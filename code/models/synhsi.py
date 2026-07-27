import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Optional
from vit_pytorch import ViT
from tqdm import tqdm
from utils import *
from bps_utils import make_bps_basis


class Sampler:
    def __init__(self, device, mask_ind, emb_f, batch_size, channel, auto_regre_num, timesteps, **kwargs):
        self.device = device
        self.mask_ind = mask_ind
        self.emb_f = emb_f
        self.batch_size = batch_size
        self.channel = channel
        self.auto_regre_num = auto_regre_num
        self.timesteps = timesteps
        self.motion_len = kwargs.get('motion_len', None)
        self.scene_type = kwargs.get('scene_type', None)
        self.objective = str(kwargs.get('objective', 'flow_matching')).lower()
        if self.objective in ('flow', 'fm', 'rectified_flow', 'rectified-flow', 'rf'):
            self.objective = 'flow_matching'
        elif self.objective in ('diffusion', 'ddpm', 'epsilon', 'eps'):
            self.objective = 'ddpm'
        if self.objective not in ('flow_matching', 'ddpm'):
            raise ValueError("objective must be 'flow_matching' or 'ddpm'.")
        self.use_aux_losses = kwargs.get('use_aux_losses', False)
        self.aux_loss_weights = kwargs.get('aux_loss_weights', {
            'pelvis_traj': 0.5,
            'completion': 0.2,
        })
        self.completion_pos_weight = kwargs.get('completion_pos_weight', None)
        self.beta_schedule = kwargs.get('beta_schedule', 'linear')
        self.beta_start = kwargs.get('beta_start', 0.0001)
        self.beta_end = kwargs.get('beta_end', 0.02)
        self.beta_schedule_s = kwargs.get('beta_schedule_s', 0.008)
        self.clip_denoised = kwargs.get('clip_denoised', False)
        self.clip_denoised_min = kwargs.get('clip_denoised_min', -1.5)
        self.clip_denoised_max = kwargs.get('clip_denoised_max', 1.5)
        self.debug_sampling = kwargs.get('debug_sampling', False)
        self.use_motion_state = kwargs.get('use_motion_state', False)
        self.motion_state_len = int(kwargs.get('motion_state_len', 0) or 0)
        self.human_object_collision_margin = float(kwargs.get('human_object_collision_margin', 0.05))
        self.human_object_collision_num_points = int(kwargs.get('human_object_collision_num_points', 128))
        self.human_object_collision_joint_ids = kwargs.get('human_object_collision_joint_ids', None)
        self.get_scheduler()

    def set_dataset_and_model(self, dataset, model):
        self.dataset = dataset
        if dataset.load_scene:
            self.grid = dataset.create_meshgrid(batch_size=self.batch_size).to(self.device)
        self.model = model
        nb_voxels = dataset.nb_voxels
        self.occ_idx = torch.arange(0, nb_voxels[1], 1).to(self.device)

    def get_scheduler(self):
        betas = get_beta_schedule(
            self.beta_schedule,
            self.timesteps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            cosine_s=self.beta_schedule_s,
        )

        # define alphas
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)
        self.betas = betas

    def q_sample(self, x_start, t, noise):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def flow_time_fraction(self, t, x_shape):
        tau = (t.to(dtype=torch.float32) + 1.0) / float(self.timesteps)
        return tau.reshape(-1, *([1] * (len(x_shape) - 1))).clamp(0.0, 1.0)

    def q_flow_sample(self, x_start, t, noise):
        tau = self.flow_time_fraction(t, x_start.shape)
        return (1.0 - tau) * x_start + tau * noise


    def p_losses(
            self,
            x_start,
            mat,
            scene_flag,
            mask,
            t,
            text_emb,
            pelvis_goal,
            hand_goal,
            is_pick,
            need_scene,
            need_pelvis_dir,
            pi,
            need_pi,
            is_loco,
            length=None,
            valid_mask=None,
            completion_label=None,
            object_present=None,
            object_motion=None,
            object_goal=None,
            object_points=None,
            object_bps=None,
            object_norm_min=None,
            object_norm_max=None,
            object_geometry_normalized=None,
            motion_state=None,
            motion_state_mask=None,
            noise=None,
            loss_type='huber',
            return_loss_dict=False,
    ):
        if noise is None:
            noise = torch.randn_like(x_start)

        if valid_mask is None:
            valid_mask = torch.ones(x_start.shape[:2], dtype=torch.bool, device=x_start.device)
        else:
            valid_mask = valid_mask.to(device=x_start.device, dtype=torch.bool)
        loss_mask = torch.logical_or(mask, torch.logical_not(valid_mask).unsqueeze(-1))

        if self.objective == 'flow_matching':
            x_noisy = self.q_flow_sample(x_start=x_start, t=t, noise=noise)
            x_noisy = x_noisy.masked_fill(loss_mask, 0.0)
            x_noisy[mask] = x_start[mask]
            target = noise - x_start
        else:
            noise = noise.clone()
            noise[loss_mask] = 0.
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            target = noise

        if self.dataset.load_scene:
            with torch.no_grad():
                x_orig = transform_points(self.dataset.denormalize_torch(x_noisy), mat)
                mat_for_query = mat.clone()
                target_ind = self.mask_ind if self.mask_ind != -1 else 0
                mat_for_query[:, :3, 3] = x_orig[:, self.emb_f, target_ind * 3: target_ind * 3 + 3]
                mat_for_query[:, 1, 3] = 0
                query_points = transform_points(self.grid, mat_for_query)
                occ = self.dataset.get_occ_for_points(query_points, scene_flag)
                nb_voxels = self.dataset.nb_voxels
                occ = occ.reshape(-1, nb_voxels[0], nb_voxels[1], nb_voxels[2]).float()

                if self.scene_type in ['plane_two', 'occ_two']:
                    mat_for_query_goal = mat.clone()
                    pelvis_goal_copy = pelvis_goal.clone()
                    pelvis_goal_copy[is_loco] = pelvis_goal_copy[is_loco] / (torch.norm(pelvis_goal_copy[is_loco], dim=-1, keepdim=True) + 1e-6) * 0.8
                    pelvis_goal_orig = transform_points(pelvis_goal_copy.unsqueeze(1), mat).squeeze(1)

                    mat_for_query_goal[need_pelvis_dir, :3, 3] = pelvis_goal_orig[need_pelvis_dir]
                    mat_for_query_goal[torch.logical_not(need_pelvis_dir), :3, 3] = mat_for_query[torch.logical_not(need_pelvis_dir), :3, 3].clone()
                    mat_for_query_goal[:, 1, 3] = 0.
                    query_points = transform_points(self.grid, mat_for_query_goal)
                    occ_goal = self.dataset.get_occ_for_points(query_points, scene_flag)
                    nb_voxels = self.dataset.nb_voxels
                    occ_goal = occ_goal.reshape(-1, nb_voxels[0], nb_voxels[1], nb_voxels[2]).float()

                if self.scene_type == 'occ':
                    occ = occ.permute(0, 2, 1, 3)
                elif self.scene_type == 'plane':
                    occ = occ.permute(0, 1, 3, 2)
                    occ_cnt = occ * self.occ_idx
                    occ = torch.argmax(occ_cnt, dim=-1).unsqueeze(1).float() / nb_voxels[1]
                elif self.scene_type == 'plane_two':
                    occ = occ.permute(0, 1, 3, 2)
                    occ_cnt = occ * self.occ_idx
                    occ = torch.argmax(occ_cnt, dim=-1).unsqueeze(1).float() / nb_voxels[1]

                    occ_goal = occ_goal.permute(0, 1, 3, 2)
                    occ_goal_cnt = occ_goal * self.occ_idx
                    occ_goal = torch.argmax(occ_goal_cnt, dim=-1).unsqueeze(1).float() / nb_voxels[1]
                    occ = torch.cat([occ, occ_goal], dim=1)
                elif self.scene_type == 'occ_two':
                    occ = occ.permute(0, 2, 1, 3)
                    occ_goal = occ_goal.permute(0, 2, 1, 3)
                    occ = torch.cat([occ, occ_goal], dim=1)

        else:
            occ = None

        model_out = self.model(
            x_noisy,
            occ,
            t,
            text_emb,
            pelvis_goal,
            hand_goal,
            is_pick,
            need_scene,
            need_pelvis_dir,
            pi,
            need_pi,
            object_motion=object_motion,
            object_goal=object_goal,
            object_points=object_points,
            object_bps=object_bps,
            object_norm_min=object_norm_min,
            object_norm_max=object_norm_max,
            object_geometry_normalized=object_geometry_normalized,
            object_present=object_present,
            motion_state=motion_state,
            motion_state_mask=motion_state_mask,
            return_dict=self.use_aux_losses or completion_label is not None,
        )
        predicted_field = model_out["pred_noise"] if isinstance(model_out, dict) else model_out

        mask_inv = torch.logical_not(loss_mask)

        if loss_type == 'l1':
            denoise_loss = F.l1_loss(target[mask_inv], predicted_field[mask_inv])
        elif loss_type == 'l2':
            denoise_loss = F.mse_loss(target[mask_inv], predicted_field[mask_inv])
        elif loss_type == "huber":
            denoise_loss = F.smooth_l1_loss(target[mask_inv], predicted_field[mask_inv])
        else:
            raise NotImplementedError()

        zero_loss = denoise_loss.new_zeros(())
        loss_terms = {
            "denoise": denoise_loss,
            "flow_matching": denoise_loss if self.objective == 'flow_matching' else zero_loss,
            "aux_total": zero_loss,
            "pelvis_traj": zero_loss,
            "completion": zero_loss,
            "human_object_collision": zero_loss,
            "pelvis_traj_weighted": zero_loss,
            "completion_weighted": zero_loss,
            "human_object_collision_weighted": zero_loss,
        }
        loss = denoise_loss

        if self.use_aux_losses:
            valid_frame = valid_mask.to(dtype=x_start.dtype)
            pelvis_gt = x_start[..., :3]
            pelvis_loss = (torch.abs(model_out["pelvis_traj_dense"] - pelvis_gt) * valid_frame.unsqueeze(-1)).sum()
            pelvis_loss = pelvis_loss / valid_frame.sum().clamp_min(1.0)

            if completion_label is None:
                if length is None:
                    length = valid_frame.sum(dim=1).long()
                else:
                    length = length.to(device=x_start.device, dtype=torch.long)
                completion_target = (length < x_start.shape[1]).to(dtype=x_start.dtype)
            else:
                completion_target = completion_label.to(device=x_start.device, dtype=x_start.dtype).reshape(-1)
            pos_weight = None
            if self.completion_pos_weight is not None:
                pos_weight = torch.as_tensor(
                    self.completion_pos_weight,
                    dtype=x_start.dtype,
                    device=x_start.device,
                )
            completion_loss = F.binary_cross_entropy_with_logits(
                model_out["completion_logits"].reshape(-1),
                completion_target,
                pos_weight=pos_weight,
            )
            pred_x0 = self.predict_x0_from_field(x_noisy, t, predicted_field)
            collision_geometry = (
                model_out.get("object_surface_points", object_points)
                if isinstance(model_out, dict)
                else object_points
            )
            human_object_collision_loss_value = human_object_collision_loss(
                pred_x0,
                object_points=collision_geometry,
                object_present=object_present,
                valid_mask=valid_mask,
                margin=self.human_object_collision_margin,
                max_object_points=self.human_object_collision_num_points,
                joint_ids=self.human_object_collision_joint_ids,
            )

            pelvis_weighted = self.aux_loss_weights.get('pelvis_traj', 0.5) * pelvis_loss
            completion_weighted = self.aux_loss_weights.get('completion', 0.2) * completion_loss
            human_object_collision_weighted = (
                self.aux_loss_weights.get('human_object_collision', 0.0)
                * human_object_collision_loss_value
            )
            aux_total = (
                pelvis_weighted
                + completion_weighted
                + human_object_collision_weighted
            )
            loss = loss + aux_total
            loss_terms.update(
                {
                    "aux_total": aux_total,
                    "pelvis_traj": pelvis_loss,
                    "completion": completion_loss,
                    "human_object_collision": human_object_collision_loss_value,
                    "pelvis_traj_weighted": pelvis_weighted,
                    "completion_weighted": completion_weighted,
                    "human_object_collision_weighted": human_object_collision_weighted,
                }
            )

        if return_loss_dict:
            loss_terms["total"] = loss
            return loss_terms
        return loss

    def predict_x0_from_field(self, x_noisy, t, predicted_field):
        if self.objective == 'flow_matching':
            tau = self.flow_time_fraction(t, x_noisy.shape)
            return x_noisy - tau * predicted_field

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_noisy.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_noisy.shape)
        return (x_noisy - sqrt_one_minus_alphas_cumprod_t * predicted_field) / sqrt_alphas_cumprod_t.clamp_min(1e-8)

    @torch.no_grad()
    def p_sample_loop(
            self,
            fixed_points,
            mat,
            scene_flag,
            text_emb,
            pelvis_goal,
            hand_goal,
            is_pick,
            need_scene,
            need_pelvis_dir,
            pi,
            need_pi,
            is_loco,
            motion_state=None,
            motion_state_mask=None,
            object_motion=None,
            object_goal=None,
            object_points=None,
            object_bps=None,
            object_norm_min=None,
            object_norm_max=None,
            object_geometry_normalized=None,
            object_present=None,
    ):
        device = next(self.model.parameters()).device
        shape = (self.batch_size, self.dataset.max_window_size, self.channel)
        points = torch.randn(shape, device=device)

        if self.auto_regre_num > 0:
            self.set_fixed_points(points, None, fixed_points, mat, joint_id=self.mask_ind, fix_mode=True, fix_goal=False)
        imgs = []
        occs = []
        final_model_out = None
        sample_desc = 'flow matching ODE step' if self.objective == 'flow_matching' else 'sampling loop time step'
        for i in tqdm(reversed(range(0, self.timesteps)), desc=sample_desc, total=self.timesteps):
            model_used = self.model

            sample_out = self.p_sample(
                model_used,
                points,
                fixed_points,
                mat,
                scene_flag,
                torch.full((self.batch_size,), i, device=device, dtype=torch.long),
                i,
                text_emb,
                pelvis_goal,
                hand_goal,
                is_pick,
                need_scene,
                need_pelvis_dir,
                pi,
                need_pi,
                is_loco,
                motion_state,
                motion_state_mask,
                object_motion=object_motion,
                object_goal=object_goal,
                object_points=object_points,
                object_bps=object_bps,
                object_norm_min=object_norm_min,
                object_norm_max=object_norm_max,
                object_geometry_normalized=object_geometry_normalized,
                object_present=object_present,
                return_model_out=(i == 0),
            )
            if i == 0:
                points, occ, final_model_out = sample_out
            else:
                points, occ = sample_out
            if self.auto_regre_num > 0:
                self.set_fixed_points(points, None, fixed_points, mat, joint_id=self.mask_ind, fix_mode=True, fix_goal=False)

            points_orig = points

            imgs.append(points_orig)
            if occ is not None:
                occs.append(occ.cpu().numpy())

        return imgs, occs, final_model_out

    @torch.no_grad()
    def p_sample(self, model, x, fixed_points, mat, scene_flag, t, t_index,
                 text_emb, pelvis_goal, hand_goal, is_pick, need_scene, need_pelvis_dir, pi, need_pi, is_loco,
                 motion_state=None, motion_state_mask=None,
                 object_motion=None, object_goal=None, object_points=None,
                 object_bps=None, object_norm_min=None, object_norm_max=None,
                 object_geometry_normalized=None, object_present=None,
                 return_model_out=False):
        if self.dataset.load_scene:
            x_orig = transform_points(self.dataset.denormalize_torch(x), mat)
            mat_for_query = mat.clone()
            target_ind = self.mask_ind if self.mask_ind != -1 else 0
            mat_for_query[:, :3, 3] = x_orig[:, self.emb_f, target_ind * 3: target_ind * 3 + 3]
            mat_for_query[:, 1, 3] = 0
            query_points = transform_points(self.grid, mat_for_query)
            occ = self.dataset.get_occ_for_points(query_points, scene_flag)
            nb_voxels = self.dataset.nb_voxels
            occ = occ.reshape(-1, nb_voxels[0], nb_voxels[1], nb_voxels[2]).float()

            if self.scene_type in ['plane_two', 'occ_two']:
                mat_for_query_goal = mat.clone()
                pelvis_goal_copy = pelvis_goal.clone()
                pelvis_goal_copy[is_loco] = pelvis_goal_copy[is_loco] / (
                            torch.norm(pelvis_goal_copy[is_loco], dim=-1, keepdim=True) + 1e-6) * 0.8
                pelvis_goal_orig = transform_points(pelvis_goal_copy, mat)

                mat_for_query_goal[need_pelvis_dir, :3, 3] = pelvis_goal_orig[need_pelvis_dir].squeeze(1)
                mat_for_query_goal[torch.logical_not(need_pelvis_dir), :3, 3] = mat_for_query[
                                                                                torch.logical_not(need_pelvis_dir), :3,
                                                                                3].clone()
                mat_for_query_goal[:, 1, 3] = 0.
                query_points_goal = transform_points(self.grid, mat_for_query_goal)
                occ_goal = self.dataset.get_occ_for_points(query_points_goal, scene_flag)
                nb_voxels = self.dataset.nb_voxels
                occ_goal = occ_goal.reshape(-1, nb_voxels[0], nb_voxels[1], nb_voxels[2]).float()

            if self.scene_type == 'occ':
                occ = occ.permute(0, 2, 1, 3)
            elif self.scene_type == 'plane':
                occ = occ.permute(0, 1, 3, 2)
                occ_cnt = occ * self.occ_idx
                occ = torch.argmax(occ_cnt, dim=-1).unsqueeze(1).float() / nb_voxels[1]
            elif self.scene_type == 'plane_two':
                occ = occ.permute(0, 1, 3, 2)
                occ_cnt = occ * self.occ_idx
                occ = torch.argmax(occ_cnt, dim=-1).unsqueeze(1).float() / nb_voxels[1]

                occ_goal = occ_goal.permute(0, 1, 3, 2)
                occ_goal_cnt = occ_goal * self.occ_idx
                occ_goal = torch.argmax(occ_goal_cnt, dim=-1).unsqueeze(1).float() / nb_voxels[1]
                occ = torch.cat([occ, occ_goal], dim=1)
            elif self.scene_type == 'occ_two':
                occ = occ.permute(0, 2, 1, 3)
                occ_goal = occ_goal.permute(0, 2, 1, 3)
                occ = torch.cat([occ, occ_goal], dim=1)

        else:
            occ = None

        model_out = model(
            x,
            occ,
            t,
            text_emb,
            pelvis_goal,
            hand_goal,
            is_pick,
            need_scene,
            need_pelvis_dir,
            pi,
            need_pi,
            object_motion=object_motion,
            object_goal=object_goal,
            object_points=object_points,
            object_bps=object_bps,
            object_norm_min=object_norm_min,
            object_norm_max=object_norm_max,
            object_geometry_normalized=object_geometry_normalized,
            object_present=object_present,
            motion_state=motion_state,
            motion_state_mask=motion_state_mask,
            return_dict=return_model_out,
        )
        predicted_field = model_out["pred_noise"] if isinstance(model_out, dict) else model_out
        if self.debug_sampling and (t_index >= self.timesteps - 3 or t_index in (90, 50, 10, 0)):
            print(
                "sample_debug",
                f"t={t_index}",
                f"x_abs_max={x.detach().abs().max().item():.6f}",
                f"field_abs_max={predicted_field.detach().abs().max().item():.6f}",
                f"x_minus_field_abs_max={(x - predicted_field).detach().abs().max().item():.6f}",
            )

        if self.objective == 'flow_matching':
            sample = x - predicted_field / float(self.timesteps)
            if return_model_out:
                return sample, occ, model_out
            return sample, occ

        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        if self.clip_denoised:
            sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x.shape)
            pred_x0 = (x - sqrt_one_minus_alphas_cumprod_t * predicted_field) / sqrt_alphas_cumprod_t.clamp_min(1e-8)
            pred_x0 = pred_x0.clamp(self.clip_denoised_min, self.clip_denoised_max)
            model_mean = (
                extract(self.posterior_mean_coef1, t, x.shape) * pred_x0
                + extract(self.posterior_mean_coef2, t, x.shape) * x
            )
        else:
            model_mean = sqrt_recip_alphas_t * (
                    x - betas_t * predicted_field / sqrt_one_minus_alphas_cumprod_t
            )

        if t_index == 0:
            if return_model_out:
                return model_mean, occ, model_out
            return model_mean, occ

        posterior_variance_t = extract(self.posterior_variance, t, x.shape)
        sample = model_mean + torch.sqrt(posterior_variance_t) * torch.randn_like(x)
        if return_model_out:
            return sample, occ, model_out
        return sample, occ


    def set_fixed_points(self, img, goal, fixed_points, mat, joint_id, fix_mode, fix_goal):
        '''
        set fixed points of goal and prefix frames

        img: [b, max_window_size, 3 * joint_num]
        fixed_points: [b, auto_regre_num, 3 * joint_num]

        '''

        if goal is not None and fix_goal:
            goal_len = goal.shape[1]
            goal = self.dataset.normalize_torch(transform_points(goal, torch.inverse(mat)))

            img[:, -goal_len:, joint_id * 3] = goal[:, :, 0]
            if joint_id != 0:
                img[:, -goal_len:, joint_id * 3 + 1] = goal[:, :, 1]
            img[:, -goal_len:, joint_id * 3 + 2] = goal[:, :, 2]

        if fixed_points is not None and fix_mode:
            img[:, :fixed_points.shape[1], :] = fixed_points


def temporal_upsample(x_anchor: torch.Tensor, target_len: int) -> torch.Tensor:
    """Linearly upsample low-frequency trajectory anchors to frame-level tracks."""
    if x_anchor.shape[1] == target_len:
        return x_anchor
    if x_anchor.shape[1] == 1:
        return x_anchor.repeat(1, target_len, 1)
    x = x_anchor.permute(0, 2, 1)
    x = F.interpolate(x, size=target_len, mode="linear", align_corners=True)
    return x.permute(0, 2, 1)


def _batch_bool_mask(value, batch_size: int, device, default: bool = True) -> torch.Tensor:
    if value is None:
        return torch.full((batch_size,), default, dtype=torch.bool, device=device)
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, device=device)
    value = value.to(device=device, dtype=torch.bool)
    if value.ndim > 1:
        value = value.reshape(batch_size, -1).any(dim=1)
    return value.reshape(batch_size)


def _joint_ids_tensor(joint_ids, num_joints: int, device) -> Optional[torch.Tensor]:
    if joint_ids is None:
        return None
    ids = torch.as_tensor(list(joint_ids), dtype=torch.long, device=device)
    ids = ids[(ids >= 0) & (ids < num_joints)]
    if ids.numel() == 0:
        return None
    return ids


def rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Invert the column-major 6D rotation representation used by the datasets."""
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"Expected 6D rotations, got shape {rotation_6d.shape}.")
    first = rotation_6d[..., [0, 2, 4]]
    second = rotation_6d[..., [1, 3, 5]]
    first_norm = first.norm(dim=-1, keepdim=True)
    b1 = first / first_norm.clamp_min(1e-8)
    second_ortho = second - (b1 * second).sum(dim=-1, keepdim=True) * b1
    second_norm = second_ortho.norm(dim=-1, keepdim=True)
    b2 = second_ortho / second_norm.clamp_min(1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    rotation = torch.stack([b1, b2, b3], dim=-1)

    invalid = (first_norm < 1e-8) | (second_norm < 1e-8)
    if bool(invalid.any()):
        identity = torch.eye(3, device=rotation.device, dtype=rotation.dtype)
        rotation = torch.where(invalid[..., None], identity, rotation)
    return rotation


def _prepare_object_bounds(value, batch_size, device, dtype):
    if value is None:
        return None
    value = torch.as_tensor(value, device=device, dtype=dtype)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.shape[0] == 1 and batch_size > 1:
        value = value.repeat(batch_size, 1)
    if value.shape != (batch_size, 3):
        raise ValueError(f"Expected object bounds with shape [{batch_size}, 3], got {value.shape}.")
    return value


def decode_bps_surface_points(
        object_bps: Optional[torch.Tensor],
        object_motion: Optional[torch.Tensor],
        basis_points: torch.Tensor,
        object_present: Optional[torch.Tensor] = None,
        object_norm_min: Optional[torch.Tensor] = None,
        object_norm_max: Optional[torch.Tensor] = None,
        object_geometry_normalized: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Decode vector-BPS proxies and place them at every object pose."""
    if object_bps is None or object_motion is None:
        return None
    object_bps = torch.as_tensor(object_bps, device=basis_points.device, dtype=basis_points.dtype)
    object_motion = torch.as_tensor(object_motion, device=basis_points.device, dtype=basis_points.dtype)
    if object_bps.ndim == 2:
        object_bps = object_bps.unsqueeze(0)
    if object_motion.ndim == 2:
        object_motion = object_motion.unsqueeze(0)
    batch_size = object_bps.shape[0]
    if object_motion.shape[0] == 1 and batch_size > 1:
        object_motion = object_motion.repeat(batch_size, 1, 1)
    if object_motion.shape[0] != batch_size or object_motion.shape[-1] < 9:
        raise ValueError(
            f"Expected object motion with shape [{batch_size}, T, >=9], got {object_motion.shape}."
        )
    if object_bps.shape[1:] != basis_points.shape:
        raise ValueError(
            f"BPS residual shape {object_bps.shape[1:]} does not match basis shape {basis_points.shape}."
        )

    local_proxies = basis_points.unsqueeze(0) + object_bps
    rotations = rotation_6d_to_matrix(object_motion[..., 3:9])
    translations = object_motion[..., :3]

    norm_min = _prepare_object_bounds(
        object_norm_min,
        batch_size,
        object_bps.device,
        object_bps.dtype,
    )
    norm_max = _prepare_object_bounds(
        object_norm_max,
        batch_size,
        object_bps.device,
        object_bps.dtype,
    )
    normalized_mask = _batch_bool_mask(
        object_geometry_normalized,
        batch_size,
        object_bps.device,
        default=False,
    )
    if bool(normalized_mask.any()):
        if norm_min is None or norm_max is None:
            raise ValueError("Normalized object geometry requires object_norm_min and object_norm_max.")
        scale = (norm_max - norm_min).clamp_min(1e-8)
        denormalized_translations = (
            (translations + 1.0) * scale[:, None, :] / 2.0
            + norm_min[:, None, :]
        )
        translations_physical = torch.where(
            normalized_mask[:, None, None],
            denormalized_translations,
            translations,
        )
    else:
        scale = None
        translations_physical = translations

    proxies_physical = torch.einsum(
        "bmk,btik->btmi",
        local_proxies,
        rotations,
    ) + translations_physical[:, :, None, :]

    if bool(normalized_mask.any()):
        proxies_normalized = (
            -1.0
            + 2.0
            * (proxies_physical - norm_min[:, None, None, :])
            / scale[:, None, None, :]
        )
        proxies_physical = torch.where(
            normalized_mask[:, None, None, None],
            proxies_normalized,
            proxies_physical,
        )

    present_mask = _batch_bool_mask(
        object_present,
        batch_size,
        object_bps.device,
        default=True,
    )
    return proxies_physical * present_mask.to(proxies_physical.dtype).view(batch_size, 1, 1, 1)


def _prepare_object_points(
        object_points: Optional[torch.Tensor],
        batch_size: int,
        device,
        dtype,
        max_object_points: int,
) -> Optional[torch.Tensor]:
    if object_points is None or max_object_points == 0:
        return None
    if not torch.is_tensor(object_points):
        object_points = torch.as_tensor(object_points, device=device)
    object_points = object_points.to(device=device, dtype=dtype)
    if object_points.ndim == 2:
        object_points = object_points.unsqueeze(0)
    if object_points.shape[0] == 1 and batch_size > 1:
        repeats = [batch_size] + [1] * (object_points.ndim - 1)
        object_points = object_points.repeat(*repeats)
    if (
        object_points.ndim not in (3, 4)
        or object_points.shape[0] != batch_size
        or object_points.shape[-1] != 3
    ):
        return None
    if max_object_points > 0 and object_points.shape[-2] > max_object_points:
        sample_idx = torch.linspace(
            0,
            object_points.shape[-2] - 1,
            max_object_points,
            device=device,
        ).round().long()
        object_points = object_points.index_select(-2, sample_idx)
    return object_points


def human_object_collision_loss(
        human_motion: torch.Tensor,
        object_points: Optional[torch.Tensor],
        object_present: Optional[torch.Tensor],
        valid_mask: Optional[torch.Tensor],
        margin: float = 0.05,
        max_object_points: int = 128,
        joint_ids=None,
) -> torch.Tensor:
    zero = human_motion.new_zeros(())
    if margin <= 0.0 or object_points is None:
        return zero
    batch_size, seq_len, motion_dim = human_motion.shape
    if motion_dim % 3 != 0:
        return zero

    object_mask = _batch_bool_mask(object_present, batch_size, human_motion.device, default=False)
    if not bool(object_mask.any()):
        return zero

    num_joints = motion_dim // 3
    human_points = human_motion.reshape(batch_size, seq_len, num_joints, 3)
    selected_joint_ids = _joint_ids_tensor(joint_ids, num_joints, human_motion.device)
    if selected_joint_ids is not None:
        human_points = human_points.index_select(2, selected_joint_ids)

    sampled_object_points = _prepare_object_points(
        object_points,
        batch_size=batch_size,
        device=human_motion.device,
        dtype=human_motion.dtype,
        max_object_points=max_object_points,
    )
    if sampled_object_points is None or sampled_object_points.shape[-2] == 0:
        return zero

    if sampled_object_points.ndim == 3:
        object_for_distance = sampled_object_points[:, None, None, :, :]
    else:
        if sampled_object_points.shape[1] == 1 and seq_len > 1:
            sampled_object_points = sampled_object_points.repeat(1, seq_len, 1, 1)
        if sampled_object_points.shape[1] != seq_len:
            raise ValueError(
                f"Dynamic object geometry has {sampled_object_points.shape[1]} frames; expected {seq_len}."
            )
        object_for_distance = sampled_object_points[:, :, None, :, :]
    diff = human_points.unsqueeze(3) - object_for_distance
    min_dist = torch.sqrt(diff.pow(2).sum(dim=-1).clamp_min(1e-12)).min(dim=-1).values
    penetration = (float(margin) - min_dist).clamp_min(0.0).pow(2)

    valid = torch.ones(batch_size, seq_len, device=human_motion.device, dtype=human_motion.dtype)
    if valid_mask is not None:
        valid = valid_mask.to(device=human_motion.device, dtype=human_motion.dtype)
    mask = valid.unsqueeze(-1) * object_mask.to(human_motion.dtype).view(batch_size, 1, 1)
    denom = mask.sum() * human_points.shape[2]
    if denom <= 0:
        return zero
    return (penetration * mask).sum() / denom.clamp_min(1.0)


class MotionStateEncoder(nn.Module):
    """Encode short clean boundary history into one conditioning token."""

    def __init__(self, dim_input: int, dim_output: int, state_len: int):
        super().__init__()
        self.dim_input = int(dim_input)
        self.state_len = max(1, int(state_len))
        self.delta_len = max(1, self.state_len - 1)
        feature_dim = (
            self.state_len * self.dim_input
            + self.delta_len * self.dim_input
            + 2 * self.dim_input
            + 8
        )
        self.net = nn.Sequential(
            nn.Linear(feature_dim, dim_output),
            nn.LayerNorm(dim_output),
            nn.SiLU(inplace=False),
            nn.Linear(dim_output, dim_output),
        )

    def _fit_length(self, value: torch.Tensor, target_len: int) -> torch.Tensor:
        if value.shape[1] == target_len:
            return value
        if value.shape[1] > target_len:
            return value[:, -target_len:]
        pad = value[:, :1].repeat(1, target_len - value.shape[1], *([1] * (value.ndim - 2)))
        return torch.cat([pad, value], dim=1)

    def forward(self, motion_state, motion_state_mask, batch_size: int, device, dtype) -> torch.Tensor:
        if motion_state is None:
            return torch.zeros(batch_size, 1, self.net[-1].out_features, device=device, dtype=dtype)
        motion_state = motion_state.to(device=device, dtype=dtype)
        if motion_state.ndim == 2:
            motion_state = motion_state.unsqueeze(0)
        if motion_state.shape[0] == 1 and batch_size > 1:
            motion_state = motion_state.repeat(batch_size, 1, 1)
        motion_state = self._fit_length(motion_state, self.state_len)

        if motion_state.shape[-1] != self.dim_input:
            raise ValueError(
                f"motion_state last dim={motion_state.shape[-1]} does not match dim_input={self.dim_input}."
            )

        if motion_state_mask is None:
            mask = torch.ones(batch_size, self.state_len, device=device, dtype=dtype)
        else:
            mask = motion_state_mask.to(device=device, dtype=torch.bool)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if mask.shape[0] == 1 and batch_size > 1:
                mask = mask.repeat(batch_size, 1)
            mask = self._fit_length(mask, self.state_len).to(dtype=dtype)
        state = motion_state * mask.unsqueeze(-1)

        if self.state_len > 1:
            deltas = motion_state[:, 1:] - motion_state[:, :-1]
            delta_mask = (mask[:, 1:] * mask[:, :-1]).unsqueeze(-1)
            deltas = deltas * delta_mask
            denom = delta_mask.sum(dim=1).clamp_min(1.0)
            mean_delta = deltas.sum(dim=1) / denom
            root_delta_mean = deltas[..., :3].sum(dim=1) / denom[..., :3]
            root_delta_last = deltas[:, -1, :3]
            root_speed = deltas[..., :3].norm(dim=-1)
            root_speed_mean = root_speed.sum(dim=1, keepdim=True) / delta_mask.squeeze(-1).sum(dim=1, keepdim=True).clamp_min(1.0)
            root_speed_last = root_speed[:, -1:].clone()
        else:
            deltas = motion_state.new_zeros(batch_size, 1, self.dim_input)
            mean_delta = motion_state.new_zeros(batch_size, self.dim_input)
            root_delta_mean = motion_state.new_zeros(batch_size, 3)
            root_delta_last = motion_state.new_zeros(batch_size, 3)
            root_speed_mean = motion_state.new_zeros(batch_size, 1)
            root_speed_last = motion_state.new_zeros(batch_size, 1)

        deltas = self._fit_length(deltas, self.delta_len)
        last_state = state[:, -1]
        features = torch.cat(
            [
                state.reshape(batch_size, -1),
                deltas.reshape(batch_size, -1),
                last_state,
                mean_delta,
                root_delta_mean,
                root_delta_last,
                root_speed_mean,
                root_speed_last,
            ],
            dim=-1,
        )
        return self.net(features).unsqueeze(1)


class ObjectEncoder(nn.Module):
    """Hybrid vector-BPS encoder with a point-cloud fallback for old inputs."""
    def __init__(
            self,
            dim_output: int,
            point_dim: int = 3,
            use_bps: bool = True,
            bps_num_points: int = 256,
            bps_radius: float = 1.0,
            bps_seed: int = 12345,
    ):
        super().__init__()
        self.use_bps = bool(use_bps)
        self.bps_num_points = int(bps_num_points)
        basis = make_bps_basis(self.bps_num_points, bps_radius, bps_seed)
        self.register_buffer(
            "bps_basis",
            torch.as_tensor(basis, dtype=torch.float32),
            persistent=True,
        )
        self.bps_net = nn.Sequential(
            nn.Linear(self.bps_num_points * 3, dim_output),
            nn.SiLU(inplace=False),
            nn.Linear(dim_output, dim_output),
        )
        self.point_mlp = nn.Sequential(
            nn.Linear(point_dim, dim_output),
            nn.SiLU(inplace=False),
            nn.Linear(dim_output, dim_output),
        )
        self.out = nn.Sequential(
            nn.Linear(dim_output, dim_output),
            nn.SiLU(inplace=False),
            nn.Linear(dim_output, dim_output),
        )

    def forward(
            self,
            object_points: Optional[torch.Tensor],
            object_present: torch.Tensor,
            batch_size: int,
            device,
            object_bps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_bps and object_bps is not None:
            object_bps = torch.as_tensor(object_bps, device=device, dtype=torch.float32)
            if object_bps.ndim == 2:
                object_bps = object_bps.unsqueeze(0)
            if object_bps.shape[0] == 1 and batch_size > 1:
                object_bps = object_bps.repeat(batch_size, 1, 1)
            expected_shape = (batch_size, self.bps_num_points, 3)
            if object_bps.shape != expected_shape:
                raise ValueError(
                    f"Expected vector-BPS residuals with shape {expected_shape}, got {object_bps.shape}."
                )
            feat = self.bps_net(object_bps.reshape(batch_size, -1))
        elif object_points is None:
            feat = torch.zeros(batch_size, self.out[-1].out_features, device=device)
        else:
            if object_points.ndim == 2:
                object_points = object_points.unsqueeze(0).repeat(batch_size, 1, 1)
            object_points = object_points.to(device=device, dtype=torch.float32)
            feat = self.point_mlp(object_points).mean(dim=1)
            feat = self.out(feat)
        feat = feat * object_present.to(feat.dtype).unsqueeze(-1)
        return feat.unsqueeze(1)


class VectorConditionEncoder(nn.Module):
    def __init__(self, dim_input: int, dim_output: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_input, dim_output),
            nn.SiLU(inplace=False),
            nn.Linear(dim_output, dim_output),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).unsqueeze(1)


class DynamicSceneQuery(nn.Module):
    """Sample trajectory-conditioned scene features from the local occupancy crop."""
    def __init__(
            self,
            dim_model: int,
            num_query_frames: int = 8,
            scene_channels: int = 0,
            coord_scale: float = 1.0,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.num_query_frames = num_query_frames
        self.scene_channels = int(scene_channels)
        self.coord_scale = float(coord_scale)
        input_dim = self.scene_channels * 2 + 7
        self.query_mlp = nn.Sequential(
            nn.Linear(input_dim, dim_model),
            nn.SiLU(inplace=False),
            nn.Linear(dim_model, dim_model),
        )

    def _sample_scene(self, scene_grid: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens = points.shape[:2]
        if scene_grid is None or scene_grid.ndim != 4 or self.scene_channels <= 0:
            return points.new_zeros(batch_size, num_tokens, self.scene_channels)

        scene_grid = scene_grid.to(device=points.device, dtype=points.dtype)
        if scene_grid.shape[1] < self.scene_channels:
            pad = scene_grid.new_zeros(
                scene_grid.shape[0],
                self.scene_channels - scene_grid.shape[1],
                scene_grid.shape[2],
                scene_grid.shape[3],
            )
            scene_grid = torch.cat([scene_grid, pad], dim=1)
        elif scene_grid.shape[1] > self.scene_channels:
            scene_grid = scene_grid[:, :self.scene_channels]

        scale = max(self.coord_scale, 1e-6)
        x_norm = (points[..., 0] / scale).clamp(-1.0, 1.0)
        z_norm = (points[..., 2] / scale).clamp(-1.0, 1.0)
        grid = torch.stack([z_norm, x_norm], dim=-1).reshape(batch_size, num_tokens, 1, 2)
        sampled = F.grid_sample(scene_grid, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return sampled.squeeze(-1).permute(0, 2, 1)

    def forward(
            self,
            scene_grid: Optional[torch.Tensor],
            pelvis_traj_dense: torch.Tensor,
            object_traj_dense: Optional[torch.Tensor] = None,
            object_geometry: Optional[torch.Tensor] = None,
            query_frame_indices: Optional[torch.Tensor] = None,
            object_present: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = pelvis_traj_dense.shape[0]
        device = pelvis_traj_dense.device
        dtype = pelvis_traj_dense.dtype
        seq_len = pelvis_traj_dense.shape[1]

        if scene_grid is None or self.scene_channels <= 0:
            num_tokens = max(1, min(self.num_query_frames, seq_len)) if query_frame_indices is None else int(query_frame_indices.numel())
            return torch.zeros(batch_size, num_tokens, self.dim_model, device=device, dtype=dtype)

        if query_frame_indices is None:
            num_tokens = max(1, min(self.num_query_frames, seq_len))
            query_frame_indices = torch.linspace(0, seq_len - 1, num_tokens, device=device).round().long()
        else:
            query_frame_indices = query_frame_indices.to(device=device, dtype=torch.long)
            num_tokens = int(query_frame_indices.numel())

        pelvis_query = pelvis_traj_dense.index_select(1, query_frame_indices)
        if object_traj_dense is None:
            object_query = torch.zeros_like(pelvis_query)
        else:
            object_query = object_traj_dense.index_select(1, query_frame_indices).to(device=device, dtype=dtype)

        object_mask = _batch_bool_mask(object_present, batch_size, device, default=False).to(dtype=dtype).view(batch_size, 1, 1)
        object_query = object_query * object_mask

        pelvis_scene = self._sample_scene(scene_grid, pelvis_query)
        object_scene = self._sample_scene(scene_grid, object_query) * object_mask
        time = query_frame_indices.to(dtype=dtype).view(1, num_tokens, 1) / max(seq_len - 1, 1)
        time = time.repeat(batch_size, 1, 1)

        query_feat = torch.cat(
            [pelvis_scene, object_scene, pelvis_query, object_query, time],
            dim=-1,
        )
        return self.query_mlp(query_feat)


class HumanSceneInteractionEncoder(nn.Module):
    """Cross-attend recent clean body points to occupied points in the scene crop."""

    def __init__(
            self,
            dim_model: int,
            dim_human: int,
            scene_type: Optional[str],
            num_scene_points: int = 64,
            num_body_frames: int = 4,
            num_heads: int = 4,
            occupancy_threshold: float = 0.5,
            coord_scale: float = 1.0,
    ):
        super().__init__()
        if dim_human % 3 != 0:
            raise ValueError(f"dim_human={dim_human} must be divisible by 3.")
        self.dim_model = int(dim_model)
        self.dim_human = int(dim_human)
        self.num_joints = self.dim_human // 3
        self.scene_type = str(scene_type or "").lower()
        self.num_scene_points = max(1, int(num_scene_points))
        self.num_body_frames = max(1, int(num_body_frames))
        self.occupancy_threshold = float(occupancy_threshold)
        self.coord_scale = float(coord_scale)

        heads = max(1, int(num_heads))
        if self.dim_model % heads != 0:
            heads = 1

        self.body_proj = nn.Sequential(
            nn.Linear(4, self.dim_model),
            nn.LayerNorm(self.dim_model),
            nn.SiLU(inplace=False),
            nn.Linear(self.dim_model, self.dim_model),
        )
        self.scene_proj = nn.Sequential(
            nn.Linear(3, self.dim_model),
            nn.LayerNorm(self.dim_model),
            nn.SiLU(inplace=False),
            nn.Linear(self.dim_model, self.dim_model),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.dim_model,
            num_heads=heads,
            batch_first=True,
        )
        self.out = nn.Sequential(
            nn.LayerNorm(self.dim_model),
            nn.Linear(self.dim_model, self.dim_model),
        )

    def _fit_length(self, value: torch.Tensor, target_len: int, take_last: bool = True) -> torch.Tensor:
        if value.shape[1] == target_len:
            return value
        if value.shape[1] > target_len:
            return value[:, -target_len:] if take_last else value[:, :target_len]
        pad = value[:, :1].repeat(1, target_len - value.shape[1], *([1] * (value.ndim - 2)))
        return torch.cat([pad, value], dim=1)

    def _body_points(
            self,
            human_motion: torch.Tensor,
            motion_state: Optional[torch.Tensor],
            motion_state_mask: Optional[torch.Tensor],
    ):
        batch_size = human_motion.shape[0]
        device = human_motion.device
        dtype = human_motion.dtype

        if motion_state is None:
            source = human_motion[:, :1]
            source = self._fit_length(source, self.num_body_frames, take_last=False)
            frame_mask = torch.ones(batch_size, self.num_body_frames, device=device, dtype=torch.bool)
        else:
            source = motion_state.to(device=device, dtype=dtype)
            if source.ndim == 2:
                source = source.unsqueeze(0)
            if source.shape[0] == 1 and batch_size > 1:
                source = source.repeat(batch_size, 1, 1)
            source = self._fit_length(source, self.num_body_frames, take_last=True)

            if motion_state_mask is None:
                frame_mask = torch.ones(batch_size, self.num_body_frames, device=device, dtype=torch.bool)
            else:
                frame_mask = motion_state_mask.to(device=device, dtype=torch.bool)
                if frame_mask.ndim == 1:
                    frame_mask = frame_mask.unsqueeze(0)
                if frame_mask.shape[0] == 1 and batch_size > 1:
                    frame_mask = frame_mask.repeat(batch_size, 1)
                frame_mask = self._fit_length(frame_mask, self.num_body_frames, take_last=True)

        if source.shape[-1] != self.dim_human:
            raise ValueError(
                f"human-scene body dim={source.shape[-1]} does not match dim_human={self.dim_human}."
            )

        points = source.reshape(batch_size, self.num_body_frames, self.num_joints, 3)
        valid = frame_mask[:, :, None].expand(-1, -1, self.num_joints).reshape(batch_size, -1)
        points = points.reshape(batch_size, -1, 3)

        has_valid = valid.any(dim=1, keepdim=True)
        first_valid = torch.zeros_like(valid)
        first_valid[:, :1] = True
        points = torch.where(has_valid.unsqueeze(-1), points, torch.zeros_like(points))
        valid = torch.where(has_valid, valid, first_valid)
        points = points * valid.to(dtype).unsqueeze(-1)

        time = torch.linspace(-1.0, 0.0, self.num_body_frames, device=device, dtype=dtype)
        time = time.view(1, self.num_body_frames, 1, 1).expand(batch_size, -1, self.num_joints, -1)
        time = time.reshape(batch_size, -1, 1)
        return torch.cat([points, time], dim=-1), valid

    def _select_scene_grid(self, scene_grid: torch.Tensor):
        if scene_grid is None or scene_grid.ndim != 4:
            return None, "occ"
        scene_kind = "plane" if "plane" in self.scene_type else "occ"
        if self.scene_type == "occ_two":
            channels = max(1, scene_grid.shape[1] // 2)
            return scene_grid[:, :channels], "occ"
        if self.scene_type == "plane_two":
            return scene_grid[:, :1], "plane"
        return scene_grid, scene_kind

    def _scene_points_from_occ(self, scene_grid: torch.Tensor):
        batch_size, channels, height, width = scene_grid.shape
        device = scene_grid.device
        dtype = scene_grid.dtype
        values = scene_grid.reshape(batch_size, -1)
        k = min(self.num_scene_points, values.shape[1])
        top_values, top_idx = torch.topk(values, k=k, dim=1)

        scale = max(self.coord_scale, 1e-6)
        y = torch.linspace(-1.0, 1.0, channels, device=device, dtype=dtype)
        x = torch.linspace(-scale, scale, height, device=device, dtype=dtype)
        z = torch.linspace(-scale, scale, width, device=device, dtype=dtype)
        yy, xx, zz = torch.meshgrid(y, x, z, indexing="ij")
        base_coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        points = base_coords.index_select(0, top_idx.reshape(-1)).reshape(batch_size, k, 3)

        valid = top_values > self.occupancy_threshold
        has_valid = valid.any(dim=1, keepdim=True)
        first_valid = torch.zeros_like(valid)
        first_valid[:, :1] = True
        points = torch.where(has_valid.unsqueeze(-1), points, torch.zeros_like(points))
        valid = torch.where(has_valid, valid, first_valid)
        points = points * valid.to(dtype).unsqueeze(-1)
        return points, valid

    def _scene_points_from_plane(self, scene_grid: torch.Tensor):
        batch_size, _, height, width = scene_grid.shape
        device = scene_grid.device
        dtype = scene_grid.dtype
        height_map = scene_grid[:, :1].reshape(batch_size, -1)
        k = min(self.num_scene_points, height_map.shape[1])
        top_values, top_idx = torch.topk(height_map, k=k, dim=1)

        scale = max(self.coord_scale, 1e-6)
        x = torch.linspace(-scale, scale, height, device=device, dtype=dtype)
        z = torch.linspace(-scale, scale, width, device=device, dtype=dtype)
        xx, zz = torch.meshgrid(x, z, indexing="ij")
        base_xz = torch.stack([xx, zz], dim=-1).reshape(-1, 2)
        xz = base_xz.index_select(0, top_idx.reshape(-1)).reshape(batch_size, k, 2)
        y = top_values.mul(2.0).sub(1.0).unsqueeze(-1)
        points = torch.cat([xz[..., :1], y, xz[..., 1:]], dim=-1)
        valid = torch.ones(batch_size, k, device=device, dtype=torch.bool)
        return points, valid

    def _scene_points(self, scene_grid: Optional[torch.Tensor], batch_size: int, device, dtype):
        selected_grid, scene_kind = self._select_scene_grid(scene_grid)
        if selected_grid is None or selected_grid.shape[1] == 0:
            points = torch.zeros(batch_size, 1, 3, device=device, dtype=dtype)
            valid = torch.ones(batch_size, 1, device=device, dtype=torch.bool)
            return points, valid

        selected_grid = selected_grid.to(device=device, dtype=dtype)
        if scene_kind == "plane":
            return self._scene_points_from_plane(selected_grid)
        return self._scene_points_from_occ(selected_grid)

    def forward(
            self,
            human_motion: torch.Tensor,
            scene_grid: Optional[torch.Tensor],
            need_scene: Optional[torch.Tensor] = None,
            motion_state: Optional[torch.Tensor] = None,
            motion_state_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = human_motion.shape[0]
        device = human_motion.device
        dtype = human_motion.dtype

        body_points, body_valid = self._body_points(human_motion, motion_state, motion_state_mask)
        scene_points, scene_valid = self._scene_points(scene_grid, batch_size, device, dtype)

        body_tokens = self.body_proj(body_points) * body_valid.to(dtype).unsqueeze(-1)
        scene_tokens = self.scene_proj(scene_points) * scene_valid.to(dtype).unsqueeze(-1)
        attended, _ = self.cross_attn(
            query=body_tokens,
            key=scene_tokens,
            value=scene_tokens,
            key_padding_mask=torch.logical_not(scene_valid),
            need_weights=False,
        )

        body_weight = body_valid.to(dtype).unsqueeze(-1)
        pooled = (attended * body_weight).sum(dim=1) / body_weight.sum(dim=1).clamp_min(1.0)
        token = self.out(pooled).unsqueeze(1)
        if need_scene is not None:
            scene_mask = _batch_bool_mask(need_scene, batch_size, device, default=True)
            token = token * scene_mask.to(dtype).view(batch_size, 1, 1)
        return token


class HumanObjectCrossQuery(nn.Module):
    """Encode nearest object-surface geometry around predicted human joints."""

    def __init__(
            self,
            dim_model: int,
            dim_human: int,
            query_joint_ids=None,
            max_object_points: int = 64,
            collision_margin: float = 0.05,
    ):
        super().__init__()
        if dim_human % 3 != 0:
            raise ValueError(f"dim_human={dim_human} must be divisible by 3.")
        self.num_joints = dim_human // 3
        self.max_object_points = int(max_object_points)
        self.collision_margin = float(collision_margin)

        use_all_joints = query_joint_ids is None
        ids = []
        if query_joint_ids is not None:
            ids = [int(idx) for idx in list(query_joint_ids) if 0 <= int(idx) < self.num_joints]
            use_all_joints = len(ids) == 0
        self.use_all_joints = use_all_joints
        self.register_buffer(
            "query_joint_ids",
            torch.as_tensor(ids, dtype=torch.long),
            persistent=False,
        )
        selected_joints = self.num_joints if self.use_all_joints else len(ids)
        self.query_mlp = nn.Sequential(
            nn.Linear(selected_joints * 5, dim_model),
            nn.LayerNorm(dim_model),
            nn.SiLU(inplace=False),
            nn.Linear(dim_model, dim_model),
        )

    def forward(
            self,
            human_motion: torch.Tensor,
            object_points: Optional[torch.Tensor],
            object_present: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len, _ = human_motion.shape
        device = human_motion.device
        dtype = human_motion.dtype
        zero_tokens = human_motion.new_zeros(batch_size, seq_len, self.query_mlp[-1].out_features)

        object_mask = _batch_bool_mask(object_present, batch_size, device, default=False)
        if object_points is None or not bool(object_mask.any()):
            return zero_tokens

        sampled_object_points = _prepare_object_points(
            object_points,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            max_object_points=self.max_object_points,
        )
        if sampled_object_points is None or sampled_object_points.shape[-2] == 0:
            return zero_tokens

        human_points = human_motion.reshape(batch_size, seq_len, self.num_joints, 3)
        if not self.use_all_joints:
            human_points = human_points.index_select(2, self.query_joint_ids.to(device=device))

        if sampled_object_points.ndim == 3:
            object_for_query = sampled_object_points[:, None, None, :, :]
        else:
            if sampled_object_points.shape[1] == 1 and seq_len > 1:
                sampled_object_points = sampled_object_points.repeat(1, seq_len, 1, 1)
            if sampled_object_points.shape[1] != seq_len:
                raise ValueError(
                    f"Dynamic object geometry has {sampled_object_points.shape[1]} frames; expected {seq_len}."
                )
            object_for_query = sampled_object_points[:, :, None, :, :]
        diff = human_points.unsqueeze(3) - object_for_query
        dist2 = diff.pow(2).sum(dim=-1).clamp_min(1e-12)
        min_dist2, min_idx = dist2.min(dim=-1)
        nearest = object_for_query.expand(-1, seq_len, human_points.shape[2], -1, -1).gather(
            3,
            min_idx[..., None, None].expand(-1, -1, -1, 1, 3),
        ).squeeze(3)

        delta = human_points - nearest
        min_dist = torch.sqrt(min_dist2)
        direction = delta / min_dist.unsqueeze(-1).clamp_min(1e-6)
        margin_violation = (self.collision_margin - min_dist).clamp_min(0.0)
        query_feat = torch.cat(
            [direction, min_dist.unsqueeze(-1), margin_violation.unsqueeze(-1)],
            dim=-1,
        ).reshape(batch_size, seq_len, -1)
        tokens = self.query_mlp(query_feat)
        return tokens * object_mask.to(dtype).view(batch_size, 1, 1)


class GlobalBranch(nn.Module):
    def __init__(self, dim_model: int, anchor_stride: int = 4, phase_dim: int = 32):
        super().__init__()
        self.anchor_stride = anchor_stride
        self.pelvis_head = nn.Linear(dim_model, 3)
        self.object_head = nn.Linear(dim_model, 3)
        self.phase_head = nn.Linear(dim_model, phase_dim)

    def forward(self, frame_tokens: torch.Tensor, object_present: torch.Tensor) -> Dict[str, torch.Tensor]:
        anchor_tokens = frame_tokens[:, ::self.anchor_stride]
        pelvis_anchor = self.pelvis_head(anchor_tokens)
        object_anchor = self.object_head(anchor_tokens)
        object_anchor = object_anchor * object_present.to(object_anchor.dtype).view(-1, 1, 1)
        return {
            "pelvis_traj_anchor": pelvis_anchor,
            "object_traj_anchor": object_anchor,
            "phase_latent": self.phase_head(frame_tokens),
        }


class LocalBranch(nn.Module):
    def __init__(self, dim_model: int, dim_human: int, dim_object: int = 9, phase_dim: int = 32):
        super().__init__()
        self.traj_proj = nn.Linear(6, dim_model)
        self.phase_proj = nn.Linear(phase_dim, dim_model)
        self.scene_proj = nn.Linear(dim_model, dim_model)
        self.fuse = nn.Sequential(
            nn.Linear(dim_model, dim_model),
            nn.SiLU(inplace=False),
            nn.Linear(dim_model, dim_model),
        )
        self.human_head = nn.Linear(dim_model, dim_human)
        self.object_head = nn.Linear(dim_model, dim_object)

    def forward(
            self,
            frame_tokens: torch.Tensor,
            pelvis_traj_dense: torch.Tensor,
            object_traj_dense: torch.Tensor,
            temporal_scene_tokens: torch.Tensor,
            phase_latent: torch.Tensor,
            object_present: torch.Tensor,
            human_object_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        traj_feat = self.traj_proj(torch.cat([pelvis_traj_dense, object_traj_dense], dim=-1))
        phase_feat = self.phase_proj(phase_latent)
        scene_tokens = temporal_upsample(temporal_scene_tokens, frame_tokens.shape[1])
        scene_feat = self.scene_proj(scene_tokens)
        if human_object_tokens is None:
            human_object_tokens = torch.zeros_like(frame_tokens)
        frame_tokens = self.fuse(frame_tokens + traj_feat + phase_feat + scene_feat + human_object_tokens)
        object_motion = self.object_head(frame_tokens)
        object_motion = object_motion * object_present.to(object_motion.dtype).view(-1, 1, 1)
        return {
            "frame_tokens": frame_tokens,
            "human_motion": self.human_head(frame_tokens),
            "object_motion": object_motion,
        }


class PhaseContactTerminationHeads(nn.Module):
    def __init__(self, dim_model: int, phase_dim: int = 32, contact_dim: int = 6):
        super().__init__()
        self.phase_head = nn.Linear(dim_model, phase_dim)
        self.contact_head = nn.Linear(dim_model, contact_dim)
        self.completion_head = nn.Sequential(
            nn.Linear(dim_model * 3, dim_model),
            nn.SiLU(inplace=False),
            nn.Linear(dim_model, 1),
        )

    def forward(self, frame_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        summary = torch.cat(
            [
                frame_tokens[:, -1],
                frame_tokens.mean(dim=1),
                frame_tokens.amax(dim=1),
            ],
            dim=-1,
        )
        completion_logits = self.completion_head(summary).squeeze(-1)
        return {
            "phase_latent": self.phase_head(frame_tokens),
            "contact_logits": self.contact_head(frame_tokens),
            "completion_logits": completion_logits,
            "completion_prob": torch.sigmoid(completion_logits),
        }


class Unet(nn.Module):
    def __init__(
            self,
            dim_model,
            num_heads,
            num_layers,
            dropout_p,
            dim_input,
            dim_output,
            nb_voxels=None,
            free_p=0.1,
            load_scene=True,
            load_language=True,
            load_hand_goal=True,
            load_pelvis_goal=True,
            language_feature_dim=768,
            scene_type=None,
            load_object=False,
            use_object=None,
            object_motion_dim=9,
            object_point_dim=3,
            trajectory_anchor_stride=4,
            num_scene_query_frames=8,
            scene_query_coord_scale=1.0,
            phase_dim=32,
            contact_dim=6,
            architecture="global_to_local",
            return_full_state=False,
            **kwargs
    ):
        super().__init__()

        self.architecture = str(architecture or kwargs.get("model_type", "ar_transformer")).lower()
        if self.architecture in ("lingo", "ar", "autoregressive", "auto_regressive", "window"):
            self.architecture = "ar_transformer"
        elif self.architecture in ("global", "global_to_local", "humoworld", "end_to_end"):
            self.architecture = "global_to_local"
        if self.architecture not in ("ar_transformer", "global_to_local"):
            raise ValueError(
                f"Unknown architecture={architecture!r}. "
                "Use ar_transformer or global_to_local."
            )
        self.model_type = "LINGOWindowDenoiser" if self.architecture == "ar_transformer" else "GlobalToLocalHOSIDenoiser"
        self.dim_model = dim_model
        self.load_scene = load_scene
        self.load_language = load_language
        self.load_hand_goal = load_hand_goal
        self.load_pelvis_goal = load_pelvis_goal
        self.load_object = load_object if use_object is None else use_object
        self.use_object = self.load_object
        self.object_motion_dim = object_motion_dim
        self.return_full_state = return_full_state
        self.scene_type = scene_type
        self.use_motion_state = bool(kwargs.get("use_motion_state", False))
        self.motion_state_len = max(1, int(kwargs.get("motion_state_len", 4)))
        self.use_human_scene_interaction = bool(kwargs.get("use_human_scene_interaction", False))
        self.human_scene_num_scene_points = int(kwargs.get("human_scene_num_scene_points", 64))
        self.human_scene_num_body_frames = int(kwargs.get("human_scene_num_body_frames", self.motion_state_len))
        self.human_scene_num_heads = int(kwargs.get("human_scene_num_heads", 4))
        self.human_scene_occupancy_threshold = float(kwargs.get("human_scene_occupancy_threshold", 0.5))
        self.use_human_object_cross_query = bool(kwargs.get("use_human_object_cross_query", True))
        self.human_object_query_num_points = int(kwargs.get("human_object_query_num_points", 64))
        self.human_object_query_joint_ids = kwargs.get("human_object_query_joint_ids", None)
        self.human_object_query_margin = float(kwargs.get("human_object_query_margin", 0.05))
        self.use_bps_object_geometry = bool(kwargs.get("use_bps_object_geometry", True))
        self.bps_num_points = int(kwargs.get("bps_num_points", 256))
        self.bps_radius = float(kwargs.get("bps_radius", 1.0))
        self.bps_seed = int(kwargs.get("bps_seed", 12345))
        vit_channels = 0

        if self.scene_type == 'plane':
            vit_channels = 1
        elif self.scene_type == 'occ':
            vit_channels = nb_voxels[1]
        elif self.scene_type == 'plane_two':
            vit_channels = 2
        elif self.scene_type == 'occ_two':
            vit_channels = 2*nb_voxels[1]

        if self.load_scene:
            self.scene_embedding = ViT(
                image_size=nb_voxels[0],
                patch_size=8,
                channels=vit_channels,
                num_classes=dim_model,
                dim=512,
                depth=6,
                heads=16,
                mlp_dim=1024,
                dropout=0.1,
                emb_dropout=0.1
            )
        self.free_p = free_p
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        state_input_dim = dim_input + (object_motion_dim if self.use_object else 0)
        self.embedding_input = nn.Linear(state_input_dim, dim_model)
        self.embedding_output = nn.Linear(dim_model, dim_output)

        if self.load_language:
            self.embedding_language = LanguageEncoder(dim_output=dim_model, dim_input=language_feature_dim)

        if self.load_hand_goal:
            self.embedding_hand_goal = GoalEncoder(mode='hand', dim_output=dim_model)

        if self.load_pelvis_goal:
            self.embedding_pelvis_goal = GoalEncoder(mode='pelvis', dim_output=dim_model)

        self.embedding_motion_state = None
        if self.use_motion_state:
            self.embedding_motion_state = MotionStateEncoder(
                dim_input=dim_input,
                dim_output=dim_model,
                state_len=self.motion_state_len,
            )

        if self.use_object:
            self.embedding_object = ObjectEncoder(
                dim_output=dim_model,
                point_dim=object_point_dim,
                use_bps=self.use_bps_object_geometry,
                bps_num_points=self.bps_num_points,
                bps_radius=self.bps_radius,
                bps_seed=self.bps_seed,
            )
            self.embedding_object_goal = VectorConditionEncoder(dim_input=3, dim_output=dim_model)

        self.embedding_human_scene = None
        if self.use_human_scene_interaction and self.load_scene:
            self.embedding_human_scene = HumanSceneInteractionEncoder(
                dim_model=dim_model,
                dim_human=dim_input,
                scene_type=scene_type,
                num_scene_points=self.human_scene_num_scene_points,
                num_body_frames=self.human_scene_num_body_frames,
                num_heads=self.human_scene_num_heads,
                occupancy_threshold=self.human_scene_occupancy_threshold,
                coord_scale=scene_query_coord_scale,
            )

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model,
                                                   nhead=num_heads,
                                                   dim_feedforward=dim_model,
                                                   dropout=dropout_p,
                                                   activation="gelu")

        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers
        )

        self.global_branch = None
        self.dynamic_scene_query = None
        self.human_object_cross_query = None
        self.local_branch = None
        if self.architecture == "global_to_local":
            self.global_branch = GlobalBranch(
                dim_model=dim_model,
                anchor_stride=trajectory_anchor_stride,
                phase_dim=phase_dim,
            )
            self.dynamic_scene_query = DynamicSceneQuery(
                dim_model=dim_model,
                num_query_frames=num_scene_query_frames,
                scene_channels=vit_channels if self.load_scene else 0,
                coord_scale=scene_query_coord_scale,
            )
            if self.use_object and self.use_human_object_cross_query:
                self.human_object_cross_query = HumanObjectCrossQuery(
                    dim_model=dim_model,
                    dim_human=dim_output,
                    query_joint_ids=self.human_object_query_joint_ids,
                    max_object_points=self.human_object_query_num_points,
                    collision_margin=self.human_object_query_margin,
                )
            self.local_branch = LocalBranch(
                dim_model=dim_model,
                dim_human=dim_output,
                dim_object=object_motion_dim,
                phase_dim=phase_dim,
            )
        self.object_output = nn.Linear(dim_model, object_motion_dim) if self.use_object else None
        self.heads = PhaseContactTerminationHeads(
            dim_model=dim_model,
            phase_dim=phase_dim,
            contact_dim=contact_dim,
        )

        self.embed_timestep = TimestepEmbedder(self.dim_model, self.positional_encoder)

    def _pack_state_input(self, x, object_motion, batch_size, device):
        if not self.use_object:
            return x
        if x.shape[-1] == self.embedding_input.in_features:
            return x
        if object_motion is None:
            object_motion = torch.zeros(batch_size, x.shape[1], self.object_motion_dim, device=device, dtype=x.dtype)
        return torch.cat([x, object_motion.to(device=device, dtype=x.dtype)], dim=-1)

    def forward(
            self,
            x,
            cond,
            timesteps,
            text_emb,
            pelvis_goal,
            hand_goal,
            is_pick,
            need_scene,
            need_pelvis_dir,
            pi,
            need_pi,
            object_motion=None,
            object_goal=None,
            object_points=None,
            object_bps=None,
            object_norm_min=None,
            object_norm_max=None,
            object_geometry_normalized=None,
            object_present=None,
            motion_state=None,
            motion_state_mask=None,
            return_dict=False,
    ):
        batch_size = x.shape[0]
        device = x.device
        t_emb = self.embed_timestep(timesteps)  # [b, 1, d]
        object_present = _batch_bool_mask(
            object_present if object_present is not None else (is_pick if self.use_object else None),
            batch_size,
            device,
            default=False,
        )
        object_surface_points = object_points
        if self.use_object and self.use_bps_object_geometry and object_bps is not None:
            decoded_surface = decode_bps_surface_points(
                object_bps=object_bps,
                object_motion=object_motion,
                basis_points=self.embedding_object.bps_basis,
                object_present=object_present,
                object_norm_min=object_norm_min,
                object_norm_max=object_norm_max,
                object_geometry_normalized=object_geometry_normalized,
            )
            if decoded_surface is not None:
                object_surface_points = decoded_surface

        if not self.load_scene:
            scene_emb = torch.zeros_like(t_emb)
        elif cond is None:
            scene_emb = torch.zeros_like(t_emb)
        else:
            scene_emb = self.scene_embedding(cond).reshape(-1, 1, self.dim_model)
            not_need_scene = torch.logical_not(need_scene)
            scene_emb[not_need_scene] = 0.
        
        if not self.load_language:
            language_emb = torch.zeros_like(t_emb)
        else:
            language_emb = self.embedding_language(text_emb, pi, need_pi)

        if not self.load_hand_goal:
            hand_goal_emb = torch.zeros_like(t_emb)
        else:
            hand_goal_emb = self.embedding_hand_goal(hand_goal)
            is_not_pick = torch.logical_not(is_pick)
            hand_goal_emb[is_not_pick] = 0.

        if not self.load_pelvis_goal:
            pelvis_goal_emb = torch.zeros_like(t_emb)
        else:
            pelvis_goal_emb = self.embedding_pelvis_goal(pelvis_goal)
            not_need_pelvis_dir = torch.logical_not(need_pelvis_dir)
            pelvis_goal_emb[not_need_pelvis_dir] = 0.

        cond_tokens = [
            t_emb + scene_emb,
            t_emb + language_emb,
            t_emb + hand_goal_emb,
            t_emb + pelvis_goal_emb,
        ]

        if self.use_object:
            object_emb = self.embedding_object(
                object_points,
                object_present,
                batch_size,
                device,
                object_bps=object_bps,
            )
            if object_goal is not None:
                object_goal_emb = self.embedding_object_goal(object_goal.to(device=device, dtype=x.dtype))
                object_emb = object_emb + object_goal_emb
            cond_tokens.append(t_emb + object_emb)

        if self.use_motion_state:
            motion_state_emb = self.embedding_motion_state(
                motion_state,
                motion_state_mask,
                batch_size=batch_size,
                device=device,
                dtype=x.dtype,
            )
            cond_tokens.append(t_emb + motion_state_emb)

        if self.embedding_human_scene is not None:
            human_scene_emb = self.embedding_human_scene(
                human_motion=x,
                scene_grid=cond,
                need_scene=need_scene,
                motion_state=motion_state,
                motion_state_mask=motion_state_mask,
            )
            cond_tokens.append(t_emb + human_scene_emb)

        cond_tokens = [token.permute(1, 0, 2) for token in cond_tokens]
        cond_token_count = sum(token.shape[0] for token in cond_tokens)
        state_input = self._pack_state_input(x, object_motion, batch_size, device)
        frame_tokens = state_input.permute(1, 0, 2)
        frame_tokens = self.embedding_input(frame_tokens) * math.sqrt(self.dim_model)

        tokens = torch.cat(cond_tokens + [frame_tokens], dim=0)
        tokens = self.positional_encoder(tokens)
        tokens = self.transformer(tokens)

        frame_tokens = tokens[cond_token_count:].permute(1, 0, 2)
        if self.architecture == "ar_transformer":
            human_motion = self.embedding_output(frame_tokens)
            if self.use_object:
                object_motion = self.object_output(frame_tokens)
                object_motion = object_motion * object_present.to(object_motion.dtype).view(-1, 1, 1)
            else:
                object_motion = frame_tokens.new_zeros(batch_size, x.shape[1], self.object_motion_dim)
            aux_out = self.heads(frame_tokens)
            pred = {
                "pred_noise": human_motion,
                "x0": human_motion,
                "human_motion": human_motion,
                "object_motion": object_motion,
                "pelvis_traj_anchor": human_motion[..., :3],
                "object_traj_anchor": object_motion[..., :3],
                "pelvis_traj_dense": human_motion[..., :3],
                "object_traj_dense": object_motion[..., :3],
                "phase_latent": aux_out["phase_latent"],
                "global_phase_latent": aux_out["phase_latent"],
                "contact_logits": aux_out["contact_logits"],
                "completion_logits": aux_out["completion_logits"],
                "completion_prob": aux_out["completion_prob"],
                "object_present": object_present,
                "object_surface_points": object_surface_points,
            }

            if return_dict:
                return pred
            if self.return_full_state and self.use_object:
                return torch.cat([pred["human_motion"], pred["object_motion"]], dim=-1)
            return pred["human_motion"]

        global_out = self.global_branch(frame_tokens, object_present)
        pelvis_dense = temporal_upsample(global_out["pelvis_traj_anchor"], x.shape[1])
        object_dense = temporal_upsample(global_out["object_traj_anchor"], x.shape[1])

        temporal_scene_tokens = self.dynamic_scene_query(
            scene_grid=cond,
            pelvis_traj_dense=pelvis_dense,
            object_traj_dense=object_dense,
            object_geometry=object_surface_points,
            object_present=object_present,
        )
        human_object_tokens = None
        if self.human_object_cross_query is not None:
            human_object_tokens = self.human_object_cross_query(
                human_motion=x,
                object_points=object_surface_points,
                object_present=object_present,
            )
        local_out = self.local_branch(
            frame_tokens=frame_tokens,
            pelvis_traj_dense=pelvis_dense,
            object_traj_dense=object_dense,
            temporal_scene_tokens=temporal_scene_tokens,
            phase_latent=global_out["phase_latent"],
            object_present=object_present,
            human_object_tokens=human_object_tokens,
        )
        aux_out = self.heads(local_out["frame_tokens"])

        pred = {
            "pred_noise": local_out["human_motion"],
            "x0": local_out["human_motion"],
            "human_motion": local_out["human_motion"],
            "object_motion": local_out["object_motion"],
            "pelvis_traj_anchor": global_out["pelvis_traj_anchor"],
            "object_traj_anchor": global_out["object_traj_anchor"],
            "pelvis_traj_dense": pelvis_dense,
            "object_traj_dense": object_dense,
            "phase_latent": aux_out["phase_latent"],
            "global_phase_latent": global_out["phase_latent"],
            "contact_logits": aux_out["contact_logits"],
            "completion_logits": aux_out["completion_logits"],
            "completion_prob": aux_out["completion_prob"],
            "object_present": object_present,
            "object_surface_points": object_surface_points,
        }

        if return_dict:
            return pred
        if self.return_full_state and self.use_object:
            return torch.cat([pred["human_motion"], pred["object_motion"]], dim=-1)
        return pred["human_motion"]


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).reshape(-1, 1)  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(inplace=False),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pos_encoding[timesteps])


class ProgressIndicatorEmbedding(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

    def forward(self, timesteps):
        return self.sequence_pos_encoder.pos_encoding[timesteps]


class ActionTransformerEncoder(nn.Module):
    def __init__(self,
                 action_number,
                 dim_model,
                 nhead,
                 num_layers,
                 dim_feedforward,
                 dropout_p,
                 activation="gelu") -> None:
        super().__init__()
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.input_embedder = nn.Linear(action_number, dim_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model,
                                                    nhead=nhead,
                                                    dim_feedforward=dim_feedforward,
                                                    dropout=dropout_p,
                                                    activation=activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers
        )

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = self.input_embedder(x)
        x = self.positional_encoder(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        x = torch.mean(x, dim=1, keepdim=True)
        return x
    

class LanguageEncoder(nn.Module):
    def __init__(self, dim_output, dim_input, **kwargs):
        super().__init__()
        self.dim_model = dim_output

        self.embedding_input1 = nn.Sequential(
            nn.Linear(dim_input, dim_output),
            nn.SiLU(inplace=False),
            nn.Linear(dim_output, dim_output),
        )

        self.embedding_input2 = nn.Sequential(
            nn.Linear(dim_output, dim_output),
            nn.SiLU(inplace=False),
            nn.Linear(dim_output, dim_output),
        )

        self.positional_encoder = PositionalEncoding(
            dim_model=dim_output, dropout_p=0.1, max_len=5000
        )

        self.embed_pi = ProgressIndicatorEmbedding(dim_output, self.positional_encoder)

    def forward(self, x, pi, need_pi):
        # x.shape: [b, 1, 768]

        x = self.embedding_input1(x)
        pi = self.embed_pi(pi)

        # normalization
        pi = pi / np.sqrt(self.dim_model // 2)
        not_need_pi = torch.logical_not(need_pi)
        pi[not_need_pi] = 0.
        x = x + pi
        x = self.embedding_input2(x)
        return x

class GoalEncoder(nn.Module):
    def __init__(self, mode, dim_output, **kwargs):
        super().__init__()

        self.mode = mode
        if mode == 'pelvis':
            self.embedding_input = nn.Sequential(nn.Linear(2, dim_output),
                                                    nn.SiLU(inplace=False),
                                                    nn.Linear(dim_output, dim_output))
        elif mode == 'hand':
            self.embedding_input = nn.Sequential(nn.Linear(3, dim_output),
                                                    nn.SiLU(inplace=False),
                                                    nn.Linear(dim_output, dim_output))

    def forward(self, x):
        # x.shape: [b, 3]
        if self.mode == 'pelvis':
            x = x[..., [0, 2]]
        x = self.embedding_input(x)
        x = x.reshape(-1, 1, x.shape[-1])
        return x
