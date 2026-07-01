import os
import sys
import datetime
import json
import math
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
os.environ.setdefault('ROOT_DIR', str(PROJECT_ROOT))
os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['CURRENT_TIME'] = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['NCCL_P2P_DISABLE'] = '0'
os.environ['NCCL_IB_DISABLE'] = '0'

import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam
from utils import *
from constants import *
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_rank_logging(cfg, rank):
    os.makedirs(cfg.exp_dir, exist_ok=True)
    log_path = os.path.join(cfg.exp_dir, f"train_rank{rank}.log")
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")
    stdout_orig = sys.stdout
    stderr_orig = sys.stderr
    sys.stdout = TeeStream(stdout_orig, log_file)
    sys.stderr = TeeStream(stderr_orig, log_file)
    return log_file, stdout_orig, stderr_orig


def write_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


LOSS_LOG_ORDER = (
    "total",
    "denoise",
    "flow_matching",
    "aux_total",
    "pelvis_traj_weighted",
    "completion_weighted",
    "pelvis_traj",
    "completion",
)


def update_loss_totals(totals, loss_dict, batch_size):
    for key, value in loss_dict.items():
        if key not in totals:
            totals[key] = torch.zeros(1, device=value.device)
        totals[key] += value.detach().reshape(1) * batch_size


def average_loss_totals(totals, total_count):
    return {
        key: (value / total_count.clamp_min(1)).item()
        for key, value in totals.items()
    }


def format_loss_parts(loss_values):
    parts = []
    for key in LOSS_LOG_ORDER:
        if key in loss_values:
            parts.append(f"{key}={loss_values[key]:.6f}")
    return " ".join(parts)


def loss_payload(prefix, loss_values):
    payload = {}
    for key in LOSS_LOG_ORDER:
        if key in loss_values:
            payload[f"{prefix}/loss_{key}"] = float(loss_values[key])
    if "total" in loss_values:
        payload[f"{prefix}/loss"] = float(loss_values["total"])
    return payload


def log_loss_scalars(writer, prefix, loss_values, step):
    if writer is None:
        return
    for key in LOSS_LOG_ORDER:
        if key in loss_values:
            writer.add_scalar(f"{prefix}/loss_{key}", loss_values[key], step)
    if "total" in loss_values:
        writer.add_scalar(f"{prefix}/loss", loss_values["total"], step)


def init_wandb(cfg, rank):
    if rank != 0 or not cfg.get("use_wandb", False):
        return None
    try:
        import wandb
    except ImportError:
        print("WandB is enabled but wandb is not installed. Install it with `pip install wandb`.", flush=True)
        return None

    wandb_kwargs = {
        "project": cfg.get("wandb_project", "HSI-ours"),
        "name": cfg.get("wandb_run_name", cfg.exp_name),
        "mode": cfg.get("wandb_mode", "online"),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "dir": cfg.exp_dir,
    }
    if cfg.get("wandb_entity", None) not in [None, "null", "None"]:
        wandb_kwargs["entity"] = cfg.wandb_entity
    try:
        return wandb.init(**wandb_kwargs)
    except Exception as exc:
        print(f"WandB initialization failed, continuing without WandB: {exc}", flush=True)
        return None


def get_split_subset(dataset, dataset_cfg, split_name):
    if split_name in [None, 'None', 'none', 'null']:
        return dataset
    if hasattr(dataset, "get_split_indices"):
        return Subset(dataset, dataset.get_split_indices(split_name))
    split_dir = dataset_cfg.get('split_dir', 'splits')
    split_path = os.path.join(dataset_cfg.folder, split_dir, f'{split_name}_idx.npy')
    split_idx = np.load(split_path).astype(np.int64)
    return Subset(dataset, split_idx)


def build_dataset(dataset_cfg):
    cfg_dict = OmegaConf.to_container(dataset_cfg, resolve=True)
    target = cfg_dict.pop("_target_", "datasets.lingo.LingoDataset")
    cfg_dict.pop("name", None)
    dataset_cls = hydra.utils.get_class(target)
    return dataset_cls(**cfg_dict)


def build_lr_scheduler(optimizer, cfg, steps_per_epoch):
    if not cfg.get("use_lr_decay", False):
        return None, None
    scheduler_type = str(cfg.get("lr_scheduler", "step")).lower()
    if scheduler_type in ("step", "steplr"):
        step_size = int(cfg.get("lr_decay_step_size", 50))
        gamma = float(cfg.get("lr_decay_gamma", 0.95))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        return scheduler, {
            "type": "step",
            "interval": "epoch",
            "step_size": step_size,
            "gamma": gamma,
        }

    if scheduler_type not in ("cosine", "warmup_cosine", "cosine_warmup"):
        raise ValueError(
            f"Unsupported lr_scheduler='{scheduler_type}'. "
            "Expected one of: step, cosine, warmup_cosine."
        )

    total_steps = int(cfg.get("lr_total_steps", 0) or int(cfg.epochs) * int(steps_per_epoch))
    warmup_steps = int(cfg.get("lr_warmup_steps", 0) or 0)
    warmup_epochs = cfg.get("lr_warmup_epochs", None)
    if warmup_steps <= 0 and warmup_epochs is not None:
        warmup_steps = int(float(warmup_epochs) * int(steps_per_epoch))
    warmup_steps = min(max(warmup_steps, 0), total_steps)

    base_lr = float(cfg.lr)
    min_lr = float(cfg.get("lr_min", 0.0))
    if min_lr < 0:
        raise ValueError("lr_min must be non-negative.")
    if min_lr > base_lr:
        raise ValueError("lr_min must be <= lr.")
    min_lr_ratio = min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(current_step):
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        decay_steps = max(total_steps - warmup_steps, 1)
        progress = min(max((current_step - warmup_steps) / decay_steps, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler, {
        "type": "warmup_cosine",
        "interval": "step",
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "min_lr": min_lr,
    }


def move_lingo_batch(batch, device):
    completion_label = None
    extra = {}
    if len(batch) == 15:
        joints, mat, scene_flag, text_clip_embedding, pelvis_goal, hand_goal, is_pick, need_scene, need_pelvis_dir, pi, need_pi, is_loco, length, valid_mask, object_present = batch
    elif len(batch) == 16:
        joints, mat, scene_flag, text_clip_embedding, pelvis_goal, hand_goal, is_pick, need_scene, need_pelvis_dir, pi, need_pi, is_loco, length, valid_mask, object_present, completion_label = batch
    else:
        joints, mat, scene_flag, text_clip_embedding, pelvis_goal, hand_goal, is_pick, need_scene, need_pelvis_dir, pi, need_pi, is_loco, length, valid_mask, object_present, completion_label, extra = batch
    extra = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in (extra or {}).items()
    }
    return joints.to(device), \
           mat.to(device), scene_flag.to(device), \
           text_clip_embedding.to(device), \
           pelvis_goal.to(device), hand_goal.to(device), \
           is_pick.to(device), need_scene.to(device), need_pelvis_dir.to(device), pi.to(device), \
           need_pi.to(device), is_loco.to(device), length.to(device), valid_mask.to(device), object_present.to(device), \
           None if completion_label is None else completion_label.to(device), extra


@torch.no_grad()
def validate(model, trainer, dataloader, cfg, device, epoch=0, show_progress=False):
    model.eval()
    total_losses = {}
    total_count = torch.zeros(1, device=device)

    progress = tqdm(dataloader, desc=f"Val {epoch}", disable=not show_progress, leave=False)
    for batch in progress:
        joints, mat, scene_flag, text_clip_embedding, pelvis_goal, hand_goal, is_pick, need_scene, need_pelvis_dir, pi, need_pi, is_loco, length, valid_mask, object_present, completion_label, extra = move_lingo_batch(batch, device)
        batch_size = joints.shape[0]
        t = torch.randint(0, trainer.timesteps, (batch_size,), device=device).long()
        mask, _, _ = get_mask(joints, -1, p=1., fixed_frame=cfg.auto_regre_num)
        loss = trainer.p_losses(
            joints, mat, scene_flag, mask, t,
            text_clip_embedding, pelvis_goal, hand_goal,
            is_pick, need_scene, need_pelvis_dir, pi, need_pi, is_loco,
            length=length, valid_mask=valid_mask, completion_label=completion_label, object_present=object_present,
            object_motion=extra.get("object_motion"),
            object_points=extra.get("object_points"),
            object_goal=extra.get("object_goal"),
            motion_state=extra.get("motion_state"),
            motion_state_mask=extra.get("motion_state_mask"),
            return_loss_dict=True,
        )
        update_loss_totals(total_losses, loss, batch_size)
        total_count += batch_size
        progress.set_postfix(
            loss=f"{loss['total'].item():.4f}",
            denoise=f"{loss['denoise'].item():.4f}",
        )

    for key in sorted(total_losses):
        torch.distributed.all_reduce(total_losses[key], op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(total_count, op=torch.distributed.ReduceOp.SUM)
    model.train()
    return average_loss_totals(total_losses, total_count)


@hydra.main(version_base=None, config_path="config", config_name="config_train")
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = find_free_port()
    world_size = cfg.num_gpus
    print('Usable GPUS: ', torch.cuda.device_count(), flush=True)
    torch.multiprocessing.spawn(train_ddp,
                                args=(world_size, cfg),
                                nprocs=world_size,
                                join=True)

def train_ddp(rank, world_size, cfg):

    OmegaConf.register_new_resolver("times", lambda x, y: int(x) * int(y))

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    cfg.device = f"cuda:{rank}"
    rank_log_file, stdout_orig, stderr_orig = setup_rank_logging(cfg, rank)
    print(f'Training on {device}', flush=True)
    print('Initializing Distributed', flush=True)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    model = init_model(list(cfg.model.values())[0], device=rank, eval=False, load_state_dict=cfg.load_state_dict)

    train_split = cfg.dataset.get('split', 'train')
    val_split = cfg.get('val_split', 'val')
    dataset_cfg = OmegaConf.create(OmegaConf.to_container(cfg.dataset, resolve=True))
    dataset_cfg.split = None
    synhsi_dataset = build_dataset(dataset_cfg)
    train_dataset = get_split_subset(synhsi_dataset, dataset_cfg, train_split)
    val_dataset = get_split_subset(synhsi_dataset, dataset_cfg, val_split) if cfg.get('use_validation', True) else None

    sampler = DistributedSampler(train_dataset)
    pin_memory = cfg.num_workers > 0
    dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, drop_last=True, num_workers=cfg.num_workers,
                            sampler=sampler, pin_memory=pin_memory)
    if val_dataset is not None:
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=cfg.val_batch_size, drop_last=True, num_workers=cfg.num_workers,
                                    sampler=val_sampler, pin_memory=pin_memory)
    else:
        val_dataloader = None

    trainer = hydra.utils.instantiate(list(cfg.sampler.values())[0])
    trainer.set_dataset_and_model(synhsi_dataset, model)

    optimizer = Adam(model.parameters(), lr=cfg.lr)
    lr_scheduler, lr_scheduler_info = build_lr_scheduler(optimizer, cfg, len(dataloader))
    if rank == 0 and lr_scheduler is not None:
        print(f"LR scheduler enabled: {lr_scheduler_info}", flush=True)

    if cfg.use_tensorboard and rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(cfg.exp_dir, 'tensorboard_logs'))
    else:
        writer = None

    wandb_run = init_wandb(cfg, rank)
    metrics_path = os.path.join(cfg.exp_dir, "metrics.jsonl")

    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(cfg.epochs):
        if rank == 0:
            print(f'Start epoch {epoch}', flush=True)
        sampler.set_epoch(epoch)
        step = 0
        progress = tqdm(dataloader, desc=f"Train {epoch}", disable=rank != 0, leave=True)
        for batch in progress:
            step += 1
            global_step = epoch * len(dataloader) + step
            optimizer.zero_grad()

            joints, mat, scene_flag, text_clip_embedding, pelvis_goal, hand_goal, is_pick, need_scene, need_pelvis_dir, pi, need_pi, is_loco, length, valid_mask, object_present, completion_label, extra = move_lingo_batch(batch, device)

            batch_size = joints.shape[0]
            t = torch.randint(0, trainer.timesteps, (batch_size,), device=device).long()
            with torch.no_grad():
                mask, _, _ = get_mask(joints, -1, p=1., fixed_frame=cfg.auto_regre_num)

            loss_dict = trainer.p_losses(
                joints, mat, scene_flag, mask, t,
                text_clip_embedding, pelvis_goal, hand_goal,
                is_pick, need_scene, need_pelvis_dir, pi, need_pi, is_loco,
                length=length, valid_mask=valid_mask, completion_label=completion_label, object_present=object_present,
                object_motion=extra.get("object_motion"),
                object_points=extra.get("object_points"),
                object_goal=extra.get("object_goal"),
                motion_state=extra.get("motion_state"),
                motion_state_mask=extra.get("motion_state_mask"),
                return_loss_dict=True,
            )
            loss = loss_dict["total"]

            if step % cfg.get("log_interval", 10) == 0:
                train_losses = {key: value.detach().item() for key, value in loss_dict.items()}
                print(
                    f"Epoch: {epoch}, Step: {step} / {len(dataloader)}   "
                    f"{format_loss_parts(train_losses)}",
                    flush=True,
                )
                if rank == 0:
                    log_payload = {
                        "epoch": epoch,
                        "step": step,
                        "global_step": global_step,
                        "train/lr": float(optimizer.param_groups[0]["lr"]),
                    }
                    log_payload.update(loss_payload("train", train_losses))
                    write_jsonl(metrics_path, log_payload)
                    if writer is not None:
                        writer.add_scalar('Loss', loss.item(), global_step)
                        writer.add_scalar('LR', optimizer.param_groups[0]["lr"], global_step)
                        log_loss_scalars(writer, "train", train_losses, global_step)
                    if wandb_run is not None:
                        wandb_run.log(log_payload, step=global_step)

            loss.backward()
            optimizer.step()
            if lr_scheduler is not None and lr_scheduler_info["interval"] == "step":
                lr_scheduler.step()
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                denoise=f"{loss_dict['denoise'].item():.4f}",
            )

        if rank == 0 and epoch % cfg.ckpt_interval == 0:
            print(f'Saving checkpoint', flush=True)
            ckpt_folder = os.path.join(cfg.exp_dir, 'checkpoints')
            os.makedirs(ckpt_folder, exist_ok=True)
            torch.save(model.module.state_dict(), os.path.join(ckpt_folder, f"{cfg.exp_name}_epoch{epoch:03d}.pth"))

        if val_dataloader is not None and epoch % cfg.val_interval == 0:
            val_losses = validate(model, trainer, val_dataloader, cfg, device, epoch=epoch, show_progress=(rank == 0))
            val_loss = val_losses["total"]
            if rank == 0:
                print(f"Epoch: {epoch}   Val {format_loss_parts(val_losses)}", flush=True)
                val_payload = {
                    "epoch": epoch,
                    "global_step": global_step,
                }
                val_payload.update(loss_payload("val", val_losses))
                write_jsonl(metrics_path, val_payload)
                if writer is not None:
                    writer.add_scalar('Val/Loss', val_loss, epoch)
                    log_loss_scalars(writer, "val", val_losses, epoch)
                if wandb_run is not None:
                    eval_payload = {
                        "epoch": epoch,
                        "global_step": global_step,
                    }
                    eval_payload.update(loss_payload("eval", val_losses))
                    wandb_run.log(eval_payload, step=global_step)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    ckpt_folder = os.path.join(cfg.exp_dir, 'checkpoints')
                    os.makedirs(ckpt_folder, exist_ok=True)
                    best_path = os.path.join(ckpt_folder, 'best_model.pth')
                    torch.save(model.module.state_dict(), best_path)
                    print(f"Saved best model to {best_path} with val loss {best_val_loss}", flush=True)
                    if wandb_run is not None:
                        wandb_run.summary["best_val_loss"] = float(best_val_loss)

        if lr_scheduler is not None:
            if lr_scheduler_info["interval"] == "epoch":
                lr_scheduler.step()
            if rank == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                if lr_scheduler_info["interval"] == "epoch":
                    print(f"Epoch: {epoch}   LR updated to {current_lr:.8f}", flush=True)
                else:
                    print(f"Epoch: {epoch}   LR at epoch end {current_lr:.8f}", flush=True)
                if writer is not None:
                    writer.add_scalar('LR/epoch', current_lr, epoch)
                if wandb_run is not None:
                    wandb_run.log({"train/lr_epoch": float(current_lr)}, step=global_step)

        torch.distributed.barrier()

        print('Clearing cache', flush=True)
        torch.cuda.empty_cache()

    if writer is not None:
        writer.close()
    if wandb_run is not None:
        wandb_run.finish()
    sys.stdout = stdout_orig
    sys.stderr = stderr_orig
    rank_log_file.close()


def get_mask(x_start, ind, p, fixed_frame=0, mask_y=True):
    '''
    get mask for the input sequence of pre frames and final goal frame
    '''
    mask_frame = torch.zeros_like(x_start).to(dtype=torch.bool, device=x_start.device)
    mask_goal = torch.zeros_like(x_start).to(dtype=torch.bool, device=x_start.device)

    # goal mask
    if ind != -1:
        rand_batch = torch.rand(x_start.shape[0]).to(x_start.device) < p
        mask_goal[rand_batch, -1, ind * 3: ind * 3 + 3] = True
        if not mask_y:
            mask_goal[rand_batch, -1, ind * 3 + 1] = False

    # prefix frame mask
    if fixed_frame > 0:
        rand_batch = torch.rand(x_start.shape[0]).to(x_start.device) < p
        mask_frame[rand_batch, :fixed_frame, :] = True
    mask = torch.logical_or(mask_frame, mask_goal)
    return mask, mask_frame, mask_goal


if __name__ == '__main__':
    train()
