import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from object_sdf import ObjectSDFStore, human_object_collision_loss


def write_box_sdf(folder, name="box.obj", half=(0.5, 0.5, 0.5), center=(0, 0, 0)):
    folder.mkdir(exist_ok=True)
    axis = np.linspace(-1, 1, 33, dtype=np.float32)
    xyz = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    q = np.abs(xyz) - np.asarray(half)
    sdf = np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(axis=-1), 0)
    np.save(folder / f"{name}.npy", sdf.astype(np.float32))
    (folder / f"{name}.json").write_text(json.dumps({
        "coord_center": list(center), "coord_min": -1, "coord_max": 1,
    }))
    return ObjectSDFStore(folder)


def loss_inputs(points, store, parts=1):
    human = torch.as_tensor(points, dtype=torch.float32)
    batch, frames, _ = human.shape
    return dict(
        human_motion=human, sdf_store=store,
        object_sdf_ids=torch.zeros(batch, parts, dtype=torch.long),
        object_sdf_rotations=torch.eye(3).expand(batch, frames, parts, 3, 3).clone(),
        object_sdf_translations=torch.zeros(batch, frames, parts, 3),
        object_sdf_scales=torch.ones(batch, frames, parts),
        object_present=torch.ones(batch, dtype=torch.bool),
        valid_mask=torch.ones(batch, frames, dtype=torch.bool),
    )


def test_deep_interior_penalty_and_gradient_move_joint_outward(tmp_path):
    store = write_box_sdf(tmp_path)
    args = loss_inputs([[[0.25, 0, 0]]], store)
    human = args["human_motion"].requires_grad_()
    loss = human_object_collision_loss(**args)
    # The joint is 25 cm from the surface: the old 0.05 proximity loss was zero.
    assert loss.item() == pytest.approx(0.25 ** 2)
    loss.backward()
    assert human.grad[0, 0, 0].item() == pytest.approx(-0.5)
    args["human_motion"] = human.detach() - 0.1 * human.grad
    assert human_object_collision_loss(**args) < loss


@pytest.mark.parametrize("point", [(0.5, 0, 0), (0.51, 0, 0), (3, 0, 0)])
def test_surface_nearby_valid_contact_and_exterior_have_zero_loss(tmp_path, point):
    args = loss_inputs([[point]], write_box_sdf(tmp_path))
    args["human_motion"].requires_grad_()
    loss = human_object_collision_loss(**args)
    loss.backward()
    assert loss.item() == 0
    assert torch.equal(args["human_motion"].grad, torch.zeros(1, 1, 3))


def test_axis_order_and_metadata_center(tmp_path):
    store = write_box_sdf(tmp_path, half=(0.5, 0.25, 0.125), center=(1, 2, 3))
    points = torch.tensor([[1.4, 2, 3], [1, 2.3, 3], [1, 2, 3.2]])
    assert store.query(0, points).tolist() == pytest.approx([-0.1, 0.05, 0.075], abs=1e-6)


def test_pose_scale_and_anisotropic_normalization_preserve_metric_loss(tmp_path):
    args = loss_inputs([[[0.25, 0, 0], [0.25, 0, 0]]], write_box_sdf(tmp_path))
    # Move and rotate the object between frames; scale it by two in frame two.
    args["object_sdf_rotations"][0, 1, 0] = torch.tensor([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    args["object_sdf_translations"][0, 1, 0] = torch.tensor([2., 3, 4])
    args["object_sdf_scales"][0, 1, 0] = 2
    physical = torch.tensor([[[0.25, 0, 0], [2, 3.5, 4]]])
    lower, upper = torch.tensor([[-2., -4, -6]]), torch.tensor([[8., 12, 14]])
    args.update(human_motion=2 * (physical - lower[:, None]) / (upper - lower)[:, None] - 1,
                norm_min=lower, norm_max=upper, motion_normalized=torch.tensor([True]))
    assert human_object_collision_loss(**args).item() == pytest.approx((0.25**2 + 0.5**2) / 2, abs=1e-6)


def test_union_of_separately_moving_parts_and_gradient(tmp_path):
    args = loss_inputs([[[2.25, 0, 0]]], write_box_sdf(tmp_path), parts=2)
    args["object_sdf_translations"][0, 0, 1, 0] = 2
    args["human_motion"].requires_grad_()
    loss = human_object_collision_loss(**args)
    loss.backward()
    assert loss.item() == pytest.approx(0.25**2)
    assert torch.isfinite(args["human_motion"].grad).all()


def test_exclusions_padding_and_absent_objects_do_not_dilute_loss(tmp_path):
    args = loss_inputs([[[0.25, 0, 0], [0, 0, 0]]] * 3, write_box_sdf(tmp_path))
    args["valid_mask"][:, 1] = False
    args["object_present"][1] = False
    args["object_sdf_ids"][2] = -1
    args["object_sdf_scales"][:, 1] = 0  # padded poses must not cause division by zero
    loss, coverage = human_object_collision_loss(**args, return_coverage=True)
    assert loss.item() == pytest.approx(0.25**2)
    assert coverage.item() == pytest.approx(0.5)


def test_metric_tolerance_allows_only_the_requested_penetration_depth(tmp_path):
    args = loss_inputs([[[0.49, 0, 0]]], write_box_sdf(tmp_path))
    assert human_object_collision_loss(**args).item() > 0
    assert human_object_collision_loss(**args, tolerance_m=0.02).item() == 0
    with pytest.raises(ValueError, match="nonnegative"):
        human_object_collision_loss(**args, tolerance_m=-0.01)


def test_out_of_grid_does_not_extend_negative_border(tmp_path):
    store = write_box_sdf(tmp_path)
    np.save(tmp_path / "box.obj.npy", -np.ones((3, 3, 3), dtype=np.float32))
    args = loss_inputs([[[3, 0, 0]]], store)
    assert human_object_collision_loss(**args).item() == 0


def test_missing_sdf_never_silently_falls_back_to_proxies(tmp_path):
    args = loss_inputs([[[0, 0, 0]]], write_box_sdf(tmp_path))
    args["sdf_store"] = None
    with pytest.raises(ValueError, match="requires SDF"):
        human_object_collision_loss(**args)


def test_empty_cavity_is_not_treated_as_solid_convex_hull(tmp_path):
    store = write_box_sdf(tmp_path)
    axis = np.linspace(-1, 1, 33, dtype=np.float32)
    xyz = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    # Exact SDF of a spherical shell, with a free cavity of radius 0.375 m.
    shell = np.abs(np.linalg.norm(xyz, axis=-1) - 0.5) - 0.125
    np.save(tmp_path / "box.obj.npy", shell.astype(np.float32))
    cavity = loss_inputs([[[0, 0, 0]]], store)
    material = loss_inputs([[[0.5, 0, 0]]], store)
    assert human_object_collision_loss(**cavity).item() == 0
    assert human_object_collision_loss(**material).item() == pytest.approx(0.125**2)


def test_sdf_loss_is_finite_under_cpu_autocast(tmp_path):
    args = loss_inputs([[[0.25, 0, 0]]], write_box_sdf(tmp_path))
    args["human_motion"].requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = human_object_collision_loss(**args)
    loss.backward()
    assert loss.item() == pytest.approx(0.25**2)
    assert torch.isfinite(args["human_motion"].grad).all()


def test_sampler_training_path_uses_sdf_and_backpropagates(tmp_path):
    from models.synhsi import Sampler

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.field = torch.nn.Parameter(torch.zeros(1, 1, 3))

        def forward(self, x, *args, **kwargs):
            return {"pred_noise": self.field.expand_as(x),
                    "pelvis_traj_dense": torch.zeros_like(x[..., :3]),
                    "completion_logits": x.new_zeros(x.shape[0])}

    store = write_box_sdf(tmp_path)
    args = loss_inputs([[[0.25, 0, 0]]], store)
    dataset = SimpleNamespace(load_scene=False, nb_voxels=[2, 2, 2],
                              load_object_sdf=True, object_sdf_dir=tmp_path)
    model = Model()
    sampler = Sampler("cpu", [], 0, 1, 3, 0, 10, use_aux_losses=True,
                      aux_loss_weights={"pelvis_traj": 0, "completion": 0, "human_object_collision": 1})
    sampler.set_dataset_and_model(dataset, model)
    x = args.pop("human_motion")
    args.pop("sdf_store")
    result = sampler.p_losses(
        x, None, None, torch.zeros_like(x, dtype=torch.bool), torch.tensor([9]),
        *([None] * 9), noise=x, return_loss_dict=True, **args,
    )
    assert result["human_object_collision"].item() == pytest.approx(0.25**2)
    assert result["human_object_collision_coverage"].item() == 1
    result["total"].backward()
    assert model.field.grad[0, 0, 0] > 0
    with pytest.raises(ValueError, match="dataset supplies no SDF"):
        sampler.set_dataset_and_model(SimpleNamespace(load_object_sdf=False), model)
