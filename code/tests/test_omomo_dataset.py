import pickle
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from datasets.omomo import OmomoDataset  # noqa: E402


def _dump(value, path):
    with open(path, "wb") as f:
        pickle.dump(value, f)


def _record(subject, offset=0.0):
    num_frames = 3
    motion = np.zeros((num_frames, 276), dtype=np.float32)
    joints = np.zeros((num_frames, 24, 3), dtype=np.float32)
    joints[..., 0] = 0.25 + offset
    joints[..., 1] = 0.5
    joints[..., 2] = np.linspace(0.8, 1.0, num_frames)[:, None]
    motion[:, :72] = joints.reshape(num_frames, 72)
    rotations = np.repeat(np.eye(3, dtype=np.float32)[None], num_frames, axis=0)
    translations = np.repeat(np.asarray([[0.2, 0.4, 0.6]], dtype=np.float32), num_frames, axis=0)
    return {
        "motion": motion,
        "seq_name": f"{subject}_box_000",
        "start_t_idx": 0,
        "end_t_idx": num_frames - 1,
        "obj_trans": translations,
        "obj_rot_mat": rotations,
        "obj_scale": np.ones(num_frames, dtype=np.float32),
        "window_obj_com_pos": translations,
    }


def test_omomo_adapter_uses_official_splits_and_flowhsi_batch_shape(tmp_path):
    stats = {
        "global_jpos_min": np.tile(np.asarray([-1.0, -1.0, 0.0], dtype=np.float32), 24),
        "global_jpos_max": np.tile(np.asarray([1.0, 1.0, 2.0], dtype=np.float32), 24),
    }
    _dump(stats, tmp_path / "stats.p")
    _dump({0: _record("sub1")}, tmp_path / "train.p")
    _dump({0: _record("sub16", offset=0.1)}, tmp_path / "test.p")
    mesh_dir = tmp_path / "captured_objects"
    mesh_dir.mkdir()
    (mesh_dir / "box_cleaned_simplified.obj").write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n",
        encoding="utf-8",
    )
    sdf_dir = tmp_path / "rest_object_sdf_256_npy_files"
    sdf_dir.mkdir()
    np.save(sdf_dir / "box_cleaned_simplified.obj.npy", np.ones((3, 3, 3), dtype=np.float32))
    (sdf_dir / "box_cleaned_simplified.obj.json").write_text(json.dumps({
        "coord_center": [0, 0, 0], "coord_min": -1, "coord_max": 1,
    }))

    dataset = OmomoDataset(
        folder=tmp_path,
        device="cpu",
        mesh_grid=[-1, 1, 0, 2, -1, 1],
        batch_size=1,
        max_window_size=4,
        object_num_points=4,
        bps_num_points=8,
        train_file="train.p",
        test_file="test.p",
        stats_file="stats.p",
        load_train=True,
        load_test=True,
        motion_state_len=2,
        load_object_sdf=True,
    )

    assert dataset.get_split_indices("train").tolist() == [0]
    assert dataset.get_split_indices("val").tolist() == [1]
    assert dataset.get_split_indices("test").tolist() == [1]
    batch = dataset[0]
    assert len(batch) == 17
    assert batch[0].shape == (4, 72)
    assert batch[13].tolist() == [True, True, True, False]
    assert batch[14].item() is True
    assert batch[16]["object_motion"].shape == (4, 9)
    assert batch[16]["object_points"].shape == (4, 3)
    assert batch[16]["object_bps"].shape == (8, 3)
    assert batch[16]["object_norm_min"].shape == (3,)
    assert batch[16]["object_norm_max"].shape == (3,)
    assert batch[16]["object_geometry_normalized"].item() is True
    assert batch[16]["motion_state"].shape == (2, 72)
    assert batch[16]["object_sdf_ids"].tolist() == [0, -1]
    assert batch[16]["object_sdf_rotations"].shape == (4, 2, 3, 3)
    # Use mesh origin obj_trans, not the COM used by the trajectory condition.
    np.testing.assert_allclose(batch[16]["object_sdf_translations"][0, 0], [0.2, 0.6, -0.4])
    np.testing.assert_allclose(batch[16]["object_sdf_rotations"][0, 0], dataset.coord_transform)
    collated = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
    assert collated[16]["object_sdf_rotations"].shape == (2, 4, 2, 3, 3)

    converted = dataset.denormalize(batch[0][:3].reshape(3, 24, 3))
    np.testing.assert_allclose(converted[0, 0], [0.25, 0.8, -0.5], atol=1e-6)


def test_articulated_objects_require_part_sdfs_and_keep_independent_poses(tmp_path):
    dataset = OmomoDataset.__new__(OmomoDataset)
    dataset.max_window_size = 4
    dataset.coord_transform = np.array([[1., 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
    dataset.object_sdf_dir = tmp_path
    dataset.object_sdf_ids = {"mop_cleaned_simplified.obj": 0}
    dataset.object_sdf_excluded_objects = set()
    record = _record("sub1")
    record.update(obj_bottom_trans=np.tile([1., 2, 3], (3, 1)),
                  obj_bottom_rot_mat=record["obj_rot_mat"], obj_bottom_scale=np.full(3, 2.))
    # A whole-object rest SDF must never be substituted for a moving part.
    with pytest.raises(FileNotFoundError, match="mop_cleaned_simplified_top"):
        dataset._object_sdf_condition(record, "mop", np.arange(3))
    dataset.object_sdf_excluded_objects = {"mop"}
    assert dataset._object_sdf_condition(record, "mop", np.arange(3))["object_sdf_ids"].tolist() == [-1, -1]
    dataset.object_sdf_excluded_objects = set()
    for idx, part in enumerate(("top", "bottom")):
        name = f"mop_cleaned_simplified_{part}.obj"
        dataset.object_sdf_ids[name] = idx + 1
        (tmp_path / f"{name}.json").write_text("{}")
    extra = dataset._object_sdf_condition(record, "mop", np.arange(3))
    assert extra["object_sdf_ids"].tolist() == [1, 2]
    np.testing.assert_allclose(extra["object_sdf_translations"][1, 1], [1, 3, -2])
    assert extra["object_sdf_scales"][1, 1] == 2
