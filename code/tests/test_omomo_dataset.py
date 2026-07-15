import pickle
import sys
from pathlib import Path

import numpy as np


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

    dataset = OmomoDataset(
        folder=tmp_path,
        device="cpu",
        mesh_grid=[-1, 1, 0, 2, -1, 1],
        batch_size=1,
        max_window_size=4,
        object_num_points=4,
        train_file="train.p",
        test_file="test.p",
        stats_file="stats.p",
        load_train=True,
        load_test=True,
        motion_state_len=2,
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
    assert batch[16]["motion_state"].shape == (2, 72)

    converted = dataset.denormalize(batch[0][:3].reshape(3, 24, 3))
    np.testing.assert_allclose(converted[0, 0], [0.25, 0.8, -0.5], atol=1e-6)
