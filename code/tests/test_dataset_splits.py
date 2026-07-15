import sys
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from create_dataset_splits import (  # noqa: E402
    OMOMO_TEST_FILE,
    OMOMO_TRAIN_FILE,
    create_trumans_splits,
    detect_dataset_type,
    main,
    split_policy,
)


def test_omomo_official_split_is_detected_and_generation_is_skipped(tmp_path, monkeypatch):
    data_dir = tmp_path / "OMOMO" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / OMOMO_TRAIN_FILE).touch()
    (data_dir / OMOMO_TEST_FILE).touch()

    dataset_type, resolved = detect_dataset_type(tmp_path / "OMOMO")
    policy = split_policy(dataset_type, resolved)
    assert dataset_type == "omomo"
    assert resolved == data_dir
    assert policy["split_required"] is False

    monkeypatch.setattr(
        sys,
        "argv",
        ["create_dataset_splits.py", "--dataset-dir", str(tmp_path / "OMOMO")],
    )
    main()
    assert not (data_dir / "splits").exists()


def test_trumans_splits_are_scene_group_disjoint(tmp_path):
    dataset_dir = tmp_path / "trumans"
    output_dir = dataset_dir / "splits"
    dataset_dir.mkdir()
    idx_start = np.asarray([0, 1, 10, 11, 20, 21, 30, 31], dtype=np.int64)
    scene_flag = np.zeros(40, dtype=np.int64)
    scene_flag[10:20] = 3
    scene_flag[20:30] = 6
    scene_flag[30:] = 9
    np.save(dataset_dir / "idx_start.npy", idx_start)
    np.save(dataset_dir / "scene_flag.npy", scene_flag)
    output_dir.mkdir()

    summary = create_trumans_splits(
        dataset_dir,
        output_dir,
        np.asarray([0.5, 0.25, 0.25], dtype=np.float64),
        seed=7,
    )

    split_indices = {
        name: np.load(output_dir / f"{name}_idx.npy")
        for name in ("train", "val", "test")
    }
    split_groups = {
        name: set(scene_flag[idx_start[indices]].tolist())
        for name, indices in split_indices.items()
    }
    assert split_groups["train"].isdisjoint(split_groups["val"])
    assert split_groups["train"].isdisjoint(split_groups["test"])
    assert split_groups["val"].isdisjoint(split_groups["test"])
    assert set(np.concatenate(list(split_indices.values())).tolist()) == set(range(len(idx_start)))
    assert summary["split_required"] is True

