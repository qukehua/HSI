import argparse
import json
import pickle as pkl
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OMOMO_TRAIN_FILE = "train_diffusion_manip_window_120_cano_joints24.p"
OMOMO_TEST_FILE = "test_diffusion_manip_window_120_processed_joints24.p"


def load_pickle(path):
    with open(path, "rb") as f:
        return pkl.load(f)


def normalize_text(value):
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    return str(value)


def scene_text_sets(start_idx, frame_scene_names, texts):
    scene_to_texts = {}
    for idx, start in enumerate(start_idx):
        scene = frame_scene_names[int(start)]
        scene_to_texts.setdefault(scene, set()).add(normalize_text(texts[idx]))
    return scene_to_texts


def select_text_covered_scenes(
    unique_scenes,
    target_count,
    target_samples,
    scene_to_texts,
    scene_to_counts,
    min_texts,
    rng,
):
    remaining = list(unique_scenes)
    rng.shuffle(remaining)
    if min_texts <= 0:
        return set(remaining[-target_count:])

    all_texts = set()
    for scene in remaining:
        all_texts.update(scene_to_texts.get(scene, ()))
    if len(all_texts) < min_texts:
        raise RuntimeError(
            f"Cannot cover {min_texts} unique texts; available scenes only have {len(all_texts)} unique texts."
        )

    selected = []
    covered = set()
    selected_count = 0
    while remaining and (len(selected) < target_count or len(covered) < min_texts):
        best_pos = None
        best_score = None
        for pos, scene in enumerate(remaining):
            texts = scene_to_texts.get(scene, set())
            new_texts = len(texts - covered)
            scene_count = max(int(scene_to_counts.get(scene, 1)), 1)
            overflow = max(0, selected_count + scene_count - target_samples * 1.25) / max(target_samples, 1)
            if new_texts == 0 and len(selected) >= target_count:
                score = -float("inf")
            else:
                score = (new_texts + 0.01) / np.sqrt(scene_count) - overflow - pos * 1e-9
            if best_score is None or score > best_score:
                best_score = score
                best_pos = pos
        scene = remaining.pop(best_pos)
        selected.append(scene)
        selected_count += int(scene_to_counts.get(scene, 0))
        covered.update(scene_to_texts.get(scene, set()))

    if len(covered) < min_texts:
        raise RuntimeError(
            f"Could only cover {len(covered)} unique test texts with all available scenes; "
            f"requested {min_texts}."
        )
    return set(selected)


def split_scene_names(
    scene_names,
    ratios,
    seed,
    scene_to_texts=None,
    scene_to_counts=None,
    min_test_texts=0,
    min_val_texts=0,
):
    rng = np.random.default_rng(seed)
    unique_scenes = np.array(sorted(set(scene_names)))
    rng.shuffle(unique_scenes)

    n_total = len(unique_scenes)
    if n_total == 0:
        return {"train": set(), "val": set(), "test": set()}
    if n_total < 3:
        return {
            "train": set(unique_scenes.tolist()),
            "val": set(),
            "test": set(),
        }

    n_train = int(round(n_total * ratios[0]))
    n_val = int(round(n_total * ratios[1]))
    n_train = min(max(n_train, 1), n_total - 2)
    n_val = min(max(n_val, 1), n_total - n_train - 1)
    n_test = n_total - n_train - n_val
    target_samples = int(round(len(scene_names) * ratios[2]))

    if scene_to_texts is not None and min_test_texts > 0:
        test_scenes = select_text_covered_scenes(
            unique_scenes.tolist(),
            n_test,
            target_samples,
            scene_to_texts,
            scene_to_counts or {},
            min_test_texts,
            rng,
        )
        remaining = [scene for scene in unique_scenes.tolist() if scene not in test_scenes]
        n_val = min(n_val, max(len(remaining) - 1, 0))
        if min_val_texts > 0 and n_val > 0:
            val_scenes = select_text_covered_scenes(
                remaining,
                n_val,
                int(round(len(scene_names) * ratios[1])),
                scene_to_texts,
                scene_to_counts or {},
                min_val_texts,
                rng,
            )
        else:
            remaining = sorted(remaining, key=lambda scene: (int((scene_to_counts or {}).get(scene, 0)), scene))
            val_scenes = set(remaining[:n_val])
        train_scenes = set(scene for scene in unique_scenes.tolist() if scene not in test_scenes and scene not in val_scenes)
    else:
        train_scenes = set(unique_scenes[:n_train].tolist())
        val_scenes = set(unique_scenes[n_train:n_train + n_val].tolist())
        test_scenes = set(unique_scenes[n_train + n_val:].tolist())
    return {
        "train": train_scenes,
        "val": val_scenes,
        "test": test_scenes,
    }


def indices_for_scenes(start_idx, frame_scene_names, scene_set):
    return np.asarray(
        [idx for idx, start in enumerate(start_idx) if frame_scene_names[int(start)] in scene_set],
        dtype=np.int64,
    )


def save_split(out_dir, name, indices, scenes):
    np.save(out_dir / f"{name}_idx.npy", indices)
    if scenes is not None:
        with open(out_dir / f"{name}_scenes.txt", "w", encoding="utf-8") as f:
            for scene in sorted(scenes):
                f.write(scene + "\n")


def create_window_splits(dataset_dir, output_dir, ratios, seed, motion_dict_rel_path, min_test_texts, min_val_texts):
    frame_scene_names = load_pickle(dataset_dir / "scene_name.pkl")
    motion_dict_path = dataset_dir / motion_dict_rel_path
    if not motion_dict_path.exists():
        raise FileNotFoundError(f"Motion dictionary does not exist: {motion_dict_path}")

    motion_dict = load_pickle(motion_dict_path)
    start_idx = np.asarray(motion_dict["start_idx"], dtype=np.int64)
    texts = motion_dict.get("text", [""] * len(start_idx))
    scene_to_texts = scene_text_sets(start_idx, frame_scene_names, texts)
    scene_to_counts = Counter(frame_scene_names[int(start)] for start in start_idx)
    scene_splits = split_scene_names(
        [frame_scene_names[int(start)] for start in start_idx],
        ratios,
        seed,
        scene_to_texts=scene_to_texts,
        scene_to_counts=scene_to_counts,
        min_test_texts=min_test_texts,
        min_val_texts=min_val_texts,
    )

    summary = {
        "dataset_type": "lingo",
        "split_required": True,
        "format": "window",
        "seed": seed,
        "ratios": {
            "train": float(ratios[0]),
            "val": float(ratios[1]),
            "test": float(ratios[2]),
        },
        "min_test_texts": int(min_test_texts),
        "min_val_texts": int(min_val_texts),
        "splits": {},
    }

    for split_name, scenes in scene_splits.items():
        indices = indices_for_scenes(start_idx, frame_scene_names, scenes)
        save_split(output_dir, split_name, indices, scenes)
        scene_counts = Counter(frame_scene_names[int(start_idx[i])] for i in indices)
        text_counts = Counter(normalize_text(texts[int(i)]) for i in indices)
        summary["splits"][split_name] = {
            "num_indices": int(len(indices)),
            "num_scenes": int(len(scenes)),
            "num_unique_texts": int(len(text_counts)),
            "top_scenes": scene_counts.most_common(10),
            "top_texts": text_counts.most_common(10),
        }
        print(
            f"{split_name:5s}: {len(indices):8d} windows, "
            f"{len(scenes):3d} scenes, {len(text_counts):3d} texts"
        )
    return summary


def create_trumans_splits(dataset_dir, output_dir, ratios, seed):
    idx_start = np.load(dataset_dir / "idx_start.npy", mmap_mode="r")
    scene_flag = np.load(dataset_dir / "scene_flag.npy", mmap_mode="r")
    if len(idx_start) == 0:
        raise RuntimeError(f"TRUMANS contains no window starts: {dataset_dir / 'idx_start.npy'}")
    if int(np.max(idx_start)) >= len(scene_flag):
        raise ValueError("TRUMANS idx_start.npy contains an index outside scene_flag.npy.")

    window_groups = np.asarray(scene_flag[np.asarray(idx_start, dtype=np.int64)])
    unique_groups = np.unique(window_groups)
    group_splits = split_scene_names([str(value) for value in unique_groups], ratios, seed)
    summary = {
        "dataset_type": "trumans",
        "split_required": True,
        "format": "raw_windows",
        "group_key": "scene_flag",
        "seed": int(seed),
        "ratios": {
            "train": float(ratios[0]),
            "val": float(ratios[1]),
            "test": float(ratios[2]),
        },
        "splits": {},
    }

    for split_name, groups in group_splits.items():
        numeric_groups = np.asarray([int(value) for value in groups], dtype=window_groups.dtype)
        indices = np.flatnonzero(np.isin(window_groups, numeric_groups)).astype(np.int64)
        save_split(output_dir, split_name, indices, groups)
        group_counts = Counter(str(window_groups[index]) for index in indices)
        summary["splits"][split_name] = {
            "num_indices": int(len(indices)),
            "num_groups": int(len(groups)),
            "top_groups": group_counts.most_common(10),
        }
        print(f"{split_name:5s}: {len(indices):8d} windows, {len(groups):3d} scene groups")
    return summary


def _has_all_files(folder, names):
    return all((folder / name).exists() for name in names)


def detect_dataset_type(dataset_dir):
    """Return ``(dataset_type, resolved_data_dir)`` from official file markers."""
    dataset_dir = Path(dataset_dir)
    candidates = [dataset_dir]
    for child_name in ("data", "trumans"):
        child = dataset_dir / child_name
        if child.is_dir():
            candidates.append(child)

    for candidate in candidates:
        if _has_all_files(candidate, (OMOMO_TRAIN_FILE, OMOMO_TEST_FILE)):
            return "omomo", candidate
    for candidate in candidates:
        if _has_all_files(candidate, ("idx_start.npy", "scene_flag.npy", "human_joints.npy")):
            return "trumans", candidate
    for candidate in candidates:
        if (candidate / "scene_name.pkl").exists() and (candidate / "language_motion_dict").is_dir():
            return "lingo", candidate
    raise ValueError(
        f"Could not detect a LINGO, OMOMO, or TRUMANS dataset under {dataset_dir}. "
        "Use --dataset-type to select the expected format and verify the dataset files."
    )


def resolve_dataset(dataset_dir, dataset_type):
    if dataset_type == "auto":
        return detect_dataset_type(dataset_dir)
    detected_type, resolved_dir = detect_dataset_type(dataset_dir)
    if detected_type != dataset_type:
        raise ValueError(
            f"--dataset-type={dataset_type} was requested, but {resolved_dir} looks like {detected_type}."
        )
    return dataset_type, resolved_dir


def split_policy(dataset_type, dataset_dir):
    if dataset_type == "omomo":
        missing = [
            name for name in (OMOMO_TRAIN_FILE, OMOMO_TEST_FILE)
            if not (dataset_dir / name).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"OMOMO must use its official train/test files; missing from {dataset_dir}: {missing}"
            )
        return {
            "dataset_type": "omomo",
            "split_required": False,
            "reason": "official subject-disjoint train/test files already exist",
        }
    if dataset_type == "lingo":
        return {
            "dataset_type": "lingo",
            "split_required": True,
            "reason": "the released motion dictionary has no train/val/test assignment",
        }
    if dataset_type == "trumans":
        return {
            "dataset_type": "trumans",
            "split_required": True,
            "reason": "idx_start.npy contains all windows without train/val/test assignment",
        }
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def mirror_lingo_splits(output_dir, window_dir):
    window_split_dir = window_dir / "splits"
    window_split_dir.mkdir(parents=True, exist_ok=True)
    source_index_path = window_dir / "source_index.npy"
    source_index = np.load(source_index_path) if source_index_path.exists() else None
    window_count = None
    length_path = window_dir / "length.npy"
    if length_path.exists():
        window_count = len(np.load(length_path, mmap_mode="r"))

    for split_name in ("train", "val", "test"):
        source_indices = np.load(output_dir / f"{split_name}_idx.npy").astype(np.int64)
        if source_index is not None:
            split_indices = np.flatnonzero(np.isin(source_index, source_indices)).astype(np.int64)
        elif window_count is not None:
            split_indices = source_indices[source_indices < window_count]
        else:
            split_indices = source_indices
        np.save(window_split_dir / f"{split_name}_idx.npy", split_indices)
    print(f"Mirrored LINGO split indices to {window_split_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create leak-free splits only for datasets that do not already provide official splits."
    )
    parser.add_argument("--dataset-dir", default=str(PROJECT_ROOT / "dataset" / "lingo"))
    parser.add_argument(
        "--dataset-type",
        choices=("auto", "lingo", "omomo", "trumans"),
        default="auto",
        help="Dataset format. auto detects it from official file markers.",
    )
    parser.add_argument("--motion-dict", default="language_motion_dict/language_motion_dict__inter_and_loco__16.pkl")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--window-dir",
        default=None,
        help="Optional preprocessed LINGO window folder; generated splits are mirrored to window-dir/splits.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--min-test-texts", type=int, default=100)
    parser.add_argument("--min-val-texts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_type, dataset_dir = resolve_dataset(Path(args.dataset_dir), args.dataset_type)
    policy = split_policy(dataset_type, dataset_dir)
    print(json.dumps({**policy, "dataset_dir": str(dataset_dir)}, indent=2))
    if not policy["split_required"]:
        print("Skipping split generation; the dataset's official split files remain authoritative.")
        return

    ratios = np.asarray([args.train_ratio, args.val_ratio, args.test_ratio], dtype=np.float64)
    if not np.isfinite(ratios).all() or np.any(ratios < 0) or ratios.sum() <= 0:
        raise ValueError("Split ratios must be finite, non-negative, and have a positive sum.")
    ratios = ratios / ratios.sum()

    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    if dataset_type == "lingo":
        summary = create_window_splits(
            dataset_dir,
            output_dir,
            ratios,
            args.seed,
            args.motion_dict,
            args.min_test_texts,
            args.min_val_texts,
        )
        window_dir = Path(args.window_dir) if args.window_dir else dataset_dir / "window_t16_s3"
        if window_dir.exists():
            mirror_lingo_splits(output_dir, window_dir)
    else:
        summary = create_trumans_splits(dataset_dir, output_dir, ratios, args.seed)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved splits to {output_dir}")


if __name__ == "__main__":
    main()
