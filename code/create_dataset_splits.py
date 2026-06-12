import argparse
import json
import pickle as pkl
from collections import Counter
from pathlib import Path

import numpy as np


def load_pickle(path):
    with open(path, "rb") as f:
        return pkl.load(f)


def split_scene_names(scene_names, ratios, seed):
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

    train_scenes = set(unique_scenes[:n_train].tolist())
    val_scenes = set(unique_scenes[n_train:n_train + n_val].tolist())
    test_scenes = set(unique_scenes[n_train + n_val:].tolist())
    return {
        "train": train_scenes,
        "val": val_scenes,
        "test": test_scenes,
    }


def split_random_indices(num_samples, ratios, seed):
    rng = np.random.default_rng(seed)
    order = np.arange(num_samples, dtype=np.int64)
    rng.shuffle(order)
    n_train = int(round(num_samples * ratios[0]))
    n_val = int(round(num_samples * ratios[1]))
    return {
        "train": np.sort(order[:n_train]),
        "val": np.sort(order[n_train:n_train + n_val]),
        "test": np.sort(order[n_train + n_val:]),
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


def create_full_horizon_splits(dataset_dir, output_dir, ratios, seed, split_mode):
    sample_scene_names = load_pickle(dataset_dir / "scene_name.pkl")
    num_samples = len(sample_scene_names)
    summary = {
        "format": "full_horizon",
        "seed": seed,
        "split_mode": split_mode,
        "ratios": {
            "train": float(ratios[0]),
            "val": float(ratios[1]),
            "test": float(ratios[2]),
        },
        "splits": {},
    }

    if split_mode == "random":
        index_splits = split_random_indices(num_samples, ratios, seed)
        for split_name, indices in index_splits.items():
            save_split(output_dir, split_name, indices, None)
            scene_counts = Counter(sample_scene_names[int(i)] for i in indices)
            summary["splits"][split_name] = {
                "num_indices": int(len(indices)),
                "num_scenes": int(len(scene_counts)),
                "top_scenes": scene_counts.most_common(10),
            }
            print(f"{split_name:5s}: {len(indices):8d} samples, {len(scene_counts):3d} scenes")
        return summary

    scene_splits = split_scene_names(sample_scene_names, ratios, seed)
    for split_name, scenes in scene_splits.items():
        indices = np.asarray(
            [idx for idx, scene in enumerate(sample_scene_names) if scene in scenes],
            dtype=np.int64,
        )
        save_split(output_dir, split_name, indices, scenes)
        scene_counts = Counter(sample_scene_names[int(i)] for i in indices)
        summary["splits"][split_name] = {
            "num_indices": int(len(indices)),
            "num_scenes": int(len(scenes)),
            "top_scenes": scene_counts.most_common(10),
        }
        print(f"{split_name:5s}: {len(indices):8d} samples, {len(scenes):3d} scenes")
    return summary


def create_window_splits(dataset_dir, output_dir, ratios, seed, motion_dict_rel_path):
    frame_scene_names = load_pickle(dataset_dir / "scene_name.pkl")
    scene_splits = split_scene_names(frame_scene_names, ratios, seed)

    summary = {
        "format": "window",
        "seed": seed,
        "ratios": {
            "train": float(ratios[0]),
            "val": float(ratios[1]),
            "test": float(ratios[2]),
        },
        "splits": {},
    }

    motion_dict_path = dataset_dir / motion_dict_rel_path
    if not motion_dict_path.exists():
        raise FileNotFoundError(f"Motion dictionary does not exist: {motion_dict_path}")

    motion_dict = load_pickle(motion_dict_path)
    start_idx = np.asarray(motion_dict["start_idx"], dtype=np.int64)

    for split_name, scenes in scene_splits.items():
        indices = indices_for_scenes(start_idx, frame_scene_names, scenes)
        save_split(output_dir, split_name, indices, scenes)
        scene_counts = Counter(frame_scene_names[int(start_idx[i])] for i in indices)
        summary["splits"][split_name] = {
            "num_indices": int(len(indices)),
            "num_scenes": int(len(scenes)),
            "top_scenes": scene_counts.most_common(10),
        }
        print(f"{split_name:5s}: {len(indices):8d} windows, {len(scenes):3d} scenes")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Create scene-disjoint train/val/test splits for LINGO/HSI.")
    parser.add_argument("--dataset-dir", default="../dataset")
    parser.add_argument("--motion-dict", default="language_motion_dict/language_motion_dict__inter_and_loco__16.pkl")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--format", choices=("auto", "window", "full_horizon"), default="auto")
    parser.add_argument("--split-mode", choices=("scene", "random"), default="scene")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)

    ratios = np.asarray([args.train_ratio, args.val_ratio, args.test_ratio], dtype=np.float64)
    ratios = ratios / ratios.sum()

    dataset_format = args.format
    if dataset_format == "auto":
        if (dataset_dir / "human_motion.npy").exists() and (dataset_dir / "length.npy").exists():
            dataset_format = "full_horizon"
        else:
            dataset_format = "window"

    if dataset_format == "full_horizon":
        summary = create_full_horizon_splits(dataset_dir, output_dir, ratios, args.seed, args.split_mode)
    else:
        if args.split_mode != "scene":
            raise ValueError("Window-format split only supports --split-mode scene.")
        summary = create_window_splits(dataset_dir, output_dir, ratios, args.seed, args.motion_dict)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved splits to {output_dir}")


if __name__ == "__main__":
    main()
