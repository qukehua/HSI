import argparse
import csv
import pickle as pkl
from pathlib import Path

import numpy as np


def load_pickle(path):
    with open(path, "rb") as f:
        return pkl.load(f)


def save_pickle(value, path):
    with open(path, "wb") as f:
        pkl.dump(value, f)


def first_text(value):
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    return str(value)


def local_to_global(point, mat):
    point = np.asarray(point, dtype=np.float64).reshape(3)
    mat = np.asarray(mat, dtype=np.float64)
    return mat[:3, :3].dot(point) + mat[:3, 3]


def make_segment(scene_name, text, start_location, end_location, hand_location, episode_num, seg_num):
    return {
        "scene_name": scene_name,
        "text": text,
        "start_location": np.asarray(start_location, dtype=np.float64),
        "end_location": np.asarray(end_location, dtype=np.float64),
        "hand_location": np.asarray(hand_location, dtype=np.float64),
        "episode_num": int(episode_num),
        "seg_num": int(seg_num),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export sample.py input pkl files from a full-horizon dataset split."
    )
    parser.add_argument("--dataset-dir", default="/share/qkh/dataset/lingo/full_horizon_t120_s3")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--split-dir", default="splits")
    parser.add_argument("--output-dir", default="../results/inputs_test")
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--max-samples", type=int, default=0, help="0 exports every sample in the split.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--episode-num",
        type=int,
        default=1,
        help="Number of sample.py generation windows for each exported segment.",
    )
    parser.add_argument(
        "--prepend-walk",
        action="store_true",
        help="Export [walk, text] two-segment inputs instead of a single text segment.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    split_path = dataset_dir / args.split_dir / f"{args.split}_idx.npy"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_path}. "
            "Create it first with create_dataset_splits.py or preprocess_full_horizon_dataset.py."
        )

    indices = np.load(split_path).astype(np.int64)
    if args.max_samples > 0 and len(indices) > args.max_samples:
        rng = np.random.default_rng(args.seed)
        indices = np.sort(rng.choice(indices, size=args.max_samples, replace=False))

    scene_names = load_pickle(dataset_dir / "scene_name.pkl")
    texts = load_pickle(dataset_dir / "text.pkl")
    mats = np.load(dataset_dir / "mat.npy", mmap_mode="r")
    pelvis_goals = np.load(dataset_dir / "pelvis_goal.npy", mmap_mode="r")
    hand_goals = np.load(dataset_dir / "hand_goal.npy", mmap_mode="r")
    lengths = np.load(dataset_dir / "length.npy", mmap_mode="r")

    prefix = args.prefix or args.split
    manifest_path = output_dir / f"{prefix}_manifest.csv"
    rows = []

    for out_idx, src_idx in enumerate(indices):
        src_idx = int(src_idx)
        scene_name = str(scene_names[src_idx])
        text = first_text(texts[src_idx])
        mat = np.asarray(mats[src_idx], dtype=np.float64)

        start_location = mat[:3, 3].copy()
        start_location[1] = 0.0
        end_location = local_to_global(pelvis_goals[src_idx], mat)
        end_location[1] = 0.0
        hand_location = local_to_global(hand_goals[src_idx], mat)

        if args.prepend_walk and text.strip().lower() != "walk":
            seg_num = 2
            data = [
                make_segment(
                    scene_name,
                    "walk",
                    start_location,
                    end_location,
                    hand_location,
                    args.episode_num,
                    seg_num,
                ),
                make_segment(
                    scene_name,
                    text,
                    start_location,
                    end_location,
                    hand_location,
                    args.episode_num,
                    seg_num,
                ),
            ]
        else:
            seg_num = 1
            data = [
                make_segment(
                    scene_name,
                    text,
                    start_location,
                    end_location,
                    hand_location,
                    args.episode_num,
                    seg_num,
                )
            ]

        file_name = f"{prefix}-{out_idx:05d}__idx-{src_idx}.pkl"
        save_pickle(data, output_dir / file_name)
        rows.append(
            {
                "file": file_name,
                "source_index": src_idx,
                "scene_name": scene_name,
                "text": text,
                "length": int(lengths[src_idx]),
                "seg_num": seg_num,
                "episode_num": int(args.episode_num),
                "start_location": start_location.tolist(),
                "end_location": end_location.tolist(),
                "hand_location": hand_location.tolist(),
            }
        )

    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["file"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} {args.split} input pkl files to {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
