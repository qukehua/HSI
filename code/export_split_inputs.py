import argparse
import csv
import json
import pickle as pkl
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


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


MANIFEST_FIELDS = [
    "file",
    "source_index",
    "granularity",
    "scene_name",
    "source_action_text",
    "segment_texts",
    "num_segments",
    "prepend_walk_applied",
    "episode_num_per_segment",
    "sampled_action_frames",
    "max_padded_frames",
    "sample_step",
    "raw_frame_start",
    "raw_frame_end",
    "raw_frame_length",
    "original_lingo_action_id",
    "is_truncated",
    "start_location",
    "end_location",
    "hand_location",
]


def build_manifest_row(
    file_name,
    src_idx,
    scene_name,
    source_action_text,
    data,
    prepend_walk_applied,
    sampled_action_frames,
    max_padded_frames,
    sample_step,
    raw_frame_start,
    raw_frame_end,
    original_lingo_action_id,
    start_location,
    end_location,
    hand_location,
    episode_num,
):
    raw_frame_length = int(raw_frame_end - raw_frame_start)
    raw_sampled_frames = (raw_frame_length + sample_step - 1) // sample_step
    segment_texts = " | ".join(seg["text"] for seg in data)
    return {
        "file": file_name,
        "source_index": src_idx,
        "granularity": "full_action",
        "scene_name": scene_name,
        "source_action_text": source_action_text,
        "segment_texts": segment_texts,
        "num_segments": len(data),
        "prepend_walk_applied": prepend_walk_applied,
        "episode_num_per_segment": int(episode_num),
        "sampled_action_frames": int(sampled_action_frames),
        "max_padded_frames": int(max_padded_frames),
        "sample_step": int(sample_step),
        "raw_frame_start": int(raw_frame_start),
        "raw_frame_end": int(raw_frame_end),
        "raw_frame_length": raw_frame_length,
        "original_lingo_action_id": int(original_lingo_action_id),
        "is_truncated": bool(raw_sampled_frames > max_padded_frames),
        "start_location": start_location.tolist(),
        "end_location": end_location.tolist(),
        "hand_location": hand_location.tolist(),
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


def load_dataset_metadata(dataset_dir):
    metadata_path = dataset_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
        return int(metadata.get("t_max", 120)), int(metadata.get("step", 3))
    return 120, 3


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
    raw_start = np.load(dataset_dir / "raw_start_idx.npy", mmap_mode="r")
    raw_end = np.load(dataset_dir / "raw_end_idx.npy", mmap_mode="r")
    source_index = np.load(dataset_dir / "source_index.npy", mmap_mode="r")
    max_padded_frames, sample_step = load_dataset_metadata(dataset_dir)

    prefix = args.prefix or args.split
    manifest_path = output_dir / f"{prefix}_manifest.csv"
    rows = []

    for out_idx, src_idx in enumerate(
        tqdm(indices, desc=f"Export {args.split}", unit="sample")
    ):
        src_idx = int(src_idx)
        scene_name = str(scene_names[src_idx])
        text = first_text(texts[src_idx])
        mat = np.asarray(mats[src_idx], dtype=np.float64)

        start_location = mat[:3, 3].copy()
        start_location[1] = 0.0
        end_location = local_to_global(pelvis_goals[src_idx], mat)
        end_location[1] = 0.0
        hand_location = local_to_global(hand_goals[src_idx], mat)
        prepend_walk_applied = bool(args.prepend_walk and text.strip().lower() != "walk")

        if prepend_walk_applied:
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
            build_manifest_row(
                file_name=file_name,
                src_idx=src_idx,
                scene_name=scene_name,
                source_action_text=text,
                data=data,
                prepend_walk_applied=prepend_walk_applied,
                sampled_action_frames=lengths[src_idx],
                max_padded_frames=max_padded_frames,
                sample_step=sample_step,
                raw_frame_start=raw_start[src_idx],
                raw_frame_end=raw_end[src_idx],
                original_lingo_action_id=source_index[src_idx],
                start_location=start_location,
                end_location=end_location,
                hand_location=hand_location,
                episode_num=args.episode_num,
            )
        )

    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} {args.split} input pkl files to {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
