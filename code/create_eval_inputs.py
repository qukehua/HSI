import argparse
import csv
import pickle as pkl
from pathlib import Path

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


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


INTERACTIVE_HAND_KEYWORDS = (
    "type on ",
    "write on",
    "wash ",
    "punch ",
    "kick ",
    "pick up",
    "put down",
    "drink ",
    "eat ",
    "read ",
    "play ",
    "listen ",
    "wave ",
    "talk on",
    "blow out",
    "take shower",
    "take photo",
)


def is_interactive_text(text):
    lowered = text.lower()
    return any(keyword in lowered for keyword in INTERACTIVE_HAND_KEYWORDS)


def fallback_hand_location(start_location, end_location, text):
    hand_location = end_location.astype(np.float64).copy()
    if is_interactive_text(text):
        hand_location[1] = max(float(start_location[1]), float(end_location[1])) + 0.60
        delta_xz = end_location[[0, 2]] - start_location[[0, 2]]
        norm = np.linalg.norm(delta_xz)
        if norm > 1e-6:
            hand_location[[0, 2]] += 0.20 * delta_xz / norm
        return hand_location

    hand_location[1] = max(float(start_location[1]), float(end_location[1])) + 0.55
    return hand_location


def window_goals(src_idx, motion_dict, joints):
    start_idx = int(motion_dict["start_idx"][src_idx])
    end_idx = int(motion_dict["end_idx"][src_idx])
    text = first_text(motion_dict["text"][src_idx])
    end_frame = max(start_idx, end_idx - 3)

    start_location = joints[start_idx, 0].astype(np.float64).copy()
    start_location[1] = 0.0

    end_location = start_location.copy()
    if motion_dict["need_pelvis_dir"][src_idx]:
        if "sit down" in text or "lie down" in text:
            end_location = joints[int(motion_dict["end_range"][src_idx]), 0].astype(np.float64).copy()
        else:
            end_location = joints[end_frame, 0].astype(np.float64).copy()
        end_location[1] = 0.0

    left_hand = int(motion_dict["left_hand_inter_frame"][src_idx])
    right_hand = int(motion_dict["right_hand_inter_frame"][src_idx])
    if left_hand != -1:
        hand_location = joints[left_hand, 24].astype(np.float64).copy()
    elif right_hand != -1:
        hand_location = joints[right_hand, 26].astype(np.float64).copy()
    elif bool(motion_dict["need_hand_goal"][src_idx]) or is_interactive_text(text):
        hand_location = joints[end_frame, [20, 21]].mean(axis=0).astype(np.float64)
    else:
        hand_location = fallback_hand_location(start_location, end_location, text)

    return start_location, end_location, hand_location, text, start_idx, end_idx


MANIFEST_FIELDS = [
    "file",
    "source_index",
    "granularity",
    "scene_name",
    "source_window_text",
    "segment_texts",
    "num_segments",
    "prepend_walk_applied",
    "episode_num_per_segment",
    "raw_frame_span",
    "sampled_window_frames",
    "window_frame_start",
    "window_frame_end",
    "full_action_id",
    "full_action_frame_start",
    "full_action_frame_end",
    "full_action_raw_length",
    "start_location",
    "end_location",
    "hand_location",
]


def resolve_full_action(frame_start, root_start, root_end):
    action_id = int(np.searchsorted(root_end, frame_start, side="right") - 1)
    action_id = max(action_id, 0)
    return {
        "full_action_id": action_id,
        "full_action_frame_start": int(root_start[action_id]),
        "full_action_frame_end": int(root_end[action_id]),
        "full_action_raw_length": int(root_end[action_id] - root_start[action_id]),
    }


def build_manifest_row(
    file_name,
    src_idx,
    scene_name,
    source_window_text,
    data,
    prepend_walk_applied,
    frame_start,
    frame_end,
    full_action,
    start_location,
    end_location,
    hand_location,
    episode_num,
    sampled_window_frames,
):
    segment_texts = " | ".join(seg["text"] for seg in data)
    return {
        "file": file_name,
        "source_index": src_idx,
        "granularity": "window_clip",
        "scene_name": scene_name,
        "source_window_text": source_window_text,
        "segment_texts": segment_texts,
        "num_segments": len(data),
        "prepend_walk_applied": prepend_walk_applied,
        "episode_num_per_segment": int(episode_num),
        "raw_frame_span": int(frame_end - frame_start),
        "sampled_window_frames": int(sampled_window_frames),
        "window_frame_start": int(frame_start),
        "window_frame_end": int(frame_end),
        **full_action,
        "start_location": start_location.tolist(),
        "end_location": end_location.tolist(),
        "hand_location": hand_location.tolist(),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create inputs_test pkl files by sampling unique text commands from a dataset split."
    )
    parser.add_argument("--dataset-dir", default="../../datasets/lingo")
    parser.add_argument(
        "--motion-dict",
        default="language_motion_dict/language_motion_dict__inter_and_loco__16.pkl",
    )
    parser.add_argument("--joints-file", default="human_joints_aligned.npy")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--split-dir", default="splits")
    parser.add_argument("--output-dir", default="../results/inputs_test")
    parser.add_argument("--prefix", default=None)
    parser.add_argument(
        "--num-files",
        type=int,
        default=None,
        help="Deprecated alias for --num-texts. Kept for old commands.",
    )
    parser.add_argument(
        "--num-texts",
        type=int,
        default=100,
        help="Sample this many different source texts from the split (0 = export all unique texts).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument(
        "--episode-num",
        type=int,
        default=10,
        help="Number of sample.py generation windows for each exported segment.",
    )
    parser.add_argument(
        "--prepend-walk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export [walk, text] two-segment inputs for non-walk clips (default: true).",
    )
    return parser.parse_args()


def select_unique_text_indices(indices, motion_dict, num_texts, seed):
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(indices, dtype=np.int64).copy()
    rng.shuffle(shuffled)

    text_to_indices = {}
    for src_idx in shuffled:
        text = first_text(motion_dict["text"][int(src_idx)])
        text_to_indices.setdefault(text, []).append(int(src_idx))

    texts = list(text_to_indices)
    if num_texts > 0 and len(texts) > num_texts:
        selected_texts = rng.choice(np.asarray(texts, dtype=object), size=num_texts, replace=False).tolist()
    else:
        selected_texts = texts

    selected_indices = []
    for text in selected_texts:
        group = text_to_indices[text]
        selected_indices.append(int(group[int(rng.integers(len(group)))]))
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    return selected_indices[np.argsort(selected_indices)], len(text_to_indices), len(selected_texts)


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    split_path = dataset_dir / args.split_dir / f"{args.split}_idx.npy"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_path}. "
            "Create it first with create_dataset_splits.py."
        )

    motion_dict = load_pickle(dataset_dir / args.motion_dict)
    indices = np.load(split_path).astype(np.int64)

    expected_span = int(args.window_size) * int(args.step)
    start_idx = np.asarray(motion_dict["start_idx"], dtype=np.int64)
    end_idx = np.asarray(motion_dict["end_idx"], dtype=np.int64)
    valid = (end_idx - start_idx) == expected_span
    indices = np.intersect1d(indices, np.flatnonzero(valid), assume_unique=False)
    if len(indices) == 0:
        raise RuntimeError(
            f"No window clips matched span={expected_span} in split {args.split}."
        )

    num_texts = args.num_texts
    if args.num_files is not None:
        num_texts = args.num_files
    indices, available_texts, selected_texts = select_unique_text_indices(
        indices, motion_dict, num_texts, args.seed
    )
    if num_texts > 0 and selected_texts < num_texts:
        print(
            f"Warning: split {args.split} only has {available_texts} unique texts after span filtering; "
            f"exporting {selected_texts} files."
        )

    joints = np.load(dataset_dir / args.joints_file, mmap_mode="r")
    scene_names = load_pickle(dataset_dir / "scene_name.pkl")
    root_start = np.load(dataset_dir / "start_idx.npy")
    root_end = np.load(dataset_dir / "end_idx.npy")

    prefix = args.prefix or args.split
    manifest_path = output_dir / f"{prefix}_manifest.csv"
    rows = []

    for out_idx, src_idx in enumerate(
        tqdm(indices, desc=f"Export {args.split}", unit="sample")
    ):
        src_idx = int(src_idx)
        start_location, end_location, hand_location, text, frame_start, frame_end = window_goals(
            src_idx, motion_dict, joints
        )
        scene_name = str(scene_names[frame_start])
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
                source_window_text=text,
                data=data,
                prepend_walk_applied=prepend_walk_applied,
                frame_start=frame_start,
                frame_end=frame_end,
                full_action=resolve_full_action(frame_start, root_start, root_end),
                start_location=start_location,
                end_location=end_location,
                hand_location=hand_location,
                episode_num=args.episode_num,
                sampled_window_frames=args.window_size,
            )
        )

    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} {args.split} input pkl files to {output_dir}")
    print(f"Manifest: {manifest_path}")
    print("Run sample.py with mm_num_repeats=30 to generate 30 outputs per input.")


if __name__ == "__main__":
    main()
