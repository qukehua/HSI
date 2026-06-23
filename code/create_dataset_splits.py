import argparse
import json
import pickle as pkl
from collections import Counter
from pathlib import Path

import numpy as np


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


def parse_args():
    parser = argparse.ArgumentParser(description="Create scene-disjoint window train/val/test splits for LINGO/HSI.")
    parser.add_argument("--dataset-dir", default="/share/qkh/dataset/lingo")
    parser.add_argument("--motion-dict", default="language_motion_dict/language_motion_dict__inter_and_loco__16.pkl")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--window-dir",
        default="/share/qkh/dataset/lingo/window_t16_s3",
        help="Optional preprocessed window dataset folder; scene splits are mirrored to window-dir/splits.",
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
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)

    ratios = np.asarray([args.train_ratio, args.val_ratio, args.test_ratio], dtype=np.float64)
    ratios = ratios / ratios.sum()

    summary = create_window_splits(
        dataset_dir,
        output_dir,
        ratios,
        args.seed,
        args.motion_dict,
        args.min_test_texts,
        args.min_val_texts,
    )

    window_dir = Path(args.window_dir) if args.window_dir else None
    if window_dir is not None and window_dir.exists():
        window_split_dir = window_dir / "splits"
        window_split_dir.mkdir(parents=True, exist_ok=True)
        for split_name in ("train", "val", "test"):
            src = output_dir / f"{split_name}_idx.npy"
            if src.exists():
                np.save(window_split_dir / f"{split_name}_idx.npy", np.load(src))
        print(f"Mirrored window split indices to {window_split_dir}")

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved splits to {output_dir}")


if __name__ == "__main__":
    main()
