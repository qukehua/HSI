import argparse
import pickle as pkl
from collections import Counter
from pathlib import Path

import numpy as np


DEFAULT_SCENE_GRID = np.array([-4.0, 0.0, -6.0, 4.0, 2.0, 6.0, 400.0, 100.0, 600.0])


def load_pickle(path):
    with open(path, "rb") as f:
        return pkl.load(f)


def normalize_text(item):
    if isinstance(item, (list, tuple)):
        return str(item[0])
    return str(item)


def collect_commands(dataset_dir, num_commands, rng):
    motion_dict_path = dataset_dir / "language_motion_dict" / "language_motion_dict__inter_and_loco__16.pkl"
    motion_dict = load_pickle(motion_dict_path)
    texts = [normalize_text(item) for item in motion_dict["text"]]
    counts = Counter(texts)

    commands = []
    seen = set()
    for text, _ in counts.most_common():
        lowered = text.lower()
        if text in seen:
            continue
        if lowered == "walk" or "maintains " in lowered:
            continue
        commands.append(text)
        seen.add(text)
        if len(commands) >= num_commands:
            break

    if len(commands) < num_commands:
        remaining = [text for text in counts if text not in seen and text.lower() != "walk"]
        rng.shuffle(remaining)
        commands.extend(remaining[: num_commands - len(commands)])
    return commands


def free_xz_points(scene_occ, scene_grid, rng, count, min_distance):
    dims = scene_grid[6:].astype(int)
    lower = scene_grid[:3]
    upper = scene_grid[3:6]
    voxel = (upper - lower) / dims

    occ = np.asarray(scene_occ).astype(bool)
    while occ.ndim > 3:
        occ = occ[0]

    y_low = max(0, int(0.05 / voxel[1]))
    y_high = min(dims[1], int(1.20 / voxel[1]) + 1)
    free_map = ~occ[:, y_low:y_high, :].any(axis=1)
    candidates = np.argwhere(free_map)
    if len(candidates) == 0:
        raise RuntimeError("No free x/z candidates found in scene occupancy.")

    points = []
    attempts = 0
    while len(points) < count and attempts < count * 500:
        attempts += 1
        ix, iz = candidates[rng.integers(0, len(candidates))]
        x = lower[0] + (ix + 0.5) * voxel[0]
        z = lower[2] + (iz + 0.5) * voxel[2]
        point = np.array([x, 0.0, z], dtype=np.float64)
        if all(np.linalg.norm(point[[0, 2]] - prev[[0, 2]]) >= min_distance for prev in points):
            points.append(point)

    if len(points) < count:
        raise RuntimeError(f"Could only sample {len(points)} free points, requested {count}.")
    return points


def make_hand_location(end_location, rng):
    offset = rng.normal(0.0, 0.25, size=3)
    offset[1] = rng.uniform(0.55, 0.90)
    return end_location + offset


def make_input(scene_name, command, start_location, end_location, hand_location, episode_num):
    seg_num = 2
    common = {
        "scene_name": scene_name,
        "start_location": start_location.astype(np.float64),
        "end_location": end_location.astype(np.float64),
        "hand_location": hand_location.astype(np.float64),
        "episode_num": int(episode_num),
        "seg_num": seg_num,
    }
    return [
        {**common, "text": "walk"},
        {**common, "text": command},
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Create evaluation input pkl files for sample_lingo.py.")
    parser.add_argument("--dataset-dir", default="../dataset")
    parser.add_argument("--output-dir", default="../results/inputs_eval")
    parser.add_argument("--num-files", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scene-grid", nargs=9, type=float, default=DEFAULT_SCENE_GRID.tolist())
    parser.add_argument("--episode-num", type=int, default=10)
    parser.add_argument("--min-distance", type=float, default=1.5)
    parser.add_argument("--prefix", default="eval")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_files = sorted((dataset_dir / "Scene_vis").glob("*.npy"))
    if not scene_files:
        raise RuntimeError(f"No Scene_vis .npy files found under {dataset_dir}")

    commands = collect_commands(dataset_dir, args.num_files, rng)
    scene_grid = np.asarray(args.scene_grid, dtype=np.float64)
    manifest = []

    for idx in range(args.num_files):
        scene_file = scene_files[idx % len(scene_files)]
        scene_name = scene_file.stem
        scene_occ = np.load(scene_file, mmap_mode="r")
        start_location, end_location = free_xz_points(scene_occ, scene_grid, rng, 2, args.min_distance)
        hand_location = make_hand_location(end_location, rng)
        command = commands[idx % len(commands)]
        data = make_input(scene_name, command, start_location, end_location, hand_location, args.episode_num)

        name = f"{args.prefix}-{idx:03d}.pkl"
        out_path = output_dir / name
        with open(out_path, "wb") as f:
            pkl.dump(data, f)
        manifest.append((name, scene_name, command, start_location.tolist(), end_location.tolist(), hand_location.tolist()))

    manifest_path = output_dir / f"{args.prefix}_manifest.tsv"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("file\tscene_name\ttext\tstart_location\tend_location\thand_location\n")
        for row in manifest:
            f.write("\t".join(map(str, row)) + "\n")

    print(f"Saved {len(manifest)} input files to {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
