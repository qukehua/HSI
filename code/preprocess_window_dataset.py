import argparse
import json
import pickle as pkl
from pathlib import Path

import numpy as np

try:
    from scipy.spatial.transform import Rotation as R
except ModuleNotFoundError:
    R = None


def load_pickle(path):
    with open(path, "rb") as f:
        return pkl.load(f)


def save_pickle(value, path):
    with open(path, "wb") as f:
        pkl.dump(value, f)


def first_text(item):
    if isinstance(item, (list, tuple)):
        return str(item[0]) if item else ""
    return str(item)


def normalize_points(points, coord_min, coord_max):
    shape = points.shape
    points = points.reshape(-1, 3)
    points = -1.0 + 2.0 * (points - coord_min) / (coord_max - coord_min)
    return points.reshape(shape)


def rotvec_to_matrix(rotvec):
    rotvec = np.asarray(rotvec, dtype=np.float32)
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = rotvec / theta
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float32)
    return (
        np.eye(3, dtype=np.float32)
        + np.sin(theta) * skew
        + (1.0 - np.cos(theta)) * (skew @ skew)
    ).astype(np.float32)


def y_rotation(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def yaw_cancel_matrix(rotvec):
    if R is not None:
        init_euler = R.from_rotvec(rotvec).as_euler("zxy")
        return R.from_euler("zxy", [0.0, 0.0, -init_euler[2]]).as_matrix().astype(np.float32)

    rot = rotvec_to_matrix(rotvec)
    yaw = np.arctan2(-rot[2, 0], rot[0, 0])
    return y_rotation(-yaw)


def make_local_motion(joints, orient, frame_ids):
    motion = np.asarray(joints[frame_ids], dtype=np.float32).copy()
    init_shift = np.array([motion[0, 0, 0], 0.0, motion[0, 0, 2]], dtype=np.float32)
    motion -= init_shift

    init_orient = np.asarray(orient[frame_ids[0]], dtype=np.float32)
    local_rot = yaw_cancel_matrix(init_orient)
    motion = motion @ local_rot.T

    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = np.linalg.inv(local_rot.T).T
    mat[:3, 3] = init_shift
    return motion, mat, init_shift, local_rot


def local_point(point, init_shift, local_rot):
    point = np.asarray(point, dtype=np.float32).copy()
    point -= init_shift
    return point @ local_rot.T


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a LINGO-style autoregressive window dataset from language_motion_dict."
    )
    parser.add_argument("--dataset-dir", default="../dataset/lingo")
    parser.add_argument("--output-dir", default="../dataset/lingo/window_t16_s3")
    parser.add_argument(
        "--motion-dict",
        default="language_motion_dict/language_motion_dict__inter_and_loco__16.pkl",
    )
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument(
        "--motion-state-len",
        type=int,
        default=4,
        help="Number of boundary-history frames saved for the optional motion-state token.",
    )
    parser.add_argument(
        "--motion-state-prefix-len",
        type=int,
        default=2,
        help="Number of current-window prefix frames included at the end of each motion-state history.",
    )
    parser.add_argument(
        "--terminal-margin",
        type=int,
        default=None,
        help="Raw-frame margin for labeling a window as task-completing. Defaults to --step.",
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--motion-file", default="human_joints_aligned.npy")
    parser.add_argument("--orient-file", default="human_orient.npy")
    parser.add_argument("--norm-file", default="norm_inter_and_loco__16frames.npy")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def split_indices(n, ratios, seed):
    rng = np.random.default_rng(seed)
    order = np.arange(n, dtype=np.int64)
    rng.shuffle(order)
    ratios = np.asarray(ratios, dtype=np.float64)
    ratios = ratios / ratios.sum()
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    return {
        "train": np.sort(order[:n_train]),
        "val": np.sort(order[n_train:n_train + n_val]),
        "test": np.sort(order[n_train + n_val:]),
    }


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    motion_dict = load_pickle(dataset_dir / args.motion_dict)
    source_count = len(motion_dict["start_idx"])
    if args.max_samples > 0:
        source_count = min(source_count, args.max_samples)

    joints = np.load(dataset_dir / args.motion_file, mmap_mode="r")
    orient = np.load(dataset_dir / args.orient_file, mmap_mode="r")
    frame_scene_names = load_pickle(dataset_dir / "scene_name.pkl")
    text2features = load_pickle(dataset_dir / "text2features_idx.pkl")
    clip_features = np.load(dataset_dir / "clip_features.npy", mmap_mode="r")

    coord_min = coord_max = None
    if args.normalize:
        norm = np.load(dataset_dir / args.norm_file)
        coord_min = norm[0].astype(np.float32)
        coord_max = norm[1].astype(np.float32)

    num_joints = joints.shape[1]
    motion_dim = num_joints * 3
    n = int(source_count)
    t = int(args.window_size)
    state_len = max(0, int(args.motion_state_len))
    state_prefix_len = max(1, int(args.motion_state_prefix_len))

    human_motion = np.lib.format.open_memmap(
        output_dir / "human_motion.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n, t, motion_dim),
    )
    valid_mask = np.lib.format.open_memmap(output_dir / "valid_mask.npy", mode="w+", dtype=np.bool_, shape=(n, t))
    motion_state = None
    motion_state_mask = None
    if state_len > 0:
        motion_state = np.lib.format.open_memmap(
            output_dir / "motion_state.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n, state_len, motion_dim),
        )
        motion_state_mask = np.lib.format.open_memmap(
            output_dir / "motion_state_mask.npy",
            mode="w+",
            dtype=np.bool_,
            shape=(n, state_len),
        )
    text_emb = np.lib.format.open_memmap(
        output_dir / "text_emb.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n, 1, clip_features.shape[1]),
    )
    mat = np.lib.format.open_memmap(output_dir / "mat.npy", mode="w+", dtype=np.float32, shape=(n, 4, 4))
    pelvis_goal = np.lib.format.open_memmap(output_dir / "pelvis_goal.npy", mode="w+", dtype=np.float32, shape=(n, 3))
    hand_goal = np.lib.format.open_memmap(output_dir / "hand_goal.npy", mode="w+", dtype=np.float32, shape=(n, 3))

    human_motion[:] = 0.0
    valid_mask[:] = False
    if motion_state is not None:
        motion_state[:] = 0.0
        motion_state_mask[:] = False
    text_emb[:] = 0.0
    mat[:] = np.eye(4, dtype=np.float32)
    pelvis_goal[:] = 0.0
    hand_goal[:] = 0.0

    lengths = np.zeros(n, dtype=np.int32)
    raw_start_out = np.zeros(n, dtype=np.int32)
    raw_end_out = np.zeros(n, dtype=np.int32)
    raw_length_out = np.zeros(n, dtype=np.int32)
    is_pick = np.zeros(n, dtype=np.bool_)
    object_present = np.zeros(n, dtype=np.bool_)
    is_terminal_window = np.zeros(n, dtype=np.bool_)
    need_scene = np.zeros(n, dtype=np.bool_)
    need_pelvis_dir = np.zeros(n, dtype=np.bool_)
    need_pi = np.zeros(n, dtype=np.bool_)
    pi = np.zeros(n, dtype=np.int32)
    scene_names = []
    texts = []
    missing_text_features = []

    for out_i in range(n):
        start = int(motion_dict["start_idx"][out_i])
        end = int(motion_dict["end_idx"][out_i])
        frame_ids = np.arange(start, end, args.step, dtype=np.int64)[:t]
        write_len = min(len(frame_ids), t)
        if write_len == 0:
            raise RuntimeError(f"Empty frame window at source index {out_i}: start={start}, end={end}")

        motion, sample_mat, init_shift, local_rot = make_local_motion(joints, orient, frame_ids[:write_len])
        goal = motion[write_len - 1, 0].copy()
        goal[1] = 0.0
        if motion_state is not None:
            prefix_frames = min(state_prefix_len, write_len)
            state_end = start + (prefix_frames - 1) * args.step
            state_ids = state_end - np.arange(state_len - 1, -1, -1, dtype=np.int64) * args.step
            min_state_frame = int(motion_dict["start_range"][out_i]) if "start_range" in motion_dict else 0
            state_valid = (state_ids >= min_state_frame) & (state_ids < joints.shape[0])
            state_ids_safe = np.where(state_valid, state_ids, start).astype(np.int64)
            state_motion = np.asarray(joints[state_ids_safe], dtype=np.float32).copy()
            state_motion -= init_shift
            state_motion = state_motion @ local_rot.T
            if args.normalize:
                state_motion = normalize_points(state_motion, coord_min, coord_max)
            motion_state[out_i] = state_motion.reshape(state_len, motion_dim)
            motion_state_mask[out_i] = state_valid
        if args.normalize:
            motion = normalize_points(motion, coord_min, coord_max)

        human_motion[out_i, :write_len] = motion.reshape(write_len, motion_dim)
        valid_mask[out_i, :write_len] = True
        mat[out_i] = sample_mat
        pelvis_goal[out_i] = goal

        lh = int(motion_dict["left_hand_inter_frame"][out_i])
        rh = int(motion_dict["right_hand_inter_frame"][out_i])
        if start <= lh < end:
            hand_goal[out_i] = local_point(joints[lh, 24], init_shift, local_rot)
            is_pick[out_i] = True
        elif start <= rh < end:
            hand_goal[out_i] = local_point(joints[rh, 26], init_shift, local_rot)
            is_pick[out_i] = True

        text = first_text(motion_dict["text"][out_i])
        texts.append(text)
        scene_names.append(frame_scene_names[start])
        need_scene[out_i] = bool(motion_dict["need_scene"][out_i])
        need_pelvis_dir[out_i] = bool(motion_dict["need_pelvis_dir"][out_i])
        need_pi[out_i] = bool(motion_dict["need_pi"][out_i])
        pi[out_i] = int(motion_dict["pi"][out_i])
        object_present[out_i] = is_pick[out_i]
        if "is_tail" in motion_dict:
            is_terminal_window[out_i] = bool(motion_dict["is_tail"][out_i])
        elif "end_range" in motion_dict:
            terminal_margin = args.step if args.terminal_margin is None else args.terminal_margin
            end_range = int(motion_dict["end_range"][out_i])
            frames_to_end = end_range - end
            is_terminal_window[out_i] = end_range >= 0 and 0 < frames_to_end <= terminal_margin
        else:
            is_terminal_window[out_i] = write_len < t

        feature_idx = text2features.get(text)
        if feature_idx is None:
            missing_text_features.append((int(out_i), text))
        else:
            feature = np.asarray(clip_features[feature_idx], dtype=np.float32)
            norm = np.linalg.norm(feature)
            text_emb[out_i, 0] = feature / max(norm, 1e-8)

        lengths[out_i] = write_len
        raw_start_out[out_i] = start
        raw_end_out[out_i] = end
        raw_length_out[out_i] = end - start

        if (out_i + 1) % 10000 == 0 or out_i + 1 == n:
            print(f"Processed {out_i + 1}/{n}")

    np.save(output_dir / "length.npy", lengths)
    np.save(output_dir / "raw_start_idx.npy", raw_start_out)
    np.save(output_dir / "raw_end_idx.npy", raw_end_out)
    np.save(output_dir / "raw_length.npy", raw_length_out)
    np.save(output_dir / "source_index.npy", np.arange(n, dtype=np.int64))
    np.save(output_dir / "is_pick.npy", is_pick)
    np.save(output_dir / "object_present.npy", object_present)
    np.save(output_dir / "is_terminal_window.npy", is_terminal_window)
    np.save(output_dir / "need_scene.npy", need_scene)
    np.save(output_dir / "need_pelvis_dir.npy", need_pelvis_dir)
    np.save(output_dir / "need_pi.npy", need_pi)
    np.save(output_dir / "pi.npy", pi)
    save_pickle(texts, output_dir / "text.pkl")
    save_pickle(scene_names, output_dir / "scene_name.pkl")

    split_dir = output_dir / "splits"
    split_dir.mkdir(exist_ok=True)
    for split_name, split_idx in split_indices(
        n,
        (args.train_ratio, args.val_ratio, args.test_ratio),
        args.seed,
    ).items():
        np.save(split_dir / f"{split_name}_idx.npy", split_idx)

    metadata = {
        "dataset_dir": str(dataset_dir),
        "motion_dict": str(dataset_dir / args.motion_dict),
        "num_samples": int(n),
        "t_max": int(t),
        "step": int(args.step),
        "motion_state_len": int(state_len),
        "motion_state_prefix_len": int(state_prefix_len),
        "normalize": bool(args.normalize),
        "motion_shape": [int(n), int(t), int(motion_dim)],
        "num_joints": int(num_joints),
        "length_min": int(lengths.min()),
        "length_max": int(lengths.max()),
        "length_mean": float(lengths.mean()),
        "need_pi_count": int(need_pi.sum()),
        "terminal_window_count": int(is_terminal_window.sum()),
        "missing_text_feature_count": len(missing_text_features),
        "split_mode": "random_window",
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    if missing_text_features:
        with open(output_dir / "missing_text_features.json", "w", encoding="utf-8") as f:
            json.dump(
                [{"source_index": idx, "text": text} for idx, text in missing_text_features[:1000]],
                f,
                indent=2,
            )

    print(json.dumps(metadata, indent=2))
    print(f"Saved autoregressive window dataset to {output_dir}")


if __name__ == "__main__":
    main()
