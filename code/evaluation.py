import argparse
import ast
import csv
import glob
import json
import math
import os
import pickle as pkl
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover - PyYAML is listed in requirements.
    yaml = None

try:
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover - hydra/omegaconf is listed in requirements.
    OmegaConf = None

try:
    from scipy import linalg
    from scipy.ndimage import distance_transform_edt
except Exception:  # pragma: no cover - scipy is listed in requirements.
    linalg = None
    distance_transform_edt = None

try:
    import torch
except Exception:  # pragma: no cover - torch is needed only for SMPL-X files.
    torch = None

try:
    import smplx
except Exception:  # pragma: no cover - smplx is needed only for SMPL-X files.
    smplx = None


DEFAULT_SCENE_GRID = (-4.0, 0.0, -6.0, 4.0, 2.0, 6.0, 400.0, 100.0, 600.0)
DEFAULT_FOOT_JOINTS = (7, 8, 10, 11)
DEFAULT_HAND_JOINTS = (20, 21, 25, 26, 27, 40, 49)
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "config_evaluation.yaml"


@dataclass
class MotionSample:
    name: str
    source: str
    joints: Optional[np.ndarray] = None
    vertices: Optional[np.ndarray] = None
    features: Optional[np.ndarray] = None
    label: Optional[str] = None
    condition_id: Optional[str] = None
    scene_name: Optional[str] = None
    goal: Optional[np.ndarray] = None


def parse_ints(value) -> Tuple[int, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    return tuple(int(item) for item in value.split(",") if item.strip())


def parse_bool_or_none(value):
    if value is None or isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in ("none", "null", ""):
        return None
    if value in ("1", "true", "yes", "y"):
        return True
    if value in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Expected true, false, or null, got {value!r}.")


def load_config(path: Optional[str]) -> Dict:
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        return {}
    if yaml is not None:
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    elif OmegaConf is not None:
        data = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True) or {}
    else:
        data = parse_simple_yaml(config_path)
    if not isinstance(data, dict):
        raise ValueError(f"Evaluation config must be a mapping: {config_path}")
    return data


def parse_simple_yaml(path: Path) -> Dict:
    data = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = strip_yaml_comment(line).strip()
            if not line:
                continue
            if ":" not in line or line.startswith((" ", "-")):
                raise RuntimeError(
                    "Install PyYAML/OmegaConf for nested YAML configs. "
                    f"The fallback parser only supports top-level key: value lines: {path}"
                )
            key, value = line.split(":", 1)
            data[key.strip()] = parse_yaml_scalar(value.strip())
    return data


def strip_yaml_comment(line: str) -> str:
    in_single = False
    in_double = False
    for idx, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:idx]
    return line


def parse_yaml_scalar(value: str):
    if value == "" or value.lower() in ("null", "none", "~"):
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        literal_value = value.replace("null", "None").replace("true", "True").replace("false", "False")
        try:
            return ast.literal_eval(literal_value)
        except (SyntaxError, ValueError):
            inner = value[1:-1].strip()
            if not inner:
                return []
            return [parse_yaml_scalar(item.strip()) for item in inner.split(",")]
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("'\"")


def config_default(config: Dict, key: str, fallback):
    return config.get(key, fallback)


def as_float_array(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return None
    return arr


def normalize_label(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        flat = [str(item) for item in value if item is not None]
        if not flat:
            return None
        counts = {}
        for item in flat:
            counts[item] = counts.get(item, 0) + 1
        return max(counts, key=counts.get)
    return str(value)


def normalize_condition_id(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is None:
        return None
    text = str(value)
    suffix = Path(text).suffix
    if suffix in (".pkl", ".npz", ".npy"):
        return Path(text).stem
    return text


def ensure_motion_array(arr: np.ndarray) -> List[np.ndarray]:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[-1] == 3:
        return [arr[i] for i in range(arr.shape[0])]
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return [arr]
    if arr.ndim == 3 and arr.shape[-1] % 3 == 0:
        return [arr[i].reshape(arr.shape[1], arr.shape[2] // 3, 3) for i in range(arr.shape[0])]
    if arr.ndim == 2 and arr.shape[-1] % 3 == 0:
        return [arr.reshape(arr.shape[0], arr.shape[1] // 3, 3)]
    raise ValueError(f"Cannot interpret array with shape {arr.shape} as joints.")


def find_first_key(data: Dict, names: Sequence[str]):
    for name in names:
        if name in data:
            return data[name]
    return None


def maybe_feature_array(arr: np.ndarray) -> Optional[np.ndarray]:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2 and arr.shape[-1] % 3 != 0:
        return arr
    return None


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pkl.load(f)


def smplx_to_motion(
    data: Dict,
    args: argparse.Namespace,
    need_vertices: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if torch is None or smplx is None:
        raise RuntimeError("torch and smplx are required to evaluate SMPL-X parameter files.")
    if args.smpl_dir is None:
        raise RuntimeError("SMPL-X params found, but --smpl-dir was not provided.")

    transl = as_float_array(data.get("transl"))
    body_pose = as_float_array(data.get("body_pose"))
    global_orient = as_float_array(data.get("global_orient"))
    if transl is None or body_pose is None or global_orient is None:
        raise ValueError("SMPL-X data must contain transl, body_pose, and global_orient.")

    num_frames = transl.shape[0]
    batch_size = min(args.smpl_batch_size, num_frames)
    model = smplx.create(
        args.smpl_dir,
        model_type="smplx",
        gender=args.gender,
        ext="npz",
        num_betas=10,
        use_pca=False,
        batch_size=batch_size,
    ).to(args.device)
    model.eval()

    joints_all = []
    vertices_all = []
    with torch.no_grad():
        for start in range(0, num_frames, batch_size):
            end = min(start + batch_size, num_frames)
            curr = end - start
            if curr != batch_size:
                model = smplx.create(
                    args.smpl_dir,
                    model_type="smplx",
                    gender=args.gender,
                    ext="npz",
                    num_betas=10,
                    use_pca=False,
                    batch_size=curr,
                ).to(args.device)
                model.eval()
            output = model(
                transl=torch.as_tensor(transl[start:end], dtype=torch.float32, device=args.device),
                body_pose=torch.as_tensor(body_pose[start:end], dtype=torch.float32, device=args.device),
                global_orient=torch.as_tensor(global_orient[start:end], dtype=torch.float32, device=args.device),
                return_verts=need_vertices,
            )
            joints_all.append(output.joints.detach().cpu().numpy())
            if need_vertices:
                vertices_all.append(output.vertices.detach().cpu().numpy())

    joints = np.concatenate(joints_all, axis=0)
    if args.joints_ind:
        joints = joints[:, list(args.joints_ind)]
    vertices = np.concatenate(vertices_all, axis=0) if vertices_all else None
    return joints, vertices


def make_samples_from_dict(data: Dict, path: Path, args: argparse.Namespace) -> List[MotionSample]:
    label = normalize_label(find_first_key(data, ("label", "text", "raw_text", "action", "instruction")))
    condition_id = normalize_condition_id(find_first_key(data, ("mm_group_id", "condition_id", "test_setting", "input_pkl_path")))
    scene_name = normalize_label(find_first_key(data, ("scene_name", "scene", "scene_id")))
    goal = find_first_key(data, ("goal", "hand_goal", "hand_location", "target", "target_location", "object_goal"))
    goal = as_float_array(goal)
    if goal is not None:
        goal = goal.reshape(-1, 3)[0]

    feature_value = find_first_key(data, ("features", "feature", "embedding", "embeddings", "motion_features"))
    features = maybe_feature_array(feature_value) if feature_value is not None else None

    joint_value = find_first_key(data, ("joints", "joint", "points", "points_orig", "points_all", "keypoints"))
    vertex_value = find_first_key(data, ("vertices", "verts"))
    joints_list = ensure_motion_array(joint_value) if joint_value is not None else []
    vertices_list = ensure_motion_array(vertex_value) if vertex_value is not None else []

    if not joints_list and {"transl", "body_pose", "global_orient"}.issubset(data.keys()):
        need_vertices = args.compute_vertices or args.body_points == "vertices"
        joints, vertices = smplx_to_motion(data, args, need_vertices)
        joints_list = [joints]
        vertices_list = [vertices] if vertices is not None else []

    if features is not None and not joints_list:
        return [
            MotionSample(
                name=f"{path.stem}_{idx:03d}",
                source=str(path),
                features=features[idx],
                label=label,
                condition_id=condition_id,
                scene_name=scene_name,
                goal=goal,
            )
            for idx in range(features.shape[0])
        ]

    if not joints_list:
        raise ValueError(f"No joints, features, or SMPL-X params found in {path}.")

    samples = []
    for idx, joints in enumerate(joints_list):
        vertices = vertices_list[idx] if idx < len(vertices_list) else None
        feature = features[idx] if features is not None and idx < len(features) else None
        name = path.stem if len(joints_list) == 1 else f"{path.stem}_{idx:03d}"
        samples.append(
            MotionSample(
                name=name,
                source=str(path),
                joints=joints,
                vertices=vertices,
                features=feature,
                label=label,
                condition_id=condition_id,
                scene_name=scene_name,
                goal=goal,
            )
        )
    return samples


def load_motion_file(path: Path, args: argparse.Namespace) -> List[MotionSample]:
    suffix = path.suffix.lower()
    if suffix == ".pkl":
        data = load_pickle(path)
        if isinstance(data, dict):
            return make_samples_from_dict(data, path, args)
        if isinstance(data, list):
            samples = []
            for idx, item in enumerate(data):
                if isinstance(item, dict):
                    curr = make_samples_from_dict(item, Path(f"{path.stem}_{idx:03d}.pkl"), args)
                    for sample in curr:
                        sample.source = str(path)
                    samples.extend(curr)
                else:
                    for midx, joints in enumerate(ensure_motion_array(item)):
                        samples.append(MotionSample(f"{path.stem}_{idx:03d}_{midx:03d}", str(path), joints=joints))
            return samples
        return [
            MotionSample(f"{path.stem}_{idx:03d}", str(path), joints=joints)
            for idx, joints in enumerate(ensure_motion_array(data))
        ]
    if suffix == ".npz":
        data_npz = np.load(path, allow_pickle=True)
        data = {key: data_npz[key] for key in data_npz.files}
        return make_samples_from_dict(data, path, args)
    if suffix == ".npy":
        arr = np.load(path, allow_pickle=True)
        features = maybe_feature_array(arr)
        if features is not None:
            return [
                MotionSample(f"{path.stem}_{idx:03d}", str(path), features=features[idx])
                for idx in range(features.shape[0])
            ]
        return [
            MotionSample(f"{path.stem}_{idx:03d}", str(path), joints=joints)
            for idx, joints in enumerate(ensure_motion_array(arr))
        ]
    raise ValueError(f"Unsupported file type: {path}")


def expand_inputs(paths: Sequence[str]) -> List[Path]:
    expanded = []
    for raw in paths:
        matches = [Path(item) for item in glob.glob(raw)]
        if not matches:
            candidate = Path(raw)
            if candidate.is_dir():
                for suffix in ("*.pkl", "*.npz", "*.npy"):
                    matches.extend(candidate.rglob(suffix))
            elif candidate.exists():
                matches.append(candidate)
        expanded.extend(matches)
    return sorted(set(path.resolve() for path in expanded))


def load_samples(paths: Sequence[str], args: argparse.Namespace) -> List[MotionSample]:
    samples = []
    for path in expand_inputs(paths):
        samples.extend(load_motion_file(path, args))
    if not samples:
        raise RuntimeError(f"No samples loaded from: {paths}")
    return samples


def bool_filter(values: Optional[np.ndarray], expected: Optional[bool], size: int) -> np.ndarray:
    if expected is None or values is None:
        return np.ones(size, dtype=bool)
    return np.asarray(values, dtype=bool) == expected


def load_reference_dataset(args: argparse.Namespace) -> List[MotionSample]:
    folder = Path(args.reference_dataset)
    if not folder.exists():
        raise RuntimeError(f"Reference dataset folder does not exist: {folder}")

    joints_path = folder / args.reference_joints_file
    motion_dict_path = folder / args.reference_motion_dict
    if not joints_path.exists():
        raise RuntimeError(f"GT joints file does not exist: {joints_path}")
    if not motion_dict_path.exists():
        raise RuntimeError(f"Language motion dict does not exist: {motion_dict_path}")

    joints_all = np.load(joints_path, mmap_mode="r", allow_pickle=True)
    motion_dict = load_pickle(motion_dict_path)
    start_idx = np.asarray(motion_dict["start_idx"], dtype=np.int64)
    end_idx = np.asarray(motion_dict["end_idx"], dtype=np.int64)
    text = motion_dict.get("text", [None] * len(start_idx))
    size = len(start_idx)

    expected_span = int(args.reference_window_size) * int(args.reference_step)
    mask = (end_idx - start_idx) == expected_span
    mask &= bool_filter(motion_dict.get("need_scene"), args.reference_need_scene, size)
    mask &= bool_filter(motion_dict.get("need_pelvis_dir"), args.reference_need_pelvis_dir, size)
    mask &= bool_filter(motion_dict.get("need_hand_goal"), args.reference_need_hand_goal, size)
    mask &= bool_filter(motion_dict.get("need_pi"), args.reference_need_pi, size)

    if args.reference_text_contains:
        query = str(args.reference_text_contains).lower()
        text_mask = np.asarray([query in (normalize_label(item) or "").lower() for item in text], dtype=bool)
        mask &= text_mask

    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise RuntimeError("No GT reference clips matched the reference_dataset filters.")

    max_samples = int(args.reference_max_samples)
    if max_samples > 0 and indices.size > max_samples:
        rng = np.random.default_rng(args.reference_sample_seed if args.reference_sample_seed is not None else args.seed)
        indices = rng.choice(indices, size=max_samples, replace=False)

    samples = []
    for idx in indices:
        start = int(start_idx[idx])
        stop = start + expected_span
        clip = np.asarray(joints_all[start:stop:int(args.reference_step)], dtype=np.float32)
        if clip.shape[0] != int(args.reference_window_size):
            continue
        samples.append(
            MotionSample(
                name=f"gt_{idx}",
                source=str(folder),
                joints=clip,
                label=normalize_label(text[idx]),
                condition_id=f"gt_{idx}",
            )
        )
    if not samples:
        raise RuntimeError("GT reference clips were found, but none could be loaded.")
    return samples


def resample_motion(joints: np.ndarray, frames: int) -> np.ndarray:
    if joints.shape[0] == frames:
        return joints
    old_x = np.linspace(0.0, 1.0, joints.shape[0])
    new_x = np.linspace(0.0, 1.0, frames)
    flat = joints.reshape(joints.shape[0], -1)
    resampled = np.stack([np.interp(new_x, old_x, flat[:, i]) for i in range(flat.shape[1])], axis=-1)
    return resampled.reshape(frames, joints.shape[1], 3)


def motion_feature_from_joints(joints: np.ndarray, frames: int) -> np.ndarray:
    joints = resample_motion(np.asarray(joints, dtype=np.float32), frames)
    root = joints[:, :1]
    centered = joints - root
    velocity = np.diff(centered, axis=0, prepend=centered[:1])
    stats = np.concatenate(
        [
            centered.mean(axis=(0, 1)),
            centered.std(axis=(0, 1)),
            velocity.mean(axis=(0, 1)),
            velocity.std(axis=(0, 1)),
        ]
    )
    return np.concatenate([centered.reshape(-1), velocity.reshape(-1), stats], axis=0)


def collect_features(samples: Sequence[MotionSample], frames: int) -> np.ndarray:
    features = []
    for sample in samples:
        if sample.features is not None:
            features.append(np.asarray(sample.features, dtype=np.float32).reshape(-1))
        elif sample.joints is not None:
            features.append(motion_feature_from_joints(sample.joints, frames))
    if not features:
        raise RuntimeError("No features or joints available for distribution metrics.")
    min_dim = min(feature.shape[0] for feature in features)
    return np.stack([feature[:min_dim] for feature in features], axis=0)


def project_features(features: np.ndarray, dim: int, rng: np.random.Generator) -> np.ndarray:
    if dim <= 0 or features.shape[1] <= dim:
        return features
    projection = rng.normal(0.0, 1.0 / math.sqrt(dim), size=(features.shape[1], dim))
    return features.dot(projection).astype(np.float32)


def project_feature_pair(
    real: np.ndarray,
    fake: np.ndarray,
    dim: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if dim <= 0 or real.shape[1] <= dim:
        return real, fake
    projection = rng.normal(0.0, 1.0 / math.sqrt(dim), size=(real.shape[1], dim))
    return real.dot(projection).astype(np.float32), fake.dot(projection).astype(np.float32)


def frechet_distance(real: np.ndarray, fake: np.ndarray, eps: float = 1e-6) -> float:
    mu_real = real.mean(axis=0)
    mu_fake = fake.mean(axis=0)
    cov_real = np.cov(real, rowvar=False)
    cov_fake = np.cov(fake, rowvar=False)
    if cov_real.ndim == 0:
        cov_real = np.array([[float(cov_real)]])
        cov_fake = np.array([[float(cov_fake)]])
        mu_real = mu_real.reshape(1)
        mu_fake = mu_fake.reshape(1)

    covmean = None
    if linalg is not None:
        covmean, _ = linalg.sqrtm(cov_real.dot(cov_fake), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(cov_real.shape[0]) * eps
            covmean = linalg.sqrtm((cov_real + offset).dot(cov_fake + offset))
    if covmean is None:
        vals, vecs = np.linalg.eigh(cov_real.dot(cov_fake))
        covmean = (vecs * np.sqrt(np.maximum(vals, 0.0))).dot(vecs.T)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_real - mu_fake
    return float(diff.dot(diff) + np.trace(cov_real + cov_fake - 2.0 * covmean))


def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = np.sum(a * a, axis=1, keepdims=True)
    b2 = np.sum(b * b, axis=1, keepdims=True).T
    dist2 = np.maximum(a2 + b2 - 2.0 * a.dot(b.T), 0.0)
    return np.sqrt(dist2)


def diversity_score(features: np.ndarray, subset_size: int, rng: np.random.Generator) -> float:
    if len(features) < 2:
        return float("nan")

    if len(features) >= subset_size:
        idx_a = rng.choice(len(features), size=subset_size, replace=False)
        idx_b = rng.choice(len(features), size=subset_size, replace=False)
    else:
        idx_a = rng.choice(len(features), size=subset_size, replace=True)
        idx_b = rng.choice(len(features), size=subset_size, replace=True)

    return float(np.linalg.norm(features[idx_a] - features[idx_b], axis=1).mean())


def multimodality(samples: Sequence[MotionSample], features: np.ndarray, pairs: int, rng: np.random.Generator) -> float:
    label_to_idx: Dict[str, List[int]] = {}
    for idx, sample in enumerate(samples):
        key = sample.condition_id or sample.label
        if key is not None:
            label_to_idx.setdefault(key, []).append(idx)
    values = []
    for indices in label_to_idx.values():
        if len(indices) >= 2:
            values.append(diversity_score(features[indices], pairs, rng))
    return float(np.nanmean(values)) if values else float("nan")


def manifold_precision_recall(real: np.ndarray, fake: np.ndarray, k: int) -> Tuple[float, float, float]:
    k = max(1, min(k, len(real) - 1, len(fake) - 1))
    if k < 1:
        return float("nan"), float("nan"), float("nan")
    real_real = pairwise_distances(real, real)
    fake_fake = pairwise_distances(fake, fake)
    np.fill_diagonal(real_real, np.inf)
    np.fill_diagonal(fake_fake, np.inf)
    real_radius = np.partition(real_real, k - 1, axis=1)[:, k - 1]
    fake_radius = np.partition(fake_fake, k - 1, axis=1)[:, k - 1]
    cross = pairwise_distances(fake, real)
    precision = (cross <= real_radius.reshape(1, -1)).any(axis=1).mean()
    recall = (cross.T <= fake_radius.reshape(1, -1)).any(axis=1).mean()
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    return float(precision), float(recall), float(f1)


def interactive_metrics(
    generated: Sequence[MotionSample],
    reference: Optional[Sequence[MotionSample]],
    args: argparse.Namespace,
) -> Dict[str, float]:
    rng = np.random.default_rng(args.seed)
    gen_features = collect_features(generated, args.feature_frames)
    if reference:
        ref_features = collect_features(reference, args.feature_frames)
        dim = min(gen_features.shape[1], ref_features.shape[1])
        gen_features = gen_features[:, :dim]
        ref_features = ref_features[:, :dim]
        ref_features, gen_features = project_feature_pair(ref_features, gen_features, args.feature_dim, rng)
    else:
        gen_features = project_features(gen_features, args.feature_dim, rng)

    metrics = {
        "diversity": diversity_score(gen_features, args.diversity_pairs, rng),
        "multi_modality": multimodality(generated, gen_features, args.multimodality_pairs, rng),
    }
    if reference:
        metrics["fid"] = frechet_distance(ref_features, gen_features)
        precision, recall, f1 = manifold_precision_recall(ref_features, gen_features, args.precision_k)
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1_score"] = f1
    return metrics


def load_scene_occ(path: Optional[str], scene_name: Optional[str]) -> Optional[np.ndarray]:
    if path is None:
        return None
    scene_path = Path(path)
    if scene_path.is_dir():
        candidates = []
        if scene_name:
            candidates.extend([scene_path / f"{scene_name}.npy", scene_path / f"{scene_name}.npz"])
        candidates.extend(sorted(scene_path.glob("*.npy")))
        candidates.extend(sorted(scene_path.glob("*.npz")))
        scene_path = next((candidate for candidate in candidates if candidate.exists()), None)
        if scene_path is None:
            return None
    if scene_path.suffix.lower() == ".npz":
        data = np.load(scene_path)
        key = "occ" if "occ" in data.files else data.files[0]
        occ = data[key]
    else:
        occ = np.load(scene_path)
    occ = np.asarray(occ)
    while occ.ndim > 3:
        occ = occ[0]
    return occ.astype(bool)


def grid_parts(scene_grid: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.asarray(scene_grid, dtype=np.float64)
    lower = grid[:3]
    upper = grid[3:6]
    dims = grid[6:].astype(int)
    voxel = (upper - lower) / dims
    return lower, dims, voxel


def voxel_indices(points: np.ndarray, scene_grid: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    lower, dims, voxel = grid_parts(scene_grid)
    idx = np.floor((points - lower) / voxel).astype(np.int64)
    valid = np.all(idx >= 0, axis=1) & np.all(idx < dims.reshape(1, 3), axis=1)
    idx[~valid] = 0
    return idx, valid


def penetration_for_points(
    points: np.ndarray,
    occ: np.ndarray,
    scene_grid: Sequence[float],
    out_of_bounds_occupied: bool,
) -> Dict[str, float]:
    if distance_transform_edt is None:
        raise RuntimeError("scipy.ndimage.distance_transform_edt is required for penetration distance.")
    flat = points.reshape(-1, 3)
    idx, valid = voxel_indices(flat, scene_grid)
    occ_flags = occ[idx[:, 0], idx[:, 1], idx[:, 2]]
    if out_of_bounds_occupied:
        occ_flags[~valid] = True
    else:
        occ_flags[~valid] = False

    _, _, voxel = grid_parts(scene_grid)
    distance_field = distance_transform_edt(occ.astype(np.uint8), sampling=voxel)
    dist = distance_field[idx[:, 0], idx[:, 1], idx[:, 2]]
    pene_dist = dist[occ_flags]
    return {
        "pene_scene_percent": float(occ_flags.mean()),
        "pene_mean": float(pene_dist.mean()) if pene_dist.size else 0.0,
        "pene_max": float(pene_dist.max()) if pene_dist.size else 0.0,
    }


def sample_body_points(sample: MotionSample, mode: str) -> Optional[np.ndarray]:
    if mode == "vertices":
        return sample.vertices
    if mode == "joints":
        return sample.joints
    return sample.vertices if sample.vertices is not None else sample.joints


def penetration_metrics(samples: Sequence[MotionSample], args: argparse.Namespace) -> Dict[str, float]:
    per_sample = []
    for sample in samples:
        points = sample_body_points(sample, args.body_points)
        if points is None:
            continue
        occ = load_scene_occ(args.scene_occ, sample.scene_name)
        if occ is None:
            continue
        per_sample.append(
            penetration_for_points(points, occ, args.scene_grid, args.out_of_bounds_occupied)
        )
    return aggregate_dicts(per_sample)


def foot_sliding_for_joints(joints: np.ndarray, args: argparse.Namespace) -> float:
    foot_ids = [idx for idx in args.foot_joints if idx < joints.shape[1]]
    if not foot_ids or joints.shape[0] < 2:
        return float("nan")
    feet = joints[:, foot_ids]
    floor = args.floor_height
    if floor is None:
        floor = float(np.percentile(feet[..., 1], 5))
    disp = feet[1:, :, [0, 2]] - feet[:-1, :, [0, 2]]
    horiz_speed = np.linalg.norm(disp, axis=-1) * args.fps
    vert_speed = np.abs(feet[1:, :, 1] - feet[:-1, :, 1]) * args.fps
    contact = (feet[:-1, :, 1] <= floor + args.contact_height) & (vert_speed <= args.contact_velocity)
    if not np.any(contact):
        return float("nan")
    return float(horiz_speed[contact].mean())


def locomotion_metrics(samples: Sequence[MotionSample], args: argparse.Namespace) -> Dict[str, float]:
    values = []
    for sample in samples:
        curr = {}
        if sample.joints is not None:
            curr["foot_sliding"] = foot_sliding_for_joints(sample.joints, args)
        values.append(curr)
    metrics = aggregate_dicts(values)
    metrics.update(penetration_metrics(samples, args))
    return metrics


def load_goal_map(path: Optional[str]) -> Dict[str, np.ndarray]:
    if path is None:
        return {}
    goal_path = Path(path)
    if goal_path.suffix.lower() == ".json":
        data = json.loads(goal_path.read_text())
        return {str(key): np.asarray(value, dtype=np.float32).reshape(3) for key, value in data.items()}
    goals = {}
    with goal_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get("name") or row.get("id") or row.get("sample")
            if key is None:
                continue
            goals[key] = np.asarray([float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float32)
    return goals


def goal_for_sample(sample: MotionSample, args: argparse.Namespace, goal_map: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    if args.goal is not None:
        return np.asarray(args.goal, dtype=np.float32)
    if sample.goal is not None:
        return sample.goal
    if sample.name in goal_map:
        return goal_map[sample.name]
    source_stem = Path(sample.source).stem
    return goal_map.get(source_stem)


def reaching_for_sample(sample: MotionSample, goal: np.ndarray, args: argparse.Namespace) -> Dict[str, float]:
    if sample.joints is None:
        return {}
    hand_ids = [idx for idx in args.hand_joints if idx < sample.joints.shape[1]]
    if not hand_ids:
        return {}
    hands = sample.joints[:, hand_ids]
    dist = np.linalg.norm(hands - goal.reshape(1, 1, 3), axis=-1).min(axis=1)
    reached = np.flatnonzero(dist <= args.reach_threshold)
    return {
        "reach_error_dist": float(dist.min()),
        "reach_final_error_dist": float(dist[-1]),
        "time_used": float(reached[0] / args.fps) if reached.size else float("nan"),
    }


def reaching_metrics(samples: Sequence[MotionSample], args: argparse.Namespace) -> Dict[str, float]:
    goal_map = load_goal_map(args.goal_file)
    values = []
    for sample in samples:
        goal = goal_for_sample(sample, args, goal_map)
        curr = reaching_for_sample(sample, goal, args) if goal is not None else {}
        values.append(curr)
    metrics = aggregate_dicts(values)
    pene = penetration_metrics(samples, args)
    if "pene_scene_percent" in pene:
        metrics["pene_scene_percent"] = pene["pene_scene_percent"]
    return metrics


def aggregate_dicts(values: Sequence[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted({key for item in values for key in item.keys()})
    out = {}
    for key in keys:
        arr = np.asarray([item[key] for item in values if key in item], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            out[key] = float(arr.mean())
            out[f"{key}_std"] = float(arr.std())
    return out


def write_json(path: Optional[str], metrics: Dict[str, Dict[str, float]]) -> None:
    if path is None:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))


def print_metrics(metrics: Dict[str, Dict[str, float]]) -> None:
    for section, values in metrics.items():
        print(f"\n[{section}]")
        if not values:
            print("  skipped")
            continue
        for key in sorted(values):
            value = values[key]
            if isinstance(value, float) and math.isnan(value):
                print(f"  {key}: nan")
            else:
                print(f"  {key}: {value:.6f}")


def build_argparser(config: Optional[Dict] = None) -> argparse.ArgumentParser:
    config = config or {}
    parser = argparse.ArgumentParser(
        description="Evaluate LINGO/HSI motion generations with the metrics used in Jiang et al. 2024."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config file. CLI arguments override it.")
    parser.add_argument(
        "--generated",
        nargs="+",
        default=config_default(config, "generated", None),
        help="Generated pkl/npz/npy files, directories, or globs.",
    )
    parser.add_argument(
        "--reference",
        nargs="*",
        default=config_default(config, "reference", None),
        help="Reference motions/features for FID and P/R/F1.",
    )
    parser.add_argument(
        "--reference-dataset",
        default=config_default(config, "reference_dataset", None),
        help="LINGO dataset folder used to auto-slice GT reference clips.",
    )
    parser.add_argument(
        "--reference-motion-dict",
        default=config_default(
            config,
            "reference_motion_dict",
            "language_motion_dict/language_motion_dict__inter_and_loco__16.pkl",
        ),
    )
    parser.add_argument("--reference-joints-file", default=config_default(config, "reference_joints_file", "human_joints_aligned.npy"))
    parser.add_argument("--reference-max-samples", type=int, default=config_default(config, "reference_max_samples", 5000))
    parser.add_argument("--reference-sample-seed", type=int, default=config_default(config, "reference_sample_seed", None))
    parser.add_argument("--reference-step", type=int, default=config_default(config, "reference_step", 3))
    parser.add_argument("--reference-window-size", type=int, default=config_default(config, "reference_window_size", 16))
    parser.add_argument("--reference-text-contains", default=config_default(config, "reference_text_contains", None))
    parser.add_argument("--reference-need-scene", type=parse_bool_or_none, default=config_default(config, "reference_need_scene", None))
    parser.add_argument("--reference-need-pelvis-dir", type=parse_bool_or_none, default=config_default(config, "reference_need_pelvis_dir", None))
    parser.add_argument("--reference-need-hand-goal", type=parse_bool_or_none, default=config_default(config, "reference_need_hand_goal", None))
    parser.add_argument("--reference-need-pi", type=parse_bool_or_none, default=config_default(config, "reference_need_pi", None))
    parser.add_argument(
        "--metrics",
        default=config_default(config, "metrics", "all"),
        choices=("all", "interactive", "locomotion", "reaching"),
        help="Metric group to run.",
    )
    parser.add_argument("--output", default=config_default(config, "output", None), help="Optional JSON output path.")
    parser.add_argument("--fps", type=float, default=config_default(config, "fps", 20.0))
    parser.add_argument("--seed", type=int, default=config_default(config, "seed", 1234))
    parser.add_argument("--feature-frames", type=int, default=config_default(config, "feature_frames", 64))
    parser.add_argument("--feature-dim", type=int, default=config_default(config, "feature_dim", 512))
    parser.add_argument("--diversity-pairs", type=int, default=config_default(config, "diversity_pairs", 200))
    parser.add_argument("--multimodality-pairs", type=int, default=config_default(config, "multimodality_pairs", 100))
    parser.add_argument("--precision-k", type=int, default=config_default(config, "precision_k", 3))

    parser.add_argument("--scene-occ", default=config_default(config, "scene_occ", None), help="Scene occupancy .npy/.npz or a directory of scene files.")
    parser.add_argument("--scene-grid", nargs=9, type=float, default=config_default(config, "scene_grid", DEFAULT_SCENE_GRID))
    parser.add_argument("--body-points", choices=("auto", "joints", "vertices"), default=config_default(config, "body_points", "auto"))
    parser.add_argument(
        "--out-of-bounds-occupied",
        dest="out_of_bounds_occupied",
        action="store_true",
        default=config_default(config, "out_of_bounds_occupied", True),
    )
    parser.add_argument("--free-out-of-bounds", dest="out_of_bounds_occupied", action="store_false")

    parser.add_argument("--foot-joints", default=config_default(config, "foot_joints", ",".join(map(str, DEFAULT_FOOT_JOINTS))))
    parser.add_argument("--floor-height", type=float, default=config_default(config, "floor_height", None))
    parser.add_argument("--contact-height", type=float, default=config_default(config, "contact_height", 0.05))
    parser.add_argument("--contact-velocity", type=float, default=config_default(config, "contact_velocity", 0.10))

    parser.add_argument("--goal", nargs=3, type=float, default=config_default(config, "goal", None))
    parser.add_argument("--goal-file", default=config_default(config, "goal_file", None), help="JSON map or CSV with name,x,y,z columns.")
    parser.add_argument("--hand-joints", default=config_default(config, "hand_joints", ",".join(map(str, DEFAULT_HAND_JOINTS))))
    parser.add_argument("--reach-threshold", type=float, default=config_default(config, "reach_threshold", 0.20))

    parser.add_argument("--smpl-dir", default=config_default(config, "smpl_dir", None), help="Required when evaluating pkl files containing SMPL-X params.")
    parser.add_argument("--gender", default=config_default(config, "gender", "male"))
    parser.add_argument("--device", default=config_default(config, "device", "cuda" if torch is not None and torch.cuda.is_available() else "cpu"))
    parser.add_argument("--smpl-batch-size", type=int, default=config_default(config, "smpl_batch_size", 256))
    parser.add_argument("--compute-vertices", action="store_true", default=config_default(config, "compute_vertices", False))
    parser.add_argument("--joints-ind", default=config_default(config, "joints_ind", None), help="Optional comma-separated SMPL-X joint ids to keep.")
    return parser


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args()
    config = load_config(pre_args.config)

    parser = build_argparser(config)
    args = parser.parse_args()
    if not args.generated:
        parser.error("Set generated in the config file or pass --generated on the command line.")
    args.foot_joints = parse_ints(args.foot_joints)
    args.hand_joints = parse_ints(args.hand_joints)
    args.joints_ind = parse_ints(args.joints_ind)

    generated = load_samples(args.generated, args)
    reference = []
    if args.reference:
        reference.extend(load_samples(args.reference, args))
    if args.reference_dataset:
        reference.extend(load_reference_dataset(args))
    reference = reference or None

    results = {}
    if args.metrics in ("all", "interactive"):
        try:
            results["interactive"] = interactive_metrics(generated, reference, args)
        except Exception as exc:
            warnings.warn(f"Interactive metrics skipped: {exc}")
            results["interactive"] = {}
    if args.metrics in ("all", "locomotion"):
        try:
            results["locomotion"] = locomotion_metrics(generated, args)
        except Exception as exc:
            warnings.warn(f"Locomotion metrics skipped: {exc}")
            results["locomotion"] = {}
    if args.metrics in ("all", "reaching"):
        try:
            results["reaching"] = reaching_metrics(generated, args)
        except Exception as exc:
            warnings.warn(f"Reaching metrics skipped: {exc}")
            results["reaching"] = {}

    print_metrics(results)
    write_json(args.output, results)


if __name__ == "__main__":
    main()
