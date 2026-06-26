import argparse
import ast
import csv
import glob
import json
import math
import os
import pickle as pkl
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is listed in requirements.
    tqdm = None

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

_SCENE_PATH_CACHE: Dict[Tuple[str, Optional[str]], Optional[Path]] = {}
_SCENE_OCC_CACHE: Dict[str, np.ndarray] = {}
_DISTANCE_FIELD_CACHE: Dict[Tuple[str, Tuple[float, ...]], np.ndarray] = {}
_OBJ_VERT_CACHE: Dict[str, np.ndarray] = {}


def progress_iter(iterable, desc: str, enabled: bool = True, total: Optional[int] = None, leave: bool = False):
    if tqdm is None or not enabled:
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=leave)


def progress_write(message: str, enabled: bool = True) -> None:
    if not enabled:
        return
    if tqdm is None:
        print(message)
    else:
        tqdm.write(message)


@dataclass
class MotionSample:
    name: str
    source: str
    joints: Optional[np.ndarray] = None
    vertices: Optional[np.ndarray] = None
    object_vertices: Optional[np.ndarray] = None
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


def ensure_object_array(arr: np.ndarray) -> List[np.ndarray]:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[-1] == 3:
        return [arr[i] for i in range(arr.shape[0])]
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return [arr]
    if arr.ndim == 2 and arr.shape[-1] == 3:
        return [arr]
    if arr.ndim == 2 and arr.shape[-1] % 3 == 0:
        return [arr.reshape(arr.shape[0], arr.shape[1] // 3, 3)]
    raise ValueError(f"Cannot interpret array with shape {arr.shape} as object vertices.")


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
    try:
        with path.open("rb") as f:
            return pkl.load(f)
    except ModuleNotFoundError as exc:
        if exc.name != "joblib":
            raise
        try:
            import joblib
        except ModuleNotFoundError as joblib_exc:
            raise ModuleNotFoundError(
                f"{path} appears to require joblib. Install joblib or convert the file to a standard pickle."
            ) from joblib_exc
        return joblib.load(path)


def load_obj_vertices(path: Path) -> np.ndarray:
    key = stable_path_key(path)
    if key in _OBJ_VERT_CACHE:
        return _OBJ_VERT_CACHE[key]
    vertices = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise RuntimeError(f"No vertices found in OBJ file: {path}")
    arr = np.asarray(vertices, dtype=np.float32)
    _OBJ_VERT_CACHE[key] = arr
    return arr


def transform_object_vertices(rest_vertices: np.ndarray, scale, rot_mat, trans) -> np.ndarray:
    rest_vertices = np.asarray(rest_vertices, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32).reshape(-1)
    rot_mat = np.asarray(rot_mat, dtype=np.float32).reshape(-1, 3, 3)
    trans = np.asarray(trans, dtype=np.float32)
    if trans.ndim == 3 and trans.shape[-1] == 1:
        trans = trans[..., 0]
    trans = trans.reshape(-1, 3)
    length = min(len(scale), len(rot_mat), len(trans))
    verts = np.einsum("tij,nj->tni", rot_mat[:length], rest_vertices)
    verts = verts * scale[:length, None, None] + trans[:length, None, :]
    return verts.astype(np.float32)


def infer_omomo_object_mesh_dir(path: Path, args: argparse.Namespace) -> Optional[Path]:
    if args.object_mesh_dir not in [None, "None", "none", "null", ""]:
        return Path(args.object_mesh_dir)
    for parent in [path.parent, *path.parents]:
        candidate = parent / "captured_objects"
        if candidate.exists():
            return candidate
    return None


def omomo_object_vertices_from_record(data: Dict, path: Path, args: argparse.Namespace) -> Optional[np.ndarray]:
    required = ("seq_name", "obj_trans", "obj_rot_mat", "obj_scale")
    if not all(key in data for key in required):
        return None
    mesh_dir = infer_omomo_object_mesh_dir(path, args)
    if mesh_dir is None:
        return None
    object_name = str(data["seq_name"]).split("_")[1]
    if object_name in ("mop", "vacuum"):
        top_path = mesh_dir / f"{object_name}_cleaned_simplified_top.obj"
        bottom_path = mesh_dir / f"{object_name}_cleaned_simplified_bottom.obj"
        if not top_path.exists() or not bottom_path.exists():
            return None
        top = transform_object_vertices(
            load_obj_vertices(top_path),
            data["obj_scale"],
            data["obj_rot_mat"],
            data["obj_trans"],
        )
        bottom = transform_object_vertices(
            load_obj_vertices(bottom_path),
            data.get("obj_bottom_scale", data["obj_scale"]),
            data.get("obj_bottom_rot_mat", data["obj_rot_mat"]),
            data.get("obj_bottom_trans", data["obj_trans"]),
        )
        length = min(len(top), len(bottom))
        return np.concatenate([top[:length], bottom[:length]], axis=1).astype(np.float32)

    mesh_path = mesh_dir / f"{object_name}_cleaned_simplified.obj"
    if not mesh_path.exists():
        return None
    return transform_object_vertices(
        load_obj_vertices(mesh_path),
        data["obj_scale"],
        data["obj_rot_mat"],
        data["obj_trans"],
    )


def select_joints(joints: np.ndarray, joint_ids: Tuple[int, ...], context: str) -> np.ndarray:
    if not joint_ids:
        return joints
    if joints.ndim != 3:
        raise ValueError(f"{context}: expected joints shape [T, J, 3], got {joints.shape}.")
    max_id = max(joint_ids)
    if max_id >= joints.shape[1]:
        raise ValueError(
            f"{context}: joint id {max_id} is out of range for {joints.shape[1]} joints. "
            "Check reference_joints_ind / motion_evaluator_joints_ind."
        )
    return joints[:, list(joint_ids)]


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
    condition_id = normalize_condition_id(find_first_key(data, ("mm_group_id", "condition_id", "test_setting", "input_pkl_path", "seq_name")))
    if "seq_name" in data and "start_t_idx" in data:
        condition_id = f"{data['seq_name']}_{int(data['start_t_idx'])}_{int(data.get('end_t_idx', -1))}"
    scene_name = normalize_label(find_first_key(data, ("scene_name", "scene", "scene_id")))
    goal = find_first_key(data, ("goal", "hand_goal", "hand_location", "target", "target_location", "object_goal"))
    goal = as_float_array(goal)
    if goal is not None:
        goal = goal.reshape(-1, 3)[0]

    feature_value = find_first_key(data, ("features", "feature", "embedding", "embeddings", "motion_features"))
    features = maybe_feature_array(feature_value) if feature_value is not None else None

    joint_value = find_first_key(data, ("joints", "joint", "points", "points_orig", "points_all", "keypoints"))
    if joint_value is None and "motion" in data:
        motion_value = np.asarray(data["motion"], dtype=np.float32)
        if motion_value.ndim == 2 and motion_value.shape[-1] >= 72:
            joint_value = motion_value[:, :72].reshape(motion_value.shape[0], 24, 3)
    vertex_value = find_first_key(data, ("vertices", "verts"))
    object_value = find_first_key(
        data,
        (
            "object_vertices",
            "object_verts",
            "obj_vertices",
            "obj_verts",
            "object_points",
            "obj_points",
            "object_mesh_verts",
            "obj_mesh_verts",
        ),
    )
    joints_list = ensure_motion_array(joint_value) if joint_value is not None else []
    vertices_list = ensure_motion_array(vertex_value) if vertex_value is not None else []
    object_list = ensure_object_array(object_value) if object_value is not None else []
    if not object_list:
        omomo_object = omomo_object_vertices_from_record(data, path, args)
        if omomo_object is not None:
            object_list = [omomo_object]

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
        object_vertices = None
        if object_list:
            object_vertices = object_list[idx] if idx < len(object_list) else object_list[0]
        samples.append(
            MotionSample(
                name=name,
                source=str(path),
                joints=joints,
                vertices=vertices,
                object_vertices=object_vertices,
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
    if suffix in (".pkl", ".p"):
        data = load_pickle(path)
        if isinstance(data, dict):
            if data and all(isinstance(item, dict) for item in data.values()):
                samples = []
                for key in sorted(data.keys(), key=lambda item: str(item)):
                    curr = make_samples_from_dict(data[key], path.parent / f"{path.stem}_{key}{path.suffix}", args)
                    for sample in curr:
                        sample.source = str(path)
                    samples.extend(curr)
                return samples
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


def group_key_for_path(path: Path) -> str:
    match = re.search(r"idx[-_](\d+)", path.as_posix())
    if match is not None:
        return f"idx-{match.group(1)}"
    return path.parent.name


def subsample_paths_by_group(
    paths: Sequence[Path],
    max_groups: int,
    seed: Optional[int],
) -> List[Path]:
    if max_groups <= 0:
        return list(paths)
    groups: Dict[str, List[Path]] = {}
    for path in paths:
        groups.setdefault(group_key_for_path(path), []).append(path)
    if len(groups) <= max_groups:
        return list(paths)
    rng = np.random.default_rng(seed)
    selected_keys = rng.choice(list(groups.keys()), size=max_groups, replace=False)
    result: List[Path] = []
    for key in selected_keys:
        result.extend(sorted(groups[key]))
    return result


def load_samples(
    paths: Sequence[str],
    args: argparse.Namespace,
    desc: str = "Load samples",
    max_idx: Optional[int] = None,
    sample_seed: Optional[int] = None,
) -> List[MotionSample]:
    samples = []
    input_paths = expand_inputs(paths)
    if max_idx is not None:
        input_paths = subsample_paths_by_group(input_paths, int(max_idx), sample_seed)
    for path in progress_iter(input_paths, desc=desc, enabled=args.show_progress, total=len(input_paths)):
        samples.extend(load_motion_file(path, args))
    if not samples:
        raise RuntimeError(f"No samples loaded from: {paths}")
    return samples


def bool_filter(values: Optional[np.ndarray], expected: Optional[bool], size: int) -> np.ndarray:
    if expected is None or values is None:
        return np.ones(size, dtype=bool)
    return np.asarray(values, dtype=bool) == expected


def resolve_relative_path(base: Path, value: Optional[str]) -> Optional[Path]:
    if value in [None, "None", "none", "null", ""]:
        return None
    path = Path(value)
    return path if path.is_absolute() else base / path


def load_optional_array(path: Path) -> Optional[np.ndarray]:
    return np.load(path, mmap_mode="r") if path.exists() else None


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
    candidate_indices = np.arange(size)

    if args.reference_split not in [None, "None", "none", "null"]:
        split_dataset = resolve_relative_path(folder, args.reference_split_dataset) or folder
        split_path = split_dataset / args.reference_split_dir / f"{args.reference_split}_idx.npy"
        if not split_path.exists():
            raise RuntimeError(f"Reference split file does not exist: {split_path}")
        candidate_indices = np.load(split_path).astype(np.int64)
        source_index_path = resolve_relative_path(split_dataset, args.reference_split_source_index)
        if source_index_path is None and split_dataset.resolve() != folder.resolve():
            candidate = split_dataset / "source_index.npy"
            source_index_path = candidate if candidate.exists() else None
        if source_index_path is not None:
            if not source_index_path.exists():
                raise RuntimeError(f"Reference split source_index file does not exist: {source_index_path}")
            source_index = np.load(source_index_path).astype(np.int64)
            if candidate_indices.size and candidate_indices.max() >= len(source_index):
                raise RuntimeError(
                    f"Reference split index max={candidate_indices.max()} exceeds "
                    f"source_index length={len(source_index)}."
                )
            candidate_indices = source_index[candidate_indices]
        if candidate_indices.size and candidate_indices.max() >= size:
            raise RuntimeError(
                f"Mapped reference split index max={candidate_indices.max()} exceeds "
                f"motion dictionary length={size}."
            )

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

    mask_indices = np.flatnonzero(mask)
    indices = np.intersect1d(candidate_indices, mask_indices, assume_unique=False)
    if indices.size == 0:
        raise RuntimeError("No GT reference clips matched the reference_dataset filters.")

    max_samples = int(args.reference_max_samples)
    if max_samples > 0 and indices.size > max_samples:
        rng = np.random.default_rng(args.reference_sample_seed if args.reference_sample_seed is not None else args.seed)
        indices = rng.choice(indices, size=max_samples, replace=False)

    samples = []
    for idx in progress_iter(indices, desc="Load GT clips", enabled=args.show_progress, total=len(indices)):
        start = int(start_idx[idx])
        stop = start + expected_span
        clip = np.asarray(joints_all[start:stop:int(args.reference_step)], dtype=np.float32)
        if clip.shape[0] != int(args.reference_window_size):
            continue
        clip = select_joints(clip, args.reference_joints_ind, f"reference clip gt_{idx}")
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


def load_reference_clips_for_indices(
    args: argparse.Namespace,
    indices: Sequence[int],
    desc: str = "Load paired GT",
) -> List[MotionSample]:
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
    expected_span = int(args.reference_window_size) * int(args.reference_step)

    unique_indices = np.unique(np.asarray(indices, dtype=np.int64))
    samples = []
    for idx in progress_iter(unique_indices, desc=desc, enabled=args.show_progress, total=len(unique_indices)):
        idx = int(idx)
        if idx < 0 or idx >= len(start_idx):
            continue
        if int(end_idx[idx] - start_idx[idx]) != expected_span:
            continue
        start = int(start_idx[idx])
        stop = start + expected_span
        clip = np.asarray(joints_all[start:stop:int(args.reference_step)], dtype=np.float32)
        if clip.shape[0] != int(args.reference_window_size):
            continue
        clip = select_joints(clip, args.reference_joints_ind, f"paired reference clip gt_{idx}")
        samples.append(
            MotionSample(
                name=f"gt_{idx}",
                source=str(folder),
                joints=clip,
                label=normalize_label(text[idx]),
                condition_id=f"gt_{idx}",
            )
        )
    return samples


def paired_reference_for_generated(
    generated: Sequence[MotionSample],
    reference: Optional[Sequence[MotionSample]],
    args: argparse.Namespace,
) -> Optional[Sequence[MotionSample]]:
    if not args.reference_dataset:
        return reference
    pair_indices = sorted(
        {
            int(pair_id)
            for sample in generated
            if (pair_id := sample_pair_id(sample)) is not None
        }
    )
    if not pair_indices:
        return reference
    return load_reference_clips_for_indices(args, pair_indices)


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


def collect_features(
    samples: Sequence[MotionSample],
    frames: int,
    desc: str = "Collect features",
    show_progress: bool = True,
) -> np.ndarray:
    features = []
    for sample in progress_iter(samples, desc=desc, enabled=show_progress, total=len(samples)):
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


def resolve_torch_device(device_name: str):
    if torch is None:
        raise RuntimeError("torch is required when --motion-evaluator-checkpoint is set.")
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def load_motion_evaluator(args: argparse.Namespace):
    if args.motion_evaluator_checkpoint in [None, "None", "none", "null", ""]:
        return None
    if torch is None:
        raise RuntimeError("torch is required to load a motion evaluator checkpoint.")
    from models.motion_evaluator import build_motion_evaluator_from_checkpoint

    device = resolve_torch_device(args.motion_evaluator_device or args.device)
    checkpoint_path = Path(args.motion_evaluator_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_motion_evaluator_from_checkpoint(checkpoint, map_location=device)
    model.eval()
    return model, checkpoint.get("config", model.config), device


def sample_motion_for_evaluator(
    sample: MotionSample,
    expected_dim: int,
    joint_ids: Tuple[int, ...],
) -> np.ndarray:
    if sample.joints is None:
        raise RuntimeError(
            f"{sample.name} has no joints. Evaluator-space metrics require joints; "
            "use plain feature inputs only without --motion-evaluator-checkpoint."
        )
    joints = select_joints(np.asarray(sample.joints, dtype=np.float32), joint_ids, sample.name)
    flat = joints.reshape(joints.shape[0], -1)
    if flat.shape[-1] != expected_dim:
        raise RuntimeError(
            f"{sample.name} motion dim {flat.shape[-1]} does not match evaluator dim {expected_dim}. "
            "Use a checkpoint trained for this dataset/joint set, or set reference_joints_ind / "
            "motion_evaluator_joints_ind consistently."
        )
    return flat


def collect_evaluator_features(
    samples: Sequence[MotionSample],
    evaluator_bundle,
    args: argparse.Namespace,
    desc: str = "Evaluator features",
) -> np.ndarray:
    if evaluator_bundle is None:
        raise RuntimeError("No motion evaluator was loaded.")
    model, config, device = evaluator_bundle
    expected_dim = int(config["motion_dim"])
    batch_size = int(args.motion_evaluator_batch_size)
    features = []

    with torch.no_grad():
        for start in progress_iter(
            range(0, len(samples), batch_size),
            desc=desc,
            enabled=args.show_progress,
            total=math.ceil(len(samples) / max(batch_size, 1)),
        ):
            batch_samples = samples[start:start + batch_size]
            motions = [
                sample_motion_for_evaluator(sample, expected_dim, args.motion_evaluator_joints_ind)
                for sample in batch_samples
            ]
            lengths = np.asarray([motion.shape[0] for motion in motions], dtype=np.int64)
            max_len = int(lengths.max())
            padded = np.zeros((len(motions), max_len, expected_dim), dtype=np.float32)
            for idx, motion in enumerate(motions):
                padded[idx, : motion.shape[0]] = motion
            motion_tensor = torch.as_tensor(padded, dtype=torch.float32, device=device)
            length_tensor = torch.as_tensor(lengths, dtype=torch.long, device=device)
            embedding = model.encode_motion(
                motion_tensor,
                length_tensor,
                normalize=args.motion_evaluator_normalize_embeddings,
            )
            features.append(embedding.detach().cpu().numpy())

    if not features:
        raise RuntimeError("No evaluator features were produced.")
    return np.concatenate(features, axis=0).astype(np.float32)


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
    evaluator_bundle = load_motion_evaluator(args)
    if evaluator_bundle is None:
        warnings.warn(
            "Interactive metrics skipped: motion_evaluator_checkpoint is required. "
            "Train with train_motion_evaluator.py and set motion_evaluator_checkpoint in the evaluation config.",
            stacklevel=2,
        )
        return {}

    progress_write(
        f"Using motion evaluator features from {args.motion_evaluator_checkpoint}",
        args.show_progress,
    )
    gen_features = collect_evaluator_features(
        generated,
        evaluator_bundle,
        args,
        desc="Generated evaluator features",
    )
    if reference:
        ref_features = collect_evaluator_features(
            reference,
            evaluator_bundle,
            args,
            desc="Reference evaluator features",
        )
    else:
        ref_features = None

    progress_write("Computing distribution metrics...", args.show_progress)
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


def sample_pair_id(sample: MotionSample) -> Optional[str]:
    fields = [sample.condition_id, sample.name, sample.source]
    patterns = (
        r"idx[-_](\d+)",
        r"gt_(\d+)",
        r"gt_full_(\d+)",
        r"source[_-]?index[-_=](\d+)",
    )
    for field in fields:
        if field is None:
            continue
        text = str(field)
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
    if sample.condition_id is not None:
        return str(sample.condition_id)
    return None


def align_motion_to_reference(generated: np.ndarray, reference: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    generated = np.asarray(generated, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    joint_count = min(generated.shape[1], reference.shape[1])
    generated = generated[:, :joint_count]
    reference = reference[:, :joint_count]

    if mode == "resample":
        generated = resample_motion(generated, reference.shape[0])
    elif mode == "min":
        length = min(generated.shape[0], reference.shape[0])
        generated = generated[:length]
        reference = reference[:length]
    else:
        raise ValueError(f"Unknown mpjpe_frame_alignment={mode!r}. Use resample or min.")
    return generated, reference


def mpjpe_for_pair(generated: np.ndarray, reference: np.ndarray, args: argparse.Namespace) -> float:
    generated, reference = align_motion_to_reference(generated, reference, args.mpjpe_frame_alignment)
    dist = np.linalg.norm(generated - reference, axis=-1)
    return float(dist.mean())


def trajectory_metrics_for_pair(
    generated: np.ndarray,
    reference: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, float]:
    generated, reference = align_motion_to_reference(generated, reference, args.mpjpe_frame_alignment)
    joint_id = min(max(int(args.trajectory_joint), 0), generated.shape[1] - 1, reference.shape[1] - 1)
    pred_traj = generated[:, joint_id]
    gt_traj = reference[:, joint_id]
    dist = np.linalg.norm(pred_traj - gt_traj, axis=-1)
    return {
        "goal_error": float(dist[-1]),
        "traj_error": float(dist.mean()),
        "traj_similarity": float((dist < float(args.trajectory_threshold)).mean()),
    }


def paired_metrics(
    generated: Sequence[MotionSample],
    reference: Optional[Sequence[MotionSample]],
    args: argparse.Namespace,
) -> Dict[str, float]:
    if not reference:
        return {}

    reference_by_id = {}
    for sample in reference:
        key = sample_pair_id(sample)
        if key is not None and sample.joints is not None:
            reference_by_id[key] = sample

    mpjpe_values = []
    goal_errors = []
    traj_errors = []
    traj_similarities = []
    for sample in progress_iter(generated, desc="Paired MPJPE", enabled=args.show_progress, total=len(generated)):
        if sample.joints is None:
            continue
        key = sample_pair_id(sample)
        ref = reference_by_id.get(key)
        if ref is None or ref.joints is None:
            continue
        mpjpe_values.append(mpjpe_for_pair(sample.joints, ref.joints, args))
        traj = trajectory_metrics_for_pair(sample.joints, ref.joints, args)
        goal_errors.append(traj["goal_error"])
        traj_errors.append(traj["traj_error"])
        traj_similarities.append(traj["traj_similarity"])

    if not mpjpe_values:
        return {}
    return {
        "mpjpe": float(np.mean(mpjpe_values)),
        "goal_error": float(np.mean(goal_errors)),
        "traj_error": float(np.mean(traj_errors)),
        "traj_similarity": float(np.mean(traj_similarities)),
    }


def stable_path_key(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def resolve_scene_occ_file(path: Optional[str], scene_name: Optional[str]) -> Optional[Path]:
    if path is None:
        return None
    scene_path = Path(path)
    cache_key = (stable_path_key(scene_path), scene_name)
    if cache_key in _SCENE_PATH_CACHE:
        return _SCENE_PATH_CACHE[cache_key]

    resolved = scene_path
    if scene_path.is_dir():
        candidates = []
        if scene_name:
            candidates.extend([scene_path / f"{scene_name}.npy", scene_path / f"{scene_name}.npz"])
        candidates.extend(sorted(scene_path.glob("*.npy")))
        candidates.extend(sorted(scene_path.glob("*.npz")))
        resolved = next((candidate for candidate in candidates if candidate.exists()), None)

    _SCENE_PATH_CACHE[cache_key] = resolved
    return resolved


def load_scene_occ_entry(path: Optional[str], scene_name: Optional[str]) -> Tuple[Optional[str], Optional[np.ndarray]]:
    scene_path = resolve_scene_occ_file(path, scene_name)
    if scene_path is None:
        return None, None

    scene_key = stable_path_key(scene_path)
    if scene_key in _SCENE_OCC_CACHE:
        return scene_key, _SCENE_OCC_CACHE[scene_key]

    if scene_path.suffix.lower() == ".npz":
        data = np.load(scene_path)
        key = "occ" if "occ" in data.files else data.files[0]
        occ = data[key]
    else:
        occ = np.load(scene_path)
    occ = np.asarray(occ)
    while occ.ndim > 3:
        occ = occ[0]
    occ = occ.astype(bool)
    _SCENE_OCC_CACHE[scene_key] = occ
    return scene_key, occ


def load_scene_occ(path: Optional[str], scene_name: Optional[str]) -> Optional[np.ndarray]:
    _, occ = load_scene_occ_entry(path, scene_name)
    return occ


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


def grid_cache_key(scene_grid: Sequence[float]) -> Tuple[float, ...]:
    return tuple(float(item) for item in np.asarray(scene_grid, dtype=np.float64).reshape(-1))


def distance_field_for_scene(
    scene_key: str,
    occ: np.ndarray,
    scene_grid: Sequence[float],
    voxel: np.ndarray,
    show_progress: bool,
) -> np.ndarray:
    if distance_transform_edt is None:
        raise RuntimeError("scipy.ndimage.distance_transform_edt is required for penetration distance.")

    cache_key = (scene_key, grid_cache_key(scene_grid))
    if cache_key not in _DISTANCE_FIELD_CACHE:
        progress_write(f"Computing distance field for {Path(scene_key).name}...", show_progress)
        _DISTANCE_FIELD_CACHE[cache_key] = distance_transform_edt(occ.astype(np.uint8), sampling=voxel)
    return _DISTANCE_FIELD_CACHE[cache_key]


def penetration_for_points(
    points: np.ndarray,
    occ: np.ndarray,
    scene_grid: Sequence[float],
    out_of_bounds_occupied: bool,
    scene_key: str = "scene",
    include_distance: bool = True,
    show_progress: bool = True,
) -> Dict[str, float]:
    _, dims, voxel = grid_parts(scene_grid)
    if tuple(occ.shape) != tuple(dims):
        raise RuntimeError(
            f"Scene occupancy shape {tuple(occ.shape)} does not match scene_grid dims {tuple(dims)} "
            f"for {scene_key}. Check scene_occ and scene_grid."
        )

    flat = points.reshape(-1, 3)
    idx, valid = voxel_indices(flat, scene_grid)
    occ_flags = occ[idx[:, 0], idx[:, 1], idx[:, 2]]
    if out_of_bounds_occupied:
        occ_flags[~valid] = True
    else:
        occ_flags[~valid] = False

    metrics = {"pene_scene_percent": float(occ_flags.mean())}
    if include_distance and np.any(occ_flags):
        distance_field = distance_field_for_scene(scene_key, occ, scene_grid, voxel, show_progress)
        dist = distance_field[idx[:, 0], idx[:, 1], idx[:, 2]]
        pene_dist = dist[occ_flags]
        metrics["pene_mean"] = float(pene_dist.mean()) if pene_dist.size else 0.0
        metrics["pene_max"] = float(pene_dist.max()) if pene_dist.size else 0.0
    elif include_distance:
        metrics["pene_mean"] = 0.0
        metrics["pene_max"] = 0.0
    return metrics


def sample_body_points(sample: MotionSample, mode: str) -> Optional[np.ndarray]:
    if mode == "vertices":
        return sample.vertices
    if mode == "joints":
        return sample.joints
    return sample.vertices if sample.vertices is not None else sample.joints


def penetration_metrics(
    samples: Sequence[MotionSample],
    args: argparse.Namespace,
    include_distance: bool = True,
    desc: str = "Penetration",
) -> Dict[str, float]:
    if args.scene_occ is None:
        return {}

    per_sample = []
    for sample in progress_iter(samples, desc=desc, enabled=args.show_progress, total=len(samples)):
        points = sample_body_points(sample, args.body_points)
        if points is None:
            continue
        scene_key, occ = load_scene_occ_entry(args.scene_occ, sample.scene_name)
        if occ is None:
            continue
        per_sample.append(
            penetration_for_points(
                points,
                occ,
                args.scene_grid,
                args.out_of_bounds_occupied,
                scene_key=scene_key or "scene",
                include_distance=include_distance,
                show_progress=args.show_progress,
            )
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
    for sample in progress_iter(samples, desc="Foot sliding", enabled=args.show_progress, total=len(samples)):
        curr = {}
        if sample.joints is not None:
            curr["foot_sliding"] = foot_sliding_for_joints(sample.joints, args)
        values.append(curr)
    metrics = aggregate_dicts(values)
    metrics.update(penetration_metrics(samples, args, include_distance=True, desc="Locomotion penetration"))
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


def contact_hand_ids(joint_count: int, args: argparse.Namespace) -> Tuple[int, ...]:
    if args.contact_hand_joints:
        candidates = args.contact_hand_joints
    elif joint_count == 24:
        candidates = (22, 23)
    else:
        candidates = args.hand_joints
    return tuple(idx for idx in candidates if 0 <= idx < joint_count)


def object_vertices_for_length(object_vertices: np.ndarray, length: int) -> np.ndarray:
    object_vertices = np.asarray(object_vertices, dtype=np.float32)
    if object_vertices.ndim == 2 and object_vertices.shape[-1] == 3:
        return np.repeat(object_vertices[None], length, axis=0)
    if object_vertices.ndim == 3 and object_vertices.shape[-1] == 3:
        if object_vertices.shape[0] == 1:
            return np.repeat(object_vertices, length, axis=0)
        return object_vertices[:length]
    raise ValueError(f"Expected object vertices shape [N, 3] or [T, N, 3], got {object_vertices.shape}.")


def min_hand_object_distance(
    joints: np.ndarray,
    object_vertices: np.ndarray,
    hand_ids: Tuple[int, ...],
    chunk_size: int = 32,
) -> np.ndarray:
    distances = []
    for start in range(0, joints.shape[0], chunk_size):
        end = min(start + chunk_size, joints.shape[0])
        hands = joints[start:end, list(hand_ids)]
        obj = object_vertices[start:end]
        diff = hands[:, :, None, :] - obj[:, None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        distances.append(dist.min(axis=(1, 2)))
    return np.concatenate(distances, axis=0)


def contact_metrics_for_pair(
    generated: MotionSample,
    reference: MotionSample,
    args: argparse.Namespace,
) -> Dict[str, float]:
    if generated.joints is None or reference.joints is None:
        return {}
    object_vertices = reference.object_vertices if reference.object_vertices is not None else generated.object_vertices
    if object_vertices is None:
        return {}

    object_len = object_vertices.shape[0] if np.asarray(object_vertices).ndim == 3 else min(len(generated.joints), len(reference.joints))
    length = min(len(generated.joints), len(reference.joints), int(object_len))
    if length <= 0:
        return {}

    pred_joints = np.asarray(generated.joints[:length], dtype=np.float32)
    gt_joints = np.asarray(reference.joints[:length], dtype=np.float32)
    joint_count = min(pred_joints.shape[1], gt_joints.shape[1])
    pred_joints = pred_joints[:, :joint_count]
    gt_joints = gt_joints[:, :joint_count]
    hand_ids = contact_hand_ids(joint_count, args)
    if not hand_ids:
        return {}

    obj = object_vertices_for_length(object_vertices, length)
    threshold = float(args.contact_threshold)
    gt_dist = min_hand_object_distance(gt_joints, obj, hand_ids)
    pred_dist = min_hand_object_distance(pred_joints, obj, hand_ids)
    gt_contact = gt_dist < threshold
    pred_contact = pred_dist < threshold

    tp = int(np.logical_and(gt_contact, pred_contact).sum())
    fp = int(np.logical_and(~gt_contact, pred_contact).sum())
    fn = int(np.logical_and(gt_contact, ~pred_contact).sum())

    precision = 0.0 if (tp + fp) == 0 else tp / float(tp + fp)
    recall = 0.0 if (tp + fn) == 0 else tp / float(tp + fn)
    f1 = 0.0 if precision == 0.0 and recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "contact_precision": float(precision),
        "contact_recall": float(recall),
        "contact_f1_score": float(f1),
    }


def reaching_metrics(
    samples: Sequence[MotionSample],
    reference: Optional[Sequence[MotionSample]],
    args: argparse.Namespace,
) -> Dict[str, float]:
    values = []
    if reference:
        reference_by_id = {}
        for sample in reference:
            key = sample_pair_id(sample)
            if key is not None:
                reference_by_id[key] = sample
        for idx, sample in enumerate(progress_iter(samples, desc="Human-object contact", enabled=args.show_progress, total=len(samples))):
            ref = None
            key = sample_pair_id(sample)
            if key is not None:
                ref = reference_by_id.get(key)
            if ref is None and idx < len(reference):
                ref = reference[idx]
            if ref is not None:
                values.append(contact_metrics_for_pair(sample, ref, args))
    metrics = aggregate_dicts(values)
    pene = penetration_metrics(samples, args, include_distance=False, desc="Reaching penetration")
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
        "--generated-max-idx",
        type=int,
        default=config_default(config, "generated_max_idx", 0),
        help="Randomly subsample this many idx groups before loading. All repeats per idx are kept. 0 keeps all idx.",
    )
    parser.add_argument(
        "--generated-sample-seed",
        type=int,
        default=config_default(config, "generated_sample_seed", None),
        help="Random seed for generated subsampling. Defaults to --seed.",
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
    parser.add_argument(
        "--reference-joints-ind",
        default=config_default(config, "reference_joints_ind", None),
        help="Optional joint ids to select when slicing clips from --reference-dataset.",
    )
    parser.add_argument("--reference-split", default=config_default(config, "reference_split", None), help="Optional GT split name: train, val, or test.")
    parser.add_argument(
        "--reference-split-dataset",
        default=config_default(config, "reference_split_dataset", None),
        help="Dataset folder that owns the split files.",
    )
    parser.add_argument("--reference-split-dir", default=config_default(config, "reference_split_dir", "splits"))
    parser.add_argument(
        "--reference-split-source-index",
        default=config_default(config, "reference_split_source_index", None),
        help="Optional source_index.npy that maps split indices back to the raw motion dictionary.",
    )
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
        choices=("all", "interactive", "locomotion", "reaching", "paired"),
        help="Metric group to run.",
    )
    parser.add_argument("--output", default=config_default(config, "output", None), help="Optional JSON output path.")
    parser.add_argument("--fps", type=float, default=config_default(config, "fps", 20.0))
    parser.add_argument("--seed", type=int, default=config_default(config, "seed", 1234))
    parser.add_argument(
        "--show-progress",
        dest="show_progress",
        action="store_true",
        default=config_default(config, "show_progress", True),
        help="Show tqdm progress bars during evaluation.",
    )
    parser.add_argument("--no-progress", dest="show_progress", action="store_false", help="Disable progress bars.")
    parser.add_argument("--feature-frames", type=int, default=config_default(config, "feature_frames", 64))
    parser.add_argument("--feature-dim", type=int, default=config_default(config, "feature_dim", 512))
    parser.add_argument("--diversity-pairs", type=int, default=config_default(config, "diversity_pairs", 200))
    parser.add_argument("--multimodality-pairs", type=int, default=config_default(config, "multimodality_pairs", 100))
    parser.add_argument("--precision-k", type=int, default=config_default(config, "precision_k", 3))
    parser.add_argument(
        "--motion-evaluator-checkpoint",
        default=config_default(config, "motion_evaluator_checkpoint", None),
        help="Optional trained motion evaluator checkpoint. When set, interactive metrics use evaluator embeddings.",
    )
    parser.add_argument(
        "--motion-evaluator-batch-size",
        type=int,
        default=config_default(config, "motion_evaluator_batch_size", 256),
    )
    parser.add_argument(
        "--motion-evaluator-device",
        default=config_default(config, "motion_evaluator_device", None),
        help="Device for evaluator encoding. Defaults to --device.",
    )
    parser.add_argument(
        "--motion-evaluator-normalize-embeddings",
        type=parse_bool_or_none,
        default=config_default(config, "motion_evaluator_normalize_embeddings", True),
        help="Whether to L2-normalize evaluator embeddings before FID/diversity/MM. Keep true for contrastive evaluators.",
    )
    parser.add_argument(
        "--motion-evaluator-joints-ind",
        default=config_default(config, "motion_evaluator_joints_ind", None),
        help="Optional joint ids to select from generated/reference samples before evaluator encoding.",
    )
    parser.add_argument(
        "--mpjpe-frame-alignment",
        choices=("resample", "min"),
        default=config_default(config, "mpjpe_frame_alignment", "resample"),
        help="How to align generated/reference frame counts before MPJPE.",
    )
    parser.add_argument(
        "--trajectory-joint",
        type=int,
        default=config_default(config, "trajectory_joint", 0),
        help="Joint id used as p_t for goal/traj metrics. Default 0 is pelvis/root.",
    )
    parser.add_argument(
        "--trajectory-threshold",
        type=float,
        default=config_default(config, "trajectory_threshold", 0.5),
        help="Distance threshold tau in meters for trajectory similarity.",
    )

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
    parser.add_argument(
        "--contact-hand-joints",
        default=config_default(config, "contact_hand_joints", None),
        help="Hand joint ids for human-object contact. Defaults to 22,23 for 24-joint OMOMO; otherwise --hand-joints.",
    )
    parser.add_argument(
        "--contact-threshold",
        type=float,
        default=config_default(config, "contact_threshold", 0.05),
        help="Distance threshold in meters for hand-object contact labels.",
    )
    parser.add_argument(
        "--object-mesh-dir",
        default=config_default(config, "object_mesh_dir", None),
        help="Optional OMOMO captured_objects folder for reconstructing object vertices from .p records.",
    )

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
    args.reference_joints_ind = parse_ints(args.reference_joints_ind)
    args.motion_evaluator_joints_ind = parse_ints(args.motion_evaluator_joints_ind)
    args.contact_hand_joints = parse_ints(args.contact_hand_joints)

    generated = load_samples(
        args.generated,
        args,
        desc="Load generated",
        max_idx=args.generated_max_idx,
        sample_seed=args.generated_sample_seed if args.generated_sample_seed is not None else args.seed,
    )
    reference = []
    if args.reference:
        reference.extend(load_samples(args.reference, args, desc="Load reference"))
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
    if args.metrics in ("all", "paired"):
        try:
            paired_reference = paired_reference_for_generated(generated, reference, args)
            results["paired"] = paired_metrics(generated, paired_reference, args)
        except Exception as exc:
            warnings.warn(f"Paired metrics skipped: {exc}")
            results["paired"] = {}
    if args.metrics in ("all", "locomotion"):
        try:
            results["locomotion"] = locomotion_metrics(generated, args)
        except Exception as exc:
            warnings.warn(f"Locomotion metrics skipped: {exc}")
            results["locomotion"] = {}
    if args.metrics in ("all", "reaching"):
        try:
            results["reaching"] = reaching_metrics(generated, reference, args)
        except Exception as exc:
            warnings.warn(f"Reaching metrics skipped: {exc}")
            results["reaching"] = {}

    print_metrics(results)
    write_json(args.output, results)


if __name__ == "__main__":
    main()
