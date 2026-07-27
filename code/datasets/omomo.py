import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from bps_utils import encode_bps, make_bps_basis

try:
    import joblib
except ModuleNotFoundError:  # pragma: no cover - joblib is part of requirements.txt
    joblib = None


_Z_UP_TO_Y_UP = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float32,
)


def _load_serialized(path, mmap_mode=None):
    if joblib is not None:
        return joblib.load(path, mmap_mode=mmap_mode)
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Loading the official OMOMO file {path} requires joblib. "
            "Install the dependencies from requirements.txt."
        ) from exc


def _ordered_items(records):
    if isinstance(records, dict):
        try:
            keys = sorted(records, key=lambda value: int(value))
        except (TypeError, ValueError):
            keys = list(records)
        return [(key, records[key]) for key in keys]
    if isinstance(records, (list, tuple)):
        return list(enumerate(records))
    raise TypeError(f"Expected an OMOMO dict/list, got {type(records).__name__}.")


def _normalize_points(points, coord_min, coord_max):
    shape = points.shape
    points = points.reshape(-1, 3)
    denom = np.maximum(coord_max - coord_min, 1e-8)
    points = -1.0 + 2.0 * (points - coord_min) / denom
    return points.reshape(shape)


def _rotation_6d(matrix):
    return np.asarray(matrix[:3, :2], dtype=np.float32).reshape(6)


def _read_obj_vertices(path):
    vertices = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise RuntimeError(f"No vertices found in OMOMO object mesh: {path}")
    return np.asarray(vertices, dtype=np.float32)


def _transform_bounds(coord_min, coord_max, matrix):
    corners = np.stack(
        np.meshgrid(
            [coord_min[0], coord_max[0]],
            [coord_min[1], coord_max[1]],
            [coord_min[2], coord_max[2]],
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    transformed = corners @ matrix.T
    return transformed.min(axis=0), transformed.max(axis=0)


class OmomoDataset(Dataset):
    """Official OMOMO train/test windows exposed through the FlowHSI batch API.

    OMOMO already ships subject-disjoint train and test files (subjects 1-15
    versus 16-17), so this adapter never creates a random split. ``train``,
    ``val`` and ``test`` are resolved by :meth:`get_split_indices`; ``val`` is
    an alias for the official test set for opt-in validation only.
    """

    def __init__(
            self,
            folder,
            device,
            mesh_grid,
            batch_size,
            step=1,
            nb_voxels=(32, 32, 32),
            train=True,
            load_scene=False,
            load_language=False,
            load_pelvis_goal=False,
            load_hand_goal=False,
            load_object=True,
            max_window_size=120,
            use_pi=False,
            split=None,
            train_file="train_diffusion_manip_window_120_cano_joints24.p",
            test_file="test_diffusion_manip_window_120_processed_joints24.p",
            stats_file="min_max_mean_std_data_window_120_cano_joints24.p",
            object_mesh_dir="captured_objects",
            load_train=True,
            load_test=True,
            max_train_samples=0,
            max_test_samples=0,
            object_num_points=1024,
            bps_num_points=256,
            bps_radius=1.0,
            bps_seed=12345,
            normalize=True,
            convert_z_up_to_y_up=True,
            mmap_mode=None,
            motion_state_len=0,
            motion_state_prefix_len=None,
            language_feature_dim=1,
            **kwargs,
    ):
        self.folder = Path(folder)
        self.device = device
        self.mesh_grid = mesh_grid
        self.batch_size = int(batch_size)
        self.step = max(1, int(step))
        self.nb_voxels = list(nb_voxels)
        self.train = bool(train)
        self.load_scene = bool(load_scene)
        self.load_language = bool(load_language)
        self.load_pelvis_goal = bool(load_pelvis_goal)
        self.load_hand_goal = bool(load_hand_goal)
        self.load_object = bool(load_object)
        self.use_pi = bool(use_pi)
        self.max_window_size = int(max_window_size)
        self.object_num_points = int(object_num_points)
        self.bps_num_points = int(bps_num_points)
        self.bps_radius = float(bps_radius)
        self.bps_seed = int(bps_seed)
        self.bps_basis = make_bps_basis(self.bps_num_points, self.bps_radius, self.bps_seed)
        self.motion_state_len = max(0, int(motion_state_len or 0))
        self.motion_state_prefix_len = int(motion_state_prefix_len or kwargs.get("auto_regre_num", 2))
        self.language_feature_dim = int(language_feature_dim)
        self.normalize_enabled = bool(normalize)
        self.coord_transform = _Z_UP_TO_Y_UP if convert_z_up_to_y_up else np.eye(3, dtype=np.float32)
        self.object_mesh_dir = self.folder / object_mesh_dir
        self.object_vertices_cache = {}

        stats_path = self.folder / stats_file
        if not stats_path.exists():
            raise FileNotFoundError(f"OMOMO normalization statistics do not exist: {stats_path}")
        stats = _load_serialized(stats_path, mmap_mode=mmap_mode)
        joint_min = np.asarray(stats["global_jpos_min"], dtype=np.float32).reshape(-1, 3)
        joint_max = np.asarray(stats["global_jpos_max"], dtype=np.float32).reshape(-1, 3)
        raw_min = joint_min.min(axis=0)
        raw_max = joint_max.max(axis=0)
        self.min, self.max = _transform_bounds(raw_min, raw_max, self.coord_transform)
        self.min = self.min.astype(np.float32)
        self.max = self.max.astype(np.float32)
        self.min_torch = torch.tensor(self.min, device=device)
        self.max_torch = torch.tensor(self.max, device=device)

        self.entries = []
        self.split_indices = {"train": [], "test": []}
        if load_train:
            self._append_split("train", self.folder / train_file, max_train_samples, mmap_mode)
        if load_test:
            self._append_split("test", self.folder / test_file, max_test_samples, mmap_mode)
        if not self.entries:
            raise RuntimeError("OMOMO loader was configured with both load_train=false and load_test=false.")

        self.indices = np.arange(len(self.entries), dtype=np.int64)
        if split not in [None, "None", "none", "null"]:
            self.indices = self.get_split_indices(split)

        self.scene_occ = None
        self.scene_dict = {}
        if self.load_scene:
            raise ValueError("OMOMO has no scene occupancy data; set load_scene=false.")

    def _append_split(self, split_name, path, max_samples, mmap_mode):
        if not path.exists():
            raise FileNotFoundError(
                f"Official OMOMO {split_name} file does not exist: {path}. "
                "OMOMO must use its supplied train/test files and must not be randomly split."
            )
        items = _ordered_items(_load_serialized(path, mmap_mode=mmap_mode))
        if max_samples not in [None, 0, "0", "None", "none", "null"]:
            items = items[:int(max_samples)]
        start = len(self.entries)
        self.entries.extend((split_name, source_key, record) for source_key, record in items)
        self.split_indices[split_name] = np.arange(start, len(self.entries), dtype=np.int64)

    def get_split_indices(self, split_name):
        split_name = str(split_name).lower()
        if split_name == "val":
            split_name = "test"
        if split_name not in self.split_indices:
            raise ValueError(f"Unknown OMOMO split={split_name!r}; use train, val, or test.")
        indices = np.asarray(self.split_indices[split_name], dtype=np.int64)
        if len(indices) == 0:
            raise RuntimeError(
                f"OMOMO split {split_name!r} was not loaded. Enable load_{split_name}=true in the dataset config."
            )
        return indices

    def _object_mesh_vertices(self, object_name, suffix=""):
        cache_key = (object_name, suffix)
        if cache_key in self.object_vertices_cache:
            return self.object_vertices_cache[cache_key]
        path = self.object_mesh_dir / f"{object_name}_cleaned_simplified{suffix}.obj"
        if not path.exists():
            raise FileNotFoundError(f"OMOMO object mesh does not exist: {path}")
        vertices = _read_obj_vertices(path)
        self.object_vertices_cache[cache_key] = vertices
        return vertices

    def _sample_points(self, points):
        if len(points) == 0:
            return np.zeros((self.object_num_points, 3), dtype=np.float32)
        if len(points) >= self.object_num_points:
            ids = np.linspace(0, len(points) - 1, self.object_num_points).round().astype(np.int64)
        else:
            ids = np.resize(np.arange(len(points), dtype=np.int64), self.object_num_points)
        return np.asarray(points[ids], dtype=np.float32)

    def _transform_object_part(self, vertices, scale, rotation, translation):
        points = float(scale) * (vertices @ np.asarray(rotation, dtype=np.float32).T)
        points += np.asarray(translation, dtype=np.float32).reshape(1, 3)
        return points @ self.coord_transform.T

    def _object_points(self, record, object_name, frame_id):
        if object_name in ("mop", "vacuum") and "obj_bottom_trans" in record:
            top = self._transform_object_part(
                self._object_mesh_vertices(object_name, "_top"),
                np.asarray(record["obj_scale"])[frame_id],
                np.asarray(record["obj_rot_mat"])[frame_id],
                np.asarray(record["obj_trans"])[frame_id],
            )
            bottom = self._transform_object_part(
                self._object_mesh_vertices(object_name, "_bottom"),
                np.asarray(record["obj_bottom_scale"])[frame_id],
                np.asarray(record["obj_bottom_rot_mat"])[frame_id],
                np.asarray(record["obj_bottom_trans"])[frame_id],
            )
            points = np.concatenate([top, bottom], axis=0)
        else:
            points = self._transform_object_part(
                self._object_mesh_vertices(object_name),
                np.asarray(record["obj_scale"])[frame_id],
                np.asarray(record["obj_rot_mat"])[frame_id],
                np.asarray(record["obj_trans"])[frame_id],
            )
        points = self._sample_points(points)
        if self.normalize_enabled:
            points = _normalize_points(points, self.min, self.max)
        return points.astype(np.float32)

    def _object_bps(self, object_points, object_translation, object_rotation):
        points = np.asarray(object_points, dtype=np.float32)
        if self.normalize_enabled:
            points = (points + 1.0) * (self.max - self.min) / 2.0 + self.min
        local_points = (
            points - np.asarray(object_translation, dtype=np.float32).reshape(1, 3)
        ) @ np.asarray(object_rotation, dtype=np.float32)
        residuals, _, _ = encode_bps(local_points, self.bps_basis)
        return residuals

    def _motion_state(self, motion, valid_len):
        if self.motion_state_len <= 0:
            return None, None
        prefix_len = max(1, min(self.motion_state_prefix_len, valid_len))
        prefix = motion[:prefix_len]
        pad_len = self.motion_state_len - prefix_len
        if pad_len > 0:
            state = np.concatenate([np.repeat(prefix[:1], pad_len, axis=0), prefix], axis=0)
            mask = np.concatenate(
                [np.zeros(pad_len, dtype=np.bool_), np.ones(prefix_len, dtype=np.bool_)], axis=0
            )
        else:
            state = prefix[-self.motion_state_len:]
            mask = np.ones(self.motion_state_len, dtype=np.bool_)
        return state.astype(np.float32), mask

    def __getitem__(self, idx):
        entry_idx = int(self.indices[idx])
        _, _, record = self.entries[entry_idx]
        source_motion = np.asarray(record["motion"], dtype=np.float32)
        if source_motion.ndim != 2 or source_motion.shape[1] < 72:
            raise ValueError(f"Invalid OMOMO motion shape: {source_motion.shape}; expected T x >=72.")

        source_len = min(source_motion.shape[0], int(record.get("end_t_idx", source_motion.shape[0] - 1))
                         - int(record.get("start_t_idx", 0)) + 1)
        frame_ids = np.arange(0, source_len, self.step, dtype=np.int64)[:self.max_window_size]
        valid_len = len(frame_ids)
        if valid_len == 0:
            raise RuntimeError(f"Empty OMOMO window at dataset index {entry_idx}.")

        raw_motion = source_motion[frame_ids, :72].reshape(valid_len, 24, 3)
        raw_motion = raw_motion @ self.coord_transform.T
        pelvis_goal = raw_motion[-1, 0].copy()
        pelvis_goal[1] = 0.0
        motion = raw_motion
        if self.normalize_enabled:
            motion = _normalize_points(motion, self.min, self.max)

        human_motion = np.zeros((self.max_window_size, 72), dtype=np.float32)
        human_motion[:valid_len] = motion.reshape(valid_len, 72)
        valid_mask = np.zeros(self.max_window_size, dtype=np.bool_)
        valid_mask[:valid_len] = True

        object_centers = np.asarray(record["window_obj_com_pos"], dtype=np.float32)[frame_ids]
        object_centers = object_centers @ self.coord_transform.T
        object_rotations = np.asarray(record["obj_rot_mat"], dtype=np.float32)[frame_ids]
        object_motion = np.zeros((self.max_window_size, 9), dtype=np.float32)
        converted_rotations = np.zeros((valid_len, 3, 3), dtype=np.float32)
        object_translation = object_centers
        if self.normalize_enabled:
            object_translation = _normalize_points(object_translation, self.min, self.max)
        object_motion[:valid_len, :3] = object_translation
        for local_idx, rotation in enumerate(object_rotations):
            converted_rotation = self.coord_transform @ rotation @ self.coord_transform.T
            converted_rotations[local_idx] = converted_rotation
            object_motion[local_idx, 3:] = _rotation_6d(converted_rotation)

        seq_name = str(record.get("seq_name", "unknown_object_000"))
        parts = seq_name.split("_")
        object_name = parts[1] if len(parts) > 1 else seq_name
        object_points = self._object_points(record, object_name, int(frame_ids[0]))
        object_bps = self._object_bps(
            object_points,
            object_centers[0],
            converted_rotations[0],
        )
        object_goal = object_centers[-1].astype(np.float32)
        motion_state, motion_state_mask = self._motion_state(
            motion.reshape(valid_len, 72), valid_len
        )
        extra = {
            "object_motion": object_motion,
            "object_points": object_points,
            "object_bps": object_bps,
            "object_goal": object_goal,
            "object_norm_min": self.min.astype(np.float32),
            "object_norm_max": self.max.astype(np.float32),
            "object_geometry_normalized": np.asarray(
                self.normalize_enabled,
                dtype=np.bool_,
            ),
        }
        if motion_state is not None:
            extra["motion_state"] = motion_state
            extra["motion_state_mask"] = motion_state_mask

        text_emb = np.zeros((1, self.language_feature_dim), dtype=np.float32)
        return (
            human_motion,
            np.eye(4, dtype=np.float32),
            np.asarray(0, dtype=np.int64),
            text_emb,
            pelvis_goal.astype(np.float32),
            np.zeros(3, dtype=np.float32),
            np.asarray([True], dtype=np.bool_),
            np.asarray(False, dtype=np.bool_),
            np.asarray(False, dtype=np.bool_),
            np.asarray(0, dtype=np.int64),
            np.asarray(False, dtype=np.bool_),
            np.asarray(False, dtype=np.bool_),
            np.asarray(valid_len, dtype=np.int64),
            valid_mask,
            np.asarray(True, dtype=np.bool_),
            np.asarray(valid_len < self.max_window_size, dtype=np.bool_),
            extra,
        )

    def __len__(self):
        return len(self.indices)

    def normalize(self, data):
        if not self.normalize_enabled:
            return data
        return _normalize_points(data, self.min, self.max)

    def normalize_torch(self, data):
        if not self.normalize_enabled:
            return data
        shape = data.shape
        data = data.reshape(-1, 3)
        data = -1.0 + 2.0 * (data - self.min_torch) / (self.max_torch - self.min_torch).clamp_min(1e-8)
        return data.reshape(shape)

    def denormalize(self, data):
        if not self.normalize_enabled:
            return data
        shape = data.shape
        data = data.reshape(-1, 3)
        data = (data + 1.0) * (self.max - self.min) / 2.0 + self.min
        return data.reshape(shape)

    def denormalize_torch(self, data):
        if not self.normalize_enabled:
            return data
        shape = data.shape
        data = data.reshape(-1, 3)
        data = (data + 1.0) * (self.max_torch - self.min_torch) / 2.0 + self.min_torch
        return data.reshape(shape)
