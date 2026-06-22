import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from scipy.spatial.transform import Rotation as R
except ModuleNotFoundError:
    R = None


def _rotvec_to_matrix(rotvec):
    rotvec = np.asarray(rotvec, dtype=np.float32)
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = rotvec / theta
    x, y, z = axis
    skew = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float32,
    )
    return (
        np.eye(3, dtype=np.float32)
        + np.sin(theta) * skew
        + (1.0 - np.cos(theta)) * (skew @ skew)
    ).astype(np.float32)


def _y_rotation(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def _yaw_cancel_matrix(rotvec):
    if R is not None:
        init_euler = R.from_rotvec(rotvec).as_euler("zxy")
        return R.from_euler("zxy", [0.0, 0.0, -init_euler[2]]).as_matrix().astype(np.float32)

    rot = _rotvec_to_matrix(rotvec)
    yaw = np.arctan2(-rot[2, 0], rot[0, 0])
    return _y_rotation(-yaw)


def _normalize_points(points, coord_min, coord_max):
    shape = points.shape
    points = points.reshape(-1, 3)
    points = -1.0 + 2.0 * (points - coord_min) / (coord_max - coord_min)
    return points.reshape(shape)


def _rotation_6d(matrix):
    return np.asarray(matrix[:3, :2], dtype=np.float32).reshape(6)


class TrumansDataset(Dataset):
    """TRUMANS raw dataset adapter with the same batch surface as LingoDataset.

    This follows the official trumans_utils layout: raw arrays live directly in
    ``folder`` and fixed-length windows are indexed by ``idx_start.npy``.
    """

    def __init__(
            self,
            folder,
            device,
            mesh_grid,
            batch_size,
            step=3,
            nb_voxels=(32, 32, 32),
            train=True,
            load_scene=True,
            load_language=True,
            load_pelvis_goal=True,
            load_hand_goal=False,
            load_object=False,
            max_window_size=None,
            use_pi=False,
            split=None,
            split_dir="splits",
            scene_mode="test",
            test_scene_name=None,
            norm_file="norm.npy",
            norm_key=None,
            normalize=True,
            split_seed=42,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            max_samples=0,
            object_num_points=512,
            normalize_object_motion=True,
            **kwargs,
    ):
        self.folder = Path(folder)
        self.device = device
        self.train = train
        self.load_scene = load_scene
        self.load_language = load_language
        self.load_pelvis_goal = load_pelvis_goal
        self.load_hand_goal = load_hand_goal
        self.load_object = load_object
        self.use_pi = use_pi
        self.split = split
        self.split_dir = split_dir
        self.scene_mode = scene_mode
        self.test_scene_name = test_scene_name
        self.step = int(step)
        self.batch_size = batch_size
        self.nb_voxels = list(nb_voxels)
        self.mesh_grid = mesh_grid
        self.max_window_size = int(max_window_size or kwargs.get("seq_len", 16))
        self.split_seed = int(split_seed)
        self.split_ratios = (float(train_ratio), float(val_ratio), float(test_ratio))
        self.object_num_points = int(object_num_points)
        self.normalize_object_motion = bool(normalize_object_motion)

        mmap = "r"
        self.human_joints = np.load(self.folder / "human_joints.npy", mmap_mode=mmap)
        self.human_orient = np.load(self.folder / "human_orient.npy", mmap_mode=mmap)
        self.action_label = np.load(self.folder / "action_label.npy", mmap_mode=mmap)
        self.idx_start = np.load(self.folder / "idx_start.npy", mmap_mode=mmap)
        self.scene_flag = np.load(self.folder / "scene_flag.npy", mmap_mode=mmap)
        self.object_flag = np.load(self.folder / "object_flag.npy", mmap_mode=mmap)
        self.object_mat = np.load(self.folder / "object_mat.npy", mmap_mode=mmap)

        self.nb_joints = int(self.human_joints.shape[1])
        self.language_feature_dim = int(self.action_label.shape[-1])

        sample_count = len(self.idx_start)
        if max_samples not in [None, 0, "0", "None", "none", "null"]:
            sample_count = min(sample_count, int(max_samples))
        self.indices = np.arange(sample_count, dtype=np.int64)

        self.normalize_enabled = bool(normalize)
        self.norm_key = norm_key or f"({self.max_window_size}, {self.step})"
        self.min = np.zeros(3, dtype=np.float32)
        self.max = np.ones(3, dtype=np.float32)
        norm_path = self.folder / norm_file
        if self.normalize_enabled and norm_path.exists():
            norm = np.load(norm_path, allow_pickle=True).item()
            if self.norm_key not in norm:
                raise KeyError(
                    f"Normalization key {self.norm_key!r} is missing from {norm_path}. "
                    f"Available keys: {sorted(norm.keys())}"
                )
            self.min = np.asarray(norm[self.norm_key][0], dtype=np.float32)
            self.max = np.asarray(norm[self.norm_key][1], dtype=np.float32)
        else:
            self.normalize_enabled = False
        self.min_torch = torch.tensor(self.min).to(device)
        self.max_torch = torch.tensor(self.max).to(device)

        self.object_files = sorted((self.folder / "Object").glob("*.npy"))
        self.object_points_cache = {}

        self.scene_occ = None
        self.scene_dict = {}
        if self.load_scene:
            self._load_scenes()

    def _load_scenes(self):
        scene_folder = self.folder / "Scene"
        scene_file_list = sorted(os.listdir(scene_folder))
        if self.test_scene_name not in [None, "None", "none", "null"]:
            scene_file_list = [name for name in scene_file_list if name.split(".")[0] == self.test_scene_name]

        scene_occ = []
        for sid, file_name in enumerate(scene_file_list):
            print(f"{sid} Loading TRUMANS Scene Mesh {file_name}")
            occ = np.load(scene_folder / file_name)
            scene_occ.append(torch.from_numpy(occ).to(device=self.device, dtype=torch.bool))
            self.scene_dict[file_name[:-4]] = sid
        if len(scene_occ) == 0:
            raise RuntimeError(f"No scene files found in {scene_folder}.")
        self.scene_occ = torch.stack(scene_occ)

        self.scene_grid_np = np.array([-3, 0, -4, 3, 2, 4, 300, 100, 400])
        self.scene_grid_torch = torch.tensor([-3, 0, -4, 3, 2, 4, 300, 100, 400]).to(self.device)

        grid_count = self.nb_voxels[0] * self.nb_voxels[1] * self.nb_voxels[2]
        self.batch_id = torch.linspace(0, self.batch_size - 1, self.batch_size).tile((grid_count, 1)).T
        self.batch_id = self.batch_id.reshape(-1, 1).to(device=self.device, dtype=torch.long)

    def get_split_indices(self, split_name):
        if split_name in [None, "None", "none", "null"]:
            return self.indices

        split_path = self.folder / self.split_dir / f"{split_name}_idx.npy"
        if split_path.exists():
            split_idx = np.load(split_path).astype(np.int64)
            return split_idx[split_idx < len(self.indices)]

        rng = np.random.default_rng(self.split_seed)
        order = np.arange(len(self.indices), dtype=np.int64)
        rng.shuffle(order)
        ratios = np.asarray(self.split_ratios, dtype=np.float64)
        ratios = ratios / ratios.sum()
        n_train = int(round(len(order) * ratios[0]))
        n_val = int(round(len(order) * ratios[1]))
        splits = {
            "train": np.sort(order[:n_train]),
            "val": np.sort(order[n_train:n_train + n_val]),
            "test": np.sort(order[n_train + n_val:]),
        }
        if split_name not in splits:
            raise ValueError(f"Unknown split={split_name!r}; use train, val, test, or provide {split_path}.")
        return splits[split_name]

    def _frame_ids(self, src_idx):
        start = int(self.idx_start[src_idx])
        frame_ids = start + np.arange(self.max_window_size, dtype=np.int64) * self.step
        valid = frame_ids < self.human_joints.shape[0]
        return frame_ids, valid

    def _local_motion(self, frame_ids, valid):
        motion = np.zeros((self.max_window_size, self.nb_joints, 3), dtype=np.float32)
        valid_ids = frame_ids[valid]
        motion[:len(valid_ids)] = np.asarray(self.human_joints[valid_ids], dtype=np.float32)

        init_shift = np.array([motion[0, 0, 0], 0.0, motion[0, 0, 2]], dtype=np.float32)
        motion[:len(valid_ids)] -= init_shift
        local_rot = _yaw_cancel_matrix(np.asarray(self.human_orient[valid_ids[0]], dtype=np.float32))
        motion[:len(valid_ids)] = motion[:len(valid_ids)] @ local_rot.T

        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = np.linalg.inv(local_rot.T).T
        mat[:3, 3] = init_shift
        return motion, mat, init_shift, local_rot

    def _primary_object_id(self, frame_ids, valid):
        flags = np.asarray(self.object_flag[frame_ids[valid]], dtype=np.int16)
        present = flags != -1
        if not present.any():
            return None
        counts = present.sum(axis=0)
        return int(np.argmax(counts))

    def _sample_object_points(self, object_id):
        if object_id in self.object_points_cache:
            return self.object_points_cache[object_id]
        points = np.asarray(np.load(self.object_files[object_id]), dtype=np.float32)
        if len(points) >= self.object_num_points:
            sample_idx = np.linspace(0, len(points) - 1, self.object_num_points).round().astype(np.int64)
            points = points[sample_idx]
        else:
            pad_idx = np.resize(np.arange(len(points), dtype=np.int64), self.object_num_points)
            points = points[pad_idx]
        self.object_points_cache[object_id] = points.astype(np.float32)
        return self.object_points_cache[object_id]

    def _object_condition(self, frame_ids, valid, init_shift, local_rot):
        object_motion = np.zeros((self.max_window_size, 9), dtype=np.float32)
        object_points = np.zeros((self.object_num_points, 3), dtype=np.float32)
        object_goal = np.zeros(3, dtype=np.float32)
        object_id = self._primary_object_id(frame_ids, valid)
        if object_id is None or object_id >= len(self.object_files):
            return object_motion, object_points, object_goal, False

        rest_points = self._sample_object_points(object_id)
        first_local_mat = None
        valid_ids = frame_ids[valid]
        flags = np.asarray(self.object_flag[valid_ids, object_id], dtype=np.int16)

        for local_i, (frame_id, slot) in enumerate(zip(valid_ids, flags)):
            if slot < 0:
                continue
            world_mat = np.asarray(self.object_mat[frame_id, int(slot)], dtype=np.float32)
            if not np.isfinite(world_mat).all():
                continue
            world_rot = world_mat[:3, :3]
            world_trans = world_mat[:3, 3]
            local_obj_rot = local_rot @ world_rot
            local_obj_trans = local_rot @ (world_trans - init_shift)

            trans_for_motion = local_obj_trans.astype(np.float32)
            if self.normalize_enabled and self.normalize_object_motion:
                trans_for_motion = _normalize_points(trans_for_motion.reshape(1, 3), self.min, self.max).reshape(3)
            object_motion[local_i, :3] = trans_for_motion
            object_motion[local_i, 3:] = _rotation_6d(local_obj_rot)
            object_goal = local_obj_trans.astype(np.float32)

            if first_local_mat is None:
                first_local_mat = (local_obj_rot.astype(np.float32), local_obj_trans.astype(np.float32))

        if first_local_mat is not None:
            obj_rot, obj_trans = first_local_mat
            object_points = (rest_points @ obj_rot.T) + obj_trans
            if self.normalize_enabled and self.normalize_object_motion:
                object_points = _normalize_points(object_points, self.min, self.max)
        else:
            return object_motion, object_points, object_goal, False

        return object_motion, object_points.astype(np.float32), object_goal, True

    def __getitem__(self, idx):
        src_idx = int(self.indices[idx])
        frame_ids, valid = self._frame_ids(src_idx)
        valid_len = int(valid.sum())
        if valid_len == 0:
            raise RuntimeError(f"Empty TRUMANS window at source index {src_idx}.")

        motion, mat, init_shift, local_rot = self._local_motion(frame_ids, valid)
        pelvis_goal = motion[valid_len - 1, 0].copy()
        pelvis_goal[1] = 0.0

        if self.normalize_enabled:
            motion = _normalize_points(motion, self.min, self.max)

        text_emb = np.zeros((self.max_window_size, self.language_feature_dim), dtype=np.float32)
        if self.load_language:
            text_emb[:valid_len] = np.asarray(self.action_label[frame_ids[valid]], dtype=np.float32)

        object_motion, object_points, object_goal, object_present = self._object_condition(
            frame_ids, valid, init_shift, local_rot
        )

        return (
            motion.reshape(self.max_window_size, self.nb_joints * 3).astype(np.float32),
            mat.astype(np.float32),
            np.asarray(self.scene_flag[int(self.idx_start[src_idx])] if self.load_scene else 0, dtype=np.int64),
            text_emb.astype(np.float32),
            pelvis_goal.astype(np.float32),
            np.zeros(3, dtype=np.float32),
            np.asarray([object_present], dtype=np.bool_),
            np.asarray(True, dtype=np.bool_),
            np.asarray(True, dtype=np.bool_),
            np.asarray(0, dtype=np.int64),
            np.asarray(False, dtype=np.bool_),
            np.asarray(False, dtype=np.bool_),
            np.asarray(valid_len, dtype=np.int64),
            valid.astype(np.bool_),
            np.asarray(object_present, dtype=np.bool_),
            np.asarray(False, dtype=np.bool_),
            {
                "object_motion": object_motion.astype(np.float32),
                "object_points": object_points.astype(np.float32),
                "object_goal": object_goal.astype(np.float32),
            },
        )

    def get_occ_for_points(self, points, scene_flag):
        batch_size = points.shape[0]
        seq_len = points.shape[1]
        points = points.reshape(-1, 3)
        voxel_size = torch.div(self.scene_grid_torch[3: 6] - self.scene_grid_torch[:3], self.scene_grid_torch[6:])
        voxel = torch.div((points - self.scene_grid_torch[:3]), voxel_size)
        voxel = voxel.to(dtype=torch.long)
        lb = torch.all(voxel >= 0, dim=-1)
        ub = torch.all(voxel < self.scene_grid_torch[6:] - 0, dim=-1)
        in_bound = torch.logical_and(lb, ub)
        voxel[torch.logical_not(in_bound)] = 0
        scene_flag = torch.as_tensor(scene_flag, device=self.device, dtype=torch.long).reshape(-1)
        if scene_flag.numel() == 1 and batch_size > 1:
            scene_flag = scene_flag.repeat(batch_size)
        if scene_flag.numel() != batch_size:
            raise ValueError(
                f"scene_flag has {scene_flag.numel()} entries, but points batch size is {batch_size}."
            )

        occ = self.scene_occ[scene_flag]
        if occ.ndim == 5 and occ.shape[1] == 1:
            occ = occ[:, 0]

        batch_id = torch.arange(batch_size, device=self.device).repeat_interleave(seq_len)
        occ_for_points = occ[batch_id, voxel[:, 0], voxel[:, 1], voxel[:, 2]]
        occ_for_points[torch.logical_not(in_bound)] = True
        return occ_for_points.reshape(batch_size, seq_len, -1)

    def create_meshgrid(self, batch_size=1):
        bbox = self.mesh_grid
        size = (self.nb_voxels[0], self.nb_voxels[1], self.nb_voxels[2])
        x = torch.linspace(bbox[0], bbox[1], size[0])
        y = torch.linspace(bbox[2], bbox[3], size[1])
        z = torch.linspace(bbox[4], bbox[5], size[2])
        xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
        grid = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        return grid.repeat(batch_size, 1, 1)

    def __len__(self):
        return len(self.indices)

    def normalize(self, data):
        if not self.normalize_enabled:
            return data
        return _normalize_points(data, self.min, self.max)

    def normalize_torch(self, data):
        if not self.normalize_enabled:
            return data
        shape_orig = data.shape
        data = data.reshape((-1, 3))
        data = -1.0 + 2.0 * (data - self.min_torch) / (self.max_torch - self.min_torch)
        return data.reshape(shape_orig)

    def denormalize(self, data):
        if not self.normalize_enabled:
            return data
        shape_orig = data.shape
        data = data.reshape((-1, 3))
        data = (data + 1.0) * (self.max - self.min) / 2.0 + self.min
        return data.reshape(shape_orig)

    def denormalize_torch(self, data):
        if not self.normalize_enabled:
            return data
        shape_orig = data.shape
        data = data.reshape((-1, 3))
        data = (data + 1.0) * (self.max_torch - self.min_torch) / 2.0 + self.min_torch
        return data.reshape(shape_orig)
