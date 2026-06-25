import json
import os
import pickle as pkl
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class LingoDataset(Dataset):
    """Windowed autoregressive LINGO/HSI dataset.

    Expected folder layout is produced by preprocess_window_dataset.py. The
    class name stays LingoDataset so the existing Hydra config and training
    script can keep importing datasets.lingo.LingoDataset.
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
            load_hand_goal=True,
            max_window_size=None,
            use_pi=True,
            split=None,
            split_dir="splits",
            scene_mode="auto",
            scene_source_dir=None,
            test_scene_name=None,
            motion_state_len=0,
            motion_state_prefix_len=None,
            **kwargs,
    ):
        self.folder = Path(folder)
        self.device = device
        self.train = train
        self.load_scene = load_scene
        self.load_language = load_language
        self.load_pelvis_goal = load_pelvis_goal
        self.load_hand_goal = load_hand_goal
        self.use_pi = use_pi
        self.split = split
        self.split_dir = split_dir
        self.scene_mode = scene_mode
        self.scene_is_dataset = bool(train)
        self.test_scene_name = test_scene_name
        self.step = step
        self.batch_size = batch_size
        self.nb_voxels = list(nb_voxels)
        self.mesh_grid = mesh_grid
        self.motion_state_len = max(0, int(motion_state_len or 0))
        self.motion_state_prefix_len = int(motion_state_prefix_len or kwargs.get("auto_regre_num", 2))

        self.human_motion = np.load(self.folder / "human_motion.npy", mmap_mode="r")
        self.valid_mask = np.load(self.folder / "valid_mask.npy", mmap_mode="r")
        self.length = np.load(self.folder / "length.npy", mmap_mode="r")
        self.text_emb = np.load(self.folder / "text_emb.npy", mmap_mode="r")
        self.mat = np.load(self.folder / "mat.npy", mmap_mode="r")
        self.pelvis_goal = np.load(self.folder / "pelvis_goal.npy", mmap_mode="r")
        self.hand_goal = np.load(self.folder / "hand_goal.npy", mmap_mode="r")
        self.is_pick = np.load(self.folder / "is_pick.npy", mmap_mode="r")
        self.need_scene = np.load(self.folder / "need_scene.npy", mmap_mode="r")
        self.need_pelvis_dir = np.load(self.folder / "need_pelvis_dir.npy", mmap_mode="r")
        self.pi = np.load(self.folder / "pi.npy", mmap_mode="r")
        self.need_pi = np.load(self.folder / "need_pi.npy", mmap_mode="r")
        self.object_present = np.load(self.folder / "object_present.npy", mmap_mode="r")
        self.motion_state = None
        self.motion_state_mask = None
        motion_state_path = self.folder / "motion_state.npy"
        motion_state_mask_path = self.folder / "motion_state_mask.npy"
        if self.motion_state_len > 0 and motion_state_path.exists():
            self.motion_state = np.load(motion_state_path, mmap_mode="r")
            if self.motion_state.shape[1] != self.motion_state_len:
                raise ValueError(
                    f"Config motion_state_len={self.motion_state_len} does not match "
                    f"motion_state length={self.motion_state.shape[1]} in {self.folder}."
                )
            if motion_state_mask_path.exists():
                self.motion_state_mask = np.load(motion_state_mask_path, mmap_mode="r")

        with open(self.folder / "scene_name.pkl", "rb") as f:
            self.scene_name = pkl.load(f)
        with open(self.folder / "text.pkl", "rb") as f:
            self.text = pkl.load(f)

        self.max_window_size = int(max_window_size or self.human_motion.shape[1])
        if self.max_window_size != self.human_motion.shape[1]:
            raise ValueError(
                f"Config max_window_size={self.max_window_size} does not match "
                f"human_motion length={self.human_motion.shape[1]} in {self.folder}."
            )
        completion_path = self.folder / "is_terminal_window.npy"
        if not completion_path.exists():
            completion_path = self.folder / "completion_label.npy"
        if completion_path.exists():
            self.completion_label = np.load(completion_path, mmap_mode="r")
        else:
            self.completion_label = np.asarray(self.length < self.max_window_size, dtype=np.bool_)

        self.indices = np.arange(len(self.length), dtype=np.int64)
        if self.split not in [None, "None", "none", "null"]:
            split_path = self.folder / self.split_dir / f"{self.split}_idx.npy"
            self.indices = np.load(split_path).astype(np.int64)

        metadata_path = self.folder / "metadata.json"
        self.metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        self.normalize_enabled = bool(self.metadata.get("normalize", True))
        raw_dataset_dir = scene_source_dir or self.metadata.get("dataset_dir")
        self.scene_source_dir = Path(raw_dataset_dir).resolve() if raw_dataset_dir else self.folder

        norm_path = self.scene_source_dir / "norm_inter_and_loco__16frames.npy"
        if norm_path.exists():
            norm = np.load(norm_path)
            self.min = norm[0].astype(np.float32)
            self.max = norm[1].astype(np.float32)
        else:
            self.min = np.zeros(3, dtype=np.float32)
            self.max = np.ones(3, dtype=np.float32)
            self.normalize_enabled = False
        self.min_torch = torch.tensor(self.min).to(device)
        self.max_torch = torch.tensor(self.max).to(device)

        self.scene_occ = None
        self.scene_dict = {}
        if self.load_scene:
            self._load_scenes()

    def _load_scenes(self):
        scene_mode = str(self.scene_mode or "auto").lower()
        if scene_mode in ("auto", "none", "null"):
            self.scene_is_dataset = bool(self.train)
        elif scene_mode in ("test", "scene", "dataset"):
            self.scene_is_dataset = True
        elif scene_mode in ("visualization", "visualisation", "vis", "scene_vis", "scenevis"):
            self.scene_is_dataset = False
        else:
            raise ValueError(
                f"Unknown scene_mode={self.scene_mode!r}. "
                "Use auto, test/scene, or visualization/scene_vis."
            )

        scene_folder = self.scene_source_dir / ("Scene" if self.scene_is_dataset else "Scene_vis")
        scene_file_list = sorted(os.listdir(scene_folder))
        if self.test_scene_name not in [None, "None", "none", "null"]:
            scene_file_list = [name for name in scene_file_list if name.split(".")[0] == self.test_scene_name]

        scene_occ = []
        for sid, file_name in enumerate(scene_file_list):
            print(f"{sid} Loading Scene Mesh {file_name}")
            occ = np.load(scene_folder / file_name)
            scene_occ.append(torch.from_numpy(occ).to(device=self.device, dtype=torch.bool))
            self.scene_dict[file_name[:-4]] = sid
        if len(scene_occ) == 0:
            raise RuntimeError(f"No scene files found in {scene_folder}.")
        self.scene_occ = torch.stack(scene_occ)

        if self.scene_is_dataset:
            self.scene_grid_np = np.array([-3, 0, -4, 3, 2, 4, 300, 100, 400])
            self.scene_grid_torch = torch.tensor([-3, 0, -4, 3, 2, 4, 300, 100, 400]).to(self.device)
        else:
            self.scene_grid_np = np.array([-4, 0, -6, 4, 2, 6, 400, 100, 600])
            self.scene_grid_torch = torch.tensor([-4, 0, -6, 4, 2, 6, 400, 100, 600]).to(self.device)

        grid_count = self.nb_voxels[0] * self.nb_voxels[1] * self.nb_voxels[2]
        self.batch_id = torch.linspace(0, self.batch_size - 1, self.batch_size).tile((grid_count, 1)).T
        self.batch_id = self.batch_id.reshape(-1, 1).to(device=self.device, dtype=torch.long)

    def _motion_state_for_window(self, src_idx):
        if self.motion_state_len <= 0:
            return None, None
        if self.motion_state is not None:
            state = np.asarray(self.motion_state[src_idx], dtype=np.float32)
            if self.motion_state_mask is None:
                mask = np.ones(self.motion_state_len, dtype=np.bool_)
            else:
                mask = np.asarray(self.motion_state_mask[src_idx], dtype=np.bool_)
            return state, mask

        motion = np.asarray(self.human_motion[src_idx], dtype=np.float32)
        valid = np.asarray(self.valid_mask[src_idx], dtype=np.bool_)
        valid_len = int(valid.sum())
        prefix_len = max(1, min(self.motion_state_prefix_len, valid_len, motion.shape[0]))
        prefix = motion[:prefix_len]
        pad_len = self.motion_state_len - prefix_len
        if pad_len > 0:
            state = np.concatenate([np.repeat(prefix[:1], pad_len, axis=0), prefix], axis=0)
            mask = np.concatenate(
                [np.zeros(pad_len, dtype=np.bool_), np.ones(prefix_len, dtype=np.bool_)],
                axis=0,
            )
        else:
            state = prefix[-self.motion_state_len:]
            mask = np.ones(self.motion_state_len, dtype=np.bool_)
        return state.astype(np.float32), mask.astype(np.bool_)

    def __getitem__(self, idx):
        src_idx = int(self.indices[idx])
        scene_name = self.scene_name[src_idx]
        scene_flag = self.scene_dict.get(scene_name, 0) if self.load_scene else 0

        pi = int(self.pi[src_idx]) if self.use_pi else 0
        need_pi = bool(self.need_pi[src_idx]) if self.use_pi else False
        motion_state, motion_state_mask = self._motion_state_for_window(src_idx)
        extra = {}
        if motion_state is not None:
            extra["motion_state"] = motion_state
            extra["motion_state_mask"] = motion_state_mask

        return (
            np.asarray(self.human_motion[src_idx], dtype=np.float32),
            np.asarray(self.mat[src_idx], dtype=np.float32),
            np.asarray(scene_flag, dtype=np.int64),
            np.asarray(self.text_emb[src_idx], dtype=np.float32),
            np.asarray(self.pelvis_goal[src_idx], dtype=np.float32),
            np.asarray(self.hand_goal[src_idx], dtype=np.float32),
            np.asarray([self.is_pick[src_idx]], dtype=np.bool_),
            np.asarray(self.need_scene[src_idx], dtype=np.bool_),
            np.asarray(self.need_pelvis_dir[src_idx], dtype=np.bool_),
            np.asarray(pi, dtype=np.int64),
            np.asarray(need_pi, dtype=np.bool_),
            np.asarray(False, dtype=np.bool_),
            np.asarray(self.length[src_idx], dtype=np.int64),
            np.asarray(self.valid_mask[src_idx], dtype=np.bool_),
            np.asarray(self.object_present[src_idx], dtype=np.bool_),
            np.asarray(self.completion_label[src_idx], dtype=np.bool_),
            extra,
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
        shape_orig = data.shape
        data = data.reshape((-1, 3))
        data = -1. + 2. * (data - self.min) / (self.max - self.min)
        return data.reshape(shape_orig)

    def normalize_torch(self, data):
        if not self.normalize_enabled:
            return data
        shape_orig = data.shape
        data = data.reshape((-1, 3))
        data = -1. + 2. * (data - self.min_torch) / (self.max_torch - self.min_torch)
        return data.reshape(shape_orig)

    def denormalize(self, data):
        if not self.normalize_enabled:
            return data
        shape_orig = data.shape
        data = data.reshape((-1, 3))
        data = (data + 1.) * (self.max - self.min) / 2. + self.min
        return data.reshape(shape_orig)

    def denormalize_torch(self, data):
        if not self.normalize_enabled:
            return data
        shape_orig = data.shape
        data = data.reshape((-1, 3))
        data = (data + 1.) * (self.max_torch - self.min_torch) / 2. + self.min_torch
        return data.reshape(shape_orig)
