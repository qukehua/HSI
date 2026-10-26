"""Metric, signed human-object penetration loss (negative SDF means inside).

SDF volumes use OMOMO's xyz array layout and metadata convention. They stay in
the object's original rest frame; never normalize their distance values using
the human-motion coordinate ranges. No point-cloud/convex-hull fallback is used.
"""

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class ObjectSDFStore:
    """Cache original-resolution grids per device, not once per frame.

    IDs are indices into the sorted .npy filenames, shared with the dataset.
    The bounded cache limits persistent device memory; autograd retains grids
    needed by the current step even if they are evicted from this cache.
    """

    def __init__(self, folder, max_cached=4):
        self.folder = Path(folder)
        self.names = tuple(sorted(p.name[:-4] for p in self.folder.glob("*.npy")))
        if not self.names:
            raise FileNotFoundError(f"No object SDF grids found in {self.folder}")
        self.name_to_id = {name: idx for idx, name in enumerate(self.names)}
        self.max_cached = max(1, int(max_cached))
        self._cache = OrderedDict()

    def _load(self, sdf_id, device, dtype):
        key = (int(sdf_id), str(device), dtype)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        if not 0 <= int(sdf_id) < len(self.names):
            raise ValueError(f"Unknown object SDF ID: {sdf_id}")
        name = self.names[int(sdf_id)]
        path = self.folder / f"{name}.npy"
        values = np.load(path, allow_pickle=False)
        if values.ndim != 3 or min(values.shape) < 2 or not np.isfinite(values).all():
            raise ValueError(f"Invalid 3-D SDF grid: {path}")
        with (self.folder / f"{name}.json").open(encoding="utf-8") as f:
            metadata = json.load(f)
        if "coord_center" in metadata:
            center = np.asarray(metadata["coord_center"], dtype=np.float32)
            lower = center + np.asarray(metadata["coord_min"], dtype=np.float32)
            upper = center + np.asarray(metadata["coord_max"], dtype=np.float32)
        elif "centroid" in metadata and "extents" in metadata:
            center = np.asarray(metadata["centroid"], dtype=np.float32)
            radius = np.asarray(metadata["extents"], dtype=np.float32).max() / 2
            lower, upper = center - radius, center + radius
        else:
            raise ValueError(f"Unsupported object SDF metadata: {name}")
        if (lower.shape != (3,) or upper.shape != (3,)
                or not np.isfinite([lower, upper]).all() or np.any(upper <= lower)):
            raise ValueError(f"Invalid object SDF bounds: {name}")
        result = (
            torch.as_tensor(values, device=device, dtype=dtype)[None, None],
            torch.as_tensor(lower, device=device, dtype=dtype),
            torch.as_tensor(upper, device=device, dtype=dtype),
        )
        self._cache[key] = result
        while len(self._cache) > self.max_cached:
            self._cache.popitem(last=False)
        return result

    def query(self, sdf_id, rest_points):
        """Trilinear signed distance in rest-frame units, with point gradients.

        Out-of-grid points cannot be classified as interior: do not extend a
        negative boundary voxel indefinitely using grid_sample's border mode.
        """
        volume, lower, upper = self._load(sdf_id, rest_points.device, rest_points.dtype)
        coordinates = 2 * (rest_points - lower) / (upper - lower) - 1
        inside_grid = (coordinates.abs() <= 1).all(dim=-1)
        # grid_sample's final coordinates are W/H/D, while OMOMO stores x/y/z.
        grid = coordinates[..., [2, 1, 0]].reshape(1, -1, 1, 1, 3)
        distances = F.grid_sample(
            volume, grid, mode="bilinear", padding_mode="border", align_corners=True,
        ).reshape(rest_points.shape[:-1])
        return torch.where(inside_grid, distances, torch.full_like(distances, float("inf")))


def human_object_collision_loss(
        human_motion, sdf_store, object_sdf_ids, object_sdf_rotations,
        object_sdf_translations, object_sdf_scales, object_present, valid_mask,
        norm_min=None, norm_max=None, motion_normalized=None,
        tolerance_m=0.0, joint_ids=None, return_coverage=False):
    """Mean squared penetration depth in m^2; surface/outside points cost zero.

    Each component's physical model-space pose is x = scale * R @ x_rest + t.
    Negative IDs explicitly disable unavailable/excluded components. The union
    of available components uses the minimum signed distance. Missing geometry
    never falls back to the former unsigned proximity penalty.
    """
    if tolerance_m < 0:
        raise ValueError("Penetration tolerance must be nonnegative (meters).")
    batch, frames, channels = human_motion.shape
    if channels % 3:
        raise ValueError("Human motion must contain xyz joint coordinates.")
    if sdf_store is None or any(value is None for value in (
            object_sdf_ids, object_sdf_rotations, object_sdf_translations, object_sdf_scales)):
        raise ValueError("Signed collision loss requires SDF grids and their per-frame poses.")
    # SDF interpolation in at least float32, also under mixed-precision training.
    dtype = torch.float64 if human_motion.dtype == torch.float64 else torch.float32
    device = human_motion.device
    points = human_motion.to(dtype).reshape(batch, frames, -1, 3)
    if joint_ids is not None:
        ids = torch.as_tensor(joint_ids, device=device, dtype=torch.long)
        if ids.ndim != 1 or not ids.numel() or bool(((ids < 0) | (ids >= points.shape[2])).any()):
            raise ValueError("Collision joint_ids must select existing joints.")
        points = points.index_select(2, ids)

    normalized = torch.zeros(batch, device=device, dtype=torch.bool)
    if motion_normalized is not None:
        normalized = torch.as_tensor(motion_normalized, device=device, dtype=torch.bool).reshape(batch)
    if bool(normalized.any()):
        if norm_min is None or norm_max is None:
            raise ValueError("Normalized motion requires coordinate bounds for metric SDF queries.")
        lower = torch.as_tensor(norm_min, device=device, dtype=dtype).reshape(batch, 3)
        upper = torch.as_tensor(norm_max, device=device, dtype=dtype).reshape(batch, 3)
        if bool((upper <= lower).any()):
            raise ValueError("Motion normalization bounds must have positive ranges.")
        physical = (points + 1) * (upper - lower)[:, None, None] / 2 + lower[:, None, None]
        points = torch.where(normalized[:, None, None, None], physical, points)

    sdf_ids = torch.as_tensor(object_sdf_ids, device=device, dtype=torch.long)
    if sdf_ids.ndim != 2 or sdf_ids.shape[0] != batch or bool((sdf_ids < -1).any()):
        raise ValueError("object_sdf_ids must have shape [B, parts], with -1 for unavailable parts.")
    parts = sdf_ids.shape[1]
    rotations = torch.as_tensor(object_sdf_rotations, device=device, dtype=dtype)
    translations = torch.as_tensor(object_sdf_translations, device=device, dtype=dtype)
    scales = torch.as_tensor(object_sdf_scales, device=device, dtype=dtype)
    if (rotations.shape != (batch, frames, parts, 3, 3)
            or translations.shape != (batch, frames, parts, 3)
            or scales.shape != (batch, frames, parts)):
        raise ValueError("SDF poses must match [B, T, parts] and contain rotations/translations/scales.")
    present = torch.as_tensor(object_present, device=device, dtype=torch.bool).reshape(batch)
    valid = (torch.ones(batch, frames, device=device, dtype=torch.bool) if valid_mask is None
             else torch.as_tensor(valid_mask, device=device, dtype=torch.bool))
    if valid.shape != (batch, frames):
        raise ValueError("valid_mask must have shape [B, T].")
    active = present[:, None] & valid
    supported = (sdf_ids >= 0).any(dim=1)
    covered = active & supported[:, None]
    coverage = covered.sum().to(dtype) / active.sum().clamp_min(1)
    zero = human_motion.sum() * 0
    if not bool(covered.any()):
        return (zero, coverage) if return_coverage else zero

    # Sanitize padded frames before querying: they do not participate in loss.
    points = torch.where(covered[:, :, None, None], points, torch.zeros_like(points))
    signed = torch.full(points.shape[:-1], float("inf"), device=device, dtype=dtype)
    with torch.autocast(device_type=device.type, enabled=False):
        for part in range(parts):
            part_active = covered & (sdf_ids[:, part] >= 0)[:, None]
            for sdf_id in torch.unique(sdf_ids[part_active.any(dim=1), part]).tolist():
                sample_ids = torch.where((sdf_ids[:, part] == sdf_id) & part_active.any(dim=1))[0]
                mask = part_active[sample_ids]
                r = rotations[sample_ids, :, part]
                t = translations[sample_ids, :, part]
                s = scales[sample_ids, :, part]
                if (not bool(torch.isfinite(r[mask]).all()) or not bool(torch.isfinite(t[mask]).all())
                        or not bool(torch.isfinite(s[mask]).all()) or bool((s[mask] <= 0).any())):
                    raise ValueError("Active SDF poses must be finite with positive object scales.")
                r = torch.where(mask[:, :, None, None], r, torch.eye(3, device=device, dtype=dtype))
                t = torch.where(mask[:, :, None], t, torch.zeros_like(t))
                s = torch.where(mask, s, torch.ones_like(s))
                rest = torch.einsum("btji,btik->btjk", points[sample_ids] - t[:, :, None], r)
                rest = rest / s[:, :, None, None]
                distance = sdf_store.query(sdf_id, rest) * s[:, :, None]
                # Functional update keeps the gradient from both part queries.
                part_distances = torch.full_like(signed, float("inf")).index_copy(0, sample_ids, distance)
                signed = torch.minimum(signed, part_distances)
        depth = (-signed - float(tolerance_m)).clamp_min(0)
        penalty = torch.where(covered[:, :, None], depth.square(), torch.zeros_like(depth))
        loss = penalty.sum() / (covered.sum() * points.shape[2]).clamp_min(1)
    return (loss, coverage) if return_coverage else loss
