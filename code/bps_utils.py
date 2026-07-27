import numpy as np


def make_bps_basis(num_points=256, radius=1.0, seed=12345):
    """Create a deterministic set of points uniformly distributed in a 3D ball."""
    num_points = int(num_points)
    radius = float(radius)
    if num_points <= 0:
        raise ValueError("num_points must be positive.")
    if radius <= 0.0:
        raise ValueError("radius must be positive.")

    rng = np.random.default_rng(int(seed))
    directions = rng.normal(size=(num_points, 3)).astype(np.float32)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.maximum(norms, 1e-8)
    radii = np.cbrt(rng.random((num_points, 1))).astype(np.float32)
    return (directions * radii * radius).astype(np.float32)


def encode_bps(points, basis, chunk_size=64):
    """
    Encode an object point cloud with vector BPS residuals.

    Returns residual vectors, reconstructed surface proxies, and source indices.
    The proxy for basis point b_m is its nearest input point q_m, so
    q_m = b_m + residual_m.
    """
    points = np.asarray(points, dtype=np.float32)
    basis = np.asarray(basis, dtype=np.float32)
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"Expected points with shape [N, 3], got {points.shape}.")
    if basis.ndim != 2 or basis.shape[-1] != 3:
        raise ValueError(f"Expected basis with shape [M, 3], got {basis.shape}.")
    if len(points) == 0:
        residuals = np.zeros_like(basis)
        indices = np.zeros(len(basis), dtype=np.int64)
        return residuals, basis.copy(), indices

    chunk_size = max(1, int(chunk_size))
    nearest_indices = np.empty(len(basis), dtype=np.int64)
    for start in range(0, len(basis), chunk_size):
        stop = min(start + chunk_size, len(basis))
        diff = basis[start:stop, None, :] - points[None, :, :]
        dist2 = np.einsum("mnc,mnc->mn", diff, diff)
        nearest_indices[start:stop] = np.argmin(dist2, axis=1)

    proxies = points[nearest_indices]
    residuals = proxies - basis
    return (
        residuals.astype(np.float32),
        proxies.astype(np.float32),
        nearest_indices,
    )
