import sys
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from bps_utils import encode_bps, make_bps_basis  # noqa: E402


def test_bps_basis_is_deterministic_and_inside_ball():
    basis_a = make_bps_basis(num_points=32, radius=0.75, seed=7)
    basis_b = make_bps_basis(num_points=32, radius=0.75, seed=7)
    np.testing.assert_allclose(basis_a, basis_b)
    assert np.linalg.norm(basis_a, axis=1).max() <= 0.75 + 1e-6


def test_vector_bps_reconstructs_nearest_surface_proxies():
    basis = np.asarray([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]], dtype=np.float32)
    points = np.asarray([[0.2, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    residuals, proxies, indices = encode_bps(points, basis)

    np.testing.assert_allclose(basis + residuals, proxies)
    np.testing.assert_allclose(proxies, points)
    assert indices.tolist() == [0, 1]
