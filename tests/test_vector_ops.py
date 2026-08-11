import numpy as np

from src.f16sim.vector_ops import skew


def test_skew_matches_cross_product():
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([4.0, 5.0, 6.0])

    result_matrix = skew(u) @ v
    result_cross = np.cross(u, v)

    assert np.allclose(result_matrix, result_cross)