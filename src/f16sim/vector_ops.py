import numpy as np


def skew(u):
    """
    Return the skew-symmetric matrix associated with a 3D vector.

    Parameters
    ----------
    u : array_like
        Three-component vector [ux, uy, uz].

    Returns
    -------
    numpy.ndarray
        3x3 skew-symmetric matrix such that skew(u) @ v = cross(u, v).
    """
    ux, uy, uz = u

    return np.array([
        [0.0, -uz,  uy],
        [uz,   0.0, -ux],
        [-uy,  ux,  0.0]
    ])