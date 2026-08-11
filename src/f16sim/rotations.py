import numpy as np


def rotation_x(phi):
    """
    Coordinate transformation matrix for a rotation about the x-axis.

    Parameters
    ----------
    phi : float
        Rotation angle in radians.

    Returns
    -------
    numpy.ndarray
        3x3 rotation matrix.
    """
    c = np.cos(phi)
    s = np.sin(phi)

    return np.array([
        [1.0, 0.0, 0.0],
        [0.0,   c,   s],
        [0.0,  -s,   c]
    ])

def rotation_y(theta):
    """
    Coordinate transformation matrix for a rotation about the y-axis.

    Parameters
    ----------
    theta : float
        Rotation angle in radians.

    Returns
    -------
    numpy.ndarray
        3x3 rotation matrix.
    """
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [ c, 0.0, -s],
        [0.0, 1.0, 0.0],
        [ s, 0.0,  c]
    ])


def rotation_z(psi):
    """
    Coordinate transformation matrix for a rotation about the z-axis.

    Parameters
    ----------
    psi : float
        Rotation angle in radians.

    Returns
    -------
    numpy.ndarray
        3x3 rotation matrix.
    """
    c = np.cos(psi)
    s = np.sin(psi)

    return np.array([
        [ c,  s, 0.0],
        [-s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])

def ned_to_body(phi, theta, psi):
    """
    Transform vector components from NED coordinates to body (FRD) coordinates.

    Parameters
    ----------
    phi : float
        Roll angle in radians.
    theta : float
        Pitch angle in radians.
    psi : float
        Yaw angle in radians.

    Returns
    -------
    numpy.ndarray
        3x3 direction cosine matrix from NED to body.
    """
    return (
        rotation_x(phi)
        @ rotation_y(theta)
        @ rotation_z(psi)
    )


def body_to_ned(phi, theta, psi):
    """
    Transform vector components from body (FRD) coordinates to NED coordinates.
    """
    C_ned_to_body = ned_to_body(phi, theta, psi)

    return C_ned_to_body.T