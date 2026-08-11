import numpy as np


def quaternion_normalize(q):
    """
    Normalize a scalar-first quaternion.

    Parameters
    ----------
    q : array_like
        Quaternion ``[q0, q1, q2, q3]`` with the scalar component first.

    Returns
    -------
    numpy.ndarray
        Unit quaternion with the scalar component first.

    Raises
    ------
    ValueError
        If the quaternion has zero magnitude.
    """
    q = np.asarray(q, dtype=float)
    magnitude = np.linalg.norm(q)

    if magnitude == 0.0:
        raise ValueError("Cannot normalize a zero-magnitude quaternion.")

    return q / magnitude


def euler_to_quaternion(phi, theta, psi):
    """
    Convert roll, pitch, and yaw angles to a scalar-first quaternion.

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
        Normalized scalar-first quaternion ``[q0, q1, q2, q3]`` describing
        the NED-to-body attitude.
    """
    half_phi = 0.5 * phi
    half_theta = 0.5 * theta
    half_psi = 0.5 * psi

    c_phi = np.cos(half_phi)
    s_phi = np.sin(half_phi)
    c_theta = np.cos(half_theta)
    s_theta = np.sin(half_theta)
    c_psi = np.cos(half_psi)
    s_psi = np.sin(half_psi)

    q = np.array([
        c_phi * c_theta * c_psi + s_phi * s_theta * s_psi,
        s_phi * c_theta * c_psi - c_phi * s_theta * s_psi,
        c_phi * s_theta * c_psi + s_phi * c_theta * s_psi,
        c_phi * c_theta * s_psi - s_phi * s_theta * c_psi,
    ])

    return quaternion_normalize(q)


def quaternion_rate(q, omega_body):
    """
    Compute the quaternion time derivative from body angular velocity.

    Parameters
    ----------
    q : array_like
        Scalar-first quaternion ``[q0, q1, q2, q3]`` describing the
        NED-to-body attitude.
    omega_body : array_like
        Body angular velocity ``[p, q, r]`` in radians per second, expressed
        in the body FRD frame.

    Returns
    -------
    numpy.ndarray
        Quaternion derivative ``[q0_dot, q1_dot, q2_dot, q3_dot]``.
    """
    q = np.asarray(q, dtype=float)
    p, q_rate, r = omega_body

    omega = np.array([
        [0.0, -p,     -q_rate, -r],
        [p,    0.0,    r,      -q_rate],
        [q_rate, -r,   0.0,     p],
        [r,     q_rate, -p,     0.0]
    ])

    return 0.5 * omega @ q


def quaternion_to_dcm(q):
    """
    Convert a scalar-first quaternion to the NED-to-body DCM.

    Parameters
    ----------
    q : array_like
        Quaternion ``[q0, q1, q2, q3]`` describing the NED-to-body attitude.

    Returns
    -------
    numpy.ndarray
        3x3 direction cosine matrix transforming NED components to body FRD
        components.
    """
    q0, q1, q2, q3 = quaternion_normalize(q)

    return np.array([
        [q0**2 + q1**2 - q2**2 - q3**2,
         2.0 * (q1 * q2 + q0 * q3),
         2.0 * (q1 * q3 - q0 * q2)],
        [2.0 * (q1 * q2 - q0 * q3),
         q0**2 - q1**2 + q2**2 - q3**2,
         2.0 * (q2 * q3 + q0 * q1)],
        [2.0 * (q1 * q3 + q0 * q2),
         2.0 * (q2 * q3 - q0 * q1),
         q0**2 - q1**2 - q2**2 + q3**2]
    ])
