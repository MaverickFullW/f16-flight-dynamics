"""Aerodynamic air-data quantities derived from body-frame velocity."""

import numpy as np


def air_data_from_body_velocity(velocity_body):
    """
    Compute airspeed and aerodynamic angles from FRD body velocity.

    The body axes use the forward-right-down (FRD) convention. Therefore,
    positive ``u``, ``v``, and ``w`` point forward, right, and down,
    respectively. Positive ``w`` relative to ``u`` produces positive angle
    of attack, and positive ``v`` produces positive sideslip angle.

    Parameters
    ----------
    velocity_body : array_like
        Body-frame velocity ``[u, v, w]`` in meters per second.

    Returns
    -------
    true_airspeed : float
        Magnitude of the body-frame velocity in meters per second.
    alpha_deg : float
        Angle of attack in degrees.
    beta_deg : float
        Sideslip angle in degrees.

    Raises
    ------
    ValueError
        If ``velocity_body`` does not have shape ``(3,)`` or if true
        airspeed is not positive.
    """
    velocity_body = np.asarray(velocity_body, dtype=float)
    if velocity_body.shape != (3,):
        raise ValueError("velocity_body must have shape (3,)")

    u, v, w = velocity_body
    true_airspeed = np.sqrt(u**2 + v**2 + w**2)
    if true_airspeed <= 0.0:
        raise ValueError("true airspeed must be positive")

    alpha_rad = np.arctan2(w, u)
    beta_ratio = np.clip(v / true_airspeed, -1.0, 1.0)
    beta_rad = np.arcsin(beta_ratio)

    return (
        float(true_airspeed),
        float(np.degrees(alpha_rad)),
        float(np.degrees(beta_rad)),
    )
