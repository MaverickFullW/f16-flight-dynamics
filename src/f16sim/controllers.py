"""Reusable feedback controllers for the F-16 model."""

import numpy as np


DEFAULT_KQ = 5.0
DEFAULT_KTHETA = 0.5


def _finite_scalar(value, name):
    try:
        value = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite scalar") from error
    if value.shape != () or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")
    return float(value)


def pitch_attitude_control(
    theta_cmd,
    theta,
    q,
    Kq=DEFAULT_KQ,
    Ktheta=DEFAULT_KTHETA,
):
    """Compute cascaded pitch-attitude and pitch-rate feedback commands.

    Angles are in radians, pitch rates are in radians per second, and the
    returned elevator perturbation is in degrees.

    Returns
    -------
    q_cmd : float
        Commanded pitch rate in radians per second.
    elevator_perturbation_deg : float
        Elevator perturbation in degrees.
    """
    theta_cmd = _finite_scalar(theta_cmd, "theta_cmd")
    theta = _finite_scalar(theta, "theta")
    q = _finite_scalar(q, "q")
    Kq = _finite_scalar(Kq, "Kq")
    Ktheta = _finite_scalar(Ktheta, "Ktheta")
    if Kq <= 0.0:
        raise ValueError("Kq must be positive")
    if Ktheta < 0.0:
        raise ValueError("Ktheta must be nonnegative")

    q_cmd = Ktheta * (theta_cmd - theta)
    elevator_perturbation_deg = Kq * (q - q_cmd)
    return float(q_cmd), float(elevator_perturbation_deg)
