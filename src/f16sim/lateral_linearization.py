"""Reduced lateral-directional dynamics and numerical linearization."""

import numpy as np

from .air_data import air_data_from_body_velocity
from .attitude import euler_to_quaternion, quaternion_normalize
from .f16_model import f16_state_derivative
from .parameters import reference_cg_fraction


_LATERAL_STATE_STEPS = np.array([1e-4, 1e-4, 1e-4, 1e-4])
_LATERAL_CONTROL_STEPS = np.array([1e-2, 1e-2])


def _pitch_angle(quaternion):
    q0, q1, q2, q3 = quaternion_normalize(quaternion)
    sin_theta = 2.0 * (q0 * q2 - q3 * q1)
    return float(np.arcsin(np.clip(sin_theta, -1.0, 1.0)))


def lateral_state_derivative(
    x_lat,
    u_lat,
    trim_state,
    throttle,
    elevator_deg,
    cg_fraction=reference_cg_fraction,
):
    """Evaluate reduced lateral dynamics about a longitudinal trim state.

    The reduced state is ``[beta, phi, p, r]`` in radians and radians per
    second. Lateral controls are ``[aileron_deg, rudder_deg]``.
    """
    x_lat = np.asarray(x_lat, dtype=float)
    u_lat = np.asarray(u_lat, dtype=float)
    trim_state = np.asarray(trim_state, dtype=float)
    if x_lat.shape != (4,):
        raise ValueError("x_lat must have shape (4,)")
    if u_lat.shape != (2,):
        raise ValueError("u_lat must have shape (2,)")
    if trim_state.shape != (14,):
        raise ValueError("trim_state must have shape (14,)")

    beta, phi, p, r = x_lat
    aileron_deg, rudder_deg = u_lat
    true_airspeed, alpha_deg, _ = air_data_from_body_velocity(trim_state[3:6])
    alpha = np.deg2rad(alpha_deg)
    theta = _pitch_angle(trim_state[6:10])

    state = trim_state.copy()
    longitudinal_speed = true_airspeed * np.cos(beta)
    state[3] = longitudinal_speed * np.cos(alpha)
    state[4] = true_airspeed * np.sin(beta)
    state[5] = longitudinal_speed * np.sin(alpha)
    state[6:10] = euler_to_quaternion(phi, theta, 0.0)
    state[10] = p
    state[12] = r

    state_dot = f16_state_derivative(
        state,
        throttle=throttle,
        elevator_deg=elevator_deg,
        aileron_deg=aileron_deg,
        rudder_deg=rudder_deg,
        cg_fraction=cg_fraction,
    )

    velocity = state[3:6]
    velocity_dot = state_dot[3:6]
    vt_dot = float(np.dot(velocity, velocity_dot) / true_airspeed)
    cos_beta = np.cos(beta)
    beta_dot = (
        velocity_dot[1] * true_airspeed - velocity[1] * vt_dot
    ) / (true_airspeed**2 * cos_beta)

    q = state[11]
    phi_dot = p + np.tan(theta) * (q * np.sin(phi) + r * np.cos(phi))
    return np.array(
        [beta_dot, phi_dot, state_dot[10], state_dot[12]], dtype=float
    )


def linearize_lateral(
    trim_state,
    throttle,
    elevator_deg,
    cg_fraction=reference_cg_fraction,
):
    """Numerically linearize ``[beta, phi, p, r]`` lateral dynamics."""
    trim_state = np.asarray(trim_state, dtype=float)
    if trim_state.shape != (14,):
        raise ValueError("trim_state must have shape (14,)")

    equilibrium_state = np.zeros(4, dtype=float)
    equilibrium_controls = np.zeros(2, dtype=float)

    def evaluate(x_lat, u_lat):
        return lateral_state_derivative(
            x_lat,
            u_lat,
            trim_state,
            throttle,
            elevator_deg,
            cg_fraction=cg_fraction,
        )

    A_lat = np.empty((4, 4), dtype=float)
    for column, step in enumerate(_LATERAL_STATE_STEPS):
        perturbation = np.zeros(4, dtype=float)
        perturbation[column] = step
        A_lat[:, column] = (
            evaluate(equilibrium_state + perturbation, equilibrium_controls)
            - evaluate(equilibrium_state - perturbation, equilibrium_controls)
        ) / (2.0 * step)

    B_lat = np.empty((4, 2), dtype=float)
    for column, step in enumerate(_LATERAL_CONTROL_STEPS):
        perturbation = np.zeros(2, dtype=float)
        perturbation[column] = step
        B_lat[:, column] = (
            evaluate(equilibrium_state, equilibrium_controls + perturbation)
            - evaluate(equilibrium_state, equilibrium_controls - perturbation)
        ) / (2.0 * step)

    return A_lat, B_lat
