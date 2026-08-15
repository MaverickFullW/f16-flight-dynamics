"""Reduced nonlinear models used for future F-16 linearization."""

import numpy as np

from .attitude import euler_to_quaternion
from .engine import tgear
from .f16_model import f16_state_derivative
from .parameters import reference_cg_fraction


_LONGITUDINAL_STATE_STEPS = np.array([0.1, 1e-3, 1e-3, 1e-3])
_LONGITUDINAL_CONTROL_STEPS = np.array([1e-3, 1e-2])


def longitudinal_state_derivative(
    x_lon,
    u_lon,
    altitude_m,
    cg_fraction=reference_cg_fraction,
):
    """Evaluate the reduced symmetric longitudinal F-16 dynamics.

    This function maps the reduced longitudinal state and controls into the
    full nonlinear 14-state F-16 model, then maps its derivative back to the
    reduced coordinates. It is intended as the nonlinear basis for future
    numerical linearization; it does not compute state or control Jacobians.

    Parameters
    ----------
    x_lon : array_like
        Reduced state ``[VT, alpha, theta, q]``, where true airspeed is in
        meters per second and angles and pitch rate are in radians and
        radians per second.
    u_lon : array_like
        Longitudinal controls ``[throttle, elevator_deg]``.
    altitude_m : float
        Altitude above the NED origin in meters.
    cg_fraction : float, optional
        Center-of-gravity position as a fraction of mean aerodynamic chord.

    Returns
    -------
    numpy.ndarray
        Reduced derivative ``[VT_dot, alpha_dot, theta_dot, q_dot]``.

    Raises
    ------
    ValueError
        If ``x_lon`` does not have shape ``(4,)``, ``u_lon`` does not have
        shape ``(2,)``, or true airspeed is not positive.
    """
    x_lon = np.asarray(x_lon, dtype=float)
    u_lon = np.asarray(u_lon, dtype=float)
    if x_lon.shape != (4,):
        raise ValueError("x_lon must have shape (4,)")
    if u_lon.shape != (2,):
        raise ValueError("u_lon must have shape (2,)")

    true_airspeed, alpha, theta, q = x_lon
    throttle, elevator_deg = u_lon
    if true_airspeed <= 0.0:
        raise ValueError("VT must be positive")

    u_body = true_airspeed * np.cos(alpha)
    w_body = true_airspeed * np.sin(alpha)

    state = np.zeros(14, dtype=float)
    state[2] = -float(altitude_m)
    state[3] = u_body
    state[5] = w_body
    state[6:10] = euler_to_quaternion(0.0, theta, 0.0)
    state[11] = q
    state[13] = tgear(throttle)

    full_state_dot = f16_state_derivative(
        state,
        throttle=throttle,
        elevator_deg=elevator_deg,
        aileron_deg=0.0,
        rudder_deg=0.0,
        cg_fraction=cg_fraction,
    )

    u_dot = full_state_dot[3]
    w_dot = full_state_dot[5]
    vt_dot = (u_body * u_dot + w_body * w_dot) / true_airspeed
    alpha_dot = (
        u_body * w_dot - w_body * u_dot
    ) / (u_body**2 + w_body**2)
    q_dot = full_state_dot[11]

    return np.array([vt_dot, alpha_dot, q, q_dot], dtype=float)


def linearize_longitudinal(
    x_equilibrium,
    u_equilibrium,
    altitude_m,
    cg_fraction=reference_cg_fraction,
):
    """Numerically linearize the reduced longitudinal F-16 model.

    Centered finite differences are evaluated about the supplied equilibrium
    using perturbations appropriate to each state and control variable. The
    reduced state is ``[VT, alpha, theta, q]`` and the controls are
    ``[throttle, elevator_deg]``.

    Parameters
    ----------
    x_equilibrium : array_like
        Equilibrium reduced state ``[VT, alpha, theta, q]``. Airspeed is in
        meters per second, angles are in radians, and pitch rate is in
        radians per second.
    u_equilibrium : array_like
        Equilibrium controls ``[throttle, elevator_deg]``.
    altitude_m : float
        Altitude above the NED origin in meters.
    cg_fraction : float, optional
        Center-of-gravity position as a fraction of mean aerodynamic chord.

    Returns
    -------
    A : numpy.ndarray
        State Jacobian with shape ``(4, 4)``.
    B : numpy.ndarray
        Control Jacobian with shape ``(4, 2)``.

    Raises
    ------
    ValueError
        If ``x_equilibrium`` does not have shape ``(4,)`` or
        ``u_equilibrium`` does not have shape ``(2,)``.
    """
    x_equilibrium = np.asarray(x_equilibrium, dtype=float)
    u_equilibrium = np.asarray(u_equilibrium, dtype=float)
    if x_equilibrium.shape != (4,):
        raise ValueError("x_equilibrium must have shape (4,)")
    if u_equilibrium.shape != (2,):
        raise ValueError("u_equilibrium must have shape (2,)")

    def evaluate(x_lon, u_lon):
        return longitudinal_state_derivative(
            x_lon,
            u_lon,
            altitude_m=altitude_m,
            cg_fraction=cg_fraction,
        )

    A = np.empty((4, 4), dtype=float)
    for column, step in enumerate(_LONGITUDINAL_STATE_STEPS):
        perturbation = np.zeros(4, dtype=float)
        perturbation[column] = step
        A[:, column] = (
            evaluate(x_equilibrium + perturbation, u_equilibrium)
            - evaluate(x_equilibrium - perturbation, u_equilibrium)
        ) / (2.0 * step)

    B = np.empty((4, 2), dtype=float)
    for column, step in enumerate(_LONGITUDINAL_CONTROL_STEPS):
        perturbation = np.zeros(2, dtype=float)
        perturbation[column] = step
        B[:, column] = (
            evaluate(x_equilibrium, u_equilibrium + perturbation)
            - evaluate(x_equilibrium, u_equilibrium - perturbation)
        ) / (2.0 * step)

    return A, B
