"""Trim calculations for the F-16 model."""

import numpy as np
from scipy.optimize import minimize

from .attitude import euler_to_quaternion
from .engine import tgear
from .f16_model import f16_state_derivative
from .parameters import FT_TO_METER, reference_cg_fraction


def _straight_level_state(true_airspeed, altitude_m, throttle, alpha_deg):
    """Construct the 14-state symmetric straight-and-level trim state."""
    alpha = np.deg2rad(alpha_deg)
    state = np.zeros(14, dtype=float)
    state[2] = -altitude_m
    state[3] = true_airspeed * np.cos(alpha)
    state[5] = true_airspeed * np.sin(alpha)
    state[6:10] = euler_to_quaternion(0.0, alpha, 0.0)
    state[13] = tgear(throttle)
    return state


def _longitudinal_rates(state, state_dot, true_airspeed):
    """Return true-airspeed, angle-of-attack, and pitch-rate derivatives."""
    u = state[3]
    w = state[5]
    u_dot = state_dot[3]
    w_dot = state_dot[5]

    vt_dot = (u * u_dot + w * w_dot) / true_airspeed
    alpha_dot = (u * w_dot - w * u_dot) / (u * u + w * w)
    q_dot = state_dot[11]
    return float(vt_dot), float(alpha_dot), float(q_dot)


def trim_straight_level(
    true_airspeed,
    altitude_m,
    initial_guess=None,
    cg_fraction=reference_cg_fraction,
    max_iterations=2000,
):
    """Find a symmetric straight-and-level longitudinal F-16 trim state.

    This routine trims throttle, elevator deflection, and angle of attack at
    a specified true airspeed and altitude. It imposes zero sideslip, bank,
    yaw, and body rates, with pitch attitude equal to angle of attack. This is
    the first longitudinal trim capability and is not a general turning or
    six-degree-of-freedom trim solver.

    Parameters
    ----------
    true_airspeed : float
        Desired true airspeed in meters per second. Must be positive.
    altitude_m : float
        Desired altitude above the NED origin in meters.
    initial_guess : array_like, optional
        Initial simplex point ordered as ``[throttle, elevator_deg,
        alpha_deg]``. The default is ``[0.2, 0.0, 3.0]``.
    cg_fraction : float, optional
        Center-of-gravity position as a fraction of mean aerodynamic chord.
    max_iterations : int, optional
        Maximum number of Nelder-Mead iterations.

    Returns
    -------
    dict
        Optimization status, trimmed controls and angle of attack, the final
        14-state vector and derivative, and the residual longitudinal rates.
        Angles in scalar outputs are in degrees; ``alpha_dot`` and ``q_dot``
        are in radians per second and radians per second squared,
        respectively.

    Raises
    ------
    ValueError
        If ``true_airspeed`` is not positive or ``initial_guess`` does not
        contain exactly three values.
    """
    true_airspeed = float(true_airspeed)
    altitude_m = float(altitude_m)
    cg_fraction = float(cg_fraction)
    if true_airspeed <= 0.0:
        raise ValueError("true_airspeed must be positive")

    if initial_guess is None:
        initial_guess = np.array([0.2, 0.0, 3.0], dtype=float)
    else:
        initial_guess = np.asarray(initial_guess, dtype=float)
        if initial_guess.shape != (3,):
            raise ValueError("initial_guess must contain exactly three values")

    def evaluate(variables):
        throttle, elevator_deg, alpha_deg = variables
        state = _straight_level_state(
            true_airspeed, altitude_m, throttle, alpha_deg
        )
        state_dot = f16_state_derivative(
            state,
            throttle=throttle,
            elevator_deg=elevator_deg,
            aileron_deg=0.0,
            rudder_deg=0.0,
            cg_fraction=cg_fraction,
        )
        rates = _longitudinal_rates(state, state_dot, true_airspeed)
        return state, state_dot, rates

    def objective(variables):
        _, _, (vt_dot, alpha_dot, q_dot) = evaluate(variables)
        vt_dot_ft_s2 = vt_dot / FT_TO_METER
        return vt_dot_ft_s2**2 + 100.0 * alpha_dot**2 + 10.0 * q_dot**2

    result = minimize(
        objective,
        initial_guess,
        method="Nelder-Mead",
        options={"maxiter": int(max_iterations)},
    )

    throttle, elevator_deg, alpha_deg = result.x
    state, state_dot, rates = evaluate(result.x)
    vt_dot, alpha_dot, q_dot = rates

    return {
        "success": bool(result.success),
        "message": str(result.message),
        "cost": float(objective(result.x)),
        "iterations": int(result.nit),
        "throttle": float(throttle),
        "elevator_deg": float(elevator_deg),
        "alpha_deg": float(alpha_deg),
        "state": state,
        "state_dot": state_dot,
        "true_airspeed": true_airspeed,
        "altitude_m": altitude_m,
        "cg_fraction": cg_fraction,
        "VT_dot": vt_dot,
        "alpha_dot": alpha_dot,
        "q_dot": q_dot,
        "engine_power": float(state[13]),
        "engine_power_dot": float(state_dot[13]),
    }
