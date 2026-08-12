"""Fixed-control time simulation for the current F-16 model."""

import numpy as np

from .f16_model import f16_state_derivative
from .integrators import rk4_step
from .parameters import reference_cg_fraction


def simulate_f16(
    initial_state,
    duration,
    dt,
    throttle,
    elevator_deg,
    aileron_deg,
    rudder_deg,
    cg_fraction=reference_cg_fraction,
):
    """
    Simulate the 14-state F-16 model with fixed controls.

    Atmospheric density and Mach number are computed internally by the F-16
    model from the state. Throttle and control-surface deflections remain
    constant throughout this first simulator.

    Parameters
    ----------
    initial_state : array_like
        Initial state ``[N, E, D, u, v, w, q0, q1, q2, q3, p, q, r,
        engine_power]``. NED position is in meters, FRD body velocity is in
        meters per second, the quaternion is scalar-first and dimensionless,
        body rates are in radians per second, and engine power uses the
        original F-16 power units.
    duration : float
        Simulation duration in seconds.
    dt : float
        Fixed integration time step in seconds.
    throttle : float
        Constant dimensionless throttle position.
    elevator_deg : float
        Constant elevator deflection in degrees.
    aileron_deg : float
        Constant aileron deflection in degrees.
    rudder_deg : float
        Constant rudder deflection in degrees.
    cg_fraction : float, optional
        Center-of-gravity position as a fraction of mean aerodynamic chord.

    Returns
    -------
    times : numpy.ndarray
        Simulation times in seconds, with shape ``(number_of_steps + 1,)``.
    states : numpy.ndarray
        State history with shape ``(number_of_steps + 1, 14)``.

    Raises
    ------
    ValueError
        If ``initial_state`` does not have shape ``(14,)``, if ``duration``
        or ``dt`` is not positive, or if the duration is not an integer
        number of time steps within floating-point tolerance.
    """
    initial_state = np.asarray(initial_state, dtype=float)
    if initial_state.shape != (14,):
        raise ValueError("initial_state must have shape (14,)")
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    step_ratio = duration / dt
    number_of_steps = round(step_ratio)
    if not np.isclose(step_ratio, number_of_steps, rtol=1e-12, atol=1e-12):
        raise ValueError("duration must be an integer number of time steps")

    times = np.linspace(0.0, duration, number_of_steps + 1)
    states = np.empty((number_of_steps + 1, 14), dtype=float)
    states[0] = initial_state

    for step in range(number_of_steps):
        states[step + 1] = rk4_step(
            f16_state_derivative,
            states[step],
            dt,
            throttle,
            elevator_deg,
            aileron_deg,
            rudder_deg,
            cg_fraction=cg_fraction,
        )

    return times, states
