"""Time simulation for the current F-16 model."""

import numpy as np

from .attitude import quaternion_normalize
from .f16_model import f16_state_derivative
from .integrators import rk4_step
from .parameters import reference_cg_fraction


def _control_value(control, time, name):
    value = control(time) if callable(control) else control
    value = np.asarray(value, dtype=float)
    if value.shape != ():
        if callable(control):
            raise ValueError(f"{name} callable must return a scalar")
        raise ValueError(f"{name} must be a scalar or callable")
    return float(value)


def _time_varying_rk4_step(state, time, dt, controls, cg_fraction):
    def derivative(stage_time, stage_state):
        stage_controls = [
            _control_value(control, stage_time, name)
            for name, control in controls
        ]
        return np.asarray(
            f16_state_derivative(
                stage_state,
                *stage_controls,
                cg_fraction=cg_fraction,
            ),
            dtype=float,
        )

    k1 = derivative(time, state)
    k2 = derivative(time + 0.5 * dt, state + 0.5 * dt * k1)
    k3 = derivative(time + 0.5 * dt, state + 0.5 * dt * k2)
    k4 = derivative(time + dt, state + dt * k3)
    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    next_state[6:10] = quaternion_normalize(next_state[6:10])
    return next_state


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
    Simulate the 14-state F-16 model with constant or time-varying controls.

    Atmospheric density and Mach number are computed internally by the F-16
    model from the state. Each control may be a scalar or a callable of time;
    callable controls are evaluated at the intermediate RK4 stage times.

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
    throttle : float or callable
        Dimensionless throttle position or function of time returning one.
    elevator_deg : float or callable
        Elevator deflection in degrees or function of time returning one.
    aileron_deg : float or callable
        Aileron deflection in degrees or function of time returning one.
    rudder_deg : float or callable
        Rudder deflection in degrees or function of time returning one.
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

    controls = (
        ("throttle", throttle),
        ("elevator_deg", elevator_deg),
        ("aileron_deg", aileron_deg),
        ("rudder_deg", rudder_deg),
    )
    has_time_varying_control = any(callable(control) for _, control in controls)

    for step in range(number_of_steps):
        if has_time_varying_control:
            states[step + 1] = _time_varying_rk4_step(
                states[step], times[step], dt, controls, cg_fraction
            )
        else:
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


def simulate_f16_feedback(
    initial_state,
    duration,
    dt,
    control_law,
    cg_fraction=0.35,
):
    """Simulate the F-16 with controls determined from time and state.

    ``control_law`` is evaluated at every classical RK4 stage and must return
    ``[throttle, elevator_deg, aileron_deg, rudder_deg]``.

    Parameters
    ----------
    initial_state : array_like
        Initial 14-element F-16 state.
    duration : float
        Simulation duration in seconds.
    dt : float
        Fixed integration time step in seconds.
    control_law : callable
        Function with signature ``control_law(t, state)`` returning four
        finite scalar-compatible controls.
    cg_fraction : float, optional
        Center-of-gravity position as a fraction of mean aerodynamic chord.

    Returns
    -------
    times : numpy.ndarray
        Simulation times with shape ``(number_of_steps + 1,)``.
    states : numpy.ndarray
        State history with shape ``(number_of_steps + 1, 14)``.

    Raises
    ------
    ValueError
        If an input shape or time value is invalid, or if ``control_law``
        does not return exactly four finite scalar-compatible values.
    """
    initial_state = np.asarray(initial_state, dtype=float)
    if initial_state.shape != (14,):
        raise ValueError("initial_state must have shape (14,)")
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if not callable(control_law):
        raise ValueError("control_law must be callable")

    step_ratio = duration / dt
    number_of_steps = round(step_ratio)
    if not np.isclose(step_ratio, number_of_steps, rtol=1e-12, atol=1e-12):
        raise ValueError("duration must be an integer number of time steps")

    def evaluate_controls(time, state):
        try:
            controls = np.asarray(control_law(time, state), dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "control_law must return four finite scalar-compatible values"
            ) from error
        if controls.shape != (4,) or not np.all(np.isfinite(controls)):
            raise ValueError(
                "control_law must return four finite scalar-compatible values"
            )
        return controls

    def derivative(time, state):
        controls = evaluate_controls(time, state)
        return np.asarray(
            f16_state_derivative(
                state,
                *controls,
                cg_fraction=cg_fraction,
            ),
            dtype=float,
        )

    def feedback_rk4_step(time, state):
        k1 = derivative(time, state)
        k2 = derivative(time + 0.5 * dt, state + 0.5 * dt * k1)
        k3 = derivative(time + 0.5 * dt, state + 0.5 * dt * k2)
        k4 = derivative(time + dt, state + dt * k3)
        next_state = state + (dt / 6.0) * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )
        next_state[6:10] = quaternion_normalize(next_state[6:10])
        return next_state

    times = np.linspace(0.0, duration, number_of_steps + 1)
    states = np.empty((number_of_steps + 1, 14), dtype=float)
    states[0] = initial_state

    for step in range(number_of_steps):
        states[step + 1] = feedback_rk4_step(times[step], states[step])

    return times, states


def simulate_f16_feedback_augmented(
    initial_state,
    initial_controller_state,
    duration,
    dt,
    control_law,
    controller_state_derivative,
    cg_fraction=0.35,
):
    """Simulate coupled F-16 and dynamic-controller states using RK4.

    The callbacks have signatures ``control_law(t, state,
    controller_state)`` and ``controller_state_derivative(t, state,
    controller_state)``. Both are evaluated at every RK4 stage.

    Returns
    -------
    times : numpy.ndarray
        Simulation times with shape ``(number_of_steps + 1,)``.
    states : numpy.ndarray
        Aircraft-state history with shape ``(number_of_steps + 1, 14)``.
    controller_states : numpy.ndarray
        Controller-state history with shape
        ``(number_of_steps + 1, number_of_controller_states)``.
    """
    initial_state = np.asarray(initial_state, dtype=float)
    initial_controller_state = np.asarray(initial_controller_state, dtype=float)
    if initial_state.shape != (14,):
        raise ValueError("initial_state must have shape (14,)")
    if initial_controller_state.ndim != 1:
        raise ValueError("initial_controller_state must be one-dimensional")
    if not np.all(np.isfinite(initial_controller_state)):
        raise ValueError("initial_controller_state must be finite")
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if not callable(control_law):
        raise ValueError("control_law must be callable")
    if not callable(controller_state_derivative):
        raise ValueError("controller_state_derivative must be callable")

    step_ratio = duration / dt
    number_of_steps = round(step_ratio)
    if not np.isclose(step_ratio, number_of_steps, rtol=1e-12, atol=1e-12):
        raise ValueError("duration must be an integer number of time steps")

    controller_shape = initial_controller_state.shape

    def evaluate(time, state, controller_state):
        try:
            controls = np.asarray(
                control_law(time, state, controller_state), dtype=float
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "control_law must return four finite scalar-compatible values"
            ) from error
        if controls.shape != (4,) or not np.all(np.isfinite(controls)):
            raise ValueError(
                "control_law must return four finite scalar-compatible values"
            )

        try:
            controller_dot = np.asarray(
                controller_state_derivative(time, state, controller_state),
                dtype=float,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "controller state derivative must match the controller state shape"
            ) from error
        if controller_dot.shape != controller_shape:
            raise ValueError(
                "controller state derivative must match the controller state shape"
            )
        if not np.all(np.isfinite(controller_dot)):
            raise ValueError("controller state derivative must be finite")

        state_dot = np.asarray(
            f16_state_derivative(
                state,
                *controls,
                cg_fraction=cg_fraction,
            ),
            dtype=float,
        )
        return state_dot, controller_dot

    times = np.linspace(0.0, duration, number_of_steps + 1)
    states = np.empty((number_of_steps + 1, 14), dtype=float)
    controller_states = np.empty(
        (number_of_steps + 1, initial_controller_state.size), dtype=float
    )
    states[0] = initial_state
    controller_states[0] = initial_controller_state

    for step in range(number_of_steps):
        time = times[step]
        state = states[step]
        controller_state = controller_states[step]
        state_k1, controller_k1 = evaluate(time, state, controller_state)
        state_k2, controller_k2 = evaluate(
            time + 0.5 * dt,
            state + 0.5 * dt * state_k1,
            controller_state + 0.5 * dt * controller_k1,
        )
        state_k3, controller_k3 = evaluate(
            time + 0.5 * dt,
            state + 0.5 * dt * state_k2,
            controller_state + 0.5 * dt * controller_k2,
        )
        state_k4, controller_k4 = evaluate(
            time + dt,
            state + dt * state_k3,
            controller_state + dt * controller_k3,
        )

        states[step + 1] = state + (dt / 6.0) * (
            state_k1 + 2.0 * state_k2 + 2.0 * state_k3 + state_k4
        )
        states[step + 1, 6:10] = quaternion_normalize(
            states[step + 1, 6:10]
        )
        controller_states[step + 1] = controller_state + (dt / 6.0) * (
            controller_k1
            + 2.0 * controller_k2
            + 2.0 * controller_k3
            + controller_k4
        )

    return times, states, controller_states
