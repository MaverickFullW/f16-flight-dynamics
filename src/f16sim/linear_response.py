"""Time response utilities for reduced linear F-16 models."""

import numpy as np


def simulate_linear_longitudinal(
    A,
    B,
    initial_perturbation,
    duration,
    dt,
    control_perturbation=None,
):
    """Simulate a reduced longitudinal linear model using RK4.

    The state and input are perturbations about the trim condition associated
    with ``A`` and ``B``. The longitudinal state has four elements and the
    control perturbation has two elements.

    Parameters
    ----------
    A : array_like
        State matrix with shape ``(4, 4)``.
    B : array_like
        Control matrix with shape ``(4, 2)``.
    initial_perturbation : array_like
        Initial longitudinal state perturbation with shape ``(4,)``.
    duration : float
        Simulation duration in seconds.
    dt : float
        Fixed integration time step in seconds.
    control_perturbation : array_like or callable, optional
        Constant control perturbation with shape ``(2,)``, or a function of
        time returning an array with that shape. Defaults to zero.

    Returns
    -------
    times : numpy.ndarray
        Simulation times with shape ``(number_of_steps + 1,)``.
    perturbations : numpy.ndarray
        State perturbation history with shape ``(number_of_steps + 1, 4)``.

    Raises
    ------
    ValueError
        If an array has the wrong shape, a time value is not positive, or the
        duration is not an integer number of time steps within floating-point
        tolerance.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    initial_perturbation = np.asarray(initial_perturbation, dtype=float)

    if A.shape != (4, 4):
        raise ValueError("A must have shape (4, 4)")
    if B.shape != (4, 2):
        raise ValueError("B must have shape (4, 2)")
    if initial_perturbation.shape != (4,):
        raise ValueError("initial_perturbation must have shape (4,)")

    if callable(control_perturbation):
        control_function = control_perturbation

        def control_at(time):
            control = np.asarray(control_function(time), dtype=float)
            if control.shape != (2,):
                raise ValueError("control_perturbation must return shape (2,)")
            return control

    else:
        if control_perturbation is None:
            constant_control = np.zeros(2, dtype=float)
        else:
            constant_control = np.asarray(control_perturbation, dtype=float)
        if constant_control.shape != (2,):
            raise ValueError("control_perturbation must have shape (2,)")

        def control_at(time):
            return constant_control

    if duration <= 0.0:
        raise ValueError("duration must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    step_ratio = duration / dt
    number_of_steps = round(step_ratio)
    if not np.isclose(step_ratio, number_of_steps, rtol=1e-12, atol=1e-12):
        raise ValueError("duration must be an integer number of time steps")

    def derivative(time, perturbation):
        return A @ perturbation + B @ control_at(time)

    def rk4_step(time, perturbation):
        half_time = time + 0.5 * dt
        k1 = derivative(time, perturbation)
        k2 = derivative(half_time, perturbation + 0.5 * dt * k1)
        k3 = derivative(half_time, perturbation + 0.5 * dt * k2)
        k4 = derivative(time + dt, perturbation + dt * k3)
        return perturbation + (dt / 6.0) * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )

    times = np.linspace(0.0, duration, number_of_steps + 1)
    perturbations = np.empty((number_of_steps + 1, 4), dtype=float)
    perturbations[0] = initial_perturbation

    for step in range(number_of_steps):
        perturbations[step + 1] = rk4_step(times[step], perturbations[step])

    return times, perturbations
