"""Numerical integration methods for the F-16 state model."""

import numpy as np

from .attitude import quaternion_normalize


def rk4_step(derivative_function, state, dt, *args, **kwargs):
    """
    Advance the 13-state model by one classical fourth-order RK4 step.

    Parameters
    ----------
    derivative_function : callable
        Function with signature ``derivative_function(state, *args,
        **kwargs)`` that returns the 13-element state derivative.
    state : array_like
        Current state ``[N, E, D, u, v, w, q0, q1, q2, q3, p, q, r]``.
    dt : float
        Integration time step in seconds.
    *args : tuple
        Positional arguments passed unchanged to ``derivative_function``.
    **kwargs : dict
        Keyword arguments passed unchanged to ``derivative_function``.

    Returns
    -------
    numpy.ndarray
        State after one time step, with its scalar-first quaternion portion
        normalized.
    """
    state = np.asarray(state, dtype=float)

    k1 = np.asarray(
        derivative_function(state, *args, **kwargs),
        dtype=float,
    )
    k2 = np.asarray(
        derivative_function(state + 0.5 * dt * k1, *args, **kwargs),
        dtype=float,
    )
    k3 = np.asarray(
        derivative_function(state + 0.5 * dt * k2, *args, **kwargs),
        dtype=float,
    )
    k4 = np.asarray(
        derivative_function(state + dt * k3, *args, **kwargs),
        dtype=float,
    )

    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    next_state[6:10] = quaternion_normalize(next_state[6:10])

    return next_state
