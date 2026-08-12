"""High-level coupling of F-16 aerodynamics and rigid-body dynamics."""

import numpy as np

from .aerodynamics import f16_aerodynamic_loads
from .dynamics import state_derivative
from .parameters import J, mass, reference_cg_fraction


def f16_state_derivative(
    state,
    elevator_deg,
    aileron_deg,
    rudder_deg,
    air_density,
    cg_fraction=reference_cg_fraction,
    mass_value=mass,
    inertia=J,
    gravity=9.80665,
):
    """
    Compute the 13-state derivative for the aerodynamic F-16 model.

    Parameters
    ----------
    state : array_like
        State vector ``[N, E, D, u, v, w, q0, q1, q2, q3, p, q, r]``.
        Position and body velocity are in meters and meters per second,
        the quaternion is scalar-first, and body rates are in radians per
        second. Body quantities use the forward-right-down convention.
    elevator_deg : float
        Elevator deflection in degrees.
    aileron_deg : float
        Aileron deflection in degrees.
    rudder_deg : float
        Rudder deflection in degrees.
    air_density : float
        Air density in kilograms per cubic meter.
    cg_fraction : float, optional
        Center-of-gravity position as a fraction of mean aerodynamic chord.
    mass_value : float, optional
        Aircraft mass in kilograms.
    inertia : array_like, optional
        Body-axis inertia tensor in kilogram-square meters.
    gravity : float, optional
        Gravitational acceleration in meters per second squared. Gravity is
        applied only by the generic rigid-body dynamics.

    Returns
    -------
    numpy.ndarray
        Derivative of the 13-state vector in the same state ordering.

    Raises
    ------
    ValueError
        If ``state`` does not have shape ``(13,)``. Lower-level validation
        errors are propagated unchanged.
    """
    state = np.asarray(state, dtype=float)
    if state.shape != (13,):
        raise ValueError("state must have shape (13,)")

    velocity_body = state[3:6]
    omega_body = state[10:13]

    forces_body, moments_body = f16_aerodynamic_loads(
        velocity_body=velocity_body,
        omega_body=omega_body,
        elevator_deg=elevator_deg,
        aileron_deg=aileron_deg,
        rudder_deg=rudder_deg,
        air_density=air_density,
        cg_fraction=cg_fraction,
    )

    return state_derivative(
        state=state,
        forces_body=forces_body,
        moments_body=moments_body,
        mass_value=mass_value,
        inertia=inertia,
        gravity=gravity,
    )
