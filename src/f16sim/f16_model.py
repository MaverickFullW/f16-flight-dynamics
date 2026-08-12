"""High-level coupling of F-16 aerodynamics and rigid-body dynamics."""

import numpy as np

from .air_data import air_data_from_body_velocity
from .aerodynamics import f16_aerodynamic_loads
from .atmosphere import f16_air_data
from .dynamics import state_derivative
from .engine import pdot, tgear, thrust_lbf
from .parameters import J, engine_angular_momentum, mass, reference_cg_fraction


LBF_TO_NEWTON = 4.4482216152605


def f16_state_derivative(
    state,
    throttle,
    elevator_deg,
    aileron_deg,
    rudder_deg,
    cg_fraction=reference_cg_fraction,
    mass_value=mass,
    inertia=J,
    gravity=9.80665,
):
    """
    Compute the 14-state derivative for the aerodynamic F-16 model.

    Parameters
    ----------
    state : array_like
        State vector ``[N, E, D, u, v, w, q0, q1, q2, q3, p, q, r,
        engine_power]``. Position and body velocity are in meters and meters
        per second, the quaternion is scalar-first, body rates are in radians
        per second, and engine power uses the original F-16 power units. Body
        quantities use the forward-right-down convention.
    throttle : float
        Dimensionless throttle position.
    elevator_deg : float
        Elevator deflection in degrees.
    aileron_deg : float
        Aileron deflection in degrees.
    rudder_deg : float
        Rudder deflection in degrees.
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
        Derivative of the 14-state vector in the same state ordering.

    Raises
    ------
    ValueError
        If ``state`` does not have shape ``(14,)``. Lower-level validation
        errors are propagated unchanged.
    """
    state = np.asarray(state, dtype=float)
    if state.shape != (14,):
        raise ValueError("state must have shape (14,)")

    rigid_body_state = state[:13]
    engine_power = state[13]
    velocity_body = rigid_body_state[3:6]
    omega_body = rigid_body_state[10:13]

    true_airspeed, _, _ = air_data_from_body_velocity(velocity_body)
    altitude_m = -rigid_body_state[2]
    air_density, mach, _ = f16_air_data(
        true_airspeed=true_airspeed,
        altitude_m=altitude_m,
    )

    aero_forces_body, moments_body = f16_aerodynamic_loads(
        velocity_body=velocity_body,
        omega_body=omega_body,
        elevator_deg=elevator_deg,
        aileron_deg=aileron_deg,
        rudder_deg=rudder_deg,
        air_density=air_density,
        cg_fraction=cg_fraction,
    )

    commanded_power = tgear(throttle)
    engine_power_dot = pdot(
        actual_power=engine_power,
        commanded_power=commanded_power,
    )
    engine_thrust_lbf = thrust_lbf(
        power=engine_power,
        altitude_ft=altitude_m / 0.3048,
        mach=mach,
    )
    engine_thrust_newtons = engine_thrust_lbf * LBF_TO_NEWTON

    total_forces_body = aero_forces_body.copy()
    total_forces_body[0] += engine_thrust_newtons

    rigid_body_dot = state_derivative(
        state=rigid_body_state,
        forces_body=total_forces_body,
        moments_body=moments_body,
        mass_value=mass_value,
        inertia=inertia,
        gravity=gravity,
        engine_angular_momentum=engine_angular_momentum,
    )

    return np.concatenate((rigid_body_dot, [engine_power_dot]))
