"""Rigid-body six-degree-of-freedom equations of motion."""

import numpy as np

from .attitude import quaternion_rate, quaternion_to_dcm
from .parameters import J, mass
from .vector_ops import skew


def translational_kinematics(velocity_body, quaternion):
    """
    Compute the NED position derivative from body-frame velocity.

    Parameters
    ----------
    velocity_body : array_like
        Body-frame FRD velocity ``[u, v, w]`` in meters per second.
    quaternion : array_like
        Scalar-first quaternion ``[q0, q1, q2, q3]`` describing the
        NED-to-body attitude.

    Returns
    -------
    numpy.ndarray
        NED position derivative ``[N_dot, E_dot, D_dot]`` in meters per
        second.
    """
    velocity_body = np.asarray(velocity_body, dtype=float)
    c_body_to_ned = quaternion_to_dcm(quaternion).T

    return c_body_to_ned @ velocity_body


def translational_dynamics(
    velocity_body,
    omega_body,
    forces_body,
    quaternion,
    mass_value=mass,
    gravity=9.80665,
):
    """
    Compute the body-frame translational acceleration.

    Parameters
    ----------
    velocity_body : array_like
        Body-frame FRD velocity ``[u, v, w]`` in meters per second.
    omega_body : array_like
        Body angular velocity ``[p, q, r]`` in radians per second.
    forces_body : array_like
        Applied body-frame force ``[Fx, Fy, Fz]`` in newtons, excluding
        gravity.
    quaternion : array_like
        Scalar-first quaternion describing the NED-to-body attitude.
    mass_value : float, optional
        Aircraft mass in kilograms.
    gravity : float, optional
        Gravitational acceleration in meters per second squared.

    Returns
    -------
    numpy.ndarray
        Body-frame velocity derivative ``[u_dot, v_dot, w_dot]`` in meters
        per second squared.
    """
    velocity_body = np.asarray(velocity_body, dtype=float)
    omega_body = np.asarray(omega_body, dtype=float)
    forces_body = np.asarray(forces_body, dtype=float)

    gravity_ned = np.array([0.0, 0.0, gravity])
    gravity_body = quaternion_to_dcm(quaternion) @ gravity_ned

    return (
        forces_body / mass_value
        + gravity_body
        - skew(omega_body) @ velocity_body
    )


def rotational_dynamics(
    omega_body,
    moments_body,
    inertia=J,
    engine_angular_momentum=0.0,
):
    """
    Compute body angular acceleration from applied moments.

    Parameters
    ----------
    omega_body : array_like
        Body angular velocity ``[p, q, r]`` in radians per second.
    moments_body : array_like
        Applied body-frame moments ``[L, M, N]`` in newton-meters.
    inertia : array_like, optional
        3x3 body-axis inertia matrix in kilograms meter squared.
    engine_angular_momentum : float, optional
        Constant engine rotor angular momentum along the positive body x
        axis in kilogram meter squared per second.

    Returns
    -------
    numpy.ndarray
        Body angular acceleration ``[p_dot, q_dot, r_dot]`` in radians per
        second squared.
    """
    omega_body = np.asarray(omega_body, dtype=float)
    moments_body = np.asarray(moments_body, dtype=float)
    inertia = np.asarray(inertia, dtype=float)

    h_engine = np.array([engine_angular_momentum, 0.0, 0.0])
    angular_momentum = inertia @ omega_body + h_engine
    gyroscopic_moment = skew(omega_body) @ angular_momentum

    return np.linalg.solve(inertia, moments_body - gyroscopic_moment)


def state_derivative(
    state,
    forces_body,
    moments_body,
    mass_value=mass,
    inertia=J,
    gravity=9.80665,
    engine_angular_momentum=0.0,
):
    """
    Compute the derivative of the 13-state rigid-body state vector.

    Parameters
    ----------
    state : array_like
        State vector ``[N, E, D, u, v, w, q0, q1, q2, q3, p, q, r]``.
        Positions are in meters, velocities in meters per second, and angular
        rates in radians per second.
    forces_body : array_like
        Applied body-frame force ``[Fx, Fy, Fz]`` in newtons, excluding
        gravity.
    moments_body : array_like
        Applied body-frame moments ``[L, M, N]`` in newton-meters.
    mass_value : float, optional
        Aircraft mass in kilograms.
    inertia : array_like, optional
        3x3 body-axis inertia matrix in kilograms meter squared.
    gravity : float, optional
        Gravitational acceleration in meters per second squared.
    engine_angular_momentum : float, optional
        Constant engine rotor angular momentum along the positive body x
        axis in kilogram meter squared per second. The default of zero
        preserves the rigid-body model without internal rotor momentum.

    Returns
    -------
    numpy.ndarray
        State derivative in the same 13-element order as ``state``.
    """
    state = np.asarray(state, dtype=float)
    velocity_body = state[3:6]
    quaternion = state[6:10]
    omega_body = state[10:13]

    position_dot_ned = translational_kinematics(velocity_body, quaternion)
    velocity_dot_body = translational_dynamics(
        velocity_body,
        omega_body,
        forces_body,
        quaternion,
        mass_value=mass_value,
        gravity=gravity,
    )
    quaternion_dot = quaternion_rate(quaternion, omega_body)
    omega_dot_body = rotational_dynamics(
        omega_body,
        moments_body,
        inertia=inertia,
        engine_angular_momentum=engine_angular_momentum,
    )

    return np.concatenate((
        position_dot_ned,
        velocity_dot_body,
        quaternion_dot,
        omega_dot_body,
    ))
