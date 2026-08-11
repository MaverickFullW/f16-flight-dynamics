import numpy as np
import pytest

from src.f16sim.dynamics import (
    rotational_dynamics,
    state_derivative,
    translational_dynamics,
    translational_kinematics,
)
from src.f16sim.parameters import J, mass


def test_translational_kinematics_with_identity_quaternion():
    velocity_body = np.array([100.0, 20.0, -5.0])
    quaternion = np.array([1.0, 0.0, 0.0, 0.0])

    position_dot_ned = translational_kinematics(velocity_body, quaternion)

    assert np.allclose(position_dot_ned, velocity_body)


def test_stationary_translational_dynamics_has_downward_gravity():
    gravity = 9.80665

    acceleration = translational_dynamics(
        velocity_body=[0.0, 0.0, 0.0],
        omega_body=[0.0, 0.0, 0.0],
        forces_body=[0.0, 0.0, 0.0],
        quaternion=[1.0, 0.0, 0.0, 0.0],
        gravity=gravity,
    )

    assert np.allclose(acceleration, [0.0, 0.0, gravity])


def test_constant_velocity_without_gravity_or_forces_has_zero_acceleration():
    acceleration = translational_dynamics(
        velocity_body=[100.0, 20.0, -5.0],
        omega_body=[0.0, 0.0, 0.0],
        forces_body=[0.0, 0.0, 0.0],
        quaternion=[1.0, 0.0, 0.0, 0.0],
        gravity=0.0,
    )

    assert np.allclose(acceleration, np.zeros(3))


def test_applied_force_produces_force_over_mass_acceleration():
    forces_body = np.array([1000.0, -2000.0, 500.0])

    acceleration = translational_dynamics(
        velocity_body=[10.0, 5.0, -2.0],
        omega_body=[0.0, 0.0, 0.0],
        forces_body=forces_body,
        quaternion=[1.0, 0.0, 0.0, 0.0],
        gravity=0.0,
    )

    assert acceleration == pytest.approx(forces_body / mass)


def test_rotational_dynamics_from_rest_matches_inertia_solution():
    moments_body = np.array([1000.0, -500.0, 250.0])

    omega_dot = rotational_dynamics(
        omega_body=[0.0, 0.0, 0.0],
        moments_body=moments_body,
    )

    assert np.allclose(omega_dot, np.linalg.solve(J, moments_body))


def test_rotation_about_principal_y_axis_has_zero_angular_acceleration():
    omega_dot = rotational_dynamics(
        omega_body=[0.0, 2.0, 0.0],
        moments_body=[0.0, 0.0, 0.0],
    )

    assert np.allclose(omega_dot, np.zeros(3))


def test_state_derivative_returns_thirteen_element_array():
    state = np.zeros(13)
    state[6] = 1.0

    derivative = state_derivative(
        state,
        forces_body=np.zeros(3),
        moments_body=np.zeros(3),
    )

    assert isinstance(derivative, np.ndarray)
    assert derivative.shape == (13,)


def test_stationary_state_derivative_contains_only_gravity_acceleration():
    gravity = 9.80665
    state = np.zeros(13)
    state[6] = 1.0

    derivative = state_derivative(
        state,
        forces_body=np.zeros(3),
        moments_body=np.zeros(3),
        gravity=gravity,
    )

    expected = np.zeros(13)
    expected[5] = gravity

    assert np.allclose(derivative, expected)
