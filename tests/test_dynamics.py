import numpy as np
import pytest

from src.f16sim.dynamics import (
    rotational_dynamics,
    state_derivative,
    translational_dynamics,
    translational_kinematics,
)
from src.f16sim.parameters import (
    FT_TO_METER,
    J,
    SLUG_TO_KILOGRAM,
    engine_angular_momentum,
    mass,
)


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


def test_rotational_dynamics_default_matches_original_rigid_body_equation():
    omega = np.array([0.3, -0.2, 0.4])
    moments = np.array([1200.0, -700.0, 350.0])
    expected = np.linalg.solve(J, moments - np.cross(omega, J @ omega))

    assert np.allclose(rotational_dynamics(omega, moments), expected)


def test_rotational_dynamics_explicit_zero_engine_momentum_matches_default():
    omega = np.array([0.3, -0.2, 0.4])
    moments = np.array([1200.0, -700.0, 350.0])

    default = rotational_dynamics(omega, moments)
    explicit_zero = rotational_dynamics(
        omega, moments, engine_angular_momentum=0.0
    )

    assert np.allclose(explicit_zero, default)


def test_rotational_dynamics_includes_pure_engine_gyroscopic_effect():
    omega = np.array([0.0, 0.25, -0.4])
    moments = np.zeros(3)
    momentum = 220.0
    h_engine = np.array([momentum, 0.0, 0.0])
    expected = np.linalg.solve(
        J,
        -np.cross(omega, J @ omega + h_engine),
    )

    actual = rotational_dynamics(
        omega,
        moments,
        engine_angular_momentum=momentum,
    )

    assert np.allclose(actual, expected)


def test_engine_angular_momentum_contribution_is_isolated():
    omega = np.array([0.15, -0.3, 0.45])
    moments = np.array([500.0, 250.0, -100.0])
    momentum = 180.0
    omega_dot_zero = rotational_dynamics(
        omega, moments, engine_angular_momentum=0.0
    )
    omega_dot_engine = rotational_dynamics(
        omega, moments, engine_angular_momentum=momentum
    )
    expected_difference = np.linalg.solve(
        J,
        -np.cross(omega, np.array([momentum, 0.0, 0.0])),
    )

    assert np.allclose(omega_dot_engine - omega_dot_zero, expected_difference)


def test_body_x_rotation_has_no_engine_gyroscopic_contribution():
    omega = np.array([0.7, 0.0, 0.0])
    moments = np.array([100.0, -200.0, 300.0])

    without_engine = rotational_dynamics(
        omega, moments, engine_angular_momentum=0.0
    )
    with_engine = rotational_dynamics(
        omega, moments, engine_angular_momentum=250.0
    )

    assert np.allclose(with_engine, without_engine)


def test_state_derivative_propagates_engine_angular_momentum_only_to_rotation():
    state = np.zeros(13)
    state[3:6] = [120.0, 3.0, 8.0]
    state[6] = 1.0
    state[10:13] = [0.2, -0.3, 0.4]
    forces = np.array([1000.0, -500.0, 750.0])
    moments = np.array([200.0, 300.0, -400.0])
    without_engine = state_derivative(
        state,
        forces,
        moments,
        engine_angular_momentum=0.0,
    )
    with_engine = state_derivative(
        state,
        forces,
        moments,
        engine_angular_momentum=engine_angular_momentum,
    )

    assert np.allclose(with_engine[:10], without_engine[:10])
    assert not np.allclose(with_engine[10:13], without_engine[10:13])
    assert np.allclose(
        with_engine[10:13],
        rotational_dynamics(
            state[10:13],
            moments,
            engine_angular_momentum=engine_angular_momentum,
        ),
    )


def test_engine_angular_momentum_parameter_has_expected_si_conversion():
    expected = 160.0 * SLUG_TO_KILOGRAM * FT_TO_METER**2

    assert np.isclose(engine_angular_momentum, expected)
