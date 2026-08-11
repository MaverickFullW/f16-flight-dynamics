import numpy as np
import pytest

from src.f16sim.attitude import quaternion_rate
from src.f16sim.dynamics import state_derivative
from src.f16sim.integrators import rk4_step


def test_rk4_step_matches_exponential_growth():
    def derivative(state):
        state_dot = np.zeros(13)
        state_dot[0] = state[0]
        return state_dot

    state = np.zeros(13)
    state[0] = 1.0
    state[6] = 1.0

    next_state = rk4_step(derivative, state, dt=0.1)

    assert next_state[0] == pytest.approx(np.exp(0.1), rel=1e-7)


def test_zero_derivative_leaves_valid_state_unchanged():
    def zero_derivative(state):
        return np.zeros_like(state)

    state = np.array([
        10.0, 20.0, 30.0,
        100.0, 5.0, -2.0,
        1.0, 0.0, 0.0, 0.0,
        0.1, -0.2, 0.3,
    ])

    next_state = rk4_step(zero_derivative, state, dt=0.1)

    assert np.allclose(next_state, state)


def test_rk4_step_normalizes_final_quaternion():
    def zero_derivative(state):
        return np.zeros_like(state)

    state = np.zeros(13)
    state[6:10] = [2.0, 1.0, -2.0, 0.5]

    next_state = rk4_step(zero_derivative, state, dt=0.1)

    assert np.isclose(np.linalg.norm(next_state[6:10]), 1.0)


def test_one_step_free_fall():
    gravity = 9.80665
    dt = 0.1
    state = np.zeros(13)
    state[6] = 1.0

    next_state = rk4_step(
        state_derivative,
        state,
        dt,
        forces_body=np.zeros(3),
        moments_body=np.zeros(3),
        gravity=gravity,
    )

    expected = np.array([
        0.0,
        0.0,
        0.5 * gravity * dt**2,
        0.0,
        0.0,
        gravity * dt,
    ])

    assert np.allclose(next_state[:6], expected, rtol=1e-12, atol=1e-12)


def test_constant_body_x_rate_matches_analytical_quaternion():
    def constant_angular_velocity_derivative(state):
        state_dot = np.zeros(13)
        state_dot[6:10] = quaternion_rate(state[6:10], state[10:13])
        return state_dot

    p = 1.0
    dt = 0.01
    state = np.zeros(13)
    state[6] = 1.0
    state[10] = p

    next_state = rk4_step(
        constant_angular_velocity_derivative,
        state,
        dt,
    )
    expected_quaternion = np.array([
        np.cos(p * dt / 2.0),
        np.sin(p * dt / 2.0),
        0.0,
        0.0,
    ])
    actual_quaternion = next_state[6:10]

    if np.dot(actual_quaternion, expected_quaternion) < 0.0:
        actual_quaternion = -actual_quaternion

    assert np.allclose(
        actual_quaternion,
        expected_quaternion,
        rtol=1e-10,
        atol=1e-12,
    )
