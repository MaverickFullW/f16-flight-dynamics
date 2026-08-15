import numpy as np
import pytest

from src.f16sim.engine import tgear
from src.f16sim.f16_model import f16_state_derivative
from src.f16sim.integrators import rk4_step
from src.f16sim.simulation import (
    simulate_f16,
    simulate_f16_feedback,
    simulate_f16_feedback_augmented,
)


def _valid_initial_state(engine_power=30.0):
    state = np.zeros(14)
    state[2] = -3000.0
    state[3] = 150.0
    state[6] = 1.0
    state[13] = engine_power
    return state


def _simulate(initial_state, duration=0.1, dt=0.01, cg_fraction=0.35, throttle=0.5):
    return simulate_f16(
        initial_state=initial_state,
        duration=duration,
        dt=dt,
        throttle=throttle,
        elevator_deg=-2.0,
        aileron_deg=0.5,
        rudder_deg=-0.5,
        cg_fraction=cg_fraction,
    )


@pytest.mark.parametrize("initial_state", [np.zeros(13), np.zeros((14, 1))])
def test_simulate_f16_rejects_invalid_initial_state_shape(initial_state):
    with pytest.raises(ValueError):
        _simulate(initial_state)


@pytest.mark.parametrize("duration", [0.0, -1.0])
def test_simulate_f16_rejects_nonpositive_duration(duration):
    with pytest.raises(ValueError):
        _simulate(_valid_initial_state(), duration=duration)


@pytest.mark.parametrize("dt", [0.0, -0.01])
def test_simulate_f16_rejects_nonpositive_time_step(dt):
    with pytest.raises(ValueError):
        _simulate(_valid_initial_state(), dt=dt)


def test_simulate_f16_rejects_noninteger_number_of_steps():
    with pytest.raises(ValueError):
        _simulate(_valid_initial_state(), duration=0.1, dt=0.03)


def test_simulate_f16_returns_expected_history_shapes():
    times, states = _simulate(_valid_initial_state())

    assert times.shape == (11,)
    assert states.shape == (11, 14)


def test_simulate_f16_time_history_has_requested_endpoints():
    times, _ = _simulate(_valid_initial_state(), duration=0.1)

    assert times[0] == 0.0
    assert times[-1] == 0.1


def test_simulate_f16_stores_supplied_initial_state_first():
    initial_state = _valid_initial_state()
    initial_state[:3] = [10.0, 20.0, -3000.0]

    _, states = _simulate(initial_state)

    assert np.array_equal(states[0], initial_state)


def test_short_f16_simulation_returns_only_finite_states():
    _, states = _simulate(_valid_initial_state())

    assert np.all(np.isfinite(states))


def test_short_f16_simulation_keeps_all_quaternions_normalized():
    _, states = _simulate(_valid_initial_state())

    assert np.allclose(np.linalg.norm(states[:, 6:10], axis=1), 1.0)


def test_simulate_f16_matches_repeated_manual_rk4_steps():
    initial_state = _valid_initial_state()
    dt = 0.01
    _, actual_states = _simulate(initial_state, duration=0.03, dt=dt)
    expected_states = np.empty_like(actual_states)
    expected_states[0] = initial_state

    for step in range(3):
        expected_states[step + 1] = rk4_step(
            f16_state_derivative,
            expected_states[step],
            dt,
            0.5,
            -2.0,
            0.5,
            -0.5,
            cg_fraction=0.35,
        )

    assert np.allclose(actual_states, expected_states)


def test_simulate_f16_propagates_nonreference_cg_fraction():
    initial_state = _valid_initial_state()
    dt = 0.01
    cg_fraction = 0.29
    _, states = _simulate(initial_state, duration=dt, dt=dt, cg_fraction=cg_fraction)
    expected = rk4_step(
        f16_state_derivative,
        initial_state,
        dt,
        0.5,
        -2.0,
        0.5,
        -0.5,
        cg_fraction=cg_fraction,
    )

    assert np.allclose(states[1], expected)


def test_simulate_f16_engine_power_evolves_toward_command():
    initial_state = _valid_initial_state(engine_power=10.0)
    _, states = _simulate(initial_state, duration=0.02, throttle=0.5)

    assert not np.isclose(states[-1, 13], initial_state[13])


def test_simulate_f16_steady_engine_power_remains_constant():
    throttle = 0.5
    steady_power = tgear(throttle)
    initial_state = _valid_initial_state(engine_power=steady_power)
    _, states = _simulate(initial_state, duration=0.02, throttle=throttle)

    assert np.allclose(states[:, 13], steady_power)


def test_constant_control_behavior_remains_unchanged():
    initial_state = _valid_initial_state()
    _, actual_states = _simulate(initial_state, duration=0.03, dt=0.01)
    expected_states = np.empty_like(actual_states)
    expected_states[0] = initial_state

    for step in range(3):
        expected_states[step + 1] = rk4_step(
            f16_state_derivative,
            expected_states[step],
            0.01,
            0.5,
            -2.0,
            0.5,
            -0.5,
            cg_fraction=0.35,
        )

    assert np.array_equal(actual_states, expected_states)


def test_zero_control_callables_match_zero_control_constants():
    initial_state = _valid_initial_state()
    arguments = {
        "initial_state": initial_state,
        "duration": 0.03,
        "dt": 0.01,
    }
    constant_times, constant_states = simulate_f16(
        **arguments,
        throttle=0.0,
        elevator_deg=0.0,
        aileron_deg=0.0,
        rudder_deg=0.0,
    )
    callable_times, callable_states = simulate_f16(
        **arguments,
        throttle=lambda time: 0.0,
        elevator_deg=lambda time: 0.0,
        aileron_deg=lambda time: 0.0,
        rudder_deg=lambda time: 0.0,
    )

    assert np.array_equal(constant_times, callable_times)
    assert np.array_equal(constant_states, callable_states)


def test_time_varying_elevator_returns_finite_state_history():
    times, states = simulate_f16(
        initial_state=_valid_initial_state(),
        duration=0.1,
        dt=0.01,
        throttle=0.5,
        elevator_deg=lambda time: -2.0 + 0.5 * np.sin(2.0 * np.pi * time),
        aileron_deg=0.0,
        rudder_deg=0.0,
    )

    assert times.shape == (11,)
    assert states.shape == (11, 14)
    assert np.all(np.isfinite(states))


@pytest.mark.parametrize(
    "control_name",
    ["throttle", "elevator_deg", "aileron_deg", "rudder_deg"],
)
def test_control_callable_rejects_nonscalar_output(control_name):
    controls = {
        "throttle": 0.5,
        "elevator_deg": -2.0,
        "aileron_deg": 0.0,
        "rudder_deg": 0.0,
    }
    controls[control_name] = lambda time: np.zeros(2)

    with pytest.raises(ValueError, match=f"{control_name} callable must return a scalar"):
        simulate_f16(
            initial_state=_valid_initial_state(),
            duration=0.01,
            dt=0.01,
            **controls,
        )


def test_constant_feedback_law_reproduces_constant_control_simulation():
    initial_state = _valid_initial_state()
    controls = (0.5, -2.0, 0.5, -0.5)
    expected_times, expected_states = simulate_f16(
        initial_state,
        duration=0.03,
        dt=0.01,
        throttle=controls[0],
        elevator_deg=controls[1],
        aileron_deg=controls[2],
        rudder_deg=controls[3],
        cg_fraction=0.30,
    )
    actual_times, actual_states = simulate_f16_feedback(
        initial_state,
        duration=0.03,
        dt=0.01,
        control_law=lambda time, state: controls,
        cg_fraction=0.30,
    )

    assert np.array_equal(actual_times, expected_times)
    assert np.array_equal(actual_states, expected_states)


def test_simulate_f16_feedback_returns_expected_shapes():
    times, states = simulate_f16_feedback(
        _valid_initial_state(),
        duration=0.1,
        dt=0.01,
        control_law=lambda time, state: [0.5, -2.0, 0.0, 0.0],
    )

    assert times.shape == (11,)
    assert states.shape == (11, 14)


def test_state_dependent_control_law_returns_finite_states():
    _, states = simulate_f16_feedback(
        _valid_initial_state(),
        duration=0.1,
        dt=0.01,
        control_law=lambda time, state: [
            0.5,
            -2.0 + 5.0 * state[11],
            -0.1 * state[10],
            -0.1 * state[12],
        ],
    )

    assert np.all(np.isfinite(states))


@pytest.mark.parametrize(
    "invalid_controls",
    [
        [0.5, -2.0, 0.0],
        [0.5, -2.0, 0.0, 0.0, 0.0],
        [[0.5, -2.0, 0.0, 0.0]],
    ],
)
def test_simulate_f16_feedback_rejects_invalid_control_shape(invalid_controls):
    with pytest.raises(ValueError, match="four finite scalar-compatible"):
        simulate_f16_feedback(
            _valid_initial_state(),
            duration=0.01,
            dt=0.01,
            control_law=lambda time, state: invalid_controls,
        )


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_simulate_f16_feedback_rejects_nonfinite_controls(invalid_value):
    with pytest.raises(ValueError, match="four finite scalar-compatible"):
        simulate_f16_feedback(
            _valid_initial_state(),
            duration=0.01,
            dt=0.01,
            control_law=lambda time, state: [0.5, invalid_value, 0.0, 0.0],
        )


def test_zero_controller_dynamics_reproduce_feedback_simulator():
    initial_state = _valid_initial_state()
    feedback_law = lambda time, state: [0.5, -2.0 + 5.0 * state[11], 0.0, 0.0]
    expected_times, expected_states = simulate_f16_feedback(
        initial_state,
        duration=0.03,
        dt=0.01,
        control_law=feedback_law,
        cg_fraction=0.30,
    )
    actual_times, actual_states, controller_states = (
        simulate_f16_feedback_augmented(
            initial_state,
            initial_controller_state=np.array([2.0]),
            duration=0.03,
            dt=0.01,
            control_law=lambda time, state, controller_state: feedback_law(
                time, state
            ),
            controller_state_derivative=lambda time, state, controller_state: np.zeros(1),
            cg_fraction=0.30,
        )
    )

    assert np.array_equal(actual_times, expected_times)
    assert np.array_equal(actual_states, expected_states)
    assert np.array_equal(controller_states, np.full((4, 1), 2.0))


def test_controller_state_is_integrated_at_rk4_intermediate_states():
    dt = 0.1
    _, _, controller_states = simulate_f16_feedback_augmented(
        _valid_initial_state(),
        initial_controller_state=np.array([1.0]),
        duration=dt,
        dt=dt,
        control_law=lambda time, state, controller_state: [0.5, -2.0, 0.0, 0.0],
        controller_state_derivative=lambda time, state, controller_state: controller_state,
    )

    expected_rk4_growth = 1.0 + dt + dt**2 / 2.0 + dt**3 / 6.0 + dt**4 / 24.0
    assert controller_states[-1, 0] == pytest.approx(expected_rk4_growth)


def test_constant_controller_state_derivative_integrates_exactly():
    times, _, controller_states = simulate_f16_feedback_augmented(
        _valid_initial_state(),
        initial_controller_state=np.array([1.0, -2.0]),
        duration=0.1,
        dt=0.01,
        control_law=lambda time, state, controller_state: [0.5, -2.0, 0.0, 0.0],
        controller_state_derivative=lambda time, state, controller_state: [
            2.0,
            -3.0,
        ],
    )

    expected = np.column_stack((1.0 + 2.0 * times, -2.0 - 3.0 * times))
    assert np.allclose(controller_states, expected, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize(
    "invalid_derivative",
    [lambda: np.zeros(2), lambda: np.array([np.nan]), lambda: np.array([np.inf])],
)
def test_augmented_feedback_rejects_invalid_controller_derivative(
    invalid_derivative,
):
    with pytest.raises(ValueError, match="controller state derivative"):
        simulate_f16_feedback_augmented(
            _valid_initial_state(),
            initial_controller_state=np.zeros(1),
            duration=0.01,
            dt=0.01,
            control_law=lambda time, state, controller_state: [
                0.5,
                -2.0,
                0.0,
                0.0,
            ],
            controller_state_derivative=lambda time, state, controller_state: invalid_derivative(),
        )
