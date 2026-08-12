import numpy as np
import pytest

from src.f16sim.engine import tgear
from src.f16sim.f16_model import f16_state_derivative
from src.f16sim.integrators import rk4_step
from src.f16sim.simulation import simulate_f16


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
