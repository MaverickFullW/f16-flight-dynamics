import numpy as np
import pytest

from src.f16sim.aerodynamics import f16_aerodynamic_loads
from src.f16sim.dynamics import state_derivative
from src.f16sim.f16_model import f16_state_derivative
from src.f16sim.parameters import J, mass, reference_cg_fraction


def _state(velocity_body=(150.0, 4.0, 12.0), omega_body=(0.2, -0.1, 0.15)):
    state = np.zeros(13)
    state[3:6] = velocity_body
    state[6] = 1.0
    state[10:13] = omega_body
    return state


def _manual_derivative(
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
    forces_body, moments_body = f16_aerodynamic_loads(
        velocity_body=state[3:6],
        omega_body=state[10:13],
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


@pytest.mark.parametrize("state", [np.zeros(12), np.zeros((13, 1))])
def test_f16_state_derivative_rejects_invalid_state_shape(state):
    with pytest.raises(ValueError):
        f16_state_derivative(state, 0.0, 0.0, 0.0, 1.225)


def test_f16_state_derivative_matches_end_to_end_pipeline():
    state = _state()
    controls = (-4.0, 6.0, -9.0)
    air_density = 1.1
    expected = _manual_derivative(state, *controls, air_density)

    actual = f16_state_derivative(state, *controls, air_density)

    assert np.allclose(actual, expected)


def test_f16_state_derivative_returns_thirteen_element_array():
    result = f16_state_derivative(_state(), 0.0, 0.0, 0.0, 1.225)

    assert result.shape == (13,)


def test_f16_state_derivative_zero_rates_matches_pipeline():
    state = _state(velocity_body=(150.0, 0.0, 10.0), omega_body=(0.0, 0.0, 0.0))
    expected = _manual_derivative(state, 0.0, 0.0, 0.0, 1.225)

    actual = f16_state_derivative(state, 0.0, 0.0, 0.0, 1.225)

    assert np.allclose(actual, expected)


def test_f16_state_derivative_propagates_cg_fraction():
    state = _state()
    cg_fraction = 0.29
    expected = _manual_derivative(
        state, -4.0, 6.0, -9.0, 1.1, cg_fraction=cg_fraction
    )

    actual = f16_state_derivative(
        state, -4.0, 6.0, -9.0, 1.1, cg_fraction=cg_fraction
    )

    assert np.allclose(actual, expected)


def test_f16_state_derivative_propagates_zero_velocity_error():
    with pytest.raises(ValueError):
        f16_state_derivative(_state(velocity_body=(0.0, 0.0, 0.0)), 0.0, 0.0, 0.0, 1.225)


@pytest.mark.parametrize("air_density", [0.0, -1.0])
def test_f16_state_derivative_propagates_invalid_density(air_density):
    with pytest.raises(ValueError):
        f16_state_derivative(_state(), 0.0, 0.0, 0.0, air_density)


def test_f16_state_derivative_propagates_custom_mass():
    state = _state()
    custom_mass = 0.8 * mass
    expected = _manual_derivative(
        state, -4.0, 6.0, -9.0, 1.1, mass_value=custom_mass
    )

    actual = f16_state_derivative(
        state, -4.0, 6.0, -9.0, 1.1, mass_value=custom_mass
    )

    assert np.allclose(actual, expected)


def test_f16_state_derivative_propagates_custom_inertia():
    state = _state()
    custom_inertia = 1.2 * J
    expected = _manual_derivative(
        state, -4.0, 6.0, -9.0, 1.1, inertia=custom_inertia
    )

    actual = f16_state_derivative(
        state, -4.0, 6.0, -9.0, 1.1, inertia=custom_inertia
    )

    assert np.allclose(actual, expected)


def test_f16_state_derivative_propagates_custom_gravity():
    state = _state()
    custom_gravity = 3.711
    expected = _manual_derivative(
        state, -4.0, 6.0, -9.0, 1.1, gravity=custom_gravity
    )

    actual = f16_state_derivative(
        state, -4.0, 6.0, -9.0, 1.1, gravity=custom_gravity
    )

    assert np.allclose(actual, expected)
