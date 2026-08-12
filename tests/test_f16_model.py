import numpy as np
import pytest

from src.f16sim.air_data import air_data_from_body_velocity
from src.f16sim.aerodynamics import f16_aerodynamic_loads
from src.f16sim.atmosphere import f16_air_data
from src.f16sim.dynamics import state_derivative
from src.f16sim.engine import pdot, tgear, thrust_lbf
from src.f16sim.f16_model import LBF_TO_NEWTON, f16_state_derivative
from src.f16sim.parameters import (
    J,
    engine_angular_momentum,
    mass,
    reference_cg_fraction,
)


def _state(
    velocity_body=(150.0, 4.0, 12.0),
    omega_body=(0.2, -0.1, 0.15),
    engine_power=40.0,
):
    state = np.zeros(14)
    state[2] = -3000.0
    state[3:6] = velocity_body
    state[6] = 1.0
    state[10:13] = omega_body
    state[13] = engine_power
    return state


def _manual_derivative(
    state,
    throttle,
    elevator_deg,
    aileron_deg,
    rudder_deg,
    cg_fraction=reference_cg_fraction,
    mass_value=mass,
    inertia=J,
    gravity=9.80665,
    engine_momentum=engine_angular_momentum,
):
    rigid_body_state = state[:13]
    velocity_body = rigid_body_state[3:6]
    omega_body = rigid_body_state[10:13]
    engine_power = state[13]
    true_airspeed, _, _ = air_data_from_body_velocity(velocity_body)
    altitude_m = -rigid_body_state[2]
    air_density, mach, _ = f16_air_data(true_airspeed, altitude_m)
    forces_body, moments_body = f16_aerodynamic_loads(
        velocity_body,
        omega_body,
        elevator_deg,
        aileron_deg,
        rudder_deg,
        air_density,
        cg_fraction,
    )
    forces_body = forces_body.copy()
    forces_body[0] += (
        thrust_lbf(engine_power, altitude_m / 0.3048, mach) * LBF_TO_NEWTON
    )
    rigid_body_dot = state_derivative(
        state=rigid_body_state,
        forces_body=forces_body,
        moments_body=moments_body,
        mass_value=mass_value,
        inertia=inertia,
        gravity=gravity,
        engine_angular_momentum=engine_momentum,
    )
    engine_power_dot = pdot(engine_power, tgear(throttle))
    return np.concatenate((rigid_body_dot, [engine_power_dot]))


@pytest.mark.parametrize("state", [np.zeros(13), np.zeros((14, 1))])
def test_f16_state_derivative_rejects_invalid_state_shape(state):
    with pytest.raises(ValueError):
        f16_state_derivative(state, 0.5, 0.0, 0.0, 0.0)


def test_f16_state_derivative_matches_end_to_end_pipeline():
    state = _state()
    controls = (0.8, -4.0, 6.0, -9.0)

    assert np.allclose(
        f16_state_derivative(state, *controls),
        _manual_derivative(state, *controls),
    )


def test_f16_state_derivative_returns_fourteen_element_array():
    assert f16_state_derivative(_state(), 0.5, 0.0, 0.0, 0.0).shape == (14,)


def test_f16_state_derivative_engine_power_derivative_matches_pdot():
    state = _state(engine_power=20.0)
    throttle = 0.8
    result = f16_state_derivative(state, throttle, 0.0, 0.0, 0.0)

    assert np.isclose(result[13], pdot(state[13], tgear(throttle)))


def test_f16_state_derivative_has_steady_engine_power_at_matching_command():
    throttle = 0.5
    state = _state(engine_power=tgear(throttle))

    result = f16_state_derivative(state, throttle, 0.0, 0.0, 0.0)

    assert result[13] == pytest.approx(0.0)


def test_f16_state_derivative_zero_rates_matches_pipeline():
    state = _state((150.0, 0.0, 10.0), (0.0, 0.0, 0.0))
    controls = (0.5, 0.0, 0.0, 0.0)

    assert np.allclose(
        f16_state_derivative(state, *controls),
        _manual_derivative(state, *controls),
    )


def test_f16_state_derivative_propagates_cg_fraction():
    state = _state()
    controls = (0.8, -4.0, 6.0, -9.0)
    cg_fraction = 0.29

    assert np.allclose(
        f16_state_derivative(state, *controls, cg_fraction=cg_fraction),
        _manual_derivative(state, *controls, cg_fraction=cg_fraction),
    )


def test_f16_state_derivative_propagates_zero_velocity_error():
    with pytest.raises(ValueError):
        f16_state_derivative(
            _state(velocity_body=(0.0, 0.0, 0.0)), 0.5, 0.0, 0.0, 0.0
        )


def test_f16_state_derivative_propagates_custom_mass():
    state = _state()
    controls = (0.8, -4.0, 6.0, -9.0)
    custom_mass = 0.8 * mass

    assert np.allclose(
        f16_state_derivative(state, *controls, mass_value=custom_mass),
        _manual_derivative(state, *controls, mass_value=custom_mass),
    )


def test_f16_state_derivative_propagates_custom_inertia():
    state = _state()
    controls = (0.8, -4.0, 6.0, -9.0)
    custom_inertia = 1.2 * J

    assert np.allclose(
        f16_state_derivative(state, *controls, inertia=custom_inertia),
        _manual_derivative(state, *controls, inertia=custom_inertia),
    )


def test_f16_state_derivative_propagates_custom_gravity():
    state = _state()
    controls = (0.8, -4.0, 6.0, -9.0)
    custom_gravity = 3.711

    assert np.allclose(
        f16_state_derivative(state, *controls, gravity=custom_gravity),
        _manual_derivative(state, *controls, gravity=custom_gravity),
    )


def test_f16_state_derivative_propagates_figure_engine_angular_momentum():
    state = _state(omega_body=(0.2, -0.35, 0.4), engine_power=38.0)
    controls = (0.72, -3.0, 4.0, -6.0)
    expected = _manual_derivative(
        state,
        *controls,
        engine_momentum=engine_angular_momentum,
    )

    actual = f16_state_derivative(state, *controls)

    assert np.allclose(actual, expected)


def test_engine_angular_momentum_affects_only_rotational_acceleration():
    state = _state(omega_body=(0.1, 0.3, -0.45), engine_power=42.0)
    controls = (0.65, -2.0, 3.0, -5.0)
    with_engine = f16_state_derivative(state, *controls)
    without_engine = _manual_derivative(
        state,
        *controls,
        engine_momentum=0.0,
    )

    assert np.allclose(with_engine[:10], without_engine[:10])
    assert not np.allclose(with_engine[10:13], without_engine[10:13])
    assert with_engine[13] == without_engine[13]


def test_engine_power_derivative_is_independent_of_engine_angular_momentum():
    throttle = 0.82
    state = _state(omega_body=(0.0, 0.25, 0.4), engine_power=30.0)
    controls = (throttle, -1.0, 2.0, -3.0)
    with_engine = f16_state_derivative(state, *controls)
    without_engine = _manual_derivative(
        state,
        *controls,
        engine_momentum=0.0,
    )
    expected_power_dot = pdot(state[13], tgear(throttle))

    assert with_engine[13] == expected_power_dot
    assert without_engine[13] == expected_power_dot
