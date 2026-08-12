import numpy as np
import pytest

from src.f16sim.air_data import air_data_from_body_velocity


def test_pure_forward_velocity():
    true_airspeed, alpha_deg, beta_deg = air_data_from_body_velocity(
        [100.0, 0.0, 0.0]
    )

    assert np.isclose(true_airspeed, 100.0)
    assert np.isclose(alpha_deg, 0.0)
    assert np.isclose(beta_deg, 0.0)


@pytest.mark.parametrize("alpha_deg", [10.0, -10.0])
def test_recovers_signed_angle_of_attack(alpha_deg):
    true_airspeed = 100.0
    alpha_rad = np.radians(alpha_deg)
    velocity_body = [
        true_airspeed * np.cos(alpha_rad),
        0.0,
        true_airspeed * np.sin(alpha_rad),
    ]

    actual_airspeed, actual_alpha, actual_beta = air_data_from_body_velocity(
        velocity_body
    )

    assert np.isclose(actual_airspeed, true_airspeed)
    assert np.isclose(actual_alpha, alpha_deg)
    assert np.isclose(actual_beta, 0.0)


@pytest.mark.parametrize("beta_deg", [15.0, -15.0])
def test_recovers_signed_sideslip(beta_deg):
    true_airspeed = 100.0
    beta_rad = np.radians(beta_deg)
    velocity_body = [
        true_airspeed * np.cos(beta_rad),
        true_airspeed * np.sin(beta_rad),
        0.0,
    ]

    actual_airspeed, actual_alpha, actual_beta = air_data_from_body_velocity(
        velocity_body
    )

    assert np.isclose(actual_airspeed, true_airspeed)
    assert np.isclose(actual_alpha, 0.0)
    assert np.isclose(actual_beta, beta_deg)


def test_recovers_combined_angle_of_attack_and_sideslip():
    true_airspeed = 175.0
    alpha_deg = 12.0
    beta_deg = -8.0
    alpha_rad = np.radians(alpha_deg)
    beta_rad = np.radians(beta_deg)
    velocity_body = [
        true_airspeed * np.cos(alpha_rad) * np.cos(beta_rad),
        true_airspeed * np.sin(beta_rad),
        true_airspeed * np.sin(alpha_rad) * np.cos(beta_rad),
    ]

    actual = air_data_from_body_velocity(velocity_body)

    assert np.isclose(actual[0], true_airspeed, rtol=1e-12, atol=1e-12)
    assert np.isclose(actual[1], alpha_deg, rtol=1e-12, atol=1e-12)
    assert np.isclose(actual[2], beta_deg, rtol=1e-12, atol=1e-12)


def test_zero_velocity_raises_value_error():
    with pytest.raises(ValueError):
        air_data_from_body_velocity([0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "velocity_body",
    [[1.0, 2.0], [1.0, 2.0, 3.0, 4.0], np.zeros((3, 1))],
)
def test_invalid_input_shape_raises_value_error(velocity_body):
    with pytest.raises(ValueError):
        air_data_from_body_velocity(velocity_body)


def test_returns_python_floats():
    result = air_data_from_body_velocity([100.0, 0.0, 0.0])

    assert all(isinstance(value, float) for value in result)
