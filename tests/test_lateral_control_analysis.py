import numpy as np
import pytest

from src.f16sim.control_analysis import (
    lateral_bank_angle_feedback_poles,
    lateral_roll_rate_feedback_poles,
)
from src.f16sim.lateral_linearization import linearize_lateral
from src.f16sim.parameters import FT_TO_METER
from src.f16sim.trim import trim_straight_level


@pytest.fixture(scope="module")
def lateral_matrices():
    trim = trim_straight_level(
        true_airspeed=502.0 * FT_TO_METER,
        altitude_m=0.0,
        cg_fraction=0.30,
    )
    assert trim["success"] is True
    return linearize_lateral(
        trim["state"],
        trim["throttle"],
        trim["elevator_deg"],
        cg_fraction=0.30,
    )


def test_zero_roll_rate_gain_reproduces_yaw_damped_system(lateral_matrices):
    A_lat, B_lat = lateral_matrices
    poles = lateral_roll_rate_feedback_poles(A_lat, B_lat, k_p=0.0, Kr=50.0)
    yaw_damped_matrix = A_lat + 50.0 * (
        B_lat[:, 1:2] @ np.array([[0.0, 0.0, 0.0, 1.0]])
    )

    assert poles.shape == (1, 4)
    assert np.allclose(
        np.sort_complex(poles[0]),
        np.sort_complex(np.linalg.eigvals(yaw_damped_matrix)),
    )


def test_zero_bank_gain_reproduces_roll_rate_feedback_system(lateral_matrices):
    A_lat, B_lat = lateral_matrices
    expected = lateral_roll_rate_feedback_poles(A_lat, B_lat, k_p=5.0)
    actual = lateral_bank_angle_feedback_poles(
        A_lat, B_lat, k_phi=0.0, Kp=5.0
    )

    assert actual.shape == (1, 4)
    assert np.allclose(np.sort_complex(actual[0]), np.sort_complex(expected[0]))


def test_lateral_feedback_pole_arrays_are_finite_and_have_expected_shapes(
    lateral_matrices,
):
    A_lat, B_lat = lateral_matrices
    roll_gains = np.array([0.0, 1.0, 5.0, 20.0])
    bank_gains = np.array([0.0, 0.1, 0.5, 2.0])
    roll_poles = lateral_roll_rate_feedback_poles(A_lat, B_lat, roll_gains)
    bank_poles = lateral_bank_angle_feedback_poles(
        A_lat, B_lat, bank_gains, Kp=5.0
    )

    assert roll_poles.shape == (roll_gains.size, 4)
    assert bank_poles.shape == (bank_gains.size, 4)
    assert np.all(np.isfinite(roll_poles))
    assert np.all(np.isfinite(bank_poles))


@pytest.mark.parametrize("invalid_gain", [[], [[1.0]], [0.0, np.nan], np.inf])
def test_roll_rate_feedback_rejects_invalid_gains(lateral_matrices, invalid_gain):
    A_lat, B_lat = lateral_matrices
    with pytest.raises(ValueError):
        lateral_roll_rate_feedback_poles(A_lat, B_lat, invalid_gain)


@pytest.mark.parametrize(
    "invalid_gain", [[], [[0.1]], -0.1, [0.0, np.nan], np.inf]
)
def test_bank_angle_feedback_rejects_invalid_gains(lateral_matrices, invalid_gain):
    A_lat, B_lat = lateral_matrices
    with pytest.raises(ValueError):
        lateral_bank_angle_feedback_poles(A_lat, B_lat, invalid_gain, Kp=5.0)
