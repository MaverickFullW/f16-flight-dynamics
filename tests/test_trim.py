import numpy as np
import pytest

from src.f16sim.parameters import FT_TO_METER
from src.f16sim.trim import trim_straight_level


@pytest.fixture(scope="module")
def table_3_6_2_trim():
    return trim_straight_level(
        true_airspeed=500.0 * FT_TO_METER,
        altitude_m=0.0,
    )


def test_trim_straight_level_matches_lewis_table_3_6_2(table_3_6_2_trim):
    result = table_3_6_2_trim

    assert result["throttle"] == pytest.approx(0.137, abs=0.005)
    assert result["alpha_deg"] == pytest.approx(2.14, abs=0.05)
    assert result["elevator_deg"] == pytest.approx(-0.756, abs=0.05)


def test_trim_straight_level_has_small_dynamic_residuals(table_3_6_2_trim):
    result = table_3_6_2_trim

    assert abs(result["VT_dot"]) < 1e-4
    assert abs(result["alpha_dot"]) < 1e-4
    assert abs(result["q_dot"]) < 1e-4
    assert abs(result["engine_power_dot"]) < 1e-12


def test_trim_straight_level_optimizer_succeeds(table_3_6_2_trim):
    result = table_3_6_2_trim

    assert result["success"] is True
    assert result["cost"] < 1e-6


def test_trim_straight_level_returns_full_state_vectors(table_3_6_2_trim):
    result = table_3_6_2_trim

    assert result["state"].shape == (14,)
    assert result["state_dot"].shape == (14,)


def test_trim_straight_level_state_satisfies_constraints(table_3_6_2_trim):
    result = table_3_6_2_trim
    state = result["state"]

    assert state[2] == pytest.approx(-result["altitude_m"])
    assert state[4] == pytest.approx(0.0)
    assert np.allclose(state[10:13], 0.0)


@pytest.mark.parametrize("true_airspeed", [0.0, -1.0])
def test_trim_straight_level_rejects_nonpositive_true_airspeed(true_airspeed):
    with pytest.raises(ValueError, match="true_airspeed must be positive"):
        trim_straight_level(true_airspeed, altitude_m=0.0)


@pytest.mark.parametrize(
    "initial_guess",
    [
        [0.2, 0.0],
        [0.2, 0.0, 3.0, 0.0],
        [[0.2, 0.0, 3.0]],
    ],
)
def test_trim_straight_level_rejects_invalid_initial_guess_shape(initial_guess):
    with pytest.raises(
        ValueError, match="initial_guess must contain exactly three values"
    ):
        trim_straight_level(
            true_airspeed=500.0 * FT_TO_METER,
            altitude_m=0.0,
            initial_guess=initial_guess,
        )
