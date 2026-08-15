import numpy as np
import pytest
from scipy.optimize import OptimizeResult

import src.f16sim.trim as trim_module
from src.f16sim.parameters import FT_TO_METER
from src.f16sim.trim import trim_straight_level


@pytest.fixture(scope="module")
def table_3_6_2_trim():
    return trim_straight_level(
        true_airspeed=500.0 * FT_TO_METER,
        altitude_m=0.0,
    )


@pytest.mark.parametrize(
    ("true_airspeed_ft_s", "throttle", "alpha_deg", "elevator_deg"),
    [
        (440.0, 0.113, 3.19, -0.671),
        (500.0, 0.137, 2.14, -0.756),
        (600.0, 0.200, 1.04, -0.846),
        (700.0, 0.282, 0.382, -0.900),
        (800.0, 0.378, -0.045, -0.943),
    ],
)
def test_trim_straight_level_matches_lewis_table_3_6_2(
    true_airspeed_ft_s,
    throttle,
    alpha_deg,
    elevator_deg,
):
    result = trim_straight_level(
        true_airspeed=true_airspeed_ft_s * FT_TO_METER,
        altitude_m=0.0,
    )

    assert result["success"] is True
    assert result["throttle"] == pytest.approx(throttle, abs=0.005)
    assert result["alpha_deg"] == pytest.approx(alpha_deg, abs=0.02)
    assert result["elevator_deg"] == pytest.approx(elevator_deg, abs=0.005)
    assert abs(result["VT_dot"]) < 1e-4
    assert abs(result["alpha_dot"]) < 1e-4
    assert abs(result["q_dot"]) < 1e-4
    assert abs(result["engine_power_dot"]) < 1e-12


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


def test_trim_straight_level_recovers_lewis_table_3_6_3_case():
    result = trim_straight_level(
        true_airspeed=502.0 * FT_TO_METER,
        altitude_m=0.0,
        cg_fraction=0.30,
    )

    assert result["success"] is True
    assert result["throttle"] == pytest.approx(0.1485, abs=0.002)
    assert result["elevator_deg"] == pytest.approx(-1.931, abs=0.02)
    assert result["alpha_deg"] == pytest.approx(2.255, abs=0.02)
    assert abs(result["VT_dot"]) < 1e-4
    assert abs(result["alpha_dot"]) < 1e-4
    assert abs(result["q_dot"]) < 1e-4
    assert abs(result["engine_power_dot"]) < 1e-12


def test_optimizer_success_with_poor_residuals_is_not_valid_trim(monkeypatch):
    def poor_minimize(*args, **kwargs):
        return OptimizeResult(
            x=np.array([0.0, 0.0, 20.0]),
            success=True,
            message="Optimization terminated successfully.",
            nit=1,
        )

    monkeypatch.setattr(trim_module, "minimize", poor_minimize)
    result = trim_straight_level(
        true_airspeed=500.0 * FT_TO_METER,
        altitude_m=0.0,
    )

    assert result["success"] is False
    assert "physical trim residual validation failed" in result["message"]
    assert any(
        abs(result[name]) >= 1e-4
        for name in ("VT_dot", "alpha_dot", "q_dot")
    )


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
