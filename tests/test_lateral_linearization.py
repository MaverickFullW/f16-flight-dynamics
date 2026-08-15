import numpy as np
import pytest

import src.f16sim.lateral_linearization as lateral_linearization
from src.f16sim.lateral_linearization import (
    lateral_state_derivative,
    linearize_lateral,
)
from src.f16sim.parameters import FT_TO_METER
from src.f16sim.trim import trim_straight_level


@pytest.fixture(scope="module")
def lewis_lateral_condition():
    trim = trim_straight_level(
        true_airspeed=502.0 * FT_TO_METER,
        altitude_m=0.0,
        cg_fraction=0.30,
    )
    assert trim["success"] is True
    return trim


def _linearize(trim):
    return linearize_lateral(
        trim["state"],
        throttle=trim["throttle"],
        elevator_deg=trim["elevator_deg"],
        cg_fraction=trim["cg_fraction"],
    )


def test_lateral_linearization_returns_finite_matrices(lewis_lateral_condition):
    A_lat, B_lat = _linearize(lewis_lateral_condition)

    assert A_lat.shape == (4, 4)
    assert B_lat.shape == (4, 2)
    assert np.all(np.isfinite(A_lat))
    assert np.all(np.isfinite(B_lat))


def test_lateral_linearization_is_deterministic(lewis_lateral_condition):
    A_first, B_first = _linearize(lewis_lateral_condition)
    A_second, B_second = _linearize(lewis_lateral_condition)

    assert np.array_equal(A_first, A_second)
    assert np.array_equal(B_first, B_second)


def test_lateral_linearization_is_insensitive_to_smaller_steps(
    lewis_lateral_condition, monkeypatch
):
    A_default, B_default = _linearize(lewis_lateral_condition)
    monkeypatch.setattr(
        lateral_linearization,
        "_LATERAL_STATE_STEPS",
        0.5 * lateral_linearization._LATERAL_STATE_STEPS,
    )
    monkeypatch.setattr(
        lateral_linearization,
        "_LATERAL_CONTROL_STEPS",
        0.5 * lateral_linearization._LATERAL_CONTROL_STEPS,
    )
    A_refined, B_refined = _linearize(lewis_lateral_condition)

    assert np.allclose(A_default, A_refined, rtol=5e-3, atol=1e-6)
    assert np.allclose(B_default, B_refined, rtol=5e-3, atol=1e-6)


def test_linear_model_matches_small_nonlinear_lateral_perturbation(
    lewis_lateral_condition,
):
    trim = lewis_lateral_condition
    A_lat, B_lat = _linearize(trim)
    x_lat = np.array([1e-5, -2e-5, 3e-5, -1e-5])
    u_lat = np.array([1e-3, -1e-3])
    nonlinear_derivative = lateral_state_derivative(
        x_lat,
        u_lat,
        trim["state"],
        trim["throttle"],
        trim["elevator_deg"],
        cg_fraction=trim["cg_fraction"],
    )

    assert np.allclose(
        nonlinear_derivative,
        A_lat @ x_lat + B_lat @ u_lat,
        rtol=2e-2,
        atol=1e-8,
    )


def test_zero_lateral_perturbation_has_zero_derivative(lewis_lateral_condition):
    trim = lewis_lateral_condition
    derivative = lateral_state_derivative(
        np.zeros(4),
        np.zeros(2),
        trim["state"],
        trim["throttle"],
        trim["elevator_deg"],
        cg_fraction=trim["cg_fraction"],
    )

    assert np.allclose(derivative, 0.0, rtol=0.0, atol=1e-10)


def test_lateral_kinematic_and_gravity_signs_are_consistent(
    lewis_lateral_condition,
):
    A_lat, _ = _linearize(lewis_lateral_condition)

    assert A_lat[1, 2] == pytest.approx(1.0, abs=1e-8)
    assert A_lat[0, 1] > 0.0


@pytest.mark.parametrize(
    ("x_lat", "u_lat", "trim_state", "message"),
    [
        (np.zeros(3), np.zeros(2), np.zeros(14), r"x_lat must have shape \(4,\)"),
        (np.zeros(4), np.zeros(3), np.zeros(14), r"u_lat must have shape \(2,\)"),
        (
            np.zeros(4),
            np.zeros(2),
            np.zeros(13),
            r"trim_state must have shape \(14,\)",
        ),
    ],
)
def test_lateral_state_derivative_rejects_invalid_shapes(
    x_lat, u_lat, trim_state, message
):
    with pytest.raises(ValueError, match=message):
        lateral_state_derivative(x_lat, u_lat, trim_state, 0.2, 0.0)


def test_linearize_lateral_rejects_invalid_trim_state_shape():
    with pytest.raises(ValueError, match=r"trim_state must have shape \(14,\)"):
        linearize_lateral(np.zeros(13), throttle=0.2, elevator_deg=0.0)
