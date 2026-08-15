import numpy as np
import pytest

import src.f16sim.linearization as linearization
from src.f16sim.linearization import (
    linearize_longitudinal,
    longitudinal_state_derivative,
)
from src.f16sim.parameters import FT_TO_METER
from src.f16sim.trim import trim_straight_level


@pytest.fixture(scope="module")
def equilibrium_500_ft_s():
    trim = trim_straight_level(
        true_airspeed=500.0 * FT_TO_METER,
        altitude_m=0.0,
    )
    alpha = np.deg2rad(trim["alpha_deg"])
    x_equilibrium = np.array([trim["true_airspeed"], alpha, alpha, 0.0])
    u_equilibrium = np.array([trim["throttle"], trim["elevator_deg"]])
    return trim, x_equilibrium, u_equilibrium


def test_longitudinal_state_derivative_has_expected_shape():
    derivative = longitudinal_state_derivative(
        x_lon=np.array([500.0 * FT_TO_METER, np.deg2rad(2.0), 0.0, 0.0]),
        u_lon=np.array([0.2, 0.0]),
        altitude_m=0.0,
    )

    assert derivative.shape == (4,)


@pytest.mark.parametrize(
    "x_lon",
    [
        [500.0 * FT_TO_METER, 0.0, 0.0],
        [500.0 * FT_TO_METER, 0.0, 0.0, 0.0, 0.0],
        [[500.0 * FT_TO_METER, 0.0, 0.0, 0.0]],
    ],
)
def test_longitudinal_state_derivative_rejects_invalid_state_shape(x_lon):
    with pytest.raises(ValueError, match=r"x_lon must have shape \(4,\)"):
        longitudinal_state_derivative(x_lon, [0.2, 0.0], altitude_m=0.0)


@pytest.mark.parametrize(
    "u_lon",
    [
        [0.2],
        [0.2, 0.0, 0.0],
        [[0.2, 0.0]],
    ],
)
def test_longitudinal_state_derivative_rejects_invalid_control_shape(u_lon):
    with pytest.raises(ValueError, match=r"u_lon must have shape \(2,\)"):
        longitudinal_state_derivative(
            [500.0 * FT_TO_METER, 0.0, 0.0, 0.0],
            u_lon,
            altitude_m=0.0,
        )


@pytest.mark.parametrize("true_airspeed", [0.0, -1.0])
def test_longitudinal_state_derivative_rejects_nonpositive_airspeed(
    true_airspeed,
):
    with pytest.raises(ValueError, match="VT must be positive"):
        longitudinal_state_derivative(
            [true_airspeed, 0.0, 0.0, 0.0],
            [0.2, 0.0],
            altitude_m=0.0,
        )


def test_longitudinal_derivative_is_zero_at_500_ft_s_trim(
    equilibrium_500_ft_s,
):
    trim, x_equilibrium, u_equilibrium = equilibrium_500_ft_s

    derivative = longitudinal_state_derivative(
        x_lon=x_equilibrium,
        u_lon=u_equilibrium,
        altitude_m=trim["altitude_m"],
        cg_fraction=trim["cg_fraction"],
    )

    assert trim["success"] is True
    assert np.allclose(derivative, 0.0, atol=1e-4, rtol=0.0)


def test_linearize_longitudinal_returns_finite_matrices(
    equilibrium_500_ft_s,
):
    trim, x_equilibrium, u_equilibrium = equilibrium_500_ft_s

    A, B = linearize_longitudinal(
        x_equilibrium,
        u_equilibrium,
        altitude_m=trim["altitude_m"],
        cg_fraction=trim["cg_fraction"],
    )

    assert A.shape == (4, 4)
    assert B.shape == (4, 2)
    assert np.all(np.isfinite(A))
    assert np.all(np.isfinite(B))


def test_linearize_longitudinal_is_repeatable(equilibrium_500_ft_s):
    trim, x_equilibrium, u_equilibrium = equilibrium_500_ft_s
    arguments = (x_equilibrium, u_equilibrium, trim["altitude_m"])

    A_first, B_first = linearize_longitudinal(*arguments)
    A_second, B_second = linearize_longitudinal(*arguments)

    assert np.array_equal(A_first, A_second)
    assert np.array_equal(B_first, B_second)


def test_linearize_longitudinal_is_insensitive_to_small_step_changes(
    equilibrium_500_ft_s,
    monkeypatch,
):
    trim, x_equilibrium, u_equilibrium = equilibrium_500_ft_s
    arguments = (x_equilibrium, u_equilibrium, trim["altitude_m"])
    A_default, B_default = linearize_longitudinal(*arguments)

    monkeypatch.setattr(
        linearization,
        "_LONGITUDINAL_STATE_STEPS",
        0.5 * linearization._LONGITUDINAL_STATE_STEPS,
    )
    monkeypatch.setattr(
        linearization,
        "_LONGITUDINAL_CONTROL_STEPS",
        0.5 * linearization._LONGITUDINAL_CONTROL_STEPS,
    )
    A_refined, B_refined = linearize_longitudinal(*arguments)

    assert np.allclose(A_default, A_refined, rtol=5e-3, atol=1e-5)
    assert np.allclose(B_default, B_refined, rtol=5e-3, atol=1e-5)


def test_longitudinal_linearization_matches_lewis_published_case():
    trim = trim_straight_level(
        true_airspeed=502.0 * FT_TO_METER,
        altitude_m=0.0,
        cg_fraction=0.30,
    )
    alpha = np.deg2rad(trim["alpha_deg"])
    x_equilibrium = np.array([trim["true_airspeed"], alpha, alpha, 0.0])
    u_equilibrium = np.array([trim["throttle"], trim["elevator_deg"]])

    A, _ = linearize_longitudinal(
        x_equilibrium,
        u_equilibrium,
        altitude_m=0.0,
        cg_fraction=0.30,
    )

    published_A_ft = np.array(
        [
            [-2.0244e-2, 7.8763, -32.170, -0.65020],
            [-2.5372e-4, -1.0190, 0.0, 0.90484],
            [0.0, 0.0, 0.0, 1.0],
            [7.9472e-11, -2.4982, 0.0, -1.3861],
        ]
    )
    coordinate_scale = np.diag([FT_TO_METER, 1.0, 1.0, 1.0])
    published_A_si = (
        coordinate_scale
        @ published_A_ft
        @ np.linalg.inv(coordinate_scale)
    )

    assert trim["success"] is True
    assert np.allclose(A, published_A_si, rtol=2e-2, atol=2e-4)

    computed_positive_imaginary = sorted(
        (value for value in np.linalg.eigvals(A) if value.imag > 0.0),
        key=lambda value: abs(value.imag),
        reverse=True,
    )
    published_positive_imaginary = np.array(
        [-1.2039 + 1.4922j, -0.0087297 + 0.073966j]
    )

    assert len(computed_positive_imaginary) == 2
    assert np.allclose(
        computed_positive_imaginary,
        published_positive_imaginary,
        rtol=1e-2,
        atol=2e-3,
    )
