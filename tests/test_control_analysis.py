import numpy as np
import pytest

from src.f16sim.control_analysis import (
    longitudinal_transfer_function,
    pitch_attitude_feedback_poles,
    pitch_attitude_pi_feedback_poles,
    pitch_rate_feedback_poles,
)
from src.f16sim.linearization import linearize_longitudinal
from src.f16sim.parameters import FT_TO_METER
from src.f16sim.trim import trim_straight_level


@pytest.fixture(scope="module")
def lewis_longitudinal_matrices():
    trim = trim_straight_level(
        true_airspeed=502.0 * FT_TO_METER,
        altitude_m=0.0,
        cg_fraction=0.30,
    )
    alpha = np.deg2rad(trim["alpha_deg"])
    x_equilibrium = np.array([trim["true_airspeed"], alpha, alpha, 0.0])
    u_equilibrium = np.array([trim["throttle"], trim["elevator_deg"]])
    A, B = linearize_longitudinal(
        x_equilibrium,
        u_equilibrium,
        altitude_m=0.0,
        cg_fraction=0.30,
    )

    assert trim["success"] is True
    return A, B


@pytest.mark.parametrize("output", ["q", "theta", "alpha"])
def test_elevator_transfer_functions_match_state_matrix_poles(
    lewis_longitudinal_matrices, output
):
    A, B = lewis_longitudinal_matrices
    result = longitudinal_transfer_function(A, B, output, "elevator")

    expected_poles = np.linalg.eigvals(A)
    assert result["output"] == output
    assert result["input"] == "elevator"
    assert result["numerator"].ndim == 1
    assert result["denominator"].shape == (5,)
    assert np.all(np.isfinite(result["numerator"]))
    assert np.all(np.isfinite(result["denominator"]))
    assert np.all(np.isfinite(result["poles"]))
    assert np.all(np.isfinite(result["zeros"]))
    assert np.allclose(
        np.sort_complex(result["poles"]),
        np.sort_complex(expected_poles),
        rtol=1e-6,
        atol=1e-8,
    )


def test_longitudinal_transfer_function_rejects_invalid_output(
    lewis_longitudinal_matrices,
):
    A, B = lewis_longitudinal_matrices

    with pytest.raises(ValueError, match="unsupported longitudinal output"):
        longitudinal_transfer_function(A, B, "altitude", "elevator")


def test_longitudinal_transfer_function_rejects_invalid_input(
    lewis_longitudinal_matrices,
):
    A, B = lewis_longitudinal_matrices

    with pytest.raises(ValueError, match="unsupported longitudinal input"):
        longitudinal_transfer_function(A, B, "q", "aileron")


@pytest.mark.parametrize(
    ("A", "B", "message"),
    [
        (np.zeros((3, 3)), np.zeros((4, 2)), r"A must have shape \(4, 4\)"),
        (np.zeros((4, 5)), np.zeros((4, 2)), r"A must have shape \(4, 4\)"),
        (np.zeros((4, 4)), np.zeros((3, 2)), r"B must have shape \(4, 2\)"),
        (np.zeros((4, 4)), np.zeros((4, 3)), r"B must have shape \(4, 2\)"),
    ],
)
def test_longitudinal_transfer_function_rejects_invalid_matrix_shapes(
    A, B, message
):
    with pytest.raises(ValueError, match=message):
        longitudinal_transfer_function(A, B, "q", "elevator")


def test_zero_pitch_rate_feedback_gain_reproduces_open_loop_poles(
    lewis_longitudinal_matrices,
):
    A, B = lewis_longitudinal_matrices
    transfer_function = longitudinal_transfer_function(A, B, "q", "elevator")

    poles = pitch_rate_feedback_poles(
        transfer_function["numerator"],
        transfer_function["denominator"],
        gains=np.array([0.0]),
    )

    assert poles.shape == (1, 4)
    assert np.allclose(
        np.sort_complex(poles[0]),
        np.sort_complex(transfer_function["poles"]),
        rtol=1e-7,
        atol=1e-9,
    )


def test_pitch_rate_feedback_poles_returns_finite_expected_shape(
    lewis_longitudinal_matrices,
):
    A, B = lewis_longitudinal_matrices
    transfer_function = longitudinal_transfer_function(A, B, "q", "elevator")
    gains = np.linspace(0.0, 10.0, 51)

    poles = pitch_rate_feedback_poles(
        transfer_function["numerator"],
        transfer_function["denominator"],
        gains,
    )

    assert poles.shape == (gains.size, 4)
    assert np.all(np.isfinite(poles))


@pytest.mark.parametrize(
    ("numerator", "denominator", "gains"),
    [
        (np.zeros((1, 2)), [1.0, 2.0], [0.0]),
        ([1.0], np.zeros((1, 2)), [0.0]),
        ([], [1.0, 2.0], [0.0]),
        ([1.0], [1.0], [0.0]),
        ([1.0, 2.0, 3.0], [1.0, 2.0], [0.0]),
        ([1.0], [0.0, 1.0], [0.0]),
        ([np.nan], [1.0, 2.0], [0.0]),
        ([1.0], [1.0, 2.0], []),
        ([1.0], [1.0, 2.0], [[0.0]]),
        ([1.0], [1.0, 2.0], [-1.0]),
        ([1.0], [1.0, 2.0], [np.inf]),
    ],
)
def test_pitch_rate_feedback_poles_rejects_invalid_inputs(
    numerator, denominator, gains
):
    with pytest.raises(ValueError):
        pitch_rate_feedback_poles(numerator, denominator, gains)


def test_zero_attitude_gain_reproduces_pitch_rate_feedback_poles(
    lewis_longitudinal_matrices,
):
    A, B = lewis_longitudinal_matrices
    transfer_function = longitudinal_transfer_function(A, B, "q", "elevator")
    pitch_rate_poles = pitch_rate_feedback_poles(
        transfer_function["numerator"],
        transfer_function["denominator"],
        gains=np.array([5.0]),
    )
    attitude_loop_poles = pitch_attitude_feedback_poles(A, B, k_theta=0.0)

    assert attitude_loop_poles.shape == (1, 4)
    assert np.allclose(
        np.sort_complex(attitude_loop_poles[0]),
        np.sort_complex(pitch_rate_poles[0]),
        rtol=1e-7,
        atol=1e-9,
    )


def test_pitch_attitude_feedback_poles_returns_finite_expected_shape(
    lewis_longitudinal_matrices,
):
    A, B = lewis_longitudinal_matrices
    gains = np.array([0.0, 0.1, 0.2, 0.5, 1.0, 2.0])

    poles = pitch_attitude_feedback_poles(A, B, gains)

    assert poles.shape == (gains.size, 4)
    assert np.all(np.isfinite(poles))


@pytest.mark.parametrize(
    "invalid_gain",
    [[], [[0.1]], -0.1, [0.1, np.nan], [0.1, np.inf]],
)
def test_pitch_attitude_feedback_poles_rejects_invalid_attitude_gains(
    lewis_longitudinal_matrices, invalid_gain
):
    A, B = lewis_longitudinal_matrices

    with pytest.raises(ValueError):
        pitch_attitude_feedback_poles(A, B, invalid_gain)


@pytest.mark.parametrize("invalid_kq", [0.0, -1.0, np.nan, np.inf, [5.0]])
def test_pitch_attitude_feedback_poles_rejects_invalid_pitch_rate_gain(
    lewis_longitudinal_matrices, invalid_kq
):
    A, B = lewis_longitudinal_matrices

    with pytest.raises(ValueError):
        pitch_attitude_feedback_poles(A, B, 0.1, Kq=invalid_kq)


def test_zero_integral_gain_adds_zero_pole_to_cascaded_attitude_loop(
    lewis_longitudinal_matrices,
):
    A, B = lewis_longitudinal_matrices
    pi_poles = pitch_attitude_pi_feedback_poles(A, B, ki=0.0)
    proportional_poles = pitch_attitude_feedback_poles(A, B, k_theta=0.5)

    assert pi_poles.shape == (1, 5)
    zero_index = np.argmin(np.abs(pi_poles[0]))
    assert pi_poles[0, zero_index] == pytest.approx(0.0, abs=1e-12)
    remaining_poles = np.delete(pi_poles[0], zero_index)
    assert np.allclose(
        np.sort_complex(remaining_poles),
        np.sort_complex(proportional_poles[0]),
        rtol=1e-7,
        atol=1e-9,
    )


def test_pitch_attitude_pi_feedback_poles_returns_finite_expected_shape(
    lewis_longitudinal_matrices,
):
    A, B = lewis_longitudinal_matrices
    gains = np.array([0.0, 0.005, 0.01, 0.02, 0.05, 0.1])

    poles = pitch_attitude_pi_feedback_poles(A, B, gains)

    assert poles.shape == (gains.size, 5)
    assert np.all(np.isfinite(poles))


@pytest.mark.parametrize(
    "invalid_gain",
    [[], [[0.01]], -0.01, [0.01, np.nan], [0.01, np.inf]],
)
def test_pitch_attitude_pi_feedback_poles_rejects_invalid_integral_gains(
    lewis_longitudinal_matrices, invalid_gain
):
    A, B = lewis_longitudinal_matrices

    with pytest.raises(ValueError):
        pitch_attitude_pi_feedback_poles(A, B, invalid_gain)
