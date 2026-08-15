"""Compare linear F-16 responses with and without pitch-rate feedback."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.f16sim.control_analysis import (
    longitudinal_transfer_function,
    pitch_rate_feedback_poles,
)
from src.f16sim.linear_response import simulate_linear_longitudinal
from src.f16sim.linearization import (
    analyze_longitudinal_modes,
    linearize_longitudinal,
)
from src.f16sim.parameters import FT_TO_METER
from src.f16sim.trim import trim_straight_level


TRUE_AIRSPEED = 502.0 * FT_TO_METER
ALTITUDE_M = 0.0
CG_FRACTION = 0.30
KQ = 5.0
DURATION = 15.0
DT = 0.01


def _lewis_linear_model():
    trim = trim_straight_level(
        true_airspeed=TRUE_AIRSPEED,
        altitude_m=ALTITUDE_M,
        cg_fraction=CG_FRACTION,
    )
    if not trim["success"]:
        raise RuntimeError(f"Unable to obtain the reference trim: {trim['message']}")

    alpha = np.deg2rad(trim["alpha_deg"])
    state = np.array([trim["true_airspeed"], alpha, alpha, 0.0])
    controls = np.array([trim["throttle"], trim["elevator_deg"]])
    return linearize_longitudinal(
        state,
        controls,
        altitude_m=ALTITUDE_M,
        cg_fraction=CG_FRACTION,
    )


def _closed_loop_matrix(A, B):
    elevator_column = B[:, 1:2]
    pitch_rate_output = np.array([[0.0, 0.0, 0.0, 1.0]])
    return A + KQ * (elevator_column @ pitch_rate_output)


def _verify_closed_loop_poles(A, B, A_closed_loop):
    transfer_function = longitudinal_transfer_function(A, B, "q", "elevator")
    root_locus_poles = pitch_rate_feedback_poles(
        transfer_function["numerator"],
        transfer_function["denominator"],
        np.array([KQ]),
    )[0]
    state_space_poles = np.linalg.eigvals(A_closed_loop)
    if not np.allclose(
        np.sort_complex(state_space_poles),
        np.sort_complex(root_locus_poles),
        rtol=1e-7,
        atol=1e-9,
    ):
        raise RuntimeError(
            "Closed-loop state-space poles do not match 1 - Kq Gq(s) = 0"
        )

    modes = analyze_longitudinal_modes(A_closed_loop)
    expected = {
        "short_period": -1.65963 + 1.35116j,
        "phugoid": -0.0089865 + 0.0661194j,
    }
    for name, expected_eigenvalue in expected.items():
        if not np.isclose(
            modes[name]["eigenvalue"],
            expected_eigenvalue,
            rtol=1e-2,
            atol=2e-3,
        ):
            raise RuntimeError(
                f"Unexpected {name} pole: {modes[name]['eigenvalue']}"
            )


def create_comparison_figure():
    """Build and return the open- and closed-loop response comparison."""
    A, B = _lewis_linear_model()
    A_closed_loop = _closed_loop_matrix(A, B)
    _verify_closed_loop_poles(A, B, A_closed_loop)

    open_loop_modes = analyze_longitudinal_modes(A)
    closed_loop_modes = analyze_longitudinal_modes(A_closed_loop)
    open_loop_eigenvalues = np.linalg.eigvals(A)
    closed_loop_eigenvalues = np.linalg.eigvals(A_closed_loop)

    print("Open-loop eigenvalues:", open_loop_eigenvalues)
    print("Closed-loop eigenvalues:", closed_loop_eigenvalues)
    print(
        "Open-loop short-period damping ratio:",
        open_loop_modes["short_period"]["damping_ratio"],
    )
    print(
        "Kq = 5 short-period damping ratio:",
        closed_loop_modes["short_period"]["damping_ratio"],
    )

    initial_perturbation = np.array([0.0, np.deg2rad(1.0), 0.0, 0.0])
    open_times, open_response = simulate_linear_longitudinal(
        A,
        B,
        initial_perturbation,
        duration=DURATION,
        dt=DT,
    )
    closed_times, closed_response = simulate_linear_longitudinal(
        A_closed_loop,
        B,
        initial_perturbation,
        duration=DURATION,
        dt=DT,
    )

    open_outputs = np.rad2deg(open_response[:, 1:4])
    closed_outputs = np.rad2deg(closed_response[:, 1:4])
    labels = (
        r"$\Delta \alpha$ [deg]",
        r"$\Delta \theta$ [deg]",
        r"$\Delta q$ [deg/s]",
    )

    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(9.0, 9.0))
    for index, (axis, label) in enumerate(zip(axes, labels)):
        axis.plot(open_times, open_outputs[:, index], label="Without q-feedback")
        axis.plot(closed_times, closed_outputs[:, index], label="Kq = 5")
        axis.set_ylabel(label)
        axis.grid(True)
        axis.legend()

    axes[-1].set_xlabel("Time [s]")
    figure.suptitle(
        "F-16 Longitudinal Response with Positive Pitch-Rate Feedback\n"
        r"$\delta_e = K_q \Delta q$, $K_q = 5$ deg/(rad/s)"
    )
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    create_comparison_figure()
    plt.show()
