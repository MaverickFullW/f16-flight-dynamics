"""Plot the positive pitch-rate-feedback root locus at the Lewis condition."""

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
from src.f16sim.linearization import linearize_longitudinal
from src.f16sim.parameters import FT_TO_METER
from src.f16sim.trim import trim_straight_level


def create_root_locus_figure():
    """Build the Lewis-condition positive pitch-rate-feedback root locus."""
    trim = trim_straight_level(
        true_airspeed=502.0 * FT_TO_METER,
        altitude_m=0.0,
        cg_fraction=0.30,
    )
    if not trim["success"]:
        raise RuntimeError(f"Unable to obtain the reference trim: {trim['message']}")

    alpha = np.deg2rad(trim["alpha_deg"])
    state = np.array([trim["true_airspeed"], alpha, alpha, 0.0])
    controls = np.array([trim["throttle"], trim["elevator_deg"]])
    A, B = linearize_longitudinal(
        state,
        controls,
        altitude_m=0.0,
        cg_fraction=0.30,
    )
    transfer_function = longitudinal_transfer_function(A, B, "q", "elevator")

    gains = np.linspace(0.0, 10.0, 501)
    closed_loop_poles = pitch_rate_feedback_poles(
        transfer_function["numerator"],
        transfer_function["denominator"],
        gains,
    )

    figure, axis = plt.subplots(figsize=(9.0, 7.0))
    for branch in range(closed_loop_poles.shape[1]):
        label = "Closed-loop trajectories" if branch == 0 else None
        axis.plot(
            closed_loop_poles[:, branch].real,
            closed_loop_poles[:, branch].imag,
            color="C0",
            label=label,
        )

    open_loop_poles = transfer_function["poles"]
    finite_zeros = transfer_function["zeros"][
        np.isfinite(transfer_function["zeros"])
    ]
    axis.scatter(
        open_loop_poles.real,
        open_loop_poles.imag,
        marker="x",
        s=80,
        color="C3",
        label="Open-loop poles",
        zorder=3,
    )
    axis.scatter(
        finite_zeros.real,
        finite_zeros.imag,
        marker="o",
        s=70,
        facecolors="none",
        edgecolors="C2",
        label="Finite zeros",
        zorder=3,
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Real axis [1/s]")
    axis.set_ylabel("Imaginary axis [rad/s]")
    axis.set_title(
        "F-16 Positive Pitch-Rate Feedback Root Locus\n"
        r"Characteristic equation: $1 - K_q G_q(s) = 0$"
    )
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    create_root_locus_figure()
    plt.show()
