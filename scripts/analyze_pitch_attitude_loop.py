"""Analyze cascaded pitch-rate and pitch-attitude feedback at the Lewis case."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.f16sim.control_analysis import pitch_attitude_feedback_poles
from src.f16sim.linearization import linearize_longitudinal
from src.f16sim.parameters import FT_TO_METER
from src.f16sim.trim import trim_straight_level


KQ = 5.0
REPORTED_GAINS = np.array([0.0, 0.1, 0.2, 0.5, 1.0, 2.0])


def create_attitude_loop_figure():
    """Build and return the cascaded attitude-loop pole-trajectory figure."""
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

    sweep_gains = np.linspace(0.0, 2.0, 501)
    sweep_poles = pitch_attitude_feedback_poles(A, B, sweep_gains, Kq=KQ)
    reported_poles = pitch_attitude_feedback_poles(A, B, REPORTED_GAINS, Kq=KQ)
    for gain, poles in zip(REPORTED_GAINS, reported_poles):
        print(f"Ktheta = {gain:g} 1/s poles: {poles}")

    figure, axis = plt.subplots(figsize=(9.0, 7.0))
    for branch in range(sweep_poles.shape[1]):
        label = "Closed-loop trajectories" if branch == 0 else None
        axis.plot(
            sweep_poles[:, branch].real,
            sweep_poles[:, branch].imag,
            color="C0",
            label=label,
        )

    axis.scatter(
        sweep_poles[0].real,
        sweep_poles[0].imag,
        marker="x",
        s=80,
        color="C3",
        label=r"$K_\theta = 0$ poles",
        zorder=3,
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Real axis [1/s]")
    axis.set_ylabel("Imaginary axis [rad/s]")
    axis.set_title(
        "F-16 Cascaded Pitch-Attitude Loop Pole Trajectories\n"
        r"$K_q = 5$ deg/(rad/s), positive $K_\theta$ sweep"
    )
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    create_attitude_loop_figure()
    plt.show()
