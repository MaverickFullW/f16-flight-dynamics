"""Compare nonlinear F-16 responses with and without a pitch-rate damper."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.f16sim.parameters import FT_TO_METER
from src.f16sim.simulation import simulate_f16, simulate_f16_feedback
from src.f16sim.trim import trim_straight_level


TRUE_AIRSPEED = 502.0 * FT_TO_METER
ALTITUDE_M = 0.0
CG_FRACTION = 0.30
KQ = 5.0
DURATION = 15.0
DT = 0.01


def _pitch_angle(quaternion):
    q0, q1, q2, q3 = quaternion
    sin_theta = 2.0 * (q0 * q2 - q3 * q1)
    return np.arcsin(np.clip(sin_theta, -1.0, 1.0))


def _initial_alpha_perturbation(trim):
    state = trim["state"].copy()
    perturbed_alpha = np.deg2rad(trim["alpha_deg"] + 1.0)
    state[3] = trim["true_airspeed"] * np.cos(perturbed_alpha)
    state[5] = trim["true_airspeed"] * np.sin(perturbed_alpha)
    return state


def create_comparison_figure():
    """Build and return the nonlinear pitch-rate-damper comparison."""
    trim = trim_straight_level(
        true_airspeed=TRUE_AIRSPEED,
        altitude_m=ALTITUDE_M,
        cg_fraction=CG_FRACTION,
    )
    if not trim["success"]:
        raise RuntimeError(f"Unable to obtain the reference trim: {trim['message']}")

    initial_state = _initial_alpha_perturbation(trim)
    open_times, open_states = simulate_f16(
        initial_state=initial_state,
        duration=DURATION,
        dt=DT,
        throttle=trim["throttle"],
        elevator_deg=trim["elevator_deg"],
        aileron_deg=0.0,
        rudder_deg=0.0,
        cg_fraction=CG_FRACTION,
    )

    def pitch_rate_damper(time, state):
        return [
            trim["throttle"],
            trim["elevator_deg"] + KQ * state[11],
            0.0,
            0.0,
        ]

    closed_times, closed_states = simulate_f16_feedback(
        initial_state=initial_state,
        duration=DURATION,
        dt=DT,
        control_law=pitch_rate_damper,
        cg_fraction=CG_FRACTION,
    )

    trim_alpha = np.deg2rad(trim["alpha_deg"])
    open_alpha = np.arctan2(open_states[:, 5], open_states[:, 3])
    closed_alpha = np.arctan2(closed_states[:, 5], closed_states[:, 3])
    open_theta = np.array(
        [_pitch_angle(quaternion) for quaternion in open_states[:, 6:10]]
    )
    closed_theta = np.array(
        [_pitch_angle(quaternion) for quaternion in closed_states[:, 6:10]]
    )
    open_outputs = np.column_stack(
        (
            np.rad2deg(open_alpha - trim_alpha),
            np.rad2deg(open_theta - trim_alpha),
            np.rad2deg(open_states[:, 11]),
        )
    )
    closed_outputs = np.column_stack(
        (
            np.rad2deg(closed_alpha - trim_alpha),
            np.rad2deg(closed_theta - trim_alpha),
            np.rad2deg(closed_states[:, 11]),
        )
    )

    elevator_perturbation = KQ * closed_states[:, 11]
    print(
        "Damper elevator perturbation [deg]: "
        f"min={elevator_perturbation.min():.6f}, "
        f"max={elevator_perturbation.max():.6f}, "
        f"peak absolute={np.max(np.abs(elevator_perturbation)):.6f}"
    )

    labels = (
        r"$\Delta \alpha$ [deg]",
        r"$\Delta \theta$ [deg]",
        r"$\Delta q$ [deg/s]",
    )
    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(9.0, 9.0))
    for index, (axis, label) in enumerate(zip(axes, labels)):
        axis.plot(
            open_times,
            open_outputs[:, index],
            label="Nonlinear without q-feedback",
        )
        axis.plot(
            closed_times,
            closed_outputs[:, index],
            label="Nonlinear Kq = 5",
        )
        axis.set_ylabel(label)
        axis.grid(True)
        axis.legend()

    axes[-1].set_xlabel("Time [s]")
    figure.suptitle("Nonlinear F-16 Response with Positive Pitch-Rate Feedback")
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    create_comparison_figure()
    plt.show()
