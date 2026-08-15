"""Compare linear and nonlinear longitudinal responses at the Lewis condition."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.f16sim.linear_response import simulate_linear_longitudinal
from src.f16sim.linearization import linearize_longitudinal
from src.f16sim.parameters import FT_TO_METER
from src.f16sim.simulation import simulate_f16
from src.f16sim.trim import trim_straight_level


TRUE_AIRSPEED = 502.0 * FT_TO_METER
ALTITUDE_M = 0.0
CG_FRACTION = 0.30
DURATION = 20.0
DT = 0.01


def elevator_pulse(time):
    """Return the 0.5-degree elevator perturbation applied for 0.5 seconds."""
    return 0.5 if 0.0 <= time < 0.5 else 0.0


def _pitch_angle(quaternion):
    """Return pitch angle from a scalar-first NED-to-body quaternion."""
    q0, q1, q2, q3 = quaternion
    sin_theta = 2.0 * (q0 * q2 - q3 * q1)
    return np.arcsin(np.clip(sin_theta, -1.0, 1.0))


def create_comparison_figure():
    """Simulate the Lewis case and return its linear/nonlinear comparison."""
    trim = trim_straight_level(
        true_airspeed=TRUE_AIRSPEED,
        altitude_m=ALTITUDE_M,
        cg_fraction=CG_FRACTION,
    )
    if not trim["success"]:
        raise RuntimeError(f"Unable to obtain the reference trim: {trim['message']}")

    trim_alpha = np.deg2rad(trim["alpha_deg"])
    reduced_trim_state = np.array(
        [trim["true_airspeed"], trim_alpha, trim_alpha, 0.0]
    )
    trim_control = np.array([trim["throttle"], trim["elevator_deg"]])
    A, B = linearize_longitudinal(
        reduced_trim_state,
        trim_control,
        altitude_m=ALTITUDE_M,
        cg_fraction=CG_FRACTION,
    )

    def linear_control(time):
        return np.array([0.0, elevator_pulse(time)])

    linear_times, linear_perturbations = simulate_linear_longitudinal(
        A,
        B,
        initial_perturbation=np.zeros(4),
        duration=DURATION,
        dt=DT,
        control_perturbation=linear_control,
    )

    nonlinear_times, nonlinear_states = simulate_f16(
        initial_state=trim["state"],
        duration=DURATION,
        dt=DT,
        throttle=trim["throttle"],
        elevator_deg=lambda time: trim["elevator_deg"] + elevator_pulse(time),
        aileron_deg=0.0,
        rudder_deg=0.0,
        cg_fraction=CG_FRACTION,
    )

    nonlinear_alpha = np.arctan2(
        nonlinear_states[:, 5], nonlinear_states[:, 3]
    )
    nonlinear_theta = np.array(
        [_pitch_angle(quaternion) for quaternion in nonlinear_states[:, 6:10]]
    )

    linear_outputs = np.column_stack(
        (
            np.rad2deg(linear_perturbations[:, 1]),
            np.rad2deg(linear_perturbations[:, 2]),
            np.rad2deg(linear_perturbations[:, 3]),
        )
    )
    nonlinear_outputs = np.column_stack(
        (
            np.rad2deg(nonlinear_alpha - trim_alpha),
            np.rad2deg(nonlinear_theta - trim_alpha),
            np.rad2deg(nonlinear_states[:, 11]),
        )
    )

    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(9.0, 9.0))
    labels = (r"$\Delta \alpha$ [deg]", r"$\Delta \theta$ [deg]", r"$\Delta q$ [deg/s]")
    for index, (axis, label) in enumerate(zip(axes, labels)):
        axis.plot(linear_times, linear_outputs[:, index], label="Linear")
        axis.plot(nonlinear_times, nonlinear_outputs[:, index], label="Nonlinear")
        axis.set_ylabel(label)
        axis.grid(True)
        axis.legend()

    axes[-1].set_xlabel("Time [s]")
    figure.suptitle(
        "F-16 Longitudinal Linear and Nonlinear Response\n"
        "502 ft/s, sea level, cg = 0.30, +0.5 deg elevator pulse"
    )
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    create_comparison_figure()
    plt.show()
