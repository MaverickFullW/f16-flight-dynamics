"""Compare linear and nonlinear F-16 responses to an attitude maneuver."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.f16sim.controllers import pitch_attitude_control
from src.f16sim.linear_response import simulate_linear_longitudinal
from src.f16sim.linearization import linearize_longitudinal
from src.f16sim.parameters import FT_TO_METER
from src.f16sim.simulation import simulate_f16_feedback
from src.f16sim.trim import trim_straight_level


TRUE_AIRSPEED = 502.0 * FT_TO_METER
ALTITUDE_M = 0.0
CG_FRACTION = 0.30
KQ = 5.0
KTHETA = 0.5
DURATION = 40.0
DT = 0.01


def _pitch_angle(quaternion):
    quaternion = np.asarray(quaternion, dtype=float)
    quaternion = quaternion / np.linalg.norm(quaternion)
    q0, q1, q2, q3 = quaternion
    sin_theta = 2.0 * (q0 * q2 - q3 * q1)
    return np.arcsin(np.clip(sin_theta, -1.0, 1.0))


def _attitude_command(time):
    return np.deg2rad(5.0) if 2.0 <= time < 12.0 else 0.0


def _print_metrics(name, times, theta, q, elevator):
    command_interval = (times >= 2.0) & (times < 12.0)
    before_removal = np.searchsorted(times, 12.0) - 1
    print(name)
    print(
        "  Maximum Delta theta during command: "
        f"{np.rad2deg(np.max(theta[command_interval])):.6f} deg"
    )
    print(
        "  Delta theta immediately before command removal: "
        f"{np.rad2deg(theta[before_removal]):.6f} deg"
    )
    print(f"  Maximum absolute q: {np.rad2deg(np.max(np.abs(q))):.6f} deg/s")
    print(
        "  Maximum absolute elevator perturbation: "
        f"{np.max(np.abs(elevator)):.6f} deg"
    )
    print(f"  Delta theta at t = 40 s: {np.rad2deg(theta[-1]):.6f} deg")


def create_comparison_figure():
    """Build and return the finite attitude-maneuver comparison."""
    trim = trim_straight_level(
        true_airspeed=TRUE_AIRSPEED,
        altitude_m=ALTITUDE_M,
        cg_fraction=CG_FRACTION,
    )
    if not trim["success"]:
        raise RuntimeError(f"Unable to obtain the reference trim: {trim['message']}")

    trim_theta = _pitch_angle(trim["state"][6:10])
    trim_alpha = np.deg2rad(trim["alpha_deg"])
    reduced_trim_state = np.array(
        [trim["true_airspeed"], trim_alpha, trim_theta, 0.0]
    )
    trim_controls = np.array([trim["throttle"], trim["elevator_deg"]])
    A, B = linearize_longitudinal(
        reduced_trim_state,
        trim_controls,
        altitude_m=ALTITUDE_M,
        cg_fraction=CG_FRACTION,
    )

    elevator_column = B[:, 1:2]
    theta_output = np.array([[0.0, 0.0, 1.0, 0.0]])
    pitch_rate_output = np.array([[0.0, 0.0, 0.0, 1.0]])
    closed_loop_A = A + KQ * (
        elevator_column @ (pitch_rate_output + KTHETA * theta_output)
    )

    def linear_command_control(time):
        _, elevator = pitch_attitude_control(
            _attitude_command(time),
            theta=0.0,
            q=0.0,
            Kq=KQ,
            Ktheta=KTHETA,
        )
        return np.array([0.0, elevator])

    linear_times, linear_states = simulate_linear_longitudinal(
        closed_loop_A,
        B,
        initial_perturbation=np.zeros(4),
        duration=DURATION,
        dt=DT,
        control_perturbation=linear_command_control,
    )

    def nonlinear_control_law(time, state):
        theta_command = trim_theta + _attitude_command(time)
        _, elevator_perturbation = pitch_attitude_control(
            theta_command,
            _pitch_angle(state[6:10]),
            state[11],
            Kq=KQ,
            Ktheta=KTHETA,
        )
        return [
            trim["throttle"],
            trim["elevator_deg"] + elevator_perturbation,
            0.0,
            0.0,
        ]

    nonlinear_times, nonlinear_states = simulate_f16_feedback(
        initial_state=trim["state"],
        duration=DURATION,
        dt=DT,
        control_law=nonlinear_control_law,
        cg_fraction=CG_FRACTION,
    )

    command = np.array([_attitude_command(time) for time in linear_times])
    linear_theta = linear_states[:, 2]
    linear_q = linear_states[:, 3]
    linear_alpha = linear_states[:, 1]
    linear_q_command = KTHETA * (command - linear_theta)
    linear_elevator = KQ * (linear_q - linear_q_command)

    nonlinear_theta_absolute = np.array(
        [_pitch_angle(quaternion) for quaternion in nonlinear_states[:, 6:10]]
    )
    nonlinear_theta = nonlinear_theta_absolute - trim_theta
    nonlinear_q = nonlinear_states[:, 11]
    nonlinear_alpha = (
        np.arctan2(nonlinear_states[:, 5], nonlinear_states[:, 3]) - trim_alpha
    )
    nonlinear_q_command = KTHETA * (command - nonlinear_theta)
    nonlinear_elevator = KQ * (nonlinear_q - nonlinear_q_command)

    _print_metrics(
        "Linear model", linear_times, linear_theta, linear_q, linear_elevator
    )
    _print_metrics(
        "Nonlinear model",
        nonlinear_times,
        nonlinear_theta,
        nonlinear_q,
        nonlinear_elevator,
    )
    maximum_theta_difference = np.max(
        np.abs(np.rad2deg(linear_theta - nonlinear_theta))
    )
    print(
        "Maximum absolute linear/nonlinear Delta theta difference: "
        f"{maximum_theta_difference:.6f} deg"
    )

    figure, axes = plt.subplots(4, 1, sharex=True, figsize=(10.0, 11.0))
    axes[0].plot(linear_times, np.rad2deg(linear_theta), label="Linear")
    axes[0].plot(nonlinear_times, np.rad2deg(nonlinear_theta), label="Nonlinear")
    axes[0].plot(
        linear_times,
        np.rad2deg(command),
        color="black",
        linestyle="--",
        label="Delta theta command",
    )
    axes[0].set_ylabel(r"$\Delta \theta$ [deg]")

    axes[1].plot(linear_times, np.rad2deg(linear_q), label="Linear q")
    axes[1].plot(nonlinear_times, np.rad2deg(nonlinear_q), label="Nonlinear q")
    axes[1].plot(
        linear_times,
        np.rad2deg(linear_q_command),
        linestyle=":",
        label="Linear q command",
    )
    axes[1].set_ylabel(r"$\Delta q$ [deg/s]")

    axes[2].plot(linear_times, np.rad2deg(linear_alpha), label="Linear")
    axes[2].plot(nonlinear_times, np.rad2deg(nonlinear_alpha), label="Nonlinear")
    axes[2].set_ylabel(r"$\Delta \alpha$ [deg]")

    axes[3].plot(linear_times, linear_elevator, label="Linear")
    axes[3].plot(nonlinear_times, nonlinear_elevator, label="Nonlinear")
    axes[3].set_ylabel(r"$\Delta$ elevator [deg]")
    axes[3].set_xlabel("Time [s]")

    for axis in axes:
        axis.grid(True)
        axis.legend()

    figure.suptitle(
        "F-16 Linear and Nonlinear Finite Attitude Maneuver\n"
        r"$K_q = 5$ deg/(rad/s), $K_\theta = 0.5$ 1/s"
    )
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    create_comparison_figure()
    plt.show()
