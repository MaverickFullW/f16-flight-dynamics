"""Simulate a linear F-16 pitch-attitude command with cascaded feedback."""

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
from src.f16sim.trim import trim_straight_level


TRUE_AIRSPEED = 502.0 * FT_TO_METER
ALTITUDE_M = 0.0
CG_FRACTION = 0.30
KQ = 5.0
KTHETA = 0.5
THETA_COMMAND = np.deg2rad(5.0)
DURATION = 60.0
DT = 0.01


def _settling_time(times, attitude_error, command):
    tolerance = 0.02 * abs(command)
    outside_band = np.flatnonzero(np.abs(attitude_error) > tolerance)
    if outside_band.size == 0:
        return float(times[0])
    last_outside = outside_band[-1]
    if last_outside == times.size - 1:
        return None
    return float(times[last_outside + 1])


def create_command_response_figure():
    """Build and return the cascaded pitch-attitude command response."""
    trim = trim_straight_level(
        true_airspeed=TRUE_AIRSPEED,
        altitude_m=ALTITUDE_M,
        cg_fraction=CG_FRACTION,
    )
    if not trim["success"]:
        raise RuntimeError(f"Unable to obtain the reference trim: {trim['message']}")

    trim_alpha = np.deg2rad(trim["alpha_deg"])
    trim_state = np.array(
        [trim["true_airspeed"], trim_alpha, trim_alpha, 0.0]
    )
    trim_controls = np.array([trim["throttle"], trim["elevator_deg"]])
    A, B = linearize_longitudinal(
        trim_state,
        trim_controls,
        altitude_m=ALTITUDE_M,
        cg_fraction=CG_FRACTION,
    )

    elevator_column = B[:, 1:2]
    theta_output = np.array([[0.0, 0.0, 1.0, 0.0]])
    pitch_rate_output = np.array([[0.0, 0.0, 0.0, 1.0]])
    feedback_output = pitch_rate_output + KTHETA * theta_output
    closed_loop_A = A + KQ * (elevator_column @ feedback_output)

    command_elevator = -KQ * KTHETA * THETA_COMMAND
    times, response = simulate_linear_longitudinal(
        closed_loop_A,
        B,
        initial_perturbation=np.zeros(4),
        duration=DURATION,
        dt=DT,
        control_perturbation=np.array([0.0, command_elevator]),
    )

    delta_alpha = response[:, 1]
    delta_theta = response[:, 2]
    delta_q = response[:, 3]
    theta_command = np.full_like(times, THETA_COMMAND)
    attitude_error = theta_command - delta_theta
    q_command = KTHETA * attitude_error
    elevator_perturbation = KQ * (delta_q - q_command)

    final_theta_deg = np.rad2deg(delta_theta[-1])
    final_error_deg = np.rad2deg(attitude_error[-1])
    maximum_theta_deg = np.rad2deg(np.max(delta_theta))
    command_deg = np.rad2deg(THETA_COMMAND)
    percent_overshoot = max(
        0.0,
        100.0 * (maximum_theta_deg - command_deg) / abs(command_deg),
    )
    maximum_elevator = np.max(np.abs(elevator_perturbation))
    settling_time = _settling_time(times, attitude_error, THETA_COMMAND)

    print(f"Final Delta theta: {final_theta_deg:.6f} deg")
    print(f"Final attitude error: {final_error_deg:.6f} deg")
    print(f"Maximum Delta theta: {maximum_theta_deg:.6f} deg")
    print(f"Percent overshoot: {percent_overshoot:.3f}%")
    print(f"Maximum absolute elevator perturbation: {maximum_elevator:.6f} deg")
    if settling_time is None:
        print("2% settling time: response does not enter and remain inside the band")
    else:
        print(f"Approximate 2% settling time: {settling_time:.2f} s")

    figure, axes = plt.subplots(4, 1, sharex=True, figsize=(9.0, 11.0))
    axes[0].plot(times, np.rad2deg(delta_theta), label="Delta theta")
    axes[0].plot(
        times,
        np.rad2deg(theta_command),
        linestyle="--",
        label="Theta command",
    )
    axes[0].set_ylabel(r"$\Delta \theta$ [deg]")
    axes[0].legend()

    axes[1].plot(times, np.rad2deg(delta_q), label="Delta q")
    axes[1].plot(times, np.rad2deg(q_command), linestyle="--", label="q command")
    axes[1].set_ylabel(r"$\Delta q$ [deg/s]")
    axes[1].legend()

    axes[2].plot(times, np.rad2deg(delta_alpha), label="Delta alpha")
    axes[2].set_ylabel(r"$\Delta \alpha$ [deg]")
    axes[2].legend()

    axes[3].plot(times, elevator_perturbation, label="Delta elevator")
    axes[3].set_ylabel(r"$\Delta$ elevator [deg]")
    axes[3].set_xlabel("Time [s]")
    axes[3].legend()

    for axis in axes:
        axis.grid(True)

    figure.suptitle(
        "Linear F-16 Pitch-Attitude Command Response\n"
        r"$K_q = 5$ deg/(rad/s), $K_\theta = 0.5$ 1/s, "
        r"$\theta_{cmd} = 5$ deg"
    )
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    create_command_response_figure()
    plt.show()
