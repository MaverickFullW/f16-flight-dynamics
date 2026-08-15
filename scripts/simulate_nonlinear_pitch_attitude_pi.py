"""Validate the cascaded pitch-attitude PI controller on the nonlinear F-16."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.f16sim.parameters import FT_TO_METER
from src.f16sim.simulation import simulate_f16_feedback_augmented
from src.f16sim.trim import trim_straight_level


TRUE_AIRSPEED = 502.0 * FT_TO_METER
ALTITUDE_M = 0.0
CG_FRACTION = 0.30
KQ = 5.0
KTHETA = 0.5
KI = 0.05
COMMAND_INCREMENT = np.deg2rad(5.0)
DURATION = 200.0
DT = 0.01


def _pitch_angle(quaternion):
    quaternion = np.asarray(quaternion, dtype=float)
    quaternion = quaternion / np.linalg.norm(quaternion)
    q0, q1, q2, q3 = quaternion
    sin_theta = 2.0 * (q0 * q2 - q3 * q1)
    return np.arcsin(np.clip(sin_theta, -1.0, 1.0))


def _settling_time(times, attitude_error):
    tolerance = 0.02 * abs(COMMAND_INCREMENT)
    outside_band = np.flatnonzero(np.abs(attitude_error) > tolerance)
    if outside_band.size == 0:
        return float(times[0])
    last_outside = outside_band[-1]
    if last_outside == times.size - 1:
        return None
    return float(times[last_outside + 1])


def create_command_response_figure():
    """Simulate and return the nonlinear attitude-PI command response."""
    trim = trim_straight_level(
        true_airspeed=TRUE_AIRSPEED,
        altitude_m=ALTITUDE_M,
        cg_fraction=CG_FRACTION,
    )
    if not trim["success"]:
        raise RuntimeError(f"Unable to obtain the reference trim: {trim['message']}")

    trim_theta = _pitch_angle(trim["state"][6:10])
    theta_command = trim_theta + COMMAND_INCREMENT

    def attitude_error(state):
        return theta_command - _pitch_angle(state[6:10])

    def control_law(time, state, controller_state):
        q_command = KTHETA * attitude_error(state) + KI * controller_state[0]
        elevator = trim["elevator_deg"] + KQ * (state[11] - q_command)
        return [trim["throttle"], elevator, 0.0, 0.0]

    def controller_derivative(time, state, controller_state):
        return np.array([attitude_error(state)])

    times, states, controller_states = simulate_f16_feedback_augmented(
        initial_state=trim["state"],
        initial_controller_state=np.zeros(1),
        duration=DURATION,
        dt=DT,
        control_law=control_law,
        controller_state_derivative=controller_derivative,
        cg_fraction=CG_FRACTION,
    )

    theta = np.array([_pitch_angle(quaternion) for quaternion in states[:, 6:10]])
    alpha = np.arctan2(states[:, 5], states[:, 3])
    delta_theta = theta - trim_theta
    delta_alpha = alpha - np.deg2rad(trim["alpha_deg"])
    delta_q = states[:, 11]
    xi = controller_states[:, 0]
    error = theta_command - theta
    q_command = KTHETA * error + KI * xi
    elevator_perturbation = KQ * (delta_q - q_command)

    final_theta = np.rad2deg(delta_theta[-1])
    final_error = np.rad2deg(error[-1])
    maximum_theta = np.rad2deg(np.max(delta_theta))
    command_degrees = np.rad2deg(COMMAND_INCREMENT)
    percent_overshoot = max(
        0.0,
        100.0 * (maximum_theta - command_degrees) / abs(command_degrees),
    )
    maximum_elevator = np.max(np.abs(elevator_perturbation))
    maximum_xi = np.max(np.abs(xi))
    settling_time = _settling_time(times, error)
    controls_finite = np.all(
        np.isfinite(
            np.column_stack(
                (
                    np.full_like(times, trim["throttle"]),
                    trim["elevator_deg"] + elevator_perturbation,
                    np.zeros_like(times),
                    np.zeros_like(times),
                )
            )
        )
    )
    states_finite = np.all(np.isfinite(states)) and np.all(
        np.isfinite(controller_states)
    )

    print(f"Final Delta theta: {final_theta:.6f} deg")
    print(f"Final attitude error: {final_error:.6f} deg")
    print(f"Maximum Delta theta: {maximum_theta:.6f} deg")
    print(f"Percent overshoot: {percent_overshoot:.3f}%")
    print(f"Maximum absolute elevator perturbation: {maximum_elevator:.6f} deg")
    print(f"Maximum absolute xi: {maximum_xi:.6f} rad*s")
    if settling_time is None:
        print("2% settling time: response does not enter and remain inside the band")
    else:
        print(f"Approximate 2% settling time: {settling_time:.2f} s")
    print(f"All simulated states remained finite: {states_finite}")
    print(f"All simulated controls remained finite: {controls_finite}")

    figure, axes = plt.subplots(5, 1, sharex=True, figsize=(10.0, 13.0))
    axes[0].plot(times, np.rad2deg(delta_theta), label="Delta theta")
    axes[0].axhline(
        command_degrees,
        color="black",
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
    axes[3].legend()

    axes[4].plot(times, xi, label="Integral state xi")
    axes[4].set_ylabel(r"$\xi$ [rad s]")
    axes[4].set_xlabel("Time [s]")
    axes[4].legend()

    for axis in axes:
        axis.grid(True)

    figure.suptitle(
        "Nonlinear F-16 Pitch-Attitude PI Command Response\n"
        r"$K_q = 5$, $K_\theta = 0.5$, $K_i = 0.05$, "
        r"$\Delta\theta_{cmd} = 5$ deg"
    )
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    create_command_response_figure()
    plt.show()
