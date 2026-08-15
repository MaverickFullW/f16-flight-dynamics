"""Compare candidate integral gains for the linear pitch-attitude controller."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.f16sim.linearization import linearize_longitudinal
from src.f16sim.parameters import FT_TO_METER
from src.f16sim.trim import trim_straight_level


TRUE_AIRSPEED = 502.0 * FT_TO_METER
ALTITUDE_M = 0.0
CG_FRACTION = 0.30
KQ = 5.0
KTHETA = 0.5
INTEGRAL_GAINS = (0.01, 0.02, 0.05)
THETA_COMMAND = np.deg2rad(5.0)
DURATION = 200.0
DT = 0.01


def _augmented_system(A, B, integral_gain):
    elevator_column = B[:, 1:2]
    theta_output = np.array([[0.0, 0.0, 1.0, 0.0]])
    pitch_rate_output = np.array([[0.0, 0.0, 0.0, 1.0]])

    augmented_matrix = np.zeros((5, 5), dtype=float)
    augmented_matrix[:4, :4] = A + KQ * (
        elevator_column @ (pitch_rate_output + KTHETA * theta_output)
    )
    augmented_matrix[:4, 4:5] = -KQ * integral_gain * elevator_column
    augmented_matrix[4:5, :4] = -theta_output

    command_vector = np.empty(5, dtype=float)
    command_vector[:4] = (
        -KQ * KTHETA * THETA_COMMAND * elevator_column[:, 0]
    )
    command_vector[4] = THETA_COMMAND
    return augmented_matrix, command_vector


def _simulate_augmented(A, B, integral_gain):
    augmented_matrix, command_vector = _augmented_system(A, B, integral_gain)
    number_of_steps = round(DURATION / DT)
    times = np.linspace(0.0, DURATION, number_of_steps + 1)
    states = np.empty((number_of_steps + 1, 5), dtype=float)
    states[0] = 0.0

    def derivative(state):
        return augmented_matrix @ state + command_vector

    for step in range(number_of_steps):
        state = states[step]
        k1 = derivative(state)
        k2 = derivative(state + 0.5 * DT * k1)
        k3 = derivative(state + 0.5 * DT * k2)
        k4 = derivative(state + DT * k3)
        states[step + 1] = state + (DT / 6.0) * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )

    return times, states


def _settling_time(times, attitude_error):
    tolerance = 0.02 * abs(THETA_COMMAND)
    outside_band = np.flatnonzero(np.abs(attitude_error) > tolerance)
    if outside_band.size == 0:
        return float(times[0])
    last_outside = outside_band[-1]
    if last_outside == times.size - 1:
        return None
    return float(times[last_outside + 1])


def _report_metrics(integral_gain, times, states, elevator_perturbation):
    delta_theta = states[:, 2]
    attitude_error = THETA_COMMAND - delta_theta
    final_theta = np.rad2deg(delta_theta[-1])
    final_error = np.rad2deg(attitude_error[-1])
    maximum_theta = np.rad2deg(np.max(delta_theta))
    command_degrees = np.rad2deg(THETA_COMMAND)
    percent_overshoot = max(
        0.0,
        100.0 * (maximum_theta - command_degrees) / abs(command_degrees),
    )
    maximum_elevator = np.max(np.abs(elevator_perturbation))
    maximum_integral_state = np.max(np.abs(states[:, 4]))
    settling_time = _settling_time(times, attitude_error)

    print(f"Ki = {integral_gain:g} 1/s^2")
    print(f"  Final Delta theta: {final_theta:.6f} deg")
    print(f"  Final attitude error: {final_error:.6f} deg")
    print(f"  Maximum Delta theta: {maximum_theta:.6f} deg")
    print(f"  Percent overshoot: {percent_overshoot:.3f}%")
    print(f"  Maximum absolute elevator perturbation: {maximum_elevator:.6f} deg")
    print(f"  Maximum absolute integral state xi: {maximum_integral_state:.6f} rad*s")
    if settling_time is None:
        print("  2% settling time: does not enter and remain inside the band")
    else:
        print(f"  Approximate 2% settling time: {settling_time:.2f} s")


def create_comparison_figure():
    """Build and return the candidate-integral-gain response comparison."""
    trim = trim_straight_level(
        true_airspeed=TRUE_AIRSPEED,
        altitude_m=ALTITUDE_M,
        cg_fraction=CG_FRACTION,
    )
    if not trim["success"]:
        raise RuntimeError(f"Unable to obtain the reference trim: {trim['message']}")

    alpha = np.deg2rad(trim["alpha_deg"])
    trim_state = np.array([trim["true_airspeed"], alpha, alpha, 0.0])
    trim_controls = np.array([trim["throttle"], trim["elevator_deg"]])
    A, B = linearize_longitudinal(
        trim_state,
        trim_controls,
        altitude_m=ALTITUDE_M,
        cg_fraction=CG_FRACTION,
    )

    figure, axes = plt.subplots(4, 1, sharex=True, figsize=(10.0, 12.0))
    for integral_gain in INTEGRAL_GAINS:
        times, states = _simulate_augmented(A, B, integral_gain)
        attitude_error = THETA_COMMAND - states[:, 2]
        q_command = KTHETA * attitude_error + integral_gain * states[:, 4]
        elevator_perturbation = KQ * (states[:, 3] - q_command)
        label = f"Ki = {integral_gain:g}"

        axes[0].plot(times, np.rad2deg(states[:, 2]), label=label)
        axes[1].plot(times, np.rad2deg(states[:, 3]), label=label)
        axes[2].plot(times, np.rad2deg(states[:, 1]), label=label)
        axes[3].plot(times, elevator_perturbation, label=label)
        _report_metrics(integral_gain, times, states, elevator_perturbation)

    axes[0].axhline(
        np.rad2deg(THETA_COMMAND),
        color="black",
        linestyle="--",
        label="Theta command",
    )
    axes[0].set_ylabel(r"$\Delta \theta$ [deg]")
    axes[1].set_ylabel(r"$\Delta q$ [deg/s]")
    axes[2].set_ylabel(r"$\Delta \alpha$ [deg]")
    axes[3].set_ylabel(r"$\Delta$ elevator [deg]")
    axes[3].set_xlabel("Time [s]")

    for axis in axes:
        axis.grid(True)
        axis.legend()

    figure.suptitle(
        "Linear F-16 Pitch-Attitude PI Gain Comparison\n"
        r"$K_q = 5$ deg/(rad/s), $K_\theta = 0.5$ 1/s, "
        r"$\theta_{cmd} = 5$ deg"
    )
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    create_comparison_figure()
    plt.show()
