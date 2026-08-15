"""Compare reduced linear and full nonlinear lateral free responses."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.f16sim.attitude import euler_to_quaternion, quaternion_normalize
from src.f16sim.lateral_linearization import linearize_lateral
from src.f16sim.linear_response import simulate_linear_longitudinal
from src.f16sim.parameters import FT_TO_METER
from src.f16sim.simulation import simulate_f16
from src.f16sim.trim import trim_straight_level


TRUE_AIRSPEED = 502.0 * FT_TO_METER
ALTITUDE_M = 0.0
CG_FRACTION = 0.30
EXPECTED_DUTCH_ROLL = -0.4399356353 + 3.2203339690j
EXPECTED_DUTCH_ROLL_WN = 3.2502452576
EXPECTED_DUTCH_ROLL_ZETA = 0.1353545965
EXPECTED_SPIRAL = -0.0128370271
EXPECTED_SPIRAL_TAU = 77.899656
STATE_LABELS = ("beta", "phi", "p", "r")
STATE_AXIS_LABELS = (
    r"$\Delta \beta$ [deg]",
    r"$\Delta \phi$ [deg]",
    r"$\Delta p$ [deg/s]",
    r"$\Delta r$ [deg/s]",
)


def _pitch_angle(quaternion):
    q0, q1, q2, q3 = quaternion_normalize(quaternion)
    sin_theta = 2.0 * (q0 * q2 - q3 * q1)
    return np.arcsin(np.clip(sin_theta, -1.0, 1.0))


def _roll_angle(quaternion):
    q0, q1, q2, q3 = quaternion_normalize(quaternion)
    numerator = 2.0 * (q0 * q1 + q2 * q3)
    denominator = 1.0 - 2.0 * (q1**2 + q2**2)
    return np.arctan2(numerator, denominator)


def _nonlinear_initial_state(trim, beta, phi):
    state = trim["state"].copy()
    alpha = np.deg2rad(trim["alpha_deg"])
    theta = _pitch_angle(trim["state"][6:10])
    longitudinal_speed = trim["true_airspeed"] * np.cos(beta)
    state[3] = longitudinal_speed * np.cos(alpha)
    state[4] = trim["true_airspeed"] * np.sin(beta)
    state[5] = longitudinal_speed * np.sin(alpha)
    state[6:10] = euler_to_quaternion(phi, theta, 0.0)
    state[10:13] = 0.0
    return state


def _extract_nonlinear_lateral(states, trim):
    true_airspeed = np.linalg.norm(states[:, 3:6], axis=1)
    beta = np.arcsin(np.clip(states[:, 4] / true_airspeed, -1.0, 1.0))
    phi = np.unwrap(
        np.array([_roll_angle(quaternion) for quaternion in states[:, 6:10]])
    )
    trim_phi = _roll_angle(trim["state"][6:10])
    return np.column_stack(
        (beta, phi - trim_phi, states[:, 10], states[:, 12])
    )


def _run_experiment(trim, A_lat, B_lat, initial_lateral, duration, dt):
    linear_times, linear_states = simulate_linear_longitudinal(
        A_lat,
        B_lat,
        initial_perturbation=initial_lateral,
        duration=duration,
        dt=dt,
    )
    nonlinear_initial = _nonlinear_initial_state(
        trim,
        beta=initial_lateral[0],
        phi=initial_lateral[1],
    )
    nonlinear_times, nonlinear_full_states = simulate_f16(
        initial_state=nonlinear_initial,
        duration=duration,
        dt=dt,
        throttle=trim["throttle"],
        elevator_deg=trim["elevator_deg"],
        aileron_deg=0.0,
        rudder_deg=0.0,
        cg_fraction=CG_FRACTION,
    )
    nonlinear_states = _extract_nonlinear_lateral(nonlinear_full_states, trim)
    return linear_times, linear_states, nonlinear_times, nonlinear_states


def _print_response_metrics(name, linear_states, nonlinear_states):
    linear_degrees = np.rad2deg(linear_states)
    nonlinear_degrees = np.rad2deg(nonlinear_states)
    difference_degrees = np.abs(linear_degrees - nonlinear_degrees)
    print(f"\n{name}")
    for index, label in enumerate(STATE_LABELS):
        rate_suffix = "/s" if label in ("p", "r") else ""
        print(
            f"  maximum absolute {label} (linear): "
            f"{np.max(np.abs(linear_degrees[:, index])):.8f} deg{rate_suffix}"
        )
        print(
            f"  maximum absolute {label} (nonlinear): "
            f"{np.max(np.abs(nonlinear_degrees[:, index])):.8f} deg{rate_suffix}"
        )
        print(
            f"  maximum absolute linear/nonlinear {label} difference: "
            f"{np.max(difference_degrees[:, index]):.8f} deg{rate_suffix}"
        )


def _estimate_dutch_roll(times, beta):
    peak_indices = np.flatnonzero(
        (beta[1:-1] > beta[:-2]) & (beta[1:-1] >= beta[2:])
    ) + 1
    peak_indices = peak_indices[beta[peak_indices] > 0.0][:5]
    if peak_indices.size < 2:
        print("\nDutch-roll response estimate: insufficient positive beta peaks")
        return

    periods = np.diff(times[peak_indices])
    peak_amplitudes = beta[peak_indices]
    logarithmic_decrements = np.log(
        peak_amplitudes[:-1] / peak_amplitudes[1:]
    )
    period = float(np.mean(periods))
    damped_frequency = 2.0 * np.pi / period
    logarithmic_decrement = float(np.mean(logarithmic_decrements))
    damping_ratio = logarithmic_decrement / np.sqrt(
        (2.0 * np.pi) ** 2 + logarithmic_decrement**2
    )

    print("\nDutch-roll estimates from linear beta peaks")
    print(f"  oscillation period: {period:.8f} s")
    print(f"  damped frequency: {damped_frequency:.8f} rad/s")
    print(f"  logarithmic decrement: {logarithmic_decrement:.8f}")
    print(f"  estimated damping ratio: {damping_ratio:.8f}")
    print("  eigenanalysis reference:")
    print(f"    lambda_DR: {EXPECTED_DUTCH_ROLL} 1/s")
    print(f"    wn: {EXPECTED_DUTCH_ROLL_WN:.10f} rad/s")
    print(f"    zeta: {EXPECTED_DUTCH_ROLL_ZETA:.10f}")
    print("  estimate minus eigenanalysis:")
    print(
        f"    damped-frequency difference: "
        f"{damped_frequency - EXPECTED_DUTCH_ROLL.imag:.8f} rad/s"
    )
    print(
        f"    damping-ratio difference: "
        f"{damping_ratio - EXPECTED_DUTCH_ROLL_ZETA:.8f}"
    )


def _discuss_spiral_time_scale(times, phi):
    tail = (times >= 40.0) & (np.abs(phi) > 1e-10)
    print("\nSlow bank-angle time-scale check")
    print(f"  identified spiral pole: {EXPECTED_SPIRAL:.10f} 1/s")
    print(f"  identified spiral time constant: {EXPECTED_SPIRAL_TAU:.6f} s")
    if np.count_nonzero(tail) < 2:
        print("  observed tail: insufficient nonzero samples for an estimate")
        return

    observed_rate = np.polyfit(times[tail], np.log(np.abs(phi[tail])), 1)[0]
    if observed_rate >= 0.0:
        print(f"  observed tail rate: {observed_rate:.10f} 1/s")
        print("  discussion: the fitted tail does not show stable exponential decay")
        return
    observed_tau = -1.0 / observed_rate
    relative_difference = abs(observed_tau - EXPECTED_SPIRAL_TAU) / EXPECTED_SPIRAL_TAU
    print(f"  observed bank-angle tail rate: {observed_rate:.10f} 1/s")
    print(f"  observed bank-angle time constant: {observed_tau:.6f} s")
    if relative_difference <= 0.2:
        print("  discussion: the observed slow response is consistent with the spiral mode")
    else:
        print(
            "  discussion: the finite-response tail differs from the isolated spiral "
            "time scale, indicating residual modal/nonlinear contributions"
        )


def _plot_experiment(name, linear_times, linear_states, nonlinear_times, nonlinear_states):
    figure, axes = plt.subplots(4, 1, sharex=True, figsize=(9.0, 11.0))
    for index, (axis, label) in enumerate(zip(axes, STATE_AXIS_LABELS)):
        axis.plot(
            linear_times,
            np.rad2deg(linear_states[:, index]),
            label="Linear",
        )
        axis.plot(
            nonlinear_times,
            np.rad2deg(nonlinear_states[:, index]),
            label="Nonlinear",
        )
        axis.set_ylabel(label)
        axis.grid(True)
        axis.legend()
    axes[-1].set_xlabel("Time [s]")
    figure.suptitle(name)
    figure.tight_layout()
    return figure


def create_comparison_figures():
    """Run both lateral experiments and return their comparison figures."""
    trim = trim_straight_level(
        true_airspeed=TRUE_AIRSPEED,
        altitude_m=ALTITUDE_M,
        cg_fraction=CG_FRACTION,
    )
    if not trim["success"]:
        raise RuntimeError(f"Unable to obtain the reference trim: {trim['message']}")

    A_lat, B_lat = linearize_lateral(
        trim["state"],
        throttle=trim["throttle"],
        elevator_deg=trim["elevator_deg"],
        cg_fraction=CG_FRACTION,
    )

    dutch_roll = _run_experiment(
        trim,
        A_lat,
        B_lat,
        initial_lateral=np.array([np.deg2rad(1.0), 0.0, 0.0, 0.0]),
        duration=20.0,
        dt=0.01,
    )
    spiral = _run_experiment(
        trim,
        A_lat,
        B_lat,
        initial_lateral=np.array([0.0, np.deg2rad(5.0), 0.0, 0.0]),
        duration=120.0,
        dt=0.02,
    )

    _print_response_metrics("Experiment 1 - Dutch-roll excitation", dutch_roll[1], dutch_roll[3])
    _estimate_dutch_roll(dutch_roll[0], dutch_roll[1][:, 0])
    _print_response_metrics("Experiment 2 - roll/spiral excitation", spiral[1], spiral[3])
    _discuss_spiral_time_scale(spiral[0], spiral[1][:, 1])

    dutch_figure = _plot_experiment(
        "Dutch-Roll Free Response at 502 ft/s",
        *dutch_roll,
    )
    spiral_figure = _plot_experiment(
        "Roll/Spiral Free Response at 502 ft/s",
        *spiral,
    )
    return dutch_figure, spiral_figure


if __name__ == "__main__":
    create_comparison_figures()
    plt.show()
