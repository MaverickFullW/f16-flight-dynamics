"""Compare linear and nonlinear F-16 responses with a rudder yaw damper."""

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
from src.f16sim.simulation import simulate_f16, simulate_f16_feedback
from src.f16sim.trim import trim_straight_level


TRUE_AIRSPEED = 502.0 * FT_TO_METER
ALTITUDE_M = 0.0
CG_FRACTION = 0.30
KR = 50.0
DURATION = 15.0
DT = 0.01


def _pitch_angle(quaternion):
    q0, q1, q2, q3 = quaternion_normalize(quaternion)
    sin_theta = 2.0 * (q0 * q2 - q3 * q1)
    return np.arcsin(np.clip(sin_theta, -1.0, 1.0))


def _roll_angle(quaternion):
    q0, q1, q2, q3 = quaternion_normalize(quaternion)
    numerator = 2.0 * (q0 * q1 + q2 * q3)
    denominator = 1.0 - 2.0 * (q1**2 + q2**2)
    return np.arctan2(numerator, denominator)


def _initial_beta_state(trim):
    beta = np.deg2rad(1.0)
    alpha = np.deg2rad(trim["alpha_deg"])
    theta = _pitch_angle(trim["state"][6:10])
    state = trim["state"].copy()
    longitudinal_speed = trim["true_airspeed"] * np.cos(beta)
    state[3] = longitudinal_speed * np.cos(alpha)
    state[4] = trim["true_airspeed"] * np.sin(beta)
    state[5] = longitudinal_speed * np.sin(alpha)
    state[6:10] = euler_to_quaternion(0.0, theta, 0.0)
    state[10:13] = 0.0
    return state


def _extract_lateral(states, trim):
    true_airspeed = np.linalg.norm(states[:, 3:6], axis=1)
    beta = np.arcsin(np.clip(states[:, 4] / true_airspeed, -1.0, 1.0))
    phi = np.unwrap(
        np.array([_roll_angle(quaternion) for quaternion in states[:, 6:10]])
    )
    trim_phi = _roll_angle(trim["state"][6:10])
    return np.column_stack(
        (
            beta,
            phi - trim_phi,
            states[:, 10] - trim["state"][10],
            states[:, 12] - trim["state"][12],
        )
    )


def _dutch_roll_damping(matrix):
    complex_poles = [
        pole for pole in np.linalg.eigvals(matrix) if pole.imag > 1e-8
    ]
    if not complex_poles:
        return None
    pole = max(complex_poles, key=lambda value: abs(value.imag))
    return float(-pole.real / abs(pole))


def _print_case_metrics(name, lateral_states):
    print(name)
    print(
        "  maximum absolute beta: "
        f"{np.rad2deg(np.max(np.abs(lateral_states[:, 0]))):.8f} deg"
    )
    print(
        "  maximum absolute r: "
        f"{np.rad2deg(np.max(np.abs(lateral_states[:, 3]))):.8f} deg/s"
    )


def create_yaw_damper_response_figure():
    """Run all four yaw-damper cases and return their comparison figure."""
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
    rudder_column = B_lat[:, 1:2]
    yaw_rate_output = np.array([[0.0, 0.0, 0.0, 1.0]])
    closed_loop_A = A_lat + KR * (rudder_column @ yaw_rate_output)
    initial_lateral = np.array([np.deg2rad(1.0), 0.0, 0.0, 0.0])

    linear_times, linear_open = simulate_linear_longitudinal(
        A_lat,
        B_lat,
        initial_perturbation=initial_lateral,
        duration=DURATION,
        dt=DT,
    )
    _, linear_closed = simulate_linear_longitudinal(
        closed_loop_A,
        B_lat,
        initial_perturbation=initial_lateral,
        duration=DURATION,
        dt=DT,
    )

    nonlinear_initial = _initial_beta_state(trim)
    nonlinear_times, nonlinear_open_full = simulate_f16(
        initial_state=nonlinear_initial,
        duration=DURATION,
        dt=DT,
        throttle=trim["throttle"],
        elevator_deg=trim["elevator_deg"],
        aileron_deg=0.0,
        rudder_deg=0.0,
        cg_fraction=CG_FRACTION,
    )

    def yaw_damper(time, state):
        yaw_rate_perturbation = state[12] - trim["state"][12]
        return [
            trim["throttle"],
            trim["elevator_deg"],
            0.0,
            KR * yaw_rate_perturbation,
        ]

    _, nonlinear_closed_full = simulate_f16_feedback(
        initial_state=nonlinear_initial,
        duration=DURATION,
        dt=DT,
        control_law=yaw_damper,
        cg_fraction=CG_FRACTION,
    )
    nonlinear_open = _extract_lateral(nonlinear_open_full, trim)
    nonlinear_closed = _extract_lateral(nonlinear_closed_full, trim)
    nonlinear_rudder = KR * nonlinear_closed[:, 3]

    print(
        "Open-loop Dutch-roll damping ratio: "
        f"{_dutch_roll_damping(A_lat):.9f}"
    )
    print(
        "Closed-loop Dutch-roll damping ratio: "
        f"{_dutch_roll_damping(closed_loop_A):.9f}"
    )
    _print_case_metrics("Linear open-loop", linear_open)
    _print_case_metrics("Linear Kr=50", linear_closed)
    _print_case_metrics("Nonlinear open-loop", nonlinear_open)
    _print_case_metrics("Nonlinear Kr=50", nonlinear_closed)
    print(
        "Maximum absolute rudder command: "
        f"{np.max(np.abs(nonlinear_rudder)):.8f} deg"
    )
    closed_difference = np.abs(linear_closed - nonlinear_closed)
    print(
        "Maximum closed-loop linear/nonlinear beta difference: "
        f"{np.rad2deg(np.max(closed_difference[:, 0])):.8f} deg"
    )
    print(
        "Maximum closed-loop linear/nonlinear r difference: "
        f"{np.rad2deg(np.max(closed_difference[:, 3])):.8f} deg/s"
    )
    nonlinear_states_finite = np.all(np.isfinite(nonlinear_open_full)) and np.all(
        np.isfinite(nonlinear_closed_full)
    )
    nonlinear_controls_finite = bool(np.all(np.isfinite(nonlinear_rudder)))
    print(f"All nonlinear states remained finite: {nonlinear_states_finite}")
    print(f"All nonlinear controls remained finite: {nonlinear_controls_finite}")

    figure, axes = plt.subplots(5, 1, sharex=True, figsize=(10.0, 13.0))
    labels = (
        r"$\Delta \beta$ [deg]",
        r"$\Delta \phi$ [deg]",
        r"$\Delta p$ [deg/s]",
        r"$\Delta r$ [deg/s]",
    )
    for index, label in enumerate(labels):
        axes[index].plot(
            linear_times,
            np.rad2deg(linear_open[:, index]),
            label="Linear open-loop",
        )
        axes[index].plot(
            linear_times,
            np.rad2deg(linear_closed[:, index]),
            label="Linear Kr=50",
        )
        axes[index].plot(
            nonlinear_times,
            np.rad2deg(nonlinear_open[:, index]),
            label="Nonlinear open-loop",
        )
        axes[index].plot(
            nonlinear_times,
            np.rad2deg(nonlinear_closed[:, index]),
            label="Nonlinear Kr=50",
        )
        axes[index].set_ylabel(label)
        axes[index].legend()

    axes[4].plot(
        nonlinear_times,
        nonlinear_rudder,
        color="C3",
        label="Nonlinear yaw-damper command",
    )
    axes[4].set_ylabel(r"$\Delta$ rudder [deg]")
    axes[4].set_xlabel("Time [s]")
    axes[4].legend()
    for axis in axes:
        axis.grid(True)

    figure.suptitle(
        "F-16 Linear and Nonlinear Rudder Yaw-Damper Response\n"
        r"$K_r = 50$ deg/(rad/s), $\delta_r = K_r \Delta r$"
    )
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    create_yaw_damper_response_figure()
    plt.show()
