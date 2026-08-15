"""Compare linear and nonlinear F-16 bank-angle maneuver responses."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.f16sim.attitude import quaternion_normalize
from src.f16sim.lateral_linearization import linearize_lateral
from src.f16sim.linear_response import simulate_linear_longitudinal
from src.f16sim.parameters import FT_TO_METER
from src.f16sim.simulation import simulate_f16_feedback
from src.f16sim.trim import trim_straight_level


TRUE_AIRSPEED = 502.0 * FT_TO_METER
ALTITUDE_M = 0.0
CG_FRACTION = 0.30
KR = 50.0
KP = 5.0
KPHI = 1.0
DURATION = 30.0
DT = 0.01


def _roll_angle(quaternion):
    q0, q1, q2, q3 = quaternion_normalize(quaternion)
    numerator = 2.0 * (q0 * q1 + q2 * q3)
    denominator = 1.0 - 2.0 * (q1**2 + q2**2)
    return np.arctan2(numerator, denominator)


def _bank_command(time):
    return np.deg2rad(20.0) if 2.0 <= time < 12.0 else 0.0


def _extract_nonlinear_lateral(states, trim):
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


def _print_metrics(name, times, lateral, aileron, rudder):
    command_interval = (times >= 2.0) & (times < 12.0)
    before_removal = np.searchsorted(times, 12.0) - 1
    print(name)
    print(
        "  maximum bank angle during command: "
        f"{np.rad2deg(np.max(lateral[command_interval, 1])):.8f} deg"
    )
    print(
        "  bank angle immediately before command removal: "
        f"{np.rad2deg(lateral[before_removal, 1]):.8f} deg"
    )
    print(
        f"  maximum absolute p: "
        f"{np.rad2deg(np.max(np.abs(lateral[:, 2]))):.8f} deg/s"
    )
    print(
        f"  maximum absolute beta: "
        f"{np.rad2deg(np.max(np.abs(lateral[:, 0]))):.8f} deg"
    )
    print(
        f"  maximum absolute r: "
        f"{np.rad2deg(np.max(np.abs(lateral[:, 3]))):.8f} deg/s"
    )
    print(f"  maximum absolute aileron command: {np.max(np.abs(aileron)):.8f} deg")
    print(f"  maximum absolute rudder command: {np.max(np.abs(rudder)):.8f} deg")
    print(
        "  Delta phi at end of simulation: "
        f"{np.rad2deg(lateral[-1, 1]):.8f} deg"
    )


def create_bank_maneuver_figure():
    """Run the selected bank controller and return its comparison figure."""
    trim = trim_straight_level(
        true_airspeed=TRUE_AIRSPEED,
        altitude_m=ALTITUDE_M,
        cg_fraction=CG_FRACTION,
    )
    if not trim["success"]:
        raise RuntimeError(f"Unable to obtain the reference trim: {trim['message']}")

    A_lat, B_lat = linearize_lateral(
        trim["state"],
        trim["throttle"],
        trim["elevator_deg"],
        cg_fraction=CG_FRACTION,
    )
    bank_output = np.array([[0.0, 1.0, 0.0, 0.0]])
    roll_rate_output = np.array([[0.0, 0.0, 1.0, 0.0]])
    yaw_rate_output = np.array([[0.0, 0.0, 0.0, 1.0]])
    closed_loop_A = (
        A_lat
        + KR * (B_lat[:, 1:2] @ yaw_rate_output)
        + KP
        * (B_lat[:, 0:1] @ (roll_rate_output + KPHI * bank_output))
    )

    def linear_command_control(time):
        return np.array([-KP * KPHI * _bank_command(time), 0.0])

    linear_times, linear_lateral = simulate_linear_longitudinal(
        closed_loop_A,
        B_lat,
        initial_perturbation=np.zeros(4),
        duration=DURATION,
        dt=DT,
        control_perturbation=linear_command_control,
    )

    trim_phi = _roll_angle(trim["state"][6:10])

    def nonlinear_control_law(time, state):
        phi = _roll_angle(state[6:10])
        p_perturbation = state[10] - trim["state"][10]
        r_perturbation = state[12] - trim["state"][12]
        phi_command = trim_phi + _bank_command(time)
        p_command = KPHI * (phi_command - phi)
        aileron = KP * (p_perturbation - p_command)
        rudder = KR * r_perturbation
        return [trim["throttle"], trim["elevator_deg"], aileron, rudder]

    nonlinear_times, nonlinear_full_states = simulate_f16_feedback(
        initial_state=trim["state"],
        duration=DURATION,
        dt=DT,
        control_law=nonlinear_control_law,
        cg_fraction=CG_FRACTION,
    )
    nonlinear_lateral = _extract_nonlinear_lateral(nonlinear_full_states, trim)

    command = np.array([_bank_command(time) for time in linear_times])
    linear_p_command = KPHI * (command - linear_lateral[:, 1])
    linear_aileron = KP * (linear_lateral[:, 2] - linear_p_command)
    linear_rudder = KR * linear_lateral[:, 3]
    nonlinear_p_command = KPHI * (command - nonlinear_lateral[:, 1])
    nonlinear_aileron = KP * (
        nonlinear_lateral[:, 2] - nonlinear_p_command
    )
    nonlinear_rudder = KR * nonlinear_lateral[:, 3]

    _print_metrics(
        "Linear model",
        linear_times,
        linear_lateral,
        linear_aileron,
        linear_rudder,
    )
    _print_metrics(
        "Nonlinear model",
        nonlinear_times,
        nonlinear_lateral,
        nonlinear_aileron,
        nonlinear_rudder,
    )
    difference = np.abs(linear_lateral - nonlinear_lateral)
    print(
        "Maximum absolute linear/nonlinear phi difference: "
        f"{np.rad2deg(np.max(difference[:, 1])):.8f} deg"
    )
    print(
        "Maximum absolute linear/nonlinear beta difference: "
        f"{np.rad2deg(np.max(difference[:, 0])):.8f} deg"
    )
    states_finite = bool(np.all(np.isfinite(nonlinear_full_states)))
    controls_finite = bool(
        np.all(np.isfinite(nonlinear_aileron))
        and np.all(np.isfinite(nonlinear_rudder))
    )
    print(f"All nonlinear states remained finite: {states_finite}")
    print(f"All nonlinear controls remained finite: {controls_finite}")

    figure, axes = plt.subplots(6, 1, sharex=True, figsize=(10.0, 15.0))
    axes[0].plot(linear_times, np.rad2deg(linear_lateral[:, 1]), label="Linear")
    axes[0].plot(
        nonlinear_times,
        np.rad2deg(nonlinear_lateral[:, 1]),
        label="Nonlinear",
    )
    axes[0].plot(
        linear_times,
        np.rad2deg(command),
        color="black",
        linestyle="--",
        label="Delta phi command",
    )
    axes[0].set_ylabel(r"$\Delta \phi$ [deg]")

    axes[1].plot(linear_times, np.rad2deg(linear_lateral[:, 2]), label="Linear p")
    axes[1].plot(
        nonlinear_times,
        np.rad2deg(nonlinear_lateral[:, 2]),
        label="Nonlinear p",
    )
    axes[1].plot(
        linear_times,
        np.rad2deg(linear_p_command),
        linestyle=":",
        label="Linear p command",
    )
    axes[1].plot(
        nonlinear_times,
        np.rad2deg(nonlinear_p_command),
        linestyle=":",
        label="Nonlinear p command",
    )
    axes[1].set_ylabel(r"$\Delta p$ [deg/s]")

    plots = (
        (2, 0, r"$\Delta \beta$ [deg]"),
        (3, 3, r"$\Delta r$ [deg/s]"),
    )
    for axis_index, state_index, label in plots:
        axes[axis_index].plot(
            linear_times,
            np.rad2deg(linear_lateral[:, state_index]),
            label="Linear",
        )
        axes[axis_index].plot(
            nonlinear_times,
            np.rad2deg(nonlinear_lateral[:, state_index]),
            label="Nonlinear",
        )
        axes[axis_index].set_ylabel(label)

    axes[4].plot(linear_times, linear_aileron, label="Linear")
    axes[4].plot(nonlinear_times, nonlinear_aileron, label="Nonlinear")
    axes[4].set_ylabel(r"$\Delta$ aileron [deg]")
    axes[5].plot(linear_times, linear_rudder, label="Linear")
    axes[5].plot(nonlinear_times, nonlinear_rudder, label="Nonlinear")
    axes[5].set_ylabel(r"$\Delta$ rudder [deg]")
    axes[5].set_xlabel("Time [s]")

    for axis in axes:
        axis.grid(True)
        axis.legend()
    figure.suptitle(
        "F-16 Linear and Nonlinear Bank-Angle Maneuver\n"
        r"$K_r=50$, $K_p=5$, $K_\phi=1$"
    )
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    create_bank_maneuver_figure()
    plt.show()
