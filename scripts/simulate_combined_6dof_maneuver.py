"""Run a combined controlled pitch-and-bank maneuver on the nonlinear F-16."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.f16sim.attitude import quaternion_normalize, quaternion_to_dcm
from src.f16sim.controllers import pitch_attitude_control
from src.f16sim.parameters import FT_TO_METER
from src.f16sim.simulation import simulate_f16_feedback
from src.f16sim.trim import trim_straight_level


TRUE_AIRSPEED = 502.0 * FT_TO_METER
ALTITUDE_M = 0.0
CG_FRACTION = 0.30
KQ = 5.0
KTHETA = 0.5
KR = 50.0
KP = 5.0
KPHI = 1.0
DURATION = 30.0
DT = 0.01


def _euler_roll_pitch(quaternion):
    q0, q1, q2, q3 = quaternion_normalize(quaternion)
    roll = np.arctan2(
        2.0 * (q0 * q1 + q2 * q3),
        1.0 - 2.0 * (q1**2 + q2**2),
    )
    sin_pitch = 2.0 * (q0 * q2 - q3 * q1)
    pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))
    return roll, pitch


def _command_perturbations(time):
    if 2.0 <= time < 12.0:
        return np.deg2rad(5.0), np.deg2rad(20.0)
    return 0.0, 0.0


def _body_axes_plot_coordinates(quaternion):
    """Return body axes in North/East/altitude-up plot coordinates."""
    body_to_ned = quaternion_to_dcm(quaternion).T
    body_axes = body_to_ned.copy()
    body_axes[2, :] *= -1.0
    return body_axes


def _set_equal_3d_limits(axis, north_ft, east_ft, altitude_ft):
    centers = np.array(
        [
            0.5 * (np.min(north_ft) + np.max(north_ft)),
            0.5 * (np.min(east_ft) + np.max(east_ft)),
            0.5 * (np.min(altitude_ft) + np.max(altitude_ft)),
        ]
    )
    spans = np.array(
        [np.ptp(north_ft), np.ptp(east_ft), np.ptp(altitude_ft)]
    )
    half_range = 0.55 * max(np.max(spans), 1.0)
    axis.set_xlim(centers[0] - half_range, centers[0] + half_range)
    axis.set_ylim(centers[1] - half_range, centers[1] + half_range)
    axis.set_zlim(centers[2] - half_range, centers[2] + half_range)
    axis.set_box_aspect((1.0, 1.0, 1.0))


def simulate_combined_maneuver():
    """Run the validated combined maneuver and return its sampled histories."""
    trim = trim_straight_level(
        true_airspeed=TRUE_AIRSPEED,
        altitude_m=ALTITUDE_M,
        cg_fraction=CG_FRACTION,
    )
    if not trim["success"]:
        raise RuntimeError(f"Unable to obtain the reference trim: {trim['message']}")

    trim_phi, trim_theta = _euler_roll_pitch(trim["state"][6:10])
    trim_p, trim_q, trim_r = trim["state"][10:13]

    def control_law(time, state):
        phi, theta = _euler_roll_pitch(state[6:10])
        delta_theta_command, delta_phi_command = _command_perturbations(time)
        theta_command = trim_theta + delta_theta_command
        phi_command = trim_phi + delta_phi_command

        _, elevator_perturbation = pitch_attitude_control(
            theta_command,
            theta,
            state[11] - trim_q,
            Kq=KQ,
            Ktheta=KTHETA,
        )
        p_command = KPHI * (phi_command - phi)
        aileron = KP * ((state[10] - trim_p) - p_command)
        rudder = KR * (state[12] - trim_r)
        return [
            trim["throttle"],
            trim["elevator_deg"] + elevator_perturbation,
            aileron,
            rudder,
        ]

    times, states = simulate_f16_feedback(
        initial_state=trim["state"],
        duration=DURATION,
        dt=DT,
        control_law=control_law,
        cg_fraction=CG_FRACTION,
    )

    command_history = np.array(
        [_command_perturbations(time) for time in times]
    )
    controls = np.array(
        [control_law(time, state) for time, state in zip(times, states)]
    )
    return {
        "times": times,
        "states": states,
        "controls": controls,
        "command_history": command_history,
        "trim": trim,
    }


def create_combined_maneuver_figures():
    """Simulate the combined maneuver and return response and trajectory figures."""
    result = simulate_combined_maneuver()
    times = result["times"]
    states = result["states"]
    controls = result["controls"]
    command_history = result["command_history"]
    trim = result["trim"]
    trim_phi, trim_theta = _euler_roll_pitch(trim["state"][6:10])
    trim_p, trim_q, trim_r = trim["state"][10:13]

    euler_angles = np.array(
        [_euler_roll_pitch(quaternion) for quaternion in states[:, 6:10]]
    )
    phi = np.unwrap(euler_angles[:, 0])
    theta = np.unwrap(euler_angles[:, 1])
    delta_phi = phi - trim_phi
    delta_theta = theta - trim_theta
    delta_p = states[:, 10] - trim_p
    delta_q = states[:, 11] - trim_q
    delta_r = states[:, 12] - trim_r

    delta_theta_command = command_history[:, 0]
    delta_phi_command = command_history[:, 1]
    q_command = KTHETA * (delta_theta_command - delta_theta)
    p_command = KPHI * (delta_phi_command - delta_phi)
    elevator_perturbation = KQ * (delta_q - q_command)
    aileron = KP * (delta_p - p_command)
    rudder = KR * delta_r

    velocity_body = states[:, 3:6]
    true_airspeed = np.linalg.norm(velocity_body, axis=1)
    beta = np.arcsin(
        np.clip(velocity_body[:, 1] / true_airspeed, -1.0, 1.0)
    )
    altitude = -states[:, 2]
    altitude_change = altitude - altitude[0]
    north = states[:, 0] - states[0, 0]
    east = states[:, 1] - states[0, 1]

    states_finite = bool(np.all(np.isfinite(states)))
    controls_finite = bool(np.all(np.isfinite(controls)))

    print(f"Maximum Delta theta: {np.rad2deg(np.max(delta_theta)):.8f} deg")
    print(f"Maximum Delta phi: {np.rad2deg(np.max(delta_phi)):.8f} deg")
    print(f"Maximum absolute beta: {np.rad2deg(np.max(np.abs(beta))):.8f} deg")
    print(f"Maximum absolute p: {np.rad2deg(np.max(np.abs(delta_p))):.8f} deg/s")
    print(f"Maximum absolute q: {np.rad2deg(np.max(np.abs(delta_q))):.8f} deg/s")
    print(f"Maximum absolute r: {np.rad2deg(np.max(np.abs(delta_r))):.8f} deg/s")
    print(
        "Maximum absolute elevator perturbation: "
        f"{np.max(np.abs(elevator_perturbation)):.8f} deg"
    )
    print(f"Maximum absolute aileron: {np.max(np.abs(aileron)):.8f} deg")
    print(f"Maximum absolute rudder: {np.max(np.abs(rudder)):.8f} deg")
    print(
        "True-airspeed range: "
        f"{np.min(true_airspeed) / FT_TO_METER:.8f} to "
        f"{np.max(true_airspeed) / FT_TO_METER:.8f} ft/s"
    )
    print(f"Total altitude change: {altitude_change[-1] / FT_TO_METER:.8f} ft")
    print(f"Final north displacement: {north[-1] / FT_TO_METER:.8f} ft")
    print(f"Final east displacement: {east[-1] / FT_TO_METER:.8f} ft")
    print(f"All states remained finite: {states_finite}")
    print(f"All controls remained finite: {controls_finite}")

    response_figure, axes = plt.subplots(7, 2, sharex=True, figsize=(14.0, 20.0))
    axes = axes.flat
    axes[0].plot(times, np.rad2deg(delta_theta), label="Delta theta")
    axes[0].plot(
        times,
        np.rad2deg(delta_theta_command),
        linestyle="--",
        label="Theta command",
    )
    axes[0].set_ylabel(r"$\Delta \theta$ [deg]")

    axes[1].plot(times, np.rad2deg(delta_phi), label="Delta phi")
    axes[1].plot(
        times,
        np.rad2deg(delta_phi_command),
        linestyle="--",
        label="Phi command",
    )
    axes[1].set_ylabel(r"$\Delta \phi$ [deg]")

    axes[2].plot(times, np.rad2deg(delta_q), label="Delta q")
    axes[2].plot(times, np.rad2deg(q_command), linestyle="--", label="q command")
    axes[2].set_ylabel(r"$\Delta q$ [deg/s]")

    axes[3].plot(times, np.rad2deg(delta_p), label="Delta p")
    axes[3].plot(times, np.rad2deg(p_command), linestyle="--", label="p command")
    axes[3].set_ylabel(r"$\Delta p$ [deg/s]")

    series = (
        (beta, r"$\beta$ [deg]", True),
        (delta_r, r"$\Delta r$ [deg/s]", True),
        (elevator_perturbation, r"$\Delta$ elevator [deg]", False),
        (aileron, r"$\Delta$ aileron [deg]", False),
        (rudder, r"$\Delta$ rudder [deg]", False),
        (true_airspeed / FT_TO_METER, "True airspeed [ft/s]", False),
        (altitude_change / FT_TO_METER, "Altitude change [ft]", False),
        (north / FT_TO_METER, "North position [ft]", False),
        (east / FT_TO_METER, "East position [ft]", False),
    )
    for axis, (values, label, convert_to_degrees) in zip(axes[4:13], series):
        plotted_values = np.rad2deg(values) if convert_to_degrees else values
        axis.plot(times, plotted_values)
        axis.set_ylabel(label)

    axes[13].axis("off")
    for axis in axes[:13]:
        axis.grid(True)
        axis.set_xlabel("Time [s]")
        if any(not line.get_label().startswith("_") for line in axis.lines):
            axis.legend()
    response_figure.suptitle("Combined Controlled Nonlinear F-16 6DoF Maneuver")
    response_figure.tight_layout()

    trajectory_figure, trajectory_axis = plt.subplots(figsize=(8.0, 7.0))
    trajectory_axis.plot(north / FT_TO_METER, east / FT_TO_METER)
    trajectory_axis.scatter(0.0, 0.0, marker="o", label="Start")
    trajectory_axis.scatter(
        north[-1] / FT_TO_METER,
        east[-1] / FT_TO_METER,
        marker="x",
        label="End",
    )
    trajectory_axis.set_xlabel("North position [ft]")
    trajectory_axis.set_ylabel("East position [ft]")
    trajectory_axis.set_title("Horizontal-Plane Trajectory")
    trajectory_axis.axis("equal")
    trajectory_axis.grid(True)
    trajectory_axis.legend()
    trajectory_figure.tight_layout()

    north_ft = north / FT_TO_METER
    east_ft = east / FT_TO_METER
    altitude_change_ft = altitude_change / FT_TO_METER
    sample_indices = np.unique(
        np.linspace(0, states.shape[0] - 1, 12, dtype=int)
    )
    trajectory_span = max(
        np.ptp(north_ft),
        np.ptp(east_ft),
        np.ptp(altitude_change_ft),
        1.0,
    )
    triad_scale = 0.04 * trajectory_span

    attitude_figure = plt.figure(figsize=(11.0, 8.0))
    attitude_axis = attitude_figure.add_subplot(111, projection="3d")
    attitude_axis.plot(
        north_ft,
        east_ft,
        altitude_change_ft,
        color="C0",
        label="Trajectory",
    )
    attitude_axis.scatter(
        north_ft[0],
        east_ft[0],
        altitude_change_ft[0],
        color="black",
        marker="o",
        label="Start",
    )
    attitude_axis.scatter(
        north_ft[-1],
        east_ft[-1],
        altitude_change_ft[-1],
        color="black",
        marker="x",
        label="End",
    )
    axis_colors = ("C3", "C2", "C4")
    axis_names = ("Body forward", "Body right", "Body down")
    for sample_number, index in enumerate(sample_indices):
        origin = np.array(
            [north_ft[index], east_ft[index], altitude_change_ft[index]]
        )
        body_axes = _body_axes_plot_coordinates(states[index, 6:10])
        for body_axis, color, name in zip(
            body_axes.T, axis_colors, axis_names
        ):
            endpoint = origin + triad_scale * body_axis
            attitude_axis.plot(
                [origin[0], endpoint[0]],
                [origin[1], endpoint[1]],
                [origin[2], endpoint[2]],
                color=color,
                linewidth=1.5,
                label=name if sample_number == 0 else None,
            )

    attitude_axis.set_xlabel("North [ft]")
    attitude_axis.set_ylabel("East [ft]")
    attitude_axis.set_zlabel("Altitude change [ft]")
    attitude_axis.set_title("Controlled F-16 6DoF Trajectory and Attitude")
    _set_equal_3d_limits(
        attitude_axis, north_ft, east_ft, altitude_change_ft
    )
    attitude_axis.legend()
    attitude_figure.tight_layout()

    trajectory_3d_figure = plt.figure(figsize=(10.0, 7.5))
    trajectory_3d_axis = trajectory_3d_figure.add_subplot(111, projection="3d")
    trajectory_3d_axis.plot(
        north_ft,
        east_ft,
        altitude_change_ft,
        color="C0",
        linewidth=2.0,
        label="Trajectory",
    )
    trajectory_3d_axis.scatter(
        north_ft[0],
        east_ft[0],
        altitude_change_ft[0],
        color="C2",
        marker="o",
        s=55,
        label="Start",
    )
    trajectory_3d_axis.scatter(
        north_ft[-1],
        east_ft[-1],
        altitude_change_ft[-1],
        color="C3",
        marker="x",
        s=65,
        label="End",
    )
    trajectory_3d_axis.set_xlabel("North [ft]")
    trajectory_3d_axis.set_ylabel("East [ft]")
    trajectory_3d_axis.set_zlabel("Altitude change [ft]")
    trajectory_3d_axis.set_title("Controlled F-16 6DoF Trajectory")
    _set_equal_3d_limits(
        trajectory_3d_axis, north_ft, east_ft, altitude_change_ft
    )
    trajectory_3d_axis.legend()
    trajectory_3d_figure.tight_layout()

    initial_axes = _body_axes_plot_coordinates(states[0, 6:10])
    final_axes = _body_axes_plot_coordinates(states[-1, 6:10])
    forward_points_north = bool(initial_axes[0, 0] > 0.99)
    maximum_pitch_index = int(np.argmax(delta_theta))
    pitched_nose_moves_up = bool(
        _body_axes_plot_coordinates(states[maximum_pitch_index, 6:10])[2, 0]
        > initial_axes[2, 0]
    )
    maximum_bank_index = int(np.argmax(delta_phi))
    banked_right_axis_moves_down = bool(
        _body_axes_plot_coordinates(states[maximum_bank_index, 6:10])[2, 1]
        < initial_axes[2, 1]
    )
    relative_orientation = initial_axes.T @ final_axes
    orientation_cosine = np.clip(
        0.5 * (np.trace(relative_orientation) - 1.0), -1.0, 1.0
    )
    final_orientation_difference = np.rad2deg(np.arccos(orientation_cosine))
    returned_close_to_initial = bool(final_orientation_difference < 5.0)
    print(f"Initial forward axis points approximately +North: {forward_points_north}")
    print(f"Positive pitch rotates the nose upward: {pitched_nose_moves_up}")
    print(
        "Positive bank rotates the right body axis downward: "
        f"{banked_right_axis_moves_down}"
    )
    print(
        "Final attitude difference from initial: "
        f"{final_orientation_difference:.8f} deg"
    )
    print(f"Attitude returned close to initial: {returned_close_to_initial}")

    return (
        response_figure,
        trajectory_figure,
        attitude_figure,
        trajectory_3d_figure,
    )


if __name__ == "__main__":
    create_combined_maneuver_figures()
    plt.show()
