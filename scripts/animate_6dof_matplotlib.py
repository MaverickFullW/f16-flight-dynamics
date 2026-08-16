"""Animate the combined controlled F-16 maneuver with a wireframe pointer."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.simulate_combined_6dof_maneuver import (
    ALTITUDE_M,
    CG_FRACTION,
    DT,
    DURATION,
    KPHI,
    KP,
    KQ,
    KR,
    KTHETA,
    TRUE_AIRSPEED,
    _command_perturbations,
    _euler_roll_pitch,
)
from src.f16sim.attitude import quaternion_normalize, quaternion_to_dcm
from src.f16sim.controllers import pitch_attitude_control
from src.f16sim.parameters import FT_TO_METER
from src.f16sim.simulation import simulate_f16_feedback
from src.f16sim.trim import trim_straight_level


FRAME_STRIDE = 5
ANIMATION_FPS = 25
VERTICAL_EXAGGERATION = 5.0


_BODY_POINTS = {
    "nose": np.array([1.5, 0.0, 0.0]),
    "tail": np.array([-1.0, 0.0, 0.0]),
    "left_wing": np.array([0.0, -1.0, 0.0]),
    "right_wing": np.array([0.0, 1.0, 0.0]),
    "left_tail": np.array([-0.8, -0.4, 0.0]),
    "right_tail": np.array([-0.8, 0.4, 0.0]),
    "fin_tip": np.array([-0.8, 0.0, -0.5]),
}
_BODY_SEGMENTS = (
    ("tail", "nose"),
    ("left_wing", "right_wing"),
    ("left_tail", "right_tail"),
    ("tail", "fin_tip"),
    ("nose", "left_wing"),
    ("nose", "right_wing"),
)


def _euler_angles(quaternion):
    q0, q1, q2, q3 = quaternion_normalize(quaternion)
    phi = np.arctan2(
        2.0 * (q0 * q1 + q2 * q3),
        1.0 - 2.0 * (q1**2 + q2**2),
    )
    theta = np.arcsin(
        np.clip(2.0 * (q0 * q2 - q3 * q1), -1.0, 1.0)
    )
    psi = np.arctan2(
        2.0 * (q0 * q3 + q1 * q2),
        1.0 - 2.0 * (q2**2 + q3**2),
    )
    return phi, theta, psi


def _simulate_maneuver():
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
        _, elevator_perturbation = pitch_attitude_control(
            trim_theta + delta_theta_command,
            theta,
            state[11] - trim_q,
            Kq=KQ,
            Ktheta=KTHETA,
        )
        p_command = KPHI * (trim_phi + delta_phi_command - phi)
        aileron = KP * ((state[10] - trim_p) - p_command)
        rudder = KR * (state[12] - trim_r)
        return [
            trim["throttle"],
            trim["elevator_deg"] + elevator_perturbation,
            aileron,
            rudder,
        ]

    return simulate_f16_feedback(
        initial_state=trim["state"],
        duration=DURATION,
        dt=DT,
        control_law=control_law,
        cg_fraction=CG_FRACTION,
    )


def _wireframe_points_ned(quaternion, scale_ft):
    body_to_ned = quaternion_to_dcm(quaternion).T
    return {
        name: body_to_ned @ (scale_ft * point)
        for name, point in _BODY_POINTS.items()
    }


def _set_plot_limits(axis, north, east, displayed_altitude):
    values = (north, east, displayed_altitude)
    centers = [0.5 * (np.min(value) + np.max(value)) for value in values]
    maximum_span = max(*(np.ptp(value) for value in values), 1.0)
    half_range = 0.55 * maximum_span
    axis.set_xlim(centers[0] - half_range, centers[0] + half_range)
    axis.set_ylim(centers[1] - half_range, centers[1] + half_range)
    axis.set_zlim(centers[2] - half_range, centers[2] + half_range)
    axis.set_box_aspect((1.0, 1.0, 1.0))


def create_6dof_animation():
    """Simulate the maneuver and return its Matplotlib ``FuncAnimation``."""
    times, states = _simulate_maneuver()
    north = (states[:, 0] - states[0, 0]) / FT_TO_METER
    east = (states[:, 1] - states[0, 1]) / FT_TO_METER
    altitude = (-states[:, 2] + states[0, 2]) / FT_TO_METER
    displayed_altitude = VERTICAL_EXAGGERATION * altitude
    frame_indices = np.arange(0, states.shape[0], FRAME_STRIDE)
    if frame_indices[-1] != states.shape[0] - 1:
        frame_indices = np.append(frame_indices, states.shape[0] - 1)

    horizontal_span = max(np.ptp(north), np.ptp(east), 1.0)
    aircraft_scale = 0.025 * horizontal_span
    figure = plt.figure(figsize=(11.0, 8.0))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(
        north,
        east,
        displayed_altitude,
        color="0.75",
        linewidth=1.0,
        label="Full trajectory",
    )
    flown_line, = axis.plot([], [], [], color="C0", linewidth=2.0, label="Flown")
    aircraft_lines = [
        axis.plot([], [], [], color="C3", linewidth=2.0)[0]
        for _ in _BODY_SEGMENTS
    ]
    time_text = axis.text2D(0.03, 0.96, "", transform=axis.transAxes)
    attitude_text = axis.text2D(0.03, 0.88, "", transform=axis.transAxes)

    axis.set_xlabel("North [ft]")
    axis.set_ylabel("East [ft]")
    axis.set_zlabel("Altitude change [ft]")
    axis.set_title(
        "Controlled F-16 6DoF Trajectory and Attitude\n"
        f"Display vertical exaggeration: {VERTICAL_EXAGGERATION:g}x"
    )
    axis.view_init(elev=25.0, azim=-60.0)
    axis.grid(True)
    axis.legend(loc="upper right")
    _set_plot_limits(axis, north, east, displayed_altitude)

    def update(frame_number):
        index = int(frame_indices[frame_number])
        flown_line.set_data_3d(
            north[: index + 1],
            east[: index + 1],
            displayed_altitude[: index + 1],
        )
        origin = np.array(
            [north[index], east[index], displayed_altitude[index]]
        )
        points_ned = _wireframe_points_ned(
            states[index, 6:10], aircraft_scale
        )
        for line, (start_name, end_name) in zip(
            aircraft_lines, _BODY_SEGMENTS
        ):
            start_vector = points_ned[start_name].copy()
            end_vector = points_ned[end_name].copy()
            start_vector[2] *= -VERTICAL_EXAGGERATION
            end_vector[2] *= -VERTICAL_EXAGGERATION
            plotted_start = origin + start_vector
            plotted_end = origin + end_vector
            line.set_data_3d(
                [plotted_start[0], plotted_end[0]],
                [plotted_start[1], plotted_end[1]],
                [plotted_start[2], plotted_end[2]],
            )

        phi, theta, psi = np.rad2deg(_euler_angles(states[index, 6:10]))
        time_text.set_text(f"t = {times[index]:4.1f} s")
        attitude_text.set_text(
            f"phi = {phi:6.2f} deg\n"
            f"theta = {theta:6.2f} deg\n"
            f"psi = {psi:6.2f} deg"
        )
        return (flown_line, *aircraft_lines, time_text, attitude_text)

    animation = FuncAnimation(
        figure,
        update,
        frames=frame_indices.size,
        interval=1000.0 / ANIMATION_FPS,
        blit=False,
        repeat=True,
    )
    return animation


if __name__ == "__main__":
    animation = create_6dof_animation()
    plt.show()
