"""Animate a long-form controlled nonlinear F-16 demonstration flight."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.f16sim.air_data import air_data_from_body_velocity
from src.f16sim.attitude import quaternion_to_dcm
from src.f16sim.controllers import pitch_attitude_control
from src.f16sim.parameters import FT_TO_METER
from src.f16sim.simulation import simulate_f16_feedback
from src.f16sim.trim import trim_straight_level
from src.f16sim.visualization import create_flight_animation


TRUE_AIRSPEED = 502.0 * FT_TO_METER
ALTITUDE_M = 0.0
CG_FRACTION = 0.30
KQ = 5.0
KTHETA = 0.5
KR = 50.0
KP = 5.0
KPHI = 1.0
DURATION = 110.0
DT = 0.01
MP4_FPS = 60
MP4_FIGURE_SIZE = (19.2, 10.8)
MP4_DPI = 100
DEFAULT_MP4_OUTPUT = PROJECT_ROOT / "media" / "f16_flight_demo.mp4"


def _smooth_step(time, start, end):
    """Return a cosine-smoothed transition from zero to one."""
    if time <= start:
        return 0.0
    if time >= end:
        return 1.0
    fraction = (time - start) / (end - start)
    return 0.5 - 0.5 * np.cos(np.pi * fraction)


def _smooth_pulse(time, rise_start, rise_end, fall_start, fall_end):
    """Return a cosine-ramped unit pulse with a constant middle section."""
    return _smooth_step(time, rise_start, rise_end) * (
        1.0 - _smooth_step(time, fall_start, fall_end)
    )


def _demo_commands(time):
    """Return pitch, bank, direct roll-rate command, and maneuver phase."""
    climb = _smooth_pulse(time, 4.0, 5.5, 28.0, 31.0)
    climbing_bank = _smooth_pulse(time, 4.5, 6.0, 21.0, 23.0)
    reverse_bank = _smooth_pulse(time, 21.0, 24.0, 35.0, 37.0)
    u_turn_bank = _smooth_pulse(time, 47.0, 49.0, 75.0, 77.0)
    exit_bank = _smooth_pulse(time, 76.0, 78.0, 82.0, 84.0)
    final_bank = _smooth_pulse(time, 92.0, 94.0, 102.0, 104.0)
    roll_rate = np.deg2rad(115.0) * _smooth_pulse(
        time, 38.0, 39.0, 44.0, 45.0
    )
    direct_roll_rate = roll_rate if 38.0 <= time <= 45.0 else None
    theta_command = np.deg2rad(18.0) * climb
    phi_command = (
        np.deg2rad(55.0) * climbing_bank
        - np.deg2rad(55.0) * reverse_bank
        + np.deg2rad(75.0) * u_turn_bank
        - np.deg2rad(45.0) * exit_bank
        + np.deg2rad(45.0) * final_bank
    )

    if time < 4.0:
        phase = "LEVEL"
    elif time < 22.0:
        phase = "CLIMBING TURN"
    elif time < 37.0:
        phase = "S-TURN"
    elif time < 38.0:
        phase = "RECOVERY"
    elif time < 45.0:
        phase = "AXIAL ROLL"
    elif time < 47.0:
        phase = "RECOVERY"
    elif time < 77.0:
        phase = "U-TURN"
    elif time < 84.0:
        phase = "CURVED EXIT"
    elif time < 92.0:
        phase = "RECOVERY"
    elif time < 104.0:
        phase = "FINAL TURN"
    else:
        phase = "FINAL STRAIGHT"
    return theta_command, phi_command, direct_roll_rate, phase


def _euler_angles(quaternion):
    """Return roll, pitch, and yaw for diagnostics only."""
    ned_to_body = quaternion_to_dcm(quaternion)
    return np.array(
        [
            np.arctan2(ned_to_body[1, 2], ned_to_body[2, 2]),
            np.arcsin(np.clip(-ned_to_body[0, 2], -1.0, 1.0)),
            np.arctan2(ned_to_body[0, 1], ned_to_body[0, 0]),
        ]
    )


def simulate_flight_demo():
    """Run the presentation maneuver and return all animation histories."""
    trim = trim_straight_level(
        TRUE_AIRSPEED, ALTITUDE_M, cg_fraction=CG_FRACTION
    )
    if not trim["success"]:
        raise RuntimeError(f"Unable to obtain reference trim: {trim['message']}")

    trim_phi, trim_theta, _ = _euler_angles(trim["state"][6:10])
    trim_p, trim_q, trim_r = trim["state"][10:13]

    def control_law(time, state):
        phi, theta, _ = _euler_angles(state[6:10])
        (
            delta_theta_command,
            delta_phi_command,
            direct_roll_rate,
            _,
        ) = _demo_commands(time)
        _, elevator_perturbation = pitch_attitude_control(
            trim_theta + delta_theta_command,
            theta,
            state[11] - trim_q,
            Kq=KQ,
            Ktheta=KTHETA,
        )
        if direct_roll_rate is None:
            p_command = KPHI * (trim_phi + delta_phi_command - phi)
        else:
            p_command = direct_roll_rate
        return np.array(
            [
                trim["throttle"],
                trim["elevator_deg"] + elevator_perturbation,
                KP * ((state[10] - trim_p) - p_command),
                KR * (state[12] - trim_r),
            ]
        )

    times, states = simulate_f16_feedback(
        trim["state"], DURATION, DT, control_law, CG_FRACTION
    )
    controls = np.array([control_law(time, state) for time, state in zip(times, states)])
    commands_and_phases = [_demo_commands(time) for time in times]
    command_history = np.array([item[:2] for item in commands_and_phases])
    phase_history = np.array([item[3] for item in commands_and_phases])
    rate_command_history = np.empty(times.size)
    for index, (time, state, command) in enumerate(
        zip(times, states, commands_and_phases)
    ):
        direct_roll_rate = command[2]
        if direct_roll_rate is None:
            phi = _euler_angles(state[6:10])[0]
            rate_command_history[index] = KPHI * (
                trim_phi + command[1] - phi
            )
        else:
            rate_command_history[index] = direct_roll_rate
    return {
        "times": times,
        "states": states,
        "controls": controls,
        "command_history": command_history,
        "phase_history": phase_history,
        "rate_command_history": rate_command_history,
        "trim": trim,
    }


def _estimate_horizontal_turn_radius(states, selection):
    """Estimate turn radius from the median local North-East circumradius."""
    position_ft = states[selection, :2] / FT_TO_METER
    if position_ft.shape[0] < 5:
        return float("nan")
    sample_count = min(121, position_ft.shape[0])
    indices = np.unique(
        np.linspace(0, position_ft.shape[0] - 1, sample_count, dtype=int)
    )
    points = position_ft[indices]
    radii = []
    for first, middle, last in zip(points[:-2], points[1:-1], points[2:]):
        side_a = np.linalg.norm(middle - first)
        side_b = np.linalg.norm(last - middle)
        side_c = np.linalg.norm(last - first)
        first_leg = middle - first
        chord = last - first
        twice_area = abs(first_leg[0] * chord[1] - first_leg[1] * chord[0])
        if twice_area > 1e-8:
            radii.append(side_a * side_b * side_c / (2.0 * twice_area))
    if not radii:
        return float("inf")
    radii = np.asarray(radii)
    lower, upper = np.percentile(radii, (10.0, 90.0))
    central_radii = radii[(radii >= lower) & (radii <= upper)]
    return float(np.median(central_radii))


def _print_diagnostics(result):
    times = result["times"]
    states = result["states"]
    controls = result["controls"]
    trim = result["trim"]
    angles = np.unwrap(
        np.array([_euler_angles(quaternion) for quaternion in states[:, 6:10]]),
        axis=0,
    )
    air_data = np.array(
        [air_data_from_body_velocity(velocity) for velocity in states[:, 3:6]]
    )
    altitude_gain = -(states[:, 2] - states[0, 2]) / FT_TO_METER
    climbing_turn = (times >= 4.5) & (times <= 23.0)
    reverse_turn = (times >= 21.0) & (times <= 37.0)
    u_turn = (times >= 47.0) & (times <= 77.0)
    exit_turn = (times >= 76.0) & (times <= 84.0)
    final_turn = (times >= 92.0) & (times <= 104.0)
    axial_roll = (times >= 38.0) & (times <= 45.0)
    before_45 = times <= 45.0
    climbing_heading_change = np.rad2deg(
        angles[climbing_turn][-1, 2] - angles[climbing_turn][0, 2]
    )
    reverse_heading_change = np.rad2deg(
        angles[reverse_turn][-1, 2] - angles[reverse_turn][0, 2]
    )
    u_turn_heading_change = np.rad2deg(
        angles[u_turn][-1, 2] - angles[u_turn][0, 2]
    )
    exit_heading_change = np.rad2deg(
        angles[exit_turn][-1, 2] - angles[exit_turn][0, 2]
    )
    final_heading_change = np.rad2deg(
        angles[final_turn][-1, 2] - angles[final_turn][0, 2]
    )
    axial_roll_angle = np.rad2deg(
        np.trapezoid(states[axial_roll, 10], times[axial_roll])
    )
    climbing_turn_radius = _estimate_horizontal_turn_radius(states, climbing_turn)
    reverse_turn_radius = _estimate_horizontal_turn_radius(states, reverse_turn)
    u_turn_radius = _estimate_horizontal_turn_radius(states, u_turn)
    final_turn_radius = _estimate_horizontal_turn_radius(states, final_turn)
    control_perturbations = controls.copy()
    control_perturbations[:, 0] -= trim["throttle"]
    control_perturbations[:, 1] -= trim["elevator_deg"]

    print(f"Total simulated duration: {times[-1]:.2f} s")
    print(f"Maximum altitude gain: {np.max(altitude_gain):.3f} ft")
    print(
        "Altitude-change range: "
        f"{np.min(altitude_gain):.3f} to {np.max(altitude_gain):.3f} ft"
    )
    print(
        "True-airspeed range: "
        f"{np.min(air_data[:, 0]) / FT_TO_METER:.3f} to "
        f"{np.max(air_data[:, 0]) / FT_TO_METER:.3f} ft/s"
    )
    index_45 = int(np.argmin(np.abs(times - 45.0)))
    altitude_at_45 = -states[index_45, 2] / FT_TO_METER
    altitude_gain_at_45 = -(states[index_45, 2] - states[0, 2]) / FT_TO_METER
    heading_change_before_45 = np.rad2deg(
        angles[index_45, 2] - angles[0, 2]
    )
    print(f"Altitude at t=45 s: {altitude_at_45:.3f} ft")
    print(f"Altitude gain by t=45 s: {altitude_gain_at_45:.3f} ft")
    print(
        "Minimum true airspeed before t=45 s: "
        f"{np.min(air_data[before_45, 0]) / FT_TO_METER:.3f} ft/s"
    )
    print(
        "Maximum alpha before t=45 s: "
        f"{np.max(np.abs(air_data[before_45, 1])):.3f} deg"
    )
    print(
        "Maximum pitch angle before t=45 s: "
        f"{np.rad2deg(np.max(angles[before_45, 1])):.3f} deg"
    )
    print(f"Heading change before t=45 s: {heading_change_before_45:.3f} deg")
    maximum_alpha = np.max(np.abs(air_data[:, 1]))
    maximum_beta = np.max(np.abs(air_data[:, 2]))
    print(f"Maximum absolute alpha: {maximum_alpha:.3f} deg")
    print(f"Maximum absolute beta: {maximum_beta:.3f} deg")
    print(f"Maximum absolute phi: {np.rad2deg(np.max(np.abs(angles[:, 0]))):.3f} deg")
    print(f"Maximum absolute theta: {np.rad2deg(np.max(np.abs(angles[:, 1]))):.3f} deg")
    rates_deg = np.rad2deg(states[:, 10:13])
    for column, label in enumerate(("p", "q", "r")):
        print(f"Maximum absolute {label}: {np.max(np.abs(rates_deg[:, column])):.3f} deg/s")
    print(f"Maximum elevator perturbation: {np.max(np.abs(control_perturbations[:, 1])):.3f} deg")
    print(f"Maximum aileron: {np.max(np.abs(controls[:, 2])):.3f} deg")
    print(f"Maximum rudder: {np.max(np.abs(controls[:, 3])):.3f} deg")
    print(f"Heading change during climbing turn: {climbing_heading_change:.3f} deg")
    print(f"Heading change during reverse turn: {reverse_heading_change:.3f} deg")
    print(f"Heading change during U-turn: {u_turn_heading_change:.3f} deg")
    print(f"Heading change during curved exit: {exit_heading_change:.3f} deg")
    print(f"Heading change during final turn: {final_heading_change:.3f} deg")
    print(f"Climbing-turn estimated radius: {climbing_turn_radius:.3f} ft")
    print(f"Opposite-turn estimated radius: {reverse_turn_radius:.3f} ft")
    print(f"U-turn estimated radius: {u_turn_radius:.3f} ft")
    print(f"Final-turn estimated radius: {final_turn_radius:.3f} ft")
    print(f"Integrated axial-roll angle: {axial_roll_angle:.3f} deg")
    print(f"Final heading: {np.rad2deg(angles[-1, 2]):.3f} deg")
    print(f"Final North displacement: {(states[-1, 0] - states[0, 0]) / FT_TO_METER:.3f} ft")
    print(f"Final East displacement: {(states[-1, 1] - states[0, 1]) / FT_TO_METER:.3f} ft")
    print(f"All states remained finite: {bool(np.all(np.isfinite(states)))}")
    print(f"All controls remained finite: {bool(np.all(np.isfinite(controls)))}")
    print("Safety / model-validity checks:")
    print(f"  Alpha unusually large (>15 deg): {bool(maximum_alpha > 15.0)}")
    print(f"  Beta unusually large (>10 deg): {bool(maximum_beta > 10.0)}")
    print(
        "  Control command unusually large (>25 deg): "
        f"{bool(np.max(np.abs(control_perturbations[:, 1:])) > 25.0)}"
    )
    print(
        "  Airspeed below 75% of trim: "
        f"{bool(np.min(air_data[:, 0]) < 0.75 * TRUE_AIRSPEED)}"
    )
    print(f"  Any state diverged/non-finite: {bool(not np.all(np.isfinite(states)))}")


def create_demo_trajectory_figures(result, vertical_exaggeration=5.0):
    """Return 3D and top-view figures for the complete demo trajectory."""
    if not np.isfinite(vertical_exaggeration) or vertical_exaggeration <= 0.0:
        raise ValueError("vertical_exaggeration must be finite and positive")
    states = result["states"]
    north = (states[:, 0] - states[0, 0]) / FT_TO_METER
    east = (states[:, 1] - states[0, 1]) / FT_TO_METER
    altitude = -(states[:, 2] - states[0, 2]) / FT_TO_METER
    displayed_altitude = vertical_exaggeration * altitude

    trajectory_figure = plt.figure(figsize=(11.0, 8.0))
    trajectory_axis = trajectory_figure.add_subplot(111, projection="3d")
    trajectory_axis.plot(north, east, displayed_altitude, linewidth=2.0)
    trajectory_axis.scatter(north[0], east[0], displayed_altitude[0], label="Start")
    trajectory_axis.scatter(
        north[-1], east[-1], displayed_altitude[-1], marker="x", label="End"
    )
    trajectory_axis.set_xlabel("North [ft]")
    trajectory_axis.set_ylabel("East [ft]")
    trajectory_axis.set_zlabel(
        "Altitude change from initial [ft], displayed scale"
    )
    trajectory_axis.set_title(
        "Nonlinear F-16 Demo Trajectory\n"
        f"Vertical scale exaggerated x{vertical_exaggeration:g}"
    )
    trajectory_axis.view_init(elev=27.0, azim=-58.0)
    trajectory_axis.grid(True)
    trajectory_axis.legend()
    trajectory_figure.tight_layout()

    top_figure, top_axis = plt.subplots(figsize=(9.0, 8.0))
    top_axis.plot(north, east, linewidth=2.0)
    top_axis.scatter(north[0], east[0], label="Start")
    top_axis.scatter(north[-1], east[-1], marker="x", label="End")
    top_axis.set_xlabel("North [ft]")
    top_axis.set_ylabel("East [ft]")
    top_axis.set_title("Nonlinear F-16 Demo Ground Track")
    top_axis.axis("equal")
    top_axis.grid(True)
    top_axis.legend()
    top_figure.tight_layout()
    return trajectory_figure, top_figure


def create_flight_demo_animation(
    camera="chase",
    fps=30.0,
    playback_speed=8.0,
    aircraft_scale=92.0,
    result=None,
    print_diagnostics=True,
):
    """Run the long demo maneuver, print diagnostics, and return its animation."""
    if result is None:
        result = simulate_flight_demo()
    if print_diagnostics:
        _print_diagnostics(result)
    return create_flight_animation(
        result["times"],
        result["states"],
        controls=result["controls"],
        command_history=result["command_history"],
        phase_history=result["phase_history"],
        rate_command_history=result["rate_command_history"],
        fps=fps,
        playback_speed=playback_speed,
        camera=camera,
        view_size=7000.0,
        vertical_view_size=3500.0,
        aircraft_scale=aircraft_scale,
        trail_duration=12.0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", choices=("fixed", "chase"), default="chase")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--speed", type=float, default=8.0)
    parser.add_argument("--aircraft-scale", type=float, default=92.0)
    parser.add_argument("--vertical-exaggeration", type=float, default=5.0)
    parser.add_argument(
        "--save-mp4",
        action="store_true",
        help="save the animation as a 1080p MP4 instead of opening it",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MP4_OUTPUT,
        help="MP4 output path (default: media/f16_flight_demo.mp4)",
    )
    arguments = parser.parse_args()
    result = simulate_flight_demo()
    _print_diagnostics(result)
    animation_fps = MP4_FPS if arguments.save_mp4 else arguments.fps
    if not arguments.save_mp4:
        trajectory_figure, top_figure = create_demo_trajectory_figures(
            result, vertical_exaggeration=arguments.vertical_exaggeration
        )
    animation = create_flight_demo_animation(
        camera=arguments.camera,
        fps=animation_fps,
        playback_speed=arguments.speed,
        aircraft_scale=arguments.aircraft_scale,
        result=result,
        print_diagnostics=False,
    )
    if arguments.save_mp4:
        output_path = arguments.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        animation._fig.set_size_inches(*MP4_FIGURE_SIZE, forward=True)
        video_duration = result["times"][-1] / arguments.speed
        print("Preparing MP4 export; rendering may take some time...")
        print(f"Output path: {output_path}")
        print(f"FPS: {MP4_FPS}")
        width = MP4_FIGURE_SIZE[0] * MP4_DPI
        height = MP4_FIGURE_SIZE[1] * MP4_DPI
        print(f"Resolution: {width:.0f}x{height:.0f}")
        print(f"Playback speed: {arguments.speed:g}x")
        print(f"Approximate video duration: {video_duration:.2f} s")
        writer = FFMpegWriter(
            fps=MP4_FPS,
            codec="libx264",
            extra_args=[
                "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"
            ],
        )
        animation.save(output_path, writer=writer, dpi=MP4_DPI)
        print(f"MP4 export complete: {output_path}")
        plt.close(animation._fig)
    else:
        plt.show()
