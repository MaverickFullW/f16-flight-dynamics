"""Animate a controlled nonlinear F-16 descending 360-degree turn."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
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
from src.f16sim.visualization import (
    apply_readme_animation_layout,
    create_flight_animation,
    select_animation_frames,
)


TRUE_AIRSPEED = 502.0 * FT_TO_METER
INITIAL_ALTITUDE = 10_000.0 * FT_TO_METER
CG_FRACTION = 0.30
KQ = 5.0
KTHETA = 0.5
KR = 50.0
KP = 5.0
KPHI = 1.0
BANK_COMMAND = np.deg2rad(55.0)
PITCH_COMMAND = np.deg2rad(-2.0)
ROLL_IN_START = 6.0
ROLL_IN_END = 12.0
ROLL_OUT_START = 111.5
ROLL_OUT_END = 119.5
DURATION = 132.0
DT = 0.01
DEFAULT_FPS = 30.0
DEFAULT_PLAYBACK_SPEED = 7.0
AIRCRAFT_SCALE = 150.0


def _smooth_step(time, start, end):
    """Return a cosine-smoothed transition from zero to one."""
    if time <= start:
        return 0.0
    if time >= end:
        return 1.0
    fraction = (time - start) / (end - start)
    return 0.5 - 0.5 * np.cos(np.pi * fraction)


def _maneuver_commands(time):
    """Return pitch and bank commands relative to straight-level trim."""
    turn = _smooth_step(time, ROLL_IN_START, ROLL_IN_END) * (
        1.0 - _smooth_step(time, ROLL_OUT_START, ROLL_OUT_END)
    )
    if time < ROLL_IN_START:
        phase = "TRIMMED FLIGHT"
    elif time < ROLL_IN_END:
        phase = "ROLL-IN / DESCENT ENTRY"
    elif time < ROLL_OUT_START:
        phase = "DESCENDING 360 TURN"
    elif time < ROLL_OUT_END:
        phase = "SMOOTH ROLLOUT"
    else:
        phase = "WINGS-LEVEL RECOVERY"
    return PITCH_COMMAND * turn, BANK_COMMAND * turn, phase


def _euler_angles(quaternion):
    """Return roll, pitch, and yaw angles in radians."""
    ned_to_body = quaternion_to_dcm(quaternion)
    return np.array(
        [
            np.arctan2(ned_to_body[1, 2], ned_to_body[2, 2]),
            np.arcsin(np.clip(-ned_to_body[0, 2], -1.0, 1.0)),
            np.arctan2(ned_to_body[0, 1], ned_to_body[0, 0]),
        ]
    )


def simulate_descending_turn():
    """Run the commanded maneuver through the nonlinear 14-state model."""
    trim = trim_straight_level(
        TRUE_AIRSPEED, INITIAL_ALTITUDE, cg_fraction=CG_FRACTION
    )
    if not trim["success"]:
        raise RuntimeError(f"Unable to obtain reference trim: {trim['message']}")

    trim_phi, trim_theta, _ = _euler_angles(trim["state"][6:10])
    trim_p, trim_q, trim_r = trim["state"][10:13]

    def control_law(time, state):
        phi, theta, _ = _euler_angles(state[6:10])
        delta_theta_command, delta_phi_command, _ = _maneuver_commands(time)
        _, elevator_perturbation = pitch_attitude_control(
            trim_theta + delta_theta_command,
            theta,
            state[11] - trim_q,
            Kq=KQ,
            Ktheta=KTHETA,
        )
        p_command = KPHI * (trim_phi + delta_phi_command - phi)
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
    controls = np.array(
        [control_law(time, state) for time, state in zip(times, states)]
    )
    commands = [_maneuver_commands(time) for time in times]
    command_history = np.array([command[:2] for command in commands])
    phase_history = np.array([command[2] for command in commands])
    rate_command_history = np.empty(times.size)
    for index, (time, state) in enumerate(zip(times, states)):
        phi = _euler_angles(state[6:10])[0]
        rate_command_history[index] = KPHI * (
            trim_phi + _maneuver_commands(time)[1] - phi
        )
    return {
        "times": times,
        "states": states,
        "controls": controls,
        "command_history": command_history,
        "phase_history": phase_history,
        "rate_command_history": rate_command_history,
        "trim": trim,
    }


def _estimate_turn_radius(states, selection):
    """Estimate horizontal radius from local North-East circumcircles."""
    points = states[selection, :2] / FT_TO_METER
    sample_count = min(181, points.shape[0])
    indices = np.unique(
        np.linspace(0, points.shape[0] - 1, sample_count, dtype=int)
    )
    points = points[indices]
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
    radii = np.asarray(radii)
    lower, upper = np.percentile(radii, (10.0, 90.0))
    return float(np.median(radii[(radii >= lower) & (radii <= upper)]))


def print_diagnostics(result):
    """Print maneuver acceptance metrics and enforce validity checks."""
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
    altitude_ft = -states[:, 2] / FT_TO_METER
    rates_deg = np.rad2deg(states[:, 10:13])
    turn_selection = (times >= ROLL_IN_START) & (times <= ROLL_OUT_END)
    heading_change = np.rad2deg(
        angles[turn_selection][-1, 2] - angles[turn_selection][0, 2]
    )
    radius = _estimate_turn_radius(states, turn_selection)
    control_perturbations = controls.copy()
    control_perturbations[:, 0] -= trim["throttle"]
    control_perturbations[:, 1] -= trim["elevator_deg"]

    print("Descending 360-degree turn diagnostics")
    print(f"Initial altitude: {altitude_ft[0]:.3f} ft")
    print(f"Minimum altitude: {np.min(altitude_ft):.3f} ft")
    print(f"Final altitude: {altitude_ft[-1]:.3f} ft")
    print(f"Total altitude loss: {altitude_ft[0] - altitude_ft[-1]:.3f} ft")
    print(f"Heading change: {heading_change:.3f} deg")
    print(f"Estimated turn radius: {radius:.3f} ft")
    print(f"Maximum |phi|: {np.rad2deg(np.max(np.abs(angles[:, 0]))):.3f} deg")
    print(f"Maximum |theta|: {np.rad2deg(np.max(np.abs(angles[:, 1]))):.3f} deg")
    print(f"Maximum |alpha|: {np.max(np.abs(air_data[:, 1])):.3f} deg")
    print(f"Maximum |beta|: {np.max(np.abs(air_data[:, 2])):.3f} deg")
    for column, label in enumerate(("p", "q", "r")):
        print(f"Maximum |{label}|: {np.max(np.abs(rates_deg[:, column])):.3f} deg/s")
    print(f"Maximum elevator: {np.max(np.abs(controls[:, 1])):.3f} deg")
    print(f"Maximum aileron: {np.max(np.abs(controls[:, 2])):.3f} deg")
    print(f"Maximum rudder: {np.max(np.abs(controls[:, 3])):.3f} deg")
    print(
        "VT range: "
        f"{np.min(air_data[:, 0]) / FT_TO_METER:.3f} to "
        f"{np.max(air_data[:, 0]) / FT_TO_METER:.3f} ft/s"
    )
    print(f"Final bank angle: {np.rad2deg(angles[-1, 0]):.3f} deg")
    print(
        "Final body rates: "
        f"p={rates_deg[-1, 0]:.3f}, q={rates_deg[-1, 1]:.3f}, "
        f"r={rates_deg[-1, 2]:.3f} deg/s"
    )

    failures = []
    if not 330.0 <= abs(heading_change) <= 390.0:
        failures.append("heading change is outside 330-390 deg")
    if np.min(altitude_ft) <= 0.0:
        failures.append("trajectory reached or crossed 0 ft")
    if np.max(np.abs(air_data[:, 1])) > 15.0:
        failures.append("|alpha| exceeded 15 deg")
    if np.max(np.abs(air_data[:, 2])) > 10.0:
        failures.append("|beta| exceeded 10 deg")
    if np.max(np.abs(control_perturbations[:, 1:])) > 25.0:
        failures.append("a control perturbation exceeded 25 deg")
    if np.min(air_data[:, 0]) < 0.75 * TRUE_AIRSPEED:
        failures.append("airspeed fell below 75% of trim")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(controls)):
        failures.append("a state or control became non-finite")
    if abs(np.rad2deg(angles[-1, 0])) > 3.0:
        failures.append("final bank magnitude exceeded 3 deg")
    if np.max(np.abs(rates_deg[-1])) > 1.0:
        failures.append("a final body-rate magnitude exceeded 1 deg/s")
    if failures:
        raise RuntimeError("Acceptance checks failed: " + "; ".join(failures))
    print("Acceptance checks: PASSED")


def create_descending_turn_animation(result, fps=DEFAULT_FPS, playback_speed=7.0):
    """Create a fixed, automatically fitted overview animation."""
    animation = create_flight_animation(
        result["times"],
        result["states"],
        controls=result["controls"],
        command_history=result["command_history"],
        phase_history=result["phase_history"],
        rate_command_history=result["rate_command_history"],
        fps=fps,
        playback_speed=playback_speed,
        camera="fixed",
        aircraft_scale=AIRCRAFT_SCALE,
        trail_duration=18.0,
        show_full_trajectory=True,
    )
    axis = animation._fig.axes[0]
    axis.view_init(elev=27.0, azim=-45.0)
    axis.set_zlabel("Altitude MSL [ft]")
    apply_readme_animation_layout(
        animation,
        "F-16 Nonlinear 6DoF Descending 360-Degree Turn",
        vertical_reference="Vertical reference: mean sea level (0 ft)",
    )

    frame_indices = select_animation_frames(
        result["times"], fps=fps, playback_speed=playback_speed
    )
    original_update = animation._func

    def update_with_msl_altitude(frame_number):
        artists = original_update(frame_number)
        index = int(frame_indices[frame_number])
        current_air_data = air_data_from_body_velocity(
            result["states"][index, 3:6]
        )
        altitude_ft = -result["states"][index, 2] / FT_TO_METER
        left_hud = axis.texts[-5]
        left_hud.set_text(
            "FLIGHT\n"
            f"VT       {current_air_data[0] / FT_TO_METER:6.1f} ft/s\n"
            f"Altitude {altitude_ft:6.0f} ft MSL\n"
            f"alpha    {current_air_data[1]:6.1f} deg\n"
            f"beta     {current_air_data[2]:6.1f} deg"
        )
        return artists

    animation._func = update_with_msl_altitude
    return animation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--speed", type=float, default=DEFAULT_PLAYBACK_SPEED)
    arguments = parser.parse_args()
    result = simulate_descending_turn()
    print_diagnostics(result)
    animation = create_descending_turn_animation(
        result, fps=arguments.fps, playback_speed=arguments.speed
    )
    plt.show()
