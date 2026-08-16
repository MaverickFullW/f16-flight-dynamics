"""Reusable Matplotlib visualization helpers for F-16 flight histories."""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from .air_data import air_data_from_body_velocity
from .atmosphere import f16_air_data
from .attitude import quaternion_to_dcm
from .parameters import FT_TO_METER


def create_f16_low_poly_geometry():
    """Return an original F-16-like mesh and per-face material tags.

    Geometry uses body FRD coordinates: forward ``+x``, right ``+y``, and
    down ``+z``. Dimensions are intentionally schematic and unitless.
    """
    vertices = []
    faces = []
    groups = []

    def add_vertices(points):
        start = len(vertices)
        vertices.extend(points)
        return tuple(range(start, len(vertices)))

    def add_face(indices, group="airframe"):
        faces.append(tuple(indices))
        groups.append(group)

    station_data = (
        (6.34, 0.040, 0.045, 0.075),
        (6.23, 0.085, 0.095, 0.070),
        (6.02, 0.150, 0.170, 0.065),
        (5.65, 0.210, 0.235, 0.060),
        (5.05, 0.255, 0.290, 0.055),
        (4.15, 0.30, 0.34, 0.05),
        (3.05, 0.44, 0.47, 0.03),
        (1.40, 0.62, 0.66, 0.04),
        (-0.40, 0.72, 0.74, 0.05),
        (-2.25, 0.59, 0.61, 0.04),
        (-3.85, 0.41, 0.43, 0.02),
        (-4.45, 0.32, 0.32, 0.01),
    )
    section_angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    fuselage_sections = []
    for x, half_width, half_height, center_z in station_data:
        section = add_vertices(
            [
                (
                    x,
                    half_width * np.sin(angle),
                    center_z - half_height * np.cos(angle),
                )
                for angle in section_angles
            ]
        )
        fuselage_sections.append(section)
    add_face(tuple(reversed(fuselage_sections[0])))
    for forward, aft in zip(fuselage_sections[:-1], fuselage_sections[1:]):
        for index in range(8):
            next_index = (index + 1) % 8
            add_face(
                (forward[index], forward[next_index], aft[next_index], aft[index])
            )

    canopy_stations = (
        (3.10, 0.08, -0.43, -0.57),
        (2.60, 0.20, -0.53, -0.82),
        (2.05, 0.30, -0.60, -1.12),
        (1.40, 0.34, -0.64, -1.32),
        (0.80, 0.31, -0.65, -1.28),
        (0.15, 0.20, -0.61, -0.92),
        (-0.20, 0.06, -0.58, -0.68),
    )
    canopy_sections = []
    for x, half_width, side_z, crown_z in canopy_stations:
        shoulder_z = side_z + 0.72 * (crown_z - side_z)
        canopy_sections.append(
            add_vertices(
                (
                    (x, -half_width, side_z),
                    (x, -0.52 * half_width, shoulder_z),
                    (x, 0.0, crown_z),
                    (x, 0.52 * half_width, shoulder_z),
                    (x, half_width, side_z),
                )
            )
        )
    for forward, aft in zip(canopy_sections[:-1], canopy_sections[1:]):
        for index in range(4):
            add_face(
                (forward[index], forward[index + 1],
                 aft[index + 1], aft[index]),
                "canopy",
            )
    add_face(canopy_sections[-1], "canopy")

    def add_thin_surface(outline, thickness, group):
        upper = add_vertices([(x, y, z - thickness / 2.0) for x, y, z in outline])
        lower = add_vertices([(x, y, z + thickness / 2.0) for x, y, z in outline])
        add_face(upper, group)
        add_face(tuple(reversed(lower)), group)
        for index in range(len(outline)):
            next_index = (index + 1) % len(outline)
            add_face((upper[index], upper[next_index], lower[next_index], lower[index]), group)

    left_wing = (
        (1.20, -0.61, -0.02), (-0.65, -3.00, 0.02),
        (-1.65, -3.00, 0.04), (-2.05, -0.59, 0.03),
    )
    add_thin_surface(left_wing, 0.08, "wing")
    add_thin_surface(tuple((x, -y, z) for x, y, z in left_wing), 0.08, "wing")

    left_lerx = (
        (2.70, -0.34, -0.29), (1.20, -0.61, -0.06),
        (0.35, -0.86, -0.18), (0.45, -0.50, -0.48),
    )
    add_face(add_vertices(left_lerx), "lerx")
    add_face(add_vertices(tuple((x, -y, z) for x, y, z in left_lerx)), "lerx")

    left_stabilator = (
        (-2.75, -0.42, -0.02), (-3.35, -1.30, 0.00),
        (-4.00, -1.06, 0.03), (-4.20, -0.31, 0.03),
    )
    add_thin_surface(left_stabilator, 0.055, "stabilator")
    add_thin_surface(
        tuple((x, -y, z) for x, y, z in left_stabilator), 0.055, "stabilator"
    )

    tail_outline = (
        (-2.10, -0.03, -0.53), (-3.00, -0.03, -2.57),
        (-3.58, -0.03, -2.37), (-4.08, -0.03, -0.37),
    )
    tail_left = add_vertices(tail_outline)
    tail_right = add_vertices(tuple((x, -y, z) for x, y, z in tail_outline))
    add_face(tail_left, "tail")
    add_face(tuple(reversed(tail_right)), "tail")
    for index in range(4):
        next_index = (index + 1) % 4
        add_face(
            (tail_left[index], tail_left[next_index], tail_right[next_index], tail_right[index]),
            "tail",
        )

    intake_lip = add_vertices(
        ((2.35, -0.31, 0.43), (2.35, 0.31, 0.43),
         (2.05, 0.28, 1.00), (2.05, -0.28, 1.00))
    )
    intake_throat = add_vertices(
        ((0.45, -0.21, 0.57), (0.45, 0.21, 0.57),
         (0.35, 0.18, 0.81), (0.35, -0.18, 0.81))
    )
    add_face(intake_lip, "intake_opening")
    for index in range(4):
        next_index = (index + 1) % 4
        add_face(
            (intake_lip[index], intake_lip[next_index],
             intake_throat[next_index], intake_throat[index]),
            "intake",
        )

    nozzle_angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    nozzle_front = add_vertices(
        [(-4.45, 0.32 * np.sin(a), 0.01 - 0.32 * np.cos(a)) for a in nozzle_angles]
    )
    nozzle_rear = add_vertices(
        [(-4.98, 0.27 * np.sin(a), 0.01 - 0.27 * np.cos(a)) for a in nozzle_angles]
    )
    for index in range(8):
        next_index = (index + 1) % 8
        add_face(
            (nozzle_front[index], nozzle_front[next_index],
             nozzle_rear[next_index], nozzle_rear[index]),
            "nozzle",
        )
    add_face(tuple(reversed(nozzle_rear)), "nozzle_interior")

    return np.asarray(vertices, dtype=float), tuple(faces), tuple(groups)


def create_fighter_geometry():
    """Return the F-16-like vertices and faces for backward compatibility."""
    vertices, faces, _ = create_f16_low_poly_geometry()
    return vertices, faces


def ned_to_plot_coordinates(points_ned):
    """Convert North/East/Down coordinates to North/East/altitude-up."""
    points_ned = np.asarray(points_ned, dtype=float)
    if points_ned.shape[-1] != 3:
        raise ValueError("points_ned must end with a coordinate dimension of 3")
    plotted = points_ned.copy()
    plotted[..., 2] *= -1.0
    return plotted


def transform_body_geometry(vertices_body, quaternion, position_ned):
    """Rotate body vertices into NED, translate, and convert to plot axes."""
    vertices_body = np.asarray(vertices_body, dtype=float)
    position_ned = np.asarray(position_ned, dtype=float)
    if vertices_body.ndim != 2 or vertices_body.shape[1] != 3:
        raise ValueError("vertices_body must have shape (number_of_vertices, 3)")
    if position_ned.shape != (3,):
        raise ValueError("position_ned must have shape (3,)")
    body_to_ned = quaternion_to_dcm(quaternion).T
    vertices_ned = (body_to_ned @ vertices_body.T).T + position_ned
    return ned_to_plot_coordinates(vertices_ned)


def select_animation_frames(times, fps=30, playback_speed=3.0):
    """Select simulation indices for time-consistent animation playback."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be finite and strictly increasing")
    fps = float(fps)
    playback_speed = float(playback_speed)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be finite and positive")
    if not np.isfinite(playback_speed) or playback_speed <= 0.0:
        raise ValueError("playback_speed must be finite and positive")
    if times.size == 1:
        return np.array([0], dtype=int)

    simulated_seconds_per_frame = playback_speed / fps
    target_times = np.arange(
        times[0],
        times[-1] + 0.5 * simulated_seconds_per_frame,
        simulated_seconds_per_frame,
    )
    target_times = np.clip(target_times, times[0], times[-1])
    right = np.searchsorted(times, target_times, side="left")
    right = np.clip(right, 0, times.size - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(target_times - times[left]) <= np.abs(
        times[right] - target_times
    )
    indices = np.where(choose_left, left, right)
    indices = np.unique(indices)
    if indices[-1] != times.size - 1:
        indices = np.append(indices, times.size - 1)
    return indices.astype(int)


def _euler_from_quaternion(quaternion):
    ned_to_body = quaternion_to_dcm(quaternion)
    phi = np.arctan2(ned_to_body[1, 2], ned_to_body[2, 2])
    theta = np.arcsin(np.clip(-ned_to_body[0, 2], -1.0, 1.0))
    psi = np.arctan2(ned_to_body[0, 1], ned_to_body[0, 0])
    return phi, theta, psi


def _validate_history(
    times, states, controls, command_history, phase_history, rate_command_history
):
    times = np.asarray(times, dtype=float)
    states = np.asarray(states, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times must be a nonempty one-dimensional array")
    if states.shape != (times.size, 14):
        raise ValueError("states must have shape (number_of_times, 14)")
    if not np.all(np.isfinite(states)):
        raise ValueError("states must be finite")
    if controls is not None:
        controls = np.asarray(controls, dtype=float)
        if controls.shape != (times.size, 4) or not np.all(np.isfinite(controls)):
            raise ValueError("controls must have shape (number_of_times, 4) and be finite")
    if command_history is not None:
        command_history = np.asarray(command_history, dtype=float)
        if command_history.shape != (times.size, 2):
            raise ValueError("command_history must have shape (number_of_times, 2)")
        if not np.all(np.isfinite(command_history)):
            raise ValueError("command_history must be finite")
    if phase_history is not None:
        phase_history = np.asarray(phase_history, dtype=str)
        if phase_history.shape != (times.size,):
            raise ValueError("phase_history must have shape (number_of_times,)")
    if rate_command_history is not None:
        rate_command_history = np.asarray(rate_command_history, dtype=float)
        if rate_command_history.shape != (times.size,):
            raise ValueError(
                "rate_command_history must have shape (number_of_times,)"
            )
        if not np.all(np.isfinite(rate_command_history)):
            raise ValueError("rate_command_history must be finite")
    return (
        times,
        states,
        controls,
        command_history,
        phase_history,
        rate_command_history,
    )


def create_flight_animation(
    times,
    states,
    controls=None,
    command_history=None,
    fps=30,
    playback_speed=3.0,
    camera="chase",
    view_size=4000.0,
    vertical_view_size=1500.0,
    aircraft_scale=75.0,
    trail_duration=10.0,
    show_full_trajectory=True,
    phase_history=None,
    rate_command_history=None,
    look_ahead_distance=1000.0,
    camera_smoothing_time=2.5,
):
    """Create a 3D flight animation from an already simulated history."""
    histories = _validate_history(
        times,
        states,
        controls,
        command_history,
        phase_history,
        rate_command_history,
    )
    times, states, controls, command_history, phase_history, rate_command_history = histories
    if camera not in ("fixed", "chase"):
        raise ValueError("camera must be 'fixed' or 'chase'")
    for value, name in (
        (view_size, "view_size"),
        (vertical_view_size, "vertical_view_size"),
        (aircraft_scale, "aircraft_scale"),
        (trail_duration, "trail_duration"),
        (look_ahead_distance, "look_ahead_distance"),
        (camera_smoothing_time, "camera_smoothing_time"),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    frame_indices = select_animation_frames(times, fps, playback_speed)
    position_ned_ft = states[:, :3] / FT_TO_METER
    position_plot_ft = ned_to_plot_coordinates(position_ned_ft)
    vertices, faces, face_groups = create_f16_low_poly_geometry()
    scaled_vertices = aircraft_scale * vertices
    material_colors = {
        "airframe": "0.015",
        "wing": "0.015",
        "lerx": "0.018",
        "stabilator": "0.015",
        "tail": "0.015",
        "canopy": "0.20",
        "nozzle": "0.22",
        "nozzle_interior": "0.035",
        "intake": "0.045",
        "intake_opening": "0.005",
    }
    face_colors = [material_colors[group] for group in face_groups]

    figure = plt.figure(figsize=(12.0, 8.0))
    axis = figure.add_subplot(111, projection="3d")
    if show_full_trajectory:
        axis.plot(
            position_plot_ft[:, 0],
            position_plot_ft[:, 1],
            position_plot_ft[:, 2],
            color="0.78",
            linewidth=0.8,
            alpha=0.22,
            label="Full trajectory",
        )
    past_line, = axis.plot(
        [], [], [], color="0.50", linewidth=1.0, alpha=0.55, label="Past path"
    )
    trail_line, = axis.plot(
        [], [], [], color="C0", linewidth=2.6, alpha=0.95, label="Recent trail"
    )
    initial_geometry = transform_body_geometry(
        scaled_vertices,
        states[0, 6:10],
        position_ned_ft[0],
    )
    aircraft = Poly3DCollection(
        [initial_geometry[np.asarray(face)] for face in faces],
        facecolor=face_colors,
        edgecolor="0.28",
        linewidth=0.7,
        alpha=0.98,
    )
    axis.add_collection3d(aircraft)

    north_min, east_min = np.min(position_plot_ft[:, :2], axis=0)
    north_max, east_max = np.max(position_plot_ft[:, :2], axis=0)
    ground_north = np.linspace(north_min - 500.0, north_max + 500.0, 8)
    ground_east = np.linspace(east_min - 500.0, east_max + 500.0, 8)
    ground_n, ground_e = np.meshgrid(ground_north, ground_east)
    axis.plot_surface(
        ground_n,
        ground_e,
        np.zeros_like(ground_n),
        color="0.85",
        alpha=0.08,
        linewidth=0.0,
    )

    left_hud = axis.text2D(
        0.02,
        0.95,
        "",
        transform=axis.transAxes,
        va="top",
        family="monospace",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "0.75"},
    )
    right_hud = axis.text2D(
        0.76,
        0.95,
        "",
        transform=axis.transAxes,
        va="top",
        family="monospace",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "0.75"},
    )
    control_hud = axis.text2D(
        0.02,
        0.27,
        "",
        transform=axis.transAxes,
        va="top",
        family="monospace",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "0.75"},
    )
    phase_hud = axis.text2D(
        0.5,
        0.95,
        "",
        transform=axis.transAxes,
        ha="center",
        va="top",
        weight="bold",
        fontsize=11,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.4"},
    )
    axis.set_xlabel("North [ft]")
    axis.set_ylabel("East [ft]")
    axis.set_zlabel("Altitude gain [ft]")
    axis.set_title("F-16 Nonlinear 6DoF Controlled Flight")
    axis.view_init(elev=24.0, azim=-58.0)
    axis.grid(True, alpha=0.22)
    axis.legend(loc="lower right", fontsize=8, framealpha=0.65)
    axis.set_box_aspect((1.0, 1.0, 0.5))
    for coordinate_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
        coordinate_axis.pane.set_facecolor((0.97, 0.97, 0.97, 0.18))
        coordinate_axis.pane.set_edgecolor((0.75, 0.75, 0.75, 0.35))
        coordinate_axis._axinfo["grid"]["color"] = (0.55, 0.55, 0.55, 0.22)
        coordinate_axis._axinfo["grid"]["linewidth"] = 0.6
    axis.tick_params(labelsize=8, colors="0.35")

    if camera == "fixed":
        horizontal_span = max(north_max - north_min, east_max - east_min, 1.0)
        horizontal_margin = 0.08 * horizontal_span
        altitude_min = np.min(position_plot_ft[:, 2])
        altitude_max = np.max(position_plot_ft[:, 2])
        altitude_margin = max(0.1 * (altitude_max - altitude_min), 100.0)
        axis.set_xlim(north_min - horizontal_margin, north_max + horizontal_margin)
        axis.set_ylim(east_min - horizontal_margin, east_max + horizontal_margin)
        axis.set_zlim(altitude_min - altitude_margin, altitude_max + altitude_margin)

    world_center_offset = np.array(
        [
            0.70 * look_ahead_distance,
            0.40 * look_ahead_distance,
            0.15 * look_ahead_distance,
        ]
    )
    camera_state = {
        "center": position_plot_ft[0] + world_center_offset,
        "azimuth": -58.0,
        "elevation": 24.0,
        "horizontal_size": float(view_size),
        "vertical_size": float(vertical_view_size),
        "last_time": float(times[0]),
    }

    def update(frame_number):
        index = int(frame_indices[frame_number])
        trail_start_time = times[index] - trail_duration
        trail_start = int(np.searchsorted(times, trail_start_time, side="left"))
        trail = position_plot_ft[trail_start : index + 1]
        past = position_plot_ft[: index + 1]
        past_line.set_data_3d(past[:, 0], past[:, 1], past[:, 2])
        trail_line.set_data_3d(trail[:, 0], trail[:, 1], trail[:, 2])

        transformed = transform_body_geometry(
            scaled_vertices,
            states[index, 6:10],
            position_ned_ft[index],
        )
        aircraft.set_verts([transformed[np.asarray(face)] for face in faces])

        north, east, altitude = position_plot_ft[index]
        if camera == "chase":
            phi_rad, _, _ = _euler_from_quaternion(states[index, 6:10])
            target_center = position_plot_ft[index] + world_center_offset
            body_rate = np.linalg.norm(states[index, 10:13])
            activity = np.clip(
                max(abs(phi_rad) / np.deg2rad(70.0), body_rate / np.deg2rad(80.0)),
                0.0,
                1.0,
            )
            target_horizontal_size = view_size * (1.0 + 0.15 * activity)
            target_vertical_size = vertical_view_size * (1.0 + 0.12 * activity)
            target_elevation = 24.0
            if frame_number == 0:
                smoothing = 1.0
            else:
                frame_dt = max(times[index] - camera_state["last_time"], 0.0)
                smoothing = 1.0 - np.exp(-frame_dt / camera_smoothing_time)
            camera_state["center"] += smoothing * (
                target_center - camera_state["center"]
            )
            azimuth_target = -58.0
            azimuth_error = (
                azimuth_target - camera_state["azimuth"] + 180.0
            ) % 360.0 - 180.0
            camera_state["azimuth"] += smoothing * azimuth_error
            camera_state["elevation"] += smoothing * (
                target_elevation - camera_state["elevation"]
            )
            camera_state["horizontal_size"] += smoothing * (
                target_horizontal_size - camera_state["horizontal_size"]
            )
            camera_state["vertical_size"] += smoothing * (
                target_vertical_size - camera_state["vertical_size"]
            )
            camera_state["last_time"] = float(times[index])
            center_north, center_east, center_altitude = camera_state["center"]
            horizontal_size = camera_state["horizontal_size"]
            vertical_size = camera_state["vertical_size"]
            axis.set_xlim(
                center_north - 0.5 * horizontal_size,
                center_north + 0.5 * horizontal_size,
            )
            axis.set_ylim(
                center_east - 0.5 * horizontal_size,
                center_east + 0.5 * horizontal_size,
            )
            axis.set_zlim(
                center_altitude - 0.5 * vertical_size,
                center_altitude + 0.5 * vertical_size,
            )
            axis.view_init(
                elev=camera_state["elevation"],
                azim=camera_state["azimuth"],
            )

        true_airspeed, alpha_deg, beta_deg = air_data_from_body_velocity(
            states[index, 3:6]
        )
        _, mach, _ = f16_air_data(true_airspeed, -states[index, 2])
        phi, theta, psi = np.rad2deg(_euler_from_quaternion(states[index, 6:10]))
        p, q, r = np.rad2deg(states[index, 10:13])
        altitude_gain = altitude - position_plot_ft[0, 2]
        left_hud.set_text(
            "FLIGHT\n"
            f"VT       {true_airspeed / FT_TO_METER:6.1f} ft/s\n"
            f"Mach     {mach:6.2f}\n"
            f"Alt gain {altitude_gain:6.0f} ft\n"
            f"alpha    {alpha_deg:6.1f} deg\n"
            f"beta     {beta_deg:6.1f} deg"
        )
        right_hud.set_text(
            "ATTITUDE\n"
            f"phi   {phi:6.1f} deg\n"
            f"theta {theta:6.1f} deg\n"
            f"psi   {psi:6.1f} deg\n\n"
            "RATES\n"
            f"p     {p:6.1f} deg/s\n"
            f"q     {q:6.1f} deg/s\n"
            f"r     {r:6.1f} deg/s"
        )
        if controls is None:
            control_text = "Controls: not supplied"
        else:
            throttle, elevator, aileron, rudder = controls[index]
            control_text = (
                "CONTROLS\n"
                f"throttle {throttle:6.2f}\n"
                f"elevator {elevator:6.1f} deg\n"
                f"aileron  {aileron:6.1f} deg\n"
                f"rudder   {rudder:6.1f} deg"
            )
        if command_history is not None:
            theta_command, phi_command = np.rad2deg(command_history[index])
            control_text += (
                f"\ntheta cmd {theta_command:5.1f} deg"
                f"\nphi cmd   {phi_command:5.1f} deg"
            )
        if rate_command_history is not None:
            control_text += (
                f"\np cmd     {np.rad2deg(rate_command_history[index]):5.1f} deg/s"
            )
        control_hud.set_text(control_text)
        phase_text = "" if phase_history is None else phase_history[index]
        phase_hud.set_text(f"t = {times[index]:5.1f} s   |   {phase_text}")
        return (
            past_line,
            trail_line,
            aircraft,
            left_hud,
            right_hud,
            control_hud,
            phase_hud,
        )

    animation = FuncAnimation(
        figure,
        update,
        frames=frame_indices.size,
        interval=1000.0 / float(fps),
        blit=False,
        repeat=True,
    )
    return animation
