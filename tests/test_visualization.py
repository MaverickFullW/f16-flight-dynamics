import numpy as np

from src.f16sim.attitude import euler_to_quaternion
from src.f16sim.visualization import (
    create_f16_low_poly_geometry,
    create_fighter_geometry,
    ned_to_plot_coordinates,
    select_animation_frames,
    transform_body_geometry,
)


def test_detailed_f16_geometry_has_expected_extents_and_material_groups():
    vertices, faces, groups = create_f16_low_poly_geometry()

    assert vertices.ndim == 2
    assert vertices.shape[1] == 3
    assert np.all(np.isfinite(vertices))
    assert len(faces) == len(groups)
    assert vertices[:, 0].max() == vertices[0, 0]
    assert vertices[:, 0].min() < -4.0
    assert vertices[:, 1].min() < 0.0 < vertices[:, 1].max()
    assert vertices[:, 2].min() < -2.0
    assert {"airframe", "canopy", "nozzle", "intake"} <= set(groups)
    for face in faces:
        assert len(face) >= 3
        assert min(face) >= 0
        assert max(face) < vertices.shape[0]


def test_detailed_f16_geometry_is_left_right_symmetric():
    vertices, _, _ = create_f16_low_poly_geometry()

    reflected = vertices.copy()
    reflected[:, 1] *= -1.0
    for vertex in reflected:
        assert np.any(np.all(np.isclose(vertices, vertex, atol=1e-12), axis=1))


def test_geometry_creation_returns_independent_arrays_and_data():
    first_vertices, first_faces, first_groups = create_f16_low_poly_geometry()
    original_vertices = first_vertices.copy()
    first_vertices[0] = 0.0

    second_vertices, second_faces, second_groups = create_f16_low_poly_geometry()

    assert np.array_equal(second_vertices, original_vertices)
    assert second_faces == first_faces
    assert second_groups == first_groups


def test_f16_geometry_has_distinct_vertical_features_and_aft_stabilators():
    vertices, faces, groups = create_f16_low_poly_geometry()

    canopy_indices = {
        index
        for face, group in zip(faces, groups)
        if group == "canopy"
        for index in face
    }
    intake_indices = {
        index
        for face, group in zip(faces, groups)
        if group in ("intake", "intake_opening")
        for index in face
    }
    stabilator_indices = {
        index
        for face, group in zip(faces, groups)
        if group == "stabilator"
        for index in face
    }
    wing_indices = {
        index
        for face, group in zip(faces, groups)
        if group == "wing"
        for index in face
    }

    assert np.min(vertices[list(canopy_indices), 2]) < -1.3
    assert np.max(vertices[list(intake_indices), 2]) > 0.9
    assert np.max(vertices[list(stabilator_indices), 0]) < np.min(
        vertices[list(wing_indices), 0]
    )


def test_fighter_geometry_has_finite_vertices_and_valid_faces():
    vertices, faces = create_fighter_geometry()

    assert vertices.ndim == 2
    assert vertices.shape[1] == 3
    assert np.all(np.isfinite(vertices))
    assert len(faces) > 0
    for face in faces:
        assert len(face) >= 3
        assert min(face) >= 0
        assert max(face) < vertices.shape[0]


def test_fighter_geometry_uses_forward_right_down_body_orientation():
    vertices, _ = create_fighter_geometry()

    assert vertices[:, 0].max() > abs(vertices[:, 0].min())
    assert vertices[:, 1].max() > 0.0
    assert vertices[:, 1].min() < 0.0
    assert vertices[:, 2].min() < 0.0


def test_identity_quaternion_preserves_horizontal_body_axes_for_plotting():
    vertices = np.eye(3)
    transformed = transform_body_geometry(
        vertices,
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        position_ned=np.zeros(3),
    )

    assert np.array_equal(
        transformed,
        np.array([[1.0, 0.0, -0.0], [0.0, 1.0, -0.0], [0.0, 0.0, -1.0]]),
    )


def test_ninety_degree_yaw_rotates_body_forward_axis_toward_east():
    forward_vertex = np.array([[1.0, 0.0, 0.0]])
    quaternion = euler_to_quaternion(0.0, 0.0, np.deg2rad(90.0))

    transformed = transform_body_geometry(
        forward_vertex,
        quaternion=quaternion,
        position_ned=np.zeros(3),
    )

    assert np.allclose(transformed[0], [0.0, 1.0, 0.0], atol=1e-12)


def test_ninety_degree_pitch_rotates_body_forward_axis_upward():
    forward_vertex = np.array([[1.0, 0.0, 0.0]])
    quaternion = euler_to_quaternion(0.0, np.deg2rad(90.0), 0.0)

    transformed = transform_body_geometry(
        forward_vertex,
        quaternion=quaternion,
        position_ned=np.zeros(3),
    )

    assert np.allclose(transformed[0], [0.0, 0.0, 1.0], atol=1e-12)


def test_positive_roll_rotates_body_right_axis_downward_in_plot_coordinates():
    body_axes = np.eye(3)
    quaternion = euler_to_quaternion(np.deg2rad(90.0), 0.0, 0.0)

    transformed = transform_body_geometry(
        body_axes, quaternion=quaternion, position_ned=np.zeros(3)
    )

    assert np.allclose(transformed[1], [0.0, 0.0, -1.0], atol=1e-12)


def test_negative_roll_rotates_body_right_axis_upward_in_plot_coordinates():
    body_axes = np.eye(3)
    quaternion = euler_to_quaternion(np.deg2rad(-90.0), 0.0, 0.0)

    transformed = transform_body_geometry(
        body_axes, quaternion=quaternion, position_ned=np.zeros(3)
    )

    assert np.allclose(transformed[1], [0.0, 0.0, 1.0], atol=1e-12)


def test_ned_to_plot_coordinates_flips_only_down_coordinate():
    points = np.array([[10.0, 20.0, 30.0], [-1.0, -2.0, -3.0]])

    plotted = ned_to_plot_coordinates(points)

    assert np.array_equal(plotted, [[10.0, 20.0, -30.0], [-1.0, -2.0, 3.0]])


def test_frame_selection_respects_fps_and_playback_speed():
    times = np.linspace(0.0, 30.0, 3001)

    indices = select_animation_frames(times, fps=30, playback_speed=3.0)

    assert indices[0] == 0
    assert indices[-1] == times.size - 1
    assert indices.size == 301
    assert np.allclose(np.diff(times[indices]), 0.1, atol=1e-12)


def test_visualization_math_does_not_mutate_inputs():
    vertices = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    position = np.array([4.0, 5.0, 6.0])
    times = np.linspace(0.0, 1.0, 101)
    states = np.zeros((times.size, 14))
    states[:, :3] = np.column_stack((times, 2.0 * times, -3.0 * times))
    originals = tuple(
        value.copy() for value in (vertices, quaternion, position, times, states)
    )

    transform_body_geometry(vertices, quaternion, position)
    ned_to_plot_coordinates(states[:, :3])
    select_animation_frames(times, fps=20, playback_speed=2.0)

    for actual, expected in zip(
        (vertices, quaternion, position, times, states), originals
    ):
        assert np.array_equal(actual, expected)
