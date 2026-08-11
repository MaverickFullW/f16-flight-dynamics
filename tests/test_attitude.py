import numpy as np
import pytest

from src.f16sim.attitude import (
    euler_to_quaternion,
    quaternion_normalize,
    quaternion_rate,
    quaternion_to_dcm,
)
from src.f16sim.rotations import rotation_x, rotation_y, rotation_z


def test_quaternion_normalize_produces_unit_quaternion():
    q = np.array([1.0, 2.0, 3.0, 4.0])

    q_normalized = quaternion_normalize(q)

    assert np.isclose(np.linalg.norm(q_normalized), 1.0)


def test_quaternion_normalize_raises_for_zero_quaternion():
    with pytest.raises(ValueError):
        quaternion_normalize([0.0, 0.0, 0.0, 0.0])


def test_identity_quaternion_returns_identity_dcm():
    dcm = quaternion_to_dcm([1.0, 0.0, 0.0, 0.0])

    assert np.allclose(dcm, np.eye(3))


def test_x_axis_quaternion_matches_rotation_x():
    angle = np.deg2rad(30.0)
    q = [np.cos(angle / 2.0), np.sin(angle / 2.0), 0.0, 0.0]

    assert np.allclose(quaternion_to_dcm(q), rotation_x(angle))


def test_y_axis_quaternion_matches_rotation_y():
    angle = np.deg2rad(30.0)
    q = [np.cos(angle / 2.0), 0.0, np.sin(angle / 2.0), 0.0]

    assert np.allclose(quaternion_to_dcm(q), rotation_y(angle))


def test_z_axis_quaternion_matches_rotation_z():
    angle = np.deg2rad(30.0)
    q = [np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)]

    assert np.allclose(quaternion_to_dcm(q), rotation_z(angle))


def test_quaternion_to_dcm_is_proper_orthogonal_matrix():
    q = quaternion_normalize([1.0, 2.0, 3.0, 4.0])

    dcm = quaternion_to_dcm(q)

    assert np.allclose(dcm.T @ dcm, np.eye(3))
    assert np.isclose(np.linalg.det(dcm), 1.0)


def test_quaternion_rate_for_identity_and_roll_rate():
    p = 0.6

    q_dot = quaternion_rate([1.0, 0.0, 0.0, 0.0], [p, 0.0, 0.0])

    assert np.allclose(q_dot, [0.0, p / 2.0, 0.0, 0.0])


def test_euler_to_quaternion_matches_ned_to_body_rotation_sequence():
    phi = np.deg2rad(20.0)
    theta = np.deg2rad(10.0)
    psi = np.deg2rad(30.0)

    q = euler_to_quaternion(phi, theta, psi)
    dcm_from_quaternion = quaternion_to_dcm(q)
    dcm_from_rotations = rotation_x(phi) @ rotation_y(theta) @ rotation_z(psi)

    assert np.allclose(dcm_from_quaternion, dcm_from_rotations)
