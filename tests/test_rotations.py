import numpy as np

from src.f16sim.rotations import (
    rotation_x,
    rotation_y,
    rotation_z,
    ned_to_body,
    body_to_ned,
)

# ============================================================
# Rotation about X-axis
# ============================================================

def test_rotation_x_zero_is_identity():
    C = rotation_x(0.0)

    assert np.allclose(C, np.eye(3))


def test_rotation_x_is_orthogonal():
    phi = np.deg2rad(30.0)

    C = rotation_x(phi)

    assert np.allclose(C.T @ C, np.eye(3))


def test_rotation_x_determinant_is_one():
    phi = np.deg2rad(30.0)

    C = rotation_x(phi)

    assert np.isclose(np.linalg.det(C), 1.0)


# ============================================================
# Rotation about Y-axis
# ============================================================

def test_rotation_y_zero_is_identity():
    C = rotation_y(0.0)

    assert np.allclose(C, np.eye(3))


def test_rotation_y_is_orthogonal():
    theta = np.deg2rad(30.0)

    C = rotation_y(theta)

    assert np.allclose(C.T @ C, np.eye(3))


def test_rotation_y_determinant_is_one():
    theta = np.deg2rad(30.0)

    C = rotation_y(theta)

    assert np.isclose(np.linalg.det(C), 1.0)


# ============================================================
# Rotation about Z-axis
# ============================================================

def test_rotation_z_zero_is_identity():
    C = rotation_z(0.0)

    assert np.allclose(C, np.eye(3))


def test_rotation_z_is_orthogonal():
    psi = np.deg2rad(30.0)

    C = rotation_z(psi)

    assert np.allclose(C.T @ C, np.eye(3))


def test_rotation_z_determinant_is_one():
    psi = np.deg2rad(30.0)

    C = rotation_z(psi)

    assert np.isclose(np.linalg.det(C), 1.0)

def test_ned_to_body_zero_angles_is_identity():
    C = ned_to_body(0.0, 0.0, 0.0)

    assert np.allclose(C, np.eye(3))


def test_body_to_ned_is_transpose():
    phi = np.deg2rad(20.0)
    theta = np.deg2rad(10.0)
    psi = np.deg2rad(30.0)

    C_nb = ned_to_body(phi, theta, psi)
    C_bn = body_to_ned(phi, theta, psi)

    assert np.allclose(C_bn, C_nb.T)


def test_ned_body_round_trip():
    phi = np.deg2rad(20.0)
    theta = np.deg2rad(10.0)
    psi = np.deg2rad(30.0)

    v_ned = np.array([100.0, 20.0, -5.0])

    v_body = ned_to_body(phi, theta, psi) @ v_ned
    v_ned_recovered = body_to_ned(phi, theta, psi) @ v_body

    assert np.allclose(v_ned_recovered, v_ned)