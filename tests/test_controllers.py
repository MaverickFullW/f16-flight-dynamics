import numpy as np
import pytest

from src.f16sim.controllers import pitch_attitude_control


def test_pitch_attitude_control_is_zero_at_zero_error_and_pitch_rate():
    q_cmd, elevator = pitch_attitude_control(0.1, 0.1, 0.0)

    assert q_cmd == 0.0
    assert elevator == 0.0


@pytest.mark.parametrize(
    ("theta_cmd", "theta", "expected_q_command", "elevator_sign"),
    [
        (np.deg2rad(5.0), 0.0, np.deg2rad(2.5), -1.0),
        (np.deg2rad(-5.0), 0.0, np.deg2rad(-2.5), 1.0),
    ],
)
def test_pitch_attitude_control_handles_signed_attitude_errors(
    theta_cmd, theta, expected_q_command, elevator_sign
):
    q_cmd, elevator = pitch_attitude_control(theta_cmd, theta, q=0.0)

    assert q_cmd == pytest.approx(expected_q_command)
    assert np.sign(elevator) == elevator_sign


def test_pitch_attitude_control_applies_positive_pitch_rate_feedback():
    q_cmd, elevator = pitch_attitude_control(0.0, 0.0, q=0.2)

    assert q_cmd == 0.0
    assert elevator == pytest.approx(1.0)


def test_pitch_attitude_control_uses_configurable_gains():
    q_cmd, elevator = pitch_attitude_control(
        theta_cmd=0.2,
        theta=0.1,
        q=0.08,
        Kq=2.0,
        Ktheta=0.25,
    )

    assert q_cmd == pytest.approx(0.025)
    assert elevator == pytest.approx(0.11)


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("theta_cmd", [0.0]),
        ("theta", np.zeros(2)),
        ("q", np.nan),
        ("Kq", np.inf),
        ("Kq", 0.0),
        ("Ktheta", -0.1),
    ],
)
def test_pitch_attitude_control_rejects_invalid_inputs(argument, value):
    arguments = {
        "theta_cmd": 0.1,
        "theta": 0.0,
        "q": 0.0,
        "Kq": 5.0,
        "Ktheta": 0.5,
    }
    arguments[argument] = value

    with pytest.raises(ValueError):
        pitch_attitude_control(**arguments)
