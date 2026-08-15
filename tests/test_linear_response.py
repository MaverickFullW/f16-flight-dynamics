import numpy as np
import pytest

from src.f16sim.linear_response import simulate_linear_longitudinal


def test_simulate_linear_longitudinal_returns_expected_shapes():
    times, perturbations = simulate_linear_longitudinal(
        A=np.diag([-1.0, -2.0, -3.0, -4.0]),
        B=np.ones((4, 2)),
        initial_perturbation=np.array([1.0, 2.0, 3.0, 4.0]),
        duration=1.0,
        dt=0.1,
        control_perturbation=np.array([0.1, -0.2]),
    )

    assert times.shape == (11,)
    assert perturbations.shape == (11, 4)
    assert np.array_equal(perturbations[0], [1.0, 2.0, 3.0, 4.0])


def test_zero_initial_perturbation_and_zero_control_stays_zero():
    times, perturbations = simulate_linear_longitudinal(
        A=np.ones((4, 4)),
        B=np.ones((4, 2)),
        initial_perturbation=np.zeros(4),
        duration=1.0,
        dt=0.1,
    )

    assert np.array_equal(times, np.linspace(0.0, 1.0, 11))
    assert np.array_equal(perturbations, np.zeros((11, 4)))


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("A", np.zeros((3, 3)), r"A must have shape \(4, 4\)"),
        ("B", np.zeros((4, 3)), r"B must have shape \(4, 2\)"),
        (
            "initial_perturbation",
            np.zeros(3),
            r"initial_perturbation must have shape \(4,\)",
        ),
        (
            "control_perturbation",
            np.zeros(3),
            r"control_perturbation must have shape \(2,\)",
        ),
    ],
)
def test_simulate_linear_longitudinal_rejects_invalid_shapes(
    argument, value, message
):
    arguments = {
        "A": np.zeros((4, 4)),
        "B": np.zeros((4, 2)),
        "initial_perturbation": np.zeros(4),
        "duration": 1.0,
        "dt": 0.1,
    }
    arguments[argument] = value

    with pytest.raises(ValueError, match=message):
        simulate_linear_longitudinal(**arguments)


@pytest.mark.parametrize(
    ("duration", "dt", "message"),
    [
        (0.0, 0.1, "duration must be positive"),
        (-1.0, 0.1, "duration must be positive"),
        (1.0, 0.0, "dt must be positive"),
        (1.0, -0.1, "dt must be positive"),
        (1.0, 0.3, "duration must be an integer number of time steps"),
    ],
)
def test_simulate_linear_longitudinal_rejects_invalid_times(
    duration, dt, message
):
    with pytest.raises(ValueError, match=message):
        simulate_linear_longitudinal(
            np.zeros((4, 4)),
            np.zeros((4, 2)),
            np.zeros(4),
            duration,
            dt,
        )


def test_simulate_linear_longitudinal_is_deterministic():
    arguments = {
        "A": np.array(
            [
                [-0.2, 0.1, 0.0, 0.0],
                [-0.1, -0.3, 0.2, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, -0.4, 0.0, -0.5],
            ]
        ),
        "B": np.arange(8, dtype=float).reshape(4, 2) / 10.0,
        "initial_perturbation": np.array([1.0, -0.1, 0.2, 0.0]),
        "duration": 2.0,
        "dt": 0.05,
        "control_perturbation": np.array([0.01, -0.02]),
    }

    first_times, first_perturbations = simulate_linear_longitudinal(**arguments)
    second_times, second_perturbations = simulate_linear_longitudinal(**arguments)

    assert np.array_equal(first_times, second_times)
    assert np.array_equal(first_perturbations, second_perturbations)


def test_constant_control_behavior_matches_equivalent_callable():
    A = np.diag([-0.1, -0.2, -0.3, -0.4])
    B = np.arange(8, dtype=float).reshape(4, 2) / 10.0
    constant_control = np.array([0.1, -0.2])
    arguments = (A, B, np.array([1.0, 0.0, -0.5, 0.2]), 1.0, 0.1)

    constant_times, constant_response = simulate_linear_longitudinal(
        *arguments, control_perturbation=constant_control
    )
    callable_times, callable_response = simulate_linear_longitudinal(
        *arguments, control_perturbation=lambda time: constant_control
    )

    assert np.array_equal(constant_times, callable_times)
    assert np.array_equal(constant_response, callable_response)


def test_zero_callable_matches_zero_constant_control():
    arguments = (
        np.diag([-0.1, -0.2, -0.3, -0.4]),
        np.ones((4, 2)),
        np.array([1.0, 2.0, 3.0, 4.0]),
        1.0,
        0.1,
    )

    _, constant_response = simulate_linear_longitudinal(
        *arguments, control_perturbation=np.zeros(2)
    )
    _, callable_response = simulate_linear_longitudinal(
        *arguments, control_perturbation=lambda time: np.zeros(2)
    )

    assert np.array_equal(constant_response, callable_response)


def test_time_varying_control_returns_finite_response_with_expected_shape():
    times, perturbations = simulate_linear_longitudinal(
        A=np.diag([-0.1, -0.2, -0.3, -0.4]),
        B=np.ones((4, 2)),
        initial_perturbation=np.zeros(4),
        duration=2.0,
        dt=0.05,
        control_perturbation=lambda time: np.array(
            [0.01 * np.sin(time), -0.2 * np.cos(time)]
        ),
    )

    assert times.shape == (41,)
    assert perturbations.shape == (41, 4)
    assert np.all(np.isfinite(perturbations))


def test_callable_control_rejects_invalid_output_shape():
    with pytest.raises(ValueError, match=r"must return shape \(2,\)"):
        simulate_linear_longitudinal(
            np.zeros((4, 4)),
            np.zeros((4, 2)),
            np.zeros(4),
            duration=1.0,
            dt=0.1,
            control_perturbation=lambda time: np.zeros(3),
        )


def test_elevator_pulse_perturbs_state_and_remains_finite():
    def elevator_pulse(time):
        return np.array([0.0, 0.5 if 0.0 <= time < 0.5 else 0.0])

    times, perturbations = simulate_linear_longitudinal(
        A=np.diag([-0.2, -0.3, -0.4, -0.5]),
        B=np.array(
            [
                [0.0, 0.1],
                [0.0, -0.2],
                [0.0, 0.0],
                [0.0, -1.0],
            ]
        ),
        initial_perturbation=np.zeros(4),
        duration=2.0,
        dt=0.05,
        control_perturbation=elevator_pulse,
    )

    pulse_end = np.searchsorted(times, 0.5)
    assert np.any(perturbations[1:pulse_end] != 0.0)
    assert np.any(perturbations[pulse_end:] != 0.0)
    assert np.all(np.isfinite(perturbations))
