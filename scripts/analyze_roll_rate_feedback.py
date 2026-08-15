"""Analyze aileron roll-rate feedback with the selected yaw damper active."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.f16sim.control_analysis import lateral_roll_rate_feedback_poles
from src.f16sim.lateral_linearization import linearize_lateral
from src.f16sim.parameters import FT_TO_METER
from src.f16sim.trim import trim_straight_level


KR = 50.0
P_OUTPUT = np.array([[0.0, 0.0, 1.0, 0.0]])
R_OUTPUT = np.array([[0.0, 0.0, 0.0, 1.0]])


def _participation(vector):
    values = np.abs(vector)
    return values / np.sum(values)


def _modal_summary(matrix):
    poles, vectors = np.linalg.eig(matrix)
    complex_indices = [i for i, pole in enumerate(poles) if pole.imag > 1e-8]
    if not complex_indices:
        raise RuntimeError("Dutch-roll mode is no longer oscillatory")
    dutch_index = max(
        complex_indices,
        key=lambda i: _participation(vectors[:, i])[0]
        + _participation(vectors[:, i])[3],
    )
    dutch = poles[dutch_index]
    real_indices = [i for i, pole in enumerate(poles) if abs(pole.imag) <= 1e-8]
    roll_index = max(real_indices, key=lambda i: _participation(vectors[:, i])[2])
    spiral_index = max(
        (i for i in real_indices if i != roll_index),
        key=lambda i: _participation(vectors[:, i])[1],
    )
    return {
        "poles": poles,
        "roll": poles[roll_index],
        "dutch": np.array([dutch, dutch.conjugate()]),
        "dutch_zeta": float(-dutch.real / abs(dutch)),
        "spiral": poles[spiral_index],
        "stable": bool(np.all(poles.real < 0.0)),
    }


def _matrix(A_lat, B_lat, gain):
    return (
        A_lat
        + KR * (B_lat[:, 1:2] @ R_OUTPUT)
        + gain * (B_lat[:, 0:1] @ P_OUTPUT)
    )


def create_roll_rate_analysis_figure():
    """Run the signed roll-rate sweep and return its analysis figure."""
    trim = trim_straight_level(
        502.0 * FT_TO_METER,
        altitude_m=0.0,
        cg_fraction=0.30,
    )
    if not trim["success"]:
        raise RuntimeError(f"Unable to obtain the reference trim: {trim['message']}")
    A_lat, B_lat = linearize_lateral(
        trim["state"], trim["throttle"], trim["elevator_deg"], 0.30
    )

    print("Signed Kp diagnostic with Kr = 50 deg/(rad/s)")
    diagnostic = {}
    for gain in (-1.0, 0.0, 1.0):
        summary = _modal_summary(_matrix(A_lat, B_lat, gain))
        diagnostic[gain] = summary
        print(f"Kp = {gain:+g}: poles={np.sort_complex(summary['poles'])}")
        print(
            f"  roll={summary['roll']}, Dutch zeta={summary['dutch_zeta']:.8f}, "
            f"spiral={summary['spiral']}, stable={summary['stable']}"
        )

    candidates = [
        gain
        for gain in (-1.0, 1.0)
        if diagnostic[gain]["stable"]
    ]
    beneficial_gain = min(candidates, key=lambda gain: diagnostic[gain]["roll"].real)
    beneficial_sign = float(np.sign(beneficial_gain))
    print(
        "Beneficial numerical direction: "
        f"{'positive' if beneficial_sign > 0.0 else 'negative'} Kp"
    )

    magnitudes = np.linspace(0.0, 250.0, 1001)
    gains = beneficial_sign * magnitudes
    poles = lateral_roll_rate_feedback_poles(A_lat, B_lat, gains, Kr=KR)
    roll_poles = np.empty(gains.size, dtype=complex)
    spiral_poles = np.empty(gains.size, dtype=complex)
    dutch_zeta = np.empty(gains.size)
    stable = np.empty(gains.size, dtype=bool)
    summaries = []
    for index, gain in enumerate(gains):
        summary = _modal_summary(_matrix(A_lat, B_lat, gain))
        summaries.append(summary)
        roll_poles[index] = summary["roll"]
        spiral_poles[index] = summary["spiral"]
        dutch_zeta[index] = summary["dutch_zeta"]
        stable[index] = summary["stable"]

    for magnitude in (0.0, 1.0, 5.0, 20.0, 100.0, 200.0, 250.0):
        index = int(np.argmin(np.abs(magnitudes - magnitude)))
        summary = summaries[index]
        print(f"\nKp = {gains[index]:g} deg/(rad/s)")
        print(f"  all poles: {np.sort_complex(summary['poles'])}")
        print(f"  roll pole: {summary['roll']}")
        print(f"  Dutch-roll poles: {summary['dutch']}")
        print(f"  Dutch-roll damping ratio: {summary['dutch_zeta']:.9f}")
        print(f"  spiral pole: {summary['spiral']}")
        print(f"  stable: {summary['stable']}")

    unstable_indices = np.flatnonzero(~stable)
    if unstable_indices.size:
        first = unstable_indices[0]
        print(
            "First unstable sampled gain: "
            f"Kp={gains[first]:.6f}, spiral pole={spiral_poles[first]}"
        )

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 9.0))
    pole_axis, roll_axis, dutch_axis, spiral_axis = axes.flat
    for branch in range(4):
        pole_axis.plot(poles[:, branch].real, poles[:, branch].imag)
    pole_axis.scatter(poles[0].real, poles[0].imag, marker="x", color="black")
    pole_axis.axhline(0.0, color="black", linewidth=0.8)
    pole_axis.axvline(0.0, color="black", linewidth=0.8)
    pole_axis.set_title("Full pole trajectories")
    pole_axis.set_xlabel("Real axis [1/s]")
    pole_axis.set_ylabel("Imaginary axis [rad/s]")
    pole_axis.grid(True)

    roll_axis.plot(gains, roll_poles.real)
    roll_axis.set_title("Roll-subsidence pole")
    roll_axis.set_xlabel(r"$K_p$ [deg/(rad/s)]")
    roll_axis.set_ylabel("Roll pole real part [1/s]")
    roll_axis.grid(True)

    dutch_axis.plot(gains, dutch_zeta)
    dutch_axis.set_title("Dutch-roll damping ratio")
    dutch_axis.set_xlabel(r"$K_p$ [deg/(rad/s)]")
    dutch_axis.set_ylabel(r"$\zeta_{DR}$")
    dutch_axis.grid(True)

    spiral_axis.plot(gains, spiral_poles.real)
    spiral_axis.axhline(0.0, color="black", linewidth=0.8)
    spiral_axis.set_title("Spiral-mode real part")
    spiral_axis.set_xlabel(r"$K_p$ [deg/(rad/s)]")
    spiral_axis.set_ylabel("Spiral pole real part [1/s]")
    spiral_axis.grid(True)

    figure.suptitle("F-16 Roll-Rate Feedback Analysis with Kr = 50")
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    create_roll_rate_analysis_figure()
    plt.show()
