"""Analyze the cascaded bank-angle loop with roll and yaw damping active."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.f16sim.control_analysis import lateral_bank_angle_feedback_poles
from src.f16sim.lateral_linearization import linearize_lateral
from src.f16sim.parameters import FT_TO_METER
from src.f16sim.trim import trim_straight_level


KR = 50.0
REPRESENTATIVE_GAINS = np.array([0.0, 0.1, 0.2, 0.5, 1.0, 2.0])


def _participation(vector):
    values = np.abs(vector)
    return values / np.sum(values)


def _modal_summary(matrix):
    poles, vectors = np.linalg.eig(matrix)
    positive_complex = [i for i, pole in enumerate(poles) if pole.imag > 1e-8]
    dutch_index = max(
        positive_complex,
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
        "bank_related": poles[spiral_index],
        "stable": bool(np.all(poles.real < 0.0)),
    }


def _matrix(A_lat, B_lat, k_phi, Kp):
    bank_output = np.array([[0.0, 1.0, 0.0, 0.0]])
    roll_rate_output = np.array([[0.0, 0.0, 1.0, 0.0]])
    yaw_rate_output = np.array([[0.0, 0.0, 0.0, 1.0]])
    return (
        A_lat
        + KR * (B_lat[:, 1:2] @ yaw_rate_output)
        + Kp * (B_lat[:, 0:1] @ (roll_rate_output + k_phi * bank_output))
    )


def create_bank_angle_analysis_figure(Kp=5.0):
    """Return a bank-loop pole analysis for a configurable candidate Kp."""
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

    gains = np.linspace(0.0, 2.0, 501)
    poles = lateral_bank_angle_feedback_poles(
        A_lat, B_lat, gains, Kp=Kp, Kr=KR
    )
    bank_related = np.empty(gains.size, dtype=complex)
    summaries = []
    for index, gain in enumerate(gains):
        summary = _modal_summary(_matrix(A_lat, B_lat, gain, Kp))
        summaries.append(summary)
        bank_related[index] = summary["bank_related"]

    print(f"Illustrative configurable Kp = {Kp:g} deg/(rad/s); not selected")
    for gain in REPRESENTATIVE_GAINS:
        index = int(np.argmin(np.abs(gains - gain)))
        summary = summaries[index]
        print(f"\nKphi = {gain:g} 1/s")
        print(f"  all poles: {np.sort_complex(summary['poles'])}")
        print(f"  roll pole: {summary['roll']}")
        print(f"  Dutch-roll poles: {summary['dutch']}")
        print(f"  Dutch-roll damping ratio: {summary['dutch_zeta']:.9f}")
        print(f"  slow bank-angle-related pole: {summary['bank_related']}")
        print(f"  stable: {summary['stable']}")

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.5))
    pole_axis, slow_axis = axes
    for branch in range(4):
        pole_axis.plot(poles[:, branch].real, poles[:, branch].imag)
    pole_axis.scatter(poles[0].real, poles[0].imag, marker="x", color="black")
    pole_axis.axhline(0.0, color="black", linewidth=0.8)
    pole_axis.axvline(0.0, color="black", linewidth=0.8)
    pole_axis.set_xlabel("Real axis [1/s]")
    pole_axis.set_ylabel("Imaginary axis [rad/s]")
    pole_axis.set_title("Bank-loop pole trajectories")
    pole_axis.grid(True)

    slow_axis.plot(gains, bank_related.real)
    slow_axis.axhline(0.0, color="black", linewidth=0.8)
    slow_axis.set_xlabel(r"$K_\phi$ [1/s]")
    slow_axis.set_ylabel("Slow pole real part [1/s]")
    slow_axis.set_title("Slow bank-angle-related pole")
    slow_axis.grid(True)

    figure.suptitle(
        f"F-16 Bank-Angle Loop Analysis: Kr = 50, candidate Kp = {Kp:g}"
    )
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kp",
        type=float,
        default=5.0,
        help="Illustrative roll-rate feedback gain in deg/(rad/s)",
    )
    arguments = parser.parse_args()
    create_bank_angle_analysis_figure(Kp=arguments.kp)
    plt.show()
