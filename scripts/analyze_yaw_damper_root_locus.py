"""Analyze yaw-rate-to-rudder feedback for the lateral F-16 model."""

from itertools import permutations
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.f16sim.lateral_linearization import linearize_lateral
from src.f16sim.parameters import FT_TO_METER
from src.f16sim.trim import trim_straight_level


R_OUTPUT = np.array([[0.0, 0.0, 0.0, 1.0]])


def _closed_loop_matrix(A_lat, B_lat, gain):
    rudder_column = B_lat[:, 1:2]
    return A_lat + gain * (rudder_column @ R_OUTPUT)


def _normalized_participation(eigenvector):
    magnitude = np.abs(eigenvector)
    total = np.sum(magnitude)
    return magnitude / total if total > 0.0 else magnitude


def _classify_open_loop(eigenvalues, eigenvectors):
    positive_complex = [
        index for index, pole in enumerate(eigenvalues) if pole.imag > 1e-8
    ]
    if not positive_complex:
        raise RuntimeError("No oscillatory Dutch-roll pair could be identified")
    dutch_positive = max(
        positive_complex,
        key=lambda index: sum(
            _normalized_participation(eigenvectors[:, index])[[0, 3]]
        ),
    )
    dutch_negative = min(
        range(eigenvalues.size),
        key=lambda index: abs(
            eigenvalues[index] - eigenvalues[dutch_positive].conjugate()
        ),
    )
    real_indices = [
        index for index, pole in enumerate(eigenvalues) if abs(pole.imag) <= 1e-8
    ]
    if len(real_indices) < 2:
        raise RuntimeError("Roll and spiral real modes could not be identified")
    roll = max(
        real_indices,
        key=lambda index: _normalized_participation(eigenvectors[:, index])[2],
    )
    spiral = max(
        (index for index in real_indices if index != roll),
        key=lambda index: _normalized_participation(eigenvectors[:, index])[1],
    )
    return {
        "dutch_positive": dutch_positive,
        "dutch_negative": dutch_negative,
        "roll": roll,
        "spiral": spiral,
    }


def _dutch_properties(poles, classification):
    pair = poles[
        [classification["dutch_positive"], classification["dutch_negative"]]
    ]
    oscillatory = pair[np.abs(pair.imag) > 1e-8]
    if oscillatory.size == 0:
        return np.nan, np.nan, np.nan
    pole = oscillatory[np.argmax(oscillatory.imag)]
    natural_frequency = abs(pole)
    damping_ratio = -pole.real / natural_frequency
    return natural_frequency, damping_ratio, abs(pole.imag)


def _match_to_previous(previous, current):
    best_permutation = min(
        permutations(range(current.size)),
        key=lambda order: sum(
            abs(previous[index] - current[order[index]])
            for index in range(current.size)
        ),
    )
    return current[list(best_permutation)]


def _diagnose_feedback_sign(A_lat, B_lat):
    results = {}
    for gain in (-1.0, 0.0, 1.0):
        matrix = _closed_loop_matrix(A_lat, B_lat, gain)
        poles, eigenvectors = np.linalg.eig(matrix)
        classification = _classify_open_loop(poles, eigenvectors)
        natural_frequency, damping_ratio, damped_frequency = _dutch_properties(
            poles, classification
        )
        results[gain] = damping_ratio
        print(f"Kr = {gain:+g} deg/(rad/s)")
        print(f"  poles: {np.sort_complex(poles)}")
        print(f"  Dutch-roll zeta: {damping_ratio:.9f}")
        print(f"  Dutch-roll wn: {natural_frequency:.9f} rad/s")
        print(f"  Dutch-roll damped frequency: {damped_frequency:.9f} rad/s")

    beneficial_sign = 1.0 if results[1.0] > results[-1.0] else -1.0
    print(
        "Beneficial numerical feedback direction from diagnostic: "
        f"{'positive' if beneficial_sign > 0.0 else 'negative'} Kr"
    )
    return beneficial_sign


def _track_sweep(A_lat, B_lat, gains):
    initial_poles, initial_vectors = np.linalg.eig(
        _closed_loop_matrix(A_lat, B_lat, gains[0])
    )
    classification = _classify_open_loop(initial_poles, initial_vectors)
    tracked = np.empty((gains.size, 4), dtype=complex)
    tracked[0] = initial_poles
    for index, gain in enumerate(gains[1:], start=1):
        current = np.linalg.eigvals(_closed_loop_matrix(A_lat, B_lat, gain))
        tracked[index] = _match_to_previous(tracked[index - 1], current)

    natural_frequency = np.full(gains.size, np.nan)
    damping_ratio = np.full(gains.size, np.nan)
    damped_frequency = np.full(gains.size, np.nan)
    roll_poles = np.empty(gains.size, dtype=complex)
    spiral_poles = np.empty(gains.size, dtype=complex)
    for index, gain in enumerate(gains):
        current_poles, current_vectors = np.linalg.eig(
            _closed_loop_matrix(A_lat, B_lat, gain)
        )
        current_classification = _classify_open_loop(
            current_poles, current_vectors
        )
        (
            natural_frequency[index],
            damping_ratio[index],
            damped_frequency[index],
        ) = _dutch_properties(current_poles, current_classification)
        roll_poles[index] = current_poles[current_classification["roll"]]
        spiral_poles[index] = current_poles[current_classification["spiral"]]
    return (
        tracked,
        classification,
        natural_frequency,
        damping_ratio,
        damped_frequency,
        roll_poles,
        spiral_poles,
    )


def _print_representative(
    gain,
    poles,
    natural_frequency,
    damping_ratio,
    roll_pole,
    spiral_pole,
):
    print(f"\nKr = {gain:g} deg/(rad/s)")
    print(f"  all poles: {np.sort_complex(poles)}")
    print(f"  Dutch-roll zeta: {damping_ratio:.9f}")
    print(f"  Dutch-roll wn: {natural_frequency:.9f} rad/s")
    print(f"  roll pole: {roll_pole}")
    print(f"  spiral pole: {spiral_pole}")
    print(f"  stable: {bool(np.all(poles.real < 0.0))}")


def create_yaw_damper_analysis_figure():
    """Run the signed yaw-damper sweep and return its analysis figure."""
    trim = trim_straight_level(
        true_airspeed=502.0 * FT_TO_METER,
        altitude_m=0.0,
        cg_fraction=0.30,
    )
    if not trim["success"]:
        raise RuntimeError(f"Unable to obtain the reference trim: {trim['message']}")
    A_lat, B_lat = linearize_lateral(
        trim["state"],
        throttle=trim["throttle"],
        elevator_deg=trim["elevator_deg"],
        cg_fraction=0.30,
    )

    print("Signed-gain diagnostic")
    beneficial_sign = _diagnose_feedback_sign(A_lat, B_lat)
    gain_magnitudes = np.linspace(0.0, 100.0, 1001)
    gains = beneficial_sign * gain_magnitudes
    (
        poles,
        classification,
        natural_frequency,
        damping_ratio,
        damped_frequency,
        roll_poles,
        spiral_poles,
    ) = _track_sweep(A_lat, B_lat, gains)

    for magnitude in (0.0, 1.0, 5.0, 20.0, 50.0, 100.0):
        gain = beneficial_sign * magnitude
        index = int(np.argmin(np.abs(gains - gain)))
        _print_representative(
            gains[index],
            poles[index],
            natural_frequency[index],
            damping_ratio[index],
            roll_poles[index],
            spiral_poles[index],
        )

    stable = np.all(poles.real < 0.0, axis=1)
    valid_damping = stable & np.isfinite(damping_ratio)
    open_loop_zeta = damping_ratio[0]
    print(f"\nOpen-loop Dutch-roll damping ratio: {open_loop_zeta:.9f}")
    if np.any(valid_damping):
        stable_indices = np.flatnonzero(valid_damping)
        best_index = stable_indices[np.argmax(damping_ratio[stable_indices])]
        print(
            "Maximum stable-system Dutch-roll damping ratio in sweep: "
            f"{damping_ratio[best_index]:.9f}"
        )
        print(f"Corresponding Kr: {gains[best_index]:.9f} deg/(rad/s)")
    else:
        print("No stable oscillatory Dutch-roll point was found in the sweep")

    roll_unstable = roll_poles.real >= 0.0
    spiral_unstable = spiral_poles.real >= 0.0
    if np.any(roll_unstable) or np.any(spiral_unstable):
        affected = []
        if np.any(roll_unstable):
            affected.append("roll")
        if np.any(spiral_unstable):
            affected.append("spiral")
        print(
            "Increasing |Kr| causes an undesirable interaction with: "
            + " and ".join(affected)
        )
    else:
        print(
            "No roll or spiral instability occurs within the analyzed "
            f"|Kr| <= {gain_magnitudes[-1]:g} range"
        )
    roll_shift = np.max(np.abs(roll_poles.real - roll_poles[0].real)) / abs(
        roll_poles[0].real
    )
    spiral_shift = np.max(
        np.abs(spiral_poles.real - spiral_poles[0].real)
    ) / abs(spiral_poles[0].real)
    if roll_shift > 0.2:
        print(
            "Large gains show a significant Dutch-roll/roll interaction: "
            f"the identified roll-pole real part shifts by {100.0 * roll_shift:.1f}%"
        )
    else:
        print("No strong Dutch-roll/roll interaction is evident in this sweep")
    print(
        "Spiral-pole real-part change across the sweep: "
        f"{100.0 * spiral_shift:.1f}% "
        f"(final pole {spiral_poles[-1]})"
    )
    nonoscillatory = ~np.isfinite(damped_frequency)
    if np.any(nonoscillatory):
        first_index = np.flatnonzero(nonoscillatory)[0]
        print(
            "The tracked Dutch-roll branches cease to be oscillatory near "
            f"Kr = {gains[first_index]:.6f} deg/(rad/s)"
        )

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 9.0))
    full_axis, zoom_axis, damping_axis, frequency_axis = axes.flat
    for branch in range(poles.shape[1]):
        label = "Closed-loop pole trajectories" if branch == 0 else None
        full_axis.plot(poles[:, branch].real, poles[:, branch].imag, label=label)
    full_axis.scatter(
        poles[0].real,
        poles[0].imag,
        marker="x",
        color="black",
        s=70,
        label="Open-loop poles",
        zorder=3,
    )
    full_axis.set_title("Full closed-loop pole trajectories")
    full_axis.set_xlabel("Real axis [1/s]")
    full_axis.set_ylabel("Imaginary axis [rad/s]")
    full_axis.axhline(0.0, color="black", linewidth=0.8)
    full_axis.axvline(0.0, color="black", linewidth=0.8)
    full_axis.grid(True)
    full_axis.legend()

    for branch_name in ("dutch_positive", "dutch_negative"):
        branch = classification[branch_name]
        zoom_axis.plot(poles[:, branch].real, poles[:, branch].imag)
    zoom_axis.scatter(
        poles[0, [classification["dutch_positive"], classification["dutch_negative"]]].real,
        poles[0, [classification["dutch_positive"], classification["dutch_negative"]]].imag,
        marker="x",
        color="black",
        s=70,
    )
    zoom_axis.set_title("Dutch-roll branch detail")
    zoom_axis.set_xlabel("Real axis [1/s]")
    zoom_axis.set_ylabel("Imaginary axis [rad/s]")
    zoom_axis.axhline(0.0, color="black", linewidth=0.8)
    zoom_axis.axvline(0.0, color="black", linewidth=0.8)
    zoom_axis.grid(True)

    damping_axis.plot(gains, damping_ratio)
    damping_axis.set_title("Dutch-roll damping ratio")
    damping_axis.set_xlabel(r"$K_r$ [deg/(rad/s)]")
    damping_axis.set_ylabel(r"$\zeta_{DR}$")
    damping_axis.grid(True)

    frequency_axis.plot(gains, natural_frequency)
    frequency_axis.set_title("Dutch-roll natural frequency")
    frequency_axis.set_xlabel(r"$K_r$ [deg/(rad/s)]")
    frequency_axis.set_ylabel(r"$\omega_n$ [rad/s]")
    frequency_axis.grid(True)

    figure.suptitle(
        "F-16 Rudder Yaw-Damper Analysis\n"
        r"Feedback law: $\delta_r = K_r \Delta r$"
    )
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    create_yaw_damper_analysis_figure()
    plt.show()
