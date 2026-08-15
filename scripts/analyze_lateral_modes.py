"""Inspect lateral-directional modes at the validated Lewis flight condition."""

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


STATE_LABELS = ("beta", "phi", "p", "r")


def _normalized_magnitudes(eigenvector):
    magnitudes = np.abs(eigenvector)
    maximum = np.max(magnitudes)
    return magnitudes / maximum if maximum > 0.0 else magnitudes


def _classify_modes(eigenvalues, eigenvectors):
    classifications = {}
    complex_indices = [
        index for index, value in enumerate(eigenvalues) if abs(value.imag) > 1e-8
    ]
    positive_complex = [
        index for index in complex_indices if eigenvalues[index].imag > 0.0
    ]
    if positive_complex:
        dutch_index = max(
            positive_complex,
            key=lambda index: (
                _normalized_magnitudes(eigenvectors[:, index])[0]
                + _normalized_magnitudes(eigenvectors[:, index])[3]
            ),
        )
        dutch_value = eigenvalues[dutch_index]
        classifications["Dutch roll"] = [
            index
            for index in complex_indices
            if np.isclose(
                eigenvalues[index],
                dutch_value,
                rtol=1e-7,
                atol=1e-10,
            )
            or np.isclose(
                eigenvalues[index],
                dutch_value.conjugate(),
                rtol=1e-7,
                atol=1e-10,
            )
        ]

    real_indices = [
        index for index, value in enumerate(eigenvalues) if abs(value.imag) <= 1e-8
    ]
    if real_indices:
        roll_index = max(
            real_indices,
            key=lambda index: _normalized_magnitudes(eigenvectors[:, index])[2],
        )
        classifications["roll subsidence"] = [roll_index]
        remaining_real = [index for index in real_indices if index != roll_index]
        if remaining_real:
            spiral_index = max(
                remaining_real,
                key=lambda index: _normalized_magnitudes(
                    eigenvectors[:, index]
                )[1],
            )
            classifications["spiral"] = [spiral_index]

    return classifications


def create_lateral_mode_figure():
    """Analyze the reference lateral model and return its s-plane pole plot."""
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
    np.set_printoptions(precision=10, suppress=False)
    print("A_lat [beta, phi, p, r]:")
    print(A_lat)
    print("\nB_lat [aileron_deg, rudder_deg]:")
    print(B_lat)

    eigenvalues, eigenvectors = np.linalg.eig(A_lat)
    print("\nLateral eigenanalysis")
    for index, eigenvalue in enumerate(eigenvalues):
        natural_frequency = abs(eigenvalue)
        print(f"\nEigenvalue {index + 1}: {eigenvalue}")
        print(f"  real part: {eigenvalue.real:.10f} 1/s")
        print(f"  imaginary part: {eigenvalue.imag:.10f} rad/s")
        print(f"  natural frequency wn: {natural_frequency:.10f} rad/s")
        if abs(eigenvalue.imag) > 1e-8 and natural_frequency > 0.0:
            damping_ratio = -eigenvalue.real / natural_frequency
            print(f"  damping ratio zeta: {damping_ratio:.10f}")
        else:
            print("  damping ratio zeta: not applicable to a real mode")
        if abs(eigenvalue.imag) <= 1e-8 and eigenvalue.real < 0.0:
            print(f"  stable-mode time constant: {-1.0 / eigenvalue.real:.10f} s")
        elif abs(eigenvalue.imag) <= 1e-8:
            print("  stable-mode time constant: not applicable")
        if eigenvalue.real > 0.0:
            print("  stability: UNSTABLE")
        elif eigenvalue.real < 0.0:
            print("  stability: stable")
        else:
            print("  stability: neutral")

        participation = _normalized_magnitudes(eigenvectors[:, index])
        print("  normalized eigenvector magnitudes:")
        for label, magnitude in zip(STATE_LABELS, participation):
            print(f"    {label}: {magnitude:.10f}")

    classifications = _classify_modes(eigenvalues, eigenvectors)
    print("\nMode classification")
    for mode_name in ("roll subsidence", "Dutch roll", "spiral"):
        indices = classifications.get(mode_name)
        if not indices:
            print(f"  {mode_name}: not identified")
            continue
        values = eigenvalues[indices]
        unstable = np.any(values.real > 0.0)
        status = "UNSTABLE" if unstable else "stable"
        print(f"  {mode_name}: {values} ({status})")

    figure, axis = plt.subplots(figsize=(8.0, 6.0))
    stable = eigenvalues.real <= 0.0
    axis.scatter(
        eigenvalues[stable].real,
        eigenvalues[stable].imag,
        marker="x",
        s=90,
        color="C0",
        label="Stable/neutral poles",
    )
    if np.any(~stable):
        axis.scatter(
            eigenvalues[~stable].real,
            eigenvalues[~stable].imag,
            marker="x",
            s=100,
            color="C3",
            label="Unstable poles",
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Real axis [1/s]")
    axis.set_ylabel("Imaginary axis [rad/s]")
    axis.set_title("F-16 Lateral-Directional Poles at 502 ft/s, Sea Level")
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    create_lateral_mode_figure()
    plt.show()
