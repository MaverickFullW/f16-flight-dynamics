"""Check the selected lateral controller across several flight speeds."""

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


FLIGHT_SPEEDS_FT_S = np.array([440.0, 502.0, 700.0])
ALTITUDE_M = 0.0
CG_FRACTION = 0.30
KR = 50.0
KP = 5.0
KPHI = 1.0
C_PHI = np.array([[0.0, 1.0, 0.0, 0.0]])
C_P = np.array([[0.0, 0.0, 1.0, 0.0]])
C_R = np.array([[0.0, 0.0, 0.0, 1.0]])


def _participation(eigenvector):
    magnitudes = np.abs(eigenvector)
    total = np.sum(magnitudes)
    return magnitudes / total if total > 0.0 else magnitudes


def _modal_summary(matrix):
    poles, eigenvectors = np.linalg.eig(matrix)
    positive_complex = [
        index for index, pole in enumerate(poles) if pole.imag > 1e-8
    ]
    if not positive_complex:
        raise RuntimeError("No oscillatory Dutch-roll pair could be identified")
    dutch_index = max(
        positive_complex,
        key=lambda index: (
            _participation(eigenvectors[:, index])[0]
            + _participation(eigenvectors[:, index])[3]
        ),
    )
    dutch = poles[dutch_index]

    real_indices = [
        index for index, pole in enumerate(poles) if abs(pole.imag) <= 1e-8
    ]
    if len(real_indices) < 2:
        raise RuntimeError("Roll and spiral real modes could not be identified")
    roll_index = max(
        real_indices,
        key=lambda index: _participation(eigenvectors[:, index])[2],
    )
    slow_index = max(
        (index for index in real_indices if index != roll_index),
        key=lambda index: _participation(eigenvectors[:, index])[1],
    )
    return {
        "poles": poles,
        "roll": poles[roll_index],
        "dutch": np.array([dutch, dutch.conjugate()]),
        "dutch_wn": float(abs(dutch)),
        "dutch_zeta": float(-dutch.real / abs(dutch)),
        "slow": poles[slow_index],
        "maximum_real": float(np.max(poles.real)),
        "stable": bool(np.all(poles.real < 0.0)),
    }


def _controlled_matrix(A_lat, B_lat):
    return (
        A_lat
        + KR * (B_lat[:, 1:2] @ C_R)
        + KP * (B_lat[:, 0:1] @ (C_P + KPHI * C_PHI))
    )


def _print_summary(label, summary):
    print(f"  {label}")
    print(f"    all poles: {np.sort_complex(summary['poles'])}")
    print(f"    roll-subsidence pole: {summary['roll']}")
    print(f"    Dutch-roll pair: {summary['dutch']}")
    print(f"    Dutch-roll wn: {summary['dutch_wn']:.9f} rad/s")
    print(f"    Dutch-roll zeta: {summary['dutch_zeta']:.9f}")
    print(f"    spiral/slow bank-angle pole: {summary['slow']}")
    print(f"    maximum pole real part: {summary['maximum_real']:.9f} 1/s")
    print(f"    stable: {summary['stable']}")


def create_robustness_figures():
    """Analyze all speeds and return pole-map and metric-comparison figures."""
    results = []
    for speed_ft_s in FLIGHT_SPEEDS_FT_S:
        trim = trim_straight_level(
            true_airspeed=speed_ft_s * FT_TO_METER,
            altitude_m=ALTITUDE_M,
            cg_fraction=CG_FRACTION,
        )
        if not trim["success"]:
            raise RuntimeError(
                f"Unable to trim at {speed_ft_s:g} ft/s: {trim['message']}"
            )
        A_lat, B_lat = linearize_lateral(
            trim["state"],
            trim["throttle"],
            trim["elevator_deg"],
            cg_fraction=CG_FRACTION,
        )
        open_summary = _modal_summary(A_lat)
        closed_summary = _modal_summary(_controlled_matrix(A_lat, B_lat))

        print(f"\nFlight condition: {speed_ft_s:g} ft/s")
        print(f"  trim throttle: {trim['throttle']:.9f}")
        print(f"  trim alpha: {trim['alpha_deg']:.9f} deg")
        print(f"  trim elevator: {trim['elevator_deg']:.9f} deg")
        _print_summary("Open-loop", open_summary)
        _print_summary("Closed-loop", closed_summary)
        results.append((speed_ft_s, open_summary, closed_summary))

    pole_figure, pole_axis = plt.subplots(figsize=(9.0, 7.0))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, (speed, open_summary, closed_summary) in enumerate(results):
        color = colors[index % len(colors)]
        open_poles = open_summary["poles"]
        closed_poles = closed_summary["poles"]
        pole_axis.scatter(
            open_poles.real,
            open_poles.imag,
            marker="x",
            s=85,
            color=color,
            label=f"{speed:g} ft/s open-loop",
        )
        pole_axis.scatter(
            closed_poles.real,
            closed_poles.imag,
            marker="o",
            s=70,
            facecolors="none",
            edgecolors=color,
            label=f"{speed:g} ft/s closed-loop",
        )
    pole_axis.axhline(0.0, color="black", linewidth=0.8)
    pole_axis.axvline(0.0, color="black", linewidth=0.8)
    pole_axis.set_xlabel("Real axis [1/s]")
    pole_axis.set_ylabel("Imaginary axis [rad/s]")
    pole_axis.set_title("F-16 Lateral Open- and Closed-Loop Poles")
    pole_axis.grid(True)
    pole_axis.legend()
    pole_figure.tight_layout()

    metric_figure, metric_axes = plt.subplots(
        3, 1, sharex=True, figsize=(9.0, 10.0)
    )
    open_zeta = [result[1]["dutch_zeta"] for result in results]
    closed_zeta = [result[2]["dutch_zeta"] for result in results]
    open_roll = [result[1]["roll"].real for result in results]
    closed_roll = [result[2]["roll"].real for result in results]
    open_slow = [result[1]["slow"].real for result in results]
    closed_slow = [result[2]["slow"].real for result in results]

    metric_axes[0].plot(
        FLIGHT_SPEEDS_FT_S, open_zeta, marker="o", label="Open-loop"
    )
    metric_axes[0].plot(
        FLIGHT_SPEEDS_FT_S, closed_zeta, marker="o", label="Closed-loop"
    )
    metric_axes[0].set_ylabel("Dutch-roll damping ratio")

    metric_axes[1].plot(
        FLIGHT_SPEEDS_FT_S, open_roll, marker="o", label="Open-loop"
    )
    metric_axes[1].plot(
        FLIGHT_SPEEDS_FT_S, closed_roll, marker="o", label="Closed-loop"
    )
    metric_axes[1].set_ylabel("Roll pole real part [1/s]")

    metric_axes[2].plot(
        FLIGHT_SPEEDS_FT_S, open_slow, marker="o", label="Open-loop"
    )
    metric_axes[2].plot(
        FLIGHT_SPEEDS_FT_S, closed_slow, marker="o", label="Closed-loop"
    )
    metric_axes[2].axhline(0.0, color="black", linewidth=0.8)
    metric_axes[2].set_ylabel("Slow pole real part [1/s]")
    metric_axes[2].set_xlabel("True airspeed [ft/s]")

    for axis in metric_axes:
        axis.grid(True)
        axis.legend()
    metric_figure.suptitle(
        "F-16 Lateral Controller Robustness\n"
        r"$K_r=50$, $K_p=5$, $K_\phi=1$, sea level, cg = 0.30"
    )
    metric_figure.tight_layout()
    return pole_figure, metric_figure


if __name__ == "__main__":
    create_robustness_figures()
    plt.show()
