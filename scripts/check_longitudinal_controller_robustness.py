"""Check longitudinal attitude-controller robustness across flight speeds."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.f16sim.linearization import (
    analyze_longitudinal_modes,
    linearize_longitudinal,
)
from src.f16sim.parameters import FT_TO_METER
from src.f16sim.trim import trim_straight_level


FLIGHT_SPEEDS_FT_S = (440.0, 502.0, 700.0)
ALTITUDE_M = 0.0
CG_FRACTION = 0.30
KQ = 5.0
KTHETA = 0.5


def _closed_loop_matrix(A, B):
    elevator_column = B[:, 1:2]
    theta_output = np.array([[0.0, 0.0, 1.0, 0.0]])
    pitch_rate_output = np.array([[0.0, 0.0, 0.0, 1.0]])
    feedback_output = pitch_rate_output + KTHETA * theta_output
    return A + KQ * (elevator_column @ feedback_output)


def _short_period_damping(A):
    try:
        return analyze_longitudinal_modes(A)["short_period"]["damping_ratio"]
    except ValueError:
        return None


def _report_poles(label, poles):
    maximum_real_part = np.max(poles.real)
    stable = bool(np.all(poles.real < 0.0))
    print(f"  {label} eigenvalues: {np.sort_complex(poles)}")
    print(f"  {label} maximum pole real part: {maximum_real_part:.9f} 1/s")
    print(f"  {label} stable: {stable}")


def create_robustness_figure():
    """Analyze all requested speeds and return their combined pole plot."""
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

        alpha = np.deg2rad(trim["alpha_deg"])
        equilibrium_state = np.array(
            [trim["true_airspeed"], alpha, alpha, 0.0]
        )
        equilibrium_controls = np.array(
            [trim["throttle"], trim["elevator_deg"]]
        )
        A, B = linearize_longitudinal(
            equilibrium_state,
            equilibrium_controls,
            altitude_m=ALTITUDE_M,
            cg_fraction=CG_FRACTION,
        )
        closed_loop_A = _closed_loop_matrix(A, B)
        open_loop_poles = np.linalg.eigvals(A)
        closed_loop_poles = np.linalg.eigvals(closed_loop_A)
        open_loop_damping = _short_period_damping(A)
        closed_loop_damping = _short_period_damping(closed_loop_A)

        print(f"\nFlight condition: {speed_ft_s:g} ft/s")
        print(f"  trim throttle: {trim['throttle']:.9f}")
        print(f"  trim alpha: {trim['alpha_deg']:.9f} deg")
        print(f"  trim elevator: {trim['elevator_deg']:.9f} deg")
        _report_poles("open-loop", open_loop_poles)
        _report_poles("closed-loop", closed_loop_poles)
        if open_loop_damping is None:
            print("  open-loop short-period damping ratio: unavailable")
        else:
            print(
                "  open-loop short-period damping ratio: "
                f"{open_loop_damping:.6f}"
            )
        if closed_loop_damping is None:
            print("  closed-loop short-period damping ratio: unavailable")
        else:
            print(
                "  closed-loop short-period damping ratio: "
                f"{closed_loop_damping:.6f}"
            )

        results.append((speed_ft_s, open_loop_poles, closed_loop_poles))

    figure, axis = plt.subplots(figsize=(9.0, 7.0))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, (speed_ft_s, open_poles, closed_poles) in enumerate(results):
        color = colors[index % len(colors)]
        axis.scatter(
            open_poles.real,
            open_poles.imag,
            marker="x",
            s=80,
            color=color,
            label=f"{speed_ft_s:g} ft/s open-loop",
        )
        axis.scatter(
            closed_poles.real,
            closed_poles.imag,
            marker="o",
            s=65,
            facecolors="none",
            edgecolors=color,
            label=f"{speed_ft_s:g} ft/s closed-loop",
        )

    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Real axis [1/s]")
    axis.set_ylabel("Imaginary axis [rad/s]")
    axis.set_title(
        "F-16 Longitudinal Attitude-Controller Robustness\n"
        r"$K_q = 5$ deg/(rad/s), $K_\theta = 0.5$ 1/s, sea level, cg = 0.30"
    )
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    create_robustness_figure()
    plt.show()
