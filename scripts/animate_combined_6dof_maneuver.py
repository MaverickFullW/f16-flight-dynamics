"""Animate the validated combined nonlinear F-16 maneuver."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.simulate_combined_6dof_maneuver import simulate_combined_maneuver
from src.f16sim.visualization import create_flight_animation


def create_combined_maneuver_animation(
    camera="chase",
    fps=30,
    playback_speed=3.0,
    aircraft_scale=75.0,
):
    """Run the shared maneuver and return its reusable flight animation."""
    result = simulate_combined_maneuver()
    return create_flight_animation(
        result["times"],
        result["states"],
        controls=result["controls"],
        command_history=result["command_history"],
        fps=fps,
        playback_speed=playback_speed,
        camera=camera,
        aircraft_scale=aircraft_scale,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera",
        choices=("fixed", "chase"),
        default="chase",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--speed", type=float, default=3.0)
    parser.add_argument("--aircraft-scale", type=float, default=75.0)
    arguments = parser.parse_args()
    animation = create_combined_maneuver_animation(
        camera=arguments.camera,
        fps=arguments.fps,
        playback_speed=arguments.speed,
        aircraft_scale=arguments.aircraft_scale,
    )
    plt.show()
