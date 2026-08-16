"""Preview the independently constructed low-poly F-16-like geometry."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.f16sim.visualization import create_f16_low_poly_geometry


MATERIAL_COLORS = {
    "airframe": "0.015",
    "wing": "0.015",
    "lerx": "0.018",
    "stabilator": "0.015",
    "tail": "0.015",
    "canopy": "0.20",
    "nozzle": "0.22",
    "nozzle_interior": "0.035",
    "intake": "0.045",
    "intake_opening": "0.005",
}


def _add_mesh(axis, vertices, faces, groups):
    collection = Poly3DCollection(
        [vertices[np.asarray(face)] for face in faces],
        facecolors=[MATERIAL_COLORS[group] for group in groups],
        edgecolors="0.32",
        linewidths=0.55,
        alpha=0.98,
    )
    axis.add_collection3d(collection)
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    half_range = 0.505 * np.ptp(vertices, axis=0).max()
    axis.set_xlim(center[0] - half_range, center[0] + half_range)
    axis.set_ylim(center[1] - half_range, center[1] + half_range)
    axis.set_zlim(center[2] + half_range, center[2] - half_range)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_zticks([])
    axis.grid(True)


def create_geometry_preview():
    """Return a four-view figure of the low-poly aircraft mesh."""
    vertices, faces, groups = create_f16_low_poly_geometry()
    figure = plt.figure(figsize=(14.0, 11.0))
    views = (
        ("Top", 90.0, -90.0),
        ("Left side", 0.0, -90.0),
        ("Rear oblique", 20.0, 125.0),
        ("Perspective", 24.0, -55.0),
    )
    for index, (title, elevation, azimuth) in enumerate(views, start=1):
        axis = figure.add_subplot(2, 2, index, projection="3d")
        _add_mesh(axis, vertices, faces, groups)
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_title(title)
    figure.suptitle("F-16 Low-Poly Geometry Preview")
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    create_geometry_preview()
    plt.show()
