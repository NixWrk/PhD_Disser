"""Render a local 3D QC view of baseline and conditional lung-heart geometry."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alias", required=True)
    args = parser.parse_args()
    lung_reference = pv.read(str(args.scenario_dir / "lung_stage_0.70.vtp"))
    heart_reference = pv.read(str(args.scenario_dir / "heart_stage_0.70.vtp"))
    lung_moved = pv.read(str(args.scenario_dir / "lung_stage_0.00.vtp"))
    heart_moved = pv.read(str(args.scenario_dir / "heart_stage_0.00.vtp"))
    if lung_reference.n_points != lung_moved.n_points or heart_reference.n_points != heart_moved.n_points:
        raise ValueError("Stage surfaces have incompatible point correspondence")
    lung_moved["displacement_mm"] = np.linalg.norm(
        np.asarray(lung_moved.points) - np.asarray(lung_reference.points), axis=1
    )
    heart_moved["displacement_mm"] = np.linalg.norm(
        np.asarray(heart_moved.points) - np.asarray(heart_reference.points), axis=1
    )
    plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1600, 850))
    plotter.set_background("#f6f8fb")
    plotter.subplot(0, 0)
    plotter.add_text(f"{args.alias}: input CT / model anchor 0.7", font_size=12, color="#263042")
    plotter.add_mesh(lung_reference, color="#5caac3", opacity=0.42, smooth_shading=True)
    plotter.add_mesh(heart_reference, color="#d75864", opacity=0.95, smooth_shading=True)
    plotter.view_isometric()
    camera = plotter.camera_position
    plotter.subplot(0, 1)
    plotter.add_text("Model scenario 0.0, relative displacement (mm)", font_size=12, color="#263042")
    plotter.add_mesh(lung_reference, color="#9da5b0", opacity=0.12)
    plotter.add_mesh(
        lung_moved, scalars="displacement_mm", cmap="viridis", clim=(0, 6.5),
        opacity=0.82, smooth_shading=True, scalar_bar_args={"title": "Lung (mm)"},
    )
    plotter.add_mesh(heart_moved, color="#c43b57", opacity=0.95, smooth_shading=True)
    plotter.camera_position = camera
    plotter.add_text("Heart shift is inherited from the lung field", position="lower_right", font_size=9, color="#263042")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(args.output))
    plotter.close()
    print(args.output)


if __name__ == "__main__":
    main()
