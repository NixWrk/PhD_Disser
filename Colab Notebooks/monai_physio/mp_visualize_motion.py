"""Interactive, local 3D review of already computed surface sequences.

The figure is for visual QC. Decimation affects display only; no metric is
computed on its reduced mesh. Keep figures with individual anatomy out of Git.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pyvista as pv
from scipy.spatial import cKDTree


def _read_surface(path: Path) -> pv.PolyData:
    if path.suffix.lower() == ".stl":
        import trimesh

        mesh = trimesh.load(str(path), process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"STL is not a single mesh: {path}")
        faces = np.column_stack((np.full(len(mesh.faces), 3), mesh.faces)).ravel()
        return pv.PolyData(np.asarray(mesh.vertices), faces)
    return pv.read(str(path))


def motion_figure(
    frames: list[tuple[str, dict[str, Path]]],
    *,
    title: str,
    subtitle: str,
    colors: dict[str, str],
    opacity: dict[str, float],
    reduction: float = 0.97,
    same_topology: bool = True,
    coordinate_system: str = "LPS",
    ras_organs: set[str] | None = None,
    initial_index: int = 0,
) -> go.Figure:
    """Build a rotatable Plotly sequence with a fixed camera and physical axes.

    With corresponding meshes, reduced vertices are mapped once to original
    nodes and the same indices are used in every frame. For independent meshes,
    each frame is reduced separately and no interpolation is implied.
    """
    if len(frames) < 2 or not 0 <= reduction < 1:
        raise ValueError("At least two frames and 0 <= reduction < 1 are required")
    if not 0 <= initial_index < len(frames):
        raise ValueError("initial_index is outside the available frames")
    organs = list(frames[0][1])
    ras_organs = ras_organs or set()
    if not ras_organs.issubset(organs):
        raise ValueError("ras_organs must refer to plotted organs")
    if not organs or any(list(paths) != organs for _, paths in frames):
        raise ValueError("Every frame must contain the same ordered organs")
    if len({label for label, _ in frames}) != len(frames):
        raise ValueError("Frame labels must be unique")

    reduced: dict[str, list[np.ndarray]] = {organ: [] for organ in organs}
    coordinates: dict[str, list[np.ndarray]] = {organ: [] for organ in organs}
    for organ in organs:
        reference = _read_surface(frames[0][1][organ]).triangulate()
        if reference.n_points == 0 or reference.n_cells == 0:
            raise ValueError(f"Empty surface: {organ}")
        display_mesh = reference.decimate(reduction) if reduction else reference
        ids = cKDTree(np.asarray(reference.points)).query(
            np.asarray(display_mesh.points), workers=-1
        )[1] if same_topology else None
        reference_faces = np.asarray(reference.faces)
        for label, paths in frames:
            mesh = _read_surface(paths[organ]).triangulate()
            if same_topology and (mesh.n_points != reference.n_points or
                                  not np.array_equal(mesh.faces, reference_faces)):
                raise ValueError(f"Topology differs for {organ} in frame {label}")
            view = display_mesh if same_topology else (mesh.decimate(reduction) if reduction else mesh)
            points = np.array(np.asarray(mesh.points)[ids] if same_topology else view.points, copy=True)
            if organ in ras_organs:
                points[:, :2] *= -1  # RAS mm -> LPS mm
            coordinates[organ].append(points)
            reduced[organ].append(np.asarray(view.faces).reshape(-1, 4)[:, 1:].astype(int))

    all_points = np.concatenate([np.concatenate(points) for points in coordinates.values()])
    axis_range = []
    for dim in range(3):
        low, high = np.min(all_points[:, dim]), np.max(all_points[:, dim])
        pad = max((high - low) * 0.08, 5.0)
        axis_range.append([float(low - pad), float(high + pad)])

    def traces(index: int) -> list[go.Mesh3d]:
        result = []
        for organ in organs:
            pts = coordinates[organ][index]
            faces = reduced[organ][index]
            result.append(go.Mesh3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color=colors[organ], opacity=opacity[organ],
                name=organ, showlegend=True, flatshading=True,
                hovertemplate=f"{organ}<br>X=%{{x:.1f}} мм<br>Y=%{{y:.1f}} мм<br>Z=%{{z:.1f}} мм<extra></extra>",
            ))
        return result

    labels = [label for label, _ in frames]
    fig = go.Figure(data=traces(initial_index), frames=[
        go.Frame(name=label, data=traces(index), traces=list(range(len(organs))))
        for index, label in enumerate(labels)
    ])
    steps = [dict(
        label=label,
        method="animate",
        args=[[label], dict(frame=dict(duration=0, redraw=True),
                            transition=dict(duration=0), mode="immediate")],
    ) for label in labels]
    fig.update_layout(
        title=dict(text=f"{title}<br><sup>{subtitle}</sup>"),
        width=950, height=760,
        margin=dict(l=0, r=0, t=85, b=0),
        scene=dict(
            xaxis=dict(title=f"{coordinate_system} X, мм", range=axis_range[0]),
            yaxis=dict(title=f"{coordinate_system} Y, мм", range=axis_range[1]),
            zaxis=dict(title=f"{coordinate_system} Z, мм", range=axis_range[2]),
            aspectmode="data",
        ),
        legend=dict(x=0.01, y=0.99),
        updatemenus=[dict(type="buttons", direction="left", x=0.02, y=0.02,
                          buttons=[
                              dict(label="▶ Воспроизвести", method="animate", args=[None,
                                   dict(frame=dict(duration=850, redraw=True), transition=dict(duration=0),
                                        fromcurrent=True, mode="immediate")]),
                              dict(label="■ Остановить", method="animate", args=[[None],
                                   dict(frame=dict(duration=0, redraw=True), transition=dict(duration=0),
                                        mode="immediate")]),
                          ])],
        sliders=[dict(active=initial_index, currentvalue=dict(prefix="Кадр: "),
                      pad=dict(t=55), steps=steps)],
    )
    return fig
