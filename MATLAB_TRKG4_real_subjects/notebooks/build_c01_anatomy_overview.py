"""Build the interactive C01 anatomy overview without running FEM.

The anatomical surfaces are visually decimated copies of the accepted STL
files.  Electrode patches are the exact boundary triangles already certified
by ``best_pose_review`` for all nine C01 assemblies.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import trimesh


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output" / "exploratory" / "transverse_consistency_20260914"
BEST = ROOT / "output" / "exploratory" / "scan_localization_20260910" / "best_pose_review"
ASSET_DIR = ROOT / "output" / "exploratory" / "notebook_2010_revision_20260914" / "assets"
FIGURE = ASSET_DIR / "c01_anatomy_overview.json"
MANIFEST = ASSET_DIR / "c01_anatomy_overview_manifest.json"

STL = {
    "Тело": ROOT / "data" / "nik" / "stl" / "body_solid_v3_1mm_r10_volume_fill.stl",
    "Лёгкие": ROOT / "data" / "nik" / "stl" / "lungs_solid_v3_volume_fill.stl",
    "Сердце": ROOT / "data" / "nik" / "stl" / "heart.stl",
    "Кости": ROOT / "data" / "nik" / "stl" / "bones.stl",
}
TARGET_FACES = {"Тело": 9000, "Лёгкие": 12000, "Сердце": 5500, "Кости": 16000}
COLORS = {
    "Тело": "#9aa5b1",
    "Лёгкие": "#00b8d9",
    "Сердце": "#b83250",
    "Кости": "#ead9ad",
    "I+": "#ef3340",
    "V+": "#f2a900",
    "V-": "#1687c9",
    "I-": "#7a49a5",
}
ROLES = ["I+", "V+", "V-", "I-"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload() -> tuple[dict, dict]:
    html_path = BEST / "index.html"
    manifest_path = BEST / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["outputs"]["index.html"] == sha(html_path)
    html = html_path.read_text(encoding="utf-8")
    match = re.search(
        r'<script id="data" type="application/json">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("В сохранённом обзоре C01 не найден блок точных площадок")
    return json.loads(match.group(1)), manifest


def _mesh(path: Path, target_faces: int) -> tuple[trimesh.Trimesh, int]:
    mesh = trimesh.load_mesh(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    mesh.remove_unreferenced_vertices()
    original_faces = len(mesh.faces)
    # Binary STL repeats vertices for adjacent triangles. Welding coincident
    # vertices is required before quadric decimation can reduce the surface.
    mesh.merge_vertices(digits_vertex=7)
    if original_faces > target_faces:
        mesh = mesh.simplify_quadric_decimation(face_count=target_faces, aggression=5)
        mesh.remove_unreferenced_vertices()
    return mesh, original_faces


def _mesh_trace(mesh: trimesh.Trimesh, name: str, opacity: float) -> go.Mesh3d:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    return go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        name=name, legendgroup="Анатомия", color=COLORS[name], opacity=opacity,
        flatshading=False, hoverinfo="name", showlegend=True,
        lighting=dict(ambient=0.68, diffuse=0.72, roughness=0.82, specular=0.08),
        lightposition=dict(x=-500, y=-700, z=600),
    )


def _patch_trace(patch: dict, size: int, role: str, visible: bool) -> go.Mesh3d:
    nodes = np.asarray(patch["nodes"], dtype=float)
    faces = np.asarray(patch["faces"], dtype=int)
    return go.Mesh3d(
        x=nodes[:, 0], y=nodes[:, 1], z=nodes[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        name=f"{role}, L={size} мм", color=COLORS[role], opacity=1.0,
        visible=visible, showlegend=False, flatshading=True,
        customdata=np.full((len(nodes), 1), float(patch["area_mm2"])),
        hovertemplate=(
            f"{role}; L={size} мм<br>x=%{{x:.1f}} мм<br>y=%{{y:.1f}} мм"
            "<br>z=%{z:.1f} мм<br>площадь=%{customdata[0]:.2f} мм²<extra></extra>"
        ),
    )


def _camera_buttons() -> list[dict]:
    cameras = [
        ("Общий вид", dict(eye=dict(x=-1.55, y=-1.45, z=0.72), up=dict(x=0, y=0, z=1))),
        ("Правый бок", dict(eye=dict(x=-2.25, y=0.08, z=0.20), up=dict(x=0, y=0, z=1))),
        ("Спереди", dict(eye=dict(x=0.02, y=-2.30, z=0.12), up=dict(x=0, y=0, z=1))),
    ]
    return [dict(label=label, method="relayout", args=[{"scene.camera": camera}]) for label, camera in cameras]


def build() -> dict:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    payload, old_manifest = _payload()
    sizes = [int(value) for value in payload["sizes"]]
    assert sizes == [50, 60, 70, 80, 90, 110, 120, 130, 140]
    assert len(payload["patches"]) == len(sizes)

    source = pd.read_csv(SOURCE / "source_electrodes.csv")
    source = source[source["candidate_id"] == "C01"].copy()
    traces: list = []
    mesh_record = {}
    for name, path in STL.items():
        mesh, original_faces = _mesh(path, TARGET_FACES[name])
        opacity = {"Тело": 0.11, "Лёгкие": 0.33, "Сердце": 0.58, "Кости": 0.20}[name]
        traces.append(_mesh_trace(mesh, name, opacity))
        mesh_record[name] = {
            "source_faces": original_faces,
            "display_faces": int(len(mesh.faces)),
            "source_sha256": sha(path),
        }

    default_size = 140
    assembly_indices: dict[int, list[int]] = {}
    max_area_error = 0.0
    max_centroid_error = 0.0
    for size, montage in zip(sizes, payload["patches"]):
        assert [p["role"] for p in montage] == ROLES
        assembly_indices[size] = []
        for role, patch in zip(ROLES, montage):
            row = source[(source["L_mm"] == size) & (source["electrode"] == role)].iloc[0]
            nodes = np.asarray(patch["nodes"], dtype=float)
            faces = np.asarray(patch["faces"], dtype=int)
            xyz = nodes[faces]
            areas = 0.5 * np.linalg.norm(
                np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1
            )
            centroid = np.sum(areas[:, None] * xyz.mean(axis=1), axis=0) / areas.sum()
            expected_centroid = row[[
                "patch_centroid_x_mm", "patch_centroid_y_mm", "patch_centroid_z_mm"
            ]].to_numpy(float)
            max_area_error = max(max_area_error, abs(float(areas.sum()) - float(row["contact_area_mm2"])))
            max_centroid_error = max(max_centroid_error, float(np.max(np.abs(centroid - expected_centroid))))
            assembly_indices[size].append(len(traces))
            traces.append(_patch_trace(patch, size, role, size == default_size))

        centres = np.asarray([p["centroid"] for p in montage], dtype=float)
        assembly_indices[size].append(len(traces))
        traces.append(go.Scatter3d(
            x=centres[:, 0], y=centres[:, 1], z=centres[:, 2], mode="lines+markers",
            line=dict(color="#2c3642", width=5), marker=dict(size=3, color=[COLORS[r] for r in ROLES]),
            name=f"Ось сборки, L={size} мм", visible=size == default_size, showlegend=False,
            hovertemplate=f"Сборка {size} мм<extra></extra>",
        ))

    assert max_area_error < 1e-8
    assert max_centroid_error < 1e-8
    centre = np.asarray(payload["centre"], dtype=float)
    traces.append(go.Scatter3d(
        x=[centre[0]], y=[centre[1]], z=[centre[2]], mode="markers",
        marker=dict(symbol="diamond", color="#111827", size=7), name="Центр C01",
        showlegend=True, hovertemplate="Центр C01<br>x=%{x:.2f} мм<br>y=%{y:.2f} мм<br>z=%{z:.2f} мм<extra></extra>",
    ))

    size_buttons = []
    for size in sizes:
        visible = [True] * 4 + [False] * (len(traces) - 5) + [True]
        for index in assembly_indices[size]:
            visible[index] = True
        size_buttons.append(dict(
            label=f"{size} мм", method="restyle", args=[{"visible": visible}],
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_white", height=800, margin=dict(l=5, r=5, t=105, b=5),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(255,255,255,0.70)", groupclick="toggleitem"),
        scene=dict(
            xaxis=dict(title="x, мм", showbackground=False),
            yaxis=dict(title="y, мм", showbackground=False),
            zaxis=dict(title="z, мм", showbackground=False),
            aspectmode="data",
            camera=dict(eye=dict(x=-1.55, y=-1.45, z=0.72), up=dict(x=0, y=0, z=1)),
        ),
        updatemenus=[
            dict(type="dropdown", direction="down", x=0.02, y=1.09, xanchor="left", yanchor="top",
                 active=sizes.index(default_size), buttons=size_buttons, showactive=True),
            dict(type="buttons", direction="right", x=0.28, y=1.09, xanchor="left", yanchor="top",
                 buttons=_camera_buttons(), showactive=False),
        ],
        annotations=[
            dict(text="Размер сборки", x=0.02, y=1.145, xref="paper", yref="paper", showarrow=False, xanchor="left"),
            dict(text="Ракурс", x=0.28, y=1.145, xref="paper", yref="paper", showarrow=False, xanchor="left"),
        ],
        meta={
            "candidate": "C01", "sizes_mm": sizes, "default_size_mm": default_size,
            "anatomy_trace_names": list(STL), "electrode_roles": ROLES,
            "electrode_geometry": "exact saved CEM boundary triangles",
            "anatomy_geometry": "visually decimated STL; not used in calculation",
            "max_patch_area_error_mm2": max_area_error,
            "max_patch_centroid_coordinate_error_mm": max_centroid_error,
        },
        uirevision="c01-anatomy-overview-v1",
    )
    FIGURE.write_text(fig.to_json(pretty=False), encoding="utf-8")
    record = {
        "schema": 1,
        "producer_sha256": sha(Path(__file__)),
        "status": "exploratory_hypothesis_not_validated",
        "new_FEM_solutions": 0,
        "purpose": "интерактивный анатомический обзор C01 в ноутбуке 20.10",
        "sources": {
            str((BEST / "index.html").relative_to(ROOT)): sha(BEST / "index.html"),
            str((BEST / "manifest.json").relative_to(ROOT)): sha(BEST / "manifest.json"),
            str((SOURCE / "source_electrodes.csv").relative_to(ROOT)): sha(SOURCE / "source_electrodes.csv"),
            **{str(path.relative_to(ROOT)): sha(path) for path in STL.values()},
        },
        "source_best_pose_manifest_sha256": sha(BEST / "manifest.json"),
        "source_best_pose_output_verified": old_manifest["outputs"]["index.html"] == sha(BEST / "index.html"),
        "sizes_mm": sizes,
        "patches": 4 * len(sizes),
        "max_patch_area_error_mm2": max_area_error,
        "max_patch_centroid_coordinate_error_mm": max_centroid_error,
        "display_meshes": mesh_record,
        "figure_sha256": sha(FIGURE),
    }
    MANIFEST.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
