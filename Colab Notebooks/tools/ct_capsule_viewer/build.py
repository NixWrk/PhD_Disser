"""Build a local, offline CT surface viewer for manual capsule landmarks.

Inputs are passed explicitly; no medical meshes or machine paths are tracked.
Output pages retain the source STL vertices and triangle faces (no decimation).
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import trimesh


SUBJECTS = ("nix", "georg", "yaroslav")
TITLES = {"nix": "Ник", "georg": "Георг", "yaroslav": "Ярослав"}
ROOT = Path(__file__).resolve().parent
RENDERING_REVISION = "normal_shading_v2"


def source_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mesh(path: Path) -> trimesh.Trimesh:
    if not path.is_file():
        raise FileNotFoundError(path)
    mesh = trimesh.load(path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"STL не содержит поверхности: {path}")
    mesh.remove_unreferenced_vertices()
    if not np.isfinite(mesh.vertices).all():
        raise ValueError(f"STL содержит некорректные координаты: {path}")
    return mesh


def surface_intensity(mesh: trimesh.Trimesh) -> np.ndarray:
    """Bake directional shading from STL normals so relief survives offline rendering."""
    normals = np.asarray(mesh.vertex_normals, dtype=float)
    key = np.array([-0.12, -0.88, 0.46], dtype=float)
    fill = np.array([0.86, -0.28, 0.42], dtype=float)
    back = np.array([0.12, 0.88, -0.46], dtype=float)
    key /= np.linalg.norm(key)
    fill /= np.linalg.norm(fill)
    back /= np.linalg.norm(back)
    return np.clip(
        0.12
        + 0.58 * np.clip(normals @ key, 0, 1)
        + 0.23 * np.clip(normals @ fill, 0, 1)
        + 0.18 * np.clip(normals @ back, 0, 1),
        0, 1,
    )


def plot_html(mesh: trimesh.Trimesh) -> str:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    figure = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                name="Поверхность тела", meta="surface",
                intensity=surface_intensity(mesh), intensitymode="vertex",
                cmin=0, cmax=1, showscale=False,
                colorscale=[
                    [0.00, "#463a36"], [0.20, "#6d5650"],
                    [0.42, "#aa8978"], [0.68, "#d9b9a1"],
                    [1.00, "#fff0d8"],
                ],
                opacity=1, flatshading=False,
                lighting={"ambient": 1, "diffuse": 0, "specular": 0,
                          "roughness": 1, "fresnel": 0},
                hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra>STL</extra>",
            ),
            go.Scatter3d(
                x=[], y=[], z=[], text=[], mode="markers+text",
                textposition="top center", name="Метки", meta="landmarks",
                marker={"size": 7, "color": "#e15831", "line": {"color": "#fff", "width": 2}},
                hovertemplate="%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
            ),
        ]
    )
    figure.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0}, showlegend=False,
        paper_bgcolor="#f7f9fa", plot_bgcolor="#f7f9fa",
        scene={
            "xaxis": {"title": "x · STL", "backgroundcolor": "#f7f9fa"},
            "yaxis": {"title": "y · STL", "backgroundcolor": "#f7f9fa"},
            "zaxis": {"title": "z · STL", "backgroundcolor": "#f7f9fa"},
            "aspectmode": "data", "dragmode": "orbit",
            "camera": {"eye": {"x": 1.5, "y": -1.5, "z": 0.75},
                       "up": {"x": 0, "y": 0, "z": 1},
                       "projection": {"type": "orthographic"}},
        },
        uirevision="capsule-surface-v1",
    )
    return pio.to_html(
        figure, full_html=False, include_plotlyjs=True,
        config={"displaylogo": False, "responsive": True, "scrollZoom": True,
                "modeBarButtonsToRemove": ["toImage", "lasso3d", "select3d"]},
        div_id="surface-plot",
    )


def render_subject(subject: str, path: Path, out: Path) -> dict:
    mesh = load_mesh(path)
    digest = source_digest(path)
    metadata = {
        "schema": "ct_capsule_surface_landmarks_v1",
        "subject": subject,
        "source_stl_basename": path.name,
        "source_stl_sha256": digest,
        "rendering_revision": RENDERING_REVISION,
        "coordinate_system": "source_stl_native_xyz",
        "units": "mm_assumed_from_ct_stl_not_independently_verified",
        "surface_faces": int(len(mesh.faces)),
        "surface_vertices": int(len(mesh.vertices)),
        "bounds_native_xyz": np.asarray(mesh.bounds).tolist(),
    }
    template = (ROOT / "subject.html").read_text(encoding="utf-8")
    safe_json = json.dumps(metadata, ensure_ascii=False).replace("<", "\\u003c")
    page = (template.replace("__TITLE__", html.escape(TITLES[subject]))
            .replace("__SUBJECT__", subject)
            .replace("__SOURCE__", html.escape(path.name))
            .replace("__META__", safe_json)
            .replace("__PLOT__", plot_html(mesh))
            .replace("__APP_JS__", (ROOT / "app.js").read_text(encoding="utf-8")))
    (out / f"{subject}.html").write_text(page, encoding="utf-8")
    return metadata


def build(paths: dict[str, Path], out: Path) -> None:
    # Validate every source before writing any output.
    for subject in SUBJECTS:
        if not paths[subject].is_file():
            raise FileNotFoundError(paths[subject])
    out.mkdir(parents=True, exist_ok=True)
    manifest = [render_subject(subject, paths[subject], out) for subject in SUBJECTS]
    index = (ROOT / "index.html").read_text(encoding="utf-8")
    (out / "index.html").write_text(index, encoding="utf-8")
    (out / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out / "index.html")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for subject in SUBJECTS:
        parser.add_argument(f"--{subject}", required=True, type=Path, metavar="BODY_STL")
    parser.add_argument("--out", required=True, type=Path, metavar="OUTPUT_DIR")
    args = parser.parse_args()
    build({subject: getattr(args, subject) for subject in SUBJECTS}, args.out)


if __name__ == "__main__":
    main()
