#!/usr/bin/env python
"""Build offline Plotly animations from 4D cardiac segmentation products.

Each subject gets one rotatable 3D figure with three independently toggleable
surfaces: whole heart, intracardiac blood pool, and myocardium.  Cardiac phases
are controlled by a slider and play/pause buttons.  The generated HTML and
notebook contain reduced surface meshes, not source CT voxels or identifiers.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import nibabel as nib
import nbformat
import numpy as np
import plotly.graph_objects as go
from nibabel.affines import apply_affine
from plotly.io import to_html
from skimage import measure


@dataclass(frozen=True)
class Structure:
    key: str
    label: str
    color: str
    opacity: float
    step_size: int


STRUCTURES = (
    Structure("whole_heart", "Всё сердце", "#B7BDC8", 0.20, 5),
    Structure("blood_pool", "Кровь в полостях", "#D7263D", 1.0, 5),
    Structure("myocardium", "Миокард", "#F4A261", 0.62, 5),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def surface(path: Path, step_size: int) -> dict[str, np.ndarray]:
    image = nib.load(str(path))
    mask = np.asanyarray(image.dataobj) > 0
    if not np.any(mask):
        raise ValueError(f"Empty mask: {path}")
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    vertices, faces, _, _ = measure.marching_cubes(
        padded, level=0.5, step_size=step_size, allow_degenerate=False
    )
    vertices = np.round(apply_affine(image.affine, vertices - 1.0), 2).astype(np.float32)
    faces = faces.astype(np.int32, copy=False)
    return {
        "x": vertices[:, 0],
        "y": vertices[:, 1],
        "z": vertices[:, 2],
        "i": faces[:, 0],
        "j": faces[:, 1],
        "k": faces[:, 2],
    }


def mesh_trace(mesh: dict[str, np.ndarray], structure: Structure, showlegend: bool) -> go.Mesh3d:
    return go.Mesh3d(
        **mesh,
        name=structure.label,
        legendgroup=structure.key,
        color=structure.color,
        opacity=structure.opacity,
        flatshading=False,
        lighting={"ambient": 0.55, "diffuse": 0.75, "specular": 0.20, "roughness": 0.65},
        lightposition={"x": 100, "y": 200, "z": 300},
        hovertemplate=f"{structure.label}<extra></extra>",
        showlegend=showlegend,
    )


def axis_range(values: list[float]) -> list[float]:
    low = float(min(values))
    high = float(max(values))
    margin = max(2.0, 0.03 * (high - low))
    return [low - margin, high + margin]


def build_figure(products_root: Path, subject: str) -> tuple[go.Figure, dict]:
    subject_root = products_root / subject / "products_v2"
    phase_dirs = sorted(path for path in subject_root.glob("phase_*") if path.is_dir())
    if not phase_dirs:
        raise FileNotFoundError(f"No phase_* directories: {subject_root}")

    all_meshes: list[list[dict[str, np.ndarray]]] = []
    all_x: list[float] = []
    all_y: list[float] = []
    all_z: list[float] = []
    phase_ids: list[str] = []
    counts: dict[str, list[int]] = {item.key: [] for item in STRUCTURES}

    for phase_dir in phase_dirs:
        phase_meshes = []
        for item in STRUCTURES:
            mesh = surface(phase_dir / f"{item.key}.nii.gz", item.step_size)
            phase_meshes.append(mesh)
            counts[item.key].append(len(mesh["x"]))
            all_x.extend(mesh["x"])
            all_y.extend(mesh["y"])
            all_z.extend(mesh["z"])
        all_meshes.append(phase_meshes)
        phase_ids.append(phase_dir.name)

    initial = [mesh_trace(mesh, item, True) for mesh, item in zip(all_meshes[0], STRUCTURES)]
    frames = []
    for phase_id, phase_meshes in zip(phase_ids, all_meshes):
        frames.append(
            go.Frame(
                name=phase_id,
                traces=list(range(len(STRUCTURES))),
                data=[mesh_trace(mesh, item, False) for mesh, item in zip(phase_meshes, STRUCTURES)],
            )
        )

    slider_steps = [
        {
            "label": phase_id.replace("phase_", ""),
            "method": "animate",
            "args": [[phase_id], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
        }
        for phase_id in phase_ids
    ]
    figure = go.Figure(data=initial, frames=frames)
    figure.update_layout(
        title={"text": f"{subject.upper()}: 4D КТ сердца — автоматические маски", "x": 0.5},
        template="plotly_white",
        height=760,
        margin={"l": 0, "r": 0, "t": 75, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "center", "x": 0.5, "itemclick": "toggle", "itemdoubleclick": "toggleothers"},
        scene={
            "xaxis": {"title": "R–L, мм", "range": axis_range(all_x)},
            "yaxis": {"title": "A–P, мм", "range": axis_range(all_y)},
            "zaxis": {"title": "I–S, мм", "range": axis_range(all_z)},
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 1.05}},
        },
        sliders=[{"active": 0, "currentvalue": {"prefix": "Фаза: "}, "pad": {"t": 55}, "steps": slider_steps}],
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": 0.0,
                "pad": {"r": 10, "t": 70},
                "buttons": [
                    {"label": "▶", "method": "animate", "args": [None, {"fromcurrent": True, "frame": {"duration": 550, "redraw": True}, "transition": {"duration": 0}}]},
                    {"label": "Ⅱ", "method": "animate", "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]},
                ],
            }
        ],
        annotations=[
            {
                "text": "ЛКМ — вращение; колесо — масштаб; легенда — включение/выключение. Маски требуют экспертной проверки.",
                "xref": "paper", "yref": "paper", "x": 0.5, "y": -0.13, "showarrow": False,
            }
        ],
        uirevision=f"cardiac4d-{subject}",
    )
    metadata = {
        "subject": subject,
        "phases": phase_ids,
        "surface_vertices": counts,
        "structures": [item.key for item in STRUCTURES],
        "status": "automatic_segmentation_pending_manual_review",
    }
    return figure, metadata


def html_page(title: str, fragments: Sequence[str]) -> str:
    cards = "\n".join(f'<section class="card">{fragment}</section>' for fragment in fragments)
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
body{{margin:0;background:#eef1f5;color:#1f2937;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1500px;margin:auto;padding:24px}}h1{{font-size:24px;margin:0 0 8px}}p{{margin:0 0 18px}}
.card{{background:white;border-radius:14px;box-shadow:0 3px 16px #0001;margin:0 0 24px;padding:8px}}
</style></head><body><main><h1>{title}</h1>
<p>Три структуры переключаются в легенде; сердечная фаза выбирается ползунком или кнопкой воспроизведения.</p>
{cards}</main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products-root", required=True, help="Root containing <subject>/products_v2")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subjects", nargs="+", default=["adam", "nix", "georg"])
    args = parser.parse_args()

    products_root = Path(args.products_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures: list[tuple[str, go.Figure]] = []
    manifest = {"schema_version": 1, "created_at": utc_now(), "generator": Path(__file__).name, "subjects": []}
    config = {"responsive": True, "displaylogo": False, "scrollZoom": True}

    for subject in args.subjects:
        figure, metadata = build_figure(products_root, subject.lower())
        output_path = output_dir / f"{subject.lower()}_cardiac_4d.html"
        figure.write_html(output_path, include_plotlyjs=True, full_html=True, auto_play=False, config=config)
        metadata["html"] = output_path.name
        manifest["subjects"].append(metadata)
        figures.append((subject.lower(), figure))
        print(subject, output_path.name, flush=True)

    fragments = [to_html(fig, full_html=False, include_plotlyjs=("inline" if index == 0 else False), auto_play=False, config=config) for index, (_, fig) in enumerate(figures)]
    combined = output_dir.parent / "21.01_4D_сердце_интерактивно.html"
    combined.write_text(html_page("Интерактивная динамика сердца по 4D КТ", fragments), encoding="utf-8")

    notebook = nbformat.v4.new_notebook(
        metadata={"language_info": {"name": "python"}, "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
        cells=[nbformat.v4.new_markdown_cell(
            "# Интерактивная динамика сердца по 4D КТ\n\n"
            "Каждый график вращается мышью. Фазы переключаются ползунком или кнопкой ▶. "
            "Структуры выключаются кликом по легенде. Это автоматические исследовательские маски, не прошедшие ручную экспертную верификацию."
        )],
    )
    for index, (subject, figure) in enumerate(figures):
        fragment = to_html(figure, full_html=False, include_plotlyjs=("inline" if index == 0 else False), auto_play=False, config=config)
        notebook.cells.append(nbformat.v4.new_markdown_cell(f"## {subject.upper()}"))
        notebook.cells.append(nbformat.v4.new_code_cell(
            source="# Интерактивный график встроен в output этой ячейки; повторная генерация выполняется скриптом build_interactive_cardiac4d.py.",
            execution_count=None,
            outputs=[nbformat.v4.new_output("display_data", data={"text/html": fragment}, metadata={})],
        ))
    notebook_path = output_dir.parent / "21.01_4D_сердце_интерактивно.ipynb"
    nbformat.write(notebook, notebook_path)

    manifest["combined_html"] = combined.name
    manifest["notebook"] = notebook_path.name
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
