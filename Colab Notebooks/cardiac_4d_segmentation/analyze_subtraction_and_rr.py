#!/usr/bin/env python
"""Show whole-heart minus blood and quantify chamber volumes over DICOM R-R phase."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import nbformat
import numpy as np
import plotly.graph_objects as go
from nibabel.affines import apply_affine
from plotly.io import to_html
from plotly.subplots import make_subplots
from skimage import measure


CHAMBERS = (
    ("heart_atrium_left", "Левое предсердие", "#3A86FF"),
    ("heart_ventricle_left", "Левый желудочек", "#0057B8"),
    ("heart_atrium_right", "Правое предсердие", "#FF8C42"),
    ("heart_ventricle_right", "Правый желудочек", "#D62828"),
)


@dataclass(frozen=True)
class SurfaceStyle:
    key: str
    label: str
    color: str
    opacity: float


SURFACES = (
    SurfaceStyle("heart_minus_blood", "Производная остаточная маска", "#B9A27A", 0.42),
    SurfaceStyle("blood_pool", "Кровь четырёх камер", "#D7263D", 0.72),
    SurfaceStyle("myocardium", "Миокард ЛЖ и перегородки", "#F4A261", 0.72),
)

NOTEBOOK_CONTEXT_MD = """# Проверка автоматической сегментации сердца в 4D-КТ

**Статус:** исследовательский вычислительный тест; автоматические маски ожидают
покадровой экспертной проверки.

## Задача

Ноутбук проверяет, можно ли использовать автоматическую сегментацию для
описания динамики формы сердца и предварительного расчёта объёмов четырёх
камер. В тест включены 39 фаз контрастной 4D-КТ трёх добровольцев: Adam, Nix и
Georg. Каноническим анатомическим референсом проекта остаётся ручная
сегментация Inobitec; в данном расчёте она ещё не подключена.

## Модели и производные маски

TotalSegmentator 2.18.0 запускался в двух независимых задачах. Задача `total`,
класс `heart`, формирует маску всего сердца. Задача
`heartchambers_highres` формирует четыре камеры и класс
`heart_myocardium`. Маска крови является объединением четырёх камер без
аорты и лёгочной артерии.

Остаточная маска «всё сердце минус кровь» — геометрическая разность выходов
двух моделей. Она включает все воксели внутри маски сердца, которые не
отнесены ко внутрисердечной крови, и не является маской полного миокарда.
Класс `heart_myocardium` в текущем прогоне охватывает преимущественно миокард
левого желудочка и межжелудочковую перегородку; стенки остальных камер не
следует считать полноценно выделенными.

Объёмы рассчитаны по числу вокселей бинарных NIfTI с учётом физического объёма
вокселя. Поверхностные STL используются только для интерактивной визуализации.

## Наблюдаемые ограничения

- межмодельный коэффициент Dice 0,906–0,947 характеризует согласие двух автоматических
  выходов, но не точность относительно анатомического референса;
- максимальный соседний скачок объёма модельного миокарда составляет
  21,5–26,2%, а отдельных камер достигает 44,1%;
- исходное поле `NominalPercentageOfCardiacPhase` содержит значения выше
  100%, а подтверждающие поля времени R–R не заполнены;
- ЭКГ над объёмными кривыми является схемой, а не измеренным сигналом;
- клапанные плоскости, устья сосудов, ушки предсердий и тонкие стенки камер
  требуют посрезовой экспертной проверки.

Поэтому представленные кривые являются кандидатным результатом автоматической
сегментации. Они не задают подтверждённые конечно-диастолические и
конечно-систолические объёмы, ударный объём или фракцию выброса. Полное
описание метода, численных результатов и плана валидации приведено в
[научном отчёте](21.04_heart_subtraction_and_RR_report.md).
"""

SCIENTIFIC_CONTEXT_HTML = """
<section class="card context">
<h2>Научная постановка и статус</h2>
<p><strong>Статус:</strong> исследовательский вычислительный тест автоматической
сегментации TotalSegmentator 2.18.0 на 39 фазах 4D-КТ Adam, Nix и Georg.
Ручная сегментация Inobitec в этот расчёт ещё не включена, поэтому поверхности
и объёмные кривые ожидают покадровой экспертной проверки.</p>
<p>Маска всего сердца получена независимой задачей <code>total/heart</code>.
Четыре камеры и класс <code>heart_myocardium</code> получены задачей
<code>heartchambers_highres</code>. Остаток «всё сердце минус кровь» является
геометрической разностью двух моделей и не представляет полный миокард.
В текущем прогоне <code>heart_myocardium</code> охватывает преимущественно
миокард ЛЖ и межжелудочковую перегородку.</p>
<h3>Основные ограничения</h3>
<ul>
<li>межмодельный коэффициент Dice 0,906–0,947 не является оценкой точности относительно
анатомического референса;</li>
<li>соседний скачок объёма модельного миокарда достигает 21,5–26,2%, а отдельных
камер — 44,1%;</li>
<li>значения DICOM-фазы выше 100% не подтверждены дополнительными полями
времени R–R;</li>
<li>схематическая ЭКГ не является измеренным сигналом испытуемого.</li>
</ul>
<p>До ручной проверки эти данные нельзя использовать как подтверждённые
конечно-диастолические или конечно-систолические объёмы, ударный объём и
фракцию выброса.</p>
</section>
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def same_grid(a: nib.Nifti1Image, b: nib.Nifti1Image) -> bool:
    return a.shape == b.shape and np.allclose(a.affine, b.affine, atol=1e-4)


def save_mask(mask: np.ndarray, reference: nib.Nifti1Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(mask.astype(np.uint8), reference.affine, reference.header)
    image.set_data_dtype(np.uint8)
    nib.save(image, path)


def volume_ml(mask: np.ndarray, affine: np.ndarray) -> float:
    return float(np.count_nonzero(mask) * abs(np.linalg.det(affine[:3, :3])) / 1000.0)


def surface(mask: np.ndarray, affine: np.ndarray, step_size: int = 6) -> dict[str, np.ndarray]:
    if not np.any(mask):
        return {name: np.asarray([], dtype=np.float32 if name in "xyz" else np.int32) for name in ("x", "y", "z", "i", "j", "k")}
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    vertices, faces, _, _ = measure.marching_cubes(padded, 0.5, step_size=step_size, allow_degenerate=False)
    vertices = np.round(apply_affine(affine, vertices - 1.0), 2).astype(np.float32)
    faces = faces.astype(np.int32, copy=False)
    return {"x": vertices[:, 0], "y": vertices[:, 1], "z": vertices[:, 2], "i": faces[:, 0], "j": faces[:, 1], "k": faces[:, 2]}


def trace(mesh: dict[str, np.ndarray], style: SurfaceStyle, showlegend: bool) -> go.Mesh3d:
    return go.Mesh3d(
        **mesh, name=style.label, legendgroup=style.key, color=style.color,
        opacity=style.opacity, flatshading=False, showlegend=showlegend,
        lighting={"ambient": 0.6, "diffuse": 0.7, "specular": 0.15, "roughness": 0.7},
        hovertemplate=style.label + "<extra></extra>",
    )


def axis_range(values: list[float]) -> list[float]:
    low, high = min(values), max(values)
    margin = max(2.0, 0.03 * (high - low))
    return [low - margin, high + margin]

def schematic_ecg(x: np.ndarray) -> np.ndarray:
    """Illustrative periodic P-QRS-T waveform; not a measured signal."""
    phase = np.mod(x, 100.0)
    def pulse(center: float, width: float, amplitude: float) -> np.ndarray:
        distance = (phase - center + 50.0) % 100.0 - 50.0
        return amplitude * np.exp(-0.5 * (distance / width) ** 2)
    return (pulse(80, 4.5, 0.16) + pulse(97.5, 1.0, -0.20) +
            pulse(0, 0.75, 1.15) + pulse(2.2, 1.2, -0.30) + pulse(25, 7.0, 0.34))



def subtraction_figure(subject: str, phases: list[dict]) -> go.Figure:
    all_meshes = []
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for phase in phases:
        meshes = [surface(phase[style.key], phase["affine"]) for style in SURFACES]
        all_meshes.append(meshes)
        for mesh in meshes[:2]:
            xs.extend(mesh["x"]); ys.extend(mesh["y"]); zs.extend(mesh["z"])
    figure = go.Figure(
        data=[trace(mesh, style, True) for mesh, style in zip(all_meshes[0], SURFACES)],
        frames=[go.Frame(
            name=phase["phase_id"], traces=list(range(len(SURFACES))),
            data=[trace(mesh, style, False) for mesh, style in zip(meshes, SURFACES)],
        ) for phase, meshes in zip(phases, all_meshes)],
    )
    steps = [{
        "label": f"{phase['rr_percent']:g}%",
        "method": "animate",
        "args": [[phase["phase_id"]], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
    } for phase in phases]
    figure.update_layout(
        title={"text": f"{subject.upper()}: геометрическое вычитание маски крови из маски сердца", "x": 0.5},
        template="plotly_white", height=760, margin={"l": 0, "r": 0, "t": 80, "b": 25},
        legend={"orientation": "h", "y": 1.01, "x": 0.5, "xanchor": "center", "yanchor": "bottom", "itemclick": "toggle", "itemdoubleclick": "toggleothers"},
        scene={
            "xaxis": {"title": "R–L, мм", "range": axis_range(xs)},
            "yaxis": {"title": "A–P, мм", "range": axis_range(ys)},
            "zaxis": {"title": "I–S, мм", "range": axis_range(zs)},
            "aspectmode": "data", "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 1.05}},
        },
        sliders=[{"active": 0, "currentvalue": {"prefix": "Исходное значение DICOM: "}, "pad": {"t": 55}, "steps": steps}],
        updatemenus=[{"type": "buttons", "direction": "left", "x": 0, "y": 0, "pad": {"t": 70}, "buttons": [
            {"label": "▶", "method": "animate", "args": [None, {"fromcurrent": True, "frame": {"duration": 550, "redraw": True}, "transition": {"duration": 0}}]},
            {"label": "Ⅱ", "method": "animate", "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]},
        ]}],
        uirevision=f"subtraction-{subject}",
    )
    return figure


def volume_figure(subject: str, phases: list[dict]) -> go.Figure:
    x_end = max(105.0, max(phase["rr_percent"] for phase in phases) + 5.0)
    ecg_x = np.linspace(0.0, x_end, int(x_end * 12) + 1)
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.27, 0.73], vertical_spacing=0.05)
    figure.add_trace(go.Scatter(
        x=ecg_x, y=schematic_ecg(ecg_x), mode="lines", name="Схематическая ЭКГ (не измерение)",
        line={"color": "#4B5563", "width": 2},
        hovertemplate="Условная ЭКГ<br>Исходная шкала: %{x:.1f}%<extra></extra>",
    ), row=1, col=1)
    for key, label, color in CHAMBERS:
        figure.add_trace(go.Scatter(
            x=[phase["rr_percent"] for phase in phases],
            y=[phase["chambers_ml"][key] for phase in phases],
            customdata=[[phase["phase_id"], phase["rr_percent"] % 100, int(phase["rr_percent"] // 100)] for phase in phases],
            mode="lines+markers", name=label, line={"color": color, "width": 3}, marker={"size": 8},
            hovertemplate="%{fullData.name}<br>Исходное значение DICOM: %{x:.0f}%<br>Остаток по модулю 100 (не валидирован): %{customdata[1]:.0f}%<br>Объём: %{y:.1f} мл<br>%{customdata[0]}<extra></extra>",
        ), row=2, col=1)
    for boundary in np.arange(0.0, x_end + 0.1, 100.0):
        figure.add_vline(x=float(boundary), line_dash="dash", line_color="#9CA3AF", line_width=1)
        figure.add_annotation(x=float(boundary), y=1.08, text="R", showarrow=False, row=1, col=1, font={"color": "#B91C1C"})
    figure.add_annotation(x=80, y=0.23, text="P", showarrow=False, row=1, col=1)
    figure.add_annotation(x=25, y=0.41, text="T", showarrow=False, row=1, col=1)
    figure.update_xaxes(title_text="NominalPercentageOfCardiacPhase, % R–R", dtick=10, range=[0, x_end], row=2, col=1)
    figure.update_yaxes(title_text="Условная ЭКГ", showticklabels=False, zeroline=True, row=1, col=1)
    figure.update_yaxes(title_text="Объём автоматической маски, мл", rangemode="tozero", row=2, col=1)
    figure.update_layout(
        title={"text": f"{subject.upper()}: объёмы камер по исходному значению DICOM", "x": 0.5},
        template="plotly_white", height=690, hovermode="x unified",
        legend={"orientation": "h", "y": 1.04, "x": 0.5, "xanchor": "center", "yanchor": "bottom", "itemclick": "toggle", "itemdoubleclick": "toggleothers"},
        margin={"l": 80, "r": 30, "t": 105, "b": 65},
    )
    return figure


def html_page(subtractions: list[str], volumes: list[str]) -> str:
    cards = [SCIENTIFIC_CONTEXT_HTML]
    for fragment in subtractions:
        cards.append(f'<section class="card">{fragment}</section>')
    cards.append("<h2>Объёмы четырёх камер</h2><p>Над кривыми показана условная стандартная форма ЭКГ для ориентации внутри R–R. Это схема, а не записанный у испытуемого сигнал. Значение фазы взято напрямую из DICOM NominalPercentageOfCardiacPhase. В части кадров оно выше 100%, а дополнительные поля R–R/trigger в исходных DICOM не заполнены. Поэтому трактовка этой шкалы как точного процента одного цикла или следующего цикла пока не подтверждена; на оси сохранены исходные значения.</p>")
    for fragment in volumes:
        cards.append(f'<section class="card">{fragment}</section>')
    return """<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Вычитание крови и объёмы камер</title><style>body{margin:0;background:#eef1f5;color:#1f2937;font-family:system-ui,-apple-system,'Segoe UI',sans-serif}main{max-width:1500px;margin:auto;padding:24px}.card{background:#fff;border-radius:14px;box-shadow:0 3px 16px #0001;margin:0 0 24px;padding:8px}h1,h2{margin-top:0}.warn{background:#fff4d6;border-left:5px solid #e0a100;padding:12px 16px;margin-bottom:20px}</style></head><body><main><h1>Вычитание крови из сердца и объёмы камер по R–R</h1><div class="warn">Разность «всё сердце − кровь» является геометрическим остатком двух независимых моделей, а не проверенной маской полного миокарда.</div>""" + "\n".join(cards) + "</main></body></html>"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path.cwd().resolve()
    data_root = project_root.parent / "Bitrix" / "ЭИТЛ" / "02 Big_data" / "3D" / "_DERIVED_CARDIAC4D"
    parser.add_argument("--derived-root", default=str(data_root))
    parser.add_argument("--output-dir", default=str(project_root / "Colab Notebooks" / "heart_rr_analysis"))
    parser.add_argument("--subjects", nargs="+", default=["adam", "nix", "georg"])
    args = parser.parse_args()
    derived_root = Path(args.derived_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {"responsive": True, "displaylogo": False, "scrollZoom": True}
    all_rows = []
    summaries = []
    subtraction_figures = []
    volume_figures = []

    for subject in args.subjects:
        subject = subject.lower()
        conversion = json.loads((derived_root / subject / "phases" / "conversion_manifest.json").read_text(encoding="utf-8"))
        phases = []
        for phase_info in conversion["phases"]:
            phase_id = phase_info["phase_id"]
            rr_percent = float(phase_info["dicom_phase_value"])
            product_dir = derived_root / subject / "products_v2" / phase_id
            raw_dir = derived_root / subject / "automatic_masks_v2" / phase_id / "heartchambers_highres"
            whole_image = nib.load(str(product_dir / "whole_heart.nii.gz"))
            blood_image = nib.load(str(product_dir / "blood_pool.nii.gz"))
            myocardium_image = nib.load(str(product_dir / "myocardium.nii.gz"))
            if not same_grid(whole_image, blood_image) or not same_grid(whole_image, myocardium_image):
                raise RuntimeError(f"Grid mismatch: {subject}/{phase_id}")
            whole = np.asanyarray(whole_image.dataobj) > 0
            blood = np.asanyarray(blood_image.dataobj) > 0
            myocardium = np.asanyarray(myocardium_image.dataobj) > 0
            remainder = whole & ~blood
            blood_outside = blood & ~whole
            phase_output = derived_root / subject / "heart_minus_blood" / phase_id
            save_mask(remainder, whole_image, phase_output / "heart_minus_blood.nii.gz")
            save_mask(blood_outside, whole_image, phase_output / "blood_outside_whole_heart.nii.gz")
            chambers_ml = {}
            for key, _, _ in CHAMBERS:
                chamber_image = nib.load(str(raw_dir / f"{key}.nii.gz"))
                if not same_grid(whole_image, chamber_image):
                    raise RuntimeError(f"Grid mismatch: {subject}/{phase_id}/{key}")
                chamber = np.asanyarray(chamber_image.dataobj) > 0
                chambers_ml[key] = volume_ml(chamber, chamber_image.affine)
            phase = {
                "phase_id": phase_id, "rr_percent": rr_percent,
                "rr_value_mod_100_unvalidated": rr_percent % 100, "rr_hundred_block_index_unvalidated": int(rr_percent // 100),
                "affine": whole_image.affine, "whole_heart": whole, "blood_pool": blood,
                "myocardium": myocardium, "heart_minus_blood": remainder,
                "blood_outside": blood_outside, "chambers_ml": chambers_ml,
                "whole_heart_ml": volume_ml(whole, whole_image.affine),
                "heart_minus_blood_ml": volume_ml(remainder, whole_image.affine),
                "blood_outside_whole_heart_ml": volume_ml(blood_outside, whole_image.affine),
            }
            phases.append(phase)
            row = {k: v for k, v in phase.items() if k not in {"affine", "whole_heart", "blood_pool", "myocardium", "heart_minus_blood", "blood_outside", "chambers_ml"}}
            row.update({f"{key}_ml": value for key, value in chambers_ml.items()})
            all_rows.append({"subject": subject, **row})

        sub_figure = subtraction_figure(subject, phases)
        vol_figure = volume_figure(subject, phases)
        subtraction_figures.append((subject, sub_figure))
        volume_figures.append((subject, vol_figure))
        sub_figure.write_html(output_dir / f"{subject}_heart_minus_blood_4d.html", include_plotlyjs=True, full_html=True, auto_play=False, config=config)
        vol_figure.write_html(output_dir / f"{subject}_chamber_volumes_rr.html", include_plotlyjs=True, full_html=True, config=config)
        summary = {"subject": subject, "phase_count": len(phases), "rr_percent_range": [phases[0]["rr_percent"], phases[-1]["rr_percent"]], "chambers": {}}
        for key, label, _ in CHAMBERS:
            values = [phase["chambers_ml"][key] for phase in phases]
            summary["chambers"][key] = {"label_ru": label, "minimum_ml": min(values), "maximum_ml": max(values), "range_ml": max(values) - min(values)}
        outside = [phase["blood_outside_whole_heart_ml"] for phase in phases]
        summary["blood_outside_whole_heart_ml"] = {"minimum": min(outside), "maximum": max(outside)}
        summaries.append(summary)
        (derived_root / subject / "heart_minus_blood" / "manifest.json").write_text(json.dumps({"schema_version": 1, "created_at": utc_now(), "subject": subject, "definition": "whole_heart AND NOT blood_pool", "warning": "geometric remainder of independent models; not validated full myocardium", "phases": [{k: v for k, v in row.items() if k not in {"subject"}} for row in all_rows if row["subject"] == subject]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with (output_dir / "chamber_volumes_by_rr.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    (output_dir / "rr_and_subtraction_summary.json").write_text(json.dumps({"schema_version": 1, "created_at": utc_now(), "dicom_phase_field": "NominalPercentageOfCardiacPhase", "phase_interpretation_status": "raw_dicom_nominal_percentage_unverified_for_values_above_100", "supporting_cardiac_timing_fields_present": False, "ecg_context": "schematic_periodic_not_measured", "subjects": summaries, "status": "automatic_segmentation_pending_manual_review"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fragments_sub = [to_html(fig, full_html=False, include_plotlyjs=("inline" if i == 0 else False), auto_play=False, config=config) for i, (_, fig) in enumerate(subtraction_figures)]
    fragments_vol = [to_html(fig, full_html=False, include_plotlyjs=False, config=config) for _, fig in volume_figures]
    combined = output_dir.parent / "21.03_Вычитание_крови_и_объёмы_камер_RR.html"
    combined.write_text(html_page(fragments_sub, fragments_vol), encoding="utf-8")

    notebook = nbformat.v4.new_notebook(metadata={"language_info": {"name": "python"}, "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    notebook.cells.append(nbformat.v4.new_markdown_cell(NOTEBOOK_CONTEXT_MD))
    for subject, _ in subtraction_figures:
        iframe = f'<iframe src="heart_rr_analysis/{subject}_heart_minus_blood_4d.html" width="100%" height="800" style="border:0"></iframe>'
        notebook.cells.append(nbformat.v4.new_markdown_cell(f"## {subject.upper()}: сопоставление геометрических масок"))
        notebook.cells.append(nbformat.v4.new_code_cell(source="# Интерактивная 4D-модель загружается из соседнего автономного HTML.", outputs=[nbformat.v4.new_output("display_data", data={"text/html": iframe}, metadata={})]))
    for index, (subject, figure) in enumerate(volume_figures):
        fragment = to_html(figure, full_html=False, include_plotlyjs=("inline" if index == 0 else False), config=config)
        notebook.cells.append(nbformat.v4.new_markdown_cell(f"## {subject.upper()}: динамика объёмов камер"))
        notebook.cells.append(nbformat.v4.new_code_cell(source="# Объёмы рассчитаны по бинарным маскам камер без сглаживания кривой.", outputs=[nbformat.v4.new_output("display_data", data={"text/html": fragment}, metadata={})]))
    nbformat.write(notebook, output_dir.parent / "21.03_Вычитание_крови_и_объёмы_камер_RR.ipynb")
    print(json.dumps({"subjects": [item[0] for item in subtraction_figures], "rows": len(all_rows), "combined_html": combined.name}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
