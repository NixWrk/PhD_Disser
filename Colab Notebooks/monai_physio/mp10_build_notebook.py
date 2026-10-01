"""Build the MP-10 reader notebook from the maintained scientific protocol."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "Дыхательное_движение_по_одной_КТ.md"
NOTEBOOK = ROOT / "21.22_MP10_Дыхательное_движение_лёгких_и_сердца.ipynb"

SETUP = '''from pathlib import Path
import json
import os
import pandas as pd
from IPython.display import display

def local_report(variable, relative_name):
    root = os.environ.get(variable)
    if not root:
        raise RuntimeError(f"Задайте {variable} как путь к локальной папке опыта")
    path = Path(root) / relative_name
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))

nix_fit = local_report("MONAI_PHYSIO_MP10_NIX_DIR", "fit_lung/fit_report.json")
nix_scenario = local_report("MONAI_PHYSIO_MP10_NIX_DIR", "scenarios/scenario_report.json")
open_fit = local_report("MONAI_PHYSIO_MP10_L2R_DIR", "fit_lung_inhale/fit_report.json")
open_scenario = local_report("MONAI_PHYSIO_MP10_L2R_DIR", "scenarios/scenario_report.json")
open_evaluation = local_report("MONAI_PHYSIO_MP10_L2R_DIR", "evaluation.json")
open_cases = [("0001", open_fit, open_scenario, open_evaluation)]
for case in ("0002", "0003"):
    fit = local_report("MONAI_PHYSIO_MP10_L2R_BATCH_DIR", f"fit_{case}/fit_report.json")
    scenario = local_report("MONAI_PHYSIO_MP10_L2R_BATCH_DIR", f"scenarios_{case}/scenario_report.json")
    evaluation = local_report("MONAI_PHYSIO_MP10_L2R_BATCH_DIR", f"evaluation_{case}.json")
    assert fit["subject_alias"] == scenario["subject_alias"] == evaluation["pair_alias"] == f"l2r_{case}"
    open_cases.append((case, fit, scenario, evaluation))
assert nix_fit["subject_alias"] == nix_scenario["subject_alias"] == "nix"
assert open_fit["subject_alias"] == open_scenario["subject_alias"] == "l2r_0001"
print("Прочитаны локальные отчёты; изображения и маски не загружались")'''

NIX_TABLE = '''rows = []
for item in nix_scenario["stages"]:
    rows.append({
        "Стадия": item["stage"],
        "Лёгкие: среднее смещение, мм": item["lung"]["vertex_displacement_mean_mm"],
        "Сердце: среднее смещение, мм": item["heart"]["vertex_displacement_mean_mm"],
        "Сердце: смещение центра, мм": item["heart"]["centroid_shift_norm_mm"],
    })
display(pd.DataFrame(rows).round(3))
print("Внутренняя ошибка подгонки, мм:", round(nix_fit["nearest_vertex_mean_mm_stride8"], 3))
print("Это расчётный сценарий; парного выдоха Ника для проверки нет")'''

OPEN_TABLE = '''rows = []
for case, fit, scenario, evaluation in open_cases:
    stage = {float(item["stage"]): item for item in scenario["stages"]}
    control = stage[0.7]["landmarks"]
    candidate = stage[0.0]["landmarks"]
    assert evaluation["modeled_lung_volume_status"] == "not_evaluated_non_watertight_mesh"
    rows.append({
        "Случай": case,
        "Ориентиры": candidate["count"],
        "Контроль: средняя ошибка, мм": control["static_tre_mean_mm"],
        "Модель: средняя ошибка, мм": candidate["predicted_tre_mean_mm"],
        "Изменение ошибки, мм": candidate["predicted_tre_mean_mm"] - control["static_tre_mean_mm"],
        "Сердце: маски, мм": evaluation["observed_heart_centroid_shift_norm_mm"],
        "Сердце: модель, мм": evaluation["modeled_heart_centroid_shift_norm_mm"],
    })
table = pd.DataFrame(rows)
display(table.round(2))
total = table["Ориентиры"].sum()
weighted_static = (table["Ориентиры"] * table["Контроль: средняя ошибка, мм"]).sum() / total
weighted_model = (table["Ориентиры"] * table["Модель: средняя ошибка, мм"]).sum() / total
print(f"Всего {total} ориентир; взвешенная средняя ошибка: {weighted_static:.2f} → {weighted_model:.2f} мм")
print("Объём открытых лёгочных поверхностей модели не интерпретируется")'''

NIX_MOTION = '''import os
from pathlib import Path
from IPython.display import HTML, display
from mp_visualize_motion import motion_figure

scenario_dir = Path(os.environ["MONAI_PHYSIO_MP10_NIX_DIR"]) / "scenarios"
stages = ["0.00", "0.50", "0.70", "0.90"]
frames = [
    (stage, {
        "Контур лёгких по КТ": scenario_dir.parent / "fit_lung" / "lung_target_surface.vtp",
        "Модель лёгких": scenario_dir / f"lung_stage_{stage}.vtp",
        "Сердце": scenario_dir / f"heart_stage_{stage}.vtp",
    })
    for stage in stages
]
figure = motion_figure(
    frames,
    title="Ник: условное дыхательное движение лёгких и сердца",
    subtitle="Серый — исходная маска КТ; голубой — подогнанная модель. Выдох Ника не измерен.",
    colors={"Контур лёгких по КТ": "#78808d", "Модель лёгких": "#5caac3", "Сердце": "#d75864"},
    opacity={"Контур лёгких по КТ": 0.16, "Модель лёгких": 0.48, "Сердце": 0.95},
    initial_index=2,  # 0.70 — опорная стадия входной КТ
)
display(HTML(figure.to_html(full_html=False, include_plotlyjs=True)))
# Для локального просмотра без Jupyter: HTML содержит индивидуальную геометрию.
output = Path(os.environ["MONAI_PHYSIO_MP10_NIX_DIR"]) / "nix_breathing_interactive_3d.html"
figure.write_html(str(output), include_plotlyjs=True)
print("Локальный интерактивный HTML:", output)'''


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    parts = [part.strip() for part in re.split(r"(?=^## )", source, flags=re.MULTILINE)]
    cells = []
    for part in parts:
        if not part:
            continue
        cells.append(nbformat.v4.new_markdown_cell(part))
        if part.startswith("## Выполненный расчёт и источники входов"):
            cells.append(nbformat.v4.new_code_cell(SETUP))
        elif part.startswith("## Результат для КТ Ника"):
            cells.append(nbformat.v4.new_code_cell(NIX_TABLE))
        elif part.startswith("## Проверка на трёх открытых парах вдох–выдох"):
            cells.append(nbformat.v4.new_code_cell(OPEN_TABLE))
        elif part.startswith("## Интерактивная трёхмерная визуализация"):
            cells.append(nbformat.v4.new_code_cell(NIX_MOTION))

    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
    )
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK)
    print(f"Created {NOTEBOOK.name}: {len(cells)} cells")


if __name__ == "__main__":
    main()
