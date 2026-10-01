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

OPEN_TABLE = '''stage = {float(item["stage"]): item for item in open_scenario["stages"]}
landmark_rows = []
for s in sorted(stage):
    result = stage[s]["landmarks"]
    landmark_rows.append({
        "Стадия": s,
        "Число ориентиров": result["count"],
        "Средняя ошибка, мм": result["predicted_tre_mean_mm"],
        "Медиана ошибки, мм": result["predicted_tre_median_mm"],
    })
display(pd.DataFrame(landmark_rows).round(3))
print("Неподвижный контроль, средняя ошибка, мм:",
      round(stage[0.0]["landmarks"]["static_tre_mean_mm"], 3))
print("Измеренное изменение объёма по маскам лёгких, мл:",
      round(open_evaluation["observed_lung_volume_change_ml"], 1))
print("Открытые края поверхности модели:", open_evaluation["lung_mesh_open_edges_source"])
assert open_evaluation["modeled_lung_volume_status"] == "not_evaluated_non_watertight_mesh"
print("Объём поверхности модели не интерпретируется")'''


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
        elif part.startswith("## Проверка на открытой паре вдох–выдох"):
            cells.append(nbformat.v4.new_code_cell(OPEN_TABLE))

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
