"""Add reproducible local 3D viewers to the cardiac forecast notebooks."""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parent
SOURCE = '''from pathlib import Path
from IPython.display import HTML, display
from mp_visualize_motion import motion_figure

surface_dir = Path("runs/mp06_nix_one_ct_forecast")
phases = [f"{number:02d}" for number in range(9)]
frames = [
    (phase, {"Сетевой прогноз сердца": surface_dir / f"phase_{phase}_predicted_surface.vtp"})
    for phase in phases
]
figure = motion_figure(
    frames,
    title="Ник: готовый сетевой прогноз формы сердца",
    subtitle="Фазы 00–08, номинальная доля R–R; кадры — прогноз из фазы 00, не измерение движения.",
    colors={"Сетевой прогноз сердца": "#d75864"},
    opacity={"Сетевой прогноз сердца": 0.96},
)
display(HTML(figure.to_html(full_html=False, include_plotlyjs=True)))
# Этот локальный HTML содержит индивидуальную геометрию и исключён из Git.
output = surface_dir / "heart_forecast_interactive_3d.html"
figure.write_html(str(output), include_plotlyjs=True)
print("Локальный интерактивный HTML:", output)'''

COMPARE_SOURCE = '''import os
from pathlib import Path
from IPython.display import HTML, display
from mp_visualize_motion import motion_figure

surface_dir = Path("runs/mp06_nix_one_ct_forecast")
observed_dir = Path(os.environ["MONAI_PHYSIO_DERIVED_4D_ROOT"]) / "nix" / "products_v2"
frames = [
    (f"{index:02d}", {
        "Прогноз из фазы 00": surface_dir / f"phase_{index:02d}_predicted_surface.vtp",
        "Отдельная маска целевой фазы": observed_dir / f"phase_{index:02d}" / "whole_heart.stl",
    })
    for index in range(9)
]
figure = motion_figure(
    frames,
    title="Ник: сетевой прогноз и отдельно сегментированная КТ",
    subtitle="Красная поверхность — прогноз; голубая — автоматическая маска снятой фазы.",
    colors={"Прогноз из фазы 00": "#d75864", "Отдельная маска целевой фазы": "#3b95b5"},
    opacity={"Прогноз из фазы 00": 0.87, "Отдельная маска целевой фазы": 0.38},
    same_topology=False,
    ras_organs={"Отдельная маска целевой фазы"},
)
display(HTML(figure.to_html(full_html=False, include_plotlyjs=True)))
output = surface_dir / "forecast_vs_observed_interactive_3d.html"
figure.write_html(str(output), include_plotlyjs=True)
print("Локальный интерактивный HTML:", output)'''

OBSERVED_SOURCE = '''import os
from pathlib import Path
from IPython.display import HTML, display
from mp_visualize_motion import motion_figure

# Локальный корень производных 4D-КТ задаётся вне ноутбука.
root = Path(os.environ["MONAI_PHYSIO_DERIVED_4D_ROOT"])
source_dir = root / "nix" / "products_v2"
frames = [
    (f"{index:02d}", {"Отдельная автоматическая маска сердца":
                      source_dir / f"phase_{index:02d}" / "whole_heart.stl"})
    for index in range(12)
]
figure = motion_figure(
    frames,
    title="Ник: 12 отдельно сегментированных фаз сердца",
    subtitle="Наблюдённые КТ; дискретные автоматические маски без соответствия вершин.",
    colors={"Отдельная автоматическая маска сердца": "#c85a62"},
    opacity={"Отдельная автоматическая маска сердца": 0.96},
    same_topology=False,
    coordinate_system="RAS",
    reduction=0.985,
)
display(HTML(figure.to_html(full_html=False, include_plotlyjs=True)))
output = Path("runs/mp09_nix_observed_interactive_3d.html")
figure.write_html(str(output), include_plotlyjs=True)
print("Локальный интерактивный HTML:", output)'''

DESCRIPTIONS = {
    "21.18": """## Интерактивный просмотр рассчитанного движения

Ниже показаны девять поверхностей Ника, полученных готовой сердечной сетью по
одной фазе 00. Сцену можно вращать и увеличивать, фазу выбирать ползунком или
воспроизводить последовательность кадров. Оси заданы в LPS, мм. Номинальные
доли R–R — вход модели; скорость воспроизведения не измерялась. Для вывода
сетка упрощена одинаковым способом во всех кадрах; показатели точности в
отчёте рассчитывались по полным поверхностям. Эта визуализация показывает
результат сети, но не подтверждает его анатомическую точность. Ячейка требует
локальные исключённые из Git поверхности в `runs/mp06_nix_one_ct_forecast/`.
""",
    "21.19": """## Интерактивный просмотр прогноза по одной КТ

Ползунок и кнопка воспроизведения показывают одновременно две формы сердца
Ника в фазах 00–08: красная получена сетевой моделью по одной фазе 00,
полупрозрачная голубая — отдельной автоматической сегментацией реально снятой
целевой фазы. Можно вращать сцену и менять масштаб. Исходные STL находятся
в RAS и переводятся в LPS, как сохранённый прогноз; оси заданы в миллиметрах.
Частота показа кадров условна и не воспроизводит время сердечного цикла.
Поверхности упрощены только для просмотра; численные метрики получены по полным
сеткам. Совпадение на экране не заменяет измерения ошибки и не делает
автоматическую маску ручным анатомическим референсом. Для запуска задайте
`MONAI_PHYSIO_DERIVED_4D_ROOT` как локальный корень `nix/products_v2/`.
Локальный HTML с индивидуальной геометрией не добавляется в Git.
""",
    "21.21": """## Интерактивный просмотр наблюдённых фаз

Ячейка ниже показывает 12 отдельных автоматических поверхностей сердца Ника,
по которым создавалась сцена OpenUSD. Модель можно вращать и увеличивать;
ползунок выбирает фазу, кнопка последовательно показывает дискретные кадры.
Каждая поверхность упрощается только для вывода. Вершины разных фаз не имеют
подтверждённого соответствия, поэтому промежуточная анимация и скорость показа
не являются измеренным непрерывным движением. Это визуальный контроль
автоматических масок, а не проверка прогноза или результата регистрации.
Индивидуальная геометрия сохраняется только в локальном HTML вне Git.
Для запуска задайте `MONAI_PHYSIO_DERIVED_4D_ROOT` как локальный корень
производных 4D-КТ, содержащий `nix/products_v2/`.
""",
}


def main() -> None:
    for prefix, description in DESCRIPTIONS.items():
        source = OBSERVED_SOURCE if prefix == "21.21" else COMPARE_SOURCE if prefix == "21.19" else SOURCE
        path = next(ROOT.glob(f"{prefix}_*.ipynb"))
        notebook = nbformat.read(path, as_version=4)
        markers = [i for i, cell in enumerate(notebook.cells)
                   if cell.cell_type == "markdown" and cell.source.startswith("## Интерактивный просмотр")]
        if markers:
            i = markers[0]
            notebook.cells[i].source = description
            notebook.cells[i + 1].source = source
            notebook.cells[i + 1].outputs = []
            notebook.cells[i + 1].execution_count = None
        else:
            notebook.cells.extend([
                nbformat.v4.new_markdown_cell(description),
                nbformat.v4.new_code_cell(source),
            ])
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        print(f"Updated {path.name}")


if __name__ == "__main__":
    main()
