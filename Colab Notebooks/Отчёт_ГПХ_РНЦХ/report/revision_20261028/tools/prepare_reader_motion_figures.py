"""Перерисовать 9.1–9.5 из сохранённых результатов; без FEM и инверсий.

Из исходных программ извлекаются только функции/блоки Matplotlib через AST.
Расчётные модули не импортируются. Данные и цвета графических объектов до и
после редакторских изменений сравниваются побитно по нормализованному JSON.
Исходные PNG, программы и входные таблицы остаются неизменными.
"""

from __future__ import annotations

import ast
import copy
import csv
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REVISION = Path(__file__).resolve().parents[1]
ROOT = REVISION.parents[3]
REPORT = REVISION.parent
SOURCE = REPORT.parent
OUTPUT = REVISION / "figures" / "reader_motion"
MANIFEST = REVISION / "reviews" / "motion_figure_edits.json"
GEO = ROOT / "Colab Notebooks/cardiac_4d_segmentation/results/nik_whole_heart_20261028/exploratory/full_geometry_v1"
LOCAL = SOURCE / "pulmovista_check_20261028/local"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def file_info(path):
    return {"path": relative(path), "sha256": digest(path)}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    def value(cell):
        if cell == "":
            return None
        try:
            return float(cell)
        except ValueError:
            return cell
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return [{key: value(cell) for key, cell in row.items()}
                for row in csv.DictReader(stream)]


def extract_plot(path, name, mode, namespace):
    """Compile only the selected plot function and literal scenario constants."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = copy.deepcopy(next(node for node in tree.body
                                  if isinstance(node, ast.FunctionDef) and node.name == name))
    constants = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                {"SCENARIOS", "SHAPES", "POSES", "STATES", "FORMS"} for t in node.targets):
            constants.append(node)

    def new_figure(node):
        return (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "subplots")

    def assignment_to(node, names):
        return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names
                                                   for t in node.targets)

    starts = [i for i, node in enumerate(function.body) if new_figure(node)]
    if mode == "paired_only":
        assert len(starts) == 2
        function.body = [n for n in function.body if assignment_to(n, {"ids", "labels"})] + function.body[starts[1]:]
    elif mode == "first_only":
        assert len(starts) == 2
        function.body = function.body[:starts[1]]
    elif mode == "last_two":
        assert len(starts) == 3
        function.body = [n for n in function.body if assignment_to(n, {"colors"})] + function.body[starts[1]:]
    elif mode != "whole":
        raise ValueError(mode)
    # Keep the Matplotlib cache in the temporary directory selected by main().
    function.body = [n for n in function.body if not (isinstance(n, ast.Assign) and
                     any(ast.unparse(t).startswith("os.environ[") for t in n.targets))]
    module = ast.fix_missing_locations(ast.Module(body=constants + [function], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    info = {**file_info(path), "function": name, "selected_block": mode,
            "selected_ast_sha256": hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest()}
    return namespace[name], info


def artist_data(fig, np):
    """Numerical series, panel limits and colours; exclude all text/layout."""
    def arr(x):
        return np.asarray(x).tolist()
    result = []
    for axis in fig.axes:
        result.append({
            "xlim": list(axis.get_xlim()), "ylim": list(axis.get_ylim()),
            "lines": [{"x": arr(line.get_xdata()), "y": arr(line.get_ydata()),
                       "color": line.get_color(), "marker": line.get_marker(),
                       "style": line.get_linestyle()} for line in axis.lines],
            "patches": [{"vertices": arr(patch.get_path().vertices),
                         "facecolor": arr(patch.get_facecolor()),
                         "edgecolor": arr(patch.get_edgecolor()),
                         "rectangle": [float(patch.get_x()), float(patch.get_y()), float(patch.get_width()), float(patch.get_height())]
                         if hasattr(patch, "get_width") else None} for patch in axis.patches],
            "collections": [{"offsets": arr(item.get_offsets()),
                             "facecolors": arr(item.get_facecolors()),
                             "edgecolors": arr(item.get_edgecolors()),
                             "sizes": arr(item.get_sizes()) if hasattr(item, "get_sizes") else None}
                            for item in axis.collections],
        })
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest(), result


def legend_text(axis, mapping):
    legend = axis.get_legend()
    if legend:
        for item in legend.get_texts():
            item.set_text(mapping.get(item.get_text(), item.get_text()))
            item.set_fontsize(10)


def footer(fig, text, y=.025):
    for item in list(fig.texts):
        if item is not fig._suptitle:
            item.remove()
    fig.text(.5, y, text, ha="center", va="bottom", fontsize=11)


def edit_paired(fig):
    axes = fig.axes
    for j in range(3):
        axes[j + 3].set_xlabel("Опора: 00–08 — первый R–R;\n09–11 — второй R–R")
        legend_text(axes[j + 3], {"Цели цикла 0": "Цели первого R–R", "Цели цикла 1": "Цели второго R–R"})
    axes[0].set_ylabel("Средний Dice по исходной\nвоксельной сетке; 11 целей")
    axes[3].set_ylabel("Перенос − фиксированный центр,\nпроцентные пункты Dice")
    fig.suptitle("Перенос при заданном объёме и неизменной форме опоры", fontsize=16, y=.98)
    footer(fig, "Объём цели известен в обоих вариантах; самосравнения исключены. Участник Н.\n"
           "Маски усечены верхней границей поля зрения; доступны части двух интервалов R–R.\n"
           "Поворот и отношения размеров фиксированы; это геометрический опыт без электрической инверсии.")
    fig.tight_layout(rect=(0, .12, 1, .95))


def edit_law(fig):
    fig.set_size_inches(15, 7.2)
    for axis in fig.axes:
        legend_text(axis, {"Наблюдение RR0": "Наблюдение: первый R–R", "Наблюдение RR1": "Наблюдение: второй R–R",
                          "k по RR0; без подгонки RR1": "Закон первого R–R; без подгонки второго",
                          "Прогноз исключённой фазы RR0": "Прогноз исключённой фазы первого R–R"})
    legend = fig.axes[0].get_legend()
    handles, _ = fig.axes[0].get_legend_handles_labels()
    labels = [item.get_text() for item in legend.get_texts()]
    legend.remove()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .88), ncol=3,
               fontsize=10, frameon=False)
    for item in fig.axes[1].texts:
        if item.get_text() == "08":
            item.set_position((-15, 6))
        elif item.get_text() == "09":
            item.set_position((6, -8))
    for item in fig.axes[2].texts:
        if item.get_text() == "08":
            item.set_position((-16, -10))
        elif item.get_text() == "07":
            item.set_position((5, -11))
    fig.suptitle("Опора 00 участника Н: смещение центроида и закон d = k·ΔV", fontsize=16, y=.98)
    footer(fig, "ΔV — известное изменение объёма усечённой КТ-маски; координаты RAS+, мм.\n"
           "Коэффициенты оценены по нескольким фазам первого R–R.\n"
           "Это проверка закона движения; восстановление по одной КТ и ТТРКГ здесь не проверено.")
    fig.tight_layout(rect=(0, .18, 1, .76))


def edit_motion(fig):
    fig.set_layout_engine(None)
    fig.set_size_inches(15, 9.2)
    for i, label in enumerate(("Д4", "Э-L1")):
        left, middle, right = fig.axes[i * 3:i * 3 + 3]
        left.set(title=label + ": отношение чувствительностей", xlabel="Ось сдвига (LPS)",
                 ylabel="Объёмный эквивалент сдвига,\nмл/мм")
        legend_text(left, {"raw": "До поправки объёма", "volume_corrected": "После поправки объёма"})
        middle.set(title="Чувствительность к объёму\nв семи положениях", ylabel="Объёмный коэффициент, мОм/мл",
                   xlabel="Смещение сердца (LPS), мм")
        middle.set_xticks(range(7), ["Опора", "x−5", "x+5", "y−5", "y+5", "z−5", "z+5"])
        right.set(title="Постоянный заданный объём;\nпоправка численного объёма", xlabel="Ось сдвига (LPS)",
                  ylabel="Кажущийся объём, мл")
        legend_text(right, {"-5 mm": "−5 мм", "+5 mm": "+5 мм"})
    fig.suptitle("ТТРКГ «рука–рука»: движение и объёмная чувствительность L1", fontsize=16, y=.98)
    footer(fig, "Д4: ρ сердца = 4 Ом·м. Э-L1: собственные условные параметры эллипсоида.\n"
           "Кажущийся объём при сдвиге не является УО. Шкалы двух материальных опор различаются.\n"
           "Монтаж записи не подтверждён; движение и УО/ФВ в этом опыте не измерены.")
    fig.tight_layout(rect=(0, .13, 1, .94), h_pad=2.0, w_pad=2.0)


def edit_errors(fig):
    fig.set_layout_engine(None)
    fig.set_size_inches(13, 9.5)
    labels = ["V−5%", "V+5%", "x−5\nмм", "x+5\nмм", "y−5\nмм", "y+5\nмм", "z−5\nмм", "z+5\nмм"]
    for col, label in enumerate(("Д4", "О-STL")):
        fig.axes[col].set_title(label + ": диагностика ЭИТ, 7 параметров")
        fig.axes[col + 2].set_title(label + ": ложная поправка S3 → ТТРКГ → ΔV")
        for index in (col, col + 2):
            fig.axes[index].set_xticks(range(8), labels)
            fig.axes[index].set_ylabel("Ошибка ΔV, мл")
    fig.suptitle("Ошибки объёма на конечных STL-пробах\nТТРКГ — отведение «рука–рука»", fontsize=16, y=.985)
    footer(fig, "STL/G1/L1 — обратные формы. Относительные веса общей базы STL; собственная база вычтена.\n"
           "Численный объём переносов скорректирован. S3: свободны ρ мягкого фона, лёгкого и сердца.\n"
           "Истинные Δρ = 0; проекция физиологически не принята. Шкалы панелей различаются.")
    fig.tight_layout(rect=(0, .14, 1, .92), h_pad=2.3, w_pad=2.0)


def edit_chain(fig):
    fig.set_layout_engine(None)
    fig.set_size_inches(12.5, 7.3)
    for axis, label in zip(fig.axes, ("Д4", "О-STL")):
        axis.set_title(label, fontsize=14)
        axis.set_ylabel("Исходный и остаточный\nобъёмные коэффициенты, мОм/мл")
        legend_text(axis, {"Kg исходный": "Исходный Kᵥ", "Keff S2": "Остаточный Kᵥ: S2", "Keff S3": "Остаточный Kᵥ: S3"})
    handles, _ = fig.axes[0].get_legend_handles_labels()
    labels = [item.get_text() for item in fig.axes[0].get_legend().get_texts()]
    for axis in fig.axes:
        axis.get_legend().remove()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .84), ncol=3,
               fontsize=10, frameon=False)
    fig.suptitle("Диагностика смешения в целевой цепочке\nТТРКГ «рука–рука»: до и после тканевой поправки", fontsize=16, y=.985)
    footer(fig, "S2: свободны ρ мягкого фона и лёгкого; ρ сердца фиксировано.\n"
           "S3: свободны ρ мягкого фона, лёгкого и сердца. Истинные Δρ = 0.\n"
           "Общие относительные веса STL; поправка численного объёма.\n"
           "Шум не задан; проекции физиологически не приняты. Шкалы панелей различаются.")
    fig.tight_layout(rect=(0, .22, 1, .78), w_pad=2.5)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tracked = {}
    records = []

    def track(path):
        tracked[path] = digest(path)
        return path

    with tempfile.TemporaryDirectory(prefix=".mpl-cache-", dir=OUTPUT) as cache:
        assert Path(cache).resolve().parent == OUTPUT.resolve()
        os.environ["MPLCONFIGDIR"] = cache
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.text import Text
        import numpy as np
        from PIL import Image
        import re

        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12,
                             "axes.labelsize": 11, "savefig.facecolor": "white"})
        mapping = {
            "paired_motion_effect.png": ("9.1", "ch16_paired_motion_effect.png", edit_paired,
                ["Русские оси и легенды; интервалы R–R вместо полных циклов", "Участник Н; усечение поля зрения и известные объёмы"]),
            "reference_00_displacement_vs_volume.png": ("9.2", "ch16_motion_volume_law.png", edit_law,
                ["Раскрыты R–R, известный объём усечённой маски и координаты RAS+", "Сохранён диагностический статус закона и исключённых фаз",
                 "Легенда вынесена за область данных; разнесены подписи близких точек 07–09"]),
            "01_tt_motion_sensitivity.png": ("9.3", "ch16_tt_motion_sensitivity.png", edit_motion,
                ["Переведены все оси, заголовки и легенды; ТТРКГ рука–рука", "Д4/Э-L1, единицы и объёмная поправка; кажущийся объём не УО"]),
            "03_finite_probe_volume_errors.png": ("9.4", "ch17_03_finite_probe_volume_errors.png", edit_errors,
                ["Д4/О-STL, общие относительные веса и поправка численного объёма", "Добавлены мм у сдвигов; ложная S3 при истинных Δρ=0, физиологически не принята"]),
            "02_target_chain_volume_sensitivity.png": ("9.5", "ch17_02_target_chain_volume_sensitivity.png", edit_chain,
                ["Целевой путь переименован в диагностику смешения", "Раскрыты S2/S3, исходный и остаточные коэффициенты; ТТРКГ рука–рука", "Нулевые истинные Δρ, отсутствие модели шума и физиологического принятия"]),
        }
        current = {}
        original_savefig = Figure.savefig

        def save_reader(fig, name, *args, **kwargs):
            number, original_name, editor, edits = mapping[Path(name).name]
            original = track(REPORT / "figures" / original_name)
            fig.canvas.draw()
            before, data = artist_data(fig, np)
            editor(fig)
            fig.canvas.draw()
            after, _ = artist_data(fig, np)
            assert before == after, "Data/colour/limit change while editing figure " + number
            texts = [item.get_text() for item in fig.findobj(Text) if item.get_visible() and item.get_text()]
            banned = re.compile(r"\b(?:Nik|Nix|Scorpion|diagnostic4|commonSTLfit|effectiveL1|whole-heart|native-voxel|corrected|raw|TT|RR0|RR1|FOV)\b", re.I)
            assert not [s for s in texts if banned.search(s)], "Internal labels remain in " + number
            target = OUTPUT / ("09_" + number.split(".")[1] + "_" + original_name.split("_", 1)[1])
            original_savefig(fig, target, dpi=240, bbox_inches="tight", pad_inches=.18)
            with Image.open(target) as image:
                dimensions = list(image.size)
            records.append({"figure": number, "original_source": file_info(original),
                "inputs": [file_info(p) for p in current["inputs"]], "generator": current["generator"],
                "new_png": {**file_info(target), "dimensions_px": dimensions,
                            "relative_from_chapter": Path(os.path.relpath(target, REVISION / "chapters")).as_posix()},
                "edits_ru": edits + (["Общая легенда вынесена за область столбцов и числовых подписей"] if number == "9.5" else []),
                "integrity": {"numeric_artist_sha256_before": before, "numeric_artist_sha256_after": after,
                    "data_colours_limits_unchanged": True, "axis_count": len(data),
                    "line_count": sum(len(a["lines"]) for a in data),
                    "patch_count": sum(len(a["patches"]) for a in data),
                    "collection_count": sum(len(a["collections"]) for a in data)},
                "visible_texts": texts, "visual_review": {"status": "pending"}})
            print("RENDERED", number, target.name, dimensions, flush=True)

        Figure.savefig = save_reader
        try:
            phase_path = track(GEO / "motion_geometry_v1/provenance.local.json")
            phases = sorted(read_json(phase_path)["phases"], key=lambda p: (p["cycle_index"], p["rr_percent"]))
            assert len(phases) == 12
            stats_path = track(GEO / "motion_geometry_v1/motion_effect_stats.json")
            program = track(ROOT / "Colab Notebooks/cardiac_4d_segmentation/check_whole_heart_motion.py")
            namespace = {"np": np, "plt": plt, "matplotlib": matplotlib, "os": os}
            function, info = extract_plot(program, "draw_plots", "paired_only", namespace)
            current.update(inputs=[phase_path, stats_path], generator=info)
            function(OUTPUT, [{"row": p} for p in phases], [], read_json(stats_path)["rows"], [], [])

            law = GEO / "motion_law_v1"
            law_paths = [track(law / name) for name in ("anchor_coefficients_R2.json", "heldout_RR0.json", "transfer_RR0_to_RR1.json")]
            fit = next(r for r in read_json(law_paths[0])["rows"] if r["reference_phase_id"] == "phase_00")
            folds, external = [[r for r in read_json(p)["rows"] if r["reference_phase_id"] == "phase_00"] for p in law_paths[1:]]
            program = track(ROOT / "Colab Notebooks/cardiac_4d_segmentation/check_whole_heart_motion_law.py")
            function, info = extract_plot(program, "plot_reference_00", "whole", dict(namespace))
            current.update(inputs=[phase_path] + law_paths, generator=info)
            function(OUTPUT, phases, np.array([p["moments"]["volume_ml"] for p in phases]),
                     np.array([p["moments"]["centroid_mm"] for p in phases]), fit, folds, external)

            motion = LOCAL / "motion_electrical_analysis_v1"
            motion_paths = [track(motion / (name + ".csv")) for name in ("position_derivatives", "volume_derivatives")]
            rows = {p.stem: read_csv(p) for p in motion_paths}
            program = track(ROOT / "MATLAB_TRKG4_real_subjects/tools/heart_motion_electrical_analysis.py")
            function, info = extract_plot(program, "figures", "first_only", dict(namespace))
            current.update(inputs=motion_paths, generator=info)
            function(OUTPUT, rows)

            shape = LOCAL / "shape_electrical_comparison_v1"
            shape_paths = [track(shape / (name + ".csv")) for name in
                           ("target_chain_effective_geometry", "cross_inversion", "geometry_to_tissue_leakage")]
            rows = {p.stem: read_csv(p) for p in shape_paths}
            program = track(ROOT / "MATLAB_TRKG4_real_subjects/tools/heart_shape_electrical_comparison.py")
            function, info = extract_plot(program, "figures", "last_two", dict(namespace))
            current.update(inputs=shape_paths, generator=info)
            function(OUTPUT, rows)
        finally:
            Figure.savefig = original_savefig
            plt.close("all")

        assert len(records) == 5 and len({r["figure"] for r in records}) == 5
        assert all(digest(path) == sha for path, sha in tracked.items()), "A source changed during rendering"
        result = {"schema": "reader_motion_figure_edits_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "scope_ru": "Только представление сохранённых данных. FEM, инверсии, подгонки и сегментация не запускались.",
            "reader_generator": file_info(__file__), "python": sys.version, "matplotlib": matplotlib.__version__,
            "numpy": np.__version__, "path_base": "repository_root", "chapter_links_modified": False,
            "originals_inputs_and_generators_unchanged": True,
            "figures": sorted(records, key=lambda r: r["figure"])}
        MANIFEST.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("MANIFEST", relative(MANIFEST), flush=True)


if __name__ == "__main__":
    main()
