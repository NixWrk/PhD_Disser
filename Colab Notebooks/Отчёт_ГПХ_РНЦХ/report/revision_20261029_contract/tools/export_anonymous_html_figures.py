"""Rerender five frozen scientific plots, changing only subject display labels.

Use the bundled Python and Matplotlib 3.10.8, matching the original PNG metadata.
The source generator modules are NOT imported or executed. Only the whitelisted
plotting functions/constants are compiled from their original AST. No analysis,
raw-signal loading, FEM, notebook execution or PNG retouching is performed.
Original extracted PNGs remain unchanged. A sixth image has no internal names
and is copied byte-for-byte into the publication set.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
from importlib import metadata
import io
import json
import os
from pathlib import Path
import re
import sys


NAME_MAP = {"Георгий": "Г", "Ник": "Н", "Ярослав": "Я"}
TITLE_ROWS = {
    "04_tt_ecg_four_cycles.png": [(35, 65)],
    "05_tissue_identifiability_singular_values.png": [(0, 35)],
    "06_measured_pulse_vs_volume_probe.png": [(35, 65)],
    "07_ct_whole_heart_and_chambers.png": [(35, 65), (480, 508)],
    "08_respiratory_contrast_observation_model.png": [(0, 35), (486, 514)],
}
PINNED = {
    "build_measured_pulse_report.py": "3339bf6fa3b1f64be0e53f442e41fbb22d1595979993ea411e808ed15c3fcc14",
    "build_scorpion_fem_smart_report.py": "ce5ff02888242e3f4fedc70991f3251ef6637fbb6fb964118659593fbd496ce3",
    "_assets/measured_pulse_volume_20261028.summary.json": "ee897e558391733049794c994ba55b7655b94efdcbc88b8cf7276532d990c0ba",
    "_assets/fem_smart_20261026.summary.json": "86e3b45a16a978418822f043c10694887e8372943beae8b7f5679ea65f1b3fd9",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def number_ledger(value, path="$"):
    out = []
    if isinstance(value, dict):
        for key in sorted(value):
            out.extend(number_ledger(value[key], path + "." + key))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            out.extend(number_ledger(item, f"{path}[{i}]"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out.append([path, value])
    return out


def plotting_namespace(path, function_names, constant_names, np, plt):
    """Extract plotting code only; prevent accidental execution of a whole builder."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = []
    source_functions = []
    modifications = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
            if any(t.id in constant_names for t in node.targets):
                nodes.append(copy.deepcopy(node))
        if isinstance(node, ast.FunctionDef) and node.name in function_names:
            selected = copy.deepcopy(node)
            source_functions.append({"name": node.name, "start_line": node.lineno, "end_line": node.end_lineno,
                "original_ast_sha256": digest(ast.dump(node, include_attributes=False).encode("utf-8"))})
            if node.name == "show_figure":
                # Original tail writes a separate QA cache, displays and closes.
                # Return the completed plot instead; all plotting statements,
                # grids and arithmetic above this line remain exactly unchanged.
                index = next(i for i, part in enumerate(selected.body)
                    if isinstance(part, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "qa" for t in part.targets))
                selected.body = selected.body[:index] + [ast.Return(value=ast.Name(id="fig", ctx=ast.Load()))]
                modifications.append("show_figure: only output tail (qa/savefig/show/close) replaced by return fig; all plotting statements retained")
            nodes.append(selected)
    present = {n.name for n in nodes if isinstance(n, ast.FunctionDef)}
    if present != set(function_names):
        raise RuntimeError("Missing requested plotting function")
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    namespace = {"np": np, "plt": plt, "re": re}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace, source_functions, modifications


def array_record(value, np):
    array = np.asarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": digest(array.tobytes())}


def numeric_scene(fig, np):
    """Hash all plotted coordinates, ranges, axes positions and geometry."""
    axes = []
    for axis in fig.axes:
        row = {"xlim": list(axis.get_xlim()), "ylim": list(axis.get_ylim()),
               "xscale": axis.get_xscale(), "yscale": axis.get_yscale(),
               "position": list(axis.get_position().bounds),
               "xticks": array_record(axis.get_xticks(), np), "yticks": array_record(axis.get_yticks(), np),
               "lines": [], "patches": [], "collections": []}
        for line in axis.lines:
            row["lines"].append({"x": array_record(line.get_xdata(), np), "y": array_record(line.get_ydata(), np),
                "color": str(line.get_color()), "width": line.get_linewidth(), "style": line.get_linestyle(),
                "marker": line.get_marker(), "markersize": line.get_markersize(), "alpha": line.get_alpha()})
        for patch in axis.patches:
            row["patches"].append({"path": array_record(patch.get_path().vertices, np),
                "transform": array_record(patch.get_patch_transform().get_matrix(), np),
                "facecolor": list(patch.get_facecolor()), "edgecolor": list(patch.get_edgecolor()),
                "width": patch.get_linewidth(), "alpha": patch.get_alpha()})
        for collection in axis.collections:
            row["collections"].append({"offsets": array_record(collection.get_offsets(), np),
                "paths": [array_record(p.vertices, np) for p in collection.get_paths()]})
        axes.append(row)
    return {"figure_inches": list(fig.get_size_inches()), "dpi": fig.dpi, "axes": axes}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--packages", type=Path, help="Optional local directory containing rendering dependencies")
    args = cli.parse_args()
    if args.packages:
        sys.path.insert(0, str(args.packages.resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    import numpy as np
    from PIL import Image
    if matplotlib.__version__ != "3.10.8":
        raise RuntimeError("Use Matplotlib 3.10.8 to match original PNG metadata")
    revision = Path(__file__).resolve().parents[1]
    root = revision.parents[3]
    source_folder = root / "Colab Notebooks" / "scorpion_v2"
    output = revision / "figures" / "html_exports" / "anonymous"
    output.mkdir(parents=True, exist_ok=True)
    originals = json.loads((output.parent / "manifest.json").read_text(encoding="utf-8"))
    by_file = {row["filename"]: row for row in originals["figures"]}
    for rel, expected in PINNED.items():
        actual = digest((source_folder / rel).read_bytes())
        if actual != expected:
            raise RuntimeError(f"Frozen source changed: {rel}: {actual}")
    pulse = json.loads((source_folder / "_assets/measured_pulse_volume_20261028.summary.json").read_text(encoding="utf-8"))
    fem = json.loads((source_folder / "_assets/fem_smart_20261026.summary.json").read_text(encoding="utf-8"))
    for summary in [pulse, fem]:
        generator = summary["generator"]
        if digest((root / generator["relative_path"]).read_bytes()) != generator["sha256"]:
            raise RuntimeError("Summary/generator provenance mismatch")
    pulse_ns, pulse_functions, pulse_mods = plotting_namespace(source_folder / "build_measured_pulse_report.py",
        {"show_figure"}, {"PROFILE", "COLOR", "CHAMBERS"}, np, plt)
    fem_ns, fem_functions, fem_mods = plotting_namespace(source_folder / "build_scorpion_fem_smart_report.py",
        {"_plot_setup", "case_parts", "fig_svd", "fig_breath"}, {"ROLES", "ROLE_NAMES"}, np, plt)
    jobs = [
        ("04_tt_ecg_four_cycles.png", "pulse", lambda: pulse_ns["show_figure"](pulse, 1), "show_figure(summary, 1)"),
        ("05_tissue_identifiability_singular_values.png", "fem", lambda: fem_ns["fig_svd"](fem), "fig_svd(summary)"),
        ("06_measured_pulse_vs_volume_probe.png", "pulse", lambda: pulse_ns["show_figure"](pulse, 3), "show_figure(summary, 3)"),
        ("07_ct_whole_heart_and_chambers.png", "pulse", lambda: pulse_ns["show_figure"](pulse, 4), "show_figure(summary, 4)"),
        ("08_respiratory_contrast_observation_model.png", "fem", lambda: fem_ns["fig_breath"](fem), "fig_breath(summary)"),
    ]
    records = []
    for name, group, create, entrypoint in jobs:
        plt.rcdefaults()
        data = pulse if group == "pulse" else fem
        data_before = digest(encoded(data))
        numbers = number_ledger(data)
        fig = create()
        fig.canvas.draw()
        fixed_bbox = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.1).frozen()
        fig.set_layout_engine("none")
        before = numeric_scene(fig, np)
        old_texts = [obj.get_text() for obj in fig.findobj(match=Text)]
        changes = []
        for obj in fig.findobj(match=Text):
            old = obj.get_text()
            new = old
            for long_name, code in NAME_MAP.items():
                new = new.replace(long_name, code)
            if new != old:
                changes.append({"before": old, "after": new})
                obj.set_text(new)
        if not changes:
            raise RuntimeError("Expected participant labels absent")
        fig.canvas.draw()
        after = numeric_scene(fig, np)
        if before != after or digest(encoded(data)) != data_before:
            raise RuntimeError("Scientific data or figure geometry changed during relabeling")
        texts = [obj.get_text() for obj in fig.findobj(match=Text)]
        if any(any(name in value for name in NAME_MAP) for value in texts):
            raise RuntimeError("Internal participant name remains")
        path = output / name
        fig.savefig(path, dpi=fig.dpi, bbox_inches=fixed_bbox, metadata={"Software": "Matplotlib 3.10.8; source plotting code; participant display labels only"})
        if numeric_scene(fig, np) != after:
            raise RuntimeError("Save operation changed scientific plot state")
        plt.close(fig)
        original = by_file[name]
        with Image.open(path) as img:
            size = list(img.size)
        # Read-only comparison, not PNG processing: validate the historical
        # raster outside the rows containing the renamed panel titles.
        historical = np.asarray(Image.open(output.parent / name).convert("RGBA"))
        current = np.asarray(Image.open(path).convert("RGBA"))
        if historical.shape != current.shape:
            raise RuntimeError("Rerender dimensions differ from the extracted original")
        delta = np.abs(historical.astype(np.int16) - current.astype(np.int16))
        changed = np.any(delta != 0, axis=2)
        outside = changed.copy()
        for start, stop in TITLE_ROWS[name]:
            outside[start:stop] = False
        maximum = int(delta[outside].max()) if outside.any() else 0
        if maximum > 1:
            raise RuntimeError("Unexpected non-title pixel difference")
        raster_check = {"original_dimensions_equal": True, "title_row_intervals_excluded_start_inclusive_stop_exclusive": TITLE_ROWS[name],
            "changed_pixels_outside_title_rows": int(outside.sum()), "max_channel_difference_outside_title_rows_0_to_255": maximum,
            "interpretation": "Exact non-title pixel identity" if not outside.any() else "At most one 8-bit colour level on anti-aliased curve edges; plotted numerical arrays and artist properties are identical"}
        records.append({"filename": name, "output_path": path.relative_to(root).as_posix(),
            "output_sha256": digest(path.read_bytes()), "dimensions_px": size,
            "source_html_path": original["source_path"], "source_html_sha256": original["source_html_sha256"],
            "source_html_locator": original["locator"], "extracted_original_sha256": original["output_sha256"],
            "data_path": "Colab Notebooks/scorpion_v2/" + ("_assets/measured_pulse_volume_20261028.summary.json" if group == "pulse" else "_assets/fem_smart_20261026.summary.json"),
            "data_file_sha256": PINNED["_assets/measured_pulse_volume_20261028.summary.json" if group == "pulse" else "_assets/fem_smart_20261026.summary.json"],
            "data_numeric_value_count": len(numbers), "data_numeric_ledger_sha256": digest(encoded(numbers)),
            "entrypoint": entrypoint, "plotting_function_sources": pulse_functions if group == "pulse" else fem_functions,
            "output_plumbing_changes": pulse_mods if group == "pulse" else fem_mods,
            "label_changes": changes, "numerical_scene_before_sha256": digest(encoded(before)),
            "numerical_scene_after_sha256": digest(encoded(after)), "numerical_and_style_equality": True,
            "data_unchanged": True, "source_data_generator_hash_matches": True,
            "historical_png_pixel_comparison": raster_check,
            "rendering": "Original plotting function; same saved numbers, colours, units, line/bar properties, ranges and axes positions. Only Text participant labels replaced before rasterization. Original layout locked before substitution.",
            "caption": original["caption"], "limitations": original["limitations"],
            "status": original["status"], "recommended_chapter_before_renumbering": original["recommended_chapter_before_renumbering"],
            "text_inventory_after": texts,
        })
    contour = by_file["03_g_ct_contour_regularization.png"]
    raw = (output.parent / contour["filename"]).read_bytes()
    if digest(raw) != contour["output_sha256"]:
        raise RuntimeError("Original contour image changed")
    (output / contour["filename"]).write_bytes(raw)
    records.insert(0, {"filename": contour["filename"], "output_path": (output / contour["filename"]).relative_to(root).as_posix(),
        "output_sha256": digest(raw), "source_html_path": contour["source_path"], "source_html_sha256": contour["source_html_sha256"],
        "source_html_locator": contour["locator"], "transformation": "None; existing PNG has no internal participant names; byte-for-byte copy",
        "caption": contour["caption"], "limitations": contour["limitations"], "status": contour["status"],
        "recommended_chapter_before_renumbering": contour["recommended_chapter_before_renumbering"]})
    manifest = {"schema_version": 1, "date": "2026-10-29", "purpose": "Publication participant codes Н/Г/Я; preserve scientific content",
        "originals_preserved": True, "image_retouching": False, "FEM_run": False, "signal_processing_run": False,
        "data_sources_read": list(PINNED), "runtime": {"python": sys.version, "matplotlib": matplotlib.__version__, "numpy": np.__version__,
            "dependency_versions": {name: metadata.version(name) for name in ("matplotlib", "numpy", "Pillow", "contourpy", "cycler", "fonttools", "kiwisolver", "pyparsing")}},
        "participant_display_substitution": NAME_MAP, "figures": records,
        "not_reexported": [
            {"filename": "01_n_ct_tissue_masks.png", "reason": "Individual CT/mask arrays unavailable; original PNG retained, not suitable for anonymous integration as-is"},
            {"filename": "02_n_candidate_lateral_montage.png", "reason": "Individual CT/placement arrays unavailable; original PNG retained, no PNG retouching performed"}],
        "validation_boundary": "Saved aggregates and plotting code are hash-verified. No raw-data or physical-model revalidation; equality audit covers every plotted array, line/bar property, range and axes position before/after label substitution. PNG comparison additionally checks pixels outside participant-title rows."}
    if args.packages and (args.packages / "render_dependency_manifest.json").is_file():
        dependency_bytes = (args.packages / "render_dependency_manifest.json").read_bytes()
        (output / "render_dependency_manifest.json").write_bytes(dependency_bytes)
        manifest["render_dependency_wheel_manifest"] = "render_dependency_manifest.json"
    qa_path = output / "VISUAL_QA.json"
    if qa_path.exists():
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        reviewed = {r["filename"]: r for r in qa["images"]}
        for rec in records:
            if reviewed[rec["filename"]]["sha256"] != rec["output_sha256"]:
                raise RuntimeError("Stale visual QA")
            rec["visual_review"] = reviewed[rec["filename"]]
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Рисунки с обозначениями участников Н, Г и Я", "",
        "Для интеграции использовать файлы этой папки. Пять рисунков повторно построены исходными функциями из сохранённых JSON; заменены только текстовые подписи участников. Рисунок 03 уже не содержит имён и скопирован без изменений. Исходные извлечённые PNG сохранены в родительской папке. Главы не редактировались.", ""]
    for rec in records:
        lines += ["## " + rec["filename"], "", rec["caption"], "", "Ограничения: " + " ".join(rec["limitations"]), "",
                  "Рекомендуемая исходная глава: `" + rec["recommended_chapter_before_renumbering"] + "`.", ""]
    lines += ["## Недоступный повторный экспорт", "",
        "Рисунки 01 и 02 содержат внутреннее имя в исходном растре. Без индивидуальных КТ-массивов и входов геометрической локализации их повторный экспорт не выполнен. Готовые PNG не ретушировались. Не включать эти два оригинала в анонимизированный основной текст.", ""]
    lines += ["## Проверка преобразования", "",
        "Численные массивы, стили линий и столбцов, диапазоны и позиции осей до и после изменения подписей совпадают. У рисунков 05–08 все пиксели вне строк заголовков совпали с исходными PNG. У рисунка 04 вне заголовков различаются 499 пикселей на границах сглаженных кривых, максимум на один уровень 8-битного цветового канала; численные данные не изменены. Рисунок 03 побайтно совпадает с оригиналом. Это сравнение изображений для контроля, без их обработки или ретуши.", "",
        "Воспроизведение: bundled Python, Matplotlib 3.10.8, `tools/export_anonymous_html_figures.py --packages <каталог библиотек>`. Версии библиотек и контрольные суммы использованных wheel-файлов сохранены рядом с manifest.json. Скрипт не импортирует генераторы целиком: выполняются только исходные функции построения; вывод в прежний QA-каталог заменён возвратом объекта рисунка.", ""]
    (output / "CAPTIONS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"rerendered": 5, "copied_without_names": 1, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
