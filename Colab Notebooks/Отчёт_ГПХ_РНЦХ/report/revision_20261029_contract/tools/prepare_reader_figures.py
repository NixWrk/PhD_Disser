"""Reader copies of seven report figures; saved results only, no research pipeline.

Run from the repository root:
  C:/PC/Python/python.exe "Colab Notebooks/Отчёт_ГПХ_РНЦХ/report/revision_20261028/tools/prepare_reader_figures.py"

The two --*-config options accept repository-relative paths to existing local
JSON configurations. Absolute data paths inside those files are read-only and
are represented in the output manifest by configuration field, never by root.
Only this revision's figures/reader/*.png and reviews/figure_edits.json are
written. Original generators are parsed, never imported or run as pipelines.
The CT outline is a display boundary of existing masks, not segmentation.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from datetime import datetime, timezone

import numpy as np
import nibabel as nib
from scipy.io import loadmat
from scipy.ndimage import binary_erosion

REV = Path(__file__).resolve().parents[1]
ROOT = REV.parents[3]
REPORT = ROOT / "Colab Notebooks/Отчёт_ГПХ_РНЦХ"
CARDIAC = ROOT / "Colab Notebooks/cardiac_4d_segmentation"
LOCAL = REPORT / "pulmovista_check_20261028/local"
OUTPUT = REV / "figures/reader"
MANIFEST = REV / "reviews/figure_edits.json"
MATLAB_TOOLS = ROOT / "MATLAB_TRKG4_real_subjects/tools"
MOTION = CARDIAC / "results/nik_whole_heart_20261028/exploratory/full_geometry_v1/motion_geometry_v1"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def configured_path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def relative_argument(value):
    if Path(value).is_absolute():
        raise ValueError("Configuration arguments must be repository-relative")
    p = (ROOT / value).resolve()
    p.relative_to(ROOT)
    return p


class Inputs:
    def __init__(self):
        self.entries = []
        self.checks = []

    def add(self, path, via=None):
        p = Path(path)
        digest = sha(p)
        # External files are identifiable through their read-only configuration.
        try:
            address = {"path": relative(p)}
        except ValueError:
            if via is None:
                raise ValueError("External input needs a configuration field")
            address = {"configuration_field": via}
        self.entries.append({**address, "sha256": digest, "bytes": p.stat().st_size})
        self.checks.append((p, digest))
        return p

    def json(self, path):
        return json.loads(self.add(path).read_text(encoding="utf-8-sig"))

    def csv(self, path):
        with self.add(path).open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    def unchanged(self):
        if any(sha(p) != digest for p, digest in self.checks):
            raise RuntimeError("A read-only input changed during rendering")


def visual_ast(source, function, replacements=None, start=None, stop=None):
    """Extract a reviewed visual function/block, with exact string replacements.

    No module-level code, pipeline entry point, or unrelated statements execute.
    Optional start/stop are statement line bounds within the chosen function.
    """
    tree = ast.parse(Path(source).read_text(encoding="utf-8-sig"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    if start is not None:
        statements = [n for n in node.body if n.lineno >= start and n.end_lineno <= stop]
    else:
        statements = [copy.deepcopy(node)]
    module = ast.Module(body=copy.deepcopy(statements), type_ignores=[])
    replacements = replacements or {}

    class Labels(ast.NodeTransformer):
        def visit_Constant(self, n):
            if isinstance(n.value, str):
                n.value = replacements.get(n.value, n.value)
            return n

    module = ast.fix_missing_locations(Labels().visit(module))
    return compile(module, str(source), "exec")


def check_visual_text(fig):
    from matplotlib.text import Text
    labels = [x.get_text() for x in fig.findobj(Text) if x.get_text()]
    prohibited = r"(?i)\b(?:Scorpion|Nik|Nix|Ник|Ника|Georg|Yaroslav)\b|SCV2-|pending_manual_review"
    if any(re.search(prohibited, text) for text in labels):
        raise ValueError("Internal name remains in a visible label")
    return list(dict.fromkeys(labels))


def save_figure(fig, name, dpi):
    texts = check_visual_text(fig)
    path = OUTPUT / name
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path, texts


def blocks_preview(inputs, _args):
    source = inputs.add(REPORT / "execution_20261026/signals/extract_signals.py")
    base = REPORT / "execution_20261026/signals/exploratory"
    blocks = []
    for n in (1, 2):
        block = inputs.json(base / f"SCV2-G-012_inhale_block{n:02d}.json")
        path = inputs.add(ROOT / block["npz_relative_path"])
        assert sha(path) == block["npz_sha256"]
        block["block_id"] = f"Г, запись 012, блок {n}"
        block["montages"] = ["ТТРКГ 1×1", "ТЕПС-2"]
        blocks.append(block)
    # Extract only render_preview; replace its save/close with the common writer.
    scope = {"np": np, "ROOT": ROOT, "LOCAL": OUTPUT}
    replacements = {
        "\nQRS candidates; ED/ES unknown": "\nКандидаты QRS; конец диастолы и систолы не установлен",
        "ECG (mV)": "ЭКГ, мВ",
        "0.7-12 Hz BEI (mOhm)": "Общий импеданс 0,7–12 Гц, мОм",
        "Time from record start (s)": "Время от начала записи, с",
        "Demeaned cycle (mOhm)": "Центрированный цикл, мОм",
        "QRS-to-QRS phase; median +/- sample SD": "Фаза QRS–QRS, доля цикла\nМедиана ± выборочное стандартное отклонение",
        "Exploratory real signals: annotations remain pending_manual_review": "Исследовательский просмотр записей: разметка ожидает ручной проверки",
    }
    # The original function only reads NPZ and plots; no filters are extracted.
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "render_preview")
    code = visual_ast(source, "render_preview", replacements)
    # Intercept savefig/close to validate text and redirect exactly one output.
    from matplotlib.figure import Figure
    old_save, old_close = Figure.savefig, plt.close
    captured = []

    def capture(fig, filename, *a, **kw):
        texts = check_visual_text(fig)
        target = OUTPUT / "05_01_selected_blocks_preview.png"
        old_save(fig, target, *a, **kw)
        captured.append((target, texts))

    try:
        Figure.savefig = capture
        exec(code, scope)
        scope["render_preview"](blocks)
    finally:
        Figure.savefig = old_save
        plt.close = old_close
    assert len(captured) == 1
    return *captured[0], {"visual_function": "render_preview", "lines": [node.lineno, node.end_lineno],
        "unchanged_arrays": "time, filtered ECG, saved pulse, median and SD; no filtering",
        "cycle_pairs_source": "saved block JSON"}


def patient_config(inputs, args):
    return inputs.json(relative_argument(args.patient_config))


def ct_belt(inputs, args):
    inputs.add(MATLAB_TOOLS / "pulmovista_patient_figures.py")
    c = patient_config(inputs, args)
    paths = {}
    for name in ("heart", "fat", "bone", "lung"):
        paths[name] = inputs.add(configured_path(c["mask_paths"][name]),
                                 via=f"{args.patient_config}:mask_paths.{name}")
    image = nib.load(paths["heart"])
    affine = image.affine
    plane = c["belt_height_m"] * 1000 + .01
    k = int(round((plane - affine[2, 3]) / affine[2, 2]))
    labels = np.zeros(image.shape[:2], int)
    for name, label in [("fat", 4), ("heart", 3), ("bone", 2), ("lung", 1)]:
        mask_image = nib.load(paths[name])
        assert mask_image.shape == image.shape and np.allclose(mask_image.affine, affine)
        # Same slice, order and >0 display rule as the original generator.
        mask = np.asarray(mask_image.dataobj)[:, :, k] > 0
        labels[mask] = label
    ij = np.indices(labels.shape)
    pts = np.c_[ij[0].ravel(), ij[1].ravel(), np.full(labels.size, k)]
    xy = nib.affines.apply_affine(affine, pts)[:, :2] * -1
    data_dir = configured_path(c["patient_output_dir"])
    el = np.loadtxt(inputs.add(data_dir / "electrode_centres_m.csv",
                   via=f"{args.patient_config}:patient_output_dir/electrode_centres_m.csv"), delimiter=",") * 1000
    from matplotlib.colors import ListedColormap
    fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
    ax.pcolormesh(xy[:, 0].reshape(labels.shape), xy[:, 1].reshape(labels.shape), labels,
        cmap=ListedColormap(["#f5f5f5", "#7cb1d4", "#d9cfbe", "#c55d69", "#eadb84"]),
        vmin=0, vmax=4, shading="nearest")
    ax.plot(np.r_[el[:, 0], el[0, 0]], np.r_[el[:, 1], el[0, 1]], "k:", lw=1)
    ax.scatter(el[:, 0], el[:, 1], s=25, c="black")
    for i, (x, y, _) in enumerate(el):
        ax.annotate(str(i + 1), (x, y), xytext=(5, 5), textcoords="offset points")
    ax.set(aspect="equal", xlim=(-230, 230), ylim=(-150, 180), xlabel="LPS X, мм", ylabel="LPS Y, мм",
        title=f"Сечение тканевых масок Н и расчётные центры пояса ЭИТ\n"
              f"Около z = {plane:.2f} мм; голубой — лёгкие, красный — сердечная маска\n"
              "Серый — кости, жёлтый — жир; положение пояса алгоритмическое")
    return *save_figure(fig, "05_02_ct_belt_section.png", 160), {
        "visual_block": "pulmovista_patient_figures.py:29–47", "slice_index": k,
        "plane_mm": plane, "display_label_order": ["fat", "heart", "bone", "lung"],
        "geometry": "same affine, slice, crop, label priority and electrode centres"}


def ct_overview(inputs, args):
    source = inputs.add(CARDIAC / "build_cardiac_ct_review.py")
    config_path = relative_argument(args.ct_config)
    config = inputs.json(config_path)
    row = next(r for r in config["phases"] if r["subject_id"] == "nix" and r["phase_id"] == "phase_09")
    manifest = inputs.json(CARDIAC / "heart_rr_analysis/function_analysis/ct_review/nix_phase_09/review_manifest.json")
    ct_path = inputs.add(configured_path(row["ct_path"]), via=f"{args.ct_config}:phases[nix,phase_09].ct_path")
    assert sha(ct_path) == manifest["inputs"]["ct_sha256"]
    original = nib.load(ct_path)
    ct = nib.as_closest_canonical(original)
    assert np.allclose(ct.affine[:3, :3], np.diag(np.diag(ct.affine[:3, :3])), atol=1e-5)
    assert np.allclose(ct.affine, manifest["affine"], atol=1e-5)
    data = np.asarray(ct.dataobj, dtype=np.float32)
    labels = np.zeros(ct.shape, dtype=np.uint8)
    chamber_keys = [
        ("left_ventricle_blood", "heart_ventricle_left"),
        ("right_ventricle_blood", "heart_ventricle_right"),
        ("left_atrium_blood", "heart_atrium_left"),
        ("right_atrium_blood", "heart_atrium_right"),
    ]
    for code, (key, original_key) in enumerate(chamber_keys, 1):
        path = inputs.add(configured_path(row["raw_masks"][key]),
                          via=f"{args.ct_config}:phases[nix,phase_09].raw_masks.{key}")
        assert sha(path) == manifest["inputs"]["mask_sha256"][original_key]
        image = nib.as_closest_canonical(nib.load(path))
        assert image.shape == ct.shape and np.allclose(image.affine, ct.affine, atol=1e-5)
        mask = np.asarray(image.dataobj) > 0
        assert not np.any(mask & (labels > 0))
        labels[mask] = code
    coords = np.column_stack(np.nonzero(labels))
    lo = np.maximum(coords.min(0) - 12, 0)
    hi = np.minimum(coords.max(0) + 13, ct.shape)
    assert [lo.tolist(), hi.tolist()] == manifest["crop_index_bounds"]
    soft = np.uint8(np.clip((data + 20) / 160, 0, 1) * 255)
    spacing = np.diag(ct.affine[:3, :3])
    centre = np.rint(coords.mean(0)).astype(int)
    # Only the existing 2-D contour display function is extracted. It never
    # writes or edits masks; binary erosion identifies boundary pixels to show.
    scope = {"np": np, "binary_erosion": binary_erosion,
             "COLORS": [(25, 105, 240), (240, 65, 55), (40, 200, 235), (255, 175, 35)]}
    exec(visual_ast(source, "outline"), scope)
    outline = scope["outline"]
    fig, axes = plt.subplots(3, 2, figsize=(10, 13), facecolor="white")
    for r, (axis, title) in enumerate([(2, "Аксиальное сечение"), (1, "Фронтальное сечение"), (0, "Сагиттальное сечение")]):
        idx = int(centre[axis])
        sl = [slice(int(lo[i]), int(hi[i])) for i in range(3)]
        sl[axis] = idx
        g = soft[tuple(sl)].T[::-1, ::-1]
        lab = labels[tuple(sl)].T[::-1, ::-1]
        remaining = [i for i in range(3) if i != axis]
        aspect = spacing[remaining[1]] / spacing[remaining[0]]
        for col in range(2):
            axes[r, col].imshow(g, cmap="gray", vmin=0, vmax=255, aspect=aspect)
            if col:
                axes[r, col].imshow(outline(lab), aspect=aspect, interpolation="nearest")
            axes[r, col].set_title(f"{title}, индекс RAS {idx}" + ("\nИсходные автоматические контуры" if col else ""), fontsize=11)
            axes[r, col].axis("off")
    fig.suptitle("Н, фаза 09 | окно: уровень 60 / ширина 160 HU\n"
                 "ЛЖ — синий, ПЖ — красный, ЛП — голубой, ПП — оранжевый\n"
                 "Автоматические контуры; ручное принятие не выполнено", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    return *save_figure(fig, "06_02_phase09_overview.png", 135), {
        "visual_block": "build_review: orthogonal overview and outline",
        "source_hashes_match_review_manifest": True, "crop_index_bounds": [lo.tolist(), hi.tolist()],
        "centre_RAS_indices": centre.tolist(), "window_HU": {"level": 60, "width": 160},
        "mask_edits": 0, "resampling": "canonical axis permutation/reflection only; no interpolation"}


def center_displacement(inputs, _args):
    source = inputs.add(CARDIAC / "check_whole_heart_motion.py")
    rows = inputs.json(MOTION / "center_displacements.json")["rows"]
    stats = inputs.json(MOTION / "center_displacement_stats.json")["rows"]
    # The 12 phase coordinates are taken from saved displacement rows.
    phase_rows = [r for r in rows if r["reference_phase_id"] == "phase_00"]
    ids = [r["target_phase_id"] for r in phase_rows]
    assert len(ids) == 12 and len(set(ids)) == 12
    phases = [{"row": {"phase_id": r["target_phase_id"], "cycle_index": r["target_cycle_index"],
                        "rr_percent": r["target_rr_percent"]}} for r in phase_rows]
    # Execute only statements from ids=... through the first close(fig).
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "draw_plots")
    start = next(n.lineno for n in node.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "ids")
    finish = next(n.end_lineno for n in node.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                  and ast.unparse(n.value.func) == "plt.close")
    replacements = {
        "Nik • центроиды исследовательских масок: цикл 0 — 00–08; цикл 1 — 09–11": "Н • центроиды исследовательских масок: цикл 0 — 00–08; цикл 1 — 09–11",
        "RMS исключает самосравнение. Один испытуемый; все 12 масок касаются верхней границы FOV.\nСмещение — референсный вход абляции, а не восстановление движения по одной КТ.":
        "Среднеквадратичное смещение исключает самосравнение. Один участник; все 12 масок касаются верхней границы поля зрения.\nСмещение — референсный вход парного сравнения; восстановление движения по одной КТ не выполнено.",
        ": max ": ": максимум ", "; RMS ": "; СКЗ ",
        "center_displacement.png": "06_06_center_displacement.png",
    }
    # Intercept only this one save to inspect labels and retain source layout.
    from matplotlib.figure import Figure
    old_save = Figure.savefig
    captured = []

    def capture(fig, filename, *a, **kw):
        captured.append((Path(filename), check_visual_text(fig)))
        old_save(fig, filename, *a, **kw)

    try:
        Figure.savefig = capture
        exec(visual_ast(source, "draw_plots", replacements, start, finish), {
            "np": np, "plt": plt, "output": OUTPUT, "phases": phases,
            "displacements": rows, "motion_stats": stats})
    finally:
        Figure.savefig = old_save
    assert len(captured) == 1
    return *captured[0], {"visual_function": "draw_plots: first figure only", "lines": [start, finish],
        "matrix": "144 saved true_displacement_mm values", "coordinates": "saved target_rr_percent, not recomputed"}


def electrical_approximation(inputs, _args):
    inputs.add(MATLAB_TOOLS / "heart_stl_electrical_summary.py")
    data = inputs.json(LOCAL / "stl_electrical_comparison/summary.local.json")
    table = data["shape_comparison"]
    methods = [r["method"] for r in table]
    delta = np.array(data["reference_volume_probe_ml"])
    scenarios = data["heart_material_scenarios"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
    for row in table:
        # These are the saved plotted ordinates; do not rerun polyfit.
        axs[0].plot(delta, [row["negative_probe_mohm"], 0, row["positive_probe_mohm"]], marker="o", label=row["method"])
    axs[0].set(xlabel="Изменение заданного объёма, мл", ylabel="Изменение ТТРКГ, мОм", title="Одинаковое ΔV; ρ сердца = 4 Ом·м")
    axs[0].legend(fontsize=8, ncol=2)
    axs[1].bar(methods, [r["electrical_delta_RMSE_mohm_vs_STL"] for r in table])
    axs[1].set(ylabel="СКО приращений относительно STL, мОм", title="Расхождение при одинаковом ρ")
    axs[2].scatter([x["heart_rho_ohm_m"] for x in scenarios], [x["dZ_dV_mohm_per_ml"] for x in scenarios])
    axs[2].axhline(0, color="black", lw=.6)
    axs[2].set(xlabel="Заданное ρ целого сердца, Ом·м", ylabel="dZ/dV, мОм/мл", title="Три сценария; одна STL и монтаж")
    for ax in axs:
        ax.grid(alpha=.2)
    fig.suptitle("Условные FEM-расчёты на КТ участника Н; пригодность сферы и точность УО не установлены", fontsize=12)
    fig.tight_layout()
    return *save_figure(fig, "07_02_electrical_approximation.png", 170), {
        "visual_block": "heart_stl_electrical_summary.py:81–96",
        "plotted_values": "saved negative/positive probes, saved RMSE, saved material scenarios",
        "refitting": False, "volume_denominator": "target geometric volume, unchanged"}


def measurement_residuals(inputs, _args):
    inputs.add(MATLAB_TOOLS / "pulmovista_measurement_audit.py")
    rows = inputs.csv(LOCAL / "measurement_audit_v1/residual_rows.csv")
    pairs = inputs.csv(LOCAL / "measurement_audit_v1/measurement_pairs.csv")
    y = np.array([float(r["reference"]) for r in rows])
    a = np.array([float(r["antisymmetric_component"]) for r in pairs])
    assert len(y) == len(a) == 208
    assert np.array_equal(y, [float(r["reference_reader_value"]) for r in pairs])
    assert [int(r["row_1based"]) for r in rows] == list(range(1, 209))
    fig, ax = plt.subplots(3, 1, figsize=(13, 9), constrained_layout=True)
    ax[0].plot(y * 1000, label="Измерения в условной шкале программы чтения")
    for key, label in [("STL_nominal", "Исходная высота пояса"), ("STL_minus20mm", "Пояс на 20 мм ниже")]:
        p = np.array([float(r[key]) for r in rows])
        ax[0].plot(p * 1000, label=label)
        ax[1].plot((y - p) * 1000, label=label)
    ax[2].plot(a * 1000, label="Антисимметричная часть измеренного вектора")
    for z in ax:
        z.grid(alpha=.2)
        z.legend()
        z.set_xlabel("Номер электрической пары в исходном порядке, отсчёт с нуля")
    ax[0].set_ylabel("Напряжение, мВ\nв условной шкале")
    ax[1].set_ylabel("Измерение − расчёт, мВ")
    ax[2].set_ylabel("Антисимметричная часть, мВ")
    return *save_figure(fig, "07_03_measurement_residuals.png", 150), {
        "visual_block": "pulmovista_measurement_audit.py:198–209",
        "calibration": "unchanged conditional reader scale", "row_order": "0–207, unchanged",
        "computed_operations": "saved columns; subtraction y-p and V-to-mV display multiplier only"}


def baseline_fit(inputs, args):
    inputs.add(MATLAB_TOOLS / "pulmovista_patient_figures.py")
    c = patient_config(inputs, args)
    data_dir = configured_path(c["patient_output_dir"])
    d = loadmat(inputs.add(data_dir / "reference_prediction.mat",
                via=f"{args.patient_config}:patient_output_dir/reference_prediction.mat"))
    v, p = d["Vref"].ravel(), d["Vpred"].ravel()
    assert len(v) == len(p) == 208
    error = 100 * np.linalg.norm(v - p) / np.linalg.norm(v)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), layout="constrained")
    index = np.arange(1, len(v) + 1)
    axes[0].plot(index, v * 1000, label="Измерения после чтения EIDORS", lw=1.1)
    axes[0].plot(index, p * 1000, label="Условная FEM-подгонка", lw=1.1)
    axes[0].set(ylabel="Напряжение, мВ", title=f"Н, запись 02, опорный участок 36–44 с: невязка {error:.2f}%")
    axes[0].legend()
    axes[1].plot(index, (v - p) * 1000, color="firebrick", lw=1)
    axes[1].axhline(0, color="gray", lw=.7)
    axes[1].set(xlabel="Номер измерения (из 208)", ylabel="Измерение − расчёт, мВ")
    for ax in axes:
        ax.grid(alpha=.2)
    return *save_figure(fig, "08_01_baseline_fit.png", 160), {
        "visual_block": "pulmovista_patient_figures.py:15–27", "relative_voltage_residual_pct": float(error),
        "new_fit": False, "vectors": "Vref and Vpred from reference_prediction.mat"}


FIGURES = [
    ("5.1", REPORT / "execution_20261026/signals/exploratory/selected_blocks_preview.png", blocks_preview,
     ["Служебные идентификаторы заменены на Г, запись 012, блок 1/2", "Оси, легенды и статус русифицированы", "SD назван выборочным стандартным отклонением, не доверительным интервалом"]),
    ("5.2", LOCAL / "patient_v3/ct_belt_section.png", ct_belt,
     ["КТ Ника заменено на сечение тканевых масок Н", "Сердце уточнено до сердечной маски", "Сохранены исходный пояс, срез и границы; добавлено алгоритмическое происхождение"]),
    ("6.2", CARDIAC / "heart_rr_analysis/function_analysis/ct_review/nix_phase_09/overview.png", ct_overview,
     ["NIX заменено на Н; phase_09 — на фаза 09", "Сечения и названия камер русифицированы", "Индексы RAS и окно HU сохранены; указан непринятый статус контуров"]),
    ("6.6", REPORT / "report/figures/ch16_center_displacement.png", center_displacement,
     ["Nik заменено на Н", "FOV заменено на поле зрения; абляция — на парное сравнение", "Легенда max/RMS русифицирована; данные и раздельные R–R сохранены"]),
    ("7.2", LOCAL / "stl_electrical_comparison/electrical_approximation.png", electrical_approximation,
     ["КТ Ника заменено на КТ участника Н", "RMSE заменено на СКО; методы STL/G1–G3b/L1–L2 сохранены", "Числа взяты из сохранённой сводки без новой подгонки"]),
    ("7.3", LOCAL / "measurement_audit_v1/measurement_residuals.png", measurement_residuals,
     ["reader заменено на программу чтения", "Все три вертикальные оси содержат мВ; сохранена условность шкалы", "Ось индекса уточнена: электрические пары с отсчётом с нуля"]),
    ("8.1", LOCAL / "patient_v3/baseline_fit.png", baseline_fit,
     ["Ник заменено на Н", "Верхняя ось уточнена: напряжение, мВ", "Подгонка не выполнялась: использованы сохранённые Vref и Vpred"]),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-config", default=relative(LOCAL / "patient_config_v3.local.json"))
    parser.add_argument("--ct-config", default=relative(CARDIAC / "pilot_blood_segmentation_manifest.local.json"))
    parser.add_argument("--record-visual-review", action="store_true",
                        help="Only after inspecting all seven current PNGs: record unchanged SHA values as visually reviewed")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        parser.error("Run from the repository root")
    if args.record_visual_review:
        report = json.loads(MANIFEST.read_text(encoding="utf-8"))
        assert len(report["figures"]) == 7 and all(r["status"] == "rendered" for r in report["figures"])
        for row in report["figures"]:
            assert sha(ROOT / row["reader_png"]) == row["reader_sha256"]
            row["visual_review"] = {"status": "inspected", "sha256": row["reader_sha256"],
                                     "checks": ["заголовки, оси, единицы и легенды", "читаемость и отсутствие обрезки", "соответствие исходной геометрии и численным рядам"]}
        report["visual_review_utc"] = datetime.now(timezone.utc).isoformat()
        MANIFEST.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("Visual review recorded for 7 unchanged PNGs")
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "script": relative(__file__), "script_sha256": sha(__file__),
              "scope": "reader rendering from saved arrays/masks; no FEM, filtering, segmentation or refitting",
              "absolute_data_roots_exported": False, "figures": []}
    # Temporary Matplotlib font cache is kept under the allowed reader folder
    # and removed when rendering finishes; no cache is left in source folders.
    with tempfile.TemporaryDirectory(prefix=".mpl_", dir=OUTPUT) as cache:
        os.environ["MPLCONFIGDIR"] = cache
        import matplotlib
        matplotlib.use("Agg")
        global plt
        import matplotlib.pyplot as plt
        for number, original, renderer, edits in FIGURES:
            inputs = Inputs()
            source_sha = sha(original)
            row = {"figure": number, "original": relative(original), "original_sha256": source_sha, "edits": edits}
            try:
                plt.rcdefaults()
                plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
                path, texts, checks = renderer(inputs, args)
                inputs.unchanged()
                assert sha(original) == source_sha
                row.update(status="rendered", reader_png=relative(path), reader_sha256=sha(path),
                           inputs=inputs.entries, rendering=checks, visible_text=texts,
                           visual_review={"status": "pending"}, originals_unchanged=True)
                print(f"{number}: rendered {path.name}", flush=True)
            except Exception as exc:
                # Error text can include a private absolute path; do not export it.
                row.update(status="blocked", error_type=type(exc).__name__, inputs=inputs.entries,
                           reason="Missing/mismatched saved input or visual-rendering failure; no research fallback was run")
                print(f"{number}: {type(exc).__name__}", flush=True)
                import traceback
                traceback.print_exc()
            finally:
                plt.close("all")
            report["figures"].append(row)
            MANIFEST.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if any(r["status"] != "rendered" for r in report["figures"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
