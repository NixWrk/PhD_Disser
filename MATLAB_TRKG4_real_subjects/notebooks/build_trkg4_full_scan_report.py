"""Offline full Cartesian CEM/experiment scan report; never runs FEM.

Input directory: centres.csv, poses.csv, summary.json, manifest.json, surface.mat.
Summary: grid={u:[...],v:[...],phi:[...]}, rho1/rho2 (ohm m),
experiment={L_mm:[nine sizes],Z_ohm:[nine measured impedances]},
counts={evaluated:int,invalid_geometry:int}, complete:bool,
bestrecord={centre_index:int,phi_deg:float,rmse_ohm:float,mae_ohm:float}
(null if no evaluated pose). Manifest inputs/outputs are lists of records
{name,path,sha256}; paths are relative to manifest.json, or absolute local paths.
All report inputs must occur in outputs; every listed hash is verified.
Optional summary.pose_files=["parts/poses_0001.csv", ...] replaces poses.csv;
only these explicit, hash-verified parts are read (no directory globbing).
surface.mat is MATLAB v7, nodes_mm Nx3, boundary_faces Mx3, one-based.

Default production grid is exactly u,v=-120:1:120, phi=-20:1:20. Partial
reports require --allow-partial and a consistent, hashed snapshot. Missing
poses are never interpolated. RMSE/MAE are checked against all nine Z values.
The optional notebook is a lightweight local viewer of the HTML and PNG.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "output" / "full_pose_scan_fullscan_20260908_v2"
REQUIRED_FILES = ("centres.csv", "poses.csv", "summary.json", "surface.mat")
# Matplotlib/ColorBrewer RdYlGn_r: smaller raw RMSE is greener.
COLOURS = [[i / 10, colour] for i, colour in enumerate([
    "#006837", "#1a9850", "#66bd63", "#a6d96a", "#d9ef8b", "#ffffbf",
    "#fee08b", "#fdae61", "#f46d43", "#d73027", "#a50026"])]
INVALID_COLOUR = "#969696"
MISSING_COLOUR = "#393939"
OUTSIDE_COLOUR = "#9874b8"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def verify_manifest(directory):
    """Verify the declared provenance, including the exact files we consume."""
    manifest_digest = sha256(directory / "manifest.json")
    manifest = read_json(directory / "manifest.json")
    verified, output_paths = {str(directory / "manifest.json"): manifest_digest}, set()
    for key in ("inputs", "outputs"):
        records = manifest.get(key)
        require(isinstance(records, list), f"manifest.{key} must be a list")
        require(bool(records), f"manifest.{key} must not be empty")
        for record in records:
            require(isinstance(record, dict), f"Invalid manifest.{key} record")
            path = Path(record["path"])
            path = (directory / path).resolve() if not path.is_absolute() else path.resolve()
            require(path.name == record["name"], f"Manifest name/path mismatch: {path}")
            expected = record["sha256"].lower()
            require(len(expected) == 64 and all(c in "0123456789abcdef" for c in expected),
                    f"Invalid SHA256: {path}")
            actual = sha256(path)
            require(actual == expected, f"SHA256 mismatch: {path}")
            require(str(path) not in verified or verified[str(path)] == actual,
                    f"Conflicting manifest records: {path}")
            verified[str(path)] = actual
            if key == "outputs":
                output_paths.add(path)
    pose_files = read_json(directory / "summary.json").get("pose_files", ["poses.csv"])
    require(isinstance(pose_files, list) and bool(pose_files) and
            all(isinstance(name, str) and Path(name).suffix.lower() == ".csv" for name in pose_files),
            "summary.pose_files must be a nonempty list of CSV paths")
    resolved_parts = [(directory / name).resolve() for name in pose_files]
    require(len(set(resolved_parts)) == len(resolved_parts), "Duplicate summary.pose_files")
    for path in [directory / name for name in REQUIRED_FILES if name != "poses.csv"] + resolved_parts:
        require(path.resolve() in output_paths,
                f"Consumed file absent from manifest.outputs: {path.name}")
    return verified


def grid_axis(grid, name, alias):
    # MATLAB jsonencode emits a one-element numeric axis as a JSON scalar.
    values = np.atleast_1d(np.asarray(grid.get(name, grid.get(alias, [])), dtype=float))
    require(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(),
            f"grid.{name}: expected a nonempty finite array")
    require((np.diff(values) > 0).all(), f"grid.{name}: duplicate or unsorted values")
    return values


def finite_numbers(frame, columns, label):
    try:
        values = frame[columns].to_numpy(dtype=float)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{label}: missing or nonnumeric columns {columns}") from exc
    require(np.isfinite(values).all(), f"{label}: nonfinite values in {columns}")
    return values


def load_scan(directory, *, allow_partial=False, enforce_production_grid=True):
    """Stream CSV into centre x angle arrays; enforce_production_grid=False is for tests."""
    directory = Path(directory).resolve()
    verified = verify_manifest(directory)
    summary = read_json(directory / "summary.json")
    grid = summary["grid"]
    u, v, phi = (grid_axis(grid, *pair) for pair in
                 (("u", "u_mm"), ("v", "v_mm"), ("phi", "phi_deg")))
    production_grid = ((np.array_equal(u, np.arange(-120, 121)) or np.array_equal(u, np.arange(-80, 81))) and
                       np.array_equal(v, np.arange(-120, 121)) and
                       np.array_equal(phi, np.arange(-20, 21)))
    if enforce_production_grid:
        require(production_grid,
                "Not a declared full grid: u [-80,80] (or historical [-120,120]), v [-120,120], phi [-20,20]")
    require(all(isinstance(summary[key], (int, float)) and not isinstance(summary[key], bool)
                for key in ("rho1", "rho2")), "rho1/rho2 must be JSON numbers, not strings or booleans")
    rho = np.array([summary["rho1"], summary["rho2"]], dtype=float)
    require(np.isfinite(rho).all() and (rho > 0).all(), "rho1/rho2 must be positive ohm m")
    experiment = summary["experiment"]
    sizes = np.asarray(experiment["L_mm"], dtype=float)
    measured = np.asarray(experiment["Z_ohm"], dtype=float)
    require(sizes.shape == measured.shape == (9,) and np.isfinite(sizes).all()
            and np.isfinite(measured).all() and (sizes > 0).all()
            and np.array_equal(sizes, sizes.astype(int)) and len(set(sizes)) == 9,
            "experiment must contain nine distinct integer L_mm and finite Z_ohm")
    if enforce_production_grid:
        require(np.array_equal(np.sort(sizes), [50, 60, 70, 80, 90, 110, 120, 130, 140]),
                "Expected nine independent sizes: 50,60,70,80,90,110,120,130,140 mm")
    z_columns = [f"Z_L{int(size):03d}_ohm" for size in sizes]
    centres = pd.read_csv(directory / "centres.csv")
    values = finite_numbers(centres, ["centre_index", "u_mm", "v_mm", "x_mm", "y_mm", "z_mm"],
                            "centres.csv")
    require("h_vertex_mm" in centres, "centres.csv missing h_vertex_mm")
    # h is diagnostic geometry, may be unavailable; never use it in the RMSE score.
    h = pd.to_numeric(centres.h_vertex_mm, errors="raise").to_numpy(float)
    require((np.isnan(h) | (np.isfinite(h) & (h >= 0))).all(), "Invalid h_vertex_mm")
    require(np.equal(values[:, 0], np.floor(values[:, 0])).all(), "Noninteger centre_index")
    require(not centres.centre_index.duplicated().any(), "Duplicate centre_index")
    require(len(centres) == len(u) * len(v), "Missing or extra centres")
    ui = pd.Index(u).get_indexer(centres.u_mm)
    vi = pd.Index(v).get_indexer(centres.v_mm)
    require((ui >= 0).all() and (vi >= 0).all(), "Centre outside grid.u/grid.v")
    cell_indices = vi * len(u) + ui
    require(len(set(cell_indices)) == len(centres), "Duplicate (u,v) centre")
    require(np.array_equal(centres.centre_index.to_numpy(), cell_indices + 1),
            "centre_index must equal 1 + iu + len(u)*iv (MATLAB ndgrid, u fastest)")
    centres = centres.iloc[np.argsort(cell_indices)].reset_index(drop=True)
    ids = pd.Index(centres.centre_index)
    shape = (len(centres), len(phi))
    rmse = np.full(shape, np.nan, dtype="<f4")
    status = np.zeros(shape, dtype=np.uint8)  # 0 missing, 1 evaluated, 2 invalid_geometry
    best_rmse64 = np.full(len(centres), np.inf)
    best_angle_index = np.full(len(centres), len(phi), dtype=np.int16)
    best = None
    best_declared = summary.get("bestrecord")
    declared_best_seen = False
    rows = 0
    pose_files = summary.get("pose_files", ["poses.csv"])
    chunks = chain.from_iterable(pd.read_csv(directory / name, chunksize=100_000) for name in pose_files)
    for chunk in chunks:
        finite_numbers(chunk, ["centre_index", "phi_deg"], "poses.csv")
        require(set(chunk.status).issubset({"evaluated", "invalid_geometry", "outside_search_region"}), "Unknown pose status")
        require(set(z_columns + ["rmse_ohm", "mae_ohm"]).issubset(chunk.columns),
                "poses.csv missing nine Z columns or RMSE/MAE")
        ci = ids.get_indexer(chunk.centre_index)
        ai = pd.Index(phi).get_indexer(chunk.phi_deg)
        require((ci >= 0).all() and (ai >= 0).all(), "Pose outside declared centre/angle grid")
        flat = ci * len(phi) + ai
        require(len(np.unique(flat)) == len(flat) and not status.ravel()[flat].any(),
                "Duplicate (centre_index,phi_deg) pose")
        evaluated = chunk.status.to_numpy() == "evaluated"
        status[ci, ai] = np.where(evaluated, 1, np.where(chunk.status.to_numpy() == "outside_search_region", 3, 2))
        require(chunk.loc[~evaluated, z_columns + ["rmse_ohm", "mae_ohm"]].isna().all().all(), "Rejected/excluded poses must not contain impedance scores")
        valid = chunk.loc[evaluated]
        if len(valid):
            metrics = finite_numbers(valid, ["rmse_ohm", "mae_ohm"], "evaluated poses")
            predicted = finite_numbers(valid, z_columns, "evaluated poses")
            residual = predicted - measured
            computed_rmse = np.sqrt(np.mean(residual ** 2, axis=1))
            computed_mae = np.mean(np.abs(residual), axis=1)
            require((metrics >= 0).all() and np.allclose(metrics[:, 0], computed_rmse, rtol=1e-8, atol=1e-8)
                    and np.allclose(metrics[:, 1], computed_mae, rtol=1e-8, atol=1e-8),
                    "RMSE/MAE disagree with the nine experimental impedances (ohm)")
            require((metrics[:, 0] <= np.finfo(np.float32).max).all(), "RMSE exceeds float32 display range")
            rmse[ci[evaluated], ai[evaluated]] = metrics[:, 0]
            previous_min = best_rmse64.copy()
            np.minimum.at(best_rmse64, ci[evaluated], metrics[:, 0])
            best_angle_index[best_rmse64 < previous_min] = len(phi)
            at_minimum = metrics[:, 0] == best_rmse64[ci[evaluated]]
            np.minimum.at(best_angle_index, ci[evaluated][at_minimum], ai[evaluated][at_minimum])
            # Deterministic global tie-break independent of CSV/part ordering.
            minimum = valid[valid.rmse_ohm == metrics[:, 0].min()]
            candidate = minimum.sort_values(["centre_index", "phi_deg"]).iloc[0].to_dict()
            best_key = lambda row: (row["rmse_ohm"], row["centre_index"], row["phi_deg"])
            if best is None or best_key(candidate) < best_key(best):
                best = candidate
            if isinstance(best_declared, dict):
                match = valid[(valid.centre_index == best_declared["centre_index"]) &
                              (valid.phi_deg == best_declared["phi_deg"])]
                if len(match):
                    for key in ("rmse_ohm", "mae_ohm"):
                        require(np.isclose(float(match.iloc[0][key]), float(best_declared[key]),
                                           rtol=1e-8, atol=1e-8), f"summary.bestrecord {key} mismatch")
                    declared_best_seen = True
        rows += len(chunk)
    counts = {"evaluated": int((status == 1).sum()), "invalid_geometry": int((status == 2).sum()),
              "outside_search_region": int((status == 3).sum()), "missing": int((status == 0).sum()), "total": int(status.size), "processed": rows}
    for key in ("evaluated", "invalid_geometry", "outside_search_region"):
        require(summary["counts"].get(key, 0) == counts[key], f"summary.counts.{key} mismatch")
    for key in ("total", "processed", "missing"):
        if key in summary["counts"]:
            require(summary["counts"][key] == counts[key], f"summary.counts.{key} mismatch")
    require(isinstance(summary.get("complete"), bool), "summary.complete must be boolean")
    covered = counts["missing"] == 0
    require(not summary["complete"] or covered, "False complete claim: centre/angle coverage has gaps")
    complete = covered and summary["complete"]
    require(complete or allow_partial, "Incomplete/unfinalised scan: use --allow-partial for an explicit snapshot")
    if best is None:
        require(best_declared is None, "summary.bestrecord must be null when no poses evaluated")
    else:
        require(declared_best_seen and np.isclose(best_declared["rmse_ohm"], best["rmse_ohm"],
                                                rtol=1e-8, atol=1e-8),
                "summary.bestrecord is absent or is not a minimum of verified poses")
    mesh = loadmat(directory / "surface.mat", variable_names=["nodes_mm", "boundary_faces"])
    nodes = np.asarray(mesh["nodes_mm"], dtype=float)
    faces = np.asarray(mesh["boundary_faces"], dtype=float)
    require(nodes.ndim == 2 and nodes.shape[1] == 3 and len(nodes) > 0 and np.isfinite(nodes).all(),
            "nodes_mm must be finite Nx3 millimetres")
    require(faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0 and np.isfinite(faces).all()
            and np.equal(faces, np.floor(faces)).all() and faces.min() >= 1 and faces.max() <= len(nodes),
            "boundary_faces must be integer Mx3, one-based, within nodes_mm")
    require((np.diff(np.sort(faces, axis=1), axis=1) > 0).all(), "Repeated vertex in surface triangle")
    if enforce_production_grid:
        require(len(faces) == 155120, "Expected the full FEM boundary: 155120 faces")
    faces = faces.astype("<u4") - 1
    # Catch a writer changing a snapshot while CSV/MAT files were being consumed.
    for path, digest in verified.items():
        require(sha256(path) == digest, f"Input changed while building report: {path}")
    valid_rmse = rmse[status == 1]
    low, high = (float(valid_rmse.min()), float(valid_rmse.max())) if len(valid_rmse) else (0., 1.)
    if low == high:
        high = low + max(1., abs(low) * 0.01)
    best_angle_index[~np.isfinite(best_rmse64)] = -1
    return dict(directory=directory, summary=summary, u=u, v=v, phi=phi, centres=centres,
                rmse=rmse, status=status, counts=counts, complete=complete, best=best,
                nodes=nodes, faces=faces, colours=COLOURS, colour_range=[low, high],
                verified=verified, best_angle_index=best_angle_index, production_grid=production_grid)


def best_layer(scan):
    angle = scan["best_angle_index"]
    row = np.arange(len(angle))
    available = angle >= 0
    values = np.where(available, scan["rmse"][row, angle], np.nan)
    state = np.where(available, 1, np.where((scan["status"] == 0).any(axis=1), 0, np.where((scan["status"] == 3).all(axis=1), 3, 2)))
    return values, np.where(available, scan["phi"][angle], np.nan), state


def grid_label(scan):
    label = (f"Рабочая сетка {len(scan['centres']):,} × {len(scan['phi'])}" if scan["production_grid"] else "Контрольная сетка (smoke)")
    if not scan["complete"]:
        label += " — НЕПОЛНЫЙ / НЕФИНАЛИЗИРОВАННЫЙ СНИМОК"
    return label


def status_text(scan):
    counts, summary = scan["counts"], scan["summary"]
    coverage = grid_label(scan) + ("; все заданные позы учтены" if scan["complete"] else "")
    qc = {True: "подтверждён в summary", False: "не пройден"}.get(summary.get("geometry_qc_passed"), "не указан")
    return (f"{coverage}. Набор: {scan['directory'].name}; проверено SHA256: {len(scan['verified'])} файлов. Центров: {len(scan['centres']):,}; углов: {len(scan['phi'])}; "
            f"поз: {counts['processed']:,}/{counts['total']:,}. Evaluated: {counts['evaluated']:,}; "
            f"invalid_geometry: {counts['invalid_geometry']:,}; вне области: {counts['outside_search_region']:,}; отсутствуют: {counts['missing']:,}. "
            f"Фиксированные ρ₁={summary['rho1']:.14g}, ρ₂={summary['rho2']:.14g} Ом·м. "
            f"Геометрический QC: {qc}. Статус: numerical_candidate_not_validated. "
            "Полнота перебора не означает физическую или экспериментальную валидацию модели.")


def encoded(array, dtype):
    return base64.b64encode(np.asarray(array, dtype=dtype).tobytes(order="C")).decode("ascii")


def write_html(scan, path):
    from plotly.offline import get_plotlyjs

    centres = scan["centres"]
    payload = dict(u=scan["u"].tolist(), v=scan["v"].tolist(), phi=scan["phi"].tolist(),
                   centre_ids=centres.centre_index.tolist(), n=len(centres),
                   rmse=encoded(scan["rmse"], "<f4"), status=encoded(scan["status"], "u1"),
                   best_angle_index=scan["best_angle_index"].tolist(),
                   xyz=encoded(centres[["x_mm", "y_mm", "z_mm"]], "<f8"),
                   h=encoded(centres.h_vertex_mm, "<f8"), nodes=encoded(scan["nodes"], "<f8"),
                   faces=encoded(scan["faces"], "<u4"), colours=COLOURS,
                   limits=scan["colour_range"], complete=scan["complete"], streaming_active=scan["summary"].get("streaming_active",False))
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    best = scan["best"]
    best_text = (f"Минимум сохранённых RMSE: {best['rmse_ohm']:.6g} Ом; "
                 f"центр {int(best['centre_index'])}, φ={best['phi_deg']:g}°."
                 if best else "Допустимых рассчитанных поз нет.")
    document = HTML_TEMPLATE.replace("__PLOTLY_JS__", get_plotlyjs())
    document = document.replace("__DATA__", data).replace("__STATUS__", html.escape(status_text(scan)))
    title = grid_label(scan) + ": близость CEM к эксперименту"
    document = document.replace("__BEST__", html.escape(best_text)).replace("__TITLE__", html.escape(title))
    Path(path).write_text(document, encoding="utf-8")


def write_png(scan, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize

    values, angles, state = best_layer(scan)
    shape = (len(scan["v"]), len(scan["u"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    extent = [scan["u"][0] - .5, scan["u"][-1] + .5, scan["v"][0] - .5, scan["v"][-1] + .5]
    for ax in axes:
        ax.imshow(state.reshape(shape), origin="lower", extent=extent, interpolation="nearest",
                  cmap=ListedColormap([MISSING_COLOUR, "white", INVALID_COLOUR, OUTSIDE_COLOUR]), vmin=0, vmax=3)
        ax.set(xlabel="u, мм", ylabel="v, мм")
    cmap = LinearSegmentedColormap.from_list("closer_experiment", [c[1] for c in COLOURS])
    im = axes[0].imshow(np.ma.masked_invalid(values.reshape(shape)), origin="lower", extent=extent,
                        interpolation="nearest", cmap=cmap, norm=Normalize(*scan["colour_range"]))
    fig.colorbar(im, ax=axes[0], label="RMSE к эксперименту, Ом (все 9 размеров)", shrink=.8)
    axes[0].set_title("Минимальный сохранённый RMSE в каждом центре")
    im = axes[1].imshow(np.ma.masked_invalid(angles.reshape(shape)), origin="lower", extent=extent,
                        interpolation="nearest", cmap="twilight", vmin=scan["phi"][0], vmax=scan["phi"][-1])
    fig.colorbar(im, ax=axes[1], label="φ, ° (при равенстве — первый угол)", shrink=.8)
    axes[1].set_title("Угол минимального RMSE в каждом центре")
    label = grid_label(scan)
    fig.suptitle(f"{label} · фиксированные ρ · модель не валидирована\n"
                 "Серый: нет допустимой геометрии; фиолетовый: вне области; тёмный: не рассчитано")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_report(scan, output_directory, *, notebook=False):
    destination = Path(output_directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    paths = [destination / "full_scan.html", destination / "full_scan.png"]
    require(not any(str(p) in scan["verified"] for p in paths), "Cannot overwrite a contracted input")
    write_html(scan, paths[0])
    write_png(scan, paths[1])
    if notebook:
        # No second copy of the multi-megabyte scan; the local viewer uses siblings.
        nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": [
                {"cell_type": "markdown", "metadata": {}, "id": "scan-status", "source": [
                    "# " + grid_label(scan) + "\n", status_text(scan),
                    "\n\n![Карта](full_scan.png)\n\n[Открыть интерактивный отчёт](full_scan.html)"]},
                {"cell_type": "code", "metadata": {}, "id": "local-viewer", "execution_count": None,
                 "outputs": [], "source": ["from IPython.display import IFrame, display\n",
                     "display(IFrame('full_scan.html', width='100%', height=1100))"]}]}
        nb_path = destination / "full_scan.ipynb"
        require(str(nb_path) not in scan["verified"], "Cannot overwrite a contracted input")
        nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
        paths.append(nb_path)
    return paths


HTML_TEMPLATE = r'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>body{font:16px system-ui,sans-serif;margin:20px;color:#20252a;background:#fff}
h1{font-size:25px}p{max-width:1250px;line-height:1.5}select,button{font:inherit;padding:6px;margin:5px}
.views{display:grid;grid-template-columns:1fr 1fr;gap:12px}.plot{height:570px;min-width:0;border:1px solid #ddd}
#map{height:670px}#status{padding:12px;background:#f1f4f7}.error{color:#b91c1c}
@media(max-width:900px){.views{grid-template-columns:1fr}}</style>
<script>__PLOTLY_JS__</script></head><body>
<h1>__TITLE__</h1>
<p id="status">__STATUS__</p><p>__BEST__</p>
<p>Зелёный означает меньший <b>RMSE в омах</b> относительно девяти экспериментальных импедансов.
Шкала общая для всех углов. Использованы исходные RMSE без априорных штрафов; FEM здесь не исполняется.
Серый — недопустимая геометрия; фиолетовый — электроды выходят на руки или искусственные срезы; тёмный — отсутствующая оценка. Пропуски не интерполируются.
XYZ и u,v заданы в мм; φ — в градусах. Геометрия показана без сокращения поверхности.</p>
<p>Минимум по углам выбирается только среди сохранённых допустимых оценок; при равенстве показан первый угол.
Для глобального минимума при равенстве RMSE выбран меньший centre_index, затем меньший угол.
В неполном снимке этот минимум предварительный. Фиксированные ρ — параметры подогнанной модели,
а не независимо измеренные свойства тканей. Карта не является оценкой точности установки электродов.
h_vertex — расстояние до вершины поверхности лёгкого по принятому алгоритму, не тканевая толщина
и не эффективное h двуслойной модели; путь может включать кость.</p>
<label for="angle">Слой карты:</label><select id="angle"><option value="best">Лучший угол в каждом центре</option></select>
<button id="reset">Сбросить ракурсы</button><span id="layer-status" aria-live="polite"></span>
<div class="views"><div><h2>Вся поверхность тела</h2><div id="body" class="plot"></div></div>
<div><h2>Область центров: автоматический масштаб</h2><div id="local" class="plot"></div></div></div>
<h2>Матрица заданных u × v</h2><div id="map" class="plot"></div>
<p>HTML содержит локальные данные и библиотеку Plotly. Передача данных на сервер не требуется.
Матрица RMSE хранится как float32 только для отображения; проверка CSV выполнена в float64.</p>
<script id="scan-data" type="application/json">__DATA__</script><script>
'use strict';
const D=JSON.parse(document.getElementById('scan-data').textContent);
function decode(text,Type){const s=atob(text),bytes=new Uint8Array(s.length);for(let i=0;i<s.length;i++)bytes[i]=s.charCodeAt(i);return new Type(bytes.buffer);}
const R=decode(D.rmse,Float32Array), S=decode(D.status,Uint8Array), xyz=decode(D.xyz,Float64Array), H=decode(D.h,Float64Array);
const nodes=decode(D.nodes,Float64Array), faces=decode(D.faces,Uint32Array), A=D.phi.length;
const axis=(data,k)=>Array.from({length:data.length/3},(_,i)=>data[i*3+k]);
const mesh={type:'mesh3d',x:axis(nodes,0),y:axis(nodes,1),z:axis(nodes,2),i:axis(faces,0),j:axis(faces,1),k:axis(faces,2),
 color:'#d0d3d6',opacity:.30,hoverinfo:'skip',showscale:false,name:'Расчётная поверхность',showlegend:false};
const select=document.getElementById('angle');D.phi.forEach((p,i)=>select.add(new Option('φ = '+p+'°',String(i))));
const bestIndex=D.best_angle_index,bestState=new Uint8Array(D.n),validCount=new Uint8Array(D.n),missingCount=new Uint8Array(D.n),outsideCount=new Uint8Array(D.n);
for(let c=0;c<D.n;c++){for(let a=0;a<A;a++){const j=c*A+a;if(S[j]===1)validCount[c]++;else if(S[j]===0)missingCount[c]++;else if(S[j]===3)outsideCount[c]++;}bestState[c]=bestIndex[c]>=0?1:(missingCount[c]>0?0:outsideCount[c]===A?3:2);}
function layer(value){
 const groups=Array.from({length:4},()=>({x:[],y:[],z:[],text:[],color:[]}));
 const z=Array.from({length:D.v.length},()=>Array(D.u.length).fill(null));
 const background=Array.from({length:D.v.length},()=>Array(D.u.length).fill(null));
 const hover=Array.from({length:D.v.length},()=>Array(D.u.length));
 for(let c=0;c<D.n;c++){const a=value==='best'?bestIndex[c]:Number(value), state=value==='best'?bestState[c]:S[c*A+a];
  const r=state===1?R[c*A+a]:null, row=Math.floor(c/D.u.length),col=c%D.u.length;
  let text='Центр '+D.centre_ids[c]+'<br>u='+D.u[col]+'; v='+D.v[row]+' мм<br>';
  text+=(state===1?'RMSE='+r.toPrecision(6)+' Ом; φ='+D.phi[a]+'°':state===2?'Нет допустимой геометрии':state===3?'Вне разрешённой поверхности грудной стенки':'Оценка отсутствует');
  text+='<br>XYZ=('+[xyz[3*c],xyz[3*c+1],xyz[3*c+2]].map(x=>x.toFixed(3)).join(', ')+') мм';
  text+='<br>h_vertex='+ (Number.isFinite(H[c])?H[c].toFixed(3)+' мм':'недоступно');
  text+='<br>Углов: evaluated '+validCount[c]+', invalid '+(A-validCount[c]-missingCount[c]-outsideCount[c])+', вне области '+outsideCount[c]+', missing '+missingCount[c];
  const g=groups[state];g.x.push(xyz[3*c]);g.y.push(xyz[3*c+1]);g.z.push(xyz[3*c+2]);g.text.push(text);if(state===1)g.color.push(r);
  z[row][col]=r;background[row][col]=state===1?null:state;hover[row][col]=text;
 }
 return {groups,z,background,hover};
}
function trace(group,state){return {type:'scatter3d',mode:'markers',x:group.x,y:group.y,z:group.z,text:group.text,hovertemplate:'%{text}<extra></extra>',
 name:state===1?'Рассчитано':state===2?'Геометрия недопустима':state===3?'Вне области':'Не рассчитано',marker:{size:state===1?2.4:1.8,opacity:1,
 color:state===1?group.color:state===2?'#969696':state===3?'#9874b8':'#393939',colorscale:D.colours,cmin:D.limits[0],cmax:D.limits[1],showscale:state===1,
 colorbar:{title:{text:'RMSE, Ом'},thickness:14,len:.6}}};}
function bounds(data){return [0,1,2].map(k=>{let lo=Infinity,hi=-Infinity;for(let i=k;i<data.length;i+=3){lo=Math.min(lo,data[i]);hi=Math.max(hi,data[i]);}const pad=Math.max(5,(hi-lo)*.08);return [lo-pad,hi+pad];});}
function scene(ranges){return {xaxis:{title:{text:'x, мм'},range:ranges[0]},yaxis:{title:{text:'y, мм'},range:ranges[1]},zaxis:{title:{text:'z, мм'},range:ranges[2]},aspectmode:'data',camera:{eye:{x:1.5,y:1.5,z:.8}}};}
const bodyScene=scene(bounds(nodes)),localScene=scene(bounds(xyz));
const config={responsive:true,displaylogo:false,scrollZoom:true,toImageButtonOptions:{format:'png',scale:2}};
const layout=s=>({margin:{l:0,r:0,t:0,b:0},scene:s,uirevision:'keep-camera',legend:{orientation:'h',y:1}});
async function start(){
 const initial=layer('best');
 await Promise.all([Plotly.newPlot('body',[mesh,trace(initial.groups[1],1),trace(initial.groups[2],2),trace(initial.groups[0],0),trace(initial.groups[3],3)],layout(bodyScene),config),
 Plotly.newPlot('local',[mesh,trace(initial.groups[1],1),trace(initial.groups[2],2),trace(initial.groups[0],0),trace(initial.groups[3],3)],layout(localScene),config),
 Plotly.newPlot('map',[
 {type:'heatmap',x:D.u,y:D.v,z:initial.background,zmin:0,zmax:3,colorscale:[[0,'#393939'],[.1666,'#393939'],[.1667,'white'],[.4999,'white'],[.5,'#969696'],[.8333,'#969696'],[.8334,'#9874b8'],[1,'#9874b8']],showscale:false,text:initial.hover,hovertemplate:'%{text}<extra></extra>',zsmooth:false,connectgaps:false},
 {type:'heatmap',x:D.u,y:D.v,z:initial.z,zmin:D.limits[0],zmax:D.limits[1],colorscale:D.colours,colorbar:{title:{text:'RMSE, Ом'}},text:initial.hover,hovertemplate:'%{text}<extra></extra>',hoverongaps:false,zsmooth:false,connectgaps:false}],
 {margin:{l:65,r:65,t:10,b:60},xaxis:{title:{text:'u, мм'}},yaxis:{title:{text:'v, мм'},scaleanchor:'x'},uirevision:'grid'},config)]);
 updateCount(initial);
}
function updateCount(l){document.getElementById('layer-status').textContent='На слое: '+l.groups[1].x.length+' рассчитано; '+l.groups[2].x.length+' недопустимо; '+l.groups[3].x.length+' вне области; '+l.groups[0].x.length+' отсутствует';}
select.disabled=true;
start().then(()=>{select.disabled=false;}).catch(showError);
function showError(error){document.getElementById('layer-status').className='error';document.getElementById('layer-status').textContent=String(error);}
select.addEventListener('change',async()=>{select.disabled=true;try{const l=layer(select.value),order=[1,2,0,3];
 await Promise.all(['body','local'].map(id=>Plotly.restyle(id,{x:order.map(s=>l.groups[s].x),y:order.map(s=>l.groups[s].y),z:order.map(s=>l.groups[s].z),text:order.map(s=>l.groups[s].text),
 'marker.color':[l.groups[1].color,'#969696','#393939','#9874b8']},[1,2,3,4])).concat([
 Plotly.restyle('map',{z:[l.background,l.z],text:[l.hover,l.hover]},[0,1])]));updateCount(l);
 }catch(e){showError(e);}finally{select.disabled=false;}});
document.getElementById('reset').addEventListener('click',()=>{Plotly.relayout('body',{scene:bodyScene});Plotly.relayout('local',{scene:localScene});});
if(D.streaming_active&&!D.complete){setTimeout(()=>location.reload(),60000);document.getElementById('status').textContent+=' Карта обновляется по завершённым блокам; страница перечитывается каждые 60 секунд.';}
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, help="Default: INPUT/report (source files are read-only)")
    parser.add_argument("--allow-partial", action="store_true", help="Label a consistent incomplete snapshot explicitly")
    parser.add_argument("--notebook", action="store_true", help="Also write a small local notebook viewer")
    parser.add_argument("--test-grid", action="store_true", help="Explicitly label a bounded verification grid")
    args = parser.parse_args()
    scan = load_scan(args.input_dir, allow_partial=args.allow_partial, enforce_production_grid=not args.test_grid)
    paths = build_report(scan, args.output_dir or args.input_dir / "report", notebook=args.notebook)
    print(status_text(scan))
    print("\n".join(map(str, paths)))


if __name__ == "__main__":
    main()
