"""Сводка готовых геометрических расчётов целого сердца Ника: без подгонки и FEM.

CLI (пути задаёт вызывающий агент; автоматически расчёты не запускаются)::

  python summarize_nik_whole_heart_geometry.py --baseline inputs/geometry_comparison.jsonl \
    --results project_reference project_sphere_overlap project_surface \
              project_ellipsoid_overlap_retry library/scored \
    --history-results project_ellipsoid_overlap \
    --comparison-results refine_ellipsoid_overlap refine_sphere_overlap refine_surface \
    --surface-scores surface_scores.local.json --output summary

Основной набор обязан содержать 29*12 записей, включая строки отказов. Оба
R–R анализируются раздельно. Все 12 фаз описываются; главные парные сводки
используют 10 неопорных фаз. T0 — контроль, I на опоре — самосравнение.

Входы --results (можно повторять): каталоги непосредственных per-fit *.json,
JSON с rows либо JSONL. Агрегат results.json не дублирует per-fit файлы.
Библиотечные записи должны быть приведены к run_heart_method_comparison schema.
Допустимы дополнительные поля constraints={volume,centroid}: constraint/estimated/
unknown и convergence={selected_solution: converged/closed_form/unknown/failed,
all_starts_converged: bool|null}. Отсутствие сходимости не скрывается наличием
метрик. Неизвестная сходимость исключает метод из Парето, но не из описания.

Supplement --surface-scores: {rows:[{subject,phase_id,method_id,status,
rms_32768_mm,p95_32768_mm,rms_sampling_difference_mm,samples:{...}}]}.
Проверяются cycle_index и target_mask_sha256, если они переданы. Эти расстояния
односторонние, от одинаковых площадочных точек целевой поверхности к форме;
p95 не является расстоянием Хаусдорфа. Незаданный хэш привязки фиксируется.
Supplement не применяется к history/comparison: там другие параметры форм.

Результаты: JSON/CSV полного массива, цикловых сводок, Парето, отказов,
вариантов и истории; русский Markdown с таблицами в details и matplotlib PNG.
--check-inputs-only проверяет готовность, ничего не пишет. --allow-incomplete
разрешает лишь явно помеченную диагностическую сводку; по умолчанию выключен.
Папка output должна быть пустой, чтобы не смешивать разные редакции.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

# Используется существующее определение главных осей, без новой аппроксимации.
from heart_reduced_geometry import _eigenframe


# Порядок совпадает с summarize_old_approximation_sweep и реестром runner.
METHODS = ("T0 I1 I2 I3 G1 G2 G3a G3b G3c G4a G4b G5 "
           "L1 L2 L3a L3b L3c L3d L4a L4b M1 M2 M3 M4 M5 M6 M7 M8 M9").split()
V_FIXED = set("T0 I1 I2 I3 G1 G3c G4a L1 L3c L3d L4a".split())
C_FIXED = set("T0 I1 I2 I3 G1 G2 G3a L1 L2 L3a L3c".split())
ANALYTIC = set("T0 I1 I2 I3 G1 G2 G3a G5 L1 L2".split())
ACCEPTED = {"closed_form", "algebraic_valid", "converged", "directional_stationary"}
FAILED = {"failed", "error", "unsupported", "solver_not_converged", "search_budget_exhausted"}
META_FILES = {"protocol.json", "status.json", "run_status.json", "library_clouds.json"}
SURFACE_RULE = "target_area_points_32768_one_sided_triangle_or_analytic_distance"


def number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def vector(value):
    try:
        a = np.asarray(value, float)
    except (ValueError, TypeError):
        return None
    return a if a.shape == (3,) and np.isfinite(a).all() else None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, hashes):
    path = Path(path).resolve()
    data = path.read_bytes()
    hashes[str(path)] = hashlib.sha256(data).hexdigest()
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in data.decode("utf-8-sig").splitlines() if line.strip()]
    return json.loads(data.decode("utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False, allow_nan=False)
                         if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows)


def phase_key(row):
    cycle = number(row.get("cycle_index"))
    if cycle is None or cycle != int(cycle):
        raise ValueError("Нет целого cycle_index: " + str(row.get("phase_id")))
    return str(row["subject"]), int(cycle), str(row["phase_id"])


def result_key(row):
    return (*phase_key(row), str(row["method_id"]))


def load_baseline(path, hashes):
    path = Path(path)
    if path.is_dir():
        path /= "geometry_comparison.jsonl"
    data = read(path, hashes)
    data = data.get("rows", []) if isinstance(data, dict) else data
    rows = [r for r in data if r.get("region") == "whole_heart"]
    if len(rows) != 12 or len({r["subject"] for r in rows}) != 1:
        raise ValueError("Baseline должен содержать 12 фаз whole_heart одного участника")
    result, groups = {}, defaultdict(list)
    for raw in rows:
        key = phase_key(raw)
        if key in result:
            raise ValueError(f"Повтор фазы baseline: {key}")
        row = dict(raw)
        timing = raw.get("timing") or {}
        percent = number(raw.get("phase_percent_within_cycle", timing.get("phase_percent_within_cycle")))
        if percent is None:
            percent = number(raw.get("rr_percent", timing.get("rr_percent")))
        if percent is None or not 0 <= percent < 100:
            raise ValueError(f"Нет корректного процента фазы внутри R–R: {key}")
        row["phase_percent"] = percent
        row["time_ms"] = number(raw.get("derived_time_from_R0_ms", timing.get("derived_time_from_R0_ms")))
        row["rr_interval_ms"] = number(raw.get("rr_interval_ms", timing.get("rr_interval_ms")))
        row["temporal_resolution_ms"] = number(raw.get("temporal_resolution_ms", timing.get("temporal_resolution_ms")))
        m = row.get("moments") or {}
        if number(m.get("volume_ml")) is None or m["volume_ml"] <= 0 or vector(m.get("centroid_mm")) is None:
            raise ValueError(f"Некорректные моменты baseline: {key}")
        if not isinstance(row.get("is_reference_phase"), bool) or not row.get("provenance", {}).get("sha256"):
            raise ValueError(f"Нет явной опоры или хэша маски: {key}")
        result[key] = row
        groups[key[:2]].append(row)
    if len(groups) != 2:
        raise ValueError("Ожидаются два раздельных R–R")
    for group, values in groups.items():
        values.sort(key=lambda r: r["phase_percent"])
        refs = [r for r in values if r["is_reference_phase"]]
        if len(refs) != 1 or any(r["reference_phase_id"] != refs[0]["phase_id"] for r in values):
            raise ValueError(f"Несогласованная опора R–R: {group}")
        if len({r["phase_percent"] for r in values}) != len(values):
            raise ValueError(f"Неоднозначное время фаз: {group}")
        times = [r["time_ms"] for r in values]
        if all(t is not None for t in times) and any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError(f"Порядок процентов противоречит времени: {group}")
    return result, groups


def unpack_records(data):
    if isinstance(data, dict) and "method_id" in data:
        return [data]
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        return data["rows"]
    if isinstance(data, list):
        return data
    return None


def load_results(paths, baseline, hashes, allow_incomplete=False):
    records, manifests = {}, []
    for supplied in paths:
        path = Path(supplied).resolve()
        manifest = {"path": str(path), "status_file": None, "protocol": None}
        if path.is_dir():
            for filename in ("status.json", "run_status.json"):
                if (path / filename).exists():
                    status = read(path / filename, hashes)
                    manifest["status_file"] = status
                    if status.get("status") != "completed" and not allow_incomplete:
                        raise ValueError(f"Расчёт ещё не завершён: {path / filename}")
            if (path / "protocol.json").exists():
                manifest["protocol"] = read(path / "protocol.json", hashes)
            files = sorted(p for p in path.glob("*.json") if p.name not in META_FILES and p.name != "results.json")
            parsed = [(p, read(p, hashes)) for p in files]
            selected = [(p, d) for p, d in parsed if unpack_records(d) is not None]
            if not selected and (path / "results.json").exists():
                selected = [(path / "results.json", read(path / "results.json", hashes))]
            manifest["ignored_nonrecord_files"] = [p.name for p, d in parsed if unpack_records(d) is None]
        else:
            selected = [(path, read(path, hashes))]
        count = 0
        for source, data in selected:
            rows = unpack_records(data)
            if rows is None:
                raise ValueError(f"Неподдерживаемый формат результатов: {source}")
            for raw in rows:
                key = result_key(raw)
                if key in records:
                    raise ValueError(f"Два результата {key}; выбирайте целый прогон явно, прежний — --history-results")
                if key[:3] not in baseline:
                    raise ValueError(f"Фаза вне baseline: {key}")
                target = baseline[key[:3]]
                for field, expected in (("target_mask_sha256", target["provenance"]["sha256"]),
                                        ("reference_phase_id", target["reference_phase_id"]),
                                        ("is_reference_phase", target["is_reference_phase"])):
                    if raw.get(field) != expected:
                        raise ValueError(f"Не совпадает {field}: {key}")
                if raw.get("region", "whole_heart") != "whole_heart":
                    raise ValueError(f"Неверная область: {key}")
                v = number(raw.get("target_volume_ml"))
                if v is None or not np.isclose(v, target["moments"]["volume_ml"], rtol=1e-8, atol=1e-6):
                    raise ValueError(f"Объём целевой маски не совпадает: {key}")
                records[key] = dict(raw, source_path=str(source), source_root=str(path))
                count += 1
        if not count and not allow_incomplete:
            raise ValueError(f"В каталоге нет записей project schema: {path}")
        manifest["record_count"] = count
        manifests.append(manifest)
    return records, manifests


def convergence(raw):
    """Выбранное решение и все старты — разные показатели, статусы не сливаются."""
    params, fit = raw.get("parameters") or {}, raw.get("fit") or {}
    opt = fit.get("optimization") or params.get("optimization") or {}
    diag = params.get("diagnostics") or fit.get("diagnostics") or {}
    trials = opt.get("trials") or diag.get("trials") or []
    flags = [t.get("success", t.get("last_step_stationary")) for t in trials]
    flags = [v if isinstance(v, bool) else None for v in flags]
    selected = "unknown"
    statuses = [raw.get("status"), params.get("status"), fit.get("status"), opt.get("status")]
    if any(s in FAILED for s in statuses):
        selected = "failed"
    elif raw["method_id"] in {f"M{i}" for i in range(1, 10)} and raw.get("library_valid") is True:
        # Проверенные адаптеры вызывают алгебраическую подгонку. Геометрическая
        # допустимость не называется сходимостью всех стартов или устойчивостью.
        selected = "algebraic_valid"
    elif raw["method_id"] in ANALYTIC and raw.get("metrics"):
        selected = "closed_form"
    elif "discrete_directional_search_stationary" in statuses:
        selected = "directional_stationary"
    elif "local_solver_converged" in statuses or opt.get("success") is True:
        selected = "converged"
    elif opt.get("success") is False:
        selected = "failed"
    explicit = raw.get("convergence") or {}
    if selected == "unknown" and explicit.get("selected_solution") in ACCEPTED | {"unknown", "failed"}:
        selected = explicit["selected_solution"]
    all_success = all(flags) if flags and all(v is not None for v in flags) else None
    if all_success is None and isinstance(explicit.get("all_starts_converged"), bool):
        all_success = explicit["all_starts_converged"]
    selected_index = opt.get("selected_start", diag.get("selected_trial"))
    selected_flag = flags[selected_index] if isinstance(selected_index, int) and 0 <= selected_index < len(flags) else None
    if selected_flag is False:
        selected = "failed"
    return {"selected_solution": selected, "all_starts_converged": all_success,
            "selected_start_index": selected_index, "selected_start_success": selected_flag,
            "starts_count": len(flags), "starts_successful": sum(v is True for v in flags),
            "starts_unknown": sum(v is None for v in flags), "source_statuses": statuses}


def constraints(raw):
    method = raw["method_id"]
    # Библиотечные M1–M9 — свободная подгонка целевого облака, по аудиту реестра.
    v = "constraint" if method in V_FIXED else "estimated" if method in METHODS else "unknown"
    c = "constraint" if method in C_FIXED else "estimated" if method in METHODS else "unknown"
    declared = raw.get("constraints") or {}
    for name, value in (("volume", v), ("centroid", c)):
        if name in declared and declared[name] not in {"constraint", "estimated", "unknown"}:
            raise ValueError(f"Непонятный статус ограничения {name}: {declared[name]}")
        if method in METHODS and name in declared and declared[name] != value:
            raise ValueError(f"Ограничения {method}/{name} противоречат проверенному реестру")
    return declared.get("volume", v), declared.get("centroid", c)


def axes_info(parameters):
    """Полуоси формы либо моментные полуоси; без чтения/подгонки поверхности."""
    a = vector(parameters.get("semi_axes_mm"))
    frame = np.asarray(parameters.get("axes_matrix", []), float)
    kind = "shape"
    if a is None or frame.shape != (3, 3):
        cov = np.asarray(parameters.get("covariance_mm2", []), float)
        if cov.shape != (3, 3) or not np.isfinite(cov).all():
            return None
        eigen, frame, degenerate = _eigenframe(cov, 1e-6)
        a, kind = np.sqrt(5 * eigen), "moment_equivalent"
    else:
        order = np.argsort(a)[::-1]
        a, frame = a[order], frame[:, order]
        degenerate = bool(parameters.get("orientation_degenerate", False))
    if np.any(a <= 0) or not np.isfinite(frame).all() or not np.allclose(frame.T @ frame, np.eye(3), atol=1e-5):
        return None
    degenerate |= any(abs(a[i] - a[j]) <= 1e-6 * max(a) for i, j in ((0, 1), (0, 2), (1, 2)))
    if np.linalg.det(frame) < 0:
        frame[:, -1] *= -1
    return {"semi_axes_mm": a.tolist(), "axes_matrix": frame.tolist(), "source": kind,
            "orientation_degenerate": bool(degenerate)}


def angle_deg(left, right):
    if left is None or right is None or left["orientation_degenerate"] or right["orientation_degenerate"]:
        return None
    a, b = np.array(left["axes_matrix"]), np.array(right["axes_matrix"])
    # Оси без направления: знак собственного вектора не является поворотом сердца.
    return min(float(np.degrees(np.arccos(np.clip((np.trace(a.T @ (b * signs)) - 1) / 2, -1, 1))))
               for signs in product((-1, 1), repeat=3) if np.prod(signs) == 1)


def load_surface(path, baseline, records, hashes):
    if path is None:
        return {}, {"provided": False}
    data = read(path, hashes)
    rows = unpack_records(data)
    if rows is None:
        raise ValueError("Surface supplement должен содержать rows")
    lookup, unbound = {}, []
    # Общий scorer хранит хэши всех 348 форм на верхнем уровне.
    # Его относительные пути заданы от корня репозитория, как и у runner.
    repo = Path(__file__).resolve().parents[2]
    source_hashes = data.get("source_hashes", {}) if isinstance(data, dict) else {}
    bound_hashes = {str((Path(p) if Path(p).is_absolute() else repo / p).resolve()): sha
                    for p, sha in source_hashes.items()}
    for row in rows:
        keys = [k for k in records if (k[0], k[2], k[3]) == (row.get("subject"), row.get("phase_id"), row.get("method_id"))]
        if len(keys) != 1:
            raise ValueError(f"Неоднозначная привязка поверхности: {row.get('phase_id')}/{row.get('method_id')}")
        key = keys[0]
        if key in lookup:
            raise ValueError(f"Повтор поверхностной оценки: {key}")
        if "cycle_index" in row and row["cycle_index"] != key[1]:
            raise ValueError(f"Неверный R–R поверхности: {key}")
        if "target_mask_sha256" in row and row["target_mask_sha256"] != baseline[key[:3]]["provenance"]["sha256"]:
            raise ValueError(f"Несовпадение маски поверхности: {key}")
        source_hash = row.get("result_sha256") or row.get("source_result_sha256") or bound_hashes.get(records[key]["source_path"])
        if source_hash and source_hash != digest(records[key]["source_path"]):
            raise ValueError(f"Поверхностная оценка относится к другой редакции формы: {key}")
        if not source_hash:
            unbound.append(list(key))
        lookup[key] = row
    return lookup, {"provided": True, "record_count": len(lookup),
                    "records_without_result_hash": unbound, "rule": SURFACE_RULE,
                    "metadata": {k: v for k, v in data.items() if k != "rows"} if isinstance(data, dict) else {}}


def normalize(raw, base, supplement=None):
    p, m = raw.get("parameters") or {}, raw.get("metrics") or {}
    control = raw["method_id"] == "T0"
    v = number(p.get("volume_ml"))
    if v is None and number(p.get("volume_mm3")) is not None:
        v = float(p["volume_mm3"]) / 1000
    center = vector(p.get("center_mm"))
    if control:
        v, center = base["moments"]["volume_ml"], vector(base["moments"]["centroid_mm"])
    target_v, target_c = float(base["moments"]["volume_ml"]), vector(base["moments"]["centroid_mm"])
    error = v - target_v if v is not None else None
    numeric = bool(raw.get("metrics")) and v is not None and v > 0 and center is not None
    conv = convergence(raw)
    vc, cc = constraints(raw)
    shape = axes_info(base["raw_moment_ellipsoid"] if control else p)
    target_shape = axes_info(base.get("raw_moment_ellipsoid") or base["moments"])
    target_cov = np.asarray(base["moments"]["covariance_mm2"], float)
    shape_cov = np.asarray(base["moments"]["covariance_mm2"] if control else p.get("covariance_mm2", []), float)
    tensor_error = (float(np.linalg.norm(shape_cov-target_cov, ord="fro"))
                    if shape_cov.shape == (3, 3) and np.isfinite(shape_cov).all() else None)
    row = {k: raw[k] for k in ("subject", "phase_id", "cycle_index", "method_id", "reference_phase_id", "is_reference_phase", "target_mask_sha256")}
    row.update(source_path=raw["source_path"], status=raw.get("status"), reason=raw.get("reason", (raw.get("fit") or {}).get("reason")),
               library_status=raw.get("library_status"), library_warning=raw.get("library_warning"),
               library_valid=raw.get("library_valid"), method_name=raw.get("method_name"),
               equivalence_note="M7 xyz — сфера; численно совпадает с M9 в данном прогоне" if raw["method_id"] in {"M7", "M9"} else None,
               control=control, self_reference=raw["method_id"].startswith("I") and raw["is_reference_phase"],
               main_phase=not raw["is_reference_phase"], numeric_available=numeric,
               optimization_acceptable=conv["selected_solution"] in ACCEPTED,
               **conv, volume_status=vc, centroid_status=cc,
               phase_percent=base["phase_percent"], time_ms=base["time_ms"], rr_interval_ms=base["rr_interval_ms"],
               temporal_resolution_ms=base["temporal_resolution_ms"],
               target_touches_CT_boundary=base["provenance"].get("touches_source_image_boundary"),
               volume_ml=v, target_volume_ml=target_v, volume_error_ml=error,
               volume_error_percent=None if error is None else 100 * error / target_v,
               center_mm=None if center is None else center.tolist(), target_center_mm=target_c.tolist(),
               center_error_mm=None if center is None else float(np.linalg.norm(center - target_c)),
               dice=number(m.get("dice")),
               sampled_volume_ml=number((m.get("overlap") or {}).get("candidate_sampled_volume_ml")),
               original_surface_rms_mm=number(m.get("target_surface_to_candidate_RMS_mm")),
               original_surface_rule=raw.get("surface_metric_rule", m.get("surface_metric_limitation")),
               shape_axes=shape, target_moment_axes=target_shape,
               orientation_error_deg=angle_deg(shape, target_shape),
               covariance_error_fro_mm2=tensor_error,
               covariance_relative_error=None if tensor_error is None else tensor_error/float(np.linalg.norm(target_cov, ord="fro")),
               common_surface_rms_mm=None, common_surface_p95_mm=None, surface_sampling_difference_mm=None,
               surface_status="not_provided", surface_rule=None)
    warnings = []
    if row["dice"] is not None and not 0 <= row["dice"] <= 1:
        raise ValueError(f"Dice вне [0,1]: {result_key(raw)}")
    if numeric and number(m.get("volume_error_percent")) is not None and not np.isclose(row["volume_error_percent"], m["volume_error_percent"], atol=1e-6):
        raise ValueError(f"Ошибка объёма не соответствует параметрам: {result_key(raw)}")
    if numeric and number(m.get("center_shift_mm")) is not None and not np.isclose(row["center_error_mm"], m["center_shift_mm"], atol=1e-5):
        raise ValueError(f"Ошибка центра не соответствует параметрам: {result_key(raw)}")
    if vc == "constraint" and error is not None and abs(error) > 1e-5:
        warnings.append("Нарушено заявленное равенство непрерывного объёма")
    if cc == "constraint" and row["center_error_mm"] is not None and row["center_error_mm"] > 1e-5:
        warnings.append("Нарушено заявленное равенство центроида")
    if supplement is not None:
        row["surface_status"] = supplement.get("status", "unknown")
        row["surface_rule"] = SURFACE_RULE
        if supplement.get("status") in {"completed", "computed", "ok", "success", "reference"}:
            for source, dest in (("rms_32768_mm", "common_surface_rms_mm"), ("p95_32768_mm", "common_surface_p95_mm"),
                                 ("rms_sampling_difference_mm", "surface_sampling_difference_mm")):
                row[dest] = number(supplement.get(source))
            if any(row[k] is not None and row[k] < 0 for k in ("common_surface_rms_mm", "common_surface_p95_mm")):
                raise ValueError(f"Отрицательное расстояние поверхности: {result_key(raw)}")
    row["warnings"] = warnings
    for index in range(3):
        row[f"semi_axis_{index+1}_mm"] = shape["semi_axes_mm"][index] if shape else None
        row[f"target_moment_semi_axis_{index+1}_mm"] = target_shape["semi_axes_mm"][index] if target_shape else None
    if row["reason"] is None and conv["selected_solution"] == "unknown":
        row["reason"] = "Нет явного статуса выбранного решения; наличие metrics не подтверждает сходимость"
    return row


def add_dynamics(rows, baseline_groups):
    by = {(r["subject"], r["cycle_index"], r["phase_id"], r["method_id"]): r for r in rows}
    methods = sorted({r["method_id"] for r in rows})
    for group, bases in baseline_groups.items():
        for method in methods:
            anchor_base = next(b for b in bases if b["is_reference_phase"])
            anchor = by.get((*group, anchor_base["phase_id"], method))
            for index, b in enumerate(bases):
                row = by.get((*group, b["phase_id"], method))
                if row is None:
                    continue
                previous = by.get((*group, bases[index - 1]["phase_id"], method)) if index else None
                row["previous_phase_id"] = bases[index - 1]["phase_id"] if index else None
                for prefix, other in (("anchor", anchor), ("step", previous)):
                    for field in ("volume_change_ml", "target_volume_change_ml", "volume_change_error_ml", "center_delta_mm",
                                  "target_center_delta_mm", "displacement_error_mm", "path_segment_mm", "target_path_segment_mm",
                                  "direction_error_deg", "elapsed_ms", "speed_mm_per_s", "target_speed_mm_per_s", "orientation_change_deg"):
                        row[prefix + "_" + field] = None
                    if other is None:
                        continue
                    dt = row["time_ms"] - other["time_ms"] if row["time_ms"] is not None and other["time_ms"] is not None else None
                    row[prefix + "_elapsed_ms"] = dt
                    if row["volume_ml"] is not None and other["volume_ml"] is not None:
                        dv = row["volume_ml"] - other["volume_ml"]
                        dtv = row["target_volume_ml"] - other["target_volume_ml"]
                        row.update({prefix + "_volume_change_ml": dv, prefix + "_target_volume_change_ml": dtv,
                                    prefix + "_volume_change_error_ml": dv - dtv})
                    c, oc = vector(row["center_mm"]), vector(other["center_mm"])
                    if c is not None and oc is not None:
                        dc = c - oc
                        dtc = np.array(row["target_center_mm"]) - other["target_center_mm"]
                        distance, tdistance = float(np.linalg.norm(dc)), float(np.linalg.norm(dtc))
                        row.update({prefix + "_center_delta_mm": dc.tolist(), prefix + "_target_center_delta_mm": dtc.tolist(),
                                    prefix + "_displacement_error_mm": float(np.linalg.norm(dc - dtc)),
                                    prefix + "_path_segment_mm": distance, prefix + "_target_path_segment_mm": tdistance})
                        if distance > 1e-8 and tdistance > 1e-8:
                            row[prefix + "_direction_error_deg"] = float(np.degrees(np.arccos(np.clip(dc @ dtc / (distance * tdistance), -1, 1))))
                        if dt is not None and dt > 0:
                            row[prefix + "_speed_mm_per_s"] = distance * 1000 / dt
                            row[prefix + "_target_speed_mm_per_s"] = tdistance * 1000 / dt
                    row[prefix + "_orientation_change_deg"] = angle_deg(row["shape_axes"], other["shape_axes"])
                    for axis in range(1, 4):
                        key = f"semi_axis_{axis}_mm"
                        row[f"{prefix}_semi_axis_{axis}_change_mm"] = (row[key] - other[key]
                            if row[key] is not None and other[key] is not None else None)
                row["anchor_optimization_acceptable"] = bool(anchor and anchor["optimization_acceptable"])
                row["step_optimization_acceptable"] = bool(previous and previous["optimization_acceptable"])


def stats(rows, field):
    values = [float(r[field]) for r in rows if number(r.get(field)) is not None]
    if not values:
        return {"n": 0, "mean": None, "rmse": None, "mae": None, "max_abs": None}
    a = np.array(values)
    return {"n": len(a), "mean": float(a.mean()), "rmse": float(np.sqrt(np.mean(a*a))),
            "mae": float(np.mean(abs(a))), "max_abs": float(max(abs(a)))}


def extrema(rows, field):
    valid = [r for r in rows if number(r.get(field)) is not None]
    if not valid:
        return {"n": 0}
    lo, hi = min(r[field] for r in valid), max(r[field] for r in valid)
    def points(value):
        return [{"phase_id": r["phase_id"], "phase_percent": r["phase_percent"], "time_ms": r["time_ms"]}
                for r in valid if np.isclose(r[field], value, rtol=1e-10, atol=1e-8)]
    return {"n": len(valid), "minimum": lo, "maximum": hi, "range": hi-lo,
            "minimum_phases": points(lo), "maximum_phases": points(hi), "observed_only": True}


def summarize(rows, groups, expected_methods):
    summaries, worst = [], []
    for group, bases in groups.items():
        for method in expected_methods:
            all_rows = [r for r in rows if (r["subject"], r["cycle_index"], r["method_id"]) == (*group, method)]
            main = [r for r in all_rows if r["main_phase"]]
            usable = [r for r in main if r["numeric_available"] and r["optimization_acceptable"]]
            row = dict(subject=group[0], cycle_index=group[1], method_id=method,
                       expected_phases=len(bases), records=len(all_rows),
                       expected_nonreference=sum(not b["is_reference_phase"] for b in bases), nonreference_records=len(main),
                       numeric_phases=sum(r["numeric_available"] for r in all_rows),
                       selected_acceptable_phases=sum(r["optimization_acceptable"] for r in all_rows),
                       selected_unknown_phases=sum(r["selected_solution"] == "unknown" for r in all_rows),
                       selected_failed_phases=sum(r["selected_solution"] == "failed" for r in all_rows),
                       closed_form_phases=sum(r["selected_solution"] == "closed_form" for r in all_rows),
                       algebraic_valid_phases=sum(r["selected_solution"] == "algebraic_valid" for r in all_rows),
                       optimizer_converged_phases=sum(r["selected_solution"] == "converged" for r in all_rows),
                       directional_stationary_phases=sum(r["selected_solution"] == "directional_stationary" for r in all_rows),
                       all_starts_success_phases=sum(r["all_starts_converged"] is True for r in all_rows),
                       nonreference_usable=len(usable),
                       reference_phase_id=next(b["phase_id"] for b in bases if b["is_reference_phase"]),
                       volume_status=all_rows[0]["volume_status"] if all_rows else "unknown",
                       centroid_status=all_rows[0]["centroid_status"] if all_rows else "unknown")
            for field in ("volume_error_ml", "volume_error_percent", "anchor_volume_change_error_ml", "step_volume_change_error_ml",
                          "center_error_mm", "anchor_displacement_error_mm", "step_displacement_error_mm", "dice",
                          "common_surface_rms_mm", "common_surface_p95_mm", "surface_sampling_difference_mm", "orientation_error_deg",
                          "covariance_error_fro_mm2", "covariance_relative_error"):
                row[field] = stats(main, field)
                available = [r for r in main if number(r.get(field)) is not None]
                if available:
                    w = min(available, key=lambda r: r[field]) if field == "dice" else max(available, key=lambda r: abs(r[field]))
                    worst.append(dict(subject=group[0], cycle_index=group[1], method_id=method, metric=field,
                                      phase_id=w["phase_id"], phase_percent=w["phase_percent"], value=w[field],
                                      selected_solution=w["selected_solution"], descriptive_includes_nonconverged=True))
            row["volume_extrema_all_phases"] = extrema(all_rows, "volume_ml")
            row["target_volume_extrema_matched_phases"] = extrema(all_rows, "target_volume_ml")
            row["target_volume_extrema_full_cycle"] = extrema([
                dict(phase_id=b["phase_id"], phase_percent=b["phase_percent"], time_ms=b["time_ms"], volume_ml=b["moments"]["volume_ml"])
                for b in bases], "volume_ml")
            e, t = row["volume_extrema_all_phases"], row["target_volume_extrema_full_cycle"]
            row["observed_volume_range_error_ml"] = e["range"]-t["range"] if e["n"] == len(bases) else None
            for extremum in ("minimum", "maximum"):
                ep, tp = e.get(extremum+"_phases", []), t.get(extremum+"_phases", [])
                row[extremum+"_timing_error_percent_points"] = (ep[0]["phase_percent"]-tp[0]["phase_percent"]
                    if e["n"] == len(bases) and len(ep) == len(tp) == 1 else None)
                row[extremum+"_timing_error_ms"] = (ep[0]["time_ms"]-tp[0]["time_ms"]
                    if e["n"] == len(bases) and len(ep) == len(tp) == 1 and ep[0]["time_ms"] is not None and tp[0]["time_ms"] is not None else None)
            edges = [r for r in all_rows if r.get("previous_phase_id") is not None]
            for key in ("step_path_segment_mm", "step_target_path_segment_mm"):
                val = [r[key] for r in edges if number(r.get(key)) is not None]
                row[key + "_sum"] = sum(val) if val else None
                row[key + "_n"] = len(val)
            row["trajectory_complete"] = row["step_path_segment_mm_n"] == len(bases)-1
            row["trajectory_length_error_mm"] = (row["step_path_segment_mm_sum"]-row["step_target_path_segment_mm_sum"]
                if row["trajectory_complete"] else None)
            row["pareto_candidate"] = (method != "T0" and len(all_rows) == len(bases) and len(usable) == row["expected_nonreference"]
                and all(r["optimization_acceptable"] and r["numeric_available"] and not r["warnings"] for r in all_rows)
                and row["dice"]["n"] == len(main) and row["anchor_displacement_error_mm"]["n"] == len(main)
                and "unknown" not in (row["volume_status"], row["centroid_status"]))
            summaries.append(row)
    return summaries, worst


def pareto_front(summaries, geometry="dice"):
    """Парето внутри одинаковых ограничений; отсутствующая цель не равна нулю."""
    groups, out = defaultdict(list), []
    for row in summaries:
        if row["pareto_candidate"]:
            groups[(row["subject"], row["cycle_index"], row["volume_status"], row["centroid_status"])].append(row)
    for group, rows in groups.items():
        dimensions = ["geometry"]
        if group[2] != "constraint":
            dimensions.append("volume")
        if group[3] != "constraint":
            dimensions.append("displacement")
        candidates = []
        for row in rows:
            surface = row["common_surface_rms_mm"]
            if geometry == "surface" and surface["n"] != row["expected_nonreference"]:
                continue
            values = {"geometry": 1-row["dice"]["mean"] if geometry == "dice" else surface["mean"],
                      "volume": row["volume_error_ml"]["rmse"],
                      "displacement": row["anchor_displacement_error_mm"]["rmse"]}
            candidates.append((row, [values[k] for k in dimensions]))
        for row, values in candidates:
            dominated = [other["method_id"] for other, v in candidates
                         if all(a <= b + 1e-10 for a, b in zip(v, values))
                         and any(a < b - 1e-10 for a, b in zip(v, values))]
            out.append(dict(subject=group[0], cycle_index=group[1], method_id=row["method_id"],
                            volume_status=group[2], centroid_status=group[3], geometry=geometry,
                            objectives=dict(zip(dimensions, values)), pareto=not dominated, dominated_by=dominated,
                            compared_methods=[r["method_id"] for r, _ in candidates]))
    return out


def side_runs(paths, baseline, groups, hashes, main_rows, role):
    summaries, phase_rows, metadata = [], [], []
    main = {result_key(r): r for r in main_rows}
    for path in paths:
        raw, meta = load_results([path], baseline, hashes)
        normalized = [normalize(r, baseline[k[:3]]) for k, r in raw.items()]
        add_dynamics(normalized, groups)
        methods = sorted({r["method_id"] for r in normalized})
        summary, _ = summarize(normalized, groups, methods)
        for r in summary:
            r.update(run=str(Path(path).resolve()), role=role)
        summaries.extend(summary)
        for row in normalized:
            row.update(run=str(Path(path).resolve()), role=role)
            source = main.get(result_key(row))
            if source:
                for field in ("volume_ml", "dice", "center_error_mm", "anchor_displacement_error_mm", "original_surface_rms_mm"):
                    row["difference_vs_main_" + field] = (row[field]-source[field]
                        if number(row.get(field)) is not None and number(source.get(field)) is not None else None)
                c, oc = vector(row["center_mm"]), vector(source["center_mm"])
                row["center_difference_vs_main_mm"] = float(np.linalg.norm(c-oc)) if c is not None and oc is not None else None
            phase_rows.append(row)
        metadata.extend(meta)
    return summaries, phase_rows, metadata


def f(value):
    return "—" if value is None else f"{value:.4g}" if isinstance(value, (float, int)) else str(value).replace("|", "\\|").replace("\n", " ")


def table(title, columns, rows):
    return "\n".join(["<details>", f"<summary>{title}</summary>", "", "| " + " | ".join(columns) + " |",
                       "| " + " | ".join("---" for _ in columns) + " |"] +
                     ["| " + " | ".join(f(x) for x in row) + " |" for row in rows] + ["", "</details>", ""])


def plots(rows, summaries, groups, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    figures = []
    for group, bases in groups.items():
        subset = [r for r in rows if (r["subject"], r["cycle_index"]) == group]
        panel = [r for method in METHODS for r in summaries
                 if (r["subject"], r["cycle_index"]) == group and r["method_id"] == method and method != "T0"]
        fig, axes = plt.subplots(1, 5, figsize=(15, 10), constrained_layout=True)
        columns = [("dice", "mean", "Средний Dice", None),
                   ("volume_error_ml", "rmse", "СКО объёма, мл", "volume_status"),
                   ("anchor_volume_change_error_ml", "rmse", "СКО ΔV от опоры, мл", "volume_status"),
                   ("anchor_displacement_error_mm", "rmse", "СКО перемещения, мм", "centroid_status"),
                   ("common_surface_rms_mm", "mean", "Средний RMS поверхности, мм", None)]
        for ax, (field, stat, title, constraint) in zip(axes, columns):
            values = np.array([r[field][stat] if r[field][stat] is not None else np.nan for r in panel])[:, None]
            im = ax.imshow(values, aspect="auto", cmap="viridis" if field == "dice" else "viridis_r")
            ax.set_xticks([]); ax.set_yticks(range(len(panel)), [r["method_id"] for r in panel]); ax.set_title(title, fontsize=9)
            for i, r in enumerate(panel):
                value = values[i, 0]
                label = "нет" if not np.isfinite(value) else f"{value:.3f}"
                if constraint and r[constraint] == "constraint":
                    label += " *"
                color = "black"
                if np.isfinite(value):
                    rgba = im.cmap(im.norm(value)); color = "black" if np.dot(rgba[:3], [.2126, .7152, .0722]) > .5 else "white"
                ax.text(0, i, label, ha="center", va="center", color=color, fontsize=8)
            fig.colorbar(im, ax=ax, fraction=.045, pad=.02)
        fig.suptitle(f"R–R {group[1]} · неопорные фазы · * величина задана построением")
        name = f"cycle_{group[1]}_overview.png"
        fig.savefig(output/name, dpi=150, facecolor="white"); plt.close(fig)
        figures.append((name, "Раздельные геометрические показатели на неопорных фазах. Общий балл не вычисляется; звёздочка отмечает заданные объём или центроид. Поверхностный столбец заполняется только общей площадочной оценкой. При неполных данных число фаз следует брать из таблиц."))
        methods = [m for m in METHODS if any(r["method_id"] == m for r in subset)]
        for offset in range(0, len(methods), 7):
            selected = methods[offset:offset+7]
            selected = ["T0"] + [m for m in selected if m != "T0"]
            fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
            x = [b["phase_percent"] for b in bases]
            lookup = {(r["method_id"], r["phase_id"]): r for r in subset}
            for method in selected:
                seq = [lookup.get((method, b["phase_id"]), {}) for b in bases]
                style = dict(marker="o", ms=3, lw=2 if method == "T0" else 1, label=method)
                for ax, field in zip(axes.flat, ("volume_ml", "anchor_volume_change_ml", "anchor_displacement_error_mm", "dice")):
                    ax.plot(x, [r.get(field) if number(r.get(field)) is not None else np.nan for r in seq], **style)
                    # Числа несошедшегося решения сохраняются, крест отмечает ограничение.
                    bad = [(xp, r.get(field)) for xp, r in zip(x, seq) if r and not r.get("optimization_acceptable") and number(r.get(field)) is not None]
                    if bad:
                        ax.scatter(*zip(*bad), marker="x", c="red", s=35)
            for ax, title, ylabel in zip(axes.flat,
                    ("Абсолютный объём", "Изменение объёма от опоры R–R", "Ошибка перемещения от опоры", "Перекрытие с маской"),
                    ("Объём, мл", "Изменение, мл", "Ошибка вектора, мм", "Dice")):
                ax.set(title=title, xlabel="Фаза внутри данного R–R, %", ylabel=ylabel)
                ax.grid(alpha=.2)
            axes[0, 0].legend(ncol=4)
            fig.suptitle(f"Цикл R–R {group[1]} · описание всех фаз · красный ×: сходимость не подтверждена")
            name = f"cycle_{group[1]}_dynamics_{offset//7+1}.png"
            fig.savefig(output/name, dpi=150, facecolor="white")
            plt.close(fig)
            figures.append((name, "Объём и движение целого сердца. Линии соединяют только соседние наблюдаемые фазы одного R–R; пропуски разрывают линию. T0 — контроль. Совпадения заданных величин не являются независимой точностью."))
            fig = plt.figure(figsize=(9, 7), constrained_layout=True)
            ax = fig.add_subplot(111, projection="3d")
            for method in selected:
                seq = [lookup.get((method, b["phase_id"]), {}) for b in bases]
                c = np.array([r.get("anchor_center_delta_mm") if vector(r.get("anchor_center_delta_mm")) is not None else [np.nan]*3 for r in seq])
                ax.plot(c[:, 0], c[:, 1], c[:, 2], "o-", ms=3, label=method)
            ax.set(xlabel="Δx, мм", ylabel="Δy, мм", zlabel="Δz, мм", title=f"Траектории относительно собственной опоры · R–R {group[1]}")
            limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
            radius = max(np.ptp(limits, axis=1).max()/2, .5)
            for setter, midpoint in zip((ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d), limits.mean(axis=1)):
                setter(midpoint-radius, midpoint+radius)
            ax.set_box_aspect((1, 1, 1)); ax.legend()
            name = f"cycle_{group[1]}_trajectory_{offset//7+1}.png"
            fig.savefig(output/name, dpi=150, facecolor="white"); plt.close(fig)
            figures.append((name, "Трёхмерные перемещения центров, мм. Оси имеют одинаковый масштаб; координаты взяты из физической системы baseline, обозначение RAS/LPS не назначается предположением. Постоянное смещение центра устраняется собственной опорой."))
    return figures


def report(output, rows, summaries, fronts, failures, history_summary, variant_summary, variant_rows, figures, manifest):
    text = ["# Сравнение динамики геометрических аппроксимаций целого сердца Ника", "",
            "Статус: описательный анализ имеющихся автоматических масок 4D-КТ. Сводка описывает выполненные пофазные геометрические расчёты; сам анализатор не выполняет повторную подгонку, сегментацию или FEM. "
            "Пофазная аппроксимация использует целевую КТ каждой фазы; её ошибки не устанавливают точность восстановления по одной КТ или электрическому сигналу.", "",
            f"Получено {len(rows)} из {manifest['expected_records']} записей. Полнота: {manifest['complete']}. "
            "Строка с отказом учитывается в покрытии. Наличие чисел и сходимость выбранного решения считаются раздельно.", "",
            "M1–M9 используют проверенные алгебраические процедуры. Статус algebraic_valid означает допустимую геометрию результата; "
            "он не является подтверждением итерационной сходимости или устойчивости. M7 с условием xyz является сферой и численно совпадает с M9; "
            "эти две строки реестра не представляют независимые семейства форм. Предупреждения библиотечного кода сохранены в полном массиве.", "",
            "## Правила и смысл показателей", "",
            "Два R–R имеют собственные опоры. Все 12 фаз приведены в полном массиве; основные ошибки усредняются по 10 неопорным фазам, раздельно по циклам. "
            "T0 оставлен контролем; самосравнения I1–I3 на опоре исключены из основных сводок. Через переход 92→1 % нет соединения, сглаживания или замыкания траектории.", "",
            r"Ошибка уровня объёма равна $\hat V_t-V_t$, мл; процентная ошибка делится на $V_t$. Ошибка приращения от опоры: "
            r"$[(\hat V_t-\hat V_0)-(V_t-V_0)]$. Для соседних фаз 0 заменяется на предыдущую наблюдаемую фазу того же R–R.", "",
            r"Ошибка движения определяется как $\|[\hat c_t-\hat c_0]-[c_t-c_0]\|$, мм. Ошибка положения центра "
            r"$\|\hat c_t-c_t\|$ приводится отдельно. СКО — корень среднего квадрата ошибки по доступным неопорным фазам; число фаз указано. "
            "Длина траектории — сумма расстояний между соседними центрами; неполный набор не сравнивается с полной длиной. Скорости вычисляются только при положительном известном интервале времени.", "",
            "Размах и времена минимумов/максимумов относятся только к наблюдаемым фазам. Это не КДО/КСО, УО или ФВ. "
            "При равных экстремумах сохраняются все фазы. Временное разрешение КТ хранится в полном массиве; интерполяция экстремумов не производится.", "",
            "Статус constraint означает величину, заданную построением из целевой КТ. Её малая ошибка — проверка сохранения ограничения. "
            "Статус estimated означает описательную подгонку; он также не является независимой валидацией. Непрерывный объём параметров и дискретно подсчитанный объём сохраняются отдельно.", "",
            "Ориентация сравнивается по главным осям с устранением неопределённости их знаков. При совпадении полуосей направление не оценивается. "
            "Полуоси индивидуальной формы, извлечённые из ковариации, являются моментными характеристиками, а не новыми эллипсоидальными аппроксимациями. "
            "Признак касания границы КТ сохраняется; полнота анатомического охвата этим анализом не подтверждается.", "",
            "## Покрытие и основные ошибки", "",
            "Таблица 1 разделяет вычисленные показатели и состояние оптимизации. Ошибки в ней описывают все доступные численные решения; "
            "для Парето применяется более строгий отбор полного набора с допустимым статусом выбранного решения. Успех всех стартов не требуется для успеха выбранного, но показывается отдельно.", "",
            table("Таблица 1. Покрытие и ошибки на неопорных фазах", ["R–R", "Метод", "Записи / ожидалось", "Числа", "Выбрано допустимых", "Все старты успешны", "Неизвестная сходимость", "V / центр", "СКО V, мл", "СКО ΔV, мл", "СКО движения, мм", "Средний Dice"],
                  [[r["cycle_index"], r["method_id"], f"{r['records']}/{r['expected_phases']}", r["numeric_phases"], r["selected_acceptable_phases"], r["all_starts_success_phases"], r["selected_unknown_phases"],
                    r["volume_status"]+" / "+r["centroid_status"], r["volume_error_ml"]["rmse"], r["anchor_volume_change_error_ml"]["rmse"], r["anchor_displacement_error_mm"]["rmse"], r["dice"]["mean"]] for r in summaries]),
            "Поверхностные показатели первоначальных реализаций сохраняются как диагностические. Они не объединяются в общий RMS. "
            "Если передан общий supplement, таблица 2 использует его односторонние расстояния на 32768 одинаково выбранных площадочных точках цели. "
            "95-й процентиль не называется расстоянием Хаусдорфа. Разность оценок на 8192 и 32768 точках описывает чувствительность к дискретизации, но не доказывает сходимость.", "",
            table("Таблица 2. Общая поверхностная оценка на неопорных фазах", ["R–R", "Метод", "Фаз", "Средний RMS, мм", "Средний p95, мм", "Максимальное изменение RMS по модулю, мм"],
                  [[r["cycle_index"], r["method_id"], r["common_surface_rms_mm"]["n"], r["common_surface_rms_mm"]["mean"], r["common_surface_p95_mm"]["mean"], r["surface_sampling_difference_mm"]["max_abs"]] for r in summaries]),
            "## Парето без общего балла", "",
            "Парето строится отдельно для каждого R–R и одинакового набора ограничений. Геометрическая цель — 1−средний Dice; "
            "при общем supplement даётся отдельный вариант по среднему одностороннему RMS. Свободные объём и перемещение добавляют СКО в мл и мм. "
            "Заданные величины исключены из целей, чтобы их нули не создавали преимущество. Группы с разными ограничениями не ранжируются между собой. "
            "Требуются все фазы, корректные численные данные, подтверждённый допустимый статус выбранного решения и отсутствие нарушенных ограничений. "
            "T0 исключён. Фронт означает отсутствие доминирования внутри данной группы, а не установленный лучший физический метод.", "",
            table("Таблица 3. Условные фронты Парето", ["R–R", "Геометрия", "V / центр", "Метод", "На фронте", "Цели", "Доминируют"],
                  [[r["cycle_index"], r["geometry"], r["volume_status"]+" / "+r["centroid_status"], r["method_id"], r["pareto"], json.dumps(r["objectives"], ensure_ascii=False), ", ".join(r["dominated_by"])] for r in fronts]),
            "## Отказы, история и уточняющие прогоны", "",
            "Численные результаты несошедшихся решений остаются в описательном массиве и отмечаются на графиках. "
            "Неизвестная сходимость не заменяется успехом по факту наличия metrics. Старый L4-прогон и уточняющие расчёты не подменяют основной набор.", "",
            table("Таблица 4. Отказы и ограничения строк основного набора", ["R–R", "Фаза", "Метод", "Статус", "Причина"],
                  [[r.get("cycle_index"), r.get("phase_id"), r.get("method_id"), r.get("selected_solution", r.get("status")), r.get("reason")] for r in failures]),
            table("Таблица 5. История предыдущих попыток", ["Прогон", "R–R", "Метод", "Записей", "Числа", "Выбранных допустимых", "Выбранных отказов"],
                  [[Path(r["run"]).name, r["cycle_index"], r["method_id"], r["records"], r["numeric_phases"], r["selected_acceptable_phases"], r["selected_failed_phases"]] for r in history_summary]),
            table("Таблица 6. Уточняющие прогоны без замены основной версии", ["Прогон", "R–R", "Метод", "Фаз / ожидалось", "Допустимых", "Средний Dice", "СКО V, мл", "СКО движения, мм"],
                  [[Path(r["run"]).name, r["cycle_index"], r["method_id"], f"{r['records']}/{r['expected_phases']}", r["selected_acceptable_phases"], r["dice"]["mean"], r["volume_error_ml"]["rmse"], r["anchor_displacement_error_mm"]["rmse"]] for r in variant_summary]),
            "Пофазные разности уточняющих прогонов с основной версией сохранены в comparison_phases.csv. "
            "Разность исходных поверхностных RMS там остаётся диагностической: изменение оценочной квадратуры может само менять показатель. "
            "Выбор окончательной версии целого метода выполняется отдельно; автоматического выбора лучших фаз здесь нет.", "",
            "## Наблюдаемые экстремумы и длины траекторий", "",
            table("Таблица 7. Размах объёма и длина траектории по всем фазам отдельного R–R", ["R–R", "Метод", "Размах, мл", "Ошибка размаха, мл", "Фазы минимума", "Фазы максимума", "Длина, мм", "Все соседние переходы"],
                  [[r["cycle_index"], r["method_id"], r["volume_extrema_all_phases"].get("range"), r["observed_volume_range_error_ml"],
                    json.dumps(r["volume_extrema_all_phases"].get("minimum_phases", []), ensure_ascii=False), json.dumps(r["volume_extrema_all_phases"].get("maximum_phases", []), ensure_ascii=False),
                    r["step_path_segment_mm_sum"], r["trajectory_complete"]] for r in summaries]),
            "## Графики", ""]
    for i, (name, caption) in enumerate(figures, 1):
        text.extend([f"На рисунке {i} показаны сохранённые геометрические параметры и их межфазные изменения.", "", f"![Рисунок {i}]({name})", "", f"Рисунок {i}. {caption}", ""])
    text += ["## Границы результата и файлы", "",
             "Сводка устанавливает описательные геометрические различия на предоставленных масках. Она не определяет точность электрической обратной задачи, "
             "анатомическую ошибку автоматической сегментации или клинические функции сердца. Два R–R одного участника не являются независимой межсубъектной проверкой.", "",
             "Все численные строки, полуоси, матрицы ориентации, динамика и временные ограничения: [phase_metrics.csv](phase_metrics.csv), "
             "[summary.local.json](summary.local.json). Покрытие и происхождение: [manifest.local.json](manifest.local.json). "
             "Худшие неопорные фазы: [worst_phases.csv](worst_phases.csv). История: [history_phases.csv](history_phases.csv). "
             "Уточнение параметров: [comparison_phases.csv](comparison_phases.csv).", ""]
    (output/"README.md").write_text("\n".join(text), encoding="utf-8")


def run(args):
    hashes = {}
    baseline, groups = load_baseline(args.baseline, hashes)
    flatten = lambda items: [p for batch in (items or []) for p in batch]
    records, manifests = load_results(flatten(args.results), baseline, hashes, args.allow_incomplete)
    expected = {(*k, method) for k in baseline for method in METHODS}
    extra, missing = set(records)-expected, expected-set(records)
    if extra or (missing and not args.allow_incomplete):
        raise ValueError(f"Неполный набор 348: отсутствует {len(missing)}, лишних {len(extra)}; "
                         f"первые пропуски {sorted(missing)[:8]}")
    surfaces, surface_meta = load_surface(args.surface_scores, baseline, records, hashes)
    normalized = [normalize(r, baseline[k[:3]], surfaces.get(k)) for k, r in sorted(records.items())]
    add_dynamics(normalized, groups)
    summaries, worst = summarize(normalized, groups, METHODS)
    fronts = pareto_front(summaries) + (pareto_front(summaries, "surface") if args.surface_scores else [])
    history_summary, history_rows, history_meta = side_runs(flatten(args.history_results), baseline, groups, hashes, normalized, "history")
    variant_summary, variant_rows, variant_meta = side_runs(flatten(args.comparison_results), baseline, groups, hashes, normalized, "comparison")
    failures = [r for r in normalized if not r["numeric_available"] or not r["optimization_acceptable"] or r["warnings"]]
    for key in sorted(missing):
        failures.append(dict(subject=key[0], cycle_index=key[1], phase_id=key[2], method_id=key[3], status="missing", reason="Нет записи результата"))
    manifest = dict(schema="nik_whole_heart_geometry_summary_v1", created_utc=datetime.now(timezone.utc).isoformat(),
                    evidence_status="exploratory_hypothesis_not_validated", complete=not missing,
                    expected_records=len(expected), observed_records=len(records), expected_methods=METHODS,
                    sources=manifests, history_sources=history_meta, comparison_sources=variant_meta,
                    surface=surface_meta, missing=[list(k) for k in sorted(missing)],
                    phases=[{k: b.get(k) for k in ("subject", "cycle_index", "phase_id", "phase_percent", "time_ms", "is_reference_phase", "reference_phase_id", "temporal_resolution_ms")}
                            for values in groups.values() for b in values],
                    checks={"same_masks": True, "two_cycle_anchors": True, "input_matrix_complete": not missing,
                            "FEM_executed": False, "fitting_executed": False, "independent_validation": False},
                    input_sha256=hashes, code_sha256={str(Path(__file__).resolve()): digest(__file__),
                                                    str(Path(__file__).with_name("heart_reduced_geometry.py").resolve()): digest(Path(__file__).with_name("heart_reduced_geometry.py"))})
    # Защита от изменения параллельно дописываемого исходника во время чтения.
    for path, sha in hashes.items():
        if digest(path) != sha:
            raise ValueError(f"Вход изменился во время анализа, повторите после завершения: {path}")
    if args.check_inputs_only:
        print(json.dumps({"complete": not missing, "records": len(records), "numeric": sum(r["numeric_available"] for r in normalized),
                          "acceptable": sum(r["optimization_acceptable"] for r in normalized), "writes": False}, ensure_ascii=False))
        return manifest
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output должен быть пустым: существующая сводка не перезаписывается")
    if any(output == Path(path).parent for path in hashes):
        raise ValueError("Output совпадает с каталогом входов")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"manifest.local.json", manifest)
    write_json(output/"summary.local.json", dict(rows=normalized, cycles=summaries, pareto=fronts, worst=worst,
                                               failures=failures, history=history_summary, comparisons=variant_summary))
    for name, rows in (("phase_metrics", normalized), ("cycle_summary", summaries), ("pareto", fronts), ("worst_phases", worst),
                       ("failures", failures), ("history_phases", history_rows), ("comparison_phases", variant_rows)):
        write_csv(output/(name+".csv"), rows)
    figures = plots(normalized, summaries, groups, output)
    report(output, normalized, summaries, fronts, failures, history_summary, variant_summary, variant_rows, figures, manifest)
    write_json(output/"completion.local.json", dict(status="completed", complete_input_matrix=not missing,
               artifacts_sha256={p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
               note="Завершение построения сводки не означает сходимость всех методов или научную валидацию"))
    print(f"Сводка сохранена: {output}; записей {len(records)}/{len(expected)}")
    return manifest


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--results", required=True, nargs="+", action="append", type=Path)
    p.add_argument("--history-results", nargs="+", action="append", type=Path)
    p.add_argument("--comparison-results", nargs="+", action="append", type=Path)
    p.add_argument("--surface-scores", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--allow-incomplete", action="store_true", help="Только диагностический неполный отчёт; по умолчанию требуется 348 строк")
    p.add_argument("--check-inputs-only", action="store_true", help="Проверить входы без создания файлов")
    return p


if __name__ == "__main__":
    try:
        run(parser().parse_args())
    except (ValueError, KeyError, OSError) as exc:
        print(f"Сводка не построена: {exc}", file=sys.stderr)
        raise SystemExit(2)
