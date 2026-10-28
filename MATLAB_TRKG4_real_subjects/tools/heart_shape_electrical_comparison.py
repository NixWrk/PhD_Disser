"""Compare saved STL/G1/L1 electrical probes. No FEM or physiological inference.

--manifest selects heart_shape_electrical_study_v1; --check-only is read-only.
--self-test runs deterministic algebra/finite-probe checks without input files.
--output selects an explicit output directory; the project-local comparison
directory is the default. Existing output files are never replaced. Weights are shared
STL-base row multipliers, NOT a noise model. All geometric EIT/joint inverses
are diagnostics outside the target PulmoVista rho -> corrected TT geometry path.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.io import loadmat

SHAPES = ("STL", "G1", "L1")
SCENARIOS = ("diagnostic4", "commonSTLfit")
POSES = ("xm5", "xp5", "ym5", "yp5", "zm5", "zp5")
STATES = (("base", 100), ("base", 95), ("base", 105)) + tuple((p, 100) for p in POSES)
MATERIALS = ["soft_background", "lung", "bone", "heart", "fat"]
TISSUES = [0, 1, 3]
PARAMETERS = ("V", "x", "y", "z", "rho_soft", "rho_lung", "rho_heart")
PARAMETER_UNITS = ("ml", "mm", "mm", "mm", "ohm*m", "ohm*m", "ohm*m")
SETTINGS = ("frequency_Hz", "current_A", "montage_basis", "breath_interval_status")
ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "Colab Notebooks/Отчёт_ГПХ_РНЦХ/pulmovista_check_20261028/local"
OUTPUT = LOCAL / "shape_electrical_comparison_v1"
LIMITATIONS = [
    "Результаты вычислительных опытов; независимая физическая/клиническая валидация отсутствует.",
    "Оба набора rho общие для всех форм и условные. commonSTLfit взят только из строки STL; это не измеренные истинные свойства тканей.",
    "STL целого сердца служит геометрической опорой. Полнота камер/миокарда и анатомическая точность не подтверждены.",
    "Изотропное масштабирование всей STL на ±5% объёма не является фазовой КТ, УО или ФВ; ±5 мм не являются измеренной траекторией.",
    "Один неподвижный фон с приоритетом сердца; замещение материалов не моделирует механическую деформацию соседних органов.",
    "Расширена только область проверки сердца; вне прежней области фон наследует центроидные метки. Равенство трёх базовых карт проверяет совместимость, а не глобальную сходимость.",
    "Численная добавка объёма при переносах сохраняется; её линейное вычитание не устраняет квадратурную/FEM-ошибку. Показаны raw и volume_corrected.",
    "Секущие ±5% и ±5 мм имеют конечный шаг; различие односторонних производных характеризует нелинейность вместе с численной ошибкой.",
    "Мягкие ткани и единое лёгкое могут давать пульсовой вклад даже на задержке дыхания; он не принят равным нулю. Кости/жир фиксированы.",
    "Производная эффективного rho сердца и геометрический вклад проверяются на различимость; их нельзя считать двумя независимыми объяснениями одного сигнала.",
    "Численный rank=7 не доказывает практическую идентифицируемость. Ошибки модели, контактов, rho и шум не измерены; шумовое распределение и амплитуда не назначены.",
    "Относительные веса 1/abs(STL base) общие внутри сценария и являются метрикой, а не обратными стандартными отклонениями шума; нулевые каналы не заменяются произвольным floor.",
    "Диагностические координаты: 1% геометрического объёма STL, 1 мм по каждой оси, 1% базового rho. Усиление дано на единицу нормы возмущения данных, без доверительных интервалов.",
    "ЭИТ и совместная ЭИТ+ТТРКГ инверсия геометрии — отдельная диагностика вне целевого пути PulmoVista rho/Δrho -> поправки -> геометрия по ТТРКГ и одной КТ.",
    "Один скаляр ТТРКГ имеет ранг не более 1: даже после известных тканевых поправок четыре свободных геометрических параметра не определяются.",
    "Монтажи ЭИТ/ТТРКГ, контакты и алгоритмический пояс унаследованы из хэшированных моделей; физическая установка во время записи не подтверждена.",
    "Совместный стек использует одну частоту 79505 Гц и ток модели. Имеющиеся разные сессии не объединяются как синхронные; перенос частоты и задержки трактов не проверены.",
    "Дифференциальная инверсия вычитает собственную базу каждой формы. Базовый межформенный offset не подмешивается к динамическому сигналу и показан отдельно.",
    "Same-shape finite-probe controls используют те же конечные пробы, что секущие: это внутренний контроль линеаризации, не независимая проверка восстановления.",
    "Оценки за пределами ±5% исходного STL-объёма или ±5 мм и rho0+Δrho<=0 помечаются без clipping. Большие значения линейной экстраполяции не являются измеренными изменениями.",
    "Линейное выравнивание базового интегрального объёма G1/L1 к STL использует собственную J_V формы. Это диагностика численного несовпадения, не точный пересчёт/перестроение сетки.",
]
REPORTED_REVIEW = dict(source="author message in this task, 2026-10-28; not a locally read audit artifact",
    reviewer="Curie",audited_analysis_sha256="0fbfaf2b91cf40903286eaf4f7810898b906c9ad0c12c5b2460acc3475987658",
    scope="diagnostic4 complete; commonSTLfit incomplete at review time",
    reported_independent_finite_transfer_lstsq_checks=144,
    reported_max_disagreement_diagnostic_units_upper_bound=1.24e-13,
    reported_selftest_groups_passed=5,reported_material_findings="none",
    interpretation="Reported numerical audit of that code version only; no physical or clinical validation")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, message, atol=1e-10):
    require(np.shape(a) == np.shape(b) and np.allclose(a, b, rtol=1e-10, atol=atol), message)


def resolve(path, parent):
    p = Path(path)
    return p.resolve() if p.is_absolute() else (parent / p).resolve()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return jsonable(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None  # rank/nullity/full-rank flags disambiguate unbounded quantities
    if isinstance(x, Path):
        return str(x)
    return x


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(jsonable(data), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in rows:
            cleaned = jsonable(row)
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in cleaned.items()})


class Sources:
    """Bind inputs to content hashes and detect modification during reading."""
    def __init__(self):
        self.records = {}

    def add(self, path, expected=None):
        p = Path(path).resolve()
        stat = p.stat()
        key = str(p)
        if key not in self.records:
            with p.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            after = p.stat()
            require((stat.st_size, stat.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                    f"Source changed while hashing: {p}")
            self.records[key] = dict(sha256=digest, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        record = self.records[key]
        require((record["bytes"], record["mtime_ns"]) == (stat.st_size, stat.st_mtime_ns),
                f"Source changed during audit: {p}")
        if expected is not None:
            require(record["sha256"].lower() == str(expected).lower(), f"SHA256 mismatch: {p}")
        return record["sha256"]

    def unchanged(self, rehash=False):
        for path in self.records:
            self.add(path)
            if rehash:
                with Path(path).open("rb") as stream:
                    digest=hashlib.file_digest(stream,"sha256").hexdigest()
                self.add(path)
                require(digest==self.records[path]["sha256"],f"Source content changed after analysis: {path}")
                self.records[path]["sha256_after_analysis"]=digest


@dataclass
class Probe:
    V: np.ndarray
    S: np.ndarray
    Z: float
    K: np.ndarray
    rho: np.ndarray
    volume: float
    target: float
    metadata: dict


def variant(shape, pose, level):
    return f"{shape}_{'' if pose == 'base' else pose + '_'}v{level:03d}_q64"


def load_inputs(manifest_path, sources):
    """Read the preparer's explicit map; never infer substitutes from filenames."""
    sources.add(manifest_path)
    sources.add(__file__)
    b = manifest_path.parent
    manifest = read_json(manifest_path)
    require(manifest["schema"] == "heart_shape_electrical_study_v1", "Unknown study schema")
    require(tuple(manifest["shapes"]) == SHAPES and set(manifest["scenarios"]) == set(SCENARIOS),
            "Require exactly STL/G1/L1 and the two COMMON rho scenarios")
    require(manifest["frequency_Hz"] == 79505 and manifest["current_A"] > 0, "Frequency/current mismatch")
    for path, digest in manifest["sources_sha256"].items():
        sources.add(resolve(path, b), digest)
    require(manifest["preparation_QC"]["common_background_arrays_identical"] is True,
            "Preparer did not pass the common-background control")
    plan_path = resolve(manifest["material_plan"], b)
    require(str(plan_path) in sources.records, "Material plan not bound by manifest hashes")
    plan = read_json(plan_path)
    require(plan["schema"] == "heart_stl_materials_v1" and plan["names"] == MATERIALS,
            "Unexpected material order/schema")
    variants = {v["id"]: v for v in plan["variants"]}
    require(len(variants) == len(plan["variants"]) == 27, "Require 27 distinct material states")
    source_plans = []
    for path, digest in plan["source_plans_sha256"].items():
        pp = resolve(path, ROOT)
        sources.add(pp, digest)
        sp = read_json(pp)
        for key in ("mesh_sha256", "stl_sha256", "seed", "quadrature_rule", "names"):
            require(sp[key] == plan[key], f"Material source setup differs: {key}")
        for src, h in sp["sources"].items():
            sources.add(resolve(src, ROOT), h)
        source_plans.append((pp, sp))
    # Recheck the arrays, not just the preparer's boolean. Different MAT headers
    # need not hash identically, and the expanded selected-tetrahedra count is legal.
    controls = []
    for shape in SHAPES:
        name = variant(shape, "base", 100)
        found = [(pp, sp, v) for pp, sp in source_plans for v in sp["variants"] if v["id"] == name]
        require(len(found) == 2, f"Missing old/new baseline control: {shape}")
        arrays, paths = [], []
        for pp, sp, v in found:
            mp = resolve(v["material_file"], pp.parent)
            sources.add(mp, v["material_sha256"])
            arrays.append(loadmat(mp, variable_names=["fractions"])["fractions"])
            paths.append(str(mp))
        require(np.array_equal(arrays[0], arrays[1]), f"Baseline fractions changed: {shape}")
        controls.append(dict(shape=shape, paths=paths, array_equal=True,
                             array_shape=list(arrays[0].shape), selected_tetrahedra=[sp["selected_tetrahedra"] for _, sp, _ in found]))
        del arrays
    for name, v in variants.items():
        sources.add(resolve(v["material_file"], plan_path.parent), v["material_sha256"])
        require(v["quadrature"] == 64, f"Mixed quadrature: {name}")
        originals = [sv for _, sp in source_plans for sv in sp["variants"]
                     if sv["id"] == name and sv["material_sha256"] == v["material_sha256"]]
        require(originals, f"Material has no hashed source-plan record: {name}")
        for key in ("volume_ml", "target_volume_ml", "method", "quadrature"):
            require(v[key] == originals[0][key], f"Merged material metadata mismatch: {name}/{key}")
    fit_path = b / "stl_effective_fit_q64_v1/effective_fit.local.json"
    require(str(fit_path.resolve()) in sources.records, "Effective fit not pinned by manifest")
    fit = read_json(fit_path)
    fit_rows = [r for r in fit["rows"] if r["id"] == "STL"]
    require(len(fit_rows) == 1 and fit_rows[0]["converged"] and fit["gain"] == 1,
            "No unique completed STL fit with fixed gain")
    close(manifest["scenarios"]["commonSTLfit"], fit_rows[0]["rho_ohm_m"], "commonSTLfit is not the STL row")
    require(manifest["scenarios"]["diagnostic4"][3] == 4, "diagnostic4 heart rho is not 4")
    close(np.asarray(manifest["scenarios"]["diagnostic4"])[[2, 4]],
          np.asarray(manifest["scenarios"]["commonSTLfit"])[[2, 4]], "Bone/fat differ between scenarios")
    models = {k: sources.add(resolve(p, b)) for k, p in manifest["model_files"].items()}
    for key, field in (("eit_model_sha256", "eit_model_file"), ("ttrkg_model_sha256", "ttrkg_model_file"),
                       ("reference_sha256", "reference_prediction_file")):
        require(fit["identity"][key] == models[field], f"Fit model mismatch: {key}")
    for key in SETTINGS:
        require(fit[key] == manifest[key], f"Fit setting mismatch: {key}")
    tt = loadmat(resolve(manifest["model_files"]["ttrkg_model_file"], b),
                 variable_names=["Jtt", "Ksigma"], simplify_cells=True)
    close(tt["Jtt"], manifest["current_A"] * tt["Ksigma"], "TT current conversion mismatch")
    candidates_path = b / "shape_candidates_v1.local.json"
    require(str(candidates_path.resolve()) in sources.records, "Geometry candidates not pinned")
    candidates = read_json(candidates_path)
    vref = float(plan["reference_STL_volume_ml"])
    for shape in SHAPES:
        base = candidates[shape]
        require(base["coordinate_system"] == "LPS" and base["length_unit"] == "mm", "Geometry units")
        require(base["source_sha256"] == plan["stl_sha256"], "Geometry/STL provenance differs")
        close(base["volume_ml"], vref, "Nominal forms are not equal-volume")
        close(base["center_mm"], candidates["STL"]["center_mm"], "Nominal centers differ")
        for pose in POSES:
            # L1 translations belong to the old candidate source, not necessarily
            # the new STL/G1 candidate JSON. Resolve through its material source.
            name = shape + "_" + pose
            candidate = candidates.get(name)
            if candidate is None:
                paths = {resolve(src, ROOT) for _, sp in source_plans for src in sp["sources"]
                         if Path(src).name == "stl_pose_candidates.local.json"}
                require(len(paths) == 1, "Ambiguous old pose candidates")
                candidate = read_json(paths.pop())[name]
            shift = np.zeros(3); shift["xyz".index(pose[0])] = -5 if pose[1] == "m" else 5
            close(np.asarray(candidate["center_mm"]) - base["center_mm"], shift, "Wrong nominal translation")
            close(candidate["volume_ml"], vref, "Translation changed nominal volume")
            for key in ("semi_axes_mm", "axes_matrix", "scale", "reference_centroid_mm"):
                if key in base:
                    close(candidate[key], base[key], f"Translation changed shape: {key}")
    configs, mat_cache, probes, missing, volumes = {}, {}, {}, [], []
    expected = {(s, h, p, l) for s in SCENARIOS for h in SHAPES for p, l in STATES}
    seen = set()
    for record in manifest["probes"]:
        s, h, p, l = (record[k] for k in ("scenario", "shape", "pose", "level"))
        key = s, h, p, l
        require(key in expected and key not in seen, f"Unexpected/duplicate probe: {key}")
        seen.add(key)
        require(record["variant"] == variant(h, p, l), "Variant map mismatch")
        close(record["rho_ohm_m"], manifest["scenarios"][s], "Per-form rho not common")
        cp = resolve(record["config"], b)
        if cp not in configs:
            sources.add(cp)
            cfg = read_json(cp)
            for field in SETTINGS:
                require(cfg[field] == manifest[field], f"Config setting mismatch: {cp}/{field}")
            for field, digest in models.items():
                sources.add(resolve(cfg[field], cp.parent), digest)
            pp = resolve(cfg["material_plan"], cp.parent)
            sources.add(pp)
            source_plan = read_json(pp)
            require(source_plan["names"] == MATERIALS, "Source material order differs")
            sources.add(resolve(cfg["mesh_path"], cp.parent), plan["mesh_sha256"])
            sources.add(resolve(cfg["stl_path"], cp.parent), plan["stl_sha256"])
            configs[cp] = cfg, pp, {v["id"]: v for v in source_plan["variants"]}
        cfg, pp, cfg_variants = configs[cp]
        material = variants[record["variant"]]
        cv = cfg_variants[record["variant"]]
        for field in ("material_sha256", "volume_ml", "target_volume_ml", "quadrature"):
            require(cv[field] == material[field], f"Config material differs: {record['variant']}/{field}")
        close(material["target_volume_ml"], vref * l / 100, "Volume law is not +/-5% original STL volume")
        path = resolve(record["path"], b)
        if not path.exists():
            missing.append(dict(key=list(key), path=str(path), reused=record["reused"]))
            continue
        require(not record["reused"] or record.get("mat_sha256"), "Reuse requires pre-existing MAT hash")
        digest = sources.add(path, record.get("mat_sha256"))
        if path not in mat_cache:
            mat_cache[path] = loadmat(path, simplify_cells=True)
        data = mat_cache[path]
        identity, row = data["identity"], data["row"]
        sources.add(cp, identity["config_sha256"])
        sources.add(pp, identity["plan_sha256"])
        for field, model_field in (("eit_model_sha256", "eit_model_file"), ("ttrkg_model_sha256", "ttrkg_model_file")):
            require(identity[field] == models[model_field], f"Different operator: {path}/{field}")
        impl = "heart_stl_electrical_check.m" if record["format"] == "electrical" else "heart_stl_effective_fit.m"
        sources.add(Path(__file__).with_name(impl), identity["implementation_sha256"])
        for field, filename in (("sensitivity_sha256", "cr_compartment_sensitivity.m"),
                                ("upstream_solve_sha256", "cr_compartment_solve.m"),
                                ("residual_sha256", "cr_relative_residual.m")):
            if field in identity:
                matches = list(resolve(cfg["pipeline_root"], cp.parent).glob("**/" + filename))
                require(len(matches) == 1, f"Ambiguous source implementation: {filename}")
                sources.add(matches[0], identity[field])
        if record["format"] == "electrical":
            require(record["probe_index"] is None and row["variant"] == record["variant"], "Electrical record mismatch")
            jobs = [j for j in cfg["jobs"] if j["id"] == row["id"]]
            require(len(jobs) == 1 and jobs[0]["variant"] == row["variant"] and
                    jobs[0]["scenario"] == row["scenario"] and jobs[0]["eit"] and row["eit_computed"], "Saved electrical job mismatch")
            require(path == resolve(cfg["output_dir"], cp.parent) / (row["id"] + ".mat"), "Wrong job result path")
            close(jobs[0]["rho_ohm_m"], record["rho_ohm_m"], "Job rho mismatch")
            entry, meta = data, row
            close(data["Ktt"], row["Ktt_ohm_per_S_m"], "Saved TT sensitivity mismatch")
        else:
            require(h == "STL" and s == "commonSTLfit" and p == "base", "Only STL fit may be reused")
            require(row["id"] == "STL" and row["base_variant"] == variant(h, "base", 100), "Wrong fit row")
            require(identity == fit["identity"], "Fit MAT/report identity differs")
            for field in ("rho_ohm_m", "volume_ml", "target_volume_ml", "Ztt_ohm"):
                close(row[field], fit_rows[0][field], f"Fit MAT/report disagreement: {field}")
            if record["format"] == "effective_fit_base":
                require(l == 100 and record["probe_index"] is None, "Invalid fit base selector")
                entry, meta = data, row
            else:
                require(record["format"] == "effective_fit_probe" and l in (95, 105), "Unknown MAT format")
                index = 0 if l == 95 else 1
                require(record["probe_index"] == index, "Wrong MATLAB probe index (must be zero-based)")
                entry = data["probes"][index]
                meta = entry
                require(entry["variant"] == record["variant"], "Wrong embedded probe variant")
        require(meta["material_sha256"] == material["material_sha256"], "MAT material hash mismatch")
        close(meta["volume_ml"], material["volume_ml"], "Integrated volumes mismatch")
        close(meta["target_volume_ml"], material["target_volume_ml"], "Nominal volumes mismatch")
        sigma = np.asarray(data["sigma"]).reshape(-1)
        require(sigma.shape == (5,) and np.isrealobj(sigma) and np.all(sigma > 0), "Invalid sigma")
        close(1 / sigma, record["rho_ohm_m"], "sigma/rho mismatch")
        close(row["rho_ohm_m"], record["rho_ohm_m"], "Row rho mismatch")
        v, sens, kt = np.asarray(entry["V_eit"]).reshape(-1), np.asarray(entry["S_eit"]), np.asarray(entry["Ktt"]).reshape(-1)
        require(all(np.isrealobj(a) for a in (v, sens, kt, entry["Ztt"])), "Complex FEM data must not be silently truncated")
        z = float(entry["Ztt"])
        require(v.shape == (208,) and sens.shape == (208, 5) and kt.shape == (5,), "Require 208 EIT and five tissue columns")
        require(np.isfinite(np.r_[v, sens.ravel(), kt, z, sigma]).all(), "Nonfinite saved FEM data")
        if entry is data:
            close(z, row["Ztt_ohm"], "TT row/array mismatch")
        for field in ("native_forward_relative_error", "reciprocity_relative_error"):
            if field in row:
                require(np.isfinite(row[field]) and row[field] < 1e-8, f"Failed saved solver QC: {field}")
        vol = np.asarray(meta["volume_ml"])
        require(vol.shape == (5,) and np.isfinite(vol).all() and np.all(vol > 0), "Invalid material volumes")
        metadata = dict(path=str(path), sha256=digest, format=record["format"], probe_index=record["probe_index"],
                        reused=record["reused"], variant=record["variant"], material_sha256=meta["material_sha256"])
        probes[key] = Probe(v, sens, z, kt, 1 / sigma, float(vol[3]), float(meta["target_volume_ml"]), metadata)
        volumes.append(dict(scenario=s, shape=h, pose=p, level=l, integrated_heart_ml=vol[3], target_heart_ml=meta["target_volume_ml"],
                            mismatch_ml=vol[3] - meta["target_volume_ml"], total_material_ml=float(vol.sum()), **metadata))
    require(seen == expected and len(manifest["probes"]) == 54, "Incomplete 54-entry study contract")
    reused_count=sum(bool(r["reused"]) for r in manifest["probes"])
    new_cfg=read_json(resolve(manifest["new_config"],b))
    require(reused_count == new_cfg["shape_comparison"]["reused_states"] and
            54-reused_count == len(new_cfg["jobs"]) == new_cfg["shape_comparison"]["new_FEM_jobs"],
            "Manifest reuse/new-job counts disagree with the hashed config")
    if volumes:
        close(np.array([v["total_material_ml"] for v in volumes]), np.full(len(volumes), volumes[0]["total_material_ml"]),
              "Total fixed-mesh material volume differs", atol=1e-6)
    for r in volumes:
        base = probes.get((r["scenario"], r["shape"], "base", 100))
        if base:
            r["integrated_delta_from_base_ml"] = r["integrated_heart_ml"] - base.volume
            r["nominal_delta_from_base_ml"] = r["target_heart_ml"] - base.target
    rerun_checks=[]
    rerun_keys=[("commonSTLfit","STL","base",level) for level in (95,100,105)]
    if all(key in probes for key in rerun_keys):
        old_path=b/"stl_effective_fit_q64_v1/STL.mat"
        old_hash=sources.add(old_path)
        historical=loadmat(old_path,simplify_cells=True)
        require(historical["identity"]==fit["identity"],"Historical fit MAT/report identity differs")
        for key in rerun_keys:
            level=key[-1]
            old=historical if level==100 else historical["probes"][0 if level==95 else 1]
            new=probes[key]
            check=dict(level=level,historical_mat_sha256=old_hash,new_mat_sha256=new.metadata["sha256"],
                historical_mat_is_operational_probe_input=False,new_probe_reused=new.metadata["reused"],
                status="diagnostic_saved_array_rerun_comparison_not_physical_validation")
            for field,value in (("V_eit",new.V),("S_eit",new.S),("Ktt",new.K),("Ztt",new.Z)):
                previous=np.asarray(old[field])
                close(np.shape(value),np.shape(previous),f"Historical/new array dimensions differ: {field}")
                check[field+"_max_abs_difference"]=float(np.max(np.abs(value-previous)))
                check[field+"_exact_equal"]=bool(np.array_equal(value,previous))
            rerun_checks.append(check)
    sources.unchanged()
    return manifest, probes, dict(status="waiting_for_inputs" if missing else "input_QC_passed",
        expected=54, available=len(probes), missing=missing, common_background_controls=controls,
        source_material_plans=[dict(path=str(pp), selected_tetrahedra=sp["selected_tetrahedra"],
            historical_implementation_sha256=sp.get("implementation_sha256")) for pp, sp in source_plans],
        integrated_volume_qc=volumes, reference_STL_volume_ml=vref,historical_STL_fit_rerun_checks=rerun_checks,
        current_conversion_checked=True, preparation_QC=manifest["preparation_QC"],
        background_check_scope="Numerical array equality of three baselines; expanded heart envelope does not require equal selected-tetrahedra counts")


def ratio(a, b):
    return float(a / b) if b != 0 else None


def relnorm(a, reference):
    return ratio(np.linalg.norm(a), np.linalg.norm(reference))


def compare_vectors(candidate, reference, weights):
    a, r = weights * np.asarray(candidate), weights * np.asarray(reference)
    nr, na = np.linalg.norm(r), np.linalg.norm(a)
    cosine = float(np.clip(a @ r / (na * nr), -1, 1)) if na and nr else None
    scale = ratio(a @ r, nr**2)
    return dict(norm_relative_error=ratio(np.linalg.norm(a - r), nr),
        signed_projection_relative_error=ratio((a - r) @ r, nr**2),
        signed_mean_error=float(np.mean(a - r)), norm_ratio=ratio(na, nr),
        cosine_uncentered=cosine,
        angle_deg=float(np.degrees(np.arctan2(np.linalg.norm(a/na-cosine*r/nr),cosine))) if cosine is not None else None,
        scale_candidate_on_reference=scale,
        residual_after_scale_relative=ratio(np.linalg.norm(a - scale * r), nr) if scale is not None else None)


def svd_inverse(a):
    """One explicit machine-precision rule for rank and Moore-Penrose inverse."""
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    tolerance = max(a.shape) * np.finfo(float).eps * (s[0] if s.size else 0)
    keep = s > tolerance
    inverse = (vh[keep].T / s[keep]) @ u[:, keep].T
    return inverse, s, int(keep.sum()), float(tolerance)


def design_diagnostics(j, weights, steps, volume_slopes=None):
    weighted = weights[:, None] * j
    norms = np.linalg.norm(weighted, axis=0)
    normalized = np.divide(weighted, norms, out=np.zeros_like(weighted), where=norms > 0)
    _, singular, rank, tolerance = svd_inverse(normalized)
    n = j.shape[1]
    singular = np.pad(singular, (0, n - len(singular)))
    gram = normalized.T @ normalized
    gram[norms == 0, :] = np.nan; gram[:, norms == 0] = np.nan
    unique = []
    for k in range(n):
        others = np.delete(normalized, k, axis=1)
        pinv, _, _, _ = svd_inverse(others)
        unique.append(float(np.linalg.norm(normalized[:, k] - others @ pinv @ normalized[:, k])) if norms[k] else None)
    # A maps the stated dimensionless diagnostic coordinates to weighted data.
    a = weighted * steps
    inverse, dimensional_s, dimensional_rank, dtol = svd_inverse(a)
    resolution = inverse @ a
    full_rank = rank == n and dimensional_rank == n
    physical_inverse=steps[:,None]*inverse
    if volume_slopes is not None:
        physical_inverse[0]+=volume_slopes@physical_inverse[1:4]
    physical_coordinates_inverse=physical_inverse/steps[:,None]
    return dict(rank=rank, unknowns=n, nullity=n-rank, full_column_rank=full_rank,
        normalized_singular_values=singular, normalized_condition=float(singular[0]/singular[-1]) if rank == n else None,
        condition_status="finite" if rank == n else "unbounded_column_identification",
        normalized_rank_tolerance=tolerance, weighted_column_norms=norms, pairwise_cosines=gram,
        unique_residual_fraction=unique, diagnostic_steps=steps, diagnostic_singular_values=dimensional_s,
        diagnostic_rank=dimensional_rank, diagnostic_rank_tolerance=dtol,
        diagnostic_condition=float(dimensional_s[0]/dimensional_s[-1]) if dimensional_rank == n else None,
        worst_coordinate_amplification_per_unit_data_norm=float(1/dimensional_s[-1]) if full_rank else None,
        amplification_status="full_rank_local_linear" if full_rank else "unbounded_unidentified_directions",
        pseudoinverse_row_norms=np.linalg.norm(inverse, axis=1),
        physical_pseudoinverse_row_norms=np.linalg.norm(physical_inverse,axis=1),
        physical_diagnostic_coordinate_row_norms=np.linalg.norm(physical_coordinates_inverse,axis=1),
        physical_worst_coordinate_amplification_per_unit_data_norm=float(np.linalg.norm(physical_coordinates_inverse,2)) if full_rank else None,
        volume_coordinate="integrated deltaV after undoing target raw translation drift" if volume_slopes is not None else "integrated deltaV (translation drift removed)",
        coordinate_resolution=resolution, measured_noise_amplitude=None,
        amplification_definition="||row(A^+)||_2 per unit ||W*delta_data||_2; A=W*J*diag(steps); no variance/noise distribution assigned")


def fit_linear(j, y, weights, steps):
    a = weights[:, None] * j * steps
    inverse, singular, rank, tolerance = svd_inverse(a)
    coordinates = inverse @ (weights * y)
    coefficients = steps * coordinates
    residual = y - j @ coefficients
    return dict(coefficients=coefficients, diagnostic_coordinates=coordinates, rank=rank,
        full_column_rank=rank == j.shape[1], rank_tolerance=tolerance,
        residual=residual, weighted_residual_norm=float(np.linalg.norm(weights * residual)),
        weighted_probe_norm=float(np.linalg.norm(weights * y)), residual_fraction=relnorm(weights * residual, weights * y),
        solution_status="unique_local_linear" if rank == j.shape[1] else "minimum_diagnostic_coordinate_norm_not_identified")


def tissue_projection(jrho, weights):
    """Exact requested (W S)^+ W in physical rho coordinates (no assumed noise)."""
    inverse, singular, rank, tolerance = svd_inverse(weights[:,None]*jrho)
    return inverse*weights, rank, tolerance


def downstream_identification(jrho, projection, krho, rank):
    """An EIT-null tissue direction may still change the subsequent TT correction."""
    null_functional=krho-krho@projection@jrho
    norm=float(np.linalg.norm(null_functional))
    tolerance=100*max(jrho.shape)*np.finfo(float).eps*np.linalg.norm(krho)
    identified=bool(norm<=tolerance)
    return dict(tissue_solution_status="unique_local_linear" if rank==jrho.shape[1] else "minimum_rho_norm_representative",
        downstream_TT_correction_identified=identified,
        downstream_TT_null_functional_norm=norm,downstream_TT_null_functional_relative=relnorm(null_functional,krho),
        downstream_TT_null_test_tolerance=float(tolerance),
        downstream_TT_status="identified_functional" if identified else "minimum_norm_representative_not_identified")


def volume_drift_slopes(probes, scenario, shape):
    return np.array([(probes[scenario,shape,axis+"p5",100].volume-
                      probes[scenario,shape,axis+"m5",100].volume)/10 for axis in "xyz"])


def estimate_flags(coefficients, rho, vref):
    """Keep out-of-probe-domain and nonpositive-rho estimates, with explicit flags."""
    row=dict(estimated_abs_dV_exceeds_5pct_STL=bool(abs(coefficients[0])>.05*vref),
             volume_probe_limit_ml=.05*vref,translation_probe_limit_mm=5.)
    for i,axis in enumerate("xyz",1):
        row[f"estimated_abs_d{axis}_exceeds_5mm"]=bool(abs(coefficients[i])>5)
    for i,name in enumerate(PARAMETERS[4:]):
        total=rho[TISSUES[i]]+coefficients[4+i]
        row[f"estimated_total_{name}_ohm_m"]=total
        row[f"estimated_total_{name}_nonpositive"]=bool(total<=0)
    row["outside_geometric_probe_range"]=any(row[k] for k in row if "exceeds" in k)
    row["any_nonpositive_rho"]=any(row[k] for k in row if k.endswith("_nonpositive"))
    row["linear_extrapolation_or_nonphysical"]=row["outside_geometric_probe_range"] or row["any_nonpositive_rho"]
    row["clipping_applied"]=False
    return row


def derivatives(probes, scenario, shape, corrected):
    base, minus, plus = (probes[scenario, shape, "base", level] for level in (100, 95, 105))
    dv = plus.volume - minus.volume
    require(minus.volume < base.volume < plus.volume, "Unordered integrated volume probes")
    jv, kv = (plus.V-minus.V)/dv, (plus.Z-minus.Z)/dv
    js, ks = [], []
    for axis in "xyz":
        m, p = (probes[scenario, shape, axis+sign+"5", 100] for sign in "mp")
        volume_delta = p.volume - m.volume if corrected else 0
        js.append((p.V-m.V-jv*volume_delta)/10)
        ks.append((p.Z-m.Z-kv*volume_delta)/10)
    rho = base.rho[TISSUES]
    je = np.column_stack([jv, *js, -base.S[:, TISSUES]/rho**2])
    jt = np.r_[kv, ks, -base.K[TISSUES]/rho**2]
    return je, jt


def finite_delta(probes, scenario, shape, pose, level, je, jt, corrected):
    base = probes[scenario, shape, "base", 100]
    p = probes[scenario, shape, pose, level]
    y, z = p.V-base.V, p.Z-base.Z
    dv = p.volume - base.volume
    nominal = p.target - base.target
    truth = np.zeros(7)
    truth[0] = dv
    if pose != "base":
        truth[1+"xyz".index(pose[0])] = -5 if pose[1] == "m" else 5
        if corrected:
            y, z = y-je[:, 0]*dv, z-jt[0]*dv
            truth[0] = 0
    return y, z, truth, dv, nominal


def stacks(je, jt, eit_weights, tt_weight, current):
    yield "EIT208", je, eit_weights
    yield "EIT208_TT1", np.vstack([je, current*jt]), np.r_[eit_weights, tt_weight]


def analyze(probes, manifest, qc):
    rows = {k: [] for k in ("baseline_errors", "baseline_channels", "jacobian_errors", "jacobian_channels",
        "derivative_nonlinearity", "distinguishability", "pairwise_cosines", "unique_and_amplification",
        "tt_leakage", "finite_tt_volume", "cross_inversion", "geometry_to_tissue_leakage",
        "target_chain_effective_geometry", "target_chain_rho_leakage")}
    designs = {}
    current, vref = manifest["current_A"], qc["reference_STL_volume_ml"]
    for scenario in SCENARIOS:
        ref = probes[scenario, "STL", "base", 100]
        require(np.all(ref.V != 0) and ref.Z != 0, "Zero STL baseline: relative metric undefined; no implicit floor")
        steps = np.r_[.01*vref, [1., 1., 1.], .01*ref.rho[TISSUES]]
        metrics = {"raw": (np.ones(208), 1.), "relative_shared_STL_base": (1/np.abs(ref.V), 1/abs(current*ref.Z))}
        for shape in SHAPES:
            base = probes[scenario, shape, "base", 100]
            bj,bk=derivatives(probes,scenario,shape,False)
            mismatch=base.volume-ref.volume
            baseline_context=dict(scenario=scenario,shape=shape,candidate_integrated_volume_ml=base.volume,
                STL_integrated_volume_ml=ref.volume,common_nominal_volume_ml=vref,integrated_volume_mismatch_ml=mismatch,
                linear_volume_correction_TT_ohm=bk[0]*mismatch,linear_volume_correction_EIT_V_norm=float(np.linalg.norm(bj[:,0]*mismatch)))
            for correction in ("raw","linear_volume_aligned_to_STL_integrated"):
                candidate_v=base.V if correction=="raw" else base.V-bj[:,0]*mismatch
                candidate_z=base.Z if correction=="raw" else base.Z-bk[0]*mismatch
                for mode,(w,_) in metrics.items():
                    rows["baseline_errors"].append(dict(**baseline_context,correction=correction,stack="EIT208",metric=mode,
                        **compare_vectors(candidate_v,ref.V,w)))
                rows["baseline_errors"].append(dict(**baseline_context,correction=correction,stack="TT1",metric="raw_ohm",
                    candidate_Z_ohm=candidate_z,STL_Z_ohm=ref.Z,signed_error_ohm=candidate_z-ref.Z,
                    signed_relative_error=ratio(candidate_z-ref.Z,ref.Z),**compare_vectors([candidate_z],[ref.Z],np.ones(1))))
            for i in range(208):
                rows["baseline_channels"].append(dict(scenario=scenario, shape=shape, channel_1based=i+1,
                    candidate_V=base.V[i], STL_V=ref.V[i], signed_error_V=base.V[i]-ref.V[i],
                    signed_relative_error=ratio(base.V[i]-ref.V[i],ref.V[i]),shared_STL_weight=1/abs(ref.V[i]),
                    linear_volume_correction_V=bj[i,0]*mismatch,
                    signed_error_after_linear_volume_alignment_V=base.V[i]-bj[i,0]*mismatch-ref.V[i]))
        for corrected in (False, True):
            correction = "volume_corrected" if corrected else "raw"
            operators = {shape: derivatives(probes, scenario, shape, corrected) for shape in SHAPES}
            rje, rjt = operators["STL"]
            for shape in SHAPES:
                je, jt = operators[shape]
                base = probes[scenario, shape, "base", 100]
                common = dict(scenario=scenario, correction=correction, shape=shape)
                drift=None if corrected else volume_drift_slopes(probes,scenario,shape)
                for k, name in enumerate(PARAMETERS):
                    for mode, (w, _) in metrics.items():
                        rows["jacobian_errors"].append(dict(**common, parameter=name, parameter_unit=PARAMETER_UNITS[k],
                            stack="EIT208", metric=mode, **compare_vectors(je[:, k], rje[:, k], w)))
                    rows["jacobian_errors"].append(dict(**common, parameter=name, parameter_unit=PARAMETER_UNITS[k],
                        stack="TT1", metric="raw_ohm", candidate_derivative=jt[k], STL_derivative=rjt[k],
                        signed_error=jt[k]-rjt[k], signed_relative_error=ratio(jt[k]-rjt[k], rjt[k]),
                        **compare_vectors([jt[k]], [rjt[k]], np.ones(1))))
                    for i in range(208):
                        rows["jacobian_channels"].append(dict(**common, parameter=name, parameter_unit=PARAMETER_UNITS[k],
                            channel_1based=i+1, candidate_derivative_V=je[i,k], STL_derivative_V=rje[i,k],
                            signed_error_V=je[i,k]-rje[i,k], signed_relative_error=ratio(je[i,k]-rje[i,k], rje[i,k])))
                    rows["tt_leakage"].append(dict(**common, parameter=name, unit=PARAMETER_UNITS[k],
                        derivative_ohm_per_unit=jt[k], diagnostic_step=steps[k], signed_diagnostic_deltaZ_ohm=jt[k]*steps[k],
                        apparent_volume_ml_per_unit=ratio(jt[k],jt[0]),
                        apparent_volume_at_diagnostic_step_ml=ratio(jt[k]*steps[k], jt[0]),
                        interpretation="volume-only TT interpretation; nuisance rho is correction error, not measured physiology"))
                tt_design = design_diagnostics(jt[None, :], np.ones(1), steps,drift)
                designs[f"{scenario}/{correction}/{shape}/TT1/raw_ohm"] = tt_design
                rows["distinguishability"].append(dict(**common, stack="TT1", metric="raw_ohm",
                    **{k: v for k, v in tt_design.items() if k not in ("pairwise_cosines", "coordinate_resolution")}))
                for mode, (w, tw) in metrics.items():
                    for tissue_model, columns in (("S3_soft_lung_heart",[4,5,6]),("S2_soft_lung_fixed_heart",[4,5])):
                        projection, rank, tolerance = tissue_projection(je[:,columns],w)
                        downstream=downstream_identification(je[:,columns],projection,jt[columns],rank)
                        leakage = projection @ je[:,:4]
                        ktissue = jt[columns] @ leakage
                        effective = jt[:4] - ktissue
                        for k,name in enumerate(PARAMETERS[:4]):
                            rows["target_chain_effective_geometry"].append(dict(**common,metric=mode,tissue_model=tissue_model,
                                parameter=name,unit=PARAMETER_UNITS[k],tissue_rank=rank,tissue_unknowns=len(columns),
                                tissue_rank_tolerance=tolerance,Kg_ohm_per_unit=jt[k],
                                tissue_subtracted_ohm_per_unit=ktissue[k],Keff_ohm_per_unit=effective[k],
                                Keff_over_Kg=ratio(effective[k],jt[k]),diagnostic_step=steps[k],
                                raw_geometry_diagnostic_TT_ohm=jt[k]*steps[k],
                                tissue_subtracted_diagnostic_TT_ohm=ktissue[k]*steps[k],
                                effective_geometry_diagnostic_TT_ohm=effective[k]*steps[k],
                                denominator_abs_ohm_per_unit=abs(jt[k]),
                                **downstream,
                                ratio_status="undefined_zero_denominator" if jt[k]==0 else "inspect_absolute_values_no_near_zero_floor",
                                status="local_geometry_leakage_into_fixed_geometry_tissue_fit_not_measured_rho"))
                            for t,column in enumerate(columns):
                                rows["target_chain_rho_leakage"].append(dict(**common,metric=mode,tissue_model=tissue_model,
                                    geometry_parameter=name,geometry_unit=PARAMETER_UNITS[k],rho_parameter=PARAMETERS[column],
                                    E_ohm_m_per_geometry_unit=leakage[t,k],
                                    false_delta_rho_at_diagnostic_step_ohm_m=leakage[t,k]*steps[k]))
                    for stack, matrix, weights in stacks(je, jt, w, tw, current):
                        d = design_diagnostics(matrix, weights, steps,drift)
                        designs[f"{scenario}/{correction}/{shape}/{stack}/{mode}"] = d
                        context = dict(**common, stack=stack, metric=mode)
                        rows["distinguishability"].append(dict(**context,
                            **{k:v for k,v in d.items() if k not in ("pairwise_cosines", "coordinate_resolution", "amplification_definition")}))
                        for k, name in enumerate(PARAMETERS):
                            rows["unique_and_amplification"].append(dict(**context, parameter=name,
                                unique_residual_fraction=d["unique_residual_fraction"][k],
                                diagnostic_step=steps[k], physical_unit=PARAMETER_UNITS[k], full_rank=d["full_column_rank"],
                                diagnostic_amplification_per_unit_weighted_data_norm=d["pseudoinverse_row_norms"][k],
                                physical_diagnostic_amplification_per_unit_weighted_data_norm=d["physical_diagnostic_coordinate_row_norms"][k],
                                physical_amplification_per_unit_weighted_data_norm=d["physical_pseudoinverse_row_norms"][k],
                                amplification_status=d["amplification_status"]))
                            for m in range(k+1,7):
                                rows["pairwise_cosines"].append(dict(**context, parameter_a=name, parameter_b=PARAMETERS[m],
                                    cosine_uncentered=d["pairwise_cosines"][k,m]))
                # One-sided probes expose finite-step curvature; rho columns are
                # analytic sensitivities and have no new finite-rho FEM probes.
                for k, name in enumerate(PARAMETERS[:4]):
                    cases = (("base",95),("base",105)) if k == 0 else ((name+"m5",100),(name+"p5",100))
                    sides = []
                    for pose, level in cases:
                        y,z,truth,dv,nominal = finite_delta(probes,scenario,shape,pose,level,je,jt,corrected)
                        denominator = dv if k == 0 else truth[k]
                        sides.append((y/denominator,z/denominator))
                    for mode,(w,_) in metrics.items():
                        rows["derivative_nonlinearity"].append(dict(**common,parameter=name,metric=mode,
                            EIT_one_sided_disagreement_fraction=relnorm(w*(sides[1][0]-sides[0][0]),w*je[:,k]),
                            TT_minus_derivative=sides[0][1],TT_plus_derivative=sides[1][1],TT_central_derivative=jt[k],
                            TT_one_sided_disagreement_fraction=ratio(abs(sides[1][1]-sides[0][1]),abs(jt[k]))))
            # Full finite source probes, each minus its OWN source baseline.
            # Target predictions are differential about their OWN target base.
            for source_shape in SHAPES:
                source_je, source_jt = operators[source_shape]
                for pose, level in STATES[1:]:
                    y,z,truth,dv,nominal = finite_delta(probes,scenario,source_shape,pose,level,source_je,source_jt,corrected)
                    source_context = dict(scenario=scenario, correction=correction, source_shape=source_shape,
                        pose=pose,level=level, integrated_delta_volume_ml=dv,nominal_delta_volume_ml=nominal,
                        linear_truth_volume_ml=truth[0], source_deltaZ_ohm=z)
                    for target_shape in SHAPES:
                        je,jt = operators[target_shape]
                        context = dict(**source_context, target_shape=target_shape, same_shape_control=source_shape==target_shape)
                        apparent = ratio(z,jt[0])
                        rows["finite_tt_volume"].append(dict(**context, apparent_volume_ml=apparent,
                            error_vs_linear_truth_ml=None if apparent is None else apparent-truth[0],
                            error_vs_nominal_ml=None if apparent is None else apparent-nominal,
                            target_Kvol_ohm_per_ml=jt[0], status="conditional_volume_only_one_scalar_rank_at_most_one"))
                        for mode,(w,tw) in metrics.items():
                            for stack,matrix,weights in stacks(je,jt,w,tw,current):
                                data = y if stack=="EIT208" else np.r_[y,current*z]
                                fit = fit_linear(matrix,data,weights,steps)
                                coefficients = fit["coefficients"]
                                physical=coefficients.copy()
                                slopes=np.zeros(3) if corrected else volume_drift_slopes(probes,scenario,target_shape)
                                physical[0]+=slopes@coefficients[1:4]
                                row = dict(**context,metric=mode,stack=stack,
                                    **{k:v for k,v in fit.items() if k not in ("coefficients","residual","diagnostic_coordinates")},
                                    TT_predicted_deltaZ_ohm=float(jt@coefficients),TT_residual_ohm=float(z-jt@coefficients),
                                    EIT_residual_raw_V_norm=float(np.linalg.norm(y-je@coefficients)),
                                    EIT_residual_relative_to_probe=relnorm(y-je@coefficients,y),
                                    model_coordinate_recovered_V=coefficients[0],
                                    model_coordinate_true_V=truth[0]-slopes@truth[1:4],
                                    recovered_translation_volume_contribution_ml=float(slopes@coefficients[1:4]),
                                    raw_target_translation_volume_slopes_ml_per_mm=slopes,
                                    volume_coordinate_note="physical recovered_V = model volume coefficient + target raw volume drift dot recovered_xyz; corrected drift zero")
                                for k,name in enumerate(PARAMETERS):
                                    row[f"true_{name}"]=truth[k]
                                    row[f"recovered_{name}"]=physical[k]
                                    row[f"error_{name}"]=physical[k]-truth[k]
                                    row[f"error_in_diagnostic_units_{name}"]=(physical[k]-truth[k])/steps[k]
                                row["volume_error_vs_nominal_ml"]=physical[0]-nominal
                                row.update(estimate_flags(physical,ref.rho,vref))
                                row["model_coordinate_abs_dV_exceeds_5pct_STL"]=bool(abs(coefficients[0])>.05*vref)
                                rows["cross_inversion"].append(row)
                            # Target-path leakage: EIT fits ONLY tissue to a pure
                            # geometry signal, then those spurious rho corrections
                            # are propagated into the target TT volume estimate.
                            for tissue_model,columns in (("S3_soft_lung_heart",[4,5,6]),("S2_soft_lung_fixed_heart",[4,5])):
                                projection,rank,_=tissue_projection(je[:,columns],w)
                                downstream=downstream_identification(je[:,columns],projection,jt[columns],rank)
                                drho=np.zeros(3);drho[np.array(columns)-4]=projection@y
                                correction_z=float(jt[4:]@drho)
                                corrected_volume=ratio(z-correction_z,jt[0])
                                tissue_residual=y-je[:,4:]@drho
                                source_base=probes[scenario,source_shape,"base",100]
                                source_probe=probes[scenario,source_shape,pose,level]
                                row=dict(**context,metric=mode,tissue_model=tissue_model,
                                    raw_finite_TT_deltaZ_ohm=source_probe.Z-source_base.Z,
                                    volume_mismatch_subtracted_TT_ohm=source_probe.Z-source_base.Z-z,
                                    false_TT_tissue_correction_ohm=correction_z,
                                    remaining_TT_deltaZ_ohm=z-correction_z,
                                    apparent_volume_before_tissue_correction_ml=ratio(z,jt[0]),
                                    volume_after_false_correction_ml=corrected_volume,
                                    volume_error_vs_linear_truth_ml=None if corrected_volume is None else corrected_volume-truth[0],
                                    volume_error_vs_nominal_ml=None if corrected_volume is None else corrected_volume-nominal,
                                    target_Kv_ohm_per_ml=jt[0],
                                    remaining_TT_minus_true_volume_prediction_ohm=z-correction_z-jt[0]*truth[0],
                                    tissue_fit_residual_fraction=relnorm(w*tissue_residual,w*y),
                                    tissue_fit_residual_raw_V_norm=float(np.linalg.norm(tissue_residual)),
                                    tissue_fit_rank=rank,tissue_unknowns=len(columns),
                                    **downstream,
                                    true_delta_rho=0.,status="spurious_tissue_estimate_from_computed_geometry_only_probe")
                                for k,name in enumerate(PARAMETERS[4:]):
                                    row["false_delta_"+name]=drho[k]
                                    row["false_delta_"+name+"_pct"]=100*drho[k]/ref.rho[TISSUES[k]]
                                # Only the volume is inferred from one TT scalar;
                                # no position estimate is silently invented here.
                                row["estimated_abs_dV_exceeds_5pct_STL"]=None if corrected_volume is None else abs(corrected_volume)>.05*vref
                                row["volume_probe_limit_ml"]=.05*vref
                                row["clipping_applied"]=False
                                for k,name in enumerate(PARAMETERS[4:]):
                                    row["estimated_total_"+name+"_ohm_m"]=ref.rho[TISSUES[k]]+drho[k]
                                    row["estimated_total_"+name+"_nonpositive"]=ref.rho[TISSUES[k]]+drho[k]<=0
                                rows["geometry_to_tissue_leakage"].append(row)
    return rows, designs


def compact_summary(rows, qc):
    """Select explicit comparable numbers, never manufacture acceptance limits."""
    findings = {}
    compact=[]
    lines = ["Условное сравнение STL, G1 и L1 при общих rho (79505 Гц).",
             "Цель: PulmoVista rho/Δrho -> тканевые поправки -> геометрия по ТТРКГ и одной КТ.",
             "Приведённая ЭИТ/совместная геометрическая инверсия — только диагностика.", ""]
    number = lambda v: "не определено" if v is None or not np.isfinite(v) else f"{v:.5g}"
    for scenario in SCENARIOS:
        results = {}
        lines.append(scenario + ":")
        for shape in SHAPES:
            base = next(r for r in rows["baseline_errors"] if r["scenario"]==scenario and r["shape"]==shape and r["stack"]=="TT1" and r["correction"]=="raw")
            eit = next(r for r in rows["baseline_errors"] if r["scenario"]==scenario and r["shape"]==shape and
                       r["stack"]=="EIT208" and r["metric"]=="raw" and r["correction"]=="raw")
            d = next(r for r in rows["distinguishability"] if r["scenario"]==scenario and r["shape"]==shape and
                     r["correction"]=="volume_corrected" and r["stack"]=="EIT208" and r["metric"]=="relative_shared_STL_base")
            recovery = [r for r in rows["cross_inversion"] if r["scenario"]==scenario and r["source_shape"]=="STL" and
                        r["target_shape"]==shape and r["correction"]=="volume_corrected" and
                        r["stack"]=="EIT208" and r["metric"]=="relative_shared_STL_base"]
            tissue = [r for r in rows["geometry_to_tissue_leakage"] if r["scenario"]==scenario and
                      r["source_shape"]=="STL" and r["target_shape"]==shape and r["correction"]=="volume_corrected" and
                      r["metric"]=="relative_shared_STL_base" and r["tissue_model"]=="S3_soft_lung_heart"]
            tt = [r for r in rows["finite_tt_volume"] if r["scenario"]==scenario and r["source_shape"]=="STL" and
                  r["target_shape"]==shape and r["correction"]=="volume_corrected"]
            worst = {k: max(recovery,key=lambda r:abs(r["error_"+k])) for k in PARAMETERS}
            finite = [r for r in tissue if r["volume_error_vs_linear_truth_ml"] is not None]
            results[shape] = dict(baseline_TT_signed_error_ohm=base["signed_error_ohm"],
                baseline_TT_signed_relative_error=base["signed_relative_error"],
                baseline_EIT_raw_norm_relative_error=eit["norm_relative_error"],
                base_integrated_volume_ml=base["candidate_integrated_volume_ml"],
                base_integrated_volume_difference_from_STL_ml=base["integrated_volume_mismatch_ml"],
                linear_baseline_volume_correction_TT_ohm=base["linear_volume_correction_TT_ohm"],
                cross_inverse_flagged_cases=sum(r["linear_extrapolation_or_nonphysical"] for r in recovery),
                cross_inverse_cases=len(recovery),
                corrected_EIT_shared_relative_rank=d["rank"],normalized_condition=d["normalized_condition"],
                max_abs_cross_error={k:abs(r["error_"+k]) for k,r in worst.items()},
                worst_cross_volume_case=worst["V"],
                max_abs_TT_volume_only_error_ml=max((abs(r["error_vs_linear_truth_ml"]) for r in tt
                                                   if r["error_vs_linear_truth_ml"] is not None),default=None),
                worst_sequential_leakage=max(finite,key=lambda r:abs(r["volume_error_vs_linear_truth_ml"])) if finite else None)
            chain=[r for r in rows["target_chain_effective_geometry"] if r["scenario"]==scenario and r["shape"]==shape and
                   r["correction"]=="volume_corrected" and r["metric"]=="relative_shared_STL_base" and r["parameter"]=="V"]
            results[shape]["target_chain_volume_sensitivity"]=chain
            small=dict(scenario=scenario,shape=shape,correction="volume_corrected")
            for metric,short in (("raw","raw"),("relative_shared_STL_base","relative")):
                diagnostic=next(r for r in rows["distinguishability"] if r["scenario"]==scenario and r["shape"]==shape and
                    r["correction"]=="volume_corrected" and r["stack"]=="EIT208" and r["metric"]==metric)
                cosine=next(r["cosine_uncentered"] for r in rows["pairwise_cosines"] if r["scenario"]==scenario and
                    r["shape"]==shape and r["correction"]=="volume_corrected" and r["stack"]=="EIT208" and
                    r["metric"]==metric and r["parameter_a"]=="V" and r["parameter_b"]=="rho_heart")
                small["condition_"+short]=diagnostic["normalized_condition"]
                small["unique_V_after_all_"+short]=diagnostic["unique_residual_fraction"][0]
                small["unique_rhoheart_after_all_"+short]=diagnostic["unique_residual_fraction"][6]
                small["cos_V_rhoheart_"+short]=cosine
                small["unique_V_after_rhoheart_only_"+short]=float(np.sqrt(max(0,1-cosine*cosine))) if np.isfinite(cosine) else None
                for r in rows["target_chain_effective_geometry"]:
                    if (r["scenario"],r["shape"],r["correction"],r["metric"],r["parameter"])==(scenario,shape,"volume_corrected",metric,"V"):
                        kind="S3" if r["tissue_model"]=="S3_soft_lung_heart" else "S2"
                        small[f"KeffV_over_KgV_{kind}_{short}"]=r["Keff_over_Kg"]
                        small[f"KeffV_mohm_per_ml_{kind}_{short}"]=1000*r["Keff_ohm_per_unit"]
                        small["KgV_mohm_per_ml"]=1000*r["Kg_ohm_per_unit"]
            compact.append(small)
            results[shape]["compact_diagnostics"]=small
            if shape != "STL":
                lines.append(f"  {shape}: база TT {number(base['signed_error_ohm'])} Ом "
                    f"({number(100*base['signed_relative_error'])}%); ошибка нормы ЭИТ {number(100*eit['norm_relative_error'])}%.")
            lines.append(f"  STL->{shape}, corrected ЭИТ relative: rank {d['rank']}/7, cond {number(d['normalized_condition'])}; "
                f"max |ошибка ΔV| {number(abs(worst['V']['error_V']))} мл, "
                f"max |ошибка xyz| {[number(abs(worst[k]['error_'+k])) for k in ('x','y','z')]} мм.")
            lines.append(f"  cond raw/relative: {number(small['condition_raw'])}/{number(small['condition_relative'])}; "
                f"unique V после всех: {number(small['unique_V_after_all_raw'])}/{number(small['unique_V_after_all_relative'])}; "
                f"unique ρheart после всех: {number(small['unique_rhoheart_after_all_raw'])}/{number(small['unique_rhoheart_after_all_relative'])}; "
                f"cos(V,ρheart): {number(small['cos_V_rhoheart_raw'])}/{number(small['cos_V_rhoheart_relative'])}.")
            if finite:
                worst_leak = results[shape]["worst_sequential_leakage"]
                lines.append(f"  ЭИТ только ткани -> TT: max |ошибка объёма| "
                    f"{number(abs(worst_leak['volume_error_vs_linear_truth_ml']))} мл "
                    f"({worst_leak['pose']}, v{worst_leak['level']}); "
                    f"ложные Δrho soft/lung/heart "
                    f"{[number(worst_leak['false_delta_'+k]) for k in PARAMETERS[4:]]} Ом·м.")
            lines.append("  Keff,V/Kg,V (S3; S2): "+"; ".join(number(r["Keff_over_Kg"]) for r in chain)+
                "; Keff,V (мОм/мл): "+"; ".join(number(1000*r["Keff_ohm_per_unit"]) for r in chain)+".")
        findings[scenario] = results
    shifts = [abs(r["integrated_delta_from_base_ml"]) for r in qc["integrated_volume_qc"] if r["pose"]!="base"]
    lines += ["", f"Максимальная численная добавка объёма при номинально фиксированном V: {number(max(shifts))} мл.",
        "Базовые integrated V и диагностическое линейное выравнивание к STL сохранены в baseline_errors; точный remesh не выполнялся.",
        f"Флаги выхода за пробы/неположительных rho в перекрёстных инверсиях: "
        f"{sum(r['linear_extrapolation_or_nonphysical'] for r in rows['cross_inversion'])}/{len(rows['cross_inversion'])}; оценки не ограничивались clipping.",
        "ТТРКГ: один скаляр, rank<=1. Нулевая невязка не означает восстановления четырёх геометрических параметров.",
        "Все максимумы относятся только к восьми конечным пробам; исправление объёмной добавки линейное.",
        "Ранг и cond — алгебраическая диагностика. Уровень шума не задан; усиление сохранено на единицу нормы данных.",
        "±5 мм — сценарии, ±5% — изотропное изменение целого сердца, не УО. Сходимость FEM и точность ткани/функции не валидированы."]
    rows["compact_diagnostics"]=compact
    return findings, "\n".join(lines) + "\n"


def figures(output, rows):
    # Keep even Matplotlib's font cache inside the explicitly permitted output.
    os.environ["MPLCONFIGDIR"] = str(output / ".matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"STL":"#30343b", "G1":"#b55334", "L1":"#277d91"}
    labels = ["V", "x", "y", "z", "ρsoft", "ρlung", "ρheart"]
    fig, axes = plt.subplots(2,2,figsize=(12,7.6),layout="constrained")
    for c, scenario in enumerate(SCENARIOS):
        matrix = np.array([[next(r["norm_relative_error"] for r in rows["jacobian_errors"] if
            r["scenario"]==scenario and r["shape"]==h and r["parameter"]==p and
            r["correction"]=="volume_corrected" and r["stack"]=="EIT208" and r["metric"]=="relative_shared_STL_base")
            for p in PARAMETERS] for h in ("G1","L1")])
        im = axes[0,c].imshow(100*matrix,aspect="auto",cmap="YlOrRd",vmin=0)
        for i in range(2):
            for j in range(7):
                axes[0,c].text(j,i,f"{100*matrix[i,j]:.1f}",ha="center",va="center",fontsize=9)
        axes[0,c].set(xticks=range(7),xticklabels=labels,yticks=[0,1],yticklabels=["G1","L1"],
            title=scenario+": ошибка J ЭИТ относительно STL, %")
        fig.colorbar(im,ax=axes[0,c],shrink=.6)
        for hidx,h in enumerate(SHAPES):
            values = [1000*next(r["signed_diagnostic_deltaZ_ohm"] for r in rows["tt_leakage"] if
                      r["scenario"]==scenario and r["shape"]==h and r["parameter"]==p and r["correction"]=="volume_corrected")
                      for p in PARAMETERS]
            axes[1,c].bar(np.arange(7)+(hidx-1)*.25,values,.24,label=h,color=colors[h])
        axes[1,c].axhline(0,color="grey",lw=.7)
        axes[1,c].set(xticks=range(7),xticklabels=labels,ylabel="Знаковый отклик TT, мОм",title="1% V(STL), 1 мм, 1% базового ρ")
        axes[1,c].legend(ncol=3,fontsize=9)
    fig.suptitle("Одинаковые ρ для STL/G1/L1; поправка численного объёма переносов\nЭИТ: общие веса базы STL; шаги диагностические",fontsize=13)
    fig.savefig(output/"01_jacobians_and_TT.png",dpi=170);plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(11,4.8),layout="constrained")
    for c,scenario in enumerate(SCENARIOS):
        for tidx,(tissue_model,label) in enumerate(((None,"Kg исходный"),("S2_soft_lung_fixed_heart","Keff S2"),("S3_soft_lung_heart","Keff S3"))):
            selected=[next(r for r in rows["target_chain_effective_geometry"] if r["scenario"]==scenario and r["shape"]==h and
                r["metric"]=="relative_shared_STL_base" and r["correction"]=="volume_corrected" and r["parameter"]=="V" and
                r["tissue_model"]==(tissue_model or "S3_soft_lung_heart")) for h in SHAPES]
            values=[1000*r["Kg_ohm_per_unit" if tissue_model is None else "Keff_ohm_per_unit"] for r in selected]
            bars=axes[c].bar(np.arange(3)+(tidx-1)*.26,values,.25,label=label,color=("#6d7780","#67a8a1","#be6948")[tidx])
            axes[c].bar_label(bars,labels=[f"{v:+.4g}" for v in values],padding=3,fontsize=8,rotation=45)
        axes[c].axhline(0,color="grey",lw=.7);axes[c].margins(y=.3)
        axes[c].set(title=scenario,ylabel="Производная TT по объёму, мОм/мл",xticks=range(3),xticklabels=SHAPES)
        axes[c].legend(fontsize=9)
    fig.suptitle("Целевой путь: утечка геометрии в тканевые ρ изменяет чувствительность TT\nKeff = Kg − Krho · (W Jrho)⁺ W G; общие relative-веса STL; шум не задан",fontsize=12)
    fig.savefig(output/"02_target_chain_volume_sensitivity.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7.8),layout="constrained")
    cases=STATES[1:]
    case_labels=["V−5%","V+5%","x−5","x+5","y−5","y+5","z−5","z+5"]
    for c,scenario in enumerate(SCENARIOS):
        for hidx,h in enumerate(SHAPES):
            for row_index,table,field in ((0,"cross_inversion","error_V"),(1,"geometry_to_tissue_leakage","volume_error_vs_linear_truth_ml")):
                values=[]
                for pose,level in cases:
                    choices=[r for r in rows[table] if r["scenario"]==scenario and r["source_shape"]=="STL" and
                             r["target_shape"]==h and r["pose"]==pose and r["level"]==level and
                             r["metric"]=="relative_shared_STL_base" and r["correction"]=="volume_corrected" and
                             (r["stack"]=="EIT208" if table=="cross_inversion" else r["tissue_model"]=="S3_soft_lung_heart")]
                    values.append(np.nan if choices[0][field] is None else choices[0][field])
                axes[row_index,c].bar(np.arange(8)+(hidx-1)*.25,values,.24,color=colors[h],label=h)
                axes[row_index,c].axhline(0,color="grey",lw=.7)
                axes[row_index,c].set(xticks=range(8),xticklabels=case_labels,ylabel="Ошибка ΔV, мл")
        axes[0,c].set_title(scenario+": ЭИТ, 7 неизвестных (диагностика)")
        axes[1,c].set_title("ЭИТ S3 (только ткани) -> поправка TT -> объём")
        axes[0,c].legend(ncol=3,fontsize=9)
    fig.suptitle("Конечные STL-пробы -> модели STL/G1/L1; собственная база вычтена\nОбщие relative-веса; corrected переносы; STL->STL — внутренний контроль",fontsize=12)
    fig.savefig(output/"03_finite_probe_volume_errors.png",dpi=170);plt.close(fig)
    return matplotlib.__version__


def self_test():
    """Independent limiting cases: signs, units, nullspace and known recovery."""
    checks=[]
    a=np.array([1.,-2.,3.]);w=np.array([1.,2.,.5])
    cmp=compare_vectors(-2*a,a,w)
    close(cmp["norm_relative_error"],3.,"Antiparallel error sign")
    close(cmp["cosine_uncentered"],-1.,"Cosine must not be centered")
    close(cmp["scale_candidate_on_reference"],-2.,"Signed scale")
    close(cmp["angle_deg"],180.,"Antiparallel angle")
    checks.append("signed_uncentered_cosine_scale_and_angle")
    rng=np.random.default_rng(20261028)
    q,_=np.linalg.qr(rng.normal(size=(208,7)))
    steps=np.array([5.8,1.,1.,1.,.03,.18,.24])
    weighted=q/steps
    d=design_diagnostics(weighted,np.ones(208),steps)
    require(d["rank"]==7,"Orthogonal full-rank control")
    close(d["unique_residual_fraction"],np.ones(7),"Orthogonal unique fractions")
    close(d["pseudoinverse_row_norms"],np.ones(7),"Known inverse amplification")
    close(d["worst_coordinate_amplification_per_unit_data_norm"],1.,"Known worst amplification")
    scalar=design_diagnostics(np.arange(1.,8.)[None,:],np.ones(1),steps)
    require(scalar["rank"]==1 and scalar["nullity"]==6 and scalar["normalized_condition"] is None,
            "One scalar must not report a finite full-column condition")
    close(scalar["unique_residual_fraction"],np.zeros(7),"Scalar no unique parameter",atol=1e-12)
    require(scalar["worst_coordinate_amplification_per_unit_data_norm"] is None,"Nullspace amplification unbounded")
    deficient=weighted.copy();deficient[:,6]=deficient[:,0]
    dd=design_diagnostics(deficient,np.ones(208),steps)
    require(dd["rank"]==6 and not dd["full_column_rank"],"Collinear nuisance must reduce rank")
    close(np.array(dd["unique_residual_fraction"])[[0,6]],[0,0],"Collinear unique residual",atol=1e-12)
    zero=design_diagnostics(np.zeros((208,7)),np.ones(208),steps)
    require(zero["rank"]==0 and zero["nullity"]==7,"Zero sensitivity design")
    checks.append("rank_unique_residual_and_noise_amplification_limiting_cases")
    truth=np.array([20.,-2.,4.,1.,.05,-.1,.2])
    recovered=fit_linear(weighted,weighted@truth,np.ones(208),steps)
    close(recovered["coefficients"],truth,"Known linear inverse")
    scale=np.geomspace(.5,2,208)
    recovered_w=fit_linear(weighted,weighted@truth,scale,steps)
    close(recovered_w["coefficients"],truth,"Weighted known inverse")
    checks.append("seven_parameter_recovery_in_physical_units")
    rho=np.array([3.,18.,19.,24.,23.])
    sens=rng.normal(size=(208,5));delta=1e-5
    for i in TISSUES:
        numeric=sens[:,i]*(1/(rho[i]+delta)-1/(rho[i]-delta))/(2*delta)
        close(numeric,-sens[:,i]/rho[i]**2,"sigma-to-rho chain rule",atol=1e-9)
    checks.append("rho_chain_rule_sign_and_units")
    tissue=q[:,:3]
    expected_E=np.arange(12.).reshape(3,4)/10
    projection,rank,_=tissue_projection(tissue,scale)
    close(projection@tissue@expected_E,expected_E,"Tissue leakage operator recovery")
    krho=np.array([.2,-.4,.6]);kg=krho@expected_E
    close(kg-krho@projection@tissue@expected_E,np.zeros(4),"Fully absorbed geometry limiting case")
    require(rank==3,"S3 tissue rank")
    identical=np.column_stack([q[:,0],q[:,0]])
    projection,rank,_=tissue_projection(identical,np.ones(208))
    require(not downstream_identification(identical,projection,np.array([1.,-1.]),rank)["downstream_TT_correction_identified"],
            "EIT-null tissue must flag unidentifiable TT correction")
    require(downstream_identification(identical,projection,np.ones(2),rank)["downstream_TT_correction_identified"],
            "Rank-deficient tissue can have identified TT functional")
    flags=estimate_flags(np.array([30.,6.,-6.,0.,-4.,0.,0.]),rho,580.)
    require(flags["estimated_abs_dV_exceeds_5pct_STL"] and flags["estimated_abs_dx_exceeds_5mm"] and
            flags["estimated_abs_dy_exceeds_5mm"] and flags["estimated_total_rho_soft_nonpositive"] and
            not flags["clipping_applied"],"Extrapolation flags must not clip")
    checks.append("two_stage_tissue_leakage_and_unclipped_domain_flags")
    # Synthetic affine forward law with deliberately different form baselines,
    # numerical volume drift on translations and NONzero tissue sensitivities.
    probes={}
    je=q*np.array([.02,.03,.04,.05,.6,.7,.8])
    jt=np.array([.01,.02,-.03,.04,.2,-.3,.4])
    s=np.zeros((208,5));s[:,TISSUES]=-je[:,4:]*rho[TISSUES]**2
    k=np.zeros(5);k[TISSUES]=-jt[4:]*rho[TISSUES]**2
    for scenario in SCENARIOS:
        for hidx,h in enumerate(SHAPES):
            baseV=np.linspace(1.,2.,208)+hidx*17.
            baseZ=90.+hidx*13.
            for pose,level in STATES:
                x=np.zeros(7);x[0]=580*(level/100-1)
                if pose!="base":
                    x[1+"xyz".index(pose[0])]=-5 if pose[1]=="m" else 5
                    x[0]=.2 if pose[1]=="m" else -.3
                probes[scenario,h,pose,level]=Probe(baseV+je@x,s,baseZ+jt@x,k,rho,580+x[0],580*level/100,{})
    for h in SHAPES:
        corrected_j,corrected_k=derivatives(probes,"diagnostic4",h,True)
        close(corrected_j,je,"Volume correction must remove drift from Jxyz",atol=1e-10)
        close(corrected_k,jt,"TT volume correction",atol=1e-10)
        for pose,level in STATES[1:]:
            y,z,true,_,_=finite_delta(probes,"diagnostic4","STL",pose,level,je,jt,True)
            fit=fit_linear(corrected_j,y,np.ones(208),steps)
            close(fit["coefficients"],true,"Differential cross-shape recovery with distinct bases",atol=1e-9)
            close(z,jt@true,"Corrected finite TT probe",atol=1e-10)
    raw_j,_=derivatives(probes,"diagnostic4","STL",False)
    require(np.linalg.norm(raw_j[:,1]-je[:,1])>1e-5,"Raw must retain volume drift")
    for pose,level in STATES[1:]:
        y,_,true,_,_=finite_delta(probes,"diagnostic4","STL",pose,level,je,jt,False)
        fit=fit_linear(raw_j,y,np.ones(208),steps)
        coefficients=fit["coefficients"].copy()
        coefficients[0]+=volume_drift_slopes(probes,"diagnostic4","STL")@coefficients[1:4]
        close(coefficients,true,"Raw inverse needs physical volume coordinate transformation",atol=1e-9)
    checks.append("finite_probe_differencing_and_volume_mismatch_correction")
    synthetic_qc=dict(reference_STL_volume_ml=580.,integrated_volume_qc=[
        dict(pose=pose,integrated_delta_from_base_ml=p.volume-580.) for (_,_,pose,_),p in probes.items()])
    all_rows,all_designs=analyze(probes,dict(current_A=.007234),synthetic_qc)
    require(len(all_rows["cross_inversion"])==1152,"Complete cross-inversion coverage")
    require(max(abs(r["error_V"]) for r in all_rows["cross_inversion"])<1e-8,
            "Raw and corrected physical cross-volumes on known affine forward law")
    require(max(abs(r["recovered_rho_soft"]) for r in all_rows["cross_inversion"])<1e-8,
            "No spurious tissue estimate in exact seven-parameter geometry inverse")
    findings,_=compact_summary(all_rows,synthetic_qc)
    json.dumps(jsonable(dict(rows=all_rows,designs=all_designs,findings=findings)),allow_nan=False)
    checks.append("end_to_end_1152_cross_cases_shared_weights_and_strict_JSON")
    # Finite nonlinear volume probes must retain same-shape error, not be
    # replaced by J*known_step (which would create a tautological inverse test).
    for level in (95,105):
        probe=probes["diagnostic4","STL","base",level]
        probe.V=probe.V+je[:,0]*2.
    jn,kn=derivatives(probes,"diagnostic4","STL",True)
    y,_,true,_,_=finite_delta(probes,"diagnostic4","STL","base",105,jn,kn,True)
    result=fit_linear(jn,y,np.ones(208),steps)
    close(result["coefficients"][0]-true[0],2.,"Finite nonlinear same-shape control",atol=1e-9)
    checks.append("same_shape_finite_nonlinearity_is_not_erased")
    return dict(status="passed",checks=checks,scope="Algebra and synthetic probes only; no physical/FEM validation")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,default=LOCAL/"shape_electrical_study_v1.local.json")
    parser.add_argument("--output",type=Path,default=OUTPUT)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only",action="store_true")
    mode.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    checks=self_test()
    if args.self_test:
        print(json.dumps(checks,ensure_ascii=False))
        return 0
    manifest_path=args.manifest.resolve()
    if not manifest_path.exists():
        print(json.dumps(dict(status="waiting_for_manifest",path=str(manifest_path),self_test=checks),ensure_ascii=False))
        return 2
    sources=Sources()
    manifest,probes,qc=load_inputs(manifest_path,sources)
    print(json.dumps(dict(status=qc["status"],available=qc["available"],expected=qc["expected"],
        missing_count=len(qc["missing"]),first_missing=[r["key"] for r in qc["missing"][:3]],
        common_background_controls=[{k:r[k] for k in ("shape","array_equal","selected_tetrahedra")}
                                    for r in qc["common_background_controls"]]),ensure_ascii=False))
    if qc["missing"]:
        return 2
    if args.check_only:
        return 0
    output=args.output.resolve()
    require(not output.exists() or not any(output.iterdir()),"Output is not empty; refusing to overwrite previous run")
    rows,designs=analyze(probes,manifest,qc)
    findings,summary=compact_summary(rows,qc)
    sources.unchanged()
    output.mkdir(parents=True,exist_ok=True)
    # Exclusive claim prevents two parallel analyzers writing into the same run.
    write_json(output/"run_started.local.json",dict(started_UTC=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=sources.add(manifest_path),implementation_sha256=sources.add(__file__)))
    try:
        rows["integrated_volume_qc"]=qc["integrated_volume_qc"]
        for name,data in rows.items():
            write_csv(output/(name+".csv"),data)
        matplotlib_version=figures(output,rows)
        with (output/"summary_ru.txt").open("x",encoding="utf-8") as stream:
            stream.write(summary)
        sources.unchanged(rehash=True)
        output_files=[p.name for p in sorted(output.iterdir()) if p.is_file()]
        result=dict(schema="heart_shape_electrical_comparison_v1",status="conditional_computational_analysis_completed",
            validation_status="exploratory_hypothesis_not_validated",clinical_validation=False,
            input_qc=qc,settings={k:manifest[k] for k in SETTINGS},scenario_rho_ohm_m=manifest["scenarios"],
            parameter_order=PARAMETERS,parameter_units=PARAMETER_UNITS,material_order=MATERIALS,
            derivative_units=dict(EIT="V per physical parameter unit",TT="ohm per physical parameter unit"),
            primary_target="PulmoVista baseline rho and synchronous delta_rho -> tissue-corrected TT geometry with one CT; EIT/joint geometry is diagnostic only",
            formulas=dict(volume="(F_v105-F_v095)/(integrated_heart_ml_v105-integrated_heart_ml_v095)",
                baseline_volume_alignment="F_shape_aligned=F_shape_base-JV_shape*(V_integrated_shape_base-V_integrated_STL_base); first order diagnostic, no remesh",
                position_raw="(F_plus5-F_minus5)/10 mm",
                position_corrected="[F_plus5-F_minus5-J_V*(integrated_volume_plus5-integrated_volume_minus5)]/10 mm",
                rho="J_rho=-J_sigma/rho0^2; only soft/lung/heart free, bone/fat fixed",
                finite_probe="F_source_probe-F_source_base; corrected translations subtract source_JV*(source_integrated_deltaV)",
                target_prediction="F_target_base+J_target*delta_p minus F_target_base; no baseline offset in differential inverse",
                comparison="candidate-reference; signed projection=(candidate-reference).reference/||reference||^2; scale=candidate.reference/||reference||^2; cos without centering",
                relative_weights="W_EIT=1/abs(V_STL_base), W_TT_joint=1/abs(current_A*Z_STL_base); same for all forms in one rho scenario",
                joint="[V_EIT;current_A*Z_TT] in volts BEFORE weighting; no fitted gain",
                coordinates="D=diag(0.01*nominal_original_STL_volume_ml,1mm,1mm,1mm,0.01*rho_soft,0.01*rho_lung,0.01*rho_heart)",
                inverse="delta_p=D*pinv(W*J*D)*W*(source_probe-source_base); rank deficient output is minimum diagnostic-coordinate norm only",
                raw_volume_coordinate="c_target=(integrated_Vplus5-integrated_Vminus5)/10mm for each axis; physical recovered_V=model_coefficient_V+c_target.dot(recovered_xyz); corrected design c=0",
                rank="SVD weighted column-normalized J; tol=max(shape)*eps*s_max; separate SVD of W*J*D for inversion/amplification",
                unique="norm(unit_column - other_normalized_columns*pinv(other_normalized_columns)*unit_column)",
                noise_amplification="row norms and operator norm of pinv(W*J*D) per unit weighted data norm, not a measured noise SD; unidentified directions unbounded",
                TT_volume="deltaV_apparent=deltaZ/K_V; nuisance leakage=K_parameter/K_V; if K_V=0 undefined",
                target_path_leakage="false_delta_rho=pinv(W*Jrho)*W*EIT_geometry_delta; TT_corrected=deltaZ-Krho*false_delta_rho; then divide by K_V; S3 main, S2 soft/lung fixed heart control",
                target_effective_geometry="E=pinv(W*Jrho)*W*G; Keff=Kg-Krho*E, where G and Kg have V,x,y,z columns; report signed ratio and absolute derivatives with explicit diagnostic steps"),
            findings=findings,summary_ru=summary,limitations=LIMITATIONS,preparer_limits=manifest.get("limits",[]),
            numerical_self_checks=checks,designs=designs,
            author_reported_independent_review={**REPORTED_REVIEW,
                "audited_version_is_final_version":sources.add(__file__)==REPORTED_REVIEW["audited_analysis_sha256"],
                "later_changes_covered_by_that_review":False},
            results={k:v for k,v in rows.items() if k not in ("baseline_channels","jacobian_channels","integrated_volume_qc")},
            source_hashes=sources.records,output_files=output_files+["analysis.local.json"],
            source_content_rehashed_after_analysis=True,
            output_sha256={name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in output_files},
            runtime=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib_version,platform=platform.platform()))
        write_json(output/"analysis.local.json",result)
    except Exception as exc:
        write_json(output/"analysis_failure.local.json",dict(status="failed_incomplete_output",error=repr(exc),source_hashes=sources.records))
        raise
    print(summary)
    print(json.dumps(dict(status=result["status"],output=str(output)),ensure_ascii=False))
    return 0


if __name__=="__main__":
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
