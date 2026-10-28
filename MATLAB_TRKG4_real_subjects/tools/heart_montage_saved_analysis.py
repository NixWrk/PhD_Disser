"""Audit and analyze saved heart_montage_study_v1 MAT results; never run FEM.

Usage: --manifest study.local.json --output NEW_DIRECTORY
       --manifest study.local.json --check-only
       --self-test

Full analysis requires the matched 720-state grid. Check-only accepts immutable
558-state q64 / 162-state q256 manifests and missing result files. It writes
nothing, returning progress AND ready small-step datasets/J matrices as JSON on
stdout for adaptive FEM preparation. A ready group requires all three montages
and all nine base/small geometry states; it is not a convergence certificate.

Outputs are local (contain provenance paths). Coordinates are integrated ml,
LPS translations in mm and effective rho in ohm*m. Jrho = -Ktt/rho**2. Single
lung; soft/lung correction uncertainty stays unknown, not physiologically zero.
Alternative montages are a hypothetical design, not simultaneous measurements.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.io import loadmat, savemat

import heart_montage_analysis as algebra
from heart_shape_electrical_comparison import Sources, read_json, write_csv, write_json


ROOT = Path(__file__).resolve().parents[2]
MATERIALS = ["soft_background", "lung", "bone", "heart", "fat"]
ORDER = list(algebra.HEART + algebra.TISSUE)
FLAGS = dict(clinical_validation=False, physical_validation=False, no_AI_training=True,
             stack_kind="alternative_montages", simultaneous_measurements=False,
             measured_noise=None, SV_EF_estimated=False,
             mesh_convergence_proven=False, smallest_step_converged=None)
AMPLITUDE_BASIS = ("Diagnostic coordinate scales only: 1% nominal reference STL volume, "
                   "1 mm per axis, 1% scenario rhoH/rhoS/rhoL; not measured amplitudes, "
                   "error bounds, SDs, or a noise distribution")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def real(value, length=None):
    a = np.asarray(value)
    require(not np.iscomplexobj(a), "Complex data cannot be silently truncated")
    a = np.asarray(a, dtype=float).reshape(-1)
    require(np.all(np.isfinite(a)), "Nonfinite saved data")
    require(length is None or a.size == length, "Unexpected saved array size")
    return a


def same(a, b, message):
    a, b = real(a), real(b)
    require(a.shape == b.shape and np.allclose(a, b, rtol=1e-11, atol=1e-12), message)


def source_path(value, parent):
    """Manifest-relative paths; historical provenance also used repo-relative paths."""
    p = Path(value)
    if p.is_absolute():
        return p.resolve()
    candidates = {(parent / p).resolve(), (ROOT / p).resolve()}
    existing = [v for v in candidates if v.exists()]
    require(len(existing) == 1, f"Missing/ambiguous relative source: {value}")
    return existing[0]


def digest_row(row):
    blob = json.dumps(algebra.jsonable(row), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def number_id(v):
    return ("p" if v > 0 else "m") + f"{abs(v):g}".replace(".", "d")


def probe_key(p):
    return p["kind"], p["axis"], float(p["signed_step"])


def expected_probes(geometry, levels):
    probes = {p["id"]: p for p in geometry["probes"] if p["quadrature"] in levels}
    if 64 in levels:
        for shape in geometry["shapes"]:
            base = probes[f"{shape}_base_q64"]
            for step in (-5., -1., 1., 5.):
                p = dict(base, id=f"{shape}_rhoH{number_id(step)}_q64", kind="rho",
                         axis="rho_heart", signed_step=step)
                probes[p["id"]] = p
    return probes


def validate_probe(row, expected, scenario_rho):
    require(row["probe"] == expected, "Record geometry differs from bound geometry manifest")
    require(row["format"] == "electrical" and row["probe_index"] is None,
            "Only complete electrical MAT states are supported")
    rho = real(scenario_rho, 5).copy()
    if expected["kind"] in ("rho", "joint"):
        rho[3] *= 1 + expected["signed_step"] / 100
    same(row["rho_ohm_m"], rho, "Probe tissue scenario/heart-rho perturbation differs")
    require(np.all(rho > 0), "Positive resistivities required")


def validate_saved(data, record, cfg, material, model_hash, config_hash, plan_hash):
    """Small pure loader contract; called after file hashes have been checked."""
    identity, row = data["identity"], data["row"]
    for key, value in (("config_sha256", config_hash), ("plan_sha256", plan_hash),
                       ("ttrkg_model_sha256", model_hash)):
        require(identity[key] == value, f"Saved identity mismatch: {key}")
    probe = record["probe"]
    legacy_alias = (record["reused"] and record["scenario"] == "diagnostic4" and
                    row["scenario"] == "heart4_author_order_diagnostic_other_tissues_conditional02")
    require(row["variant"] == probe["material_variant"] and
            (row["scenario"] == record["scenario"] or legacy_alias), "MAT row variant/scenario mismatch")
    jobs = [j for j in cfg["jobs"] if j["id"] == row["id"]]
    require(len(jobs) == 1, "Saved job absent/duplicated in bound config")
    job = jobs[0]
    require(job["variant"] == row["variant"] and job["scenario"] == row["scenario"], "Job/row identity differs")
    require(bool(job["eit"]) == bool(row["eit_computed"]), "EIT job flag differs")
    require(record["reused"] or not job["eit"], "New study states must be TT-only")
    require(row["quadrature"] == probe["quadrature"] == material["quadrature"], "Quadrature mismatch")
    require(row["material_sha256"] == record["material_sha256"] == material["material_sha256"], "Material hash mismatch")
    rho = real(record["rho_ohm_m"], 5)
    for values in (row["rho_ohm_m"], job["rho_ohm_m"]):
        same(values, rho, "Saved rho differs")
    sigma = real(data["sigma"], 5)
    require(np.all(sigma > 0), "Invalid conductivity")
    same(sigma, 1 / rho, "sigma/rho mismatch")
    volumes = real(row["volume_ml"], 5)
    require(np.all(volumes > 0), "Invalid integrated material volumes")
    same(volumes, material["volume_ml"], "Integrated volumes differ from material plan")
    same(row["target_volume_ml"], material["target_volume_ml"], "Nominal volume differs")
    k, z = real(data["Ktt"], 5), float(real(data["Ztt"], 1)[0])
    same(k, row["Ktt_ohm_per_S_m"], "Ktt array/row mismatch")
    same(z, row["Ztt_ohm"], "Ztt array/row mismatch")
    if row["id"] == cfg["jobs"][0]["id"]:
        require(all(f in row for f in ("native_forward_relative_error", "reciprocity_relative_error")),
                "First saved job lacks native/reciprocity QC")
    for field in ("native_forward_relative_error", "reciprocity_relative_error"):
        if field in row:
            require(0 <= real(row[field], 1)[0] < 1e-8, "Failed saved solver QC: " + field)
    return dict(Ztt=z, Ktt=k, Jrho=-k / rho**2, volume_ml=float(volumes[3]),
                material_volumes_ml=volumes, nominal_volume_ml=float(row["target_volume_ml"]),
                row_sha256=digest_row(row), saved_scenario=row["scenario"], legacy_scenario_alias=legacy_alias)


def load_study(manifest_path):
    """Verify immutable inputs and every available MAT; no output writes."""
    path = Path(manifest_path).resolve()
    sources = Sources()
    sources.add(path)
    study = read_json(path)
    require(study["schema"] == "heart_montage_study_v1", "Unsupported study schema")
    require(study["stack_kind"] == "alternative_montages" and study["measured_noise"] is None,
            "Do not silently change simultaneity or invent a noise model")
    for name, sha in study["source_sha256"].items():
        sources.add(source_path(name, path.parent), sha)
    for code in (__file__, algebra.__file__, Path(__file__).with_name("heart_shape_electrical_comparison.py")):
        sources.add(code)

    def bound_json(value, parent=path.parent):
        p = source_path(value, parent)
        require(str(p) in sources.records, f"Manifest does not bind source hash: {p}")
        return p, read_json(p)

    gp, geometry = bound_json(study["geometry_manifest"])
    cp, contacts = bound_json(study["contact_manifest"])
    require(geometry["schema"] == "heart_montage_geometry_v1", "Geometry schema")
    require(contacts["status"] == "ready" and contacts["mesh"]["exact_EIT_TTRKG_raw_mesh_identity"], "Contact/mesh QC not ready")
    provenance = contacts["provenance"]
    runtime_path = source_path(provenance["runtime_config_name"], cp.parent)
    sources.add(runtime_path, provenance["runtime_config_sha256"])
    runtime = read_json(runtime_path)
    for field, entry in provenance["sources"].items():
        sources.add(source_path(runtime[field], runtime_path.parent), entry["sha256"])
    for function, sha in provenance["implementation"].items():
        candidates = [p for folder in ("tools", "src", "vendor_stl_eidors")
                      for p in (ROOT/"MATLAB_TRKG4_real_subjects"/folder).glob("**/"+function+".m")]
        require(len(candidates) == 1, "Ambiguous/missing contact implementation: " + function)
        sources.add(candidates[0], sha)
    require(study["shapes"] == geometry["shapes"] == ["STL", "G1", "L1"], "Shape list differs")
    require(study["scenarios"] == geometry["scenarios"] and len(study["scenarios"]) == 2, "Scenario definitions differ")
    for field in ("reference_volume_ml", "frequency_Hz", "current_A"):
        same(study[field], geometry[field], "Geometry/study context differs: " + field)
    levels = sorted({r["probe"]["quadrature"] for r in study["records"]})
    require(levels in ([64], [256], [64, 256]), "Unexpected quadrature levels")
    require(study.get("quadrature_levels", levels) == levels, "Declared quadrature levels differ")
    grid = expected_probes(geometry, levels)
    montage_ids = [m["id"] for m in study["montages"]]
    require(len(montage_ids) == len(set(montage_ids)) == 3, "Expected three distinct alternative montages")
    expected = {(m, s, p) for m in montage_ids for s in study["scenarios"] for p in grid}
    require(len(study["records"]) == len(expected) == {1: (558 if levels == [64] else 162), 2: 720}[len(levels)], "Incomplete manifest grid")
    require(sum(bool(r["reused"]) for r in study["records"]) == (54 if 64 in levels else 0), "Reuse count differs")
    if 64 in levels:
        controls = study["baseline_controls"]
        require(len(controls) == 3 and {v["shape"] for v in controls} == set(study["shapes"]) and
                all(v["exact_fraction_array_equality"] is True for v in controls), "Baseline equality QC missing")
    contact_rows = {m["id"]: m for m in contacts["montages"]}
    montage_hashes = {}
    for m in study["montages"]:
        model = source_path(m["model_path"], path.parent)
        cm = contact_rows[m["id"]]
        require(model == source_path(cm["ttrkg_model_file"], cp.parent), "Contact model path differs")
        require(cm["model_sha256"] == m["model_sha256"], "Contact model hash differs")
        same(cm["current_A"], study["current_A"], "Montage current differs")
        montage_hashes[m["id"]] = sources.add(model, m["model_sha256"])

    configs, plans, implementations, available, missing, seen = {}, {}, {}, [], [], set()
    declared_configs = {source_path(c["path"], path.parent) for c in study["configurations"]}
    for r in study["records"]:
        p = r["probe"]
        key = r["montage"], r["scenario"], p["id"]
        require(key in expected and key not in seen, "Duplicate/unexpected record")
        seen.add(key)
        validate_probe(r, grid[p["id"]], study["scenarios"][r["scenario"]])
        cfg_path = source_path(r["config"], path.parent)
        require(r["reused"] or cfg_path in declared_configs, "New job from undeclared config")
        if cfg_path not in configs:
            _, cfg = bound_json(cfg_path)
            for field in ("frequency_Hz", "current_A", "breath_interval_status"):
                require(cfg[field] == study[field], "Config context differs: " + field)
            plan_path = source_path(cfg["material_plan"], cfg_path.parent)
            sources.add(plan_path)  # reused plans also bound by MAT identity below
            plan = read_json(plan_path)
            require(plan["schema"] == "heart_stl_materials_v1" and plan["names"] == MATERIALS, "Material schema/order differs")
            for field, h in (("mesh_path", "mesh_sha256"), ("stl_path", "stl_sha256")):
                sources.add(source_path(cfg[field], cfg_path.parent), plan[h])
            require(plan["mesh_sha256"] == contacts["provenance"]["sources"]["mesh_file"]["sha256"], "Material/contact mesh differs")
            for name, sha in plan["sources"].items():
                sources.add(source_path(name, plan_path.parent), sha)
            eit_hash = sources.add(source_path(cfg["eit_model_file"], cfg_path.parent))
            require(eit_hash == contacts["provenance"]["sources"]["eit_model_file"]["sha256"], "EIT mesh model differs")
            tt_hash = sources.add(source_path(cfg["ttrkg_model_file"], cfg_path.parent))
            variants = {v["id"]: v for v in plan["variants"]}
            require(len(variants) == len(plan["variants"]), "Duplicate material variant")
            configs[cfg_path] = cfg, plan_path, variants, eit_hash, tt_hash
            plans[str(plan_path)] = sources.add(plan_path)
        cfg, plan_path, variants, eit_hash, tt_hash = configs[cfg_path]
        require(tt_hash == montage_hashes[r["montage"]], "Wrong physical montage model for record")
        material = variants[p["material_variant"]]
        sources.add(source_path(material["material_file"], plan_path.parent), material["material_sha256"])
        require(r["material_sha256"] == material["material_sha256"], "Record/material hash mismatch")
        same(material["target_volume_ml"], study["reference_volume_ml"] * p["volume_factor"], "Wrong nominal volume law")
        result_path = Path(r["path"])
        if not result_path.is_absolute():
            result_path = path.parent / result_path
        result_path = result_path.resolve()
        if not result_path.is_file():
            require(not r["reused"], "Reused result is missing")
            missing.append(dict(montage=r["montage"], scenario=r["scenario"], probe=p["id"], path=str(result_path)))
            continue
        require(not r["reused"] or r.get("mat_sha256"), "Reused MAT lacks prior hash")
        result_hash = sources.add(result_path, r.get("mat_sha256"))
        data = loadmat(result_path, simplify_cells=True)
        values = validate_saved(data, r, cfg, material, tt_hash, sources.add(cfg_path), sources.add(plan_path))
        identity = data["identity"]
        require(identity["eit_model_sha256"] == eit_hash, "Saved EIT model identity differs")
        require("sensitivity_sha256" in identity, "Saved sensitivity implementation hash missing")
        require(result_path == Path(cfg["output_dir"]).resolve() / (data["row"]["id"] + ".mat"), "Saved result path differs from job")
        sources.add(Path(__file__).with_name("heart_stl_electrical_check.m"), identity["implementation_sha256"])
        pipeline = source_path(cfg["pipeline_root"], cfg_path.parent)
        for field, filename in (("sensitivity_sha256", "cr_compartment_sensitivity.m"),
                                ("upstream_solve_sha256", "cr_compartment_solve.m"),
                                ("residual_sha256", "cr_relative_residual.m")):
            if field not in identity:
                continue
            cache_key = pipeline, filename
            if cache_key not in implementations:
                matches = list(pipeline.glob("**/" + filename))
                require(len(matches) == 1, "Ambiguous/missing operator implementation: " + filename)
                implementations[cache_key] = matches[0]
            sources.add(implementations[cache_key], identity[field])
        # All state-independent metadata stays hash bound; retain row hash too.
        available.append(dict(record=r, **values, source=dict(path=str(result_path), sha256=result_hash,
            row_sha256=values["row_sha256"], config_path=str(cfg_path), config_sha256=sources.add(cfg_path),
            material_plan=str(plan_path), plan_sha256=sources.add(plan_path),
            material_sha256=r["material_sha256"], model_sha256=tt_hash, identity=identity, reused=r["reused"],
            saved_scenario=values["saved_scenario"], legacy_scenario_alias=values["legacy_scenario_alias"])))
    require(seen == expected, "Manifest missing matched states")
    return study, available, missing, sources, plans


def transport_groups(study, available):
    """Only bundle a probe after all montage rows pass and share its context."""
    ids = [m["id"] for m in study["montages"]]
    gathered = {}
    for value in available:
        r = value["record"]
        p = r["probe"]
        key = p["shape"], r["scenario"], p["quadrature"]
        gathered.setdefault(key, {}).setdefault(p["id"], {})[r["montage"]] = value
    groups = []
    for (shape, scenario, q), probes in sorted(gathered.items()):
        rho = np.asarray(study["scenarios"][scenario])
        records = []
        for probe_id, montages in sorted(probes.items()):
            if set(montages) != set(ids):
                continue
            rows = [montages[m] for m in ids]
            first = rows[0]
            p = first["record"]["probe"]
            for value in rows[1:]:
                require(value["record"]["probe"] == p, "Montages use different geometry probes")
                require(value["source"]["material_sha256"] == first["source"]["material_sha256"], "Montages use different material maps")
                same(value["record"]["rho_ohm_m"], first["record"]["rho_ohm_m"], "Montage rho differs")
                same(value["material_volumes_ml"], first["material_volumes_ml"], "Montage volumes differ")
            rr = first["record"]["rho_ohm_m"]
            records.append(dict(id=probe_id, heart=[first["volume_ml"], *p["center_shift_mm"], rr[3]],
                tissue=[rr[0], rr[1]], volume_ml=first["volume_ml"], nominal_volume_ml=first["nominal_volume_ml"],
                impedance_ohm=[v["Ztt"] for v in rows], roi_ok=None,
                source_id="sha256:" + ":".join(v["source"]["sha256"] for v in rows),
                sources=[v["source"] for v in rows], probe=p,
                rho_material_order=list(rr), material_volumes_ml=first["material_volumes_ml"],
                Ktt_ohm_per_S_m=[v["Ktt"] for v in rows], Jrho_material_order=[v["Jrho"] for v in rows]))
        if not records:
            continue
        groups.append(dict(schema="heart_montage_analysis_v1", subject="Nik", size_coordinate="V",
            heart_order=list(algebra.HEART), tissue_order=list(algebra.TISSUE),
            amplitudes=[.01 * study["reference_volume_ml"], 1., 1., 1., .01*rho[3], .01*rho[0], .01*rho[1]],
            amplitude_basis=AMPLITUDE_BASIS, montage_ids=ids, stack_kind="alternative_montages",
            context=dict(shape=shape, scenario=scenario, quadrature=q, frequency_Hz=study["frequency_Hz"],
                current_A=study["current_A"], coordinate_frame="LPS_mm",
                coordinate_origin="translation relative to each form's bound reference center",
                background_id="frozen_background_heart_priority:geometry_manifest=" + study["geometry_manifest"],
                contact_model_id=";".join(m["model_sha256"] for m in study["montages"]),
                breath_interval_status=study["breath_interval_status"]),
            records=records, difference_pairs=[]))
    return groups


def index_group(group):
    return {probe_key(r["probe"]): r for r in group["records"]}


def get_probe(index, parameter, step):
    kind = "volume" if parameter == "V" else "rho" if parameter == "rho_heart" else "translation"
    return index[(kind, parameter, float(step))]


def jacobian(group, step):
    index = index_group(group)
    base = index[("base", "", 0.)]
    v = algebra.finite_difference(base, get_probe(index, "V", -step), get_probe(index, "V", step), "V")
    kv = v["raw"]["central"]
    coordinates = [algebra.finite_difference(base, get_probe(index, a, -step), get_probe(index, a, step), a, kv=kv) for a in "xyz"]
    jr = np.asarray(base["Jrho_material_order"])
    j = np.column_stack([kv, *(d["volume_corrected"]["central"] for d in coordinates), jr[:, 3], jr[:, 0], jr[:, 1]])
    raw = np.column_stack([kv, *(d["raw"]["central"] for d in coordinates), jr[:, 3], jr[:, 0], jr[:, 1]])
    return dict(step_mm=step, volume_step_percent=step, parameter_order=ORDER,
                derivative_units=["ohm/ml", *(["ohm/mm"]*3), *(["ohm/(ohm*m)"]*3)],
                J=j, J_raw_coordinates=raw, difference_estimates=[v, *coordinates],
                derivative_status="finite_step_with_discretization_not_automatically_converged",
                volume_correction="coordinate difference minus this step's K_V times integrated-volume drift",
                heart_rho_derivative="analytic conductivity-chain derivative -Ktt[heart]/rhoH^2")


def ready_groups(groups):
    ready = []
    for group in groups:
        try:
            j = jacobian(group, 1.)
        except KeyError:
            continue
        base_id = next(r["id"] for r in group["records"] if r["probe"]["kind"] == "base")
        selected = {base_id, *(i for e in j["difference_estimates"] for i in e["record_ids"])}
        payload = dict(group, records=[r for r in group["records"] if r["id"] in selected])
        payload["difference_pairs"] = [dict(parameter=e["parameter"], minus=e["record_ids"][0], plus=e["record_ids"][1]) for e in j["difference_estimates"]]
        algebra.validate_export(payload)
        ready.append(dict(dataset=payload, jacobian=j,
                          status="preliminary_small_step_group_for_adaptive_FEM_not_full_analysis",
                          finite_compensation_checked=False, **FLAGS))
    return ready


def target_functional(j, scales, mapping=None):
    """Volume row-space test for seven parameters or an explicit reduced law."""
    b = np.eye(j.shape[1]) * scales if mapping is None else np.asarray(mapping)
    a = j @ b
    _, s, vh = np.linalg.svd(a, full_matrices=True)
    tol = max(a.shape) * np.finfo(float).eps * (s[0] if s.size else 0)
    rank = int(np.count_nonzero(s > tol))
    null = vh[rank:].T
    target = b[0]
    require(np.linalg.norm(target) > 0, "Restriction contains no volume coordinate")
    residual = null @ (null.T @ target)
    relative = np.linalg.norm(residual) / np.linalg.norm(target)
    # A small relative target/null overlap alone can falsely identify V in
    # the presence of a weak retained nuisance. Test the rank increment too,
    # using an orthonormal basis of ker(target) and the SAME matrix threshold.
    _, _, target_vh = np.linalg.svd(target[None, :], full_matrices=True)
    nuisance = a @ target_vh[1:].T
    nuisance_s = np.linalg.svd(nuisance, compute_uv=False)
    nuisance_rank = int(np.count_nonzero(nuisance_s > tol))
    identified = bool(rank - nuisance_rank == 1 and
                      relative <= 100 * max(a.shape) * np.finfo(float).eps)
    return dict(volume_identified=identified, rowspace_relative_residual=float(relative), rank=rank,
                nuisance_rank=nuisance_rank, volume_rank_increase=rank-nuisance_rank,
                rank_tolerance=float(tol), unknowns=a.shape[1], null_volume_ml=target @ null,
                finite_noise_accuracy=None, status="local_functional_test_not_validation")


def analyze(groups):
    tables = {name: [] for name in ("derivatives", "step_changes", "rho_checks", "identifiability",
        "joint_decomposition", "joint_components", "finite_volume_errors", "restrictions",
        "restriction_finite_errors", "omission_units", "exclusion_error_budgets",
        "quadrature_comparison", "quadrature_derivatives")}
    detail, datasets = [], []
    for group in groups:
        algebra.validate_export(group)
        ctx = group["context"]
        prefix = {k: ctx[k] for k in ("shape", "scenario", "quadrature")}
        ids, scales = group["montage_ids"], np.asarray(group["amplitudes"])
        index = index_group(group)
        base = index[("base", "", 0.)]
        basep = np.r_[base["heart"], base["tissue"]]
        levels = (1., 2.5, 5.) if ctx["quadrature"] == 64 else (1.,)
        matrices = [jacobian(group, step) for step in levels]
        primary = matrices[0]["J"]
        estimates = []
        for matrix in matrices:
            estimates.extend(matrix["difference_estimates"])
        if ctx["quadrature"] == 64:
            for step in (1., 5.):
                e = algebra.finite_difference(base, get_probe(index, "rho_heart", -step), get_probe(index, "rho_heart", step), "rho_heart")
                estimates.append(e)
                check = algebra.derivative_check(primary[:, 4], e)
                for i, mid in enumerate(ids):
                    tables["rho_checks"].append(dict(**prefix, montage=mid, rho_step_percent=step,
                        analytic=check["analytic"][i], finite=check["finite_central"][i], signed_error=check["signed_error"][i],
                        left=e["raw"]["minus"][i], right=e["raw"]["plus"][i]))
        for e in estimates:
            for component in ("raw", "volume_corrected"):
                if e[component] is None:
                    continue
                for i, mid in enumerate(ids):
                    tables["derivatives"].append(dict(**prefix, montage=mid, parameter=e["parameter"], component=component,
                        derivative_unit="ohm/ml" if e["parameter"] == "V" else "ohm/mm" if e["parameter"] in "xyz" else "ohm/(ohm*m)",
                        minus_id=e["record_ids"][0], plus_id=e["record_ids"][1],
                        minus_step=e["steps"][0], plus_step=e["steps"][1],
                        minus_volume_drift_ml=e["integrated_delta_volume_ml"][0], plus_volume_drift_ml=e["integrated_delta_volume_ml"][1],
                        central=e[component]["central"][i], secant=e[component]["secant"][i],
                        left=e[component]["minus"][i], right=e[component]["plus"][i],
                        one_sided_difference=e[component]["one_sided_difference"][i],
                        correction_KV=None if e["volume_drift_KV_ohm_per_ml"] is None else e["volume_drift_KV_ohm_per_ml"][i]))
        convergence = []
        for parameter in ORDER[:5]:
            family = [e for e in estimates if e["parameter"] == parameter]
            if len(family) < 2:
                continue
            for component in (("raw", "volume_corrected") if parameter in "xyz" else ("raw",)):
                conv = algebra.step_convergence(family, component=component)
                convergence.append(dict(parameter=parameter, **conv))
                for c in conv["comparisons"]:
                    for i, mid in enumerate(ids):
                        tables["step_changes"].append(dict(**prefix, montage=mid, parameter=parameter, component=component,
                            coarse_steps=c["coarse_steps"], fine_steps=c["fine_steps"], signed_change=c["signed_change"][i],
                            relative_stack_norm_change=c["relative_norm_change"], converged=None,
                            correction_basis="matched-volume-step KV; changes include its variation"))
        diagnostics = []
        for matrix in matrices:
            for component, j in (("raw", matrix["J_raw_coordinates"]), ("volume_corrected", matrix["J"])):
                diag = algebra.montage_diagnostics(j[:, :5], j[:, 5:], scales, montage_ids=ids, stack_kind="alternative_montages")
                diagnostics.append(dict(step=matrix["step_mm"], component=component, result=diag))
                for name, selection in [*((mid, [i]) for i, mid in enumerate(ids)), ("hypothetical_stack", list(range(len(ids))))]:
                    d = diag["stack"] if name == "hypothetical_stack" else diag["per_montage"][name]
                    for n, key in ((5, "heart_if_exact_tissue_corrections"), (7, "heart_plus_free_tissue")):
                        dd = d[key]
                        target = d["volume_identified_given_nuisance"] if n == 5 else target_functional(j[selection], scales)
                        tables["identifiability"].append(dict(**prefix, montage=name, step=matrix["step_mm"], component=component,
                            unknowns=n, rank=dd["rank"], nullity=dd["nullity"], singular_values=dd["singular_values"],
                            volume_identified=target["volume_identified"], eta_V=dd["unique_fraction"][0],
                            volume_functional=target, practical_identifiability="not_assessed_noise_and_model_error_unknown"))
        finite = []
        cases = [dict(kind="joint" if r["probe"]["kind"] == "joint" else "finite_single", record=r,
                      tissue_correction_ohm=np.zeros(len(ids))) for r in group["records"] if r["id"] != base["id"]]
        # These controls prescribe identical soft/lung rho across states. Zero
        # correction below describes that control, not physiological pulsation.
        for matrix in matrices:
            for name, omitted in (("omit_XYZ_and_rhoH", [1, 2, 3, 4]), ("omit_XYZ", [1, 2, 3]), ("omit_rhoH", [4])):
                rows = algebra.omission_table(base, cases, matrix["J"], omitted, scales)
                for r in rows:
                    tables["finite_volume_errors"].append(dict(**prefix, montage=ids[r["montage_index"]],
                        derivative_step=matrix["step_mm"], omission=name,
                        tissue_scope="prescribed_fixed_soft_lung_in_this_FEM_probe_only", **r))
            finite.append(dict(step=matrix["step_mm"], checks=[algebra.finite_probe_check(base, c["record"], matrix["J"],
                                    tissue_correction_ohm=np.zeros(len(ids))) for c in cases]))
        joints = []
        if ctx["quadrature"] == 64:
            for sign in (-1., 1.):
                joint = index[("joint", "Vxyzrho", sign)]
                singles = [get_probe(index, a, sign * (-1 if a == "y" else 1)) for a in ORDER[:5]]
                decomp = algebra.joint_decomposition(base, singles, joint, primary)
                joints.append(dict(joint_id=joint["id"], single_ids=[r["id"] for r in singles], **decomp))
                dy = np.asarray([np.asarray(r["impedance_ohm"]) - base["impedance_ohm"] for r in singles])
                for i, mid in enumerate(ids):
                    denominator = np.abs(dy[:, i]).sum()
                    tables["joint_decomposition"].append(dict(**prefix, montage=mid, sign=sign,
                        joint_delta_ohm=decomp["joint_delta_ohm"][i], sum_single_ohm=decomp["sum_single_delta_ohm"][i],
                        sum_abs_single_ohm=denominator,
                        cancellation_fraction=None if denominator == 0 else 1-abs(dy[:, i].sum())/denominator,
                        joint_minus_sum_ohm=decomp["joint_minus_sum_ohm"][i],
                        volume_step_mismatch_ml=decomp["parameter_step_mismatch"][0],
                        step_mismatch_ohm=decomp["mismatch_linear_response_ohm"][i],
                        aligned_interaction_ohm=decomp["interaction_after_linear_step_alignment_ohm"][i],
                        closure_ohm=decomp["decomposition_closure_ohm"][i]))
                    for k, parameter in enumerate(ORDER):
                        tables["joint_components"].append(dict(**prefix, montage=mid, sign=sign, parameter=parameter,
                            joint_linear_contribution_ohm=primary[i, k]*decomp["joint_delta_parameters"][k],
                            finite_single_delta_ohm=dy[k, i] if k < 5 else None,
                            controlled_tissue_change=parameter in algebra.TISSUE))
        # Unit response factors preserve unknown correction errors explicitly.
        omission = algebra.volume_omission(primary[:, :5], primary[:, 5:], [None]*4, [None]*2, base_size=base["volume_ml"])
        for i, mid in enumerate(ids):
            for k, parameter in enumerate(ORDER[1:], 1):
                kv = primary[i, 0]
                tables["omission_units"].append(dict(**prefix, montage=mid, parameter=parameter,
                    derivative=primary[i, k], diagnostic_amplitude=scales[k],
                    diagnostic_response_ohm=primary[i, k]*scales[k],
                    equivalent_volume_per_parameter_unit=None if kv == 0 else primary[i, k]/kv,
                    equivalent_volume_at_diagnostic_amplitude=None if kv == 0 else primary[i, k]*scales[k]/kv,
                    total_volume_error_ml=None, measured_error_amplitude=None))
                ki = primary[i, k]
                parameter_unit = "mm" if k <= 3 else "ohm*m"
                amplitude = 1. if k <= 3 else .01 * basep[k]
                for budget in (1., 5., 10.):
                    bound = None if kv == 0 or ki == 0 else budget * abs(kv) / abs(ki)
                    tables["exclusion_error_budgets"].append(dict(**prefix, montage=mid, parameter=parameter,
                        K_i=ki, K_V_ohm_per_ml=kv, parameter_unit=parameter_unit,
                        response_unit="1_mm" if k <= 3 else "1_percent_of_baseline_rho",
                        signed_ml_per_response_unit=None if kv == 0 else ki*amplitude/kv,
                        illustrative_budget_ml=budget, isolated_abs_parameter_change=bound,
                        isolated_abs_rho_change_percent=None if k <= 3 or bound is None else 100*bound/basep[k],
                        status="undefined_zero_KV" if kv == 0 else "no_local_first_order_bound_zero_Ki" if ki == 0 else "conditional_isolated_linear_bound",
                        label="illustrative_error_budget_not_clinical_requirement",
                        assumption="all_other_contributions_and_residuals_exactly_removed_in_this_isolated_test",
                        total_error_bound=False, physiological_range=False,
                        finite_probe_range_checked=False, derivative_converged=None))
        restrictions = restriction_outputs(group, primary, base, cases, tables, prefix)
        detail.append(dict(**prefix, diagnostics=diagnostics, convergence=convergence, joint=joints,
                           finite_controls=finite, unknown_error_budget=omission, restrictions=restrictions))
        datasets.append(dict(dataset=group, jacobians=matrices, **FLAGS))
    compare_quadratures(datasets, tables)
    return tables, detail, datasets


def restriction_outputs(group, j, base, cases, tables, prefix):
    scales = np.asarray(group["amplitudes"])
    # Mappings use dimensionless reduced coordinates. The tie is a declared
    # diagnostic hypothesis, NOT fitted from cardiac-cycle data or validated.
    bvolume = np.zeros((5, 1)); bvolume[0, 0] = scales[0]
    bvrho = np.zeros((5, 2)); bvrho[0, 0] = scales[0]; bvrho[4, 1] = scales[4]
    btie = np.array([[scales[0]], [1.], [-1.], [1.], [scales[4]]])
    mappings = {"fixed_XYZ_rhoH": bvolume, "free_V_rhoH_fixed_XYZ": bvrho,
                "assumed_tie_XYZ_rhoH_to_V_joint_direction": btie}
    ids = group["montage_ids"]
    p0 = np.asarray(base["heart"])
    results = []
    for law, b in mappings.items():
        evidence = dict(law_id=law, no_ai=True, construction_ids=[], holdout_ids=[],
            current_law_holdout_checked=False, previous_4d_only=False, operational_phase_inputs=[],
            roi_conditions="not independently established", amplitude_basis=AMPLITUDE_BASIS)
        for name, selection in [*((mid, [i]) for i, mid in enumerate(ids)), ("hypothetical_stack", list(range(len(ids))))]:
            jj = j[selection, :5]
            target = target_functional(jj, scales[:5], b)
            diag = algebra.linear_diagnostics(jj @ b, np.ones(b.shape[1]))
            tables["restrictions"].append(dict(**prefix, montage=name, law=law, mapping=b,
                rank=diag["rank"], unknowns=b.shape[1], volume_identified=target["volume_identified"],
                functional=target, validated_law=False, assumption="soft/lung corrections exact; no physiological claim"))
            law_cases = []
            a = jj @ b
            for case in cases:
                trial = case["record"]
                dp = np.asarray(trial["heart"]) - p0
                check = algebra.restriction_check(jj, b, dp, scales[:5], evidence=evidence)
                dz = (np.asarray(trial["impedance_ohm"]) - base["impedance_ohm"])[selection]
                coordinates = np.linalg.lstsq(a, dz, rcond=None)[0]
                fitted = b @ coordinates
                residual = dz - a @ coordinates
                tables["restriction_finite_errors"].append(dict(**prefix, montage=name, law=law, probe=trial["id"],
                    true_delta_volume_ml=dp[0], representative_delta_volume_ml=fitted[0],
                    representative_error_ml=fitted[0]-dp[0], volume_identified=target["volume_identified"],
                    electrical_residual_ohm=residual,
                    linear_omission_error_ml=check["representative_linear_volume_error_ml"],
                    finite_minus_linear_error_ml=(fitted[0]-dp[0])-check["representative_linear_volume_error_ml"],
                    status="minimum_norm_representative_not_unique" if not target["volume_identified"] else "conditional_finite_control_not_validation"))
                law_cases.append(dict(probe=trial["id"], linear=check))
            results.append(dict(law=law, montage=name, mapping=b, evidence=algebra.law_evidence_check(evidence), cases=law_cases,
                prior_4D_law="ch16.7 RAS/frame law not imported into static LPS FEM; registration and new transfer checks required"))
    return results


def compare_quadratures(datasets, tables):
    bykey = {(d["dataset"]["context"]["shape"], d["dataset"]["context"]["scenario"],
              d["dataset"]["context"]["quadrature"]): d for d in datasets}
    for shape, scenario in sorted({k[:2] for k in bykey}):
        coarse, fine = bykey[shape, scenario, 64], bykey[shape, scenario, 256]
        ci, fi = (index_group(d["dataset"]) for d in (coarse, fine))
        cb, fb = ci["base", "", 0.], fi["base", "", 0.]
        for key, trial in fi.items():
            check = algebra.discretization_comparison(cb, ci[key], fb, trial)
            for i, mid in enumerate(fine["dataset"]["montage_ids"]):
                tables["quadrature_comparison"].append(dict(shape=shape, scenario=scenario, montage=mid,
                    probe=trial["id"], coarse_probe=ci[key]["id"],
                    baseline_difference_ohm=check["baseline_difference_ohm"][i],
                    delta_difference_ohm=check["delta_difference_ohm"][i],
                    baseline_volume_difference_ml=check["baseline_volume_difference_ml"],
                    trial_volume_difference_ml=check["trial_volume_difference_ml"], converged=None))
        for component in ("J", "J_raw_coordinates"):
            jc, jf = (d["jacobians"][0][component] for d in (coarse, fine))
            for i, mid in enumerate(fine["dataset"]["montage_ids"]):
                for k, parameter in enumerate(ORDER):
                    scale = fine["dataset"]["amplitudes"][k]
                    dz = (jf[i, k]-jc[i, k])*scale
                    # q256 is an explicit comparison denominator, not a truth
                    # or global convergence limit. q64 denominator stays too.
                    tables["quadrature_derivatives"].append(dict(shape=shape, scenario=scenario, montage=mid,
                        parameter=parameter, component=component, q64=jc[i, k], q256=jf[i, k],
                        signed_change=jf[i, k]-jc[i, k], relative_change=None if jf[i, k] == 0 else (jf[i, k]-jc[i, k])/abs(jf[i, k]),
                        diagnostic_parameter_amplitude=scale, delta_J_response_ohm=dz,
                        delta_J_equivalent_ml_using_q256_KV=None if jf[i, 0] == 0 else dz/jf[i, 0],
                        delta_J_equivalent_ml_using_q64_KV=None if jc[i, 0] == 0 else dz/jc[i, 0],
                        amplitude_basis=AMPLITUDE_BASIS,
                        converged=None, scope="local_quadrature_not_global_mesh_convergence"))


def self_test():
    """Small MAT loader/provenance rejection tests, plus independent algebra."""
    checks = []
    def rejects(f):
        try:
            f()
        except (ValueError, KeyError):
            return
        raise AssertionError("Invalid saved result accepted")
    rho = [3., 18., 19., 4., 23.]
    material = dict(quadrature=64, material_sha256="material", volume_ml=[10., 20., 3., 500., 5.], target_volume_ml=501.)
    probe = dict(kind="base", material_variant="base", quadrature=64, signed_step=0.)
    record = dict(probe=probe, scenario="s", material_sha256="material", rho_ohm_m=rho, reused=False, format="electrical", probe_index=None)
    job = dict(id="base", scenario="s", variant="base", eit=False, rho_ohm_m=rho)
    row = dict(job, eit_computed=False, quadrature=64, material_sha256="material", volume_ml=material["volume_ml"],
        target_volume_ml=501., Ztt_ohm=12., Ktt_ohm_per_S_m=[2., -3., 0., 8., 1.], native_forward_relative_error=1e-15, reciprocity_relative_error=1e-15)
    identity = dict(config_sha256="config", plan_sha256="plan", ttrkg_model_sha256="model")
    data = dict(identity=identity, row=row, sigma=1/np.asarray(rho), Ztt=12., Ktt=row["Ktt_ohm_per_S_m"])
    cfg = dict(jobs=[job])
    def validate(d):
        return validate_saved(d, record, cfg, material, "model", "config", "plan")
    with tempfile.TemporaryDirectory(prefix="heart_montage_adapter_") as directory:
        p = Path(directory)/"small.mat"
        savemat(p, data)
        loaded = loadmat(p, simplify_cells=True)
        result = validate(loaded)
        assert result["volume_ml"] == 500. and result["nominal_volume_ml"] == 501.
        np.testing.assert_allclose(result["Jrho"], -np.asarray(row["Ktt_ohm_per_S_m"])/np.asarray(rho)**2)
        hashes = Sources(); digest = hashes.add(p); hashes.add(p, digest)
        rejects(lambda: hashes.add(p, "0"*64))
        checks.append("small_real_MAT_load_integrated_not_nominal_volume_signed_rho_chain_hash_rejection")
        for field, value in (("rho_ohm_m", [3., 18., 19., 5., 23.]), ("quadrature", 256),
                             ("volume_ml", [10., 20., 3., 501., 5.]), ("material_sha256", "wrong"),
                             ("Ztt_ohm", 13.), ("scenario", "wrong"), ("reciprocity_relative_error", 1e-3)):
            bad = copy.deepcopy(loaded); bad["row"][field] = value
            rejects(lambda: validate(bad))
        bad = copy.deepcopy(loaded); bad["Ktt"] = np.asarray(bad["Ktt"], complex)
        rejects(lambda: validate(bad))
        bad = copy.deepcopy(loaded); bad["identity"]["ttrkg_model_sha256"] = "wrong"
        rejects(lambda: validate(bad))
        checks.append("context_material_model_row_array_QC_complex_rejections")
    validated = dict(record)
    validate_probe(validated, probe, rho)
    rejects(lambda: validate_probe(dict(validated, probe=dict(probe, quadrature=256)), probe, rho))
    rejects(lambda: validate_probe(dict(validated, rho_ohm_m=[3,18,19,4.1,23]), probe, rho))
    checks.append("manifest_geometry_and_scenario_rejections")
    identified = np.array([[1., 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1]])
    assert target_functional(identified, np.ones(7))["volume_identified"]
    identified[0, 6] = 1
    assert not target_functional(identified, np.ones(7))["volume_identified"]
    assert not target_functional(np.ones((1, 7)), np.ones(7))["volume_identified"]
    weak = np.zeros((1, 7)); weak[0, 0] = 1.; weak[0, 1] = 1e-14
    assert not target_functional(weak, np.ones(7))["volume_identified"]
    checks.append("seven_parameter_volume_functional_not_inferred_from_rank_alone")
    # Forty tiny vectors exercise adapter table assembly, quadrature matching,
    # and volume-drift bookkeeping; these are algebra fixtures, never FEM data.
    j = np.array([[2., 3., -4., 5., 6., 7., -8.],
                  [-1., 2., 3., -4., 5., -6., 7.],
                  [3., -1., 2., 1., -2., 4., 3.]])
    fixtures = []
    for q in (64, 256):
        p0 = np.array([500. + q/10000, 0., 0., 0., 4., 3., 18.])
        records = []
        def add(kind, axis, step):
            dp = np.zeros(7)
            if kind in ("volume", "joint"):
                dp[0] = 5*step
            if kind == "translation":
                dp[ORDER.index(axis)] = step
            if kind == "rho":
                dp[4] = .04*step
            if kind == "joint":
                dp[1:4] = np.array([1., -1., 1.])*step
                dp[4] = .04*step
            dp[0] += (64/q)*.1*dp[1]  # explicit integrated-volume drift
            pp = p0+dp
            ident = f"{kind}_{axis}_{step}_{q}"
            jr = np.column_stack([j[:, 5], j[:, 6], np.zeros(3), j[:, 4], np.zeros(3)])
            records.append(dict(id=ident, heart=pp[:5], tissue=pp[5:], volume_ml=pp[0],
                nominal_volume_ml=500.+(5*step if kind in ("volume", "joint") else 0),
                impedance_ohm=np.array([100., 110., 120.])+j@dp,
                roi_ok=None, source_id="synthetic_math_only:"+ident, Jrho_material_order=jr,
                probe=dict(kind=kind, axis=axis, signed_step=step)))
        add("base", "", 0.)
        for step in ((1., 2.5, 5.) if q == 64 else (1.,)):
            for sign in (-1., 1.):
                for axis in ORDER[:4]:
                    add("volume" if axis == "V" else "translation", axis, sign*step)
        if q == 64:
            for step in (-5., -1., 1., 5.):
                add("rho", "rho_heart", step)
            for sign in (-1., 1.):
                add("joint", "Vxyzrho", sign)
        fixtures.append(dict(schema="heart_montage_analysis_v1", subject="Nik", size_coordinate="V",
            heart_order=ORDER[:5], tissue_order=ORDER[5:], amplitudes=[5., 1., 1., 1., .04, .03, .18],
            amplitude_basis="synthetic algebra fixture only", montage_ids=["A", "B", "C"],
            stack_kind="alternative_montages", difference_pairs=[], records=records,
            context=dict(shape="synthetic", scenario="synthetic", quadrature=q,
                         frequency_Hz=1., coordinate_frame="synthetic", background_id="synthetic",
                         contact_model_id="synthetic", breath_interval_status="synthetic")))
    tables, detail, dataset = analyze(fixtures)
    for item in dataset:
        np.testing.assert_allclose(item["jacobians"][0]["J"], j, atol=2e-12)
    assert len(tables["quadrature_comparison"]) == 27
    assert len(tables["exclusion_error_budgets"]) == 108
    budget_row = next(r for r in tables["exclusion_error_budgets"] if r["parameter"] == "x" and r["montage"] == "A")
    np.testing.assert_allclose(budget_row["isolated_abs_parameter_change"], 2/3)
    np.testing.assert_allclose(budget_row["signed_ml_per_response_unit"], 1.5)
    assert max(abs(r["closure_ohm"]) for r in tables["joint_decomposition"]) < 1e-12
    algebra.jsonable(dict(tables=tables, detail=detail, dataset=dataset, ready=ready_groups(fixtures)))
    checks.append("forty_small_vector_table_smoke_test_corrected_not_double_counted_volume_drift")
    math_result = algebra.self_test()
    return dict(status="passed", adapter_checks=checks, helper_checks=math_result["checks"],
                scope="loader_and_mathematical_selftests_only", **FLAGS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        require(not args.manifest and not args.output and not args.check_only, "Self-test is a separate mode")
        print(json.dumps(algebra.jsonable(self_test()), ensure_ascii=True, allow_nan=False))
        return
    require(args.manifest is not None, "--manifest is required")
    require(args.check_only or args.output is not None, "--output is required for full analysis")
    study, available, missing, sources, plans = load_study(args.manifest)
    groups = transport_groups(study, available)
    progress = dict(schema="heart_montage_saved_progress_v1", manifest=str(args.manifest.resolve()),
        expected=len(study["records"]), loaded=len(available), missing=missing,
        full_720_ready=len(available) == len(study["records"]) == 720,
        ready_groups=ready_groups(groups), material_plan_hashes=plans, **FLAGS)
    if args.check_only:
        sources.unchanged(rehash=True)
        progress["source_hashes"] = sources.records
        print(json.dumps(algebra.jsonable(progress), ensure_ascii=True, allow_nan=False))
        return
    require(progress["full_720_ready"], f"Full analysis requires all 720 states; loaded {len(available)}/{len(study['records'])}")
    tables, detail, datasets = analyze(groups)
    sources.unchanged(rehash=True)
    output = args.output.resolve()
    require(not output.exists() or not any(output.iterdir()), "Output must be new/empty; existing results are never replaced")
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_csv(output/(name+".csv"), algebra.jsonable(rows))
    write_json(output/"dataset.local.json", algebra.jsonable(dict(schema="heart_montage_saved_dataset_v1",
        manifest=progress["manifest"], parameter_order=ORDER, groups=datasets,
        source_hashes=sources.records, **FLAGS)))
    write_json(output/"analysis.local.json", algebra.jsonable(dict(schema="heart_montage_saved_analysis_v1",
        states=len(available), groups=detail, table_rows={k: len(v) for k,v in tables.items()},
        source_hashes=sources.records, limits=study["limits"],
        remaining_uncertainty="Noise, contact error, anatomical mismatch and physiological tissue-correction error unknown",
        **FLAGS)))
    print(json.dumps(dict(status="saved_MAT_analysis_completed_not_validation", output=str(output),
        states=len(available), table_rows={k:len(v) for k,v in tables.items()}, **FLAGS), ensure_ascii=True))


if __name__ == "__main__":
    main()
