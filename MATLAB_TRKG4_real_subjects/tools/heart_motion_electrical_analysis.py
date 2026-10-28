"""Audit saved L1 FEM motion probes; no FEM, measured pulses, or 4D trajectories.

Run with explicit --config PATH and --output PATH. The config is the unchanged
heart_stl_electrical_check input. All 42 TT probes and 18 core EIT probes are
required; EIT at shifted v095/v105 is optional. --check-only audits inputs and
available results without producing incomplete scientific diagnostics.
Weights are ROW MULTIPLIERS W, so the objective is ||W(y-Ax)||_2^2.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.io import loadmat

from pulmovista_pulse_identifiability import general_design, design_summary, jsonable


SCENARIOS = ("diagnostic4", "effectiveL1")
POSES = ("base", "xm5", "xp5", "ym5", "yp5", "zm5", "zp5")
LEVELS = (95, 100, 105)
NAMES = ["soft_background", "lung", "bone", "heart", "fat"]
SETTINGS = ("frequency_Hz", "current_A", "montage_basis", "breath_interval_status")
UNITS = dict(V_eit="V", S_eit="V/(S/m)", Ztt="ohm", Ktt="ohm/(S/m)",
             Jvol="V/ml", Jpos="V/mm", Kvol="ohm/ml", Kpos="ohm/mm",
             volume="ml", position="mm in LPS", conductivity="S/m")
LIMITATIONS = [
    "Computational probes, not observations or physiological amplitudes; no measured pulse input.",
    "Both diagnostic4 and effectiveL1 are conditional material scenarios, not tissue truth.",
    "L1 is the saved volume-preserving moment ellipsoid, not the original STL surface.",
    "No universal motion/volume law selected; no registered dynamic CT trajectories used.",
    "Algorithmic EIT belt and diagnostic TT montage are not verified recording coordinates.",
    "Fixed background with heart priority is a material replacement, not mechanical organ motion.",
    "Soft and single-lung conductivity pulse sensitivities are retained: target tissue pulses are corrected using EIT estimates, not assumed absent; bone/fat remain fixed in this audit.",
    "Optional heart conductivity is a separate nuisance, not a second explanation of volume.",
    "Numerical rank is not practical identifiability or independent validation; no noise model is inferred.",
    "Relative-base weights are a metric, not measured inverse noise standard deviations; no gain fitted.",
    "Linear volume-mismatch subtraction is diagnostic and does not undo quadrature or FEM error.",
    "Global mesh/material convergence and anatomical coverage of the source heart remain unproven.",
    "Apparent volume is an effective model parameter; it is not ventricular stroke volume or EF.",
    "The displacement-ball bound is local and linear, excludes nonlinear remainder, and is not measured motion.",
    "Joint EIT+TT geometry inversion is only a supplemental idealized shared-frequency/model diagnostic, outside the target method; current sessions are not combined.",
    "Inter-device phase synchronization, frequency transfer, noise and simultaneous electrode/source interactions remain unverified.",
    "Target: synchronous PulmoVista and TT with shared ECG; PulmoVista is the required source of rho0+delta_rho(t), while geometry/functions come from tissue-corrected TT and one CT phase.",
    "Even with specified tissue corrections, one scalar TT channel cannot identify four free geometric unknowns; six/seven-column scalar tests illustrate additional unconstrained tissue uncertainty, not the target inversion.",
    "Baseline and pulse conductivity uncertainty is not zero or quantified by this audit; two fixed-rho scenarios are not actual synchronous rho(t), and no new synchronized experimental data are produced.",
    "Fixed conductivities in a pure displacement computational probe isolate motion; they do not assert absence of physiological soft/lung pulses.",
    "Geometry-to-tissue leakage fits computed geometry-only voltage differences with a fixed-geometry tissue inverse; inferred delta_rho is spurious here, not measured tissue dynamics.",
]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(jsonable(value), ensure_ascii=False,
                                    indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows({k: jsonable(v) for k, v in row.items()} for row in rows)


def variant(pose, level):
    return f"L1_{'' if pose == 'base' else pose + '_'}v{level:03d}_q64"


def resolve(path, parent):
    p = Path(path)
    return p.resolve() if p.is_absolute() else (parent / p).resolve()


def close(a, b, message, atol=1e-10):
    require(np.shape(a) == np.shape(b) and np.allclose(a, b, rtol=1e-11, atol=atol), message)


class Sources:
    """Hash once per immutable source, rejecting changes within the audit."""
    def __init__(self):
        self.records = {}

    def add(self, path, expected=None):
        p = Path(path).resolve()
        stat = p.stat()
        key = str(p)
        if key not in self.records:
            with p.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            require((stat.st_size, stat.st_mtime_ns) == (p.stat().st_size, p.stat().st_mtime_ns),
                    f"Source changed while hashing: {p}")
            self.records[key] = dict(sha256=digest, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        entry = self.records[key]
        require((entry["bytes"], entry["mtime_ns"]) == (stat.st_size, stat.st_mtime_ns),
                f"Source changed during analysis: {p}")
        if expected is not None:
            require(entry["sha256"] == expected, f"SHA256 mismatch: {p}")
        return entry["sha256"]

    def unchanged(self):
        for path in list(self.records):
            self.add(path)


def inspect_inputs(config_path, sources):
    root = Path(__file__).resolve().parents[2]
    config_path = Path(config_path).resolve()
    b = config_path.parent
    cfg = read_json(config_path)
    plan_path = resolve(cfg["material_plan"], b)
    plan = read_json(plan_path)
    require(plan["schema"] == "heart_stl_materials_v1" and plan["names"] == NAMES,
            "Material schema/order mismatch")
    require(cfg["frequency_Hz"] > 0 and cfg["current_A"] > 0, "Invalid frequency/current")
    identity = dict(config_sha256=sources.add(config_path), plan_sha256=sources.add(plan_path),
                    eit_model_sha256=sources.add(resolve(cfg["eit_model_file"], b)),
                    ttrkg_model_sha256=sources.add(resolve(cfg["ttrkg_model_file"], b)),
                    implementation_sha256=sources.add(Path(__file__).with_name("heart_stl_electrical_check.m")))
    sensitivity = list(resolve(cfg["pipeline_root"], b).glob("**/cr_compartment_sensitivity.m"))
    require(len(sensitivity) == 1, "Ambiguous/missing upstream sensitivity implementation")
    identity["sensitivity_sha256"] = sources.add(sensitivity[0])
    tt_model = loadmat(resolve(cfg["ttrkg_model_file"], b), variable_names=["Jtt", "Ksigma"], simplify_cells=True)
    close(tt_model["Jtt"], cfg["current_A"]*tt_model["Ksigma"], "Saved TT model current scaling inconsistent")
    current_error = relative_norm(tt_model["Jtt"]-cfg["current_A"]*tt_model["Ksigma"], tt_model["Jtt"])
    sources.add(resolve(cfg["mesh_path"], b), plan["mesh_sha256"])
    sources.add(resolve(cfg["stl_path"], b), plan["stl_sha256"])
    sources.add(resolve(cfg["reference_prediction_file"], b))
    sources.add(__file__)
    sources.add(Path(__file__).with_name("pulmovista_pulse_identifiability.py"))
    source_variants = {}
    for path, digest in plan["source_plans_sha256"].items():
        pp = resolve(path, root)
        sources.add(pp, digest)
        source_plan = read_json(pp)
        require(source_plan["names"] == NAMES, "Source material order mismatch")
        for field in ("mesh_sha256", "stl_sha256", "quadrature_rule", "seed"):
            require(source_plan[field] == plan[field], f"Mixed material settings: {field}")
        for src, h in source_plan["sources"].items():
            sources.add(resolve(src, root), h)
        for material in source_plan["variants"]:
            source_variants[material["id"]] = material
    variants = {v["id"]: v for v in plan["variants"]}
    expected_variants = {variant(p, v) for p in POSES for v in LEVELS}
    require(set(variants) == expected_variants and len(plan["variants"]) == 21,
            "Require exactly the 21 L1 q64 volume/position variants")
    for name, material in variants.items():
        original = source_variants[name]
        for key in ("material_sha256", "quadrature", "target_volume_ml", "volume_ml", "method"):
            require(material[key] == original[key], f"Material plan not consistent with source: {name}/{key}")
        require(material["quadrature"] == 64, "Mixed quadrature")
        sources.add(resolve(material["material_file"], plan_path.parent), material["material_sha256"])
        volumes = np.array(material["volume_ml"])
        require(volumes.shape == (5,) and np.all(np.isfinite(volumes)) and np.all(volumes > 0),
                "Invalid integrated material volumes")
    # Read the saved geometry provenance, not a 4D motion trace.
    candidates_path = b / "stl_pose_candidates.local.json"
    sources.add(candidates_path)
    candidates = read_json(candidates_path)
    base = candidates["L1"]
    require(base["coordinate_system"] == "LPS" and base["length_unit"] == "mm", "Geometry units mismatch")
    for pose in POSES[1:]:
        candidate = candidates["L1_" + pose]
        expected = np.zeros(3)
        expected["xyz".index(pose[0])] = -5 if pose[1] == "m" else 5
        close(np.array(candidate["center_mm"]) - base["center_mm"], expected, "Position is not prescribed LPS +/-5 mm")
        for key in ("semi_axes_mm", "axes_matrix", "volume_ml"):
            close(candidate[key], base[key], "Translation changed nominal shape/volume")
    for pose in POSES:
        for level in LEVELS:
            close(variants[variant(pose, level)]["target_volume_ml"], base["volume_ml"]*level/100,
                  "Prescribed volume factors are not +/-5%")
    effective_path = b / "stl_effective_fit_q64_v1/effective_fit.local.json"
    effective = read_json(effective_path)
    sources.add(effective_path, cfg.get("motion_comparison", {}).get("effective_fit_sha256"))
    effective_row = next(r for r in effective["rows"] if r["id"] == "L1")
    old_cfg_path = b / "stl_pose_electrical_q64_v1.local.json"
    old_cfg = read_json(old_cfg_path)
    old_report_path = b / "stl_pose_electrical_q64_v1/electrical_check.local.json"
    old_report = read_json(old_report_path)
    sources.add(old_cfg_path, old_report["identity"]["config_sha256"])
    sources.add(old_report_path)
    for report in (old_report, effective):
        for key in SETTINGS:
            require(cfg[key] == report[key], f"Changed shared settings: {key}")
        for key in ("eit_model_sha256", "ttrkg_model_sha256", "sensitivity_sha256"):
            require(report["identity"][key] == identity[key], f"Changed source operator: {key}")
    require(effective["gain"] == 1, "Unexpected gain in source effective fit")
    expected_rho = dict(diagnostic4=np.array(old_report["rows"][0]["rho_ohm_m"]),
                        effectiveL1=np.array(effective_row["rho_ohm_m"]))
    require(expected_rho["diagnostic4"][3] == 4, "Diagnostic source is not rho_heart=4")
    jobs = cfg["jobs"]
    require(len(jobs) == 42 and {j["id"] for j in jobs} ==
            {f"{s}_{v}" for s in SCENARIOS for v in expected_variants}, "42-job contract mismatch")
    probes, missing, eit_missing, volumes_qc = {}, [], [], []
    for job in jobs:
        scenario, name = job["scenario"], job["variant"]
        require(scenario in SCENARIOS and name in variants and job["id"] == f"{scenario}_{name}", "Job identity mismatch")
        close(job["rho_ohm_m"], expected_rho[scenario], "Material scenario differs from its saved source")
        pose, level = next((p, v) for p in POSES for v in LEVELS if variant(p, v) == name)
        core = pose == "base" or level == 100
        require(not core or job["eit"], "Core EIT probe disabled in config")
        path = resolve(cfg["output_dir"], b) / (job["id"] + ".mat")
        if not path.exists():
            missing.append(job["id"])
            continue
        sources.add(path)
        data = loadmat(path, simplify_cells=True)
        row = data["row"]
        require(data["identity"] == identity, f"Saved identity mismatch: {path.name}")
        for key in ("id", "variant", "scenario"):
            require(row[key] == job[key], f"Saved job mismatch: {key}")
        material = variants[name]
        require(row["material_sha256"] == material["material_sha256"] and row["quadrature"] == 64,
                "Saved material identity differs")
        close(row["volume_ml"], material["volume_ml"], "Saved integrated volumes differ from plan")
        close(row["target_volume_ml"], material["target_volume_ml"], "Target volume differs")
        sigma = np.asarray(data["sigma"]).reshape(-1)
        require(sigma.shape == (5,) and np.all(sigma > 0), "Invalid sigma")
        close(row["rho_ohm_m"], 1/sigma, "rho/sigma units or values disagree")
        close(job["rho_ohm_m"], 1/sigma, "Job/sigma mismatch")
        k = np.asarray(data["Ktt"]).reshape(-1)
        z = float(data["Ztt"])
        require(k.shape == (5,), "Expected five TT sensitivities")
        close(k, row["Ktt_ohm_per_S_m"], "TT sensitivity arrays disagree")
        close(z, row["Ztt_ohm"], "TT impedance arrays disagree")
        v = np.asarray(data["V_eit"]).reshape(-1)
        s = np.asarray(data["S_eit"])
        require(all(np.isrealobj(a) for a in (v, s, k, data["Ztt"])),
                "This audit requires real saved FEM operators; complex phase must not be discarded")
        has_eit = v.size != 0
        require(has_eit == bool(row["eit_computed"]) == bool(job["eit"]), "EIT status inconsistent with config")
        require((v.shape == (208,) and s.shape == (208, 5)) if has_eit else s.size == 0,
                "EIT dimensions mismatch")
        require(np.isfinite(np.r_[v, s.ravel(), k, sigma, z]).all(), "Nonfinite FEM output")
        require(not core or has_eit, "Core EIT result is empty")
        if not has_eit:
            eit_missing.append(job["id"])
        for key in ("native_forward_relative_error", "reciprocity_relative_error"):
            if key in row:
                require(np.isfinite(row[key]) and row[key] < 1e-8, f"Failed numerical check: {key}")
        volume = float(np.asarray(row["volume_ml"])[3])
        probes[scenario, pose, level] = dict(V=v, S=s, K=k, Z=z, volume=volume, row=row)
        volumes_qc.append(dict(scenario=scenario, pose=pose, level=level, volume_ml=volume,
            target_volume_ml=material["target_volume_ml"], error_ml=volume-material["target_volume_ml"],
            error_pct=100*(volume/material["target_volume_ml"]-1), total_volume_ml=sum(row["volume_ml"]),
            eit_computed=has_eit, source_sha256=sources.add(path), material_sha256=material["material_sha256"]))
    # The old TT-only probes are an independent rerun comparison, never substituted for new EIT.
    old_checks = []
    for row in old_report["rows"]:
        key = next((("diagnostic4", p, v) for p in POSES[1:] for v in LEVELS if variant(p, v) == row["variant"]), None)
        if key in probes:
            current = probes[key]
            require(row["material_sha256"] == current["row"]["material_sha256"], "Old/new material mismatch")
            old_checks.append(dict(variant=row["variant"], old_Ztt_ohm=row["Ztt_ohm"],
                                   new_Ztt_ohm=current["Z"], difference_ohm=current["Z"]-row["Ztt_ohm"]))
    for row in volumes_qc:
        key = row["scenario"], "base", row["level"]
        if key in probes:
            baseline = probes[key]
            row["translation_volume_delta_ml"] = row["volume_ml"]-baseline["volume"]
            row["total_volume_delta_ml"] = row["total_volume_ml"]-sum(baseline["row"]["volume_ml"])
            require(abs(row["total_volume_delta_ml"]) < 1e-6, "Fixed-mesh total material volume not conserved")
    return cfg, plan, probes, dict(identity=identity, available=len(probes), expected=42,
        missing=missing, optional_EIT_absent=eit_missing, settings={k: cfg[k] for k in SETTINGS},
        units=UNITS, scenario_rho_ohm_m=expected_rho, integrated_volume_qc=volumes_qc,
        old_TT_comparison=old_checks, geometry_step_mm=5,
        TT_model_current_scaling_relative_error=current_error,
        volume_basis="plan and MAT integrated heart material volume, not target volume",
        unit_check_basis="Hashed MATLAB implementation: V_eit in V; Ktt=Jtt/current_A; Ztt=Vtt/current_A. Geometry provenance LPS/mm.")


def ratio(a, b):
    return float(a/b) if b != 0 else None


def relative_norm(a, b):
    return ratio(np.linalg.norm(a), np.linalg.norm(b))


def fwl(j, nuisance, y, w):
    """Residualize BOTH weighted target and weighted probe; no noise assumption."""
    n = w[:, None]*nuisance
    n /= np.linalg.norm(n, axis=0)
    projector = np.eye(len(w)) - n @ np.linalg.pinv(n)
    q = projector @ (w*j)
    yp = projector @ (w*y)
    require(np.linalg.norm(q) > 1e-12*np.linalg.norm(w*j), "Volume numerically lost in nuisance span")
    beta = float(q @ yp / (q @ q))
    fit = general_design(np.column_stack([j, nuisance]), y, w)
    require(fit["rank_identified"], "No-motion inverse is rank deficient")
    close(beta, float(fit["coefficients"][0, 0]), "FWL and simultaneous WLS disagree", atol=1e-8)
    return beta, fit, dict(
        nuisance_only_projected_norm=float(np.linalg.norm(yp)),
        volume_unique_fraction=float(np.linalg.norm(q)/np.linalg.norm(w*j)),
        fwl_projected_residual_norm=float(np.linalg.norm(yp-q*beta)),
        direct_probe_norm=float(np.linalg.norm(w*y)),
        direct_residual_after_volume_only_at_FWL_norm=float(np.linalg.norm(w*(y-j*beta))),
        full_inverse_residual_norm=float(np.linalg.norm(fit["weighted_residual"])),
        full_inverse_residual_fraction=relative_norm(fit["weighted_residual"].ravel(), w*y),
        nuisance_projection_removed_norm=float(np.linalg.norm(w*y-yp)))


def calculate(probes, radii, current_A):
    rows = {k: [] for k in ("volume_derivatives", "position_derivatives", "bounds", "identifiability",
                            "scalar_identifiability", "nuisance_equivalent_volume", "stack_comparison",
                            "false_volume_linear", "false_volume", "nonlinearity", "derivative_channels", "shift_channels")}
    details = {}
    for scenario in SCENARIOS:
        base = probes[scenario, "base", 100]
        minus, plus = (probes[scenario, "base", level] for level in (95, 105))
        dv = plus["volume"]-minus["volume"]
        require(dv > 0 and minus["volume"] < base["volume"] < plus["volume"], "Unordered volumes")
        jvol = (plus["V"]-minus["V"])/dv
        kvol = (plus["Z"]-minus["Z"])/dv
        for tissue, idx in (("soft_background", 0), ("lung", 1), ("heart_sigma_optional", 3)):
            rho = float(np.asarray(base["row"]["rho_ohm_m"])[idx])
            rows["nuisance_equivalent_volume"].append(dict(scenario=scenario, tissue=tissue,
                Ksigma_ohm_per_S_m=base["K"][idx], Kvol_ohm_per_ml=kvol,
                scalar_equivalent_volume_ml_per_S_m=ratio(base["K"][idx], kvol),
                scenario_rho_ohm_m=rho,
                scalar_equivalent_volume_ml_per_ohm_m=ratio(-base["K"][idx]/rho**2, kvol),
                uncertainty_magnitude=None,
                status="first_order_transfer_of_correction_error_not_pulse_amplitude_uncertainty_not_assumed_zero"))
        for pose in POSES:
            m, b, p = (probes[scenario, pose, level] for level in LEVELS)
            dm, dp = b["volume"]-m["volume"], p["volume"]-b["volume"]
            require(dm > 0 and dp > 0, "Unordered translated volume probes")
            kc = (p["Z"]-m["Z"])/(dm+dp)
            km, kp = (b["Z"]-m["Z"])/dm, (p["Z"]-b["Z"])/dp
            row = dict(scenario=scenario, pose=pose, minus_step_ml=dm, plus_step_ml=dp,
                Kvol_ohm_per_ml=kc, Kvol_minus_ohm_per_ml=km, Kvol_plus_ohm_per_ml=kp,
                Kvol_over_base=ratio(kc, kvol), Kvol_change_pct=100*(kc/kvol-1) if kvol else None,
                one_sided_disagreement_fraction=ratio(abs(kp-km), abs(kc)))
            if m["V"].size and p["V"].size:
                jc = (p["V"]-m["V"])/(dm+dp)
                jm, jp = (b["V"]-m["V"])/dm, (p["V"]-b["V"])/dp
                row.update(EIT_one_sided_disagreement_raw=relative_norm(jp-jm, jc),
                           EIT_Jvol_change_from_base_raw=relative_norm(jc-jvol, jvol))
            rows["volume_derivatives"].append(row)
        require(np.all(base["V"] != 0), "Zero base channel: relative metric undefined, no implicit floor")
        weight_modes = dict(raw=np.ones(208), relative_baseV=1/np.abs(base["V"]))
        jd, kd, shift = {}, {}, {}
        for correction in ("raw", "volume_corrected"):
            js, ks = [], []
            for axis in "xyz":
                m, p = probes[scenario, axis+"m5", 100], probes[scenario, axis+"p5", 100]
                ym, yp = m["V"]-base["V"], p["V"]-base["V"]
                zm, zp = m["Z"]-base["Z"], p["Z"]-base["Z"]
                dvm, dvp = m["volume"]-base["volume"], p["volume"]-base["volume"]
                if correction == "volume_corrected":
                    ym, yp = ym-jvol*dvm, yp-jvol*dvp
                    zm, zp = zm-kvol*dvm, zp-kvol*dvp
                jc, kc = (yp-ym)/10, (zp-zm)/10
                jm, jp, km, kp = -ym/5, yp/5, -zm/5, zp/5
                js.append(jc); ks.append(kc)
                rows["position_derivatives"].append(dict(scenario=scenario, correction=correction, axis=axis,
                    Kpos_ohm_per_mm=kc, Kpos_minus_ohm_per_mm=km, Kpos_plus_ohm_per_mm=kp,
                    Kvol_ohm_per_ml=kvol, apparent_volume_ml_per_mm=ratio(kc, kvol),
                    minus_5mm_bias_ml=ratio(zm, kvol), plus_5mm_bias_ml=ratio(zp, kvol),
                    minus_integrated_delta_ml=dvm, plus_integrated_delta_ml=dvp))
                for mode, w in weight_modes.items():
                    rows["nonlinearity"].append(dict(scenario=scenario, correction=correction, axis=axis, weight=mode,
                        EIT_one_sided_disagreement_fraction=relative_norm(w*(jp-jm), w*jc),
                        EIT_minus_vs_central_fraction=relative_norm(w*(jm-jc), w*jc),
                        EIT_plus_vs_central_fraction=relative_norm(w*(jp-jc), w*jc),
                        EIT_curvature_norm_per_mm2=float(np.linalg.norm(w*(yp+ym)/25)),
                        TT_one_sided_disagreement_fraction=ratio(abs(kp-km), abs(kc)),
                        TT_curvature_ohm_per_mm2=(zp+zm)/25))
                for sign, y, z, delta in (("m", ym, zm, dvm), ("p", yp, zp, dvp)):
                    shift[correction, axis+sign+"5"] = y, z, delta
            jd[correction], kd[correction] = np.column_stack(js), np.array(ks)
            for design_name, nuisance_indices in (("geometry_only", []), ("geometry_soft_lung", [0, 1]),
                                                  ("geometry_soft_lung_heart_sigma", [0, 1, 3])):
                scalar = np.r_[kvol, ks, base["K"][nuisance_indices]]
                normalized = np.sign(scalar)[None, :]
                rank = int(np.linalg.matrix_rank(normalized))
                ncols = len(scalar)
                singular = np.r_[np.linalg.svd(normalized, compute_uv=False), np.zeros(ncols-1)]
                rows["scalar_identifiability"].append(dict(scenario=scenario, correction=correction,
                    design=design_name, measurements=1, unknowns=ncols, rank=rank, nullity=ncols-rank,
                    full_column_rank=False, condition="unbounded_for_column_identification",
                    normalized_singular_values=json.dumps(singular.tolist()),
                    Kvol_ohm_per_ml=kvol, Kx_ohm_per_mm=ks[0], Ky_ohm_per_mm=ks[1], Kz_ohm_per_mm=ks[2],
                    Ksoft_ohm_per_S_m=base["K"][0] if 0 in nuisance_indices else None,
                    Klung_ohm_per_S_m=base["K"][1] if 1 in nuisance_indices else None,
                    Kheart_ohm_per_S_m=base["K"][3] if 3 in nuisance_indices else None,
                    status="one_scalar_cannot_separate_unknowns_zero_residual_not_recovery"))
            bound = ratio(np.linalg.norm(ks), abs(kvol))
            for radius in radii:
                rows["bounds"].append(dict(scenario=scenario, correction=correction, radius_mm=radius,
                    sensitivity_ml_per_mm=bound, bound_ml=radius*bound if bound is not None else None,
                    x_ml_per_mm=ratio(ks[0], kvol), y_ml_per_mm=ratio(ks[1], kvol), z_ml_per_mm=ratio(ks[2], kvol),
                    excludes_finite_shift_nonlinear_remainder=True,
                    status="local_linear_scenario_bound_not_measured_amplitude"))
            for channel in range(208):
                rows["derivative_channels"].append(dict(scenario=scenario, correction=correction,
                    channel_1based=channel+1, base_V=base["V"][channel], Jvol_V_per_ml=jvol[channel],
                    Jx_V_per_mm=js[0][channel], Jy_V_per_mm=js[1][channel], Jz_V_per_mm=js[2][channel],
                    Ssoft_V_per_S_m=base["S"][channel, 0], Slung_V_per_S_m=base["S"][channel, 1],
                    Sheart_V_per_S_m=base["S"][channel, 3]))
            for mode, w in weight_modes.items():
                for include_heart in (False, True):
                    nuisance_idx = [0, 1, 3] if include_heart else [0, 1]
                    nuisance = base["S"][:, nuisance_idx]
                    labels = ["volume_ml", "x_mm", "y_mm", "z_mm", "soft_sigma_S_m", "lung_sigma_S_m"]
                    if include_heart:
                        labels.append("heart_sigma_S_m")
                    a = np.column_stack([jvol, jd[correction], nuisance])
                    # Convert the entire TT row to volts BEFORE either weighting.
                    # current_A is the saved forward-model current, not fitted gain.
                    tt_row_V = current_A*np.r_[kvol, kd[correction], base["K"][nuisance_idx]]
                    tt_base_V = current_A*base["Z"]
                    require(tt_base_V != 0, "Zero TT base: relative stack metric undefined")
                    joint = np.vstack([a, tt_row_V])
                    joint_w = np.r_[w, 1. if mode == "raw" else 1/abs(tt_base_V)]
                    designs = {}
                    for stack, matrix, weights in (("EIT208", a, w), ("EIT208_TT1", joint, joint_w)):
                        design = general_design(matrix, np.zeros(len(weights)), weights)
                        designs[stack] = design
                        key = f"{scenario}/{correction}/{mode}/{stack}/{'with_heart_sigma' if include_heart else 'soft_lung'}"
                        details[key] = dict(columns=labels, TT_row_V_per_parameter=tt_row_V if stack=="EIT208_TT1" else None,
                            TT_base_V=tt_base_V if stack=="EIT208_TT1" else None, **design_summary(design))
                        for i, label in enumerate(labels):
                            rows["identifiability"].append(dict(scenario=scenario, correction=correction,
                                weight=mode, measurement_stack=stack, n_measurements=len(weights),
                                heart_sigma_nuisance=include_heart, parameter=label,
                                n_columns=len(labels), rank=design["rank"], full_rank=design["rank_identified"],
                                normalized_condition=design["normalized_condition"],
                                normalized_smallest_singular=design["normalized_singular_values"][-1],
                                unique_norm_fraction=design["partial_unique_fractions"][i],
                                unique_energy_fraction=design["partial_unique_fractions"][i]**2))
                    eit_d, joint_d = designs["EIT208"], designs["EIT208_TT1"]
                    for i, label in enumerate(labels):
                        rows["stack_comparison"].append(dict(scenario=scenario, correction=correction,
                            weight=mode, heart_sigma_nuisance=include_heart, parameter=label,
                            EIT_rank=eit_d["rank"], joint_rank=joint_d["rank"],
                            EIT_condition=eit_d["normalized_condition"], joint_condition=joint_d["normalized_condition"],
                            EIT_unique_fraction=eit_d["partial_unique_fractions"][i],
                            joint_unique_fraction=joint_d["partial_unique_fractions"][i],
                            unique_fraction_ratio=ratio(joint_d["partial_unique_fractions"][i], eit_d["partial_unique_fractions"][i]),
                            smallest_singular_ratio=ratio(joint_d["normalized_singular_values"][-1], eit_d["normalized_singular_values"][-1])))
                    for i, axis in enumerate("xyz"):
                        coefficient, _, metrics = fwl(jvol, nuisance, jd[correction][:, i], w)
                        rows["false_volume_linear"].append(dict(scenario=scenario, correction=correction,
                            weight=mode, heart_sigma_nuisance=include_heart, axis=axis,
                            FWL_false_volume_ml_per_mm=coefficient, **metrics))
                    for pose in POSES[1:]:
                        y, z, delta = shift[correction, pose]
                        beta, fit, metrics = fwl(jvol, nuisance, y, w)
                        coefficients = fit["coefficients"][:, 0]
                        tt_nuisance = float(base["K"][nuisance_idx] @ coefficients[1:])
                        rows["false_volume"].append(dict(scenario=scenario, correction=correction,
                            weight=mode, heart_sigma_nuisance=include_heart, pose=pose,
                            input_status="computed_shift_probe_not_observation", signed_shift_mm=-5 if pose[1]=="m" else 5,
                            intended_volume_change_ml=0, actual_integrated_delta_ml=delta,
                            false_volume_ml=beta, beta_minus_actual_delta_ml=beta-delta if correction=="raw" else beta,
                            delta_sigma_soft_S_m=coefficients[1], delta_sigma_lung_S_m=coefficients[2],
                            delta_sigma_heart_S_m=coefficients[3] if include_heart else None,
                            TT_probe_delta_ohm=z, TT_volume_prediction_ohm=kvol*beta,
                            TT_fitted_nuisance_projection_ohm=tt_nuisance,
                            TT_direct_residual_after_volume_only_ohm=z-kvol*beta,
                            TT_residual_after_volume_and_nuisance_ohm=z-kvol*beta-tt_nuisance,
                            TT_apparent_volume_from_scalar_ml=ratio(z, kvol), **metrics))
                        if not include_heart:
                            for channel in range(208):
                                rows["shift_channels"].append(dict(scenario=scenario, correction=correction,
                                    weight=mode, pose=pose, channel_1based=channel+1,
                                    computed_shift_V=y[channel], no_motion_prediction_V=fit["prediction"][channel, 0],
                                    no_motion_residual_V=fit["residual"][channel, 0]))
    return rows, details


def compact_findings(rows, qc):
    """Descriptive computational findings; no acceptance or physiology threshold."""
    out = {}
    for scenario in SCENARIOS:
        volumes = [r for r in rows["volume_derivatives"] if r["scenario"] == scenario]
        b = next(r for r in volumes if r["pose"] == "base")
        k = np.array([r["Kvol_ohm_per_ml"] for r in volumes])
        bias = [r for r in rows["position_derivatives"] if r["scenario"] == scenario and r["correction"] == "volume_corrected"]
        out[scenario] = dict(base_Kvol_ohm_per_ml=b["Kvol_ohm_per_ml"],
            base_volume_one_sided_disagreement_fraction=b["one_sided_disagreement_fraction"],
            Kvol_at_positions_min_max_ohm_per_ml=[float(k.min()), float(k.max())],
            Kvol_changes_sign_across_positions=bool(k.min() < 0 < k.max()),
            corrected_TT_apparent_volume_ml_per_mm={r["axis"]:r["apparent_volume_ml_per_mm"] for r in bias},
            corrected_finite_5mm_TT_bias_ml={r["axis"]:[r["minus_5mm_bias_ml"],r["plus_5mm_bias_ml"]] for r in bias},
            corrected_ball_bounds=[r for r in rows["bounds"] if r["scenario"]==scenario and r["correction"]=="volume_corrected"],
            max_integrated_volume_error_ml=max(abs(r["error_ml"]) for r in qc["integrated_volume_qc"] if r["scenario"]==scenario),
            interpretation="Conditional apparent-volume error of a volume-only scalar interpretation, not recovered cardiac motion or function")
    return out


def geometry_to_tissue_leakage(probes):
    """False tissue changes from pure geometry at fixed true conductivities.

    This is the sequential EIT tissue estimate -> TT correction diagnostic.
    It does not use the joint geometry inverse. All deltas refer to base v100.
    """
    rows, skipped, summary = [], [], {}
    indices = [0, 1, 3]
    for scenario in SCENARIOS:
        base = probes[scenario, "base", 100]
        minus, plus = (probes[scenario, "base", level] for level in (95, 105))
        kvol = (plus["Z"]-minus["Z"])/(plus["volume"]-minus["volume"])
        sigma = 1/np.asarray(base["row"]["rho_ohm_m"])[indices]
        for pose in POSES:
            for level in LEVELS:
                if pose == "base" and level == 100:
                    continue
                probe = probes[scenario, pose, level]
                if not probe["V"].size:
                    skipped.append(dict(scenario=scenario, pose=pose, level=level,
                                        reason="optional_EIT_not_computed"))
                    continue
                y = probe["V"]-base["V"]
                dz = probe["Z"]-base["Z"]
                kind = "pure_volume" if pose == "base" else (
                    "translation_with_integration_mismatch" if level == 100 else "volume_and_translation")
                for mode, w in (("raw", np.ones(208)), ("relative_baseV", 1/np.abs(base["V"]))):
                    fit = general_design(base["S"][:, indices], y, w)
                    ds = fit["coefficients"][:, 0]
                    drho = -ds/sigma**2
                    total_sigma = sigma+ds
                    finite_drho = np.full(3, np.nan)
                    positive = total_sigma > 0
                    finite_drho[positive] = 1/total_sigma[positive]-1/sigma[positive]
                    tt_terms = base["K"][indices]*ds
                    predicted = float(tt_terms.sum())
                    residual = dz-predicted
                    row = dict(scenario=scenario, pose=pose, level=level, geometry_kind=kind,
                        weight=mode, input_status="computed_geometry_only_probe_not_observation",
                        true_delta_sigma_S_m=0., intended_volume_factor=level/100,
                        integrated_volume_delta_ml=probe["volume"]-base["volume"],
                        source_variant=variant(pose, level),
                        tissue_inverse_rank=fit["rank"], tissue_inverse_full_rank=fit["rank_identified"],
                        tissue_inverse_normalized_condition=fit["normalized_condition"],
                        EIT_tissue_fit_residual_fraction=relative_norm(fit["weighted_residual"].ravel(), w*y),
                        TT_actual_geometry_delta_ohm=dz, TT_false_tissue_correction_ohm=predicted,
                        TT_after_false_tissue_correction_ohm=residual,
                        geometry_TT_fraction_subtracted=ratio(predicted, dz),
                        TT_actual_delta_relative_to_base=ratio(dz, base["Z"]),
                        apparent_volume_before_correction_ml=ratio(dz, kvol),
                        apparent_volume_after_false_correction_ml=ratio(residual, kvol),
                        apparent_volume_removed_ml=ratio(predicted, kvol), Kvol_ohm_per_ml=kvol,
                        finite_rho_all_positive=bool(positive.all()),
                        finite_rho_validity="algebraic_inverse_only_not_nonlinear_FEM_or_physiological_validation")
                    for i, tissue in enumerate(("soft", "lung", "heart")):
                        row.update({f"false_delta_sigma_{tissue}_S_m":ds[i],
                            f"false_delta_rho_{tissue}_first_order_ohm_m":drho[i],
                            f"false_delta_rho_{tissue}_finite_ohm_m":finite_drho[i],
                            f"sigma_plus_delta_{tissue}_positive":bool(positive[i]),
                            f"delta_sigma_over_base_sigma_{tissue}":ds[i]/sigma[i],
                            f"TT_false_correction_{tissue}_ohm":tt_terms[i]})
                    rows.append(row)
        for mode in ("raw", "relative_baseV"):
            group = [r for r in rows if r["scenario"]==scenario and r["weight"]==mode]
            motion = [r for r in group if r["level"]==100]
            summary[f"{scenario}/{mode}"] = dict(
                pure_volume_minus_plus=[r for r in group if r["pose"]=="base"],
                maximum_motion_by_absolute_TT_subtraction=max(motion,key=lambda r:abs(r["TT_false_tissue_correction_ohm"])),
                maximum_motion_by_absolute_remaining_apparent_volume=max(motion,key=lambda r:abs(r["apparent_volume_after_false_correction_ml"])),
                nonlinear_finite_rho_invalid_rows=sum(not r["finite_rho_all_positive"] for r in group))
    return rows, dict(status="computational_geometry_leakage_check", skipped=skipped, cases=summary,
        interpretation="A fixed-geometry EIT tissue inverse can assign volume/motion to delta_rho; using those estimates for TT correction can remove geometry or create spurious residuals. This is a computed counterexample/diagnostic, not observed double counting.")


def figures(output, rows):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for i, scenario in enumerate(SCENARIOS):
        ax = axes[i, 0]
        for correction, marker in (("raw", "o"), ("volume_corrected", "x")):
            selected = [r for r in rows["position_derivatives"] if r["scenario"]==scenario and r["correction"]==correction]
            ax.plot(range(3), [r["apparent_volume_ml_per_mm"] for r in selected], marker+"-", label=correction)
        ax.set(xticks=range(3), xticklabels=list("xyz"), ylabel="TT apparent volume [ml/mm]", title=scenario)
        ax.legend(fontsize=8); ax.axhline(0, color="grey", lw=.6)
        selected = [r for r in rows["volume_derivatives"] if r["scenario"]==scenario]
        axes[i, 1].plot(range(7), [1000*r["Kvol_ohm_per_ml"] for r in selected], "o-")
        axes[i, 1].set(xticks=range(7), xticklabels=POSES, ylabel="Kvol [mOhm/ml]", title="Volume sensitivity at seven positions")
        selected = [r for r in rows["position_derivatives"] if r["scenario"]==scenario and r["correction"]=="volume_corrected"]
        axes[i, 2].bar(np.arange(3)-.18, [r["minus_5mm_bias_ml"] for r in selected], .36, label="-5 mm")
        axes[i, 2].bar(np.arange(3)+.18, [r["plus_5mm_bias_ml"] for r in selected], .36, label="+5 mm")
        axes[i, 2].set(xticks=range(3), xticklabels=list("xyz"), ylabel="Finite-shift TT apparent volume [ml]", title="Volume-corrected, base Kvol")
        axes[i, 2].legend(fontsize=8)
    fig.suptitle("Conditional FEM probes: TT motion and volume sensitivity\nUnverified recording montage; no measured motion or SV/EF inference", fontsize=13)
    fig.savefig(output/"01_tt_motion_sensitivity.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    for i, scenario in enumerate(SCENARIOS):
        for mode, style, offset in (("raw", "o-", 0), ("relative_baseV", "x--", .25)):
            selected = [r for r in rows["geometry_to_tissue_leakage"] if r["scenario"]==scenario and r["weight"]==mode]
            volumes = [r for r in selected if r["pose"]=="base"]
            motion = [r for r in selected if r["level"]==100]
            if mode=="raw":
                axes[i, 0].bar(np.arange(2)-.25, [1000*r["TT_actual_geometry_delta_ohm"] for r in volumes], .25, label="Actual geometry TT")
                axes[i, 2].plot(range(6), [r["apparent_volume_before_correction_ml"] for r in motion], "k:", label="Before false correction")
            axes[i, 0].bar(np.arange(2)+offset, [1000*r["TT_false_tissue_correction_ohm"] for r in volumes], .25, label="False correction: "+mode)
            axes[i, 1].plot(range(6), [100*r["geometry_TT_fraction_subtracted"] for r in motion], style, label=mode)
            axes[i, 2].plot(range(6), [r["apparent_volume_after_false_correction_ml"] for r in motion], style, label=mode)
        axes[i, 0].set(xticks=range(2), xticklabels=["volume -5%", "volume +5%"], ylabel="TT delta or correction [mOhm]", title=scenario+": pure volume probes")
        axes[i, 1].set(xticks=range(6), xticklabels=POSES[1:], ylabel="Signed fraction subtracted [%]", title="Movement assigned to tissue correction")
        axes[i, 2].set(xticks=range(6), xticklabels=POSES[1:], ylabel="Residual / base Kvol [ml]", title="Apparent volume after false correction")
        for ax in axes[i]:
            ax.axhline(0, color="grey", lw=.6); ax.legend(fontsize=8)
    fig.suptitle("Geometry leakage into EIT tissue estimates and subsequent TT correction\nComputed probes at fixed true conductivity; not measured delta-rho(t), motion, SV or EF", fontsize=13)
    fig.savefig(output/"02_geometry_to_tissue_leakage.png", dpi=170); plt.close(fig)


def self_check():
    rng = np.random.default_rng(281026)
    n = rng.normal(size=(208, 2)); j = n[:, 0]*.7+rng.normal(size=208)*.1
    w = np.exp(rng.normal(size=208)*.2)
    y = 2.4*j+n@np.array([.04, -.07])
    beta, fit, _ = fwl(j, n, y, w)
    close(beta, 2.4, "Synthetic FWL recovery failure")
    beta2, _, _ = fwl(j, n, y-j*.15, w)
    close(beta2, 2.25, "Linear volume correction failure")
    # Adding an exactly dependent column must not be called identified.
    rank_test = general_design(np.column_stack([j, n, n[:, 0]]), y, w)
    require(not rank_test["rank_identified"], "Rank deficiency missed")
    return dict(status="passed", checks=["known volume with nonzero soft/lung nuisance",
        "FWL versus simultaneous WLS", "linear mismatch correction", "rank-deficient design detected"],
        volume_error_ml=beta-2.4, corrected_volume_error_ml=beta2-2.25,
        residual_norm=float(np.linalg.norm(fit["residual"])), physiological_validation=False)


def five_parameter_analysis(output, probes, qc, sources):
    """Append a conditional TT comparison without replacing earlier artifacts.

    x,y,z retain their corrected +/-5 mm secant derivatives. Radius is only a
    coordinate for the integrated L1 volume. rho uses the SI chain rule on the
    saved conductivity Jacobian. No forward solver is called here.
    """
    names = ["five_parameter_sensitivity.csv", "five_parameter_combined.csv",
             "five_parameter_checks.csv", "five_parameter_sensitivity.local.json",
             "03_five_parameter_sensitivity.png"]
    # Preserve every other file, including prior summaries with their historical
    # producer hashes. Do not relabel old results as produced by this version.
    preserved = {}
    for path in output.rglob("*"):
        if path.is_file() and path.name not in names:
            with path.open("rb") as stream:
                preserved[str(path.relative_to(output))] = hashlib.file_digest(stream, "sha256").hexdigest()
    rows, combined, checks, baselines = [], [], [], {}
    parameters = ["x", "y", "z", "rho_heart", "r_eq"]
    for scenario in SCENARIOS:
        base = probes[scenario, "base", 100]
        minus, plus = (probes[scenario, "base", level] for level in (95, 105))
        volume = base["volume"]  # integrated heart material volume, ml
        r_mm = (3*volume*1000/(4*np.pi))**(1/3)
        kvol = (plus["Z"]-minus["Z"])/(plus["volume"]-minus["volume"])
        rho = float(np.asarray(base["row"]["rho_ohm_m"])[3])
        sigma = 1/rho  # row/saved sigma agreement already checked by inspect_inputs
        ksigma = float(base["K"][3])  # ohm/(S/m), not voltage sensitivity
        jr = kvol*3*volume/r_mm  # ohm/mm; V remains ml in dV_ml/dr_mm
        jrho = -ksigma/rho**2  # ohm/(ohm*m)
        raw_position, corrected_position = [], []
        for axis in "xyz":
            m, p = (probes[scenario, axis+sign+"5", 100] for sign in ("m", "p"))
            raw = (p["Z"]-m["Z"])/10
            corrected = raw-kvol*(p["volume"]-m["volume"])/10
            raw_position.append(raw); corrected_position.append(corrected)
        derivatives = np.r_[corrected_position, jrho, jr]
        steps = np.array([1., 1., 1., .01*rho, .01*r_mm])
        contributions = derivatives*steps
        ranking = np.argsort(-np.abs(contributions), kind="stable")
        ranks = {int(index): rank+1 for rank, index in enumerate(ranking)}
        baselines[scenario] = dict(integrated_heart_volume_ml=volume,
            equivalent_volume_radius_mm=r_mm, rho_heart_ohm_m=rho, sigma_heart_S_m=sigma,
            Ksigma_heart_ohm_per_S_m=ksigma, Kvol_ohm_per_ml=kvol,
            radius_step_mm=.01*r_mm, radius_step_linear_volume_ml=.03*volume,
            radius_step_exact_coordinate_volume_ml=volume*(1.01**3-1),
            absolute_contribution_ranking=[parameters[i] for i in ranking])
        for i, parameter in enumerate(parameters):
            unit = "ohm*m" if parameter=="rho_heart" else "mm"
            rows.append(dict(scenario=scenario, parameter=parameter,
                signed_derivative_ohm_per_parameter=derivatives[i],
                derivative_units=f"ohm/({unit})", diagnostic_positive_step=steps[i], step_units=unit,
                signed_diagnostic_deltaZ_ohm=contributions[i], signed_diagnostic_deltaZ_mohm=1000*contributions[i],
                absolute_contribution_rank=ranks[i],
                step_basis="1% baseline rho" if parameter=="rho_heart" else (
                    "1% equivalent-volume radius" if parameter=="r_eq" else "+1 mm LPS"),
                derivative_before_volume_correction_ohm_per_mm=raw_position[i] if i<3 else None,
                derivative_volume_correction_ohm_per_mm=raw_position[i]-derivatives[i] if i<3 else None,
                equivalent_abs_delta_rho_heart_ohm_m=ratio(abs(contributions[i]),abs(jrho)),
                equivalent_signed_delta_rho_heart_ohm_m=ratio(contributions[i],jrho),
                equivalent_abs_delta_rho_heart_pct=100*abs(contributions[i]/jrho)/rho if jrho else None,
                equivalence_basis="linear same-magnitude TT response; not a measured rho change",
                model_status="conditional_computational_sensitivity_not_physiological_amplitude"))
        # Each state supplies one scalar equation for five parameters. The
        # dimensionless step coordinates make the row units homogeneous (ohm).
        rank = int(np.linalg.matrix_rank(contributions[None, :]))
        signed_sum = float(contributions.sum())
        box_max = float(np.abs(contributions).sum())
        combined.append(dict(scenario=scenario, parameter_order="x,y,z,rho_heart,r_eq",
            all_positive_steps_deltaZ_ohm=signed_sum, all_positive_steps_deltaZ_mohm=1000*signed_sum,
            independent_signed_step_box_min_mohm=-1000*box_max,
            independent_signed_step_box_max_mohm=1000*box_max,
            signs_for_maximum_positive_response=json.dumps(np.sign(contributions).astype(int).tolist()),
            cancellation_ratio=ratio(abs(signed_sum),box_max), measurements=1, parameters=5,
            rank=rank, nullity=5-rank, full_column_rank=False,
            normalized_nonzero_singular_value=float(np.linalg.norm(np.sign(derivatives))),
            remaining_singular_values="0,0,0,0", column_identification_condition="unbounded",
            status="first_order_superposition_only_no_mixed_FEM_or_probability_model"))
        # A finite difference of Ksigma*(sigma(rho)-sigma0) tests only the chain
        # rule and units. It is deliberately NOT a new nonlinear FEM response.
        for epsilon in (1e-2, 1e-3, 1e-4):
            h = epsilon*rho
            ds_plus, ds_minus = 1/(rho+h)-sigma, 1/(rho-h)-sigma
            d_fd = ksigma*(ds_plus-ds_minus)/(2*h)
            expected_fd = jrho/(1-epsilon**2)
            close(d_fd, expected_fd, "rho/sigma affine-surrogate finite difference failed", atol=1e-12)
            linear = jrho*h
            finite = ksigma*ds_plus
            close(finite, linear/(1+epsilon), "rho/sigma one-sided increment mismatch", atol=1e-12)
            checks.append(dict(scenario=scenario, check="sigma_to_rho_affine_surrogate_only",
                relative_rho_step=epsilon, delta_rho_ohm_m=h,
                analytic_derivative_ohm_per_ohm_m=jrho, central_difference_ohm_per_ohm_m=d_fd,
                relative_central_error=abs(d_fd/jrho-1) if jrho else None,
                expected_relative_central_error=epsilon**2/(1-epsilon**2),
                analytic_central_remainder_error=abs(d_fd-expected_fd),
                positive_step_linear_deltaZ_ohm=linear, positive_step_affine_sigma_deltaZ_ohm=finite,
                passed=True, new_forward_validation=False))
        h_r = r_mm*1e-4
        dvolume_fd = ((4*np.pi/3)*(r_mm+h_r)**3/1000 -
                      (4*np.pi/3)*(r_mm-h_r)**3/1000)/(2*h_r)
        close(dvolume_fd, 3*volume/r_mm*(1+1e-8/3), "Equivalent radius ml/mm conversion failed", atol=1e-8)
        checks.append(dict(scenario=scenario, check="equivalent_radius_volume_coordinate_only",
            analytic_dvolume_ml_per_mm=3*volume/r_mm, central_dvolume_ml_per_mm=dvolume_fd,
            relative_central_error=abs(dvolume_fd/(3*volume/r_mm)-1),
            passed=True, new_forward_validation=False))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    labels = ["x: +1 mm", "y: +1 mm", "z: +1 mm", "rho: +1%", "r_eq: +1%"]
    for i, scenario in enumerate(SCENARIOS):
        selected = [r for r in rows if r["scenario"]==scenario]
        values = [r["signed_diagnostic_deltaZ_mohm"] for r in selected]
        bars = axes[i, 0].bar(range(5), values, color=["#aa4938" if v<0 else "#267da8" for v in values])
        axes[i, 0].bar_label(bars, labels=[f"{v:+.4f}" for v in values], padding=3, fontsize=9)
        axes[i, 0].margins(y=.22)
        axes[i, 0].axhline(0, color="grey", lw=.7)
        axes[i, 0].set(xticks=range(5), xticklabels=labels, ylabel="Signed linear TT contribution [mOhm]", title=scenario)
        equivalent = [r["equivalent_abs_delta_rho_heart_ohm_m"] for r in selected if r["parameter"]!="rho_heart"]
        bars = axes[i, 1].bar(range(4), equivalent, color="#678457")
        axes[i, 1].bar_label(bars, labels=[f"{v:.5g}" for v in equivalent], padding=3, fontsize=9)
        axes[i, 1].margins(y=.22)
        axes[i, 1].set(xticks=range(4), xticklabels=[labels[j] for j in (0,1,2,4)],
            ylabel="Equivalent |delta rho_heart| [Ohm m]", title="Equal-magnitude local TT response")
    fig.suptitle("Five-parameter TT sensitivity at explicit diagnostic steps\nL1 equivalent-volume radius; no spherical FEM, measured amplitudes or probabilities", fontsize=13)
    fig.savefig(output/names[4], dpi=180); plt.close(fig)
    write_csv(output/names[0], rows)
    write_csv(output/names[1], combined)
    write_csv(output/names[2], checks)
    sources.unchanged()
    for relative, digest in preserved.items():
        with (output/relative).open("rb") as stream:
            require(hashlib.file_digest(stream,"sha256").hexdigest()==digest,
                    f"Prior output changed: {relative}")
    result = dict(schema="heart_motion_five_parameter_v1", status="conditional_computational_comparison_completed",
        input_QC=qc, source_hashes=sources.records, baselines=baselines, parameters=rows,
        combined=combined, transformation_checks=checks,
        formulas=dict(radius="r_eq_mm=(3*1000*V_ml/(4*pi))^(1/3)",
            J_r="Kvol_ohm_per_ml * 3*V_ml/r_eq_mm [ohm/mm]",
            J_rho="-Ktt_heart_ohm_per_S_m/rho_heart_ohm_m^2 [ohm/(ohm*m)]",
            J_position="(Z_plus-Z_minus-Kvol*(Vheart_plus_ml-Vheart_minus_ml))/10mm",
            diagnostic_contribution="signed raw physical derivative times explicit positive diagnostic step",
            equivalent_rho="abs(J_parameter*diagnostic_step)/abs(J_rho); first-order magnitude equality",
            combined="deltaZ=sum(J_i*delta_p_i); independent sign box gives +/-sum(abs(J_i*step_i))"),
        limitations=LIMITATIONS+[
            "Ranking uses 1mm translations, 1% radius and 1% rho chosen for diagnostics; changing steps can change the ordering. Derivatives with different units cannot be ranked directly.",
            "Equivalent-volume radius parametrizes the saved L1 volume law, not a new spherical FEM geometry or an anatomical chamber radius.",
            "The radius derivative inherits the +/-5% volume secant and its observed nonlinearity; positional derivatives inherit +/-5mm probes, not a new 1mm forward solve.",
            "sigma-to-rho differences evaluate an affine surrogate in conductivity and check only reparameterization, signs and units; they do not validate nonlinear tissue forward response.",
            "Joint action is first-order superposition: mixed geometric/material interactions were not computed. Signed box bounds are not probabilities or physiological ranges.",
            "One TT scalar has rank at most one for these five unknowns (nullity at least four). Tissue uncertainty and soft/lung pulses remain; this table does not assume them physiologically zero.",
        ], previous_outputs_preserved_sha256=preserved, previous_outputs_unchanged=True,
        output_files=names)
    write_json(output/names[3], result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--five-parameter-only", action="store_true",
                      help="Append five-parameter comparison; preserve all earlier analysis outputs")
    parser.add_argument("--radius-mm", nargs="+", type=float, default=[1., 5.])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    require(all(np.isfinite(args.radius_mm)) and all(r >= 0 for r in args.radius_mm), "Invalid scenario radius")
    sources = Sources()
    checks = self_check()
    try:
        cfg, plan, probes, qc = inspect_inputs(args.config, sources)
        sources.unchanged()
        qc.update(status="waiting_for_FEM" if qc["missing"] else "input_QC_passed",
                  numerical_self_checks=checks, source_hashes=sources.records, limitations=LIMITATIONS)
        if args.five_parameter_only:
            require(not qc["missing"], "Five-parameter addition requires the completed 42-probe input")
            result = five_parameter_analysis(args.output, probes, qc, sources)
            print(json.dumps(dict(status=result["status"], previous_outputs_unchanged=True,
                                  files=result["output_files"])))
            return 0
        write_json(args.output/"input_qc.local.json", qc)
        print(json.dumps(dict(status=qc["status"], available=qc["available"], expected=42,
                              optional_EIT_absent=len(qc["optional_EIT_absent"]))))
        if args.check_only or qc["missing"]:
            return 0 if args.check_only else 2
        rows, details = calculate(probes, args.radius_mm, cfg["current_A"])
        rows["geometry_to_tissue_leakage"], leakage_summary = geometry_to_tissue_leakage(probes)
        rows["integrated_volume_qc"] = qc["integrated_volume_qc"]
        rows["old_TT_comparison"] = qc["old_TT_comparison"]
        figures(args.output, rows)
        sources.unchanged()
        for name, values in rows.items():
            write_csv(args.output/(name+".csv"), values)
        summary = dict(schema="heart_motion_electrical_analysis_v1", status="conditional_computational_analysis_completed",
            primary_operational_target="Synchronous PulmoVista+TT with common ECG and one CT phase: EIT is required for rho0+delta_rho(t) tissue corrections ONLY; geometry and functions are inferred from corrected TT.",
            primary_TT_conclusion="At specified rho and tissue-pulse corrections, one scalar TT measurement still cannot identify four freely varying geometric unknowns. Rank <=1; zero TT residual is not recovery. Additional validated constraints would be needed. Tissue correction uncertainty remains nonzero/unknown.",
            research_reference="EIT-only and idealized joint EIT+TT geometric designs are supplemental diagnostics of a broader information set, explicitly outside the target inversion. Six/seven-column scalar checks show the cost of unconstrained correction errors.",
            conductivity_uncertainty=dict(baseline="Two conditional parameter scenarios; not a confidence interval or actual synchronized rho(t)",
                pulse="Soft/lung pulse correction is retained; its unknown estimation error is propagated by Ksigma/Kvol, not set to zero",
                measured_synchronous_rho_t_used=False, new_synchronous_experiment_obtained=False,
                baseline_and_pulse_uncertainty_assumed_zero=False, quantified_noise_or_covariance=False),
            primary_TT_findings=compact_findings(rows, qc),
            geometry_to_tissue_leakage=leakage_summary,
            units=UNITS, limitations=LIMITATIONS, input_qc=qc, designs=details, results=rows,
            formulas=dict(volume="Jvol=(V105-V095)/(integrated_ml105-integrated_ml095); same for Kvol and Ztt",
                motion="Jpos_i=(V_i+5-V_i-5)/10 mm; same for Kpos and Ztt",
                correction="DeltaV_corrected=DeltaV_raw-Jvol*(integrated_volume_shift-integrated_volume_base); TT uses Kvol",
                apparent_TT="Kpos_i/Kvol ml/mm; finite shift uses (Z_shift-Z_base)/Kvol",
                ball_bound="abs(apparent_delta_volume)<=r*norm(Kpos,2)/abs(Kvol), local linear scenario only",
                TT_scalar_nullspace="At fixed deltaZ, delta_volume=(deltaZ-Kpos.dot(displacement)-Ksoft*delta_sigma_soft-Klung*delta_sigma_lung[-Kheart*delta_sigma_heart])/Kvol; infinitely many choices unless additional validated constraints are available",
                correction_uncertainty="With epsilon_sigma=delta_sigma_true-delta_sigma_estimated, TT apparent-volume error is (Kpos.dot(d)+Ksoft*epsilon_soft+Klung*epsilon_lung[+Kheart*epsilon_heart])/Kvol. Locally epsilon_sigma=-epsilon_rho/rho0^2. Coefficients are conditional on uncertain baseline rho0; uncertainty magnitude is not assigned.",
                geometry_leakage="For each computed nonbase geometry probe at fixed true sigma: y=V_probe-V_base; false_delta_sigma=argmin||W*(y-S_base[:,soft,lung,heart]*delta_sigma)||. False_TT_correction=K_base.dot(false_delta_sigma); residual_TT=(Z_probe-Z_base)-false_TT_correction. First-order false_delta_rho=-false_delta_sigma/sigma0^2; finite inverse only for positive sigma0+false_delta_sigma.",
                weights="W=diag(ones) or diag(1/abs(base_V_eit)); W is row multiplier, no gain or offset",
                joint_stack="A_joint=[A_EIT; current_A*[Kvol,Kx,Ky,Kz,Ksoft,Klung,(Kheart)]]; base_joint=[V_EIT; current_A*Ztt]; apply raw or 1/abs(base_joint) row multipliers",
                FWL="N=W*[Ssoft,Slung,(optional Sheart)], M=I-N*pinv(N); q=M*W*Jvol; beta=q.T*M*W*y/(q.T*q)",
                residuals="Direct probe, nuisance-only projected probe, direct residual after volume, and full inverse residual reported separately",
                unique_fraction="norm((I-P_other_normalized_columns)*unit_column); energy fraction is its square",
                rank="SVD after weighted column L2 normalization; tolerance=max(shape)*eps*s_max"),
            runtime=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__, platform=platform.platform()),
            source_hashes=sources.records, output_files=[p.name for p in sorted(args.output.glob("*.csv"))]+[
                "01_tt_motion_sensitivity.png", "02_geometry_to_tissue_leakage.png"])
        # Channel arrays are kept in CSV; avoid duplicating 10k rows in the JSON summary.
        summary["results"] = {k:v for k,v in rows.items() if k not in ("derivative_channels", "shift_channels")}
        write_json(args.output/"analysis.local.json", summary)
        print(json.dumps(dict(status=summary["status"], output=str(args.output))))
        return 0
    except Exception as exc:
        write_json(args.output/"analysis_failure.local.json", dict(status="failed", error=repr(exc),
            source_hashes=sources.records, numerical_self_checks=checks, limitations=LIMITATIONS))
        raise


if __name__ == "__main__":
    sys.exit(main())
