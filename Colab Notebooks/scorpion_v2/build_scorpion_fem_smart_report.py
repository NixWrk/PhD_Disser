"""Read saved three-subject FEM only; export de-identified notebook/reader.

No CT/MAT/signal loading, field assembly, optimization, or MATLAB invocation.
Final publication requires 24 original heart results and six material-range
forward reports. --draft keeps missing results visible.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[1]
STEM = "33.43_Индивидуальная_КТ_FEM_проверка_трёх_участников_Scorpion"
ASSET = "fem_smart_20261026.summary.json"
SUBJECTS = ("georg", "nik", "yaroslav")
SUBJECT_NAMES = {"georg":"Георгий", "nik":"Ник", "yaroslav":"Ярослав"}
ACTIVE = (0, 1, 3, 4)
ROLES = {"ttrkg": (1, 0, 2), "lateral": (3, 2, 0)}
ROLE_NAMES = {"ttrkg": "ТТРКГ", "lateral": "Боковая сборка"}
MATERIALS = ("Фон", "Лёгкое", "Всё сердце", "Жир")
EXPECTED = tuple(f"{g}_q{q}" for g in ("wholeheart_ct", "sphere_v095", "sphere_v100", "sphere_v105") for q in (1024, 4096))


def records(value):
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return value
    raise ValueError("Expected scalar struct or list")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def assert_safe(value):
    """Check the actual serialization, including retained source metadata."""
    text = json.dumps(value, ensure_ascii=False, allow_nan=False)
    if re.search(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]", text):
        raise ValueError("Absolute drive/UNC path in export")
    prohibited = ("nodes_lps", "centre_lps", "nodes_mm", "boundary_face_ids", "raw_lps", "source_relative_path")
    if any(f'"{key}' in text for key in prohibited):
        raise ValueError("Individual spatial or raw-record fields in export")


class Snapshot:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.sources = {}

    def json(self, path, optional=False):
        path = Path(path)
        if optional and not path.is_file():
            return None
        payload = path.read_bytes()
        try:
            value = json.loads(payload.decode("utf-8-sig"))
        except (json.JSONDecodeError, UnicodeError):
            if optional:return None
            raise
        relative = path.resolve().relative_to(self.root).as_posix()
        self.sources[relative] = hashlib.sha256(payload).hexdigest()
        return value

    def verify(self):
        for path, expected in self.sources.items():
            if digest(self.root/path) != expected:
                raise ValueError("Input changed during report snapshot: " + path)


def case_parts(case):
    match = re.fullmatch(r"L(\d+)_TT([a-zA-Z0-9]+)", case)
    if not match:
        raise ValueError("Unsupported montage ID")
    return int(match[1]), match[2].replace("x", "×")


def scalar_transfers(t):
    matrix = np.asarray(t, float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Invalid saved transfer matrix")
    return {role: dict(own_ohm=float(matrix[i, j]), cross_ohm=float(matrix[i, other]))
            for role, (i, j, other) in ROLES.items()}


def joint_grid_diagnostics(scan, observations):
    """Descriptive own-drive grid minima for confirmed time-division operation."""
    if not scan["ready"]:return []
    fine=[r for r in scan["rows"] if r["rule"]==4096]
    points=defaultdict(dict)
    for r in fine:points[r["point_id"]][r["case_id"]]=r
    cases=sorted({r["case_id"] for r in fine},key=case_parts)
    output=[]
    for state in ("inhale","exhale"):
        obs={(r["case_id"],r["role"]):r["observed_median_ohm"] for r in observations if r["state"]==state}
        for scope,chosen in (("primary_pair",["L140_TT2x2"]),("all_cases",cases)):
            labels=[dict(case_id=case,role=role) for case in chosen for role in ROLES]
            if not all((r["case_id"],r["role"]) in obs for r in labels):
                raise ValueError("Measured montage/role missing for joint grid comparison")
            observed=np.array([obs[(r["case_id"],r["role"])] for r in labels],float)
            sampled=[]
            for point,by_case in sorted(points.items()):
                if any(case not in by_case for case in chosen):raise ValueError("Incomplete joint grid point")
                model=np.array([by_case[r["case_id"]]["transfers"][r["role"]]["own_ohm"] for r in labels],float)
                residual=model-observed
                sampled.append(dict(point_id=point,rho_ohm_m=by_case[chosen[0]]["rho_ohm_m"],
                    predicted_ohm=model.tolist(),residual_model_minus_observed_ohm=residual.tolist(),
                    euclidean_norm_ohm=float(np.linalg.norm(residual)),rms_ohm=float(np.sqrt(np.mean(residual**2)))))
            metric="euclidean_norm_ohm" if scope=="primary_pair" else "rms_ohm"
            minimum=min(sampled,key=lambda r:r[metric])
            primary_ranges={}
            if scope=="primary_pair":
                for i,role in enumerate(ROLES):
                    values=[p["L140_TT2x2"]["transfers"][role]["own_ohm"] for p in points.values()]
                    own_min=min(values)
                    primary_ranges[role]=dict(own_min_ohm=own_min,own_max_ohm=max(values),
                        observed_below_all_own=bool(observed[i]<own_min))
            output.append(dict(state=state,scope=scope,response_operator="own_ohm",excitation_mode="time_division",
                row_labels=labels,observed_ohm=observed.tolist(),sampled_points=len(sampled),
                metric=metric,minimum_on_sampled_grid=minimum,noise_weighted=False,
                primary_grid_ranges=primary_ranges,
                continuous_rectangle_excluded=False,individual_parameters_established=False))
    return output


def checked_lung_only_scenario(snap,base,subject,forward):
    """Verify the saved contribution adapter against each own finite baseline pair."""
    path=base/"heart_linear_predictions.local.json"
    linear=snap.json(path)
    if linear.get("schema")!="scorpion_heart_scenarios_v1" or linear["subject_alias"]!=subject:
        raise ValueError("Linear contribution adapter identity mismatch")
    expected_sources={str((base/"electrical/forward.local.json").resolve()),
        str((base/"heart_scenarios.local.json").resolve()),
        str((ROOT/"MATLAB_TRKG4_real_subjects/tools/scorpion_fem_heart_pilot.py").resolve())}
    if {str(Path(r["path"]).resolve()) for r in linear["sources"]}!=expected_sources:
        raise ValueError("Unexpected linear contribution source set")
    for source in linear["sources"]:
        if digest(source["path"])!=source["sha256"]:raise ValueError("Linear contribution source changed")
    snap.json(base/"heart_scenarios.local.json")
    by={(r["case_id"],r["rule"],r["state"]):r for r in records(forward["answers"])}
    rows=records(linear["rows"])
    if len(rows)*2!=len(by):raise ValueError("Incomplete finite inhale/exhale contribution pairs")
    for row in rows:
        inhale=by[(row["case_id"],row["rule"],"inhale")]
        exhale=by[(row["case_id"],row["rule"],"exhale")]
        ri=np.asarray(inhale["rho_ohm_m"]);re=np.asarray(exhale["rho_ohm_m"])
        if not np.array_equal(ri[[0,2,3,4]],re[[0,2,3,4]]) or ri[1]==re[1]:
            raise ValueError("Finite breathing scenario is not lung-only")
        ti=np.asarray(inhale["transfer_matrix_ohm"]);te=np.asarray(exhale["transfer_matrix_ohm"])
        if not np.array_equal(row["inhale_transfer_ohm"],ti) or not np.array_equal(row["exhale_transfer_ohm"],te) or not np.array_equal(row["exhale_minus_inhale_ohm"],te-ti):
            raise ValueError("Finite contribution pair changed from own baseline")
        if np.any(np.asarray(row["linear_background_only_ohm"])!=0) or not np.array_equal(row["linear_lung_only_ohm"],row["linear_delta_ohm"]):
            raise ValueError("Inconsistent zero background contribution")
    producer=ROOT/"MATLAB_TRKG4_real_subjects/tools/scorpion_fem_heart_pilot.py"
    return dict(source_count=3,checked_finite_pairs=len(rows),only_lung_resistivity_changes=True,
        background_only_zero_by_construction=True,producer_relative_path=producer.relative_to(ROOT).as_posix(),
        producer_sha256=digest(producer),physiological_background_constancy_established=False)


def center_identifiability(answers):
    """Own 1.35-heart Jacobian only; no transfer from the reference baseline."""
    import importlib.util
    path=ROOT/"MATLAB_TRKG4_real_subjects/tools/scorpion_fem_sensitivity.py"
    spec=importlib.util.spec_from_file_location("smart_signed_sensitivity",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    selected=[r for r in answers if r["rule"]==4096 and r["geometry"]=="wholeheart_ct" and
              r["rho_ohm_m"][0]==2.5 and r["rho_ohm_m"][1]==7 and r["rho_ohm_m"][3]==1.35]
    if not selected:raise ValueError("New material center is missing")
    if len({r["case_id"] for r in selected})!=len(selected):raise ValueError("Duplicate center case")
    result=[]
    for subset,roles in (("lateral",("lateral",)),("ttrkg",("ttrkg",)),("joint",tuple(ROLES)),("primary_pair",tuple(ROLES))):
        matrix=[];labels=[]
        for r in sorted(selected,key=lambda v:v["case_id"]):
            if subset=="primary_pair" and r["case_id"]!="L140_TT2x2":continue
            t=np.asarray(r["transfer_matrix_ohm"],float);j=np.asarray(r["jacobian_logrho"],float)
            if j.shape!=(4,4,4) or r["active_material_indices"]!=[1,2,4,5]:raise ValueError("Wrong range Jacobian order")
            for role in roles:
                i,own,_=ROLES[role]
                if t[i,own]==0:raise ValueError("Zero range-center transfer cannot be normalized")
                matrix.append(j[i,own,:]/t[i,own]);labels.append(dict(case_id=r["case_id"],role=role))
        a=np.asarray(matrix,float)
        result.append(dict(subset=subset,row_labels=labels,
            all_four=module.matrix_diagnostics(a),soft_lung_only=module.matrix_diagnostics(a[:,[0,1]]),
            full_parameters=["soft","lung","wholeheart","fat"],
            reduced_parameters=["soft","lung"],reduced_fixed_parameters=["wholeheart=1.35","fat=reference","bone=reference"],
            normalization="signed diag(1/Z) Jlog at own scenario center",
            practical_individual_parameters_established=False))
    return dict(rho_center_ohm_m=[2.5,7,1.35],rho_center_order=["soft","lung","wholeheart"],
                matrices=result,analysis_relative_path=path.relative_to(ROOT).as_posix(),analysis_sha256=digest(path))


def aggregate_comparisons(forward):
    groups = defaultdict(list)
    excluded = defaultdict(int)
    for row in records(forward["comparison"]):
        if row["status"] != "predicted":
            excluded[row.get("exclusion_reason", "unspecified")] += 1
            continue
        if row.get("fine_Z_ohm") is None or row["fine_Z_ohm"] == []:
            raise ValueError("q4096 comparison is missing")
        groups[(row["case_id"], row["state"], row["role"])].append(row)
    result = []
    for (case, state, role), rows in sorted(groups.items()):
        size, variant = case_parts(case)
        observed = [float(r["observed_Z_ohm"]) for r in rows]
        predicted = [float(r["fine_Z_ohm"]) for r in rows]
        if not np.allclose(predicted, predicted[0], rtol=0, atol=1e-10):
            raise ValueError("Multiple inconsistent predictions for one montage")
        result.append(dict(case_id=case, size_mm=size, tt_variant=variant, state=state, role=role,
            n=len(rows), observed_median_ohm=float(np.median(observed)), observed_min_ohm=min(observed),
            observed_max_ohm=max(observed), predicted_own_ohm=predicted[0],
            median_residual_own_ohm=predicted[0]-float(np.median(observed)),
            annotation_statuses=sorted({r["annotation_status"] for r in rows}),
            all_integration_passed=all(r["integration_passed"] for r in rows)))
    return result, dict(excluded)


def heart_own_controls(rows):
    by={(r["geometry"],r["rule"]):r for r in rows}
    expected={(g,q) for g in ("wholeheart_ct","sphere_v095","sphere_v100","sphere_v105") for q in (1024,4096)}
    if len(rows)!=8 or set(by)!=expected:raise ValueError("Incomplete own heart contrast controls")
    controls={}
    for role in ROLES:
        get=lambda g,q:by[(g,q)]["transfers"][role]["own_ohm"]
        shape=lambda q:get("sphere_v100",q)-get("wholeheart_ct",q)
        volume=lambda q:get("sphere_v105",q)-get("sphere_v095",q)
        controls[role]=dict(equal_volume_shape_effect_ohm=shape(4096),
            minus_to_plus_volume_response_ohm=volume(4096),
            sphere_v100_own_q_difference_ohm=get("sphere_v100",4096)-get("sphere_v100",1024),
            wholeheart_own_q_difference_ohm=get("wholeheart_ct",4096)-get("wholeheart_ct",1024),
            shape_contrast_q_difference_ohm=shape(4096)-shape(1024),
            volume_contrast_q_difference_ohm=volume(4096)-volume(1024))
    return controls


def heart_results(snap, base, subject):
    directory = base/"heart_pilot"
    plan = snap.json(directory/"plan.local.json", optional=True)
    result = dict(ready=False, completed=0, expected=8, missing=[], rows=[], volume_reference_ml=None)
    if plan is None:
        result["missing"] = list(EXPECTED)
        return result
    if plan["subject_alias"] != subject:
        raise ValueError("Heart plan subject mismatch")
    plan_sha = snap.sources[(directory/"plan.local.json").relative_to(snap.root).as_posix()]
    result["volume_reference_ml"] = plan["geometry"]["sphere"]["volume_ml"]
    geometry = {r["id"]: r for r in plan["geometry"]["rows"]}
    for variant in EXPECTED:
        report = snap.json(directory/("solve_"+variant)/"report.local.json", optional=True)
        if report is None:
            result["missing"].append(variant)
            continue
        if report.get("schema") != "scorpion_heart_forward_v1" or not report.get("FEM_was_run"):
            raise ValueError("Uncompleted heart solve")
        if report["subject_alias"] != subject or report["identity"]["plan_sha256"] != plan_sha:
            raise ValueError("Heart result belongs to another participant/plan")
        rows = [r for r in records(report["answers"]) if r["case_id"] == "L140_TT2x2"]
        if len(rows) != 1:
            raise ValueError("Primary montage missing in heart solve")
        row = rows[0]
        if row["system_checks"]["system_relative_residual"] >= 1e-7 or row["system_checks"]["reciprocity_ohm"] >= 1e-7:
            raise ValueError("Heart system checks failed")
        result["rows"].append(dict(variant=variant, geometry=variant.rsplit("_q", 1)[0],
            rule=row["rule"], volume_integrated_ml=geometry[variant]["volume_ml_by_material"][3],
            target_volume_ml=geometry[variant]["target_heart_volume_ml"],
            transfers=scalar_transfers(row["transfer_matrix_ohm"]),
            qc={k: row["system_checks"][k] for k in ("system_relative_residual", "reciprocity_ohm")}))
    result["completed"] = len(result["rows"])
    result["ready"] = result["completed"] == 8
    adapter=snap.json(base/"heart_pilot_summary.local.json",optional=True)
    if adapter is not None:
        if adapter.get("schema")!="scorpion_heart_pilot_summary_v1" or adapter["subject_alias"]!=subject:
            raise ValueError("Heart summary adapter mismatch")
        for source in adapter["sources"]:
            if digest(source["path"])!=source["sha256"]:raise ValueError("Heart adapter source changed")
        result["adapter_status"]=adapter["status"]
    else:result["adapter_status"]="pending"
    result["own_controls"]=heart_own_controls(result["rows"]) if result["ready"] else {}
    return result


def range_results(snap, base, subject, action, baseline):
    """Adapter stays explicitly pending until the producer schema is available."""
    directory = base/"material_range"/action
    plan = snap.json(directory/"plan.local.json", optional=True)
    forward = snap.json(directory/"results/forward.local.json", optional=True)
    out = dict(action=action, ready=False, status="pending", rows=[])
    if forward is None:
        return out
    out["producer_schema"] = forward.get("schema")
    if forward.get("schema") != "scorpion_material_range_forward_v1":
        out["status"] = "schema_adapter_pending"
        return out
    if not forward.get("FEM_was_run") or forward.get("subject_alias") != subject or plan is None:
        raise ValueError("Material-range result not complete or not own subject")
    if forward.get("frequency_hz")!=100000 or plan.get("frequency_hz")!=100000 or plan.get("subject_alias")!=subject:
        raise ValueError("Material-range frequency/subject mismatch")
    for key in ("mesh_sha256","cem_sha256","reference_sha256","registry_sha256",
                "field_kernel_sha256","scale_kernel_sha256","response_kernel_sha256"):
        if forward["identity"].get(key)!=baseline["identity"][key] or plan["identity"].get(key)!=baseline["identity"][key]:
            raise ValueError("Material-range source identity mismatch: "+key)
    plan_sha = snap.sources[(directory/"plan.local.json").relative_to(snap.root).as_posix()]
    if forward.get("identity", {}).get("plan_sha256") != plan_sha:
        raise ValueError("Material-range plan identity mismatch")
    # Known numeric fields only. No recursive copying of paths/identity objects.
    answers=records(forward.get("answers", []))
    if forward.get("action")!=action or len(answers)!=len(plan["rows"])*len(plan["case_ids"]):
        raise ValueError("Incomplete range row/case coverage")
    expected={(p["id"],case) for p in plan["rows"] for case in plan["case_ids"]}
    actual=[(r["row_id"],r["case_id"]) for r in answers]
    if len(set(actual))!=len(actual) or set(actual)!=expected:
        raise ValueError("Duplicate or missing material-range row/case")
    planned={p["id"]:p for p in plan["rows"]}
    reference_rho=records(baseline["answers"])[0]["rho_ohm_m"]
    for row in answers:
        if "transfer_matrix_ohm" not in row or "rho_ohm_m" not in row:
            out["status"] = "schema_adapter_pending"
            return out
        rho = list(map(float, row["rho_ohm_m"]))
        if len(rho) != 5 or rho[3] != 1.35:
            raise ValueError("Expected explicit new uniform-heart 1.35 scenario")
        p=planned[row["row_id"]]
        if rho!=p["rho_ohm_m"] or any(row[k]!=p[k] for k in ("rule","geometry","operator_id","point_id")):
            raise ValueError("Material-range answer differs from its planned scenario")
        if rho[2]!=reference_rho[2] or rho[4]!=reference_rho[4]:
            raise ValueError("Material-range fixed bone/fat changed from the reference")
        if row["port_order"]!=["TT_drive","TT_measure","lateral_drive","lateral_measure"]:
            raise ValueError("Material-range port order mismatch")
        result = dict(case_id=row["case_id"], rule=row["rule"], rho_ohm_m=rho,
            transfers=scalar_transfers(row["transfer_matrix_ohm"]))
        for key in ("variant", "geometry", "volume_ml", "volume_integrated_ml", "point_id", "row_id"):
            if key in row:
                result[key] = row[key]
        out["rows"].append(result)
    if not out["rows"]:
        raise ValueError("Empty material-range forward")
    out.update(ready=True, status="completed")
    checks=records(forward["system_checks"])
    if len(checks)!=len(answers) or any(c["system_relative_residual"]>=1e-7 or c["reciprocity_ohm"]>=1e-7 for c in checks):
        raise ValueError("Range system QC failed")
    out["derivative_checks"]={k:v for k,v in forward["derivative_checks"].items() if k in (
        "performed","passed","scope","relative_errors","relative_tolerance","case_id","row_id","rule")}
    out["quadrature"]=[{k:r[k] for k in ("point_id","geometry","case_id","evaluated","passed",
        "ttrkg_difference_ohm","lateral_difference_ohm")} for r in records(forward["q_checks"])]
    evaluated=[r for r in out["quadrature"] if r["evaluated"]]
    if len(evaluated)!=sum(p["rule"]==1024 for p in plan["rows"])*len(plan["case_ids"]) or not all(r["passed"] for r in evaluated):
        raise ValueError("Material-range planned quadrature control not passed")
    out["scenario_identity_sha256"]=plan_sha
    if action=="scan":
        if not forward["derivative_checks"]["performed"] or not forward["derivative_checks"]["passed"]:
            raise ValueError("New center FD not passed")
        out["center_identifiability"]=center_identifiability(answers)
    else:
        pilot=snap.json(base/"heart_pilot/plan.local.json")
        volumes={r["id"]:r["volume_ml_by_material"][3] for r in pilot["geometry"]["rows"]}
        for row in out["rows"]:
            row["volume_integrated_ml"]=volumes[f"{row['geometry']}_q{row['rule']}"]
        out["own_controls"]=heart_own_controls(out["rows"])
        scan=snap.json(base/"material_range/scan/results/forward.local.json")
        links=[]
        for rule in (1024,4096):
            whole=next(r for r in answers if r["geometry"]=="wholeheart_ct" and r["rule"]==rule)
            matching=[r for r in records(scan["answers"]) if r["case_id"]==whole["case_id"] and
                r["geometry"]=="wholeheart_ct" and r["rule"]==rule and r["rho_ohm_m"]==whole["rho_ohm_m"]]
            if len(matching)!=1:raise ValueError("Matching own scan-center control missing")
            a=np.asarray(whole["transfer_matrix_ohm"]);b=np.asarray(matching[0]["transfer_matrix_ohm"])
            links.append(dict(rule=rule,transfer_matrix_exact=bool(np.array_equal(a,b)),
                max_abs_transfer_difference_ohm=float(np.max(np.abs(a-b)))))
        out["wholeheart_scan_center_links"]=links
    return out


def collect(model_root):
    snap = Snapshot(model_root)
    subjects = []
    for index, subject in enumerate(SUBJECTS, 1):
        base = snap.root/subject
        forward_path = base/"electrical/forward.local.json"
        forward = snap.json(forward_path)
        if forward.get("schema") != "scorpion_own_CT_forward_v2" or not forward.get("FEM_was_run"):
            raise ValueError("Completed v2 baseline required")
        if forward["subject_alias"] != subject or forward["frequency_hz"] != 100000 or forward.get("limited_run"):
            raise ValueError("Own full 100kHz baseline required")
        source_sha = snap.sources[forward_path.relative_to(snap.root).as_posix()]
        reference=snap.json(base/"reference_materials.local.json")
        if snap.sources[(base/"reference_materials.local.json").relative_to(snap.root).as_posix()]!=forward["identity"]["reference_sha256"]:
            raise ValueError("Reference profile changed after baseline")
        cem_manifest=snap.json(base/"cem/cem_contacts_manifest.local.json")
        cem_input_path=Path(cem_manifest["input_json"])
        cem_input=snap.json(cem_input_path)
        if snap.sources[cem_input_path.resolve().relative_to(snap.root).as_posix()]!=cem_manifest["input_sha256"]:
            raise ValueError("CEM input changed after contact construction")
        impedances=sorted({float(c["z_contact_ohm_m2"]) for c in cem_input["contact_catalog"]})
        impedance_sources=sorted({c["z_contact_source"] for c in cem_input["contact_catalog"]})
        if not cem_manifest["CEM_contacts_QC_passed"] or impedances!=[.01] or not all(impedance_sources):
            raise ValueError("Unexpected CEM contact impedance scenario")
        sensitivity_path = base/("sensitivity_v2" if (base/"sensitivity_v2/sensitivity.local.json").exists() else "sensitivity")/"sensitivity.local.json"
        sensitivity = snap.json(sensitivity_path)
        if sensitivity["subject_alias"] != subject or sensitivity["source"]["sha256"] != source_sha:
            raise ValueError("Stale or different-subject sensitivity")
        comparison, excluded = aggregate_comparisons(forward)
        role_evidence=None
        if subject=="georg":
            inputs_path=base/"electrical_inputs.local.json"
            inputs=snap.json(inputs_path)
            if snap.sources[inputs_path.relative_to(snap.root).as_posix()]!=forward["identity"]["inputs_sha256"]:
                raise ValueError("Observation role evidence changed after baseline")
            evidence={r["record_id"]:r["role_evidence"] for r in inputs["observations"]}
            selected=[evidence[f"SCV2-G-{k:03d}"] for k in range(7,11)]
            if not all(e["ttrkg"]["variant"]=="1x1" and e["ttrkg"]["status"]=="author_confirmed" for e in selected):
                raise ValueError("Late TT1x1 author confirmation missing")
            if not all(e["status"]=="late_block_continuity_protocol_assumption" for e in selected[1:]):
                raise ValueError("Late physical-role assumption status changed")
            role_evidence=dict(tt1x1_007_010_author_confirmed=True,
                physical_roles_008_010_are_connection_continuity_assumption=True)
        breathing = defaultdict(list)
        for row in records(forward.get("breathing_differences", [])):
            breathing[(row["case_id"], row["role"])].append(row)
        breath = [dict(case_id=k[0], role=k[1], n=len(rows),
            observed_delta_ohm=float(np.median([r["observed_delta_Z_ohm"] for r in rows])),
            predicted_delta_ohm=float(rows[0]["predicted_delta_Z_ohm"])) for k, rows in sorted(breathing.items())]
        primary = [r for r in records(forward["answers"]) if r["case_id"] == "L140_TT2x2" and r["state"] == "inhale" and r["rule"] == 4096]
        if len(primary) != 1:
            raise ValueError("One primary baseline required")
        primary = primary[0]
        exhale=next(r for r in records(forward["answers"]) if r["case_id"]=="L140_TT2x2" and r["state"]=="exhale" and r["rule"]==4096)
        transfers = scalar_transfers(primary["transfer_matrix_ohm"])
        primary_comparison = {role: dict(v) for role, v in transfers.items()}
        for role, control in primary_comparison.items():
            observations = [r for r in comparison if r["case_id"] == "L140_TT2x2" and r["state"] == "inhale" and r["role"] == role]
            control["observed_median_ohm"] = observations[0]["observed_median_ohm"] if observations else None
        j = np.asarray(primary["jacobian_logrho"], float)
        basis = {role: (.01*j[i, own, :]).tolist() for role, (i, own, _) in ROLES.items()}
        fd = forward["derivative_checks"]
        checks = records(forward["system_checks"])
        mesh=snap.json(base/"materials/mesh_quality.local.json")
        if not mesh["passed"] or mesh["mesh_sha256"]!=forward["identity"]["mesh_sha256"]:
            raise ValueError("Baseline mesh QC identity mismatch")
        placement = snap.json(base/"placement_prepared/contact_plan.local.json")
        negative = []
        for row in records(placement["scenarios"]):
            p = row["placement"]
            distances = [r["projection_distance_mm"] for r in records(p["shift_details"])]
            negative.append(dict(offset_mm=p["requested_offset_mm"], passed=row["CEM_contacts_QC_passed"],
                failure_identifier=row.get("failure", {}).get("identifier"), projection_min_mm=min(distances),
                projection_max_mm=max(distances)))
        subjects.append(dict(id=SUBJECT_NAMES[subject], baseline_materials_inhale_ohm_m=primary["rho_ohm_m"],
            baseline_materials_exhale_ohm_m=exhale["rho_ohm_m"],
            tissue_reference=dict(frequency_hz=reference["frequency_hz"],registry_sha256=reference["registry_sha256"],
                source="Gabriel frequency entries in tissue_literature_v2.json",individual_fitted=False),
            contact_impedance_ohm_m2=impedances[0],contact_impedance_is_measured=False,
            contact_impedance_sources=impedance_sources,
            segmentation=dict(tool="TotalSegmentator",version="2.18.0",individual_mask_anatomical_validation=False),
            mesh_qc={k:mesh[k] for k in ("nodes","tetrahedra","passed","mean_ratio_minimum","edge_condition_maximum")},
            comparison=comparison, excluded_observations_by_reason=excluded, breath=breath,
            observation_role_evidence=role_evidence,
            breathing_scenario_checks=checked_lung_only_scenario(snap,base,subject,forward),
            sensitivity=[dict(state=a["state"], subset=a["subset"], rule=a["rule"], row_labels=a["row_labels"],
                matrix=a["matrices"]["dimensionless_signed_elasticity"],
                quadrature=a["quadrature_control"]["matrices"]["dimensionless_signed_elasticity"]) for a in sensitivity["analyses"]],
            basis_delta_logrho=.01, basis_delta_Z_ohm=basis, primary_transfers=transfers, primary_comparison=primary_comparison,
            qc=dict(system_checks=len(checks), max_relative_residual=max(r["system_relative_residual"] for r in checks),
                max_reciprocity_ohm=max(r["reciprocity_ohm"] for r in checks),
                integration_passed=forward["integration_passed"], fd_passed=fd["passed"],
                fd_max_relative_error=max(fd["relative_errors"]), fd_tolerance=fd["tolerance"],
                fd_case_id=fd["case_id"], fd_state=fd["state"], mesh_convergence=False,
                practical_identifiability="not_established"), placement=negative,
            heart=heart_results(snap, base, subject),
            material_range={action: range_results(snap, base, subject, action, forward) for action in ("scan", "heart")}))
    snap.verify()
    for subject in subjects:
        subject["material_range"]["scan"]["joint_grid_diagnostics"]=joint_grid_diagnostics(
            subject["material_range"]["scan"],subject["comparison"])
    summary = dict(schema="scorpion_fem_smart_reader_v1", generated_utc=datetime.now(timezone.utc).isoformat(),
        subjects=subjects, sources=[dict(relative_path=p, sha256=h) for p, h in sorted(snap.sources.items())],
        generator=dict(relative_path=Path(__file__).relative_to(ROOT).as_posix(), sha256=digest(__file__)),
        frequency_hz=100000, own_drive_is_scenario=False, source_relative_phase="not_applicable_time_division",
        author_overlay=dict(mode="time_division",author_confirmation="2026-10-26",phase_not_applicable=True,
            confirmation_source="direct_author_confirmation",own_drive_matches_confirmed_mode=True,
            cross_transfers_summed_into_prediction=False,original_source_reports_unchanged=True),
        heart_results_completed=sum(s["heart"]["completed"] for s in subjects), heart_results_expected=24,
        material_range_completed=sum(r["ready"] for s in subjects for r in s["material_range"].values()),
        material_range_expected=6, mesh_convergence=False, clinical_validation=False, stroke_volume_estimated=False,
        reader_tables_default_open=False)
    summary["final_inputs_complete"] = summary["heart_results_completed"] == 24 and summary["material_range_completed"] == 6
    assert_safe(summary)
    return summary


def table_html(headers, rows, caption):
    def cell(value):
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.6g}".replace(".", ",")
        return html.escape(str(value))
    return '<details><summary>'+html.escape(caption)+'</summary><div class="scientific-table"><table><thead><tr>'+''.join('<th>'+cell(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+cell(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div></details>'


def qc_table(summary):
    return table_html(["Участник", "Систем", "Макс. невязка", "Взаимность, Ом", "FD: макс. относительная ошибка", "q1024/q4096"],
        [[s["id"], s["qc"]["system_checks"], s["qc"]["max_relative_residual"], s["qc"]["max_reciprocity_ohm"],
          s["qc"]["fd_max_relative_error"], "Пройден" if s["qc"]["integration_passed"] else "Не пройден"] for s in summary["subjects"]],
        "Таблица 1. Контроль сохранённых базовых расчётов")


def placement_table(summary):
    return table_html(["Участник", "Сдвиг Y, мм", "Проекция: минимум, мм", "Проекция: максимум, мм", "Статус"],
        [[s["id"], p["offset_mm"], p["projection_min_mm"], p["projection_max_mm"], p["failure_identifier"] or "Пройден"]
         for s in summary["subjects"] for p in s["placement"]], "Таблица 2. Отказы переноса при прежнем допуске проекции 1 мм")


def drive_table(summary):
    return table_html(["Участник", "Отведение", "Измерение, Ом", "Прогноз own_drive, Ом", "Перекрёстная передача cross, Ом"],
        [[s["id"], ROLE_NAMES[role], m["observed_median_ohm"], m["own_ohm"], m["cross_ohm"]]
         for s in summary["subjects"] for role, m in s["primary_comparison"].items()],
        "Таблица 3. Временное разделение: собственный прогноз и отдельный контроль перекрёстных передач, 140 мм/ТТРКГ 2×2 на вдохе")


def status_table(summary):
    return table_html(["Участник", "Замещение сердца: готово / 8", "Сканирование диапазона", "Сфера при 1,35 Ом·м"],
        [[s["id"], s["heart"]["completed"], s["material_range"]["scan"]["status"], s["material_range"]["heart"]["status"]]
         for s in summary["subjects"]], "Таблица 4. Готовность дополнительных прямых расчётов")


def range_tables(summary):
    levels=[];center=[];joint=[]
    for s in summary["subjects"]:
        scan=s["material_range"]["scan"]
        if not scan["ready"]:continue
        groups=defaultdict(list)
        for r in scan["rows"]:
            if r["rule"]==4096:groups[r["case_id"]].append(r)
        for case,values in sorted(groups.items()):
            for role in ROLES:
                z=[r["transfers"][role]["own_ohm"] for r in values]
                levels.append([s["id"],case,ROLE_NAMES[role],"own_drive · временное разделение",min(z),max(z),len(z)])
        for r in scan["center_identifiability"]["matrices"]:
            for key,label in (("all_four","Четыре rho"),("soft_lung_only","Фон и лёгкое; прочие фиксированы")):
                d=r[key];center.append([s["id"],r["subset"],label,d["shape"][0],d["rank"],d["nullity"],
                    "; ".join(f"{v:.6g}" for v in d["parameter_space_singular_values"]),d["condition_number_full_parameter_space"]])
        for d in scan["joint_grid_diagnostics"]:
            m=d["minimum_on_sampled_grid"]
            joint.append([s["id"],{"inhale":"Вдох","exhale":"Выдох"}[d["state"]],
                "Первичная пара" if d["scope"]=="primary_pair" else "Все монтажи",
                "own_drive · временное разделение",
                m["rho_ohm_m"][0],m["rho_ohm_m"][1],"Норма 2" if d["scope"]=="primary_pair" else "RMS",
                m[d["metric"]],len(d["row_labels"]),"; ".join(f"{v:+.6g}" for v in m["residual_model_minus_observed_ohm"])])
    return table_html(["Участник","Монтаж","Отведение","Режим возбуждения","Z мин., Ом","Z макс., Ом","Точек"],levels,
        "Таблица 5а. Новая сетка сопротивлений: диапазон ответов при временном разделении каналов")+table_html(
        ["Участник","Наблюдаемое состояние","Набор","Режим возбуждения","ρфона точки минимума","ρлёгкого точки минимума","Метрика","Минимум, Ом","Число наблюдений","Знаковые невязки, Ом"],joint,
        "Таблица 5б. Описательные совместные минимумы среди рассчитанных точек при временном разделении")+table_html(
        ["Участник","Набор","Свободные параметры","Строк","Ранг","Нуль-мерность","Безразмерные сингулярные числа","Обусловленность"],center,
        "Таблица 6. Собственный Jacobian новой ветви в центре 2,5 / 7 / 1,35 Ом·м")


def _plot_setup():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False,
                         "figure.dpi":120, "savefig.bbox":"tight", "axes.formatter.useoffset":False})
    return plt


def fig_baseline(summary):
    plt = _plot_setup()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for column, subject in enumerate(summary["subjects"]):
        for row, role in enumerate(ROLES):
            ax=axes[row,column]
            values=[v for v in subject["comparison"] if v["state"]=="inhale" and v["role"]==role]
            for color, variant in zip(("#2166ac", "#b35806", "#5e3c99"), sorted({v["tt_variant"] for v in values})):
                chosen=sorted((v for v in values if v["tt_variant"]==variant),key=lambda v:v["size_mm"])
                x=[v["size_mm"] for v in chosen]
                ax.plot(x,[v["observed_median_ohm"] for v in chosen],"o-",color=color,label=f"Измерение; ТТ {variant}")
                ax.plot(x,[v["predicted_own_ohm"] for v in chosen],"s--",color=color,label=f"FEM own; ТТ {variant}")
            ax.set(title=subject["id"]+" · "+ROLE_NAMES[role],xlabel="Токовая база боковой сборки, мм",ylabel="Z, Ом")
            ax.legend(fontsize=8);ax.grid(alpha=.2)
    return fig


def fig_breath(summary):
    plt=_plot_setup();fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for column,subject in enumerate(summary["subjects"]):
        for row,role in enumerate(ROLES):
            ax=axes[row,column];values=sorted((v for v in subject["breath"] if v["role"]==role),key=lambda v:case_parts(v["case_id"]))
            x=np.arange(len(values))
            ax.bar(x-.18,[v["observed_delta_ohm"] for v in values],.36,label="Парное измерение",color="#2166ac")
            ax.bar(x+.18,[v["predicted_delta_ohm"] for v in values],.36,label="FEM own; та же КТ",color="#b35806")
            ax.set_xticks(x,[f"{case_parts(v['case_id'])[0]} / {case_parts(v['case_id'])[1]}" for v in values],rotation=40,ha="right")
            ax.axhline(0,color="#444",lw=.7);ax.set(title=subject["id"]+" · "+ROLE_NAMES[role],ylabel="Zвыдох − Zвдох, Ом",xlabel="База, мм / вариант ТТ")
            ax.legend(fontsize=8);ax.grid(axis="y",alpha=.2)
    return fig


def fig_svd(summary):
    plt=_plot_setup();fig,axes=plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True)
    names={"lateral":"Боковые", "ttrkg":"ТТРКГ", "joint":"Совместно"}
    for ax,subject in zip(axes,summary["subjects"]):
        for color,subset in zip(("#2166ac","#b35806","#1b7837"),names):
            value=next(v for v in subject["sensitivity"] if v["state"]=="inhale" and v["subset"]==subset)
            singular=value["matrix"]["parameter_space_singular_values"]
            ax.semilogy(range(1,5),singular,"o-",color=color,label=names[subset])
            norm=value["quadrature"]["spectral_norm"]
            if norm>0:ax.axhline(norm,color=color,ls=":",lw=1,label=names[subset]+": разность q")
        ax.set(title=subject["id"],xlabel="Номер сингулярного числа",ylabel="Безразмерное сингулярное число",xticks=range(1,5))
        ax.legend(fontsize=7);ax.grid(alpha=.2)
    return fig


def fig_basis(summary):
    plt=_plot_setup();fig,axes=plt.subplots(2,3,figsize=(15,7),constrained_layout=True)
    for column,subject in enumerate(summary["subjects"]):
        for row,role in enumerate(ROLES):
            ax=axes[row,column];ax.bar(MATERIALS,subject["basis_delta_Z_ohm"][role],color=("#2166ac","#67a9cf","#b2182b","#ef8a62"))
            ax.axhline(0,color="#333",lw=.7);ax.set(title=subject["id"]+" · "+ROLE_NAMES[role],ylabel="Линейное δZ, Ом")
            ax.tick_params(axis="x",rotation=20);ax.grid(axis="y",alpha=.2)
    return fig


def fig_sphere(summary):
    plt=_plot_setup();fig,axes=plt.subplots(2,3,figsize=(15,7),constrained_layout=True)
    for column,subject in enumerate(summary["subjects"]):
        for row,role in enumerate(ROLES):
            ax=axes[row,column];available=False
            for rule,color in ((1024,"#999999"),(4096,"#2166ac")):
                values=sorted((v for v in subject["heart"]["rows"] if v["rule"]==rule and v["geometry"].startswith("sphere_")),key=lambda v:v["volume_integrated_ml"])
                if values:
                    available=True;ax.plot([v["volume_integrated_ml"] for v in values],[v["transfers"][role]["own_ohm"] for v in values],"o-",color=color,label=f"Сфера, q{rule}")
                whole=[v for v in subject["heart"]["rows"] if v["rule"]==rule and v["geometry"]=="wholeheart_ct"]
                if whole:
                    v=whole[0];ax.scatter([v["volume_integrated_ml"]],[v["transfers"][role]["own_ohm"]],marker="*",s=90,color=color,label=f"Всё сердце КТ, q{rule}")
            if not available:
                ax.text(.5,.5,"Результаты сфер ещё не получены",ha="center",va="center",transform=ax.transAxes);ax.set_axis_off()
            ax.set(title=subject["id"]+" · "+ROLE_NAMES[role],xlabel="Объём сердечного материала, мл",ylabel="Z own, Ом")
            if available:ax.legend(fontsize=7)
            ax.grid(alpha=.2)
    return fig


def show_figure(fig):
    from IPython.display import display
    import matplotlib.pyplot as plt
    display(fig);plt.close(fig)


def fig_material_grid(summary):
    plt=_plot_setup();fig,axes=plt.subplots(2,3,figsize=(15,7.5),constrained_layout=True)
    for column,s in enumerate(summary["subjects"]):
        scan=s["material_range"]["scan"]
        for row,role in enumerate(ROLES):
            ax=axes[row,column]
            if not scan["ready"]:
                ax.text(.5,.5,"Прямые расчёты диапазона ещё не завершены",ha="center",va="center",transform=ax.transAxes);ax.set_axis_off()
            else:
                values=[v for v in scan["rows"] if v["case_id"]=="L140_TT2x2" and v["rule"]==4096]
                for soft,color in zip((2.,2.5,3.),("#2166ac","#b35806","#1b7837")):
                    selected=sorted((v for v in values if v["rho_ohm_m"][0]==soft),key=lambda v:v["rho_ohm_m"][1])
                    if len(selected)!=5:raise ValueError("Incomplete 15-point primary grid")
                    ax.plot([v["rho_ohm_m"][1] for v in selected],[v["transfers"][role]["own_ohm"] for v in selected],"o-",color=color,label=f"ρфона={soft:g} Ом·м")
                for state,style,label in (("inhale","--","Измерение: вдох"),("exhale",":","Измерение: выдох")):
                    values=[v for v in s["comparison"] if v["case_id"]=="L140_TT2x2" and v["role"]==role and v["state"]==state]
                    if values:ax.axhline(values[0]["observed_median_ohm"],color="#222",ls=style,label=label)
                ax.set_xticks([2,4,7,10,12]);ax.legend(fontsize=7);ax.grid(alpha=.2)
            ax.set(title=s["id"]+" · "+ROLE_NAMES[role],xlabel="ρлёгкого, Ом·м",ylabel="Z own_drive, Ом")
    return fig


def fig_blood_sphere(summary):
    plt=_plot_setup();fig,axes=plt.subplots(2,3,figsize=(15,7),constrained_layout=True)
    for column,s in enumerate(summary["subjects"]):
        data=s["material_range"]["heart"]
        for row,role in enumerate(ROLES):
            ax=axes[row,column]
            if not data["ready"]:
                ax.text(.5,.5,"Прямые объёмные расчёты при 1,35 Ом·м\nещё не завершены",ha="center",va="center",transform=ax.transAxes);ax.set_axis_off()
            else:
                for q,color in ((1024,"#999999"),(4096,"#b2182b")):
                    values=sorted((v for v in data["rows"] if v["rule"]==q and v["case_id"]=="L140_TT2x2" and v["geometry"].startswith("sphere_")),key=lambda v:v["volume_integrated_ml"])
                    if len(values)!=3:raise ValueError("Incomplete blood-sphere volume states")
                    ax.plot([v["volume_integrated_ml"] for v in values],[v["transfers"][role]["own_ohm"] for v in values],"o-",color=color,label=f"Сфера, q{q}")
                    whole=next(v for v in data["rows"] if v["rule"]==q and v["case_id"]=="L140_TT2x2" and v["geometry"]=="wholeheart_ct")
                    ax.scatter([whole["volume_integrated_ml"]],[whole["transfers"][role]["own_ohm"]],marker="*",s=90,color=color,label=f"Маска целого сердца, q{q}")
                ax.legend(fontsize=7);ax.grid(alpha=.2)
            ax.set(title=s["id"]+" · "+ROLE_NAMES[role],xlabel="Объём сердечного материала, мл",ylabel="Z own_drive, Ом")
    return fig


def notebook(summary, draft):
    import nbformat as nb
    cells=[]
    def md(text):cells.append(nb.v4.new_markdown_cell(text.strip()))
    def code(text):cells.append(nb.v4.new_code_cell(text.strip()))
    def figure(number, purpose, function, caption, discussion):
        md(purpose+f" (рисунок {number}).")
        code(f"F.show_figure(F.{function}(S))")
        md(f"**Рисунок {number}. {caption}**\n\n"+discussion)
    status="ПРЕДВАРИТЕЛЬНЫЙ ОТЧЁТ: дополнительные прямые расчёты не завершены" if draft else "Сводка завершённого ограниченного вычислительного цикла"
    title="Индивидуальная КТ/FEM-проверка трёх участников Scorpion"+(" — ПРЕДВАРИТЕЛЬНЫЙ ОТЧЁТ" if draft else "")
    md(f"""# {title}

**{status}.** Исследовательский этап разработки методики, 26 октября 2026 года.

Цель — проверить, что объясняют индивидуальные прямые модели базового импеданса,
какие тканевые параметры различимы локально и как условные тканевые и объёмные
сценарии изменяют электрический ответ. Представлены три самостоятельные модели;
данные между участниками не переносятся. Георгий, Ник и Ярослав — принятые
проектом псевдонимы; ФИО и пространственные исходники не раскрываются.

**Авторское решение о дальнейшей схеме.** Конечным входом методики принят
сигнал ТТРКГ. Боковые сборки исключены из дальнейшей схемы определения
тканевых ρ; их результаты сохраняются здесь как отрицательная диагностическая
проверка и не предлагаются в качестве рабочей инверсии сопротивлений.
Текущий вычислительный цикл сохраняет полный набор сравнений для разбора
полученного отрицательного результата.

Проверки численного решения и интегрирования пройдены в указанном ниже объёме.
Совпадение с экспериментальными уровнями и практическая идентифицируемость
параметров этим не установлены. Шесть пробных переносов электродов отклонены
до проверки новых площадок. Готовность расчётов замещения сердца: {summary['heart_results_completed']} из 24;
новых материальных ветвей: {summary['material_range_completed']} из 6.
Все таблицы раскрываются по действию читателя.

1. [Входы, прямая модель и контроль](#inputs)
2. [Базовые уровни и временное разделение](#baseline)
3. [Парное дыхательное различие](#breath)
4. [Различимость сопротивлений](#svd)
5. [Четыре тканевых возмущения](#basis)
6. [Объём эквивалентной сферы](#sphere)
7. [Авторский диапазон сопротивлений](#range)
8. [Вывод и происхождение](#conclusion)""")
    code("""from pathlib import Path
import sys,json
here=Path.cwd()
folder=here if (here/'build_scorpion_fem_smart_report.py').is_file() else next(p/'Colab Notebooks/scorpion_v2' for p in [here,*here.parents] if (p/'Colab Notebooks/scorpion_v2/build_scorpion_fem_smart_report.py').is_file())
sys.path.insert(0,str(folder))
import build_scorpion_fem_smart_report as F
from IPython.display import HTML,display
get_ipython().run_line_magic('matplotlib','inline')
S=json.loads((folder/'_assets/fem_smart_20261026.summary.json').read_text(encoding='utf-8'))
F.assert_safe(S)
""")
    md("""<a id="inputs"></a>
## 1. Индивидуальные входы, прямая модель и границы проверок

**Подтверждённые входы.** Использованы собственные КТ и подготовленные сетки трёх
участников. КТ получены на глубоком вдохе с задержкой дыхания. Фон, лёгкое,
кость, всё сердце и жир образуют пять материалов. Индивидуальная геометрия
и собственные маски TotalSegmentator v2.18.0 служат входами прямого расчёта.
Паспорта связывают исходную КТ и каждую маску SHA; для сетки используются
отдельные маски вычислительной области. Нативные анатомические STL и сетки
вычислительной области сохраняют разные назначения. Анатомический просмотр
автоматических масок остаётся отдельной проверкой. Качество электрического прогноза
не следует из одного успешного геометрического контроля.

**Модельные допущения.** Расчёт выполнен при 100 кГц в MATLAB/EIDORS с конечными
контактами в полной электродной модели (CEM), методом конечных элементов (FEM):
номинальные диски диаметром 5 мм, фактические дискретные площадки
и группировка физически присутствующих электродов. Контактный импеданс задан
сценарием. Кость фиксирована. Сердце рассматривается как единая эффективная
область, без разделения крови и миокарда. Система решается в СИ; передаточный
импеданс — отношение напряжения к току, в омах.

В реализации используются вещественные удельные сопротивления и проводимости
σ=1/ρ. Комплексные проводимости тканей не используются.

**Смешанные ткани внутри элемента.** В каждом тетраэдре fₖ — объёмная доля
ткани k, fₖ≥0 и Σₖfₖ=1. Для линейных тетраэдральных элементов используется
эффективная проводимость σeff=Σₖfₖ/ρₖ. Усредняется проводимость, а не
арифметическое значение удельного сопротивления. Тканевые матрицы Kₖ
учитывают доли ткани и геометрию элементов; объёмная матрица проводимости
собирается как K=ΣₖKₖ/ρₖ. Это соответствует сохранённой сборке
Fᵀdiag(fₖ)F с повторением весов для трёх компонент поля. Контактные члены
полной электродной модели добавляются отдельно. При изменении ρ в авторской
сетке сценариев индивидуальные тканевые доли и геометрия сохраняются.

Для всех контактов
трёх участников задано zcontact=0,01 Ом·м²; это рабочий CEM-сценарий,
контактный импеданс независимо не измерен. Числа и основание назначения
связаны с сохранённым контактным паспортом.

Цилиндрические руки имеют принятый обхват 370 мм и радиус 370/(2π)≈58,89 мм.
Для Георгия это указание автора от 08.10.2026, для Ника — от 22.10.2026;
для Ярослава это приближённое назначение. Эти разные статусы не объединяются
в утверждение о трёх одинаково измеренных обхватах.

Для оценки вычислительной согласованности сопоставляются невязка линейной
системы и взаимность с допусками 10⁻⁷, производные с центральной разностью
по lnρ с шагом 0,001 и относительным допуском 10⁻⁴. Правила q1024/q4096
используют вложенные наборы из 1024 и 4096 точек Sobol для оценки fₖ внутри
прежних линейных тетраэдров; первые 1024 точки входят в набор 4096.
Число тетраэдров и узлов при этом не растёт. Такой контроль интегрирования
не устраняет ошибку пространственной дискретизации и не доказывает сеточную
сходимость по одной неизменной сетке. Различие правил
проверяется для базовых уровней по критерию 0,02 Ом + 0,001·|Zq4096|.
Таблица 1 показывает фактический охват сохранённых проверок.""")
    profiles=defaultdict(list)
    for s in summary["subjects"]:
        profiles[(tuple(s["baseline_materials_inhale_ohm_m"]),tuple(s["baseline_materials_exhale_ohm_m"]))].append(s["id"])
    materials=[]
    for (inhale,exhale),names in profiles.items():
        f=lambda value:f"{value:.9f}".replace(".",",")
        materials.append(" / ".join(names)+f": фон {f(inhale[0])}; лёгкое на вдохе {f(inhale[1])}, в условном сценарии выдоха {f(exhale[1])}; кость {f(inhale[2])}; целое сердце {f(inhale[3])}; жир {f(inhale[4])} Ом·м")
    md("**Параметры контрольного расчёта.** "+"; ".join(materials)+". Источник — частотные значения Gabriel при 100 кГц в [числовом реестре тканей](../../MATLAB_TRKG4_real_subjects/configs/tissue_literature_v2.json), его SHA связан с каждым расчётом. В новой ветви с однородной кровью жир и кость сохраняют именно эти значения. Индивидуальная подгонка не выполнялась. Число знаков отражает сохранённый числовой профиль и не задаёт метрологическую точность. Одинаковые опорные свойства не означают одинаковых индивидуальных геометрий или чувствительностей.")
    code("display(HTML(F.qc_table(S)))")
    md("""FD-проверка выполнена для монтажа 140 мм с ТТРКГ 2×2 на вдохе; остальные
монтажи и условный выдох не получили отдельной FD-проверки. Сеточная сходимость,
ошибка сегментации и экспериментальная точность остаются самостоятельными
вопросами. При переносе на пять миллиметров сначала проверялась проекция новых
центров на прежнюю поверхность (таблица 2).""")
    code("display(HTML(F.placement_table(S)))")
    md("""**Отрицательный результат проверки размещения.** Все шесть сценариев
остановились на прежнем допуске проекции 1 мм. Задавалось смещение исходного
центра на [0,±5,0] мм в LPS перед проекцией; геодезическая длина не фиксировалась.
Допуски не расширялись. Этот отказ относится к геометрической операции по
заданной оси: новые footprint и электрическая устойчивость к размещению
остались непроверенными. Необходимость сгущения сетки данным отказом не установлена.
Сценарные ±5 мм не являются доверительной погрешностью наклейки.""")
    md("""<a id="baseline"></a>
## 2. Базовые уровни при временном разделении каналов

**Экспериментальные наблюдения.** Сопоставляются сохранённые уровни на
ретроспективно выделенных интервалах дыхательного протокола. Разметка остаётся
кандидатной; повторения одного монтажа сведены медианой, число повторений
сохраняется в агрегате. Разные размеры регистрировались последовательно.
Время, контакт и состояние участника могут различаться между записями.

У Георгия вариант ТТРКГ 1×1 в записях 007–010 подтверждён автором.
При этом соответствие физических ролей отведений в записях 008–010
остаётся допущением сохранения подключения; подтверждение варианта ТТРКГ
не превращает это допущение в подтверждённый факт о ролях отведений.

**Подтверждённый режим прибора.** По прямому уточнению автора от 26.10.2026
каналы Scorpion разделяются по времени. Относительная фаза одновременно
действующих источников к этому режиму не применима. Основной прогноз
использует собственный источник отведения — own_drive — и соответствует
подтверждённому временному разделению. Сохранённые решения с отдельными
правыми частями уже дают требуемые портовые ответы; пересчёт прямой задачи
для этого уточнения не требуется. Геометрия, свойства тканей и контактные
условия остаются заданными модельными предпосылками.
Проверим отдельно размерную зависимость и варианты ТТРКГ.""")
    residuals="; ".join(f"{s['id']}: "+", ".join(f"{ROLE_NAMES[role]} {s['primary_comparison'][role]['own_ohm']-s['primary_comparison'][role]['observed_median_ohm']:+.3f} Ом" for role in ROLES if s['primary_comparison'][role]['observed_median_ohm'] is not None) for s in summary["subjects"])
    figure(1,"Сравнение на вдохе сохраняет различия размеров и группировки электродов", "fig_baseline",
        "Измеренные медианные уровни и FEM-прогноз own_drive на вдохе при подтверждённом временном разделении каналов. Столбцы — участники; верхний ряд — ТТРКГ, нижний — боковая сборка. Цвет различает вариант ТТРКГ, сплошные линии с кружками — наблюдение, пунктир с квадратами — модель. База указана по боковой сборке.",
        "На рисунке 1 представлены прямые прогнозы для заданных материалов и CEM без подгонки сопротивлений или усиления. Невязки первичного монтажа 140 мм/2×2: "+residuals+". Эти различия относятся к выбранным материалам, геометрии и контактам; совпадение формы размерной зависимости не заменяет проверки абсолютных уровней.")
    md("""Для подтверждённого режима используются ZТТРКГ=T(2,1) и
Zбок=T(4,3) (1), где T — сохранённая матрица портовых передач.
Перекрёстные передачи T(2,3) и T(4,1) оставлены отдельно для контроля
ответов и взаимосвязей портов. При временном разделении они не прибавляются
к собственному ответу и не входят в основной прогноз измеряемого Z.
Таблица 3 относится к первичному монтажу 140 мм и ТТРКГ 2×2 на вдохе.""")
    code("display(HTML(F.drive_table(S)))")
    md("Ненулевые перекрёстные передачи характеризуют ответ измерительной пары на ток соседней пары в модели. Они сохраняют самостоятельное диагностическое значение; основной ответ при подтверждённом временном разделении определяется своим источником.")
    md("""<a id="breath"></a>
## 3. Парное различие между вдохом и выдохом

Дыхательный контраст определяется как ΔZ=Zвыдох−Zвдох (2). Каждая разность
формируется внутри одной записи и отведения до объединения повторений.
Это сохраняет парность сравнения и отделяет контраст от абсолютного уровня.
В этом FEM-сценарии меняется только сопротивление лёгкого:
9,315252925→3,681694081 Ом·м. Сопротивления фона, кости, целого сердца и жира
фиксированы, исходная КТ глубокого вдоха остаётся неизменной. Следовательно,
модельный выдох является условным материальным сценарием.
Изолированный конечный ответ lung-only совпадает с тем же расчётом выдоха;
background-only даёт нулевой контраст по конструкции сценария. Это не
эмпирический результат «фон не меняется при дыхании». Поэтому идентичные
задачи прямого расчёта повторно не решались: проверены сохранённые пары
вдох/выдох и три источника расчёта вкладов каждого участника. Конечное
изменение ответа и его локальная линейная аппроксимация различаются;
равенство lung-only и полного конечного контраста не утверждает их равенства
линейному прогнозу Jacobian.""")
    breath_text=[]
    for s in summary["subjects"]:
        v=next(v for v in s["breath"] if v["case_id"]=="L140_TT2x2" and v["role"]=="ttrkg")
        mismatch=sum(v["observed_delta_ohm"]*v["predicted_delta_ohm"]<0 for v in s["breath"] if v["role"]=="lateral")
        breath_text.append(f"{s['id']}: первичный ТТРКГ — наблюдение {v['observed_delta_ohm']:+.4g} Ом, модель {v['predicted_delta_ohm']:+.4g} Ом; число боковых монтажей с различным знаком наблюдения и модели — {mismatch}")
    figure(2,"Проверим величину и знак парного контраста для каждого монтажа", "fig_breath",
        "Парные дыхательные разности: наблюдение и модельный lung-only контраст own_drive при временном разделении. Синий столбец — медиана парных разностей, оранжевый — модель. Категории обозначают базу боковой сборки в миллиметрах и вариант ТТРКГ.",
        "По рисунку 2: "+"; ".join(breath_text)+". Геометрическое изменение грудной клетки здесь не рассчитывалось. Ретроспективная разметка, последовательность монтажей и отсутствие оценки межсессионной неопределённости ограничивают количественную интерпретацию расхождения.")
    md("""<a id="svd"></a>
## 4. Локальная различимость четырёх сопротивлений

Сохранённый adjoint Jacobian задаёт Jlog=dZ/dlnρ. Его столбцы соответствуют
фону, лёгкому, всему сердцу и жиру; кость фиксирована. Ненормированная
производная равна dZ/dρj=Jlog,j/ρj (3), с единицами 1/м.
Для сравнения масштабов используется явно безразмерная матрица
Eij=Jlog,ij/Zi (4). Знаки Z и производных сохранены; эта нормировка не задаёт
шумовую дисперсию. SVD разлагает E на ортогональные направления, а сингулярные
числа характеризуют величину ответа на возмущения lnρ в этих направлениях.

Боковые строки, ТТРКГ и их совместный набор рассматриваются раздельно.
Совместный набор объединяет все сохранённые монтажи одного участника и состояния.
Одна пара ТТРКГ/боковой сборки даёт два наблюдения для четырёх rho.
Формальный ранг при машинном пороге не доказывает практической определимости.
Эти сравнения служат диагностической проверкой; совместный набор с боковыми
сборками не предлагается как дальнейшая рабочая схема определения ρ.
Один сигнал ТТРКГ сам по себе также не идентифицирует четыре произвольных
тканевых сопротивления без дополнительных ограничений и референсов.""")
    conditions="; ".join(f"{s['id']}: "+", ".join(f"{name} {next(v for v in s['sensitivity'] if v['state']=='inhale' and v['subset']==key)['matrix']['condition_number_full_parameter_space']:.3g}" for key,name in (("lateral","боковые"),("ttrkg","ТТРКГ"),("joint","совместно"))) for s in summary["subjects"])
    figure(3,"Сопоставим сингулярный спектр с наблюдаемым изменением матрицы при смене правила интегрирования", "fig_svd",
        "Сингулярные числа безразмерной E на вдохе, логарифмическая шкала. Сплошные линии — боковые, ТТРКГ и совместный набор; цветные пунктирные уровни — спектральная норма Eq4096−Eq1024 для соответствующего набора.",
        "Обусловленность на рисунке 3: "+conditions+". Слабейшее совместное направление у всех трёх участников преимущественно связано с эффективным сопротивлением всего сердца. Разность интеграционных правил является наблюдаемым численным контролем, её нельзя считать строгой границей ошибки. Шум и неопределённость положения не оценены; раздельная практическая оценка четырёх rho не установлена.")
    md("""<a id="basis"></a>
## 5. Четыре знаковых тканевых возмущения

Для первичного монтажа задаются четыре независимых базисных сценария:
δlnρj=0,01 для одного материала, остальные возмущения равны нулю.
Линейный прогноз δZ≈Jlog·δlnρ (5) вычисляется по сохранённым производным.
Это заданные малые возмущения свойств тканей; их амплитуда не измерена
как пульсовое изменение участника. Кость и геометрия остаются постоянными.""")
    heart_basis="; ".join(f"{s['id']} {s['basis_delta_Z_ohm']['ttrkg'][2]:+.6g} Ом" for s in summary["subjects"])
    figure(4,"Сравним знаки и масштабы ответа двух отведений на одинаковое логарифмическое возмущение", "fig_basis",
        "Линейное δZ для четырёх базисных сценариев δlnρ=0,01 на вдохе. Монтаж 140 мм/ТТРКГ 2×2, q4096. Каждый столбец соответствует самостоятельному возмущению; одновременно они не суммируются.",
        "На рисунке 4 ответ ТТРКГ на изолированное δlnρ целого сердца=0,01 составляет: "+heart_basis+". Знак передаточной чувствительности определяется полями токовой и измерительной пар. Полученные столбцы не являются измеренными источниками пульса; ЭКГ-синхронность не разделяет их автоматически. Точность линейного приближения для выбранной конечной амплитуды здесь отдельно не проверена.")
    md("""<a id="sphere"></a>
## 6. Эквивалентная сфера и объёмная чувствительность

Сфера строится по объёму маски всего сердца конкретного участника.
Рассматриваются целевые объёмы 0,95V₀, V₀ и 1,05V₀ при фиксированном центре.
Исходная область сердца заменяется фоном; внутри сферы материал сердца имеет
приоритет, после выхода сферы восстанавливается фоновый материал.
Это явное назначение материалов на неподвижной КТ, без механического переноса органов.
Контроль с исходной маской целого сердца должен воспроизвести базовый расчёт до
интерпретации изменения формы. Три объёмных состояния задают секущую;
сходимость по шагу объёма и обратная задача этим не проверены.""")
    range_caption=" Контрольный профиль: ρфона=2,763586, ρлёгкого=9,315253, ρцелого сердца=4,648809 Ом·м; жир и кость фиксированы."
    figure(5,"Рассмотрим только фактически сохранённые прямые ответы объёмных вариантов", "fig_sphere",
        "Импеданс own_drive в зависимости от интегрированного объёма сердечного материала. Кружки и линии — три сферических состояния, звёзды — исходная маска целого сердца; серый — q1024, синий — q4096."+range_caption,
        "Рисунок 5 сопоставляет геометрию при одном фиксированном векторе сопротивлений: замена исходной маски сферой и изменение объёма сферы рассматриваются внутри указанного контрольного профиля. Размах модельных объёмов не является ударным объёмом. Объёмная чувствительность не переводится в УО/ФВ без отдельной проверенной связи с физиологическими камерными объёмами.")
    shape=[]
    for s in summary["subjects"]:
        if s["heart"]["ready"]:
            v=s["heart"]["own_controls"]["ttrkg"]
            shape.append(f"{s['id']}: изменение ТТРКГ при равнообъёмной замене формой сферы {v['equal_volume_shape_effect_ohm']:+.6g} Ом, отклик между 0,95V₀ и 1,05V₀ {v['minus_to_plus_volume_response_ohm']:+.6g} Ом, разность q для собственного ТТРКГ сферы V₀ {v['sphere_v100_own_q_difference_ohm']:+.6g} Ом")
    if shape:md("**Фактический контроль формы и объёма.** "+"; ".join(shape)+". Если эффект формы превосходит проверенный объёмный контраст, объём сферы особенно нельзя отождествлять с анатомическим объёмом. Эти два эффекта сопоставлены в омах; пересчёт ошибки формы в миллилитры по секущей не выполняется. Разность интегрирования собственного ТТРКГ отделена от максимума по всем элементам портовой матрицы.")
    md("""<a id="range"></a>
## 7. Авторский диапазон сопротивлений и отдельный сценарий всего сердца

**Авторское условие новой ветви.** Мягкотканному фону назначается диапазон
2–3 Ом·м, лёгкому — 2–12 Ом·м, всему сердцу — однородное 1,35 Ом·м.
Значение крови используется здесь как заданное свойство всей сердечной области;
смесь крови и миокарда им независимо не определена. Жир и кость фиксированы.
Диапазон мышц присвоен эффективному фону неразмеченных мягких тканей;
самостоятельная анатомическая маска мышц в этой постановке отсутствует.
Сканирование использует фон 2; 2,5; 3 Ом·м и лёгкое 2; 4; 7; 10; 12 Ом·м:
15 точек q4096 и пять контрольных точек q1024. Это сценарная сетка, а не
доверительный диапазон индивидуальных параметров. Отдельные восемь объёмных
результатов новой ветви вычисляются при фоне 2,5, лёгком 7 и сердце 1,35 Ом·м.

Новые прямые результаты принимаются только по собственным завершённым
планам и SHA. Прогноз own_drive соответствует подтверждённому временному
разделению каналов и сопоставляется без подгонки усиления.
До завершения прямых расчётов численные поля
остаются пустыми. Диапазон Z по такой сетке нельзя восстановить из контрольного
расчёта при сопротивлении целого сердца 4,6488 Ом·м
или линейной экстраполяцией. Таблица 4 показывает фактическую готовность.""")
    code("display(HTML(F.status_table(S)))")
    md("""Скриншот расчёта коллеги служит качественным ориентиром изменения ответа
при повышении сопротивления фона и лёгкого. Исходный файл, геометрия и определение
измеряемого Z недоступны. По уточнению автора возможна двуслойная сфера
«миокард + кровь»; текущая первая ветвь по его поручению использует однородную
кровь 1,35 Ом·м. Скриншот не оцифровывается и не служит независимой валидацией.

Для центра новой сетки ρфон=2,5, ρлёгкое=7, ρсердце=1,35 Ом·м заново
используется собственный Jlog этой точки. Таблица 6 раздельно показывает
четыре столбца и редуцированные [фон, лёгкое] при фиксированных сердце, жире
и кости. Одна первичная пара даёт две строки и не определяет четыре rho.
Полный ранг редуцированной матрицы означает лишь локальную условную
определимость при перечисленных фиксированных параметрах; реальные
индивидуальные сопротивления этим не восстановлены. Таблица 5 сохраняет
границы сценарной сетки Z без интерпретации их как доверительных интервалов.
Для отсутствующих результатов числовые строки остаются пустыми.""")
    code("display(HTML(F.range_tables(S)))")
    center_text=[]
    for s in summary["subjects"]:
        data=s["material_range"]["scan"]
        if not data["ready"]:continue
        primary=next(r for r in data["center_identifiability"]["matrices"] if r["subset"]=="primary_pair")
        q_count=sum(bool(r["evaluated"]) for r in data["quadrature"])
        center_text.append(f"{s['id']}: первичная пара имеет ранг {primary['all_four']['rank']} из четырёх и размерность ядра {primary['all_four']['nullity']}; при свободных фоне и лёгком κ={primary['soft_lung_only']['condition_number_full_parameter_space']:.6g}. Максимальная относительная ошибка FD центра {max(data['derivative_checks']['relative_errors']):.6g}; q-контроль пройден для {q_count} из {len(data['quadrature'])} комбинаций точка/монтаж")
    if center_text:
        md("**Собственные результаты новой материальной точки.** "+"; ".join(center_text)+". FD относится только к первичному монтажу центра. Второе правило интегрирования рассчитано в углах сетки и в центре; остальные точки не получили отдельного q-контроля. Условная определимость двух параметров требует фиксированных свойств сердца, жира и кости; эти свойства не установлены индивидуальными измерениями.")
    joint_text=[]
    for s in summary["subjects"]:
        diagnostics=s["material_range"]["scan"]["joint_grid_diagnostics"]
        if not diagnostics:continue
        statements=[]
        for state,label in (("inhale","вдох"),("exhale","выдох")):
            primary=next(d for d in diagnostics if d["state"]==state and d["scope"]=="primary_pair")
            all_cases=next(d for d in diagnostics if d["state"]==state and d["scope"]=="all_cases")
            p=primary["minimum_on_sampled_grid"];a=all_cases["minimum_on_sampled_grid"]
            observed="; ".join(f"{v:.6f}" for v in primary["observed_ohm"])
            residual="; ".join(f"{v:+.6g}" for v in p["residual_model_minus_observed_ohm"])
            statements.append(f"{label}: измеренные [ТТРКГ; боковая] = [{observed}] Ом. Среди {primary['sampled_points']} точек минимальная норма двух невязок {p['euclidean_norm_ohm']:.6g} Ом при [ρфона; ρлёгкого]=[{p['rho_ohm_m'][0]:g}; {p['rho_ohm_m'][1]:g}] Ом·м, знаковые невязки [{residual}] Ом. По всем {len(all_cases['row_labels'])} наблюдениям минимум RMS {a['rms_ohm']:.6g} Ом при [{a['rho_ohm_m'][0]:g}; {a['rho_ohm_m'][1]:g}] Ом·м")
        joint_text.append(s["id"]+": "+"; ".join(statements)+".")
    if joint_text:
        md("**Совместное сопоставление на сетке, own_drive.** "+"\n\n".join(joint_text)+"\n\nНевязка определена как модель минус измерение. Норма двух невязок равна √(rТТ²+rбок²), RMS — √(Σr²/n); все компоненты имеют одинаковый арифметический вес. Это описательные минимумы только среди рассчитанных точек, без модели шума, доверительных интервалов или оценки индивидуальных параметров. Они не исключают совпадение в непрерывном прямоугольнике сопротивлений. Представленные совместные невязки не подтверждают согласия обоих отведений с одним набором свойств при выбранных тканевых свойствах, геометрии и контактах. Режим возбуждения соответствует подтверждённому временному разделению.")
        bounds=[]
        for s in summary["subjects"]:
            diagnostics=[d for d in s["material_range"]["scan"]["joint_grid_diagnostics"] if d["scope"]=="primary_pair"]
            if not diagnostics:continue
            r=diagnostics[0]["primary_grid_ranges"]["ttrkg"]
            below_own=sum(d["primary_grid_ranges"]["ttrkg"]["observed_below_all_own"] for d in diagnostics)
            bounds.append(f"{s['id']}: ТТРКГ own_drive {r['own_min_ohm']:.6f}–{r['own_max_ohm']:.6f} Ом. Ниже всех рассчитанных точек находятся {below_own} из 2 наблюдаемых уровней")
        md("**Проверка первичного ТТРКГ на тех же точках.** "+"; ".join(bounds)+". Сравниваются отдельно измеренные вдох и выдох без назначения дыхательных состояний сценарным точкам. Если уровень ниже минимального прогноза, ни одна из рассчитанных точек его не покрывает. Это не исключает иные непрерывные параметры и не отменяет неопределённость геометрии, контактных условий и свойств тканей.")
    grid_text=[]
    for s in summary["subjects"]:
        data=s["material_range"]["scan"]
        if data["ready"]:
            values=[r for r in data["rows"] if r["rule"]==4096 and r["case_id"]=="L140_TT2x2"]
            grid_text.append(s["id"]+": "+", ".join(f"{ROLE_NAMES[role]} {min(r['transfers'][role]['own_ohm'] for r in values):.6g}–{max(r['transfers'][role]['own_ohm'] for r in values):.6g} Ом" for role in ROLES))
    grid_discussion=("На рассчитанных точках рисунка 6 диапазон own_drive составляет: "+"; ".join(grid_text)+". ") if grid_text else "Прямые расчёты диапазона ещё не завершены; рисунок 6 обозначает ожидаемые результаты. "
    figure(6,"Сравним фактически рассчитанную 15-точечную сетку с двумя наблюдаемыми дыхательными уровнями", "fig_material_grid",
        "Новая сетка сопротивлений для 140 мм/ТТРКГ 2×2, q4096, ρцелого сердца=1,35 Ом·м. По оси X — ρлёгкого, три цветные линии соединяют пять рассчитанных точек при ρфона 2; 2,5; 3 Ом·м. Горизонтальные линии — измеренные медианы вдоха и выдоха. Жир и кость фиксированы; own_drive соответствует подтверждённому временному разделению каналов.",
        grid_discussion+"Каждая отмеченная точка является сохранённым прямым расчётом. Соединяющие отрезки организуют точки; дополнительные значения между ними не рассчитывались. Сценарным точкам свойств тканей не назначается состояние вдоха или выдоха: оба измеренных уровня показаны как отдельные ориентиры. Совпадение уровня с точкой не устанавливает единственность сопротивлений или индивидуальную физиологическую оценку.")
    blood_text=[]
    for s in summary["subjects"]:
        data=s["material_range"]["heart"]
        if not data["ready"]:continue
        c=data["own_controls"]["ttrkg"]
        link=next(r for r in data["wholeheart_scan_center_links"] if r["rule"]==4096)
        reproduction="точное совпадение" if link["transfer_matrix_exact"] else f"максимальное различие {link['max_abs_transfer_difference_ohm']:.6g} Ом"
        fd="не выполнялась" if not data["derivative_checks"]["performed"] else "выполнена"
        blood_text.append(f"{s['id']}: ТТРКГ, сфера V₀ минус исходная маска {c['equal_volume_shape_effect_ohm']:+.6g} Ом; ответ между 0,95V₀ и 1,05V₀ {c['minus_to_plus_volume_response_ohm']:+.6g} Ом. Разности q4096−q1024 для этих контрастов формы и объёма соответственно {c['shape_contrast_q_difference_ohm']:+.6g} и {c['volume_contrast_q_difference_ohm']:+.6g} Ом; для абсолютного Z сферы V₀ отдельно {c['sphere_v100_own_q_difference_ohm']:+.6g} Ом. Контроль исходной маски с центром сетки сопротивлений при q4096: {reproduction}; FD-проверка этой объёмной ветви {fd}")
    blood_discussion=("Сохранённые результаты рисунка 7: "+"; ".join(blood_text)+". ") if blood_text else "Прямые объёмные расчёты при 1,35 Ом·м ещё не завершены; численных объёмных выводов по этой ветви пока нет. "
    figure(7,"Рассмотрим геометрическое сравнение внутри нового фиксированного материального профиля", "fig_blood_sphere",
        "Объёмные ответы own_drive при ρфона=2,5, ρлёгкого=7 и ρцелого сердца=1,35 Ом·м. Кружки — сферы с тремя объёмами, звёзды — исходная маска целого сердца при тех же свойствах. Серый — q1024, красный — q4096. На оси X — интегрированный объём, мл.",
        blood_discussion+"Для этой ветви используются собственные материальные блоки каждого объёмного варианта. Внутри рисунка 7 геометрия сравнивается при неизменных сопротивлениях всех материалов. Между рисунками 5 и 7 одновременно изменены сопротивления фона, лёгкого и целого сердца: их разность относится ко всему материальному профилю, а не к изолированному свойству крови. Однофакторное сравнение только ρ целого сердца при постоянных фоне и лёгком не проводилось. Разность сферы и маски при одинаковом целевом объёме характеризует эффект формы внутри заданного профиля; три точки не дают обоснования для расчёта ударного объёма или экстраполяции секущей за проверенный диапазон.")
    md("""<a id="conclusion"></a>
## 8. Вывод и воспроизведение

**Модельный результат.** Индивидуальные прямые модели дают знаковые базовые
ответы, локальные производные и условный дыхательный контраст при явно заданных
материалах и источниках. В диагностическом сравнении совместный набор монтажей
улучшает обусловленность по сравнению с каждым отведением отдельно; слабое
направление, связанное с сердцем, сохраняется. Надёжное индивидуальное
определение тканевых ρ в проверенной постановке не получено. Этот результат
ограничен рассмотренными моделями, измерениями и сценариями; универсальная
теорема неприменимости боковых сборок для любых условий не утверждается.

**Решение автора и граница его обоснования.** Конечным входом дальнейшей
методики выбран только сигнал ТТРКГ. Боковые результаты сохраняются как
отрицательная проверка и не становятся предлагаемой рабочей инверсией ρ.
Это принятое направление разработки; его выбор сам по себе не валидирует
оценку сердечной функции по ТТРКГ. Свойства тканей для этой ветви остаются
сценарными либо референсными. Один сигнал или одна пара отведений не
идентифицирует произвольно четыре тканевых ρ; дополнительные предпосылки
и независимые проверки остаются необходимыми.

**Экспериментальное сопоставление.** Абсолютные уровни и парные разности
сопоставлены отдельно. Контактные условия, геометрия, тканевые свойства,
ретроспективно выделенные дыхательные интервалы и состояние между последовательными монтажами
ограничивают объяснение невязок. Шесть отказов размещения не дают электрической
оценки его погрешности. Сеточная сходимость и клиническая валидация не доказаны;
УО/ФВ в этом вычислительном цикле не получены.

Отчёт читает сохранённые JSON и обезличенный агрегат. При чистом запуске
ноутбука выполняются загрузка агрегата, арифметическая постобработка и построение
семи рисунков; FEM, КТ и сигналы не пересчитываются. Полные источники доступны
локально, а ниже сохранены только относительные пути и SHA. HTML экспортируется
без исходного Python-кода. Окончательная готовность входов не повышает
доказательный статус научных выводов.""")
    range_conclusion=[]
    for s in summary["subjects"]:
        diagnostics=s["material_range"]["scan"]["joint_grid_diagnostics"]
        if not diagnostics:continue
        values=[next(d for d in diagnostics if d["state"]==state and d["scope"]=="primary_pair")["minimum_on_sampled_grid"]["euclidean_norm_ohm"] for state in ("inhale","exhale")]
        range_conclusion.append(f"{s['id']}: минимальная норма двух невязок на сетке {values[0]:.6g} Ом для вдоха и {values[1]:.6g} Ом для выдоха")
    if range_conclusion:
        md("**Совместность новой материальной сетки.** "+"; ".join(range_conclusion)+". Эти числа относятся к own_drive при подтверждённом временном разделении и 15 рассчитанным точкам; непрерывный подбор сопротивлений здесь не выполнялся. Снижение отдельной невязки нельзя принимать за согласие обоих отведений.")
    code("display(HTML(F.table_html(['Относительный источник','SHA-256'],[[r['relative_path'],r['sha256']] for r in S['sources']],'Таблица 7. Происхождение сохранённых агрегатов и прямых результатов')))")
    for cell in cells:
        if cell.cell_type=="markdown":
            if not draft:
                for missing in ("Для отсутствующих результатов числовые строки остаются пустыми.",
                    "До завершения прямых расчётов численные поля\nостаются пустыми. "):
                    cell.source=cell.source.replace(missing,"")
    book=nb.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(display_name="Python 3",language="python",name="python3"),
        language_info=dict(name="python",version="3"),generator="build_scorpion_fem_smart_report",draft=draft,
        title=title,final_inputs_complete=summary["final_inputs_complete"],source_summary_sha256=None))
    return book


def build(model_root, *, draft=False, execute=False, export_html=False):
    import nbformat
    summary=collect(model_root)
    if not draft and not summary["final_inputs_complete"]:
        raise ValueError("Final report blocked: require 24 completed heart solves and six own material-range forwards; use --draft")
    target=FOLDER/(STEM+".ipynb");asset=FOLDER/"_assets"/ASSET
    if target.exists() and nbformat.read(target,as_version=4).metadata.get("generator")!="build_scorpion_fem_smart_report":
        raise ValueError("Notebook number already occupied by another generator")
    if asset.exists() and read(asset).get("schema")!="scorpion_fem_smart_reader_v1":
        raise ValueError("Asset name already occupied")
    asset.parent.mkdir(parents=True,exist_ok=True)
    asset.write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8",newline="\n")
    book=notebook(summary,draft);book.metadata.source_summary_sha256=digest(asset)
    if execute:
        from nbclient import NotebookClient
        with tempfile.TemporaryDirectory(prefix="fem_smart_reader_",dir=Path(model_root)) as runtime:
            old={key:os.environ.get(key) for key in ("JUPYTER_RUNTIME_DIR","IPYTHONDIR","MPLCONFIGDIR")}
            for key in old:
                directory=Path(runtime)/key.lower();directory.mkdir();os.environ[key]=str(directory)
            try:NotebookClient(book,timeout=180,startup_timeout=45,iopub_timeout=45,kernel_name="python3",resources={"metadata":{"path":str(FOLDER)}}).execute()
            finally:
                for key,value in old.items():
                    if value is None:os.environ.pop(key,None)
                    else:os.environ[key]=value
        book.metadata.executed_clean_local=True
    else:book.metadata.executed_clean_local=False
    payload=nbformat.writes(book);assert_safe(json.loads(payload))
    target.write_text(payload,encoding="utf-8",newline="\n")
    if export_html:
        from nbconvert import HTMLExporter
        exporter=HTMLExporter(template_name="lab",exclude_input=True,exclude_input_prompt=True,exclude_output_prompt=True)
        exporter.embed_images=True
        rendered,_=exporter.from_notebook_node(book)
        rendered=re.sub(r"<title>.*?</title>","<title>"+html.escape(book.metadata.title)+"</title>",rendered,flags=re.S)
        rendered=re.sub(r'<script\b[^>]*\bsrc=[^>]*>.*?</script>',"",rendered,flags=re.S|re.I)
        rendered=rendered.replace("</head>","<style>body{font-size:16px;line-height:1.7}.jp-Notebook{max-width:1180px;margin:auto}details{margin:1em 0}summary{cursor:pointer;font-weight:600}.scientific-table{overflow:auto}img{max-width:100%}</style></head>")
        if re.search(r'<details[^>]*\bopen\b',rendered,re.I) or "F.show_figure(" in rendered or "import build_scorpion_fem_smart_report" in rendered:
            raise ValueError("Reader contains open table or source code")
        assert_safe(dict(reader=rendered))
        (FOLDER/(STEM+".html")).write_text(rendered,encoding="utf-8",newline="\n")
    print(json.dumps(dict(notebook=target.name,draft=draft,executed=execute,
        heart_ready=summary["heart_results_completed"],range_ready=summary["material_range_completed"],
        final_inputs_complete=summary["final_inputs_complete"]),ensure_ascii=False))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root",type=Path,required=True)
    parser.add_argument("--draft",action="store_true")
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--html",action="store_true")
    args=parser.parse_args();build(args.model_root,draft=args.draft,execute=args.execute,export_html=args.html)


if __name__=="__main__":main()
