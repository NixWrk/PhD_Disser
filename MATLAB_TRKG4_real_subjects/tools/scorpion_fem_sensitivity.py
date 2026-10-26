"""Signed local identifiability analysis of saved canonical Scorpion FEM outputs.

No field assembly, PDE solve, fitting, noise model, or cross-subject pooling.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ACTIVE = (1, 2, 4, 5)  # MATLAB indices; bone (3) is fixed
PARAMETERS = ("soft", "lung", "wholeheart", "fat")
ROLES = {"ttrkg": (1, 0), "lateral": (3, 2)}  # Python indices


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def records(value):
    """MATLAB jsonencode emits a scalar struct as an object, not a one-row list."""
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return value
    raise ValueError("Expected a MATLAB scalar struct or struct array")


def matrix_diagnostics(matrix, *, rank_rtol=None) -> dict:
    """Full parameter-space SVD, including structural nullspace for m < n."""
    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or not a.size or not np.isfinite(a).all():
        raise ValueError("Expected a nonempty finite real matrix")
    if rank_rtol is not None and not (np.isfinite(rank_rtol) and 0 < rank_rtol < 1):
        raise ValueError("rank_rtol must lie strictly between zero and one")
    _, values, right = np.linalg.svd(a, full_matrices=True)
    # On Windows longdouble aliases float64; a prior dependency import can
    # populate NumPy's finfo cache with a longdouble eps scalar for this dtype.
    # Keep the float64 numerical value, but export an ordinary JSON number.
    rtol = float(max(a.shape) * float(np.finfo(np.float64).eps) if rank_rtol is None else rank_rtol)
    threshold = float(values[0] * rtol)
    rank = int(np.count_nonzero(values > threshold))
    # Deterministic signs aid comparison; a repeated singular value still has
    # an arbitrary orthonormal basis, not separately identified directions.
    for row in right:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    n = a.shape[1]
    spectrum = np.pad(values, (0, max(0, n - len(values))))
    full = rank == n
    return dict(matrix=a.tolist(), shape=list(a.shape), rank=rank,
                parameter_count=n, nullity=n-rank, rank_relative_tolerance=rtol,
                rank_absolute_threshold=threshold,
                singular_values=values.tolist(),
                parameter_space_singular_values=spectrum.tolist(),
                condition_number_full_parameter_space=float(values[0]/values[n-1]) if full else None,
                condition_status="finite" if full else "rank_deficient_infinite",
                condition_number_on_resolved_subspace=float(values[0]/values[rank-1]) if rank else None,
                full_column_rank=full, right_singular_vectors=right.tolist(),
                nullspace_basis=right[rank:].tolist(),
                weakest_parameter_direction=right[-1].tolist(),
                direction_coordinates="columns_of_this_matrix",
                rank_is_local_algebraic_not_practical_identifiability=True)


def _validate_baseline(report: dict, subject: str | None):
    if report.get("schema") not in ("scorpion_own_CT_forward_v1", "scorpion_own_CT_forward_v2", "scorpion_smart_forward_v1"):
        raise ValueError("Unsupported baseline schema; an explicit adapter is required")
    if not report.get("FEM_was_run") or report.get("frequency_hz") != 100000:
        raise ValueError("A completed 100 kHz canonical FEM baseline is required")
    alias = report.get("subject_alias")
    if not alias or (subject is not None and alias != subject):
        raise ValueError("Subject identity mismatch")
    if report.get("identity", {}).get("subject_alias", alias) != alias:
        raise ValueError("Conflicting baseline subject identity")
    if report.get("source_relative_phase") != "unknown":
        raise ValueError("Expected the explicitly unknown source relative phase")
    if not report.get("own_drive_is_working_assumption"):
        raise ValueError("Expected the own_drive baseline assumption")
    active = report.get("active_material_indices", report.get("derivative_checks", {}).get("active_material_indices", ACTIVE))
    if list(active) != list(ACTIVE):
        raise ValueError("Jacobian must use active material order [1,2,4,5]")
    for container in (report.get("reference_materials", {}), report.get("identity", {})):
        order = container.get("material_order")
        if order is not None and (len(order) != 5 or
                order[0] not in ("soft", "soft_background") or order[1:3] != ["lung", "bone"] or
                order[3] not in ("heart", "wholeheart") or order[4] != "fat"):
            raise ValueError("Unexpected five-material ordering")
    if not isinstance(report.get("answers"), list) or not report["answers"]:
        raise ValueError("No completed baseline answers")
    if report.get("schema") == "scorpion_own_CT_forward_v2":
        required = ("identity", "comparison", "system_checks", "derivative_checks", "active_material_indices", "limited_run")
        if any(key not in report for key in required) or len(report["system_checks"]) != len(report["answers"]):
            raise ValueError("Incomplete v2 forward report; do not use individual answer/prepared files")


def _rows(report):
    rows = {}
    by_state = {}
    for answer in report["answers"]:
        state, rule, case = answer["state"], answer["rule"], answer["case_id"]
        if state not in ("inhale", "exhale") or rule not in (1024, 4096):
            raise ValueError("Only separately analysed inhale/exhale q1024/q4096 are supported")
        if answer.get("subject_alias", report["subject_alias"]) != report["subject_alias"]:
            raise ValueError("Mixed subjects in baseline answers")
        if list(answer.get("active_material_indices", ACTIVE)) != list(ACTIVE):
            raise ValueError("Inconsistent per-answer active material order")
        rho = np.asarray(answer["rho_ohm_m"], dtype=float)
        t = np.asarray(answer["transfer_matrix_ohm"], dtype=float)
        j = np.asarray(answer["jacobian_logrho"], dtype=float)
        if rho.shape != (5,) or t.shape != (4, 4) or j.shape != (4, 4, 4):
            raise ValueError("Expected rho(5), T(4,4), J(4,4,4) in MATLAB JSON axis order")
        if not (np.isfinite(rho).all() and np.all(rho > 0) and np.isfinite(t).all() and np.isfinite(j).all()):
            raise ValueError("Invalid FEM numeric values")
        if state in by_state and not np.array_equal(rho, by_state[state]):
            raise ValueError("Different rho within one state cannot be stacked")
        by_state[state] = rho
        for role, (i, k) in ROLES.items():
            if not np.isclose(answer[role+"_Z_ohm"], t[i, k], rtol=1e-12, atol=1e-12):
                raise ValueError("Reported Z disagrees with signed transfer matrix entry")
            key = (state, rule, case, role)
            if key in rows:
                raise ValueError("Duplicate baseline case/state/rule/role")
            rows[key] = (float(t[i, k]), j[i, k, :], rho[list(np.array(ACTIVE)-1)])
    if len({rho[2] for rho in by_state.values()}) != 1:
        raise ValueError("Bone must remain fixed across breathing scenarios")
    return rows


def _subset(rows, state, rule, roles):
    keys = sorted(k for k in rows if k[0] == state and k[1] == rule and k[3] in roles)
    if not keys:
        raise ValueError("Empty measurement subset")
    z = np.array([rows[k][0] for k in keys])
    jlog = np.array([rows[k][1] for k in keys])
    rho = np.array([rows[k][2] for k in keys])
    return keys, z, jlog, jlog / rho


def analyze_baseline(report: dict, *, subject=None, rank_rtol=None) -> dict:
    report = dict(report)
    if "answers" in report:
        report["answers"] = records(report["answers"])
    if "system_checks" in report:
        report["system_checks"] = records(report["system_checks"])
    _validate_baseline(report, subject)
    rows = _rows(report)
    states = sorted({key[0] for key in rows})
    analyses = []
    checks = report.get("derivative_checks", {})
    for state in states:
        for name, roles in (("lateral", ("lateral",)), ("ttrkg", ("ttrkg",)), ("joint", tuple(ROLES))):
            available_rules = {k[1] for k in rows if k[0] == state}
            rule = max(available_rules)
            fine_keys, z, jlog, jrho = _subset(rows, state, rule, roles)
            paired = available_rules == {1024, 4096}
            if paired:
                coarse_keys, coarse_z, coarse_jlog, coarse_jrho = _subset(rows, state, 1024, roles)
                if [(k[2], k[3]) for k in fine_keys] != [(k[2], k[3]) for k in coarse_keys]:
                    raise ValueError("Unpaired q1024/q4096 measurement rows")
            normalized = None if np.any(z == 0) else jlog / z[:, None]
            coarse_normalized = None if not paired or np.any(coarse_z == 0) else coarse_jlog / coarse_z[:, None]
            matrices = dict(dZ_drho=matrix_diagnostics(jrho, rank_rtol=rank_rtol),
                            dZ_dlogrho=matrix_diagnostics(jlog, rank_rtol=rank_rtol))
            matrices["dimensionless_signed_elasticity"] = (
                matrix_diagnostics(normalized, rank_rtol=rank_rtol) if normalized is not None else None)
            for label, diagnostics in matrices.items():
                if diagnostics is not None:
                    diagnostics["perturbation_coordinates"] = (
                        "delta_rho_ohm_m" if label == "dZ_drho" else "delta_logrho")
            differences = {}
            comparison_matrices = (("dZ_drho", jrho, coarse_jrho), ("dZ_dlogrho", jlog, coarse_jlog),
                                    ("dimensionless_signed_elasticity", normalized, coarse_normalized)) if paired else ()
            for label, fine, coarse in comparison_matrices:
                if fine is None or coarse is None:
                    differences[label] = dict(status="unavailable_zero_Z")
                    continue
                delta = fine - coarse
                norm = float(np.linalg.norm(delta, 2))
                scale = float(np.linalg.norm(fine, 2))
                differences[label] = dict(signed_difference=delta.tolist(), spectral_norm=norm,
                    relative_spectral_norm=norm/scale if scale else None,
                    fine_singular_values_above_observed_difference=int(np.count_nonzero(np.linalg.svd(fine, compute_uv=False) > norm)),
                    coarse_diagnostics=matrix_diagnostics(coarse, rank_rtol=rank_rtol),
                    difference_is_not_error_bound_or_noise_variance=True)
            fd_covered = bool(checks.get("passed") and checks.get("state") == state and checks.get("rule",4096) == rule and
                              any(k[2] == checks.get("case_id") for k in fine_keys))
            analyses.append(dict(state=state, subset=name, rule=rule,
                row_labels=[dict(case_id=k[2], role=k[3]) for k in fine_keys],
                Z_ohm=z.tolist(), rho_active_ohm_m=rows[fine_keys[0]][2].tolist(),
                matrices=matrices, normalization_status="undefined_zero_Z" if normalized is None else "defined",
                quadrature_control=dict(status="paired" if paired else "not_run_both_rules",
                    available_rules=sorted(available_rules),
                    Z_signed_fine_minus_coarse_ohm=(z-coarse_z).tolist() if paired else None, matrices=differences),
                finite_difference_control_applies_to_at_least_one_case=fd_covered,
                finite_difference_control_covers_all_cases=bool(fd_covered and len({k[2] for k in fine_keys}) == 1),
                interpretation="local_model_result_at_fixed_CT_mesh_contacts_and_bone",
                exhale_is_fixed_inhale_CT_conditional_scenario=(state == "exhale")))
    return dict(schema="scorpion_fem_sensitivity_v1", subject_alias=report["subject_alias"],
        generated_utc=datetime.now(timezone.utc).isoformat(), parameters=list(PARAMETERS),
        active_material_indices=list(ACTIVE), fixed_material="bone", frequency_hz=100000,
        source_rule="own_drive", source_relative_phase="unknown", baseline_identity=report.get("identity", {}),
        definitions=dict(dZ_drho="Jlog[:,j]/rho[j]; unit ohm/(ohm m) = 1/m",
            dZ_dlogrho="rho[j]*dZ/drho[j]; unit ohm",
            dimensionless_signed_elasticity="(rho[j]/Z[i])*dZ[i]/drho[j] = Jlog[i,j]/Z[i]",
            normalization_is_noise_whitening=False, direction_order=list(PARAMETERS)),
        analyses=analyses, baseline_derivative_checks=checks,
        baseline_system_checks=report.get("system_checks", []),
        baseline_integration_passed=report.get("integration_passed"),
        baseline_limited_run=report.get("limited_run", False),
        baseline_coverage_complete=report.get("coverage_complete"),
        baseline_reported_numerical_checks_passed=bool(checks.get("passed") and report.get("integration_passed")),
        numerical_controls_complete=bool(report.get("integration_passed") and
            all(a["finite_difference_control_covers_all_cases"] and a["quadrature_control"]["status"] == "paired" for a in analyses)),
        baseline_numbers_available=True, placement_test_complete=False,
        noise_variance_assigned=False, confidence_intervals_computed=False,
        practically_identifiable=False, practical_identifiability_status="not_established",
        quantitative_result_accepted=False,
        clinical_validation=False, stroke_volume_estimated=False,
        limitations=["SVD rank is local and tolerance dependent; no statistical identifiability claim.",
            "Contact and anatomy uncertainty and source phase have not been marginalized.",
            "q1024/q4096 differences are integration diagnostics, not mesh convergence or rigorous bounds.",
            "Inhale and fixed-CT exhale and distinct subjects are never pooled.",
            "Finite-difference control applies only to the reported case and state."])


def summarize_placement(report: dict) -> dict:
    """Compare only newly solved placement scenarios with their own baseline."""
    if report.get("schema") != "scorpion_contact_sensitivity_v1" or not report.get("FEM_was_run"):
        raise ValueError("Completed contact sensitivity result required")
    base = report["baseline"]
    if report.get("frequency_hz") != 100000 or report.get("active_material_indices") != list(ACTIVE):
        raise ValueError("Placement material/frequency contract differs from baseline")
    if report.get("source_rule") != "own_drive" or report.get("source_relative_phase") != "unknown":
        raise ValueError("Placement source assumptions differ")
    if report.get("case_id") != "L140_TT2x2" or not report.get("CT_mesh_materials_unchanged"):
        raise ValueError("Expected the fixed-CT L140_TT2x2 placement pilot")
    scenarios = []
    ids = set()
    for row in records(report["answers"]):
        t, jt = np.asarray(row["transfer_matrix_ohm"]), np.asarray(row["jacobian_logrho"])
        bt, bj = np.asarray(base["transfer_matrix_ohm"]), np.asarray(base["jacobian_logrho"])
        if t.shape != (4,4) or jt.shape != (4,4,4) or bt.shape != (4,4) or bj.shape != (4,4,4) or not all(np.isfinite(a).all() for a in (t,jt,bt,bj)):
            raise ValueError("Invalid placement matrix")
        if row["scenario_id"] in ids:
            raise ValueError("Duplicate placement scenario")
        ids.add(row["scenario_id"])
        offset = row["placement"]["requested_offset_mm"]
        if offset not in (-5, 5) or row["placement"]["requested_axis_LPS"] != [0, 1, 0]:
            raise ValueError("Expected explicit +/-5mm LPS-Y placement scenarios")
        scenarios.append(dict(scenario_id=row["scenario_id"],
            requested_offset_mm=offset,
            roles={role:dict(Z_ohm=float(t[i,k]), signed_delta_Z_ohm=float(t[i,k]-bt[i,k]),
                jacobian_logrho=jt[i,k,:].tolist(), signed_delta_jacobian_logrho=(jt[i,k,:]-bj[i,k,:]).tolist())
                for role,(i,k) in ROLES.items()}))
    return dict(subject_alias=report["subject_alias"], case_id=report["case_id"], scenarios=scenarios,
                both_requested_scenarios_complete=(len(scenarios) == 2 and
                    {s["requested_offset_mm"] for s in scenarios} == {-5, 5}),
                scenario_offsets_are_confidence_intervals=False, geometry_uncertainty_exhausted=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh local excluded results directory")
    parser.add_argument("--subject", choices=("georg", "nik", "yaroslav"))
    parser.add_argument("--rank-rtol", type=float, default=None, help="Explicit relative algebraic rank threshold")
    parser.add_argument("--placement-result", type=Path)
    args = parser.parse_args(argv)
    root = args.model_root.resolve()
    source = root / "electrical" / "forward.local.json"
    source_digest = sha256(source)
    baseline = read_json(source)
    result = analyze_baseline(baseline, subject=args.subject, rank_rtol=args.rank_rtol)
    result["analysis_code"] = dict(path=str(Path(__file__).resolve()), sha256=sha256(Path(__file__)))
    result["numpy_version"] = np.__version__
    result["source"] = dict(path=str(source), sha256=source_digest)
    result["supporting_inputs"] = [dict(path=str(p), present=p.is_file()) for p in (
        root/"electrical"/"prepared_materials.mat", root/"cem"/"cem_contacts.mat")]
    if args.placement_result:
        placement = read_json(args.placement_result)
        if placement["subject_alias"] != result["subject_alias"] or placement["baseline_forward_sha256"] != source_digest:
            raise ValueError("Placement result belongs to a different subject/baseline")
        result["placement"] = summarize_placement(placement)
        result["placement_test_complete"] = result["placement"]["both_requested_scenarios_complete"]
        result["placement_source"] = dict(path=str(args.placement_result.resolve()), sha256=sha256(args.placement_result))
    if sha256(source) != source_digest:
        raise ValueError("Baseline changed during analysis")
    output = args.output_dir.resolve()
    if output == root or output in source.parents or output.exists():
        raise ValueError("Use a fresh separate output directory")
    output.mkdir(parents=True, exist_ok=False)
    target = output / "sensitivity.local.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(subject=result["subject_alias"], output=str(target),
                         baseline_numbers_available=True, numerical_controls_complete=result["numerical_controls_complete"],
                         placement_test_complete=result["placement_test_complete"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
