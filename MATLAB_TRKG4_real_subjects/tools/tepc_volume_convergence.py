"""Independent-volume convergence workflow for the TEPC montage comparison.

The outer CT-derived body surface, tissue masks, contact diameter and rho states
remain fixed. L12 is the accepted baseline; L09 and L06 are independently
tetrahedralised, reclassified, contact-refined, solved and compared.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PARAMS = ("soft", "heart", "lung")
MONTAGES = tuple(f"tepc_{number}" for number in range(2, 8))
STATUS = "exploratory_hypothesis_not_validated"
ARTIFACTS = {"identity.json", "source_used.zip", "channel_results.csv"}
_EXPECTED_TRANSITIONS = ("L12_to_L09", "L09_to_L06")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _selected_l12_scales(baseline_frame, metric_specs):
    require(
        not baseline_frame.empty and set(baseline_frame["level"]) == {"L12"},
        "Scale source must contain selected L12 rows only",
    )
    return {
        (metric, compartment): float(baseline_frame[field].abs().max())
        for metric, compartment, field in metric_specs
    }


def _transition_passes(frame):
    result = {}
    for transition in _EXPECTED_TRANSITIONS:
        rows = frame[frame["transition"] == transition]
        result[transition] = bool(not rows.empty and rows["passed"].all())
    return result


def _convergence_checks(comparison_frame, volume_frame, top_frame, pair_frame):
    final_metric = comparison_frame[comparison_frame["transition"] == "L09_to_L06"]
    final_volume = volume_frame[volume_frame["transition"] == "L09_to_L06"]
    metric_passes = _transition_passes(comparison_frame)
    volume_passes = _transition_passes(volume_frame)
    return {
        "all_final_metric_changes_within_tolerance": bool(final_metric["passed"].all()),
        "all_transition_metric_changes_within_tolerance": bool(all(metric_passes.values())),
        "all_selected_top_montages_stable_across_three_levels": bool(top_frame["top_stable"].all()),
        "all_fine_top_gaps_exceed_mesh_uncertainty": bool(top_frame["fine_top_gap_resolved"].all()),
        "all_resolved_pair_preferences_stable": bool(pair_frame["acceptable_stable_or_unresolved"].all()),
        "all_final_volume_changes_within_tolerance": bool(final_volume["passed"].all()),
        "all_transition_volume_changes_within_tolerance": bool(all(volume_passes.values())),
    }


def sha(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def relative(path: Path, base: Path) -> str:
    return os.path.relpath(Path(path).resolve(), Path(base).resolve()).replace("\\", "/")


def configure(deps: Path) -> None:
    for path in (Path(deps).resolve(), ROOT / "tools"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def verified_baseline(base: Path, deps: Path):
    configure(deps)
    import run_surface_sensitivity as runner

    runner.configure(deps)
    plan, profile, studies = runner.check_inputs(base)
    runner.verified_pilot(base, plan, studies)
    require(tuple(plan["montages"]) == MONTAGES, "Unexpected baseline montage set")
    summary_path = base / "analysis/summary.json"
    require(summary_path.exists(), "Baseline analysis is missing")
    require(
        sha(summary_path) == (summary_path.parent / "summary.sha256").read_text().strip(),
        "Baseline analysis hash mismatch",
    )
    return plan, profile, studies, read(base / "manifest.json")


def refine_contacts(args) -> None:
    configure(args.deps)
    import numpy as np
    from scipy.io import loadmat, savemat
    import refine_surface_contact_mesh as refinement

    refinement.np = np
    level = args.level.resolve()
    classification_path = level / "classification/classification.json"
    source_path = level / "classification/classified_mesh.mat"
    require(classification_path.exists() and source_path.exists(), "Classified mesh is missing")
    classification = read(classification_path)
    require(
        classification["status"] == "passed_independent_mesh_geometry_and_tissue_classification",
        "Independent mesh classification did not pass",
    )
    require(classification["classified_mesh_sha256"] == sha(source_path), "Classification hash mismatch")
    output = level / "refinement"
    mesh_path = output / "refined_mesh.mat"
    report_path = output / "refinement.json"
    if mesh_path.exists() and report_path.exists():
        report = read(report_path)
        require(report["refined_mesh_sha256"] == sha(mesh_path), "Existing refinement hash mismatch")
        require(math.isclose(report["target_edge_mm"], args.target_edge_mm), "Existing target differs")
        print(json.dumps({"status": "reused_verified_refinement", "level": str(level)}))
        return
    require(not output.exists(), "Partial refinement directory exists; preserve it for diagnosis")
    output.mkdir(parents=True)

    data = loadmat(
        source_path,
        variable_names=["nodes_mm", "elems", "boundary", "tissue_id", "group_id"],
    )
    require(
        {"nodes_mm", "elems", "boundary", "tissue_id", "group_id"} <= set(data),
        "Classified MAT is incomplete or not MATLAB v7 format",
    )
    nodes = np.asarray(data["nodes_mm"], dtype=float)
    elems = np.asarray(data["elems"], dtype=np.int32) - 1
    boundary = np.asarray(data["boundary"], dtype=np.int32) - 1
    tissue = np.asarray(data["tissue_id"], dtype=np.int32).reshape(-1)
    groups = np.asarray(data["group_id"], dtype=np.int32).reshape(-1)
    require(len(elems) == len(tissue) == len(groups), "Element labels are incomplete")
    original_nodes = nodes.copy()
    original_tetrahedra = len(elems)
    original_volume = refinement.volumes(nodes, elems)
    require((original_volume > 0).all(), "Classified mesh contains nonpositive tetrahedra")
    before = np.bincount(groups, weights=original_volume, minlength=6)

    baseline = args.baseline.resolve()
    manifest = read(baseline / "manifest.json")
    centres = []
    for item in manifest["inputs"]:
        source = read(baseline / item["json"])
        require(sha(baseline / item["json"]) == item["sha256"], "Baseline landmark hash mismatch")
        centres.extend(source["preliminary_electrode_centers_xyz_mm"].values())
    centres = np.asarray(centres, dtype=float)

    nodes, elems, boundary, tissue, groups, history = refinement.refine(
        nodes,
        elems,
        boundary,
        tissue,
        groups,
        centres,
        radius=args.radius_mm,
        target=args.target_edge_mm,
    )
    volume = refinement.volumes(nodes, elems)
    after = np.bincount(groups, weights=volume, minlength=6)
    require((volume > 0).all(), "Refined mesh contains nonpositive tetrahedra")
    require(np.array_equal(nodes[: len(original_nodes)], original_nodes), "Parent vertices changed")
    require(np.allclose(before, after, rtol=1e-10, atol=1e-5), "Group volumes changed")
    refinement.verify_boundary(elems, boundary)
    savemat(
        mesh_path,
        {
            "nodes_mm": nodes,
            "elems": elems + 1,
            "boundary": boundary + 1,
            "tissue_id": tissue[:, None],
            "group_id": groups[:, None],
        },
        do_compression=False,
    )
    report = {
        "schema": "tepc_volume_contact_refinement_v1",
        "status": "passed_contact_refinement_for_volume_convergence",
        "scope": "identical_contact_neighbourhood_refinement_on_independent_volume_mesh",
        "level_id": classification["level_id"],
        "classification_sha256": sha(classification_path),
        "classified_mesh_sha256": sha(source_path),
        "refined_mesh_sha256": sha(mesh_path),
        "target_edge_mm": args.target_edge_mm,
        "contact_neighbourhood_radius_mm": args.radius_mm,
        "rounds": history,
        "counts": {
            "nodes": len(nodes),
            "tetrahedra": len(elems),
            "boundary": len(boundary),
            "added_nodes": len(nodes) - len(original_nodes),
            "added_tetrahedra": len(elems) - original_tetrahedra,
        },
        "checks": {
            "original_vertices_unchanged": True,
            "all_positive_tetrahedra": True,
            "group_volumes_preserved": True,
            "conforming_boundary_verified": True,
            "contact_neighbourhood_edges_le_target": True,
        },
        "group_volume_before_mm3": before.tolist(),
        "group_volume_after_mm3": after.tolist(),
        "min_tetra_volume_mm3": float(volume.min()),
        "physical_validation": False,
        "mesh_solution_convergence": False,
        "source_sha256": sha(Path(__file__)),
    }
    write(report_path, report)
    print(json.dumps(report, ensure_ascii=False))


def configure_level(args) -> None:
    configure(args.deps)
    from prepare_surface_sensitivity import import_archive, verify_manifest

    baseline = args.baseline.resolve()
    level = args.level.resolve()
    _, _, _, baseline_manifest = verified_baseline(baseline, args.deps)
    if (level / "manifest.json").exists():
        verify_manifest(level)
        print(json.dumps({"status": "reused_verified_manifest", "level": str(level)}))
        return

    mesh_path = level / "refinement/refined_mesh.mat"
    refinement_path = level / "refinement/refinement.json"
    export_path = level / "export/volume_export.json"
    prepared_path = level / "export/prepared_fem.mat"
    require(all(path.exists() for path in (mesh_path, refinement_path, export_path, prepared_path)),
            "Refinement/export artifacts are incomplete")
    refinement = read(refinement_path)
    export = read(export_path)
    require(
        refinement["status"] == "passed_contact_refinement_for_volume_convergence"
        and refinement["refined_mesh_sha256"] == sha(mesh_path),
        "Refinement provenance mismatch",
    )
    require(
        export["status"] == "passed_independent_volume_level_export"
        and export["matrix_error"] < 1e-12
        and export["prepared_fem_sha256"] == sha(prepared_path)
        and export["refined_mesh_sha256"] == sha(mesh_path),
        "Independent FEM export failed or is stale",
    )

    configuration = level / "configuration"
    configuration.mkdir(parents=True, exist_ok=False)
    acceptance_path = configuration / "acceptance.json"
    identity_path = configuration / "identity.json"
    acceptance = {
        "status": "accepted_for_cross_level_solution_convergence_test",
        "accepted_mesh_sha256": sha(mesh_path),
        "checks": refinement["checks"] | {"eidors_volume_contact_matrix_agreement": True},
        "classification_sha256": refinement["classification_sha256"],
        "refinement_sha256": sha(refinement_path),
        "export_sha256": sha(export_path),
        "physical_validation": False,
        "mesh_solution_convergence": False,
    }
    identity = {
        "status": "independent_volume_mesh_algebraic_check_passed",
        "prepared_fem_sha256": sha(prepared_path),
        "mesh_sha256": sha(mesh_path),
        "export_sha256": sha(export_path),
        "classification_sha256": refinement["classification_sha256"],
        "refinement_sha256": sha(refinement_path),
    }
    write(acceptance_path, acceptance)
    write(identity_path, identity)

    baseline_plan = read(baseline / "surface_plan.json")
    template_source = baseline / baseline_plan["pilots"][0]["study"]
    template = read(template_source)
    template_path = configuration / "template_study.json"
    template["source"]["prepared_fem"] = relative(prepared_path, configuration)
    template["source"]["prepared_sha256"] = sha(prepared_path)
    template["montages"][0]["contact_matrix"] = relative(
        (template_source.parent / template["montages"][0]["contact_matrix"]).resolve(),
        configuration,
    )
    write(template_path, template)

    source_surface = (baseline / baseline_manifest["source"]["surface"]).resolve()
    import_archive(
        SimpleNamespace(
            archive=baseline / "source_archive.zip",
            template=template_path,
            rho_profile=baseline / "rho_profile.json",
            mesh=mesh_path,
            mesh_acceptance=acceptance_path,
            model_identity=identity_path,
            surface=source_surface,
            output=level,
            prepared_format="si_geometry_si_blocks",
            diameter_mm=baseline_manifest["contact"]["diameter_mm"],
            z_contact=baseline_manifest["contact"]["z_contact_ohm_m2"],
        )
    )
    print(json.dumps({"status": "configured_for_contact_preparation", "level": str(level)}))

def _float(row, name: str) -> float:
    value = float(row[name])
    require(math.isfinite(value), f"Nonfinite {name}")
    return value


def _rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def select_states(args) -> None:
    baseline = args.baseline.resolve()
    plan, _, _, _ = verified_baseline(baseline, args.deps)
    output = args.output.resolve()
    state_rows = [item for item in plan["states"] if item["montage"] == MONTAGES[0]]
    state_map = {item["state"]: item for item in state_rows}
    require(len(state_map) == 189, "Baseline state catalogue is incomplete")
    reasons: dict[str, set[str]] = defaultdict(set)

    def add(state: str, reason: str) -> None:
        require(state in state_map, "Selected state is absent from baseline: " + state)
        reasons[state].add(reason)

    axes = plan["axes_rho"]
    grid_lookup = {
        tuple(item["rho"][name] for name in PARAMS): item["state"]
        for item in state_rows
        if item["state"].startswith("rho_grid_n5_")
    }
    require(len(grid_lookup) == 125, "Expected complete five-node grid")
    for indices in itertools.product((0, 4), repeat=3):
        add(grid_lookup[tuple(axes[name][index] for name, index in zip(PARAMS, indices))],
            "physiological_range_corner")
    centre = (2, 2, 2)
    add(grid_lookup[tuple(axes[name][2] for name in PARAMS)], "physiological_range_centre")
    for axis in range(3):
        for endpoint in (0, 4):
            indices = list(centre)
            indices[axis] = endpoint
            add(grid_lookup[tuple(axes[name][index] for name, index in zip(PARAMS, indices))],
                f"physiological_axis_endpoint_{PARAMS[axis]}")

    analysis = baseline / "analysis"
    interpolation_path = analysis / "interpolation_checks.csv"
    pair_path = analysis / "pair_checks.csv"
    direct_path = analysis / "states_rho.csv"
    interpolation = _rows(interpolation_path)
    pairs = _rows(pair_path)
    direct = _rows(direct_path)

    for compartment in PARAMS:
        group = [row for row in interpolation if row["compartment"] == compartment]
        for field, label in (
            ("S_error_over_tolerance", "largest_interpolation_S_error"),
            ("Q_absolute_error", "largest_interpolation_Q_error"),
        ):
            chosen = max(group, key=lambda row: _float(row, field))
            add(f"rho5_check_{int(chosen['check_index']):06d}", f"{label}_{compartment}")

        pair_group = [row for row in pairs if row["compartment"] == compartment]
        for field, label in (
            ("S_error_over_tolerance", "largest_pair_S_error"),
            ("Q_error_over_tolerance", "largest_pair_Q_error"),
        ):
            chosen = max(pair_group, key=lambda row: _float(row, field))
            add(f"rho5_check_{int(chosen['check_index']):06d}", f"{label}_{compartment}")
        for field, label in (
            ("S_direct_difference", "smallest_direct_pair_S_gap"),
            ("Q_direct_difference", "smallest_direct_pair_Q_gap"),
        ):
            chosen = min(pair_group, key=lambda row: abs(_float(row, field)))
            add(f"rho5_check_{int(chosen['check_index']):06d}", f"{label}_{compartment}")

    for row in interpolation:
        if row["S_sign_reversal"].casefold() == "true":
            add(f"rho5_check_{int(row['check_index']):06d}", "interpolation_sign_reversal")
    for row in pairs:
        if row["S_preference_reversal"].casefold() == "true":
            add(f"rho5_check_{int(row['check_index']):06d}", "pair_S_preference_reversal")
        if row["Q_preference_reversal"].casefold() == "true":
            add(f"rho5_check_{int(row['check_index']):06d}", "pair_Q_preference_reversal")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in direct:
        grouped[row["state"]].append(row)
    for metric in ("S", "B", "Q"):
        for compartment in PARAMS:
            candidates: dict[str, tuple[float, str]] = {}
            field = f"{metric}_{compartment}"
            for state, rows in grouped.items():
                require(len(rows) == len(MONTAGES), "Direct state lacks one or more montages")
                ordered = sorted(
                    ((_float(row, field), row["montage"]) for row in rows),
                    key=lambda item: (-item[0], item[1]),
                )
                top_value, top = ordered[0]
                margin = top_value - ordered[1][0]
                if top not in candidates or margin > candidates[top][0]:
                    candidates[top] = (margin, state)
            for top, (_margin, state) in candidates.items():
                add(state, f"observed_top_{metric}_{compartment}_{top}")

    ordered_states = sorted(
        reasons,
        key=lambda state: (
            0 if state.startswith("rho_grid") else 1,
            int(state.rsplit("_", 1)[1]),
        ),
    )
    selected = [
        {
            "state": state,
            "role": state_map[state]["role"],
            "rho": state_map[state]["rho"],
            "reasons": sorted(reasons[state]),
        }
        for state in ordered_states
    ]
    report = {
        "schema": "tepc_volume_convergence_state_selection_v1",
        "status": "selected_before_new_mesh_results",
        "selection_rule": {
            "fixed_grid_states": "centre, eight corners and six axial endpoints",
            "interpolation": "global largest S and Q errors for each compartment",
            "pair_accuracy": "global largest S and Q pair errors for each compartment",
            "pair_resolution": "smallest direct S and Q gaps for each compartment",
            "reversals": "all observed S/Q pair preference reversals and S sign reversals",
            "ranking": "one largest-margin state for every observed top montage by S, B and Q",
        },
        "baseline": str(baseline),
        "baseline_plan_sha256": sha(baseline / "surface_plan.json"),
        "baseline_analysis_sha256": sha(analysis / "summary.json"),
        "source_tables_sha256": {
            "interpolation_checks.csv": sha(interpolation_path),
            "pair_checks.csv": sha(pair_path),
            "states_rho.csv": sha(direct_path),
        },
        "montages": list(MONTAGES),
        "states_per_montage": len(selected),
        "total_new_solves_per_level": len(selected) * len(MONTAGES),
        "selected": selected,
        "physical_validation": False,
        "mesh_solution_convergence": False,
        "source_sha256": sha(Path(__file__)),
    }
    if output.exists():
        require(read(output) == report, "Existing state selection differs from deterministic rule")
    else:
        write(output, report)
    print(json.dumps({"status": report["status"], "states": len(selected),
                      "solves_per_level": report["total_new_solves_per_level"]}, ensure_ascii=False))


def make_studies(args) -> None:
    configure(args.deps)
    from electrode_sensitivity_contract import validate_study
    from prepare_surface_sensitivity import verify_manifest

    level = args.level.resolve()
    selection_path = args.selection.resolve()
    manifest = verify_manifest(level)
    selection = read(selection_path)
    require(selection["status"] == "selected_before_new_mesh_results", "State selection is invalid")
    require(selection["montages"] == list(MONTAGES), "State selection montage set differs")
    preparation_path = level / "contacts/preparation.json"
    preparation = read(preparation_path)
    require(
        preparation["status"] == "passed"
        and preparation["manifest_sha256"] == sha(level / "manifest.json")
        and preparation["prepared_sha256"] == manifest["source"]["prepared_sha256"],
        "Contact preparation failed or is stale",
    )
    contacts = {item["id"]: item for item in preparation["montages"]}
    require(set(contacts) == set(MONTAGES), "Prepared contacts do not match TEPC montages")
    for item in contacts.values():
        require(
            item["status"] == "passed"
            and item["contact_sha256"] == sha(level / item["contact_matrix"])
            and item["matrix_error"] < 1e-12,
            "Contact matrix failed QC",
        )

    plan_path = level / "validation_plan.json"
    if plan_path.exists():
        existing = read(plan_path)
        require(existing["selection_sha256"] == sha(selection_path), "Existing plan uses another selection")
        print(json.dumps({"status": "reused_verified_validation_plan", "level": str(level)}))
        return
    studies_dir = level / "studies"
    require(not studies_dir.exists(), "Partial studies directory exists")
    studies_dir.mkdir()
    bone_sigma = 1 / read(level / "rho_profile.json")["bone_fixed_rho_ohm_m"]
    jobs = []
    for montage in MONTAGES:
        contact = contacts[montage]
        states = []
        for selected in selection["selected"]:
            rho = selected["rho"]
            states.append({
                "id": selected["state"],
                "conductivity": {
                    "soft": 1 / rho["soft"],
                    "heart": 1 / rho["heart"],
                    "lung": 1 / rho["lung"],
                    "bone": bone_sigma,
                },
            })
        study_path = studies_dir / f"{montage}.json"
        study = {
            "schema_version": "electrode_sensitivity_v1",
            "status": STATUS,
            "source": {
                "prepared_fem": relative(level / manifest["source"]["prepared_fem"], studies_dir),
                "prepared_sha256": manifest["source"]["prepared_sha256"],
                "frequency_hz": manifest["frequency_hz"],
            },
            "parameters": copy.deepcopy(manifest["parameters"]),
            "states": states,
            "montages": [{
                "id": montage,
                "contact_matrix": relative(level / contact["contact_matrix"], studies_dir),
                "contact_sha256": contact["contact_sha256"],
                "electrodes": ["E1", "E2", "E3", "E4"],
                "contact_geometry_provenance":
                    "User surface JSON; four equal 5 mm contacts; independent volume mesh and level-specific contact QC.",
                "channels": [{
                    "id": "standard",
                    "drive": [1, 0, 0, -1],
                    "measure": [0, 1, -1, 0],
                }],
            }],
            "analysis": {"map_states": [], "save_fields": False},
        }
        write(study_path, validate_study(study))
        jobs.append({
            "montage": montage,
            "study": relative(study_path, level),
            "study_sha256": sha(study_path),
            "output": f"results/{montage}",
            "states": len(states),
        })
    plan = {
        "schema": "tepc_volume_convergence_plan_v1",
        "status": "prepared_not_computed",
        "level_id": read(level / "classification/classification.json")["level_id"],
        "manifest_sha256": sha(level / "manifest.json"),
        "preparation_sha256": sha(preparation_path),
        "selection": relative(selection_path, level),
        "selection_sha256": sha(selection_path),
        "prepared_sha256": manifest["source"]["prepared_sha256"],
        "contact_target_edge_mm": read(level / "refinement/refinement.json")["target_edge_mm"],
        "montages": list(MONTAGES),
        "states_per_montage": len(selection["selected"]),
        "total_states": len(selection["selected"]) * len(MONTAGES),
        "jobs": jobs,
        "solver_threads": args.threads,
        "physical_validation": False,
        "mesh_solution_convergence": False,
        "sources_sha256": {
            name: sha(ROOT / "tools" / name)
            for name in (
                "run_electrode_sensitivity.py",
                "electrode_sensitivity_contract.py",
                "compartment_sensitivity.py",
                "tepc_volume_convergence.py",
            )
        },
    }
    write(plan_path, plan)
    print(json.dumps({"status": plan["status"], "level": plan["level_id"],
                      "states": plan["total_states"]}, ensure_ascii=False))


def verify_level(args) -> None:
    configure(args.deps)
    import numpy as np
    import pandas as pd
    from electrode_sensitivity_contract import validate_study
    from run_electrode_sensitivity import verify_completed_output

    level = args.level.resolve()
    plan_path = level / "validation_plan.json"
    plan = read(plan_path)
    require(plan["manifest_sha256"] == sha(level / "manifest.json"), "Level manifest changed")
    require(plan["selection_sha256"] == sha(level / plan["selection"]), "Selection changed")
    completions = {}
    for job in plan["jobs"]:
        study_path = level / job["study"]
        require(job["study_sha256"] == sha(study_path), "Study changed: " + job["montage"])
        study = validate_study(read(study_path))
        output = level / job["output"]
        verify_completed_output(output, ARTIFACTS, job["states"], 0, STATUS)
        identity = read(output / "identity.json")
        require(identity["study"] == study and identity["study_sha256"] == job["study_sha256"],
                "Result identity differs from study")
        require(identity["threads"] == plan["solver_threads"] and identity["solver"] == "PARDISO float64",
                "Unexpected solver configuration")
        require(identity["sources"] == {
            name: plan["sources_sha256"][name]
            for name in ("run_electrode_sensitivity.py", "compartment_sensitivity.py",
                         "electrode_sensitivity_contract.py")
        }, "Solver sources differ")
        frame = pd.read_csv(output / "channel_results.csv", float_precision="round_trip")
        require(len(frame) == job["states"] and frame["state"].is_unique, "Result rows are incomplete")
        require(set(frame["state"]) == {item["id"] for item in study["states"]}, "Unexpected result states")
        require(frame["montage"].eq(job["montage"]).all(), "Unexpected montage in results")
        numeric = ["Z_ohm", "reciprocal_Z_ohm", "max_relative_residual"] + [
            "dZ_dsigma_" + name for name in PARAMS
        ]
        require(np.isfinite(frame[numeric].to_numpy(float)).all(), "Nonfinite numerical result")
        require((frame["max_relative_residual"] < 1e-7).all(), "Linear solve residual failed")
        completions[job["montage"]] = sha(output / "completion.json")
    report = {
        "schema": "tepc_volume_convergence_level_verification_v1",
        "status": "completed_numerical_checks_passed",
        "level_id": plan["level_id"],
        "plan_sha256": sha(plan_path),
        "states": plan["total_states"],
        "completion_sha256": completions,
        "physical_validation": False,
        "mesh_solution_convergence": False,
    }
    write(level / "verification.json", report)
    print(json.dumps(report, ensure_ascii=False))

def _load_baseline_frame(base: Path, selection: dict, deps: Path):
    configure(deps)
    import pandas as pd

    path = base / "analysis/states_rho.csv"
    require(selection["source_tables_sha256"]["states_rho.csv"] == sha(path),
            "Baseline states table differs from preselected input")
    frame = pd.read_csv(path, float_precision="round_trip")
    wanted = {item["state"] for item in selection["selected"]}
    frame = frame[frame["state"].isin(wanted)].copy()
    require(len(frame) == len(wanted) * len(MONTAGES), "Baseline selected results are incomplete")
    require(not frame.duplicated(["montage", "state"]).any(), "Duplicate baseline results")
    frame["level"] = "L12"
    return frame


def _load_level_frame(level: Path, deps: Path):
    configure(deps)
    import pandas as pd
    from analyze_resistivity_coverage import add_metrics

    verification = read(level / "verification.json")
    require(verification["status"] == "completed_numerical_checks_passed", "Level is not verified")
    plan = read(level / "validation_plan.json")
    require(verification["plan_sha256"] == sha(level / "validation_plan.json"), "Level plan changed")
    profile = read(level / "rho_profile.json")
    frames = []
    for job in plan["jobs"]:
        study = read(level / job["study"])
        lookup = {item["id"]: item["conductivity"] for item in study["states"]}
        frame = pd.read_csv(level / job["output"] / "channel_results.csv", float_precision="round_trip")
        for name in PARAMS:
            frame["sigma_" + name] = [lookup[state][name] for state in frame["state"]]
        frames.append(add_metrics(frame, profile))
    result = pd.concat(frames, ignore_index=True)
    require(len(result) == plan["total_states"], "Level results are incomplete")
    require(not result.duplicated(["montage", "state"]).any(), "Duplicate level results")
    result["level"] = plan["level_id"]
    return result


def _mesh_volumes(base: Path, levels: dict[str, Path]):
    baseline = read(base / "refinement/refinement.json")
    baseline_values = baseline["group_volume_after_mm3"]
    if len(baseline_values) == 6:
        baseline_values = baseline_values[1:]
    require(len(baseline_values) == 5, "Unexpected baseline group volume vector")
    result = {"L12": [float(value) for value in baseline_values]}
    for level_id, level in levels.items():
        classification = read(level / "classification/classification.json")
        require(classification["level_id"] == level_id, "Classification level mismatch")
        values = classification["group_volumes_mm3"]
        require(len(values) == 5, "Unexpected independent-mesh group volume vector")
        result[level_id] = [float(value) for value in values]
    return result


def _verify_independent_mesh_builds(base: Path, levels: dict[str, Path]):
    baseline_manifest = read(base / "manifest.json")
    surface_sha256 = baseline_manifest["source"]["surface_sha256"]
    expected_trunk_size = {"L09": 9.0, "L06": 6.0}
    result = {}
    for level_id, level in levels.items():
        reports = list((level / "mesh").glob("gmsh_build.json"))
        require(len(reports) == 1, f"Expected one Gmsh build report for {level_id}")
        report_path = reports[0]
        report = read(report_path)
        require(
            report["status"] == "candidate_mesh_requires_external_quality_and_matlab_qc",
            f"Unexpected Gmsh status for {level_id}",
        )
        require(report["source_sha256"] == surface_sha256, f"Surface hash mismatch for {level_id}")
        require(math.isclose(float(report["arm_size_mm"]), 1.0), f"Arm size mismatch for {level_id}")
        require(
            math.isclose(float(report["electrode_size_mm"]), 1.0),
            f"Electrode-neighbourhood size mismatch for {level_id}",
        )
        require(
            math.isclose(float(report["trunk_size_mm"]), expected_trunk_size[level_id]),
            f"Trunk size mismatch for {level_id}",
        )
        require(
            math.isclose(float(report["transition_mm"]), 5.0),
            f"Transition size mismatch for {level_id}",
        )
        require(int(report["algorithm_3d"]) == 1, f"Gmsh algorithm mismatch for {level_id}")
        mesh_path = Path(report["target"])
        require(mesh_path.exists(), f"Gmsh mesh is missing for {level_id}")
        require(
            mesh_path.resolve().parent == (level / "mesh").resolve(),
            f"Gmsh target lies outside the declared level for {level_id}",
        )
        classification = read(level / "classification/classification.json")
        require(classification["mesh_sha256"] == sha(mesh_path), f"Mesh hash mismatch for {level_id}")
        result[level_id] = {
            "report": relative(report_path, level),
            "report_sha256": sha(report_path),
            "mesh": relative(mesh_path, level),
            "mesh_sha256": sha(mesh_path),
            "source_surface_sha256": surface_sha256,
            "arm_size_mm": float(report["arm_size_mm"]),
            "electrode_size_mm": float(report["electrode_size_mm"]),
            "trunk_size_mm": float(report["trunk_size_mm"]),
            "transition_mm": float(report["transition_mm"]),
            "algorithm_3d": int(report["algorithm_3d"]),
            "nodes": int(report["nodes"]),
            "volume_elements": int(report["volume_elements"]),
        }
    return result


def analyse(args) -> None:
    configure(args.deps)
    import numpy as np
    import pandas as pd

    baseline = args.baseline.resolve()
    selection_path = args.selection.resolve()
    selection = read(selection_path)
    require(selection["status"] == "selected_before_new_mesh_results", "Invalid selection")
    levels = {"L09": args.level09.resolve(), "L06": args.level06.resolve()}
    baseline_frame = _load_baseline_frame(baseline, selection, args.deps)
    level_frames = [_load_level_frame(path, args.deps) for path in levels.values()]
    frames = [baseline_frame, *level_frames]
    data = pd.concat(frames, ignore_index=True)
    wanted = len(selection["selected"]) * len(MONTAGES) * 3
    require(len(data) == wanted, "Cross-level result matrix is incomplete")
    require(set(data["level"]) == {"L12", "L09", "L06"}, "Expected exactly three levels")
    mesh_builds = _verify_independent_mesh_builds(baseline, levels)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    require(not any(output.iterdir()), "Analysis output must be new or empty")
    data.to_csv(output / "level_results.csv", index=False)

    metric_specs = [("Z", "all", "Z_ohm")]
    metric_specs.extend(
        (metric, compartment, f"{metric}_{compartment}")
        for metric in ("S", "B", "Q")
        for compartment in PARAMS
    )
    scale = _selected_l12_scales(baseline_frame, metric_specs)
    indexed = data.set_index(["level", "montage", "state"])
    comparisons = []
    for coarse, fine in (("L12", "L09"), ("L09", "L06")):
        for montage in MONTAGES:
            for selected in selection["selected"]:
                state = selected["state"]
                for metric, compartment, field in metric_specs:
                    a = float(indexed.loc[(coarse, montage, state), field])
                    b = float(indexed.loc[(fine, montage, state), field])
                    difference = abs(b - a)
                    if metric == "Z":
                        tolerance = 1e-8 + 0.01 * max(abs(a), abs(b))
                        rule = "1% relative plus 1e-8 ohm"
                    elif metric in ("S", "B"):
                        tolerance = 0.02 * max(abs(a), abs(b)) + 0.005 * scale[(metric, compartment)]
                        rule = "2% local relative plus 0.5% global scale floor"
                    else:
                        tolerance = 0.01
                        rule = "0.01 absolute selectivity"
                    comparisons.append({
                        "transition": f"{coarse}_to_{fine}",
                        "coarse_level": coarse,
                        "fine_level": fine,
                        "montage": montage,
                        "state": state,
                        "metric": metric,
                        "compartment": compartment,
                        "coarse_value": a,
                        "fine_value": b,
                        "absolute_change": difference,
                        "relative_change": difference / max(abs(a), abs(b), np.finfo(float).tiny),
                        "global_scale": scale[(metric, compartment)],
                        "tolerance": tolerance,
                        "criterion": rule,
                        "passed": bool(difference <= tolerance),
                    })
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(output / "metric_convergence.csv", index=False)

    score_rows = []
    top_rows = []
    pair_rows = []
    for state in [item["state"] for item in selection["selected"]]:
        for metric in ("S", "B", "Q"):
            for compartment in PARAMS:
                field = f"{metric}_{compartment}"
                values = {}
                for level in ("L12", "L09", "L06"):
                    values[level] = {
                        montage: abs(float(indexed.loc[(level, montage, state), field]))
                        for montage in MONTAGES
                    }
                    ordered = sorted(values[level].items(), key=lambda item: (-item[1], item[0]))
                    for rank, (montage, score) in enumerate(ordered, 1):
                        score_rows.append({
                            "level": level,
                            "state": state,
                            "metric": metric,
                            "compartment": compartment,
                            "rank": rank,
                            "montage": montage,
                            "score": score,
                        })
                ordered_fine = sorted(values["L06"].items(), key=lambda item: (-item[1], item[0]))
                top, top_value = ordered_fine[0]
                second, second_value = ordered_fine[1]
                top_uncertainty = abs(values["L06"][top] - values["L09"][top])
                second_uncertainty = abs(values["L06"][second] - values["L09"][second])
                fine_gap = top_value - second_value
                top_by_level = {
                    level: sorted(
                        values[level], key=lambda montage: (-values[level][montage], montage)
                    )[0]
                    for level in ("L12", "L09", "L06")
                }
                top_rows.append({
                    "state": state,
                    "metric": metric,
                    "compartment": compartment,
                    "top_L12": top_by_level["L12"],
                    "top_L09": top_by_level["L09"],
                    "top_L06": top_by_level["L06"],
                    "top_stable": len(set(top_by_level.values())) == 1,
                    "fine_top": top,
                    "fine_second": second,
                    "fine_gap": fine_gap,
                    "fine_mesh_uncertainty_sum": top_uncertainty + second_uncertainty,
                    "fine_top_gap_resolved": bool(fine_gap > top_uncertainty + second_uncertainty),
                })
                for assembly_a, assembly_b in itertools.combinations(MONTAGES, 2):
                    differences = {
                        level: values[level][assembly_a] - values[level][assembly_b]
                        for level in ("L12", "L09", "L06")
                    }
                    signs = [int(np.sign(differences[level])) for level in ("L12", "L09", "L06")]
                    uncertainty = (
                        abs(values["L06"][assembly_a] - values["L09"][assembly_a])
                        + abs(values["L06"][assembly_b] - values["L09"][assembly_b])
                    )
                    fine_gap_pair = abs(differences["L06"])
                    resolved = fine_gap_pair > uncertainty
                    pair_rows.append({
                        "state": state,
                        "metric": metric,
                        "compartment": compartment,
                        "assembly_a": assembly_a,
                        "assembly_b": assembly_b,
                        "difference_L12": differences["L12"],
                        "difference_L09": differences["L09"],
                        "difference_L06": differences["L06"],
                        "preference_stable": len(set(signs)) == 1,
                        "fine_gap": fine_gap_pair,
                        "fine_mesh_uncertainty_sum": uncertainty,
                        "fine_pair_resolved": bool(resolved),
                        "acceptable_stable_or_unresolved": bool(len(set(signs)) == 1 or not resolved),
                    })
    ranking_frame = pd.DataFrame(score_rows)
    top_frame = pd.DataFrame(top_rows)
    pair_frame = pd.DataFrame(pair_rows)
    ranking_frame.to_csv(output / "rankings.csv", index=False)
    top_frame.to_csv(output / "top_stability.csv", index=False)
    pair_frame.to_csv(output / "pair_stability.csv", index=False)

    volume_names = ("soft_trunk", "soft_arms", "heart", "lungs", "bones")
    volumes = _mesh_volumes(baseline, levels)
    volume_rows = []
    for coarse, fine in (("L12", "L09"), ("L09", "L06")):
        total_a, total_b = sum(volumes[coarse]), sum(volumes[fine])
        total_change = abs(total_b - total_a) / max(abs(total_a), abs(total_b))
        volume_rows.append({
            "transition": f"{coarse}_to_{fine}",
            "region": "body_total",
            "coarse_volume_mm3": total_a,
            "fine_volume_mm3": total_b,
            "relative_change": total_change,
            "tolerance": 0.001,
            "passed": bool(total_change <= 0.001),
        })
        for index, name in enumerate(volume_names):
            a, b = volumes[coarse][index], volumes[fine][index]
            change = abs(b - a) / max(abs(a), abs(b))
            volume_rows.append({
                "transition": f"{coarse}_to_{fine}",
                "region": name,
                "coarse_volume_mm3": a,
                "fine_volume_mm3": b,
                "relative_change": change,
                "tolerance": 0.02,
                "passed": bool(change <= 0.02),
            })
    volume_frame = pd.DataFrame(volume_rows)
    volume_frame.to_csv(output / "volume_convergence.csv", index=False)

    checks = _convergence_checks(comparison_frame, volume_frame, top_frame, pair_frame)
    status = (
        "passed_selected_state_independent_volume_mesh_convergence"
        if all(checks.values())
        else "not_converged_or_preferences_unresolved_refinement_required"
    )
    worst = {}
    for metric, group in comparison_frame.groupby("metric"):
        item = group.loc[(group["absolute_change"] / group["tolerance"]).idxmax()]
        worst[metric] = {
            "transition": item["transition"],
            "montage": item["montage"],
            "state": item["state"],
            "compartment": item["compartment"],
            "error_over_tolerance": float(item["absolute_change"] / item["tolerance"]),
            "relative_change": float(item["relative_change"]),
        }
    summary = {
        "schema": "tepc_independent_volume_mesh_convergence_v1",
        "status": status,
        "scope": "selected rho states; fixed CT surfaces/masks; independent Gmsh volumes; identical 5 mm contacts",
        "levels": {"L12": str(baseline), "L09": str(levels["L09"]), "L06": str(levels["L06"])},
        "independent_mesh_builds": mesh_builds,
        "selection_sha256": sha(selection_path),
        "states_per_montage": len(selection["selected"]),
        "rows_compared": len(comparison_frame),
        "criteria": {
            "Z": "1% relative plus 1e-8 ohm",
            "S_and_B": "2% local relative plus 0.5% global scale floor",
            "S_and_B_scale_origin": {
                "basis": "selected L12 rows only",
                "source_level": "L12",
                "source_file": relative(baseline / "analysis/states_rho.csv", baseline),
                "selection": "state_selection.json",
                "rows": int(len(baseline_frame)),
                "states_per_montage": len(selection["selected"]),
                "montages": list(MONTAGES),
                "scales": {
                    f"{metric}_{compartment}": scale[(metric, compartment)]
                    for metric in ("S", "B")
                    for compartment in PARAMS
                },
            },
            "Q": "0.01 absolute",
            "body_volume": "0.1% relative",
            "compartment_volume": "2% relative",
            "ranking": "top montage unchanged across L12/L09/L06",
            "preference_resolution": "fine-mesh score gap exceeds summed L09-to-L06 changes",
        },
        "checks": checks,
        "failed_metric_rows": int((~comparison_frame["passed"]).sum()),
        "unstable_top_rows": int((~top_frame["top_stable"]).sum()),
        "unresolved_top_rows": int((~top_frame["fine_top_gap_resolved"]).sum()),
        "resolved_pair_reversals": int((~pair_frame["acceptable_stable_or_unresolved"]).sum()),
        "failed_volume_rows": int((~volume_frame["passed"]).sum()),
        "worst_metric_cases": worst,
        "interpretation": (
            "Passing supports numerical mesh adequacy only for the tested states, fixed CT geometry, "
            "fixed compartment masks and fixed contact model. It does not constitute physical validation."
        ),
        "physical_validation": False,
        "source_sha256": sha(Path(__file__)),
        "producer_source_sha256": {
            "gmsh_mesh_builder": sha(ROOT / "tools/gmsh_arm_parameter_mesh.py"),
            "mesh_classification": sha(ROOT / "src/trkg4_classify_volume_convergence_mesh.m"),
            "contact_refinement": sha(ROOT / "tools/refine_surface_contact_mesh.py"),
            "model_export": sha(ROOT / "src/trkg4_export_volume_convergence_level.m"),
        },
        "outputs_sha256": {
            name: sha(output / name)
            for name in (
                "level_results.csv",
                "metric_convergence.csv",
                "rankings.csv",
                "top_stability.csv",
                "pair_stability.csv",
                "volume_convergence.csv",
            )
        },
        "input_sha256": {
            "selection": sha(selection_path),
            "baseline_analysis": sha(baseline / "analysis/summary.json"),
            "level09_verification": sha(levels["L09"] / "verification.json"),
            "level06_verification": sha(levels["L06"] / "verification.json"),
        },
    }
    write(output / "summary.json", summary)
    (output / "summary.sha256").write_text(sha(output / "summary.json") + "\n", encoding="ascii")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    command = sub.add_parser("refine-contacts")
    command.add_argument("--baseline", required=True, type=Path)
    command.add_argument("--level", required=True, type=Path)
    command.add_argument("--deps", required=True, type=Path)
    command.add_argument("--target-edge-mm", type=float, default=1.0)
    command.add_argument("--radius-mm", type=float, default=8.0)

    command = sub.add_parser("configure-level")
    command.add_argument("--baseline", required=True, type=Path)
    command.add_argument("--level", required=True, type=Path)
    command.add_argument("--deps", required=True, type=Path)

    command = sub.add_parser("select-states")
    command.add_argument("--baseline", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command.add_argument("--deps", required=True, type=Path)

    command = sub.add_parser("make-studies")
    command.add_argument("--level", required=True, type=Path)
    command.add_argument("--selection", required=True, type=Path)
    command.add_argument("--deps", required=True, type=Path)
    command.add_argument("--threads", type=int, default=4)

    command = sub.add_parser("verify-level")
    command.add_argument("--level", required=True, type=Path)
    command.add_argument("--deps", required=True, type=Path)

    command = sub.add_parser("analyse")
    command.add_argument("--baseline", required=True, type=Path)
    command.add_argument("--level09", required=True, type=Path)
    command.add_argument("--level06", required=True, type=Path)
    command.add_argument("--selection", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command.add_argument("--deps", required=True, type=Path)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    commands = {
        "refine-contacts": refine_contacts,
        "configure-level": configure_level,
        "select-states": select_states,
        "make-studies": make_studies,
        "verify-level": verify_level,
        "analyse": analyse,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()