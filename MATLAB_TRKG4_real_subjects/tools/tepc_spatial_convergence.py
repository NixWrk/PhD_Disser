"""Prepare and analyse the first TEPC spatial-convergence screen.

The screen refines only the conforming volume/boundary neighbourhood of all
surface contacts from the accepted 1 mm edge limit to 0.5 mm.  It is therefore
a contact-discretisation screen, not evidence of global volume/interface mesh
convergence and not physical validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
PARAMS = ("soft", "heart", "lung")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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


def configure_dependencies(path: Path) -> None:
    resolved = str(Path(path).resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    tools = str(ROOT / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)


def verified_baseline(base: Path):
    manifest_path = base / "manifest.json"
    require(manifest_path.exists(), "Baseline manifest is missing")
    require(
        sha(manifest_path) == (base / "manifest.sha256").read_text().strip(),
        "Baseline manifest hash mismatch",
    )
    manifest = read(manifest_path)
    mesh = (base / manifest["source"]["mesh"]).resolve()
    require(sha(mesh) == manifest["source"]["mesh_sha256"], "Baseline mesh hash mismatch")
    return manifest, mesh


def refine_command(args) -> None:
    configure_dependencies(args.deps)
    import numpy as np
    from scipy.io import loadmat, savemat
    import refine_surface_contact_mesh as refinement

    refinement.np = np
    baseline = args.baseline.resolve()
    output = args.output.resolve()
    manifest, parent_mesh = verified_baseline(baseline)
    parent_qc = read(baseline / manifest["source"]["refinement_qc"])
    require(
        parent_qc["status"] == "passed_geometry_preserving_local_refinement",
        "Baseline refinement was not accepted",
    )
    require(0 < args.target_edge_mm < parent_qc["target_edge_mm"], "Target must refine the baseline")
    require(args.radius_mm >= parent_qc["contact_neighbourhood_radius_mm"], "Fine neighbourhood cannot be smaller")

    directory = output / "refinement"
    mesh_path = directory / "refined_mesh.mat"
    qc_path = directory / "refinement.json"
    if qc_path.exists() and mesh_path.exists():
        existing = read(qc_path)
        require(existing["refined_mesh_sha256"] == sha(mesh_path), "Existing fine mesh hash mismatch")
        require(math.isclose(existing["target_edge_mm"], args.target_edge_mm), "Existing target differs")
        print(json.dumps({"status": "reused_verified_refinement", "mesh": str(mesh_path)}))
        return
    require(not directory.exists(), "Partial refinement directory exists; preserve it and use a new output")
    directory.mkdir(parents=True)

    data = loadmat(
        parent_mesh,
        variable_names=["nodes_mm", "elems", "boundary", "tissue_id", "group_id"],
    )
    required = {"nodes_mm", "elems", "boundary", "tissue_id", "group_id"}
    require(required <= set(data), "Baseline refined MAT lacks required arrays")
    nodes = np.asarray(data["nodes_mm"], dtype=float)
    elems = np.asarray(data["elems"], dtype=np.int32) - 1
    boundary = np.asarray(data["boundary"], dtype=np.int32) - 1
    tissue = np.asarray(data["tissue_id"], dtype=np.int32).reshape(-1)
    groups = np.asarray(data["group_id"], dtype=np.int32).reshape(-1)
    require(len(elems) == len(tissue) == len(groups), "Element labels are incomplete")
    old_nodes = nodes.copy()
    old_count = len(elems)
    old_volume = refinement.volumes(nodes, elems)
    require((old_volume > 0).all(), "Baseline contains nonpositive tetrahedra")
    before = np.bincount(groups, weights=old_volume, minlength=6)

    centres = []
    for item in manifest["inputs"]:
        source = read(baseline / item["json"])
        require(sha(baseline / item["json"]) == item["sha256"], "Surface JSON hash mismatch")
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
    require((volume > 0).all(), "Fine mesh contains nonpositive tetrahedra")
    require(np.array_equal(nodes[: len(old_nodes)], old_nodes), "Parent vertices changed")
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

    parent_manifest = baseline / "manifest_before_refinement.json"
    original = read(parent_manifest)
    report = {
        "status": "passed_geometry_preserving_local_refinement",
        "scope": "contact_neighbourhood_only_not_global_volume_or_interface_convergence",
        "parent_mesh_sha256": sha(parent_mesh),
        "parent_prepared_sha256": original["source"]["prepared_sha256"],
        "manifest_sha256": sha(parent_manifest),
        "refined_mesh_sha256": sha(mesh_path),
        "target_edge_mm": args.target_edge_mm,
        "contact_neighbourhood_radius_mm": args.radius_mm,
        "rounds": history,
        "counts": {
            "nodes": len(nodes),
            "tetrahedra": len(elems),
            "boundary": len(boundary),
            "added_nodes": len(nodes) - len(old_nodes),
            "added_tetrahedra": len(elems) - old_count,
        },
        "checks": {
            "original_vertices_unchanged": True,
            "all_positive_tetrahedra": True,
            "group_volumes_preserved": True,
            "conforming_boundary_verified": True,
            "contact_neighbourhood_edges_le_1mm": True,
            "contact_neighbourhood_edges_le_target": True,
        },
        "group_volume_before_mm3": before.tolist(),
        "group_volume_after_mm3": after.tolist(),
        "min_tetra_volume_mm3": float(volume.min()),
        "physical_validation": False,
        "spatial_solution_convergence": False,
        "source_sha256": sha(Path(__file__)),
    }
    write(qc_path, report)
    print(json.dumps(report, ensure_ascii=False))


def configure_command(args) -> None:
    configure_dependencies(args.deps)
    from prepare_surface_sensitivity import import_archive, verify_manifest

    baseline = args.baseline.resolve()
    output = args.output.resolve()
    manifest, _ = verified_baseline(baseline)
    if (output / "manifest.json").exists():
        verify_manifest(output)
        print(json.dumps({"status": "reused_verified_manifest", "base": str(output)}))
        return
    refinement_path = output / "refinement/refinement.json"
    mesh_path = output / "refinement/refined_mesh.mat"
    export_path = output / "refinement/export/refinement_export.json"
    prepared_path = output / "refinement/export/prepared_fem.mat"
    qc = read(refinement_path)
    export = read(export_path)
    require(qc["refined_mesh_sha256"] == sha(mesh_path), "Fine mesh/QC mismatch")
    require(export["status"] == "passed" and export["matrix_error"] < 1e-12, "Fine export failed")
    require(export["new_prepared_sha256"] == sha(prepared_path), "Fine prepared FEM hash mismatch")
    require(export["refined_mesh_sha256"] == sha(mesh_path), "Fine export used another mesh")

    acceptance_path = output / "refinement/acceptance.json"
    identity_path = output / "refinement/identity.json"
    acceptance = {
        "status": "accepted_for_contact_discretisation_screen_not_global_solution_convergence",
        "accepted_mesh_sha256": sha(mesh_path),
        "checks": qc["checks"] | {"eidors_volume_contact_matrix_agreement": True},
        "parent_mesh_sha256": qc["parent_mesh_sha256"],
        "refinement_qc_sha256": sha(refinement_path),
        "export_sha256": sha(export_path),
        "physical_validation": False,
        "spatial_solution_convergence": False,
    }
    identity = {
        "prepared_fem_sha256": sha(prepared_path),
        "mesh_sha256": sha(mesh_path),
        "parent_prepared_sha256": qc["parent_prepared_sha256"],
        "export_sha256": sha(export_path),
        "status": "contact_local_refinement_algebraic_check_passed",
    }
    write(acceptance_path, acceptance)
    write(identity_path, identity)

    baseline_plan = read(baseline / "surface_plan.json")
    first = baseline_plan["pilots"][0]
    baseline_study_path = baseline / first["study"]
    template = read(baseline_study_path)
    template_dir = output / "template"
    template_dir.mkdir(parents=True, exist_ok=False)
    template_path = template_dir / "study.json"
    template["source"]["prepared_fem"] = relative(prepared_path, template_dir)
    template["source"]["prepared_sha256"] = sha(prepared_path)
    contact = (baseline_study_path.parent / template["montages"][0]["contact_matrix"]).resolve()
    template["montages"][0]["contact_matrix"] = relative(contact, template_dir)
    write(template_path, template)

    source_surface = (baseline / manifest["source"]["surface"]).resolve()
    import_archive(
        SimpleNamespace(
            archive=baseline / "source_archive.zip",
            template=template_path,
            rho_profile=baseline / "rho_profile.json",
            mesh=mesh_path,
            mesh_acceptance=acceptance_path,
            model_identity=identity_path,
            surface=source_surface,
            output=output,
            prepared_format="si_geometry_si_blocks",
            diameter_mm=manifest["contact"]["diameter_mm"],
            z_contact=manifest["contact"]["z_contact_ohm_m2"],
        )
    )
    print(json.dumps({"status": "configured_for_contact_preparation", "base": str(output)}))


def analyse_command(args) -> None:
    configure_dependencies(args.deps)
    import numpy as np
    import pandas as pd
    import run_surface_sensitivity as runner
    from analyze_resistivity_coverage import add_metrics

    runner.configure(args.deps)
    baseline = args.baseline.resolve()
    fine = args.output.resolve()
    baseline_plan, baseline_profile, baseline_studies = runner.check_inputs(baseline)
    fine_plan, fine_profile, fine_studies = runner.check_inputs(fine)
    require(baseline_profile == fine_profile, "Rho profiles differ between meshes")
    runner.verified_pilot(baseline, baseline_plan, baseline_studies)
    runner.verified_pilot(fine, fine_plan, fine_studies)
    coarse_frame, _ = runner.load_frames(baseline, baseline_plan, baseline_studies, baseline_plan["pilots"])
    fine_frame, _ = runner.load_frames(fine, fine_plan, fine_studies, fine_plan["pilots"])
    coarse_frame = add_metrics(coarse_frame, baseline_profile)
    fine_frame = add_metrics(fine_frame, fine_profile)
    keys = ["montage", "state"]
    merged = coarse_frame.merge(fine_frame, on=keys, suffixes=("_1mm", "_0p5mm"), validate="one_to_one")
    require(len(merged) == 78, "Expected 78 paired pilot states")

    maxima = {}
    for name in ("S", "B"):
        for compartment in PARAMS:
            maxima[(name, compartment)] = float(coarse_frame[f"{name}_{compartment}"].abs().max())
    rows = []
    for _, item in merged.iterrows():
        base_value = float(item["Z_ohm_1mm"])
        fine_value = float(item["Z_ohm_0p5mm"])
        difference = abs(fine_value - base_value)
        tolerance = 1e-8 + 0.01 * abs(base_value)
        rows.append({
            "montage": item.montage,
            "state": item.state,
            "metric": "Z",
            "compartment": "all",
            "value_1mm": base_value,
            "value_0p5mm": fine_value,
            "absolute_change": difference,
            "relative_change": difference / abs(base_value) if base_value else math.nan,
            "tolerance": tolerance,
            "passed": difference <= tolerance,
        })
        for name in ("S", "B", "Q"):
            for compartment in PARAMS:
                base_value = float(item[f"{name}_{compartment}_1mm"])
                fine_value = float(item[f"{name}_{compartment}_0p5mm"])
                difference = abs(fine_value - base_value)
                if name == "Q":
                    tolerance = 0.01
                else:
                    tolerance = 0.02 * abs(base_value) + 0.005 * maxima[(name, compartment)]
                rows.append({
                    "montage": item.montage,
                    "state": item.state,
                    "metric": name,
                    "compartment": compartment,
                    "value_1mm": base_value,
                    "value_0p5mm": fine_value,
                    "absolute_change": difference,
                    "relative_change": difference / abs(base_value) if base_value else math.nan,
                    "tolerance": tolerance,
                    "passed": difference <= tolerance,
                })
    comparison = pd.DataFrame(rows)
    central = merged[merged.state == "pilot_base"]
    ranking_rows = []
    for metric in ("B", "Q"):
        for compartment in PARAMS:
            coarse_order = central.sort_values(f"{metric}_{compartment}_1mm", ascending=False).montage.tolist()
            fine_order = central.sort_values(f"{metric}_{compartment}_0p5mm", ascending=False).montage.tolist()
            ranking_rows.append({
                "metric": metric,
                "compartment": compartment,
                "top_1mm": coarse_order[0],
                "top_0p5mm": fine_order[0],
                "top_stable": coarse_order[0] == fine_order[0],
                "order_1mm": ">".join(coarse_order),
                "order_0p5mm": ">".join(fine_order),
                "complete_order_stable": coarse_order == fine_order,
            })
    rankings = pd.DataFrame(ranking_rows)
    analysis = fine / "spatial_analysis"
    analysis.mkdir(exist_ok=True)
    comparison.to_csv(analysis / "mesh_metric_comparison.csv", index=False)
    rankings.to_csv(analysis / "central_ranking_comparison.csv", index=False)
    summary = {
        "status": "completed_contact_local_two_level_mesh_screen",
        "scope": "contact_neighbourhood_refinement_1mm_to_0p5mm",
        "paired_states": len(merged),
        "metric_checks": len(comparison),
        "failed_metric_checks": int((~comparison.passed).sum()),
        "max_relative_Z_change": float(comparison[comparison.metric == "Z"].relative_change.max()),
        "max_relative_S_change": float(comparison[comparison.metric == "S"].relative_change.replace([np.inf], np.nan).max()),
        "max_absolute_Q_change": float(comparison[comparison.metric == "Q"].absolute_change.max()),
        "stable_top_choices": bool(rankings.top_stable.all()),
        "stable_complete_orders": bool(rankings.complete_order_stable.all()),
        "contact_discretisation_screen_passed": bool(comparison.passed.all() and rankings.top_stable.all()),
        "global_volume_interface_convergence_evaluated": False,
        "full_spatial_convergence_established": False,
        "physical_validation": False,
        "criteria": {
            "Z_relative": 0.01,
            "S_and_B_relative": 0.02,
            "S_and_B_scale_floor_fraction": 0.005,
            "Q_absolute": 0.01,
            "top_choice_must_be_stable": True,
        },
        "outputs_sha256": {
            "mesh_metric_comparison.csv": sha(analysis / "mesh_metric_comparison.csv"),
            "central_ranking_comparison.csv": sha(analysis / "central_ranking_comparison.csv"),
        },
    }
    write(analysis / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("refine", "configure", "analyse"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--baseline", required=True, type=Path)
        sub.add_argument("--output", required=True, type=Path)
        sub.add_argument("--deps", required=True, type=Path)
        if name == "refine":
            sub.add_argument("--target-edge-mm", type=float, default=0.5)
            sub.add_argument("--radius-mm", type=float, default=8.0)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    {"refine": refine_command, "configure": configure_command, "analyse": analyse_command}[args.command](args)


if __name__ == "__main__":
    main()
