"""Read-only Nix registration -> existing FEM contract audit.

Reads aggregate QC and an existing exploratory FEM plan. It never loads or
exports medical arrays and never changes mesh, materials, or FEM inputs.
"""

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_FEM = REPO / "MATLAB_TRKG4_real_subjects" / "output" / "exploratory" / "nix_cardiac_cycle_20260916"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(fem_root):
    coordinate_file = fem_root / "coordinate_audit.json"
    geometry_file = fem_root / "materials" / "geometry.json"
    plan_file = fem_root / "electrical" / "plan.json"
    coordinate, geometry, plan = map(read, (coordinate_file, geometry_file, plan_file))
    candidates = {
        "phase_01": read(HERE / "mp03_nix_fullres_00_01_qc.json"),
        "phase_07": read(HERE / "mp03_nix_fullres_00_07_smooth_qc.json"),
    }
    states = {state["id"]: state for state in plan["states"]}
    same_mesh = (geometry["source_prepared_sha256"] == plan["source_prepared_sha256"]
                 and geometry["node_count"] == plan["node_count"]
                 and geometry["element_count"] == plan["element_count"])
    same_background = geometry["background_mat_sha256"] == plan["background_mat_sha256"]
    plan_geometry_hash_ok = plan["materials_geometry"]["sha256"] == sha256(geometry_file)
    mask_hashes_match = all(
        qc["input_sha256"]["reference_mask"] == states["phase_00"]["mask_sha256"]
        and qc["input_sha256"]["target_mask"] == states[phase]["mask_sha256"]
        for phase, qc in candidates.items()
    )
    same_frame = coordinate["same_dicom_patient_frame"]
    axis_map_agrees = coordinate["transform_ras_mm_to_fem_mm"] == geometry["ras_mm_to_fem_mm"]
    fields = {
        phase: {
            "field_sha256": qc["field_qc"]["local_field_sha256"],
            "local_field_exists": (HERE / qc["field_qc"]["local_field_relative_path"]).is_file(),
            "local_field_hash_matches_qc": (
                sha256(HERE / qc["field_qc"]["local_field_relative_path"])
                == qc["field_qc"]["local_field_sha256"]
                if (HERE / qc["field_qc"]["local_field_relative_path"]).is_file() else False
            ),
            "minimum_sampled_pullback_jacobian": qc["field_qc"]["jacobian_of_pullback"]["minimum"],
            "nonpositive_sampled_jacobian_fraction": qc["field_qc"]["jacobian_of_pullback"]["nonpositive_fraction"],
            "touches_source_fov_boundary": states[phase]["touches_source_boundary"],
            "mask_hash_matches_fem_state": qc["input_sha256"]["target_mask"] == states[phase]["mask_sha256"],
        }
        for phase, qc in candidates.items()
    }
    out = {
        "schema": "mp02_nix_fem_preflight_v2",
        "subject_alias": "nix",
        "source_phase": "phase_00",
        "target_phases": ["phase_01", "phase_07"],
        "source_plan_sha256": sha256(plan_file),
        "source_coordinate_audit_sha256": sha256(coordinate_file),
        "existing_fem": {
            "scientific_status": plan["scientific_status"],
            "node_count": plan["node_count"],
            "element_count": plan["element_count"],
            "required_solutions": plan["required_solutions"],
            "frequency_hz": plan["frequency_hz"],
            "model": plan["model"],
            "contact_diameter_mm": plan["contact_parameters"]["diameter_mm"],
            "prepared_mesh_sha256": plan["source_prepared_sha256"],
            "background_material_sha256": plan["background_mat_sha256"],
            "same_mesh_between_plan_and_materials": same_mesh,
            "same_background_between_plan_and_materials": same_background,
            "plan_material_geometry_sha256_matches_file": plan_geometry_hash_ok,
            "automatic_mask_sha256_matches_registration_inputs": mask_hashes_match,
        },
        "coordinate_audit": {
            "same_static_dynamic_dicom_patient_frame": same_frame,
            "axis_map_agrees_with_materials": axis_map_agrees,
            "body_distance_p50_p90_p95_mm": coordinate["body"]["distance_p50_p90_p95_mm"],
            "bones_distance_p50_p90_p95_mm": coordinate["bones"]["distance_p50_p90_p95_mm"],
            "independent_registration_error_bound_mm": None,
        },
        "candidate_fields": fields,
        "upper_fov_contact": {
            "observed": True,
            "interpretation": "The 4D CT covers less anatomy than the static CT (author clarification, 30.09.2026)",
            "independent_blocker_for_comparison": False,
        },
        "independent_same_phase_manual_heart_reference_available": False,
        "anatomically_accepted_geometry": False,
        "transfer_to_new_fem_comparison": "blocked",
        "new_fem_z_or_delta_z_computed": False,
        "reasons": [
            "No accepted independent manual anatomy for the dynamic target phases",
            "CT-to-FEM coordinate map has no independent registration error bound",
            "Existing FEM states use separate observed automatic masks, not the new transported surface",
        ],
    }
    if not all((same_mesh, same_background, plan_geometry_hash_ok, mask_hashes_match,
                same_frame, axis_map_agrees)):
        out["reasons"].append("Existing FEM provenance or coordinate consistency check failed")
    if any(item["nonpositive_sampled_jacobian_fraction"] > 0 for item in fields.values()):
        out["reasons"].append("A candidate field has nonpositive sampled Jacobian")
    if any(not item["local_field_hash_matches_qc"] for item in fields.values()):
        out["reasons"].append("A local candidate field is absent or its hash differs from QC")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fem-root", type=Path, default=DEFAULT_FEM)
    parser.add_argument("--output", type=Path, default=HERE / "mp02_nix_fem_preflight.json")
    args = parser.parse_args()
    result = audit(args.fem_root)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "transfer": result["transfer_to_new_fem_comparison"],
                      "reasons": result["reasons"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
