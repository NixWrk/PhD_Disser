"""Verify the accepted arm-electrode mesh against explicit engineering gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import gmsh
import numpy as np

from check_msh_quality import inspect_mesh


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shape_quality(path: Path) -> dict[str, object]:
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        element_types, element_tags, _ = gmsh.model.mesh.getElements(3)
        tetra_tags: list[int] = []
        for element_type, tags in zip(element_types, element_tags):
            if int(element_type) != 4:
                raise RuntimeError(f"Unsupported 3D element type: {element_type}")
            tetra_tags.extend(np.asarray(tags, dtype=np.int64).tolist())
        values = np.asarray(
            gmsh.model.mesh.getElementQualities(tetra_tags, "minSICN"),
            dtype=float,
        )
        return {
            "minimum": float(values.min()),
            "p1": float(np.percentile(values, 1)),
            "median": float(np.median(values)),
            "count_le_0_001": int(np.count_nonzero(values <= 0.001)),
            "fraction_le_0_001": float(np.mean(values <= 0.001)),
        }
    finally:
        gmsh.finalize()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-mesh", type=Path, required=True)
    parser.add_argument("--surface-qc", type=Path, required=True)
    parser.add_argument("--matlab-summary", type=Path, required=True)
    parser.add_argument("--matlab-patches", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    surface = json.loads(args.surface_qc.read_text(encoding="utf-8"))
    mesh_basic = inspect_mesh(args.mesh, 1e-12)
    candidate_basic = inspect_mesh(args.candidate, 1e-12)
    mesh_shape = shape_quality(args.mesh)
    reference_shape = shape_quality(args.reference_mesh)
    summary = read_csv(args.matlab_summary)
    patches = read_csv(args.matlab_patches)

    candidate_hash = sha256(args.candidate)
    accepted_hash = sha256(args.mesh)
    max_reciprocity = max(float(row["reciprocity_relative_error"]) for row in summary)
    if any(row["electrode_kind"] == "cross_section_plane" for row in patches):
        raise ValueError("Historical four-plane montage is retired; run the corrected arm_electrode_report profile before generating a new acceptance report.")
    if any("contact_geometry" not in row for row in patches):
        raise ValueError("Contact geometry metadata is required for the mixed montage.")
    surface_patches = [row for row in patches if row["contact_geometry"] != "internal_plane"]
    plane_patches = [row for row in patches if row["contact_geometry"] == "internal_plane"]
    if not plane_patches:
        raise ValueError("The corrected four-kind comparison must contain outer plane contacts.")
    mixed_inner = [row for row in surface_patches if row["electrode_kind"] == "outer_planes_inner_rings"]
    surface_area_ratios = np.asarray(
        [float(row["area_ratio"]) for row in surface_patches]
    )
    centroid_offsets = np.asarray(
        [
            abs(float(row["patch_x_centroid_mm"]) - float(row["x_mm"]))
            for row in patches
        ]
    )
    plane_spans_mm = np.asarray(
        [
            float(row["patch_x_max_mm"]) - float(row["patch_x_min_mm"])
            for row in plane_patches
        ]
    )
    plane_projected_area_ratios = np.asarray(
        [float(row["projected_cross_section_area_ratio"]) for row in plane_patches]
    )
    kind_counts = {
        kind: sum(row["electrode_kind"] == kind for row in summary)
        for kind in {row["electrode_kind"] for row in summary}
    }

    checks = {
        "accepted_file_matches_candidate": accepted_hash == candidate_hash,
        "surface_qc_pass": bool(surface["surface_qc_pass"]),
        "zero_surface_self_intersections": (
            int(surface["self_intersecting_faces"]) == 0
        ),
        "surface_volume_change_le_0_5_percent": abs(
            float(surface["relative_volume_change"])
        )
        <= 0.005,
        "mesh_volume_matches_surface_le_0_1_percent": abs(
            mesh_basic["absolute_volume_l"] - float(surface["target_volume_l"])
        )
        / float(surface["target_volume_l"])
        <= 0.001,
        "zero_negative_tetrahedra": mesh_basic["negative_tetrahedra"] == 0,
        "zero_degenerate_tetrahedra": mesh_basic["degenerate_tetrahedra"] == 0,
        "positive_minSICN": mesh_shape["minimum"] > 0,
        "p1_not_worse_than_reference": mesh_shape["p1"] >= reference_shape["p1"],
        "low_quality_fraction_not_worse_than_reference": (
            mesh_shape["fraction_le_0_001"]
            <= reference_shape["fraction_le_0_001"]
        ),
        "sixteen_matlab_scenarios": len(summary) == 16,
        "four_scenarios_per_electrode_kind": kind_counts == {
            "point_disc_5mm": 4,
            "circumferential_ring": 4,
            "wide_cuff_equivalent_area": 4,
            "outer_planes_inner_rings": 4,
        },
        "all_reciprocity_pass": all(row["reciprocity_pass"] == "1" for row in summary),
        "max_reciprocity_relative_error_le_1e_8": max_reciprocity <= 1e-8,
        "all_position_clearance_pass": all(
            row["position_clearance_pass"] == "1" for row in summary
        ),
        "all_geometry_screening_pass": all(
            row["within_screening_factor_of_point_disc"] == "1"
            and row["same_real_sign_as_point_disc"] == "1"
            for row in summary
        ),
        "sixty_four_electrode_patches": len(patches) == 64,
        "fifty_six_surface_patches": len(surface_patches) == 56,
        "eight_outer_plane_patches": len(plane_patches) == 8,
        "planes_are_outer_current_contacts": all(
            row["label"] in {"I_plus", "I_minus"}
            and row["electrode_kind"] == "outer_planes_inner_rings" for row in plane_patches
        ),
        "mixed_inner_contacts_are_5mm_cuffs": len(mixed_inner) == 8 and all(
            row["label"] in {"V_plus", "V_minus"}
            and row["contact_geometry"] == "surface_cuff"
            and float(row["axial_width_mm"]) == 5 for row in mixed_inner
        ),
        "all_patch_clearance_pass": all(row["pass_clearance"] == "1" for row in patches),
        "surface_patch_area_within_5_percent": bool(
            np.all(
                (surface_area_ratios >= 0.95) & (surface_area_ratios <= 1.05)
            )
        ),
        "patch_centroid_offset_le_0_5_mm": bool(np.all(centroid_offsets <= 0.5)),
        "plane_projected_area_within_5_percent": bool(
            np.all(
                (plane_projected_area_ratios >= 0.95)
                & (plane_projected_area_ratios <= 1.05)
            )
        ),
        "plane_mesh_span_le_3_mm": bool(np.all(plane_spans_mm <= 3.0)),
        "all_patches_nonempty": all(
            int(row["face_count"]) > 0 and int(row["node_count"]) > 0
            for row in patches
        ),
        "candidate_basic_qc_matches_accepted": (
            candidate_basic["nodes"] == mesh_basic["nodes"]
            and candidate_basic["tetrahedra"] == mesh_basic["tetrahedra"]
            and candidate_basic["negative_tetrahedra"]
            == mesh_basic["negative_tetrahedra"]
        ),
    }
    accepted = all(checks.values())
    report = {
        "status": (
            "accepted_for_parameterized_arm_electrode_method_development"
            if accepted
            else "rejected"
        ),
        "physical_validation": False,
        "accepted_mesh": str(args.mesh.resolve()),
        "accepted_mesh_sha256": accepted_hash,
        "candidate_mesh": str(args.candidate.resolve()),
        "reference_mesh": str(args.reference_mesh.resolve()),
        "checks": checks,
        "mesh_basic_quality": mesh_basic,
        "mesh_shape_quality": mesh_shape,
        "reference_shape_quality": reference_shape,
        "surface_parameter_band_edge_stats_mm": surface[
            "parameter_band_edge_stats_mm"
        ],
        "matlab": {
            "scenarios": len(summary),
            "patches": len(patches),
            "kind_counts": kind_counts,
            "max_reciprocity_relative_error": max_reciprocity,
            "surface_area_ratio_min": float(surface_area_ratios.min()),
            "surface_area_ratio_max": float(surface_area_ratios.max()),
            "centroid_offset_max_mm": float(centroid_offsets.max()),
            "plane_mesh_span_min_mm": float(plane_spans_mm.min()),
            "plane_mesh_span_max_mm": float(plane_spans_mm.max()),
            "plane_projected_area_ratio_min": float(
                plane_projected_area_ratios.min()
            ),
            "plane_projected_area_ratio_max": float(
                plane_projected_area_ratios.max()
            ),
        },
        "scope_limitation": (
            "Technical acceptance supports the parameterized arm-electrode "
            "FEM tests only; it is not physiological or experimental validation."
        ),
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
