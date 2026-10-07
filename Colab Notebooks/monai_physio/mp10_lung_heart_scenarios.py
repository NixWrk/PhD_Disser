"""Compute lung and coupled-heart respiratory scenarios from one segmented CT.

The published lung model is used without retraining. Stage 0.7 is its g070
technical reference, not an inferred breathing state of the patient's CT.
In particular, fitting a deep-inspiration CT to that reference does not yield
a calibrated inspiration-to-expiration forecast. All individual outputs remain
in the caller's local output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import itk
import nibabel as nib
import numpy as np
import pyvista as pv
from scipy.ndimage import map_coordinates
from skimage.measure import marching_cubes

from monai_physio import ProcessTransforms, WorkflowInferMovement, WorkflowInferPhysicsNeMo


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def surface_from_mask(path: Path) -> pv.PolyData:
    image = nib.load(path)
    mask = np.asarray(image.dataobj) > 0
    if not np.any(mask):
        raise ValueError("Heart mask is empty")
    if any(
        np.any(np.take(mask, index, axis=axis))
        for axis in range(3)
        for index in (0, mask.shape[axis] - 1)
    ):
        raise ValueError("Heart mask touches the acquired CT boundary")
    bounds = np.argwhere(mask)
    lo = np.maximum(bounds.min(axis=0) - 2, 0)
    hi = np.minimum(bounds.max(axis=0) + 3, mask.shape)
    cropped = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    vertices, triangles, _, _ = marching_cubes(cropped.astype(np.uint8), 0.5)
    ras = nib.affines.apply_affine(image.affine, vertices + lo)
    lps = ras * np.array([-1.0, -1.0, 1.0])
    faces = np.column_stack((np.full(len(triangles), 3), triangles)).ravel()
    mesh = pv.PolyData(lps.astype(np.float32), faces.astype(np.int64))
    return mesh.clean()


def points_in_field(points_lps: np.ndarray, field: itk.Image) -> np.ndarray:
    vectors = itk.array_from_image(field)
    direction = np.asarray(field.GetDirection())
    affine3 = direction @ np.diag(np.asarray(field.GetSpacing()))
    ijk = np.linalg.solve(affine3, (points_lps - np.asarray(field.GetOrigin())).T)
    coords = [ijk[2], ijk[1], ijk[0]]
    return np.column_stack(
        [map_coordinates(vectors[..., axis], coords, order=1, mode="constant", cval=0.0)
         for axis in range(3)]
    )


def describe_displacement(displacement: np.ndarray) -> dict[str, object]:
    lengths = np.linalg.norm(displacement, axis=1)
    return {
        "centroid_shift_lps_mm": [float(value) for value in displacement.mean(axis=0)],
        "centroid_shift_norm_mm": float(np.linalg.norm(displacement.mean(axis=0))),
        "vertex_displacement_mean_mm": float(lengths.mean()),
        "vertex_displacement_p95_mm": float(np.quantile(lengths, 0.95)),
        "vertex_displacement_max_mm": float(lengths.max()),
        "nonzero_vertex_fraction_gt_0_1mm": float(np.mean(lengths > 0.1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--heart-mask", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--anchor-stage", type=float)
    parser.add_argument("--source-phase", default="unverified")
    parser.add_argument("--allow-unverified-phase", action="store_true",
                        help="Allow an explicitly anchored scenario with unknown acquisition phase; not a phase-verified prediction")
    parser.add_argument("--save-fields", action="store_true")
    parser.add_argument("--stages", nargs="+", type=float, default=[0.0, 0.5, 0.7, 0.9])
    parser.add_argument("--source-landmarks", type=Path)
    parser.add_argument("--target-landmarks", type=Path)
    parser.add_argument("--landmark-grid", type=Path)
    args = parser.parse_args()
    from mp10_respiratory_phases import check_fit_phase, phase_contract
    contract = phase_contract(args.source_phase, args.anchor_stage, args.stages,
                              allow_unverified=args.allow_unverified_phase)
    if any((args.source_landmarks, args.target_landmarks, args.landmark_grid)) and not all(
        (args.source_landmarks, args.target_landmarks, args.landmark_grid)
    ):
        raise ValueError("Both landmark files and their NIfTI grid must be supplied together")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    anchor = contract["source_stage"]
    stages = args.stages
    if any(not 0.0 <= stage < 1.0 for stage in stages) or len(stages) != len(set(stages)):
        raise ValueError("Stages must be unique fractions in [0, 1)")
    if anchor not in stages:
        raise ValueError("The source stage must be one of the evaluated stages")
    sigma_mm = 15.0
    offset_mm = 3.0
    transition_mm = 5.0
    falloff_mm = 15.0

    fit_report = json.loads((args.fit_dir / "fit_report.json").read_text(encoding="utf-8"))
    check_fit_phase(fit_report, args.source_phase)
    fitted_path = args.fit_dir / "lung_fitted_reference.vtp"
    coeff_path = args.fit_dir / "lung_pca_coefficients.json"
    reference_image = itk.imread(str(args.fit_dir / "ct_fit_grid.nii.gz"))
    labelmap = itk.imread(str(args.fit_dir / "lung_labels_fit_grid.nii.gz"))
    lung_mask = itk.image_from_array(
        (itk.array_from_image(labelmap) > 0).astype(np.uint8)
    )
    lung_mask.CopyInformation(labelmap)
    fitted_lung = pv.read(str(fitted_path))
    heart_source = surface_from_mask(args.heart_mask)
    heart_source_path = args.output_dir / "heart_source.vtp"
    heart_source.save(str(heart_source_path))

    network = WorkflowInferPhysicsNeMo(model_directory=args.model_dir, epoch=None)
    infer = WorkflowInferMovement(network)
    tools = ProcessTransforms()
    coeffs = np.asarray(json.loads(coeff_path.read_text(encoding="utf-8")), dtype=float)
    anchor_lung = network.predict(coeffs, anchor)
    print(f"Anchor stage {anchor:.2f}; heart vertices {heart_source.n_points}", flush=True)

    results: list[dict[str, object]] = []
    anchor_heart_field: itk.Image | None = None
    for stage in stages:
        print(f"Calculating respiratory stage {stage:.2f}", flush=True)
        prediction = network.predict(coeffs, stage)
        lung_shift = prediction - anchor_lung
        lung = fitted_lung.copy(deep=True)
        lung.points = (np.asarray(fitted_lung.points) + lung_shift).astype(np.float32)
        lung_path = args.output_dir / f"lung_stage_{stage:.2f}.vtp"
        lung.save(str(lung_path))

        raw = infer.create_deformation_field(
            shape_parameters=coeff_path,
            stage=stage,
            reference_image=reference_image,
            fitted_reference_mesh=fitted_path,
            direction="forward",
        )
        smooth = tools.smooth_deformation_field_transform(
            raw["deformation_field"], sigma_mm, raw["weight_image"]
        )
        normals = tools.smooth_deformation_field_transform(
            raw["normal_image"], sigma_mm, raw["weight_image"]
        )
        restricted = tools.restrict_deformation_field_to_normal_falloff_outside_mask(
            smooth.GetDisplacementField(), normals.GetDisplacementField(),
            lung_mask, offset_mm, transition_mm, falloff_mm,
        )
        field = restricted.GetDisplacementField()
        if args.save_fields:
            serializable = itk.image_from_array(
                itk.array_from_image(field).astype(np.float32), is_vector=True
            )
            serializable.CopyInformation(field)
            itk.imwrite(
                serializable, str(args.output_dir / f"field_stage_{stage:.2f}.mha"),
                compression=True,
            )
        if stage == anchor:
            anchor_heart_field = field
        results.append({
            "stage": stage,
            "lung_shift": lung_shift,
            "lung_path": lung_path,
            "heart_field": field,
        })

    assert anchor_heart_field is not None
    baseline_heart_displacement = points_in_field(heart_source.points, anchor_heart_field)
    landmark_points_lps = None
    landmark_targets_lps = None
    baseline_landmark_displacement = None
    if args.source_landmarks is not None:
        grid = nib.load(args.landmark_grid)
        source_voxels = np.atleast_2d(np.loadtxt(args.source_landmarks, delimiter=","))
        target_voxels = np.atleast_2d(np.loadtxt(args.target_landmarks, delimiter=","))
        if source_voxels.shape != target_voxels.shape or source_voxels.shape[1] != 3:
            raise ValueError("Landmark pairs must have matching Nx3 voxel coordinates")
        landmark_points_lps = nib.affines.apply_affine(grid.affine, source_voxels) * [-1, -1, 1]
        landmark_targets_lps = nib.affines.apply_affine(grid.affine, target_voxels) * [-1, -1, 1]
        baseline_landmark_displacement = points_in_field(landmark_points_lps, anchor_heart_field)
    summary: list[dict[str, object]] = []
    for item in results:
        stage = float(item["stage"])
        heart_shift = points_in_field(heart_source.points, item["heart_field"]) - baseline_heart_displacement
        if stage == anchor:
            heart_shift[:] = 0.0
        heart = heart_source.copy(deep=True)
        heart.points = (np.asarray(heart_source.points) + heart_shift).astype(np.float32)
        heart_path = args.output_dir / f"heart_stage_{stage:.2f}.vtp"
        heart.save(str(heart_path))
        record = {
            "stage": stage,
            "lung": describe_displacement(item["lung_shift"]),
            "heart": describe_displacement(heart_shift),
            "lung_surface_open_edges": int(pv.read(str(item["lung_path"])).n_open_edges),
            "heart_surface_volume_ml": float(heart.volume / 1000.0),
            "lung_mesh_sha256": sha256(item["lung_path"]),
            "heart_mesh_sha256": sha256(heart_path),
        }
        if landmark_points_lps is not None:
            assert landmark_targets_lps is not None and baseline_landmark_displacement is not None
            landmark_shift = points_in_field(landmark_points_lps, item["heart_field"]) - baseline_landmark_displacement
            baseline_error = np.linalg.norm(landmark_points_lps - landmark_targets_lps, axis=1)
            predicted_error = np.linalg.norm(landmark_points_lps + landmark_shift - landmark_targets_lps, axis=1)
            record["landmarks"] = {
                "count": int(len(baseline_error)),
                "static_tre_mean_mm": float(baseline_error.mean()),
                "static_tre_median_mm": float(np.median(baseline_error)),
                "predicted_tre_mean_mm": float(predicted_error.mean()),
                "predicted_tre_median_mm": float(np.median(predicted_error)),
                "predicted_tre_p95_mm": float(np.quantile(predicted_error, 0.95)),
                "mean_predicted_displacement_mm": float(np.linalg.norm(landmark_shift, axis=1).mean()),
            }
        summary.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    report = {
        "schema": "mp10_lung_heart_scenarios_v1",
        "subject_alias": args.alias,
        "status": "model_scenarios_not_validated_breathing_predictions",
        "source_fit_report_sha256": sha256(args.fit_dir / "fit_report.json"),
        "source_heart_mask_sha256": sha256(args.heart_mask),
        "heart_source_surface_sha256": sha256(heart_source_path),
        "source_landmarks_sha256": sha256(args.source_landmarks) if args.source_landmarks else None,
        "target_landmarks_sha256": sha256(args.target_landmarks) if args.target_landmarks else None,
        "model_checkpoint_sha256": sha256(args.model_dir / "mgn_stage_model.pt"),
        "fit_status": fit_report["status"],
        "stage_anchor": anchor,
        "source_phase": args.source_phase,
        "phase_contract": contract,
        "source_phase_binding": (
            "source phase recorded with CT hash in fit report"
            if fit_report.get("source_phase") else "legacy fit: verify phase using input CT manifest and hash"
        ),
        "saved_deformation_fields": args.save_fields,
        "stage_anchor_assumption": (
            "g070 is the published model reference; supplied CT phase must be checked"
            if anchor == 0.7 else
            "non-reference CT phase fitted to a g070-trained shape model; out-of-protocol application"
        ),
        "respiratory_sigma_mm": sigma_mm,
        "direction_offset_mm": offset_mm,
        "direction_transition_mm": transition_mm,
        "falloff_distance_mm": falloff_mm,
        "field_grid_shape_xyz": list(itk.size(reference_image)),
        "field_grid_spacing_mm": [float(x) for x in reference_image.GetSpacing()],
        "cardiac_stage": "fixed; no cardiac network added",
        "stages": summary,
        "validation": "This report contains model scenarios only; evaluate target CT in a separate held-out comparison",
    }
    (args.output_dir / "scenario_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
