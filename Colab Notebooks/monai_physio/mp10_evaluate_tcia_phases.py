"""Evaluate a phase-labelled TCIA 4D-Lung case without using target phases to fit.

The fitted model and neural prediction enter from a separate run.  Each target
phase contributes only its independently computed TotalSegmentator masks to
the final comparison.  All coordinates are converted to physical LPS mm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import itk
import numpy as np
import pyvista as pv
from scipy.ndimage import center_of_mass, map_coordinates
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes


LOBES = (
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def mask_surface(seg_dir: Path, names: tuple[str, ...], output: Path) -> tuple[np.ndarray, dict]:
    first = nib.load(seg_dir / f"{names[0]}.nii.gz")
    mask = np.zeros(first.shape, dtype=bool)
    hashes = {}
    for name in names:
        path = seg_dir / f"{name}.nii.gz"
        image = nib.load(path)
        if image.shape != first.shape or not np.allclose(image.affine, first.affine, atol=1e-4):
            raise ValueError(f"Incompatible segmentation grids in {seg_dir}")
        mask |= np.asarray(image.dataobj) > 0
        hashes[name] = sha256(path)
    if not mask.any():
        raise ValueError(f"Empty mask in {seg_dir}")
    touches = any(np.any(np.take(mask, i, axis=a)) for a in range(3)
                  for i in (0, mask.shape[a] - 1))
    bounds = np.argwhere(mask)
    lo = np.maximum(bounds.min(axis=0) - 2, 0)
    hi = np.minimum(bounds.max(axis=0) + 3, mask.shape)
    cropped = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    vertices, triangles, _, _ = marching_cubes(cropped.astype(np.uint8), 0.5)
    ras = nib.affines.apply_affine(first.affine, vertices + lo)
    lps = ras * [-1.0, -1.0, 1.0]
    faces = np.column_stack((np.full(len(triangles), 3), triangles)).ravel()
    output.parent.mkdir(parents=True, exist_ok=True)
    pv.PolyData(lps.astype(np.float32), faces.astype(np.int64)).save(str(output))
    centroid = nib.affines.apply_affine(first.affine, center_of_mass(mask)) * [-1.0, -1.0, 1.0]
    meta = {
        "volume_ml": float(mask.sum() * abs(np.linalg.det(first.affine[:3, :3])) / 1000),
        "centroid_lps_mm": [float(x) for x in centroid],
        "touches_ct_boundary": bool(touches),
        "mask_sha256": hashes,
        "mask_surface_sha256": sha256(output),
    }
    return lps, meta


def surface_error(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    # Fixed spatial sampling only limits runtime; it is identical for static
    # and predicted model surfaces.  CT mask surfaces have their own grid.
    a = np.asarray(a)[::4]
    b = np.asarray(b)[::4]
    d_ab = cKDTree(b).query(a, workers=-1)[0]
    d_ba = cKDTree(a).query(b, workers=-1)[0]
    all_distances = np.concatenate([d_ab, d_ba])
    return {
        "symmetric_mean_mm": float(all_distances.mean()),
        "symmetric_rms_mm": float(np.sqrt(np.mean(all_distances**2))),
        "symmetric_p95_mm": float(np.quantile(all_distances, 0.95)),
        "predicted_to_observed_mean_mm": float(d_ab.mean()),
        "observed_to_predicted_mean_mm": float(d_ba.mean()),
    }


def sample_field(points_lps: np.ndarray, field: itk.Image) -> tuple[np.ndarray, float]:
    vectors = itk.array_from_image(field)
    direction = np.asarray(field.GetDirection())
    axes = direction @ np.diag(np.asarray(field.GetSpacing()))
    ijk = np.linalg.solve(axes, (points_lps - np.asarray(field.GetOrigin())).T)
    size = np.asarray(field.GetLargestPossibleRegion().GetSize())
    inside = np.all((ijk >= 0) & (ijk <= size[:, None] - 1), axis=0)
    coords = [ijk[2], ijk[1], ijk[0]]
    displacement = np.column_stack([
        map_coordinates(vectors[..., axis], coords, order=1, mode="constant", cval=0.0)
        for axis in range(3)
    ])
    return displacement, float(np.mean(inside))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--seg-root", type=Path, required=True)
    parser.add_argument("--scenario-dir", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phases", nargs="+", default=["g000", "g050", "g070", "g090"])
    parser.add_argument("--source-phase", default="g070")
    args = parser.parse_args()
    source_phase = args.source_phase
    if source_phase not in args.phases:
        raise ValueError("The source phase must be included")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_ct_hashes = {}
    observed_lung = {}
    observed_heart = {}
    lung_meta = {}
    heart_meta = {}
    for phase in args.phases:
        ct_path = args.case_dir / f"100_HM10395_{phase}.nii.gz"
        input_ct_hashes[phase] = sha256(ct_path)
        seg = args.seg_root / phase
        observed_lung[phase], lung_meta[phase] = mask_surface(
            seg, LOBES, args.output_dir / f"observed_lung_{phase}.vtp"
        )
        observed_heart[phase], heart_meta[phase] = mask_surface(
            seg, ("heart",), args.output_dir / f"observed_heart_{phase}.vtp"
        )
        if lung_meta[phase]["touches_ct_boundary"]:
            raise ValueError(f"Lung reaches acquired CT boundary in {phase}")
    source_fit = pv.read(str(args.fit_dir / "lung_fitted_reference.vtp")).points
    source_stage = int(source_phase[1:]) / 100
    source_ct_surface = pv.read(str(args.output_dir / f"observed_lung_{source_phase}.vtp"))
    anchor_field_path = args.scenario_dir / f"field_stage_{source_stage:.2f}.mha"
    have_fields = anchor_field_path.is_file()
    anchor_lung_displacement = None
    if have_fields:
        anchor_lung_displacement, anchor_coverage = sample_field(
            source_ct_surface.points, itk.imread(str(anchor_field_path))
        )
        if anchor_coverage < 0.99:
            raise ValueError(f"Reference field covers only {anchor_coverage:.1%} of source lung surface")
    source_heart_mesh = pv.read(str(args.scenario_dir / f"heart_stage_{source_stage:.2f}.vtp"))
    rows = []
    for phase in args.phases:
        stage = int(phase[1:]) / 100
        predicted_lung = pv.read(str(args.scenario_dir / f"lung_stage_{stage:.2f}.vtp"))
        predicted_heart = pv.read(str(args.scenario_dir / f"heart_stage_{stage:.2f}.vtp"))
        target_lung = observed_lung[phase]
        observed_lung_shift = np.array(lung_meta[phase]["centroid_lps_mm"]) - np.array(
            lung_meta[source_phase]["centroid_lps_mm"]
        )
        predicted_lung_shift = predicted_lung.points.mean(axis=0) - source_fit.mean(axis=0)
        observed_heart_shift = np.array(heart_meta[phase]["centroid_lps_mm"]) - np.array(
            heart_meta[source_phase]["centroid_lps_mm"]
        )
        predicted_heart_shift = predicted_heart.points.mean(axis=0) - source_heart_mesh.points.mean(axis=0)
        row = {
            "phase": phase,
            "model_stage": stage,
            "lung_volume_ml": lung_meta[phase]["volume_ml"],
            "lung_volume_change_from_source_ml": lung_meta[phase]["volume_ml"] - lung_meta[source_phase]["volume_ml"],
            "lung_fit_source_vs_observed": surface_error(source_fit, target_lung),
            "lung_neural_prediction_vs_observed": surface_error(predicted_lung.points, target_lung),
            "lung_source_mask_copy_vs_observed": surface_error(observed_lung[source_phase], target_lung),
            "lung_predicted_vertex_displacement_mean_mm": float(np.linalg.norm(predicted_lung.points - source_fit, axis=1).mean()),
            "lung_observed_centroid_shift_lps_mm": [float(x) for x in observed_lung_shift],
            "lung_predicted_centroid_shift_lps_mm": [float(x) for x in predicted_lung_shift],
            "lung_static_centroid_error_mm": float(np.linalg.norm(observed_lung_shift)),
            "lung_predicted_centroid_error_mm": float(np.linalg.norm(predicted_lung_shift - observed_lung_shift)),
            "heart_volume_ml": heart_meta[phase]["volume_ml"],
            "heart_predicted_volume_ml": (
                float(predicted_heart.volume / 1000) if predicted_heart.n_open_edges == 0 else None
            ),
            "heart_predicted_surface_open_edges": int(predicted_heart.n_open_edges),
            "heart_surface_static_vs_observed": surface_error(
                source_heart_mesh.points, observed_heart[phase]
            ),
            "heart_surface_predicted_vs_observed": surface_error(
                predicted_heart.points, observed_heart[phase]
            ),
            "heart_observed_centroid_shift_lps_mm": [float(x) for x in observed_heart_shift],
            "heart_predicted_centroid_shift_lps_mm": [float(x) for x in predicted_heart_shift],
            "heart_static_centroid_error_mm": float(np.linalg.norm(observed_heart_shift)),
            "heart_predicted_centroid_error_mm": float(np.linalg.norm(predicted_heart_shift - observed_heart_shift)),
            "heart_touches_ct_boundary": heart_meta[phase]["touches_ct_boundary"],
        }
        if have_fields:
            field_path = args.scenario_dir / f"field_stage_{stage:.2f}.mha"
            if not field_path.is_file():
                raise FileNotFoundError(field_path)
            displacement, coverage = sample_field(source_ct_surface.points, itk.imread(str(field_path)))
            if coverage < 0.99:
                raise ValueError(f"Target field covers only {coverage:.1%} of source lung surface")
            warped = source_ct_surface.copy(deep=True)
            warped.points = (source_ct_surface.points + displacement - anchor_lung_displacement).astype(np.float32)
            warped_path = args.output_dir / f"predicted_ct_lung_{phase}.vtp"
            warped.save(str(warped_path))
            row["lung_warped_ct_surface_vs_observed"] = surface_error(warped.points, target_lung)
            row["lung_field_coverage_fraction"] = coverage
            row["lung_warped_ct_surface_open_edges"] = int(warped.n_open_edges)
            row["lung_predicted_volume_ml"] = (
                float(warped.volume / 1000) if warped.n_open_edges == 0 else None
            )
            row["lung_predicted_volume_error_ml"] = (
                row["lung_predicted_volume_ml"] - lung_meta[phase]["volume_ml"]
                if row["lung_predicted_volume_ml"] is not None else None
            )
            row["warped_ct_lung_surface_sha256"] = sha256(warped_path)
        rows.append(row)
    report = {
        "schema": "mp10_tcia_heldout_phase_evaluation_v1",
        "status": "phase_labelled_external_heldout_case_automatic_mask_validation",
        "case_id": "100_HM10395",
        "source_phase": source_phase,
        "input_to_network": f"{source_phase} CT via TotalSegmentator lobe masks, affine ICP and six PCA coefficients",
        "target_use": "target CT and masks only for evaluation; no target fitting or phase selection",
        "model_stage_rule": "gPPP maps to stage PPP/100 from TCIA filenames; not inferred from lung volumes",
        "target_segmentation": "TotalSegmentator v2 automatic; no accepted manual contour",
        "actual_source_ct_surface_warped_by_network_field": have_fields,
        "model_checkpoint_sha256": sha256(args.model_dir / "mgn_stage_model.pt"),
        "fit_report_sha256": sha256(args.fit_dir / "fit_report.json"),
        "scenario_report_sha256": sha256(args.scenario_dir / "scenario_report.json"),
        "ct_sha256": input_ct_hashes,
        "lung_masks": lung_meta,
        "heart_masks": heart_meta,
        "metrics": rows,
        "limitations": [
            "Tutorial 9 designates this case as held out for the motion network; the packaged checkpoint training manifest was not independently audited.",
            "Tutorial 6 builds the PCA shape model from TCIA g070 cases and does not exclude case 100; this is not a full-pipeline leave-one-case-out test.",
            "Surface distances include the CT-to-shape-model fitting error and automatic mask error.",
            "Heart displacement is extrapolated from the lung field, not learned from heart CT.",
            "The fitted model lung mesh is open; predicted volume is reported only for the watertight source-CT surface warped by the network field.",
            *( ["Non-g070 source is outside the published fitting protocol and is a direct inspiration-input stress test."]
               if source_phase != "g070" else [] ),
        ],
    }
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    for row in rows:
        print(json.dumps({
            "phase": row["phase"],
            "volume_ml": row["lung_volume_ml"],
            "static_mask_surface_mean_mm": row["lung_source_mask_copy_vs_observed"]["symmetric_mean_mm"],
            "predicted_surface_mean_mm": row["lung_neural_prediction_vs_observed"]["symmetric_mean_mm"],
            "warped_ct_surface_mean_mm": (
                row["lung_warped_ct_surface_vs_observed"]["symmetric_mean_mm"] if have_fields else None
            ),
            "heart_static_centroid_error_mm": row["heart_static_centroid_error_mm"],
            "heart_predicted_centroid_error_mm": row["heart_predicted_centroid_error_mm"],
        }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
