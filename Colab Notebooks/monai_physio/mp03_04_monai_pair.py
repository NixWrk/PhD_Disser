"""One-pair-at-a-time, local-only MONAI Physio Greedy registration pilot.

The fixed image is the selected target; the moving image is the reference.
Medical arrays and dense fields are written only under an ignored runs/ path.
The JSON output contains aggregate QC, never raw images or absolute paths.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import itk
import monai_physio
import nibabel as nib
import numpy as np
import picsl_greedy
import scipy
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage.measure import marching_cubes

from mp03_04_pilot import bounds, checked_image, comparison, digest, surface, touches_full_fov


F = np.diag([-1.0, -1.0, 1.0])  # NIfTI RAS -> ITK LPS


def make_itk(array_xyz: np.ndarray, affine_ras: np.ndarray) -> itk.Image:
    image = itk.image_from_array(np.ascontiguousarray(array_xyz.transpose(2, 1, 0)))
    axes_lps = F @ affine_ras[:3, :3]
    spacing = np.linalg.norm(axes_lps, axis=0)
    image.SetSpacing(tuple(float(x) for x in spacing))
    image.SetOrigin(tuple(float(x) for x in F @ affine_ras[:3, 3]))
    image.SetDirection(itk.GetMatrixFromArray(axes_lps / spacing))
    return image


def affine_for_crop(source: np.ndarray, start: np.ndarray, stride: np.ndarray) -> np.ndarray:
    affine = source.copy()
    affine[:3, 3] += source[:3, :3] @ start
    affine[:3, :3] = source[:3, :3] @ np.diag(stride)
    return affine


def mesh_distances(reference: np.ndarray, target: np.ndarray, affine_ras: np.ndarray,
                   rigid_shift_ijk: np.ndarray, moving_to_fixed: itk.Transform) -> dict:
    vertices, faces, _, _ = marching_cubes(reference.astype(np.uint8), 0.5)
    target_distance = ndi.distance_transform_edt(~surface(target),
        sampling=np.linalg.norm(affine_ras[:3, :3], axis=0))
    stationary = ndi.map_coordinates(target_distance, vertices.T, order=1, mode="constant", cval=np.nan)
    rigid = ndi.map_coordinates(target_distance, (vertices + rigid_shift_ijk).T,
                                order=1, mode="constant", cval=np.nan)
    points_lps = (F @ (affine_ras[:3, :3] @ vertices.T + affine_ras[:3, 3:4])).T
    mapped_lps = np.asarray([moving_to_fixed.TransformPoint(tuple(p)) for p in points_lps])
    mapped_ras = (F @ mapped_lps.T).T
    mapped_index = np.linalg.solve(affine_ras[:3, :3],
                                   (mapped_ras - affine_ras[:3, 3]).T).T
    deformable = ndi.map_coordinates(target_distance, mapped_index.T,
                                     order=1, mode="constant", cval=np.nan)
    return {"vertex_count": int(len(vertices)), "triangle_count": int(len(faces)),
            "same_vertices_and_faces_for_all_three_candidates": True,
            "mean_distance_mm": {"stationary": float(np.nanmean(stationary)),
                                 "rigid": float(np.nanmean(rigid)),
                                 "greedy": float(np.nanmean(deformable))},
            "greedy_p95_distance_mm": float(np.nanpercentile(deformable, 95)),
            "greedy_vertices_outside_crop_fraction": float(np.mean(~np.isfinite(deformable)))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase-dir", type=Path, required=True)
    p.add_argument("--products-dir", type=Path, required=True)
    p.add_argument("--reference-phase", default="phase_00")
    p.add_argument("--target-phase", default="phase_01")
    p.add_argument("--conversion-manifest", type=Path, required=True)
    p.add_argument("--rigid-qc", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--field-output", type=Path, required=True)
    p.add_argument("--stride", type=int, nargs=3, default=(3, 3, 2))
    p.add_argument("--iterations", type=int, nargs=3, default=(12, 6, 3))
    p.add_argument("--deformable-smoothing", default="2.0vox 0.5vox",
                   help="Greedy warp and gradient smoothing, e.g. '3.0vox 1.0vox'")
    p.add_argument("--crop-source-only", action="store_true",
                   help="Define registration ROI from the reference mask only; target mask is evaluation-only")
    p.add_argument("--margin-mm", type=float, default=20.0)
    args = p.parse_args()
    stride = np.asarray(args.stride, dtype=int)
    if np.any(stride < 1) or np.any(np.asarray(args.iterations) < 1):
        p.error("stride and iterations must be positive")
    if not np.isfinite(args.margin_mm) or args.margin_mm <= 0:
        p.error("margin-mm must be positive")
    if args.reference_phase == args.target_phase:
        p.error("reference and target must differ")
    ref_path = args.phase_dir / f"{args.reference_phase}.nii.gz"
    target_path = args.phase_dir / f"{args.target_phase}.nii.gz"
    mask_paths = [args.products_dir / phase / "whole_heart.nii.gz"
                  for phase in (args.reference_phase, args.target_phase)]
    ref_ct = checked_image(ref_path)
    target_ct = checked_image(target_path, ref_ct)
    masks = [checked_image(path, ref_ct) for path in mask_paths]
    manifest = json.loads(args.conversion_manifest.read_text(encoding="utf-8"))
    entries = {entry["nifti"]: entry for entry in manifest["phases"]}
    spacing = np.linalg.norm(ref_ct.affine[:3, :3], axis=0)
    for name in (ref_path.name, target_path.name):
        entry = entries[name]
        if entry["affine_coordinate_system"] != "NIfTI RAS+" or not np.allclose(
            spacing, entry["voxel_spacing_mm"], atol=1e-3):
            raise ValueError("conversion manifest does not establish RAS/mm geometry")
    prior = json.loads(args.rigid_qc.read_text(encoding="utf-8"))
    if prior["reference"] != args.reference_phase or prior["target"] != args.target_phase:
        raise ValueError("rigid control phase IDs differ")
    if prior["input_sha256"] != {
        "reference_ct": digest(ref_path), "target_ct": digest(target_path),
        "reference_mask": digest(mask_paths[0]), "target_mask": digest(mask_paths[1])}:
        raise ValueError("rigid control input hashes differ from current images")
    full_masks = [np.asarray(image.dataobj) > 0 for image in masks]
    boxes = [bounds(mask) for mask in full_masks]
    margin = np.ceil(args.margin_mm / spacing).astype(int)
    if args.crop_source_only:
        lo = np.maximum(0, boxes[0][0] - margin)
        hi = np.minimum(ref_ct.shape, boxes[0][1] + margin)
    else:
        lo = np.maximum(0, np.minimum(boxes[0][0], boxes[1][0]) - margin)
        hi = np.minimum(ref_ct.shape, np.maximum(boxes[0][1], boxes[1][1]) + margin)
    if np.any(boxes[1][0] < lo) or np.any(boxes[1][1] > hi):
        raise ValueError("target mask exceeds registration crop; source-only ROI is insufficient")
    sl = tuple(slice(int(x), int(y), int(z)) for x, y, z in zip(lo, hi, stride))
    coarse_affine = affine_for_crop(ref_ct.affine, lo, stride)
    moving_mask = np.asarray(full_masks[0][sl], dtype=np.uint8)
    fixed_mask = np.asarray(full_masks[1][sl], dtype=np.uint8)
    del full_masks
    # Every array is sampled at the same physical voxel centres; coarse-grid
    # evaluation of the three methods avoids comparing different resolutions.
    moving_ct = np.asarray(ref_ct.dataobj[sl], dtype=np.float32)
    fixed_ct = np.asarray(target_ct.dataobj[sl], dtype=np.float32)
    fixed = make_itk(fixed_ct, coarse_affine)
    moving = make_itk(moving_ct, coarse_affine)
    registrar = monai_physio.RegisterImagesGreedy()
    registrar.set_modality("ct")
    registrar.set_metric("CC")
    registrar.set_number_of_iterations(list(args.iterations))
    registrar.deformable_smoothing = args.deformable_smoothing
    registrar.set_transform_type("Deformable")
    registrar.set_fixed_image(fixed)
    print("Greedy registration started", flush=True)
    result = registrar.register(moving)
    print("Greedy registration returned", flush=True)
    f2m = result["fixed_to_moving_transform"]
    m2f = result["moving_to_fixed_transform"]
    transforms = monai_physio.ProcessTransforms()
    warped_itk = transforms.transform_image(make_itk(moving_mask, coarse_affine),
                                            f2m, fixed, interpolation_method="nearest")
    warped = itk.array_from_image(warped_itk).transpose(2, 1, 0) > 0
    warped_ct_itk = transforms.transform_image(moving, f2m, fixed,
                                               interpolation_method="linear",
                                               background_value=-1000.0)
    warped_ct = itk.array_from_image(warped_ct_itk).transpose(2, 1, 0)
    shift_ras = np.asarray(prior["registration"]["shift_ras_mm_reference_to_target"])
    shift_ijk = np.linalg.solve(coarse_affine[:3, :3], shift_ras)
    rigid = ndi.shift(moving_mask, shift_ijk, order=0, mode="constant", cval=0,
                      prefilter=False) > 0
    rigid_ct = ndi.shift(moving_ct, shift_ijk, order=1, mode="constant", cval=-1000.0,
                         prefilter=False)
    static = moving_mask > 0
    truth = fixed_mask > 0
    print("Mask resampling complete", flush=True)
    field_itk = transforms.convert_transform_to_displacement_field(
        f2m, fixed, np_component_type=np.float32)
    field_zyx_lps = itk.array_from_image(field_itk)
    # Both API transforms must agree on opposite point directions.
    samples = []
    for i in np.linspace(15, truth.shape[0] - 16, 5):
        for j in np.linspace(15, truth.shape[1] - 16, 5):
            for k in np.linspace(15, truth.shape[2] - 16, 5):
                p_lps = F @ (coarse_affine[:3, :3] @ np.array([i, j, k]) + coarse_affine[:3, 3])
                q_lps = f2m.TransformPoint(tuple(p_lps))
                restored = m2f.TransformPoint(q_lps)
                samples.append(float(np.linalg.norm(np.asarray(restored) - p_lps)))
    field_sitk = sitk.GetImageFromArray(field_zyx_lps, isVector=True)
    field_sitk.SetSpacing(tuple(float(x) for x in np.linalg.norm(coarse_affine[:3, :3], axis=0)))
    field_sitk.SetOrigin(tuple(float(x) for x in F @ coarse_affine[:3, 3]))
    field_sitk.SetDirection(tuple((F @ coarse_affine[:3, :3] /
        np.linalg.norm(coarse_affine[:3, :3], axis=0)).ravel()))
    jac = sitk.GetArrayFromImage(sitk.DisplacementFieldJacobianDeterminant(field_sitk))
    # Fixed-grid physical pullback x_target -> x_reference; displacement vectors LPS mm.
    delta_ijk = field_zyx_lps @ np.linalg.inv(F @ coarse_affine[:3, :3]).T
    nx, ny, nz = truth.shape
    outside = ((np.arange(nx)[None, None, :] + delta_ijk[..., 0] < 0) |
               (np.arange(nx)[None, None, :] + delta_ijk[..., 0] > nx - 1) |
               (np.arange(ny)[None, :, None] + delta_ijk[..., 1] < 0) |
               (np.arange(ny)[None, :, None] + delta_ijk[..., 1] > ny - 1) |
               (np.arange(nz)[:, None, None] + delta_ijk[..., 2] < 0) |
               (np.arange(nz)[:, None, None] + delta_ijk[..., 2] > nz - 1))
    # Do not write outside ROI unless required for local verification.
    args.field_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.field_output, displacement_zyx_lps_mm=field_zyx_lps,
                        origin_lps_mm=np.asarray(field_sitk.GetOrigin()),
                        spacing_mm=np.asarray(field_sitk.GetSpacing()),
                        direction_lps=np.asarray(field_sitk.GetDirection()).reshape(3, 3))
    metrics = {"stationary": comparison(static, truth, coarse_affine),
               "rigid": comparison(rigid, truth, coarse_affine),
               "greedy": comparison(warped, truth, coarse_affine)}
    eval_roi = ndi.binary_dilation(truth | static, iterations=5)
    ct_mse = {"stationary": float(np.mean((moving_ct[eval_roi] - fixed_ct[eval_roi]) ** 2)),
              "rigid": float(np.mean((rigid_ct[eval_roi] - fixed_ct[eval_roi]) ** 2)),
              "greedy": float(np.mean((warped_ct[eval_roi] - fixed_ct[eval_roi]) ** 2))}
    mesh = mesh_distances(static, truth, coarse_affine, shift_ijk, m2f)
    report = {"status": "one_pair_monai_physio_greedy_candidate_unvalidated",
        "subject_alias": prior["subject_alias"], "moving_reference": args.reference_phase,
        "fixed_target": args.target_phase,
        "input_sha256": prior["input_sha256"],
        "unit_evidence": {"nifti_header_space_units": ref_ct.header.get_xyzt_units()[0],
                          "conversion_manifest_spacing_mm": entries[ref_path.name]["voxel_spacing_mm"],
                          "manifest_coordinate_system": "NIfTI RAS+",
                          "itk_transform_system": "LPS millimetres"},
        "method": {"api": "monai_physio.RegisterImagesGreedy.register",
                   "backend": "picsl_greedy in-memory Python extension",
                   "transform_type": "Deformable (Greedy affine initialization + displacement)",
                   "metric": "CC/WNCC 2x2x2", "iterations_per_level": list(args.iterations),
                   "deformable_smoothing": registrar.deformable_smoothing,
                   "masks_used_to_fit": False,
                   "mask_role": ("source mask defines crop; target mask evaluation only"
                                 if args.crop_source_only else "both masks define common crop; evaluation"),
                   "fixed_to_moving": "target LPS point -> source LPS point; pullback for image/mask resampling",
                   "moving_to_fixed": "source LPS point -> target LPS point; pushforward for vertices",
                   "loss_reported_by_api": float(result["loss"])},
        "software": {"python": platform.python_version(),
                     "monai_physio": monai_physio.__version__,
                     "itk": itk.Version.GetITKVersion(),
                     "picsl_greedy_module": getattr(picsl_greedy, "__version__", "not_exposed"),
                     "script_sha256": digest(Path(__file__))},
        "grid_qc": {"full_shape_ijk": list(ref_ct.shape), "crop_start_ijk": lo.tolist(),
                    "crop_stop_ijk_exclusive": hi.tolist(), "stride_ijk": stride.tolist(),
                    "crop_source_only": bool(args.crop_source_only),
                    "crop_margin_mm": float(args.margin_mm),
                    "evaluation_shape_ijk": list(truth.shape),
                    "evaluation_spacing_mm": np.linalg.norm(coarse_affine[:3, :3], axis=0).tolist(),
                    "source_mask_touches_full_fov": touches_full_fov(*boxes[0], ref_ct.shape),
                    "target_mask_touches_full_fov": touches_full_fov(*boxes[1], ref_ct.shape),
                    "crop_touches_full_fov": touches_full_fov(lo, hi, ref_ct.shape)},
        "field_qc": {"field_direction": "fixed target -> moving reference, LPS mm",
                     "field_array_axes": "z,y,x,component_LPS_xyz", "local_field_saved": True,
                     "local_field_relative_path": "runs/" + args.field_output.name,
                     "local_field_sha256": digest(args.field_output),
                     "forward_inverse_point_error_mm": {"median": float(np.median(samples)),
                                                        "maximum": float(np.max(samples))},
                     "jacobian_of_pullback": {"minimum": float(np.min(jac)),
                         "p01": float(np.percentile(jac, 1)), "median": float(np.median(jac)),
                         "p99": float(np.percentile(jac, 99)), "maximum": float(np.max(jac)),
                         "nonpositive_fraction": float(np.mean(jac <= 0))},
                     "pullback_samples_outside_moving_crop_fraction": float(np.mean(outside)),
                     "displacement_magnitude_mm": {"p50": float(np.percentile(np.linalg.norm(field_zyx_lps, axis=-1), 50)),
                                                   "p95": float(np.percentile(np.linalg.norm(field_zyx_lps, axis=-1), 95))}},
        "mask_qc": metrics, "ct_mse_hu2_in_dilated_union_mask": ct_mse,
        "same_surface_qc": mesh,
        "independent_anatomical_reference": False,
        "interpretation": (("Technical one-pair comparison on the native CT voxel grid in a cropped ROI; "
                            if np.all(stride == 1) else
                            "Technical one-pair comparison on a decimated common grid; ") +
                           "automatic target mask is not an independent anatomical reference. "
                           "No ED/ES, EF, SV, FEM or full-4D claim.")}
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Aggregate QC written", flush=True)


if __name__ == "__main__":
    main()
