"""Small, read-only 4D CT registration and geometry QC pilot.

Inputs are existing phase HU NIfTI and TotalSegmentator v2 product masks.
Only aggregate JSON is emitted. This rigid CT-only comparator is not MONAI
Physio, deformable registration, or independent anatomical validation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import minimize
from skimage.measure import marching_cubes


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked_image(path: Path, reference: nib.Nifti1Image | None = None) -> nib.Nifti1Image:
    image = nib.load(str(path))
    affine = image.affine
    if len(image.shape) != 3 or not np.isfinite(affine).all():
        raise ValueError(f"invalid 3D geometry: {path.name}")
    if min(np.linalg.norm(affine[:3, :3], axis=0)) <= 0 or abs(np.linalg.det(affine[:3, :3])) < 1e-8:
        raise ValueError(f"singular affine: {path.name}")
    if reference is not None and (image.shape != reference.shape or not np.allclose(affine, reference.affine, rtol=0, atol=1e-4)):
        raise ValueError(f"FOV or affine differs: {path.name}")
    return image


def bounds(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.argwhere(mask)
    if len(points) == 0:
        raise ValueError("empty mask")
    return points.min(0), points.max(0) + 1


def geometry(mask: np.ndarray, affine: np.ndarray) -> dict:
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    count = int(mask.sum())
    if count == 0:
        raise ValueError("empty mask")
    ijk = np.argwhere(mask)
    center = affine[:3, :3] @ ijk.mean(0) + affine[:3, 3]
    area = 0.0
    # Exposed voxel-face area is a reproducible discrete surface proxy.
    for axis in range(3):
        pad = [(0, 0)] * 3
        pad[axis] = (1, 1)
        face_count = np.count_nonzero(np.diff(np.pad(mask.astype(np.int8), pad), axis=axis))
        area += face_count * np.prod(np.delete(spacing, axis))
    return {"volume_ml": float(count * abs(np.linalg.det(affine[:3, :3])) / 1000),
            "center_ras_mm": center.tolist(), "voxel_face_area_mm2": float(area),
            "voxels": count}


def surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndi.binary_erosion(mask, border_value=0)


def touches_full_fov(low: np.ndarray, high: np.ndarray, shape: tuple[int, ...]) -> bool:
    return bool(np.any(low == 0) or np.any(high == shape))


def comparison(candidate: np.ndarray, target: np.ndarray, affine: np.ndarray) -> dict:
    a, b = geometry(candidate, affine), geometry(target, affine)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    sa, sb = surface(candidate), surface(target)
    da = ndi.distance_transform_edt(~sa, sampling=spacing)
    db = ndi.distance_transform_edt(~sb, sampling=spacing)
    distances = np.concatenate((db[sa], da[sb]))
    overlap = int(np.count_nonzero(candidate & target))
    return {"dice_vs_auto_mask": float(2 * overlap / (a["voxels"] + b["voxels"])),
            "center_difference_mm": float(np.linalg.norm(np.array(a["center_ras_mm"]) - b["center_ras_mm"])),
            "volume_difference_ml": a["volume_ml"] - b["volume_ml"],
            "volume_difference_percent_of_target": 100 * (a["volume_ml"] / b["volume_ml"] - 1),
            "voxel_face_area_difference_mm2": a["voxel_face_area_mm2"] - b["voxel_face_area_mm2"],
            "surface_assd_mm": float(distances.mean()),
            "surface_hd95_mm": float(np.percentile(distances, 95))}


def same_surface_qc(reference: np.ndarray, target: np.ndarray, shift_ijk: np.ndarray,
                    spacing: np.ndarray) -> dict:
    # One coarse mesh; vertex IDs and connectivity are unchanged under translation.
    stride = np.array([2, 2, 2])
    vertices, faces, _, _ = marching_cubes(reference[::2, ::2, ::2].astype(np.uint8), 0.5)
    vertices_ijk = vertices * stride
    distance = ndi.distance_transform_edt(~surface(target), sampling=spacing)
    def errors(shift: np.ndarray) -> np.ndarray:
        positions = (vertices_ijk + shift).T
        return ndi.map_coordinates(distance, positions, order=1, mode="constant", cval=np.nan)
    stationary = errors(np.zeros(3))
    moved = errors(shift_ijk)
    return {"reference_mesh_stride_ijk": stride.tolist(), "vertex_count": int(len(vertices)),
            "triangle_count": int(len(faces)), "same_vertex_ids_and_faces_all_targets": True,
            "stationary_mean_target_surface_distance_mm": float(np.nanmean(stationary)),
            "translated_mean_target_surface_distance_mm": float(np.nanmean(moved)),
            "translated_p95_target_surface_distance_mm": float(np.nanpercentile(moved, 95)),
            "translated_vertices_outside_crop_fraction": float(np.mean(~np.isfinite(moved))),
            "mesh_and_mask_interpolation": "coarse marching cubes reference mesh, continuous translation; separate nearest-neighbor mask"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-dir", required=True, type=Path)
    parser.add_argument("--products-dir", required=True, type=Path)
    parser.add_argument("--reference", default="phase_00")
    parser.add_argument("--target", default="phase_01")
    parser.add_argument("--label", choices=("whole_heart", "blood_pool", "myocardium"), default="whole_heart")
    parser.add_argument("--subject-alias", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.reference == args.target:
        parser.error("reference and target must differ")
    names = sorted(path.name for path in args.phase_dir.glob("phase_*.nii.gz"))
    if f"{args.reference}.nii.gz" not in names or f"{args.target}.nii.gz" not in names:
        parser.error("selected phase absent")
    ref_path = args.phase_dir / f"{args.reference}.nii.gz"
    target_path = args.phase_dir / f"{args.target}.nii.gz"
    ref_ct = checked_image(ref_path)
    target_ct = checked_image(target_path, ref_ct)
    # Header-only cohort check: phase names alone do not establish common FOV.
    for name in names:
        checked_image(args.phase_dir / name, ref_ct)
    mask_paths = [args.products_dir / phase / f"{args.label}.nii.gz" for phase in (args.reference, args.target)]
    ref_mask_image = checked_image(mask_paths[0], ref_ct)
    target_mask_image = checked_image(mask_paths[1], ref_ct)
    ref_mask = np.asarray(ref_mask_image.dataobj) > 0
    target_mask = np.asarray(target_mask_image.dataobj) > 0
    lo0, hi0 = bounds(ref_mask)
    lo1, hi1 = bounds(target_mask)
    spacing = np.linalg.norm(ref_ct.affine[:3, :3], axis=0)
    margin = np.ceil(25 / spacing).astype(int)
    lo = np.maximum(0, np.minimum(lo0, lo1) - margin)
    hi = np.minimum(ref_ct.shape, np.maximum(hi0, hi1) + margin)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    crop_affine = ref_ct.affine.copy()
    crop_affine[:3, 3] += ref_ct.affine[:3, :3] @ lo
    a = ref_mask[sl].copy()
    b = target_mask[sl].copy()
    del ref_mask, target_mask
    # Coarse CT-only rigid translation. Masks determine a common ROI, never the fit.
    step = np.maximum(1, np.ceil(2 / spacing).astype(int))
    coarse = tuple(slice(None, None, int(s)) for s in step)
    fixed = np.asarray(target_ct.dataobj[sl], dtype=np.float32)[coarse]
    moving = np.asarray(ref_ct.dataobj[sl], dtype=np.float32)[coarse]
    fixed = ndi.gaussian_filter(np.clip(fixed, -200, 800), 0.7)
    moving = ndi.gaussian_filter(np.clip(moving, -200, 800), 0.7)
    # Optimize in coarse voxel units; a constant interior excludes fill artifacts.
    interior = tuple(slice(5, -5) for _ in range(3))
    def objective(x: np.ndarray) -> float:
        shifted = ndi.shift(moving, x, order=1, mode="nearest", prefilter=False)
        d = fixed[interior] - shifted[interior]
        return float(np.mean(d * d))
    limit = 8 / (spacing * step)
    result = minimize(objective, np.zeros(3), method="Powell",
                      bounds=[(-float(v), float(v)) for v in limit],
                      options={"maxiter": 18, "xtol": 0.05, "ftol": 1e-4})
    shift_ijk = result.x * step
    translated = ndi.shift(a.astype(np.uint8), shift_ijk, order=0, mode="constant", cval=0, prefilter=False) > 0
    translated_full_volume = geometry(a, crop_affine)["volume_ml"]
    retained = geometry(translated, crop_affine)["volume_ml"]
    # Constant translation has Jacobian determinant exactly one throughout its domain.
    reference_geometry = geometry(a, crop_affine)
    target_geometry = geometry(b, crop_affine)
    # Avoid persisting individual absolute coordinates or image origin.
    reference_geometry.pop("center_ras_mm")
    target_geometry.pop("center_ras_mm")
    report = {"status": "rigid_ct_pilot_not_validated", "subject_alias": args.subject_alias,
              "phase_ids": names, "reference": args.reference, "target": args.target,
              "phase_timing": "not inferred from filename; ED/ES unassigned",
              "input_kind": "measured CT reconstruction; TotalSegmentator v2 auto masks",
              "independent_anatomical_reference": False,
              "input_sha256": {"reference_ct": digest(ref_path), "target_ct": digest(target_path),
                               "reference_mask": digest(mask_paths[0]), "target_mask": digest(mask_paths[1])},
              "software": {"python": platform.python_version(), "numpy": np.__version__,
                           "scipy": __import__("scipy").__version__,
                           "monai_physio_available": importlib.util.find_spec("monai_physio") is not None},
              "coordinate_qc": {"convention": "NIfTI RAS millimetres", "shape_ijk": list(ref_ct.shape),
                                "axis_vectors_ras_mm": ref_ct.affine[:3, :3].tolist(), "voxel_spacing_mm": spacing.tolist(),
                                "all_ct_phases_same_fov_affine": True, "selected_masks_same_fov_affine": True,
                                "crop_start_ijk": lo.tolist(), "crop_stop_ijk_exclusive": hi.tolist(),
                                "crop_touches_image_boundary": touches_full_fov(lo, hi, ref_ct.shape),
                                "reference_mask_touches_image_boundary": touches_full_fov(lo0, hi0, ref_ct.shape),
                                "target_mask_touches_image_boundary": touches_full_fov(lo1, hi1, ref_ct.shape)},
              "registration": {"method": "CT intensity rigid translation Powell MSE coarse ROI",
                               "shift_ijk_voxels_reference_to_target": shift_ijk.tolist(),
                               "shift_ras_mm_reference_to_target": (ref_ct.affine[:3, :3] @ shift_ijk).tolist(),
                               "coarse_step_ijk": step.tolist(), "objective_initial": objective(np.zeros(3)),
                               "objective_final": objective(result.x), "optimizer_success": bool(result.success),
                               "translation_at_search_bound": bool(np.any(np.abs(result.x) >= 0.95 * limit)),
                               "jacobian_determinant": 1.0, "nonpositive_jacobian_fraction": 0.0,
                               "deformable_field_checked": False},
              "geometry": {"reference_auto_mask": reference_geometry,
                           "target_auto_mask": target_geometry,
                           "stationary_vs_target": comparison(a, b, crop_affine),
                           "translated_vs_target": comparison(translated, b, crop_affine),
                           "same_surface_qc": same_surface_qc(a, b, shift_ijk, spacing),
                           "translated_mask_volume_retained_in_crop_fraction": retained / translated_full_volume},
              "mp04_status": "rigid same-vertex surface pilot only; deformable all-phase geometry blocked until field QC",
              "interpretation": "Within-series auto-mask consistency only. The same algorithm generated both masks; no independent registration, segmentation, motion, ED/ES, or FEM validation."}
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
