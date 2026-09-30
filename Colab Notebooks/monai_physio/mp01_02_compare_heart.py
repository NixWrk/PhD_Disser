"""Compare MONAI highres cardiac labels 140-144 with project raw masks.

Only a diagnostic comparison of two automatic runs. Both source masks and all
output reports remain local; the report path is restricted to ignored runs/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


LABELS = {
    "highres_myocardium": "heart_myocardium.nii.gz",
    "highres_atrium_left": "heart_atrium_left.nii.gz",
    "highres_ventricle_left": "heart_ventricle_left.nii.gz",
    "highres_atrium_right": "heart_atrium_right.nii.gz",
    "highres_ventricle_right": "heart_ventricle_right.nii.gz",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_grid(reference: nib.Nifti1Image, other: nib.Nifti1Image) -> None:
    if reference.shape != other.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} vs {other.shape}")
    gap = float(np.max(np.abs(reference.affine - other.affine)))
    if gap > 1e-4:
        raise ValueError(f"RAS affine mismatch; maximum difference {gap:g} mm")


def binary(path: Path, reference: nib.Nifti1Image) -> np.ndarray:
    image = nib.load(str(path))
    check_grid(reference, image)
    values = np.asanyarray(image.dataobj)
    if not np.isin(np.unique(values), [0, 1]).all():
        raise ValueError(f"Expected binary mask: {path.name}")
    return values.astype(bool)


def metrics(a: np.ndarray, b: np.ndarray, voxel_ml: float) -> dict:
    n_a = int(a.sum())
    n_b = int(b.sum())
    both = int(np.count_nonzero(a & b))
    either = n_a + n_b - both
    changed = n_a + n_b - 2 * both
    return {
        "baseline_voxels": n_a,
        "candidate_voxels": n_b,
        "baseline_volume_ml": n_a * voxel_ml,
        "candidate_volume_ml": n_b * voxel_ml,
        "dice": 2 * both / (n_a + n_b) if n_a + n_b else None,
        "jaccard": both / either if either else None,
        "symmetric_difference_voxels": changed,
        "symmetric_difference_ml": changed * voxel_ml,
        "candidate_minus_baseline_ml": (n_b - n_a) * voxel_ml,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--whole-heart", type=Path,
                        help="Raw total/heart for cross-class containment only; never segmentation accuracy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assume-mm", action="store_true")
    args = parser.parse_args()
    allowed_root = (Path(__file__).resolve().parent / "runs").resolve()
    out = args.output.resolve()
    if out.parent != allowed_root or out.exists():
        parser.error("Output must be a new JSON directly inside the ignored runs/ directory")

    manifest = json.loads((args.candidate_dir / "mp01_02_manifest.json").read_text(encoding="utf-8"))
    is_partial = (manifest.get("status") == "partial_monai_physio_heart_only_diagnostic"
                  and manifest.get("full_monai_physio_workflow") is False
                  and manifest.get("executed_tasks") == ["total", "heartchambers_highres"]
                  and manifest.get("skipped_auxiliary_tasks") == ["tissue_4_types", "lung_vessels", "body"])
    is_full = (manifest.get("status") == "automatic_monai_physio_candidate_only"
               and manifest.get("full_monai_physio_workflow") is True
               and manifest.get("executed_tasks") == ["total", "heartchambers_highres", "tissue_4_types", "lung_vessels", "body"]
               and manifest.get("skipped_auxiliary_tasks") == [])
    if (not (is_partial or is_full) or not manifest.get("academic_license_offline_precheck")
        or manifest.get("contrast_variant")):
        raise ValueError("Candidate manifest does not describe the approved full or partial route")
    if manifest["ct_sha256"] != sha256(args.ct):
        raise ValueError("Candidate was generated from a different CT file")

    ct = nib.load(str(args.ct))
    spatial_unit = ct.header.get_xyzt_units()[0]
    if spatial_unit != "mm" and not args.assume_mm:
        raise ValueError("CT spatial unit is unknown; require explicit --assume-mm")
    candidate_union_path = args.candidate_dir / "mp01_02_highres_heart_union.nii.gz"
    candidate_union = binary(candidate_union_path, ct)
    voxel_ml = abs(float(np.linalg.det(ct.affine[:3, :3]))) / 1000
    baseline_union = np.zeros(ct.shape, dtype=bool)
    per_class = {}
    for name, baseline_filename in LABELS.items():
        baseline = binary(args.baseline_dir / baseline_filename, ct)
        candidate = binary(args.candidate_dir / f"mp01_02_{name}.nii.gz", ct)
        per_class[name] = metrics(baseline, candidate, voxel_ml)
        baseline_union |= baseline
    _, baseline_components = ndimage.label(baseline_union)
    _, candidate_components = ndimage.label(candidate_union)
    report = {
        "status": ("partial_automatic_heart_labels_diagnostic_only" if is_partial
                   else "full_wrapper_automatic_heart_labels_candidate_only"),
        "comparison_semantics": "MONAI labels 140-144 versus union of five raw TotalSegmentator heartchambers_highres classes; excludes residual 51, aorta and pulmonary artery",
        "same_ct_sha256_verified_for_candidate": True,
        "baseline_same_phase_based_on_project_provenance_not_ct_hash": True,
        "same_grid": True,
        "shape": list(ct.shape),
        "spacing_header": [float(value) for value in ct.header.get_zooms()[:3]],
        "spatial_unit_header": spatial_unit,
        "assumed_mm": bool(args.assume_mm),
        "affine_max_difference_mm": 0.0,
        "per_class": per_class,
        "highres_union": metrics(baseline_union, candidate_union, voxel_ml),
        "components_6_connected": {"baseline": int(baseline_components),
                                   "candidate": int(candidate_components)},
        "limitations": (["Not the full MONAI Physio wrapper; three auxiliary tasks replaced by zero masks."]
                        if is_partial else []) + [
                        "MONAI resamples CT to 1 mm isotropic before inference; project TotalSegmentator run used separate preprocessing and ROI settings.",
                        "Agreement between automatic masks is not independent anatomical validation."],
    }
    if args.whole_heart:
        whole = binary(args.whole_heart, ct)
        whole_count = int(whole.sum())

        def relation(highres: np.ndarray) -> dict:
            total_count = int(highres.sum())
            inside = int(np.count_nonzero(highres & whole))
            outside = total_count - inside
            return {"highres_volume_ml": total_count * voxel_ml,
                    "inside_raw_total_heart_ml": inside * voxel_ml,
                    "outside_raw_total_heart_ml": outside * voxel_ml,
                    "outside_fraction_of_highres": outside / total_count if total_count else None,
                    "fraction_of_raw_total_heart_covered": inside / whole_count if whole_count else None}

        report["cross_class_geometry_only"] = {
            "interpretation": "highres myocardium+chambers versus raw total/heart; containment only, not segmentation accuracy or a model ranking",
            "raw_total_heart_volume_ml": whole_count * voxel_ml,
            "project_highres_union": relation(baseline_union),
            "candidate_monai_highres_union": relation(candidate_union),
        }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "highres_union": report["highres_union"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
