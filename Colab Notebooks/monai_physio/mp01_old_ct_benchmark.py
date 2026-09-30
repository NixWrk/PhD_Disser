"""Compare MONAI Physio and project TotalSegmentator on OLD CT phases 00/08.

Direct contours are candidate, unblinded annotations and never treated as
independent ground truth. Outputs contain pseudonym-free aggregate values only.
"""

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


CLASSES = {
    "LV": ("highres_ventricle_left", "heart_ventricle_left.nii.gz", "LV_cavity_envelope.nii.gz"),
    "RV": ("highres_ventricle_right", "heart_ventricle_right.nii.gz", "RV_cavity_envelope.nii.gz"),
    "LA": ("highres_atrium_left", "heart_atrium_left.nii.gz", "LA_cavity_envelope.nii.gz"),
    "RA": ("highres_atrium_right", "heart_atrium_right.nii.gz", "RA_cavity_envelope.nii.gz"),
    "myocardium": ("highres_myocardium", "heart_myocardium.nii.gz", None),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_mask(path, ct):
    image = nib.load(str(path))
    if image.shape != ct.shape or not np.allclose(image.affine, ct.affine, atol=1e-4):
        raise ValueError(f"Mask and CT physical grids differ: {path.name}")
    arr = np.asanyarray(image.dataobj)
    if not np.isin(np.unique(arr), [0, 1]).all():
        raise ValueError(f"Mask is not binary: {path.name}")
    result = arr.astype(bool)
    if not result.any():
        raise ValueError(f"Mask is empty: {path.name}")
    return result


def volume(mask, voxel_ml):
    return float(mask.sum() * voxel_ml)


def comparison(a, b, spacing):
    """Symmetric metrics on a tight physical ROI; a and b must share a grid."""
    va, vb = int(a.sum()), int(b.sum())
    overlap = int(np.count_nonzero(a & b))
    bounds = np.argwhere(a | b)
    lo = np.maximum(bounds.min(axis=0) - 3, 0)
    hi = np.minimum(bounds.max(axis=0) + 4, a.shape)
    sl = tuple(slice(int(x), int(y)) for x, y in zip(lo, hi))
    aa, bb = a[sl], b[sl]
    structure = ndi.generate_binary_structure(3, 1)
    edge_a = aa & ~ndi.binary_erosion(aa, structure=structure)
    edge_b = bb & ~ndi.binary_erosion(bb, structure=structure)
    dist_to_b = ndi.distance_transform_edt(~edge_b, sampling=spacing)[edge_a]
    dist_to_a = ndi.distance_transform_edt(~edge_a, sampling=spacing)[edge_b]
    distances = np.concatenate((dist_to_b, dist_to_a))
    return {
        "dice": 2 * overlap / (va + vb),
        "assd_mm": float(distances.mean()),
        "hd95_mm": float(np.percentile(distances, 95)),
        "a_minus_b_volume_ml": float((va - vb) * np.prod(spacing) / 1000),
    }


def inspect(old_root, run_root):
    phases = {}
    for phase in ("phase_00", "phase_08"):
        ct_path = old_root / "sequence_02" / "phases" / f"{phase}.nii.gz"
        ct = nib.load(str(ct_path))
        spacing = np.linalg.norm(ct.affine[:3, :3], axis=0)
        voxel_ml = abs(float(np.linalg.det(ct.affine[:3, :3]))) / 1000
        run = run_root / f"mp01_old_{phase.replace('phase_', 'phase')}_full"
        manifest = json.loads((run / "mp01_02_manifest.json").read_text(encoding="utf-8"))
        if (manifest["ct_sha256"] != sha256(ct_path) or
                not manifest["full_monai_physio_workflow"] or manifest["contrast_variant"]):
            raise ValueError("MONAI output does not match this CT and the declared full workflow")
        ts_root = old_root / "sequence02_ts_comparator_v1" / "automatic_masks_v2" / phase / "heartchambers_highres"
        direct_root = old_root / "anatomical_audit_v02" / "direct_v01" / phase
        per_class = {}
        for name, (monai_name, ts_name, direct_name) in CLASSES.items():
            monai = read_mask(run / f"mp01_02_{monai_name}.nii.gz", ct)
            ts = read_mask(ts_root / ts_name, ct)
            record = {
                "monai_volume_ml": volume(monai, voxel_ml),
                "totalsegmentator_volume_ml": volume(ts, voxel_ml),
                "monai_vs_totalsegmentator": comparison(monai, ts, spacing),
            }
            if direct_name:
                direct = read_mask(direct_root / direct_name, ct)
                record["candidate_direct_volume_ml"] = volume(direct, voxel_ml)
                record["monai_vs_candidate_direct"] = comparison(monai, direct, spacing)
                record["totalsegmentator_vs_candidate_direct"] = comparison(ts, direct, spacing)
            per_class[name] = record
        phases[phase] = {
            "ct_sha256": manifest["ct_sha256"],
            "monai_labelmap_sha256": manifest["labelmap_sha256"],
            "shape": list(ct.shape),
            "spacing_mm_from_affine": spacing.tolist(),
            "voxel_volume_ml": voxel_ml,
            "classes": per_class,
        }
    dynamics = {}
    for name in CLASSES:
        a, b = (phases[phase]["classes"][name] for phase in ("phase_00", "phase_08"))
        delta = {
            "monai_delta_ml": b["monai_volume_ml"] - a["monai_volume_ml"],
            "totalsegmentator_delta_ml": b["totalsegmentator_volume_ml"] - a["totalsegmentator_volume_ml"],
        }
        if "candidate_direct_volume_ml" in a:
            delta["candidate_direct_delta_ml"] = b["candidate_direct_volume_ml"] - a["candidate_direct_volume_ml"]
            delta["monai_minus_candidate_delta_ml"] = delta["monai_delta_ml"] - delta["candidate_direct_delta_ml"]
            delta["totalsegmentator_minus_candidate_delta_ml"] = delta["totalsegmentator_delta_ml"] - delta["candidate_direct_delta_ml"]
        dynamics[name] = delta
    return {
        "schema": "mp01_old_ct_benchmark_v1",
        "case_alias": "historical_01_sequence_02",
        "method": "Full MONAI Physio 2026.09.1 TotalSegmentator wrapper versus saved project TotalSegmentator v2; same phases and physical grid",
        "reference_status": "Direct contour envelopes on two phases were created after viewing TotalSegmentator; not blinded, expert-accepted, or independent",
        "phase_00_percent": 0,
        "phase_08_percent": 40,
        "phases": phases,
        "volume_changes_00_to_08": dynamics,
        "decision": "pending_anatomical_review_and_scope_assessment",
        "interpretation": "Agreement among automatic outputs or unblinded candidate contours cannot establish absolute segmentation accuracy",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path(__file__).resolve().parent / "runs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.old_root, args.run_root)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["decision"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
