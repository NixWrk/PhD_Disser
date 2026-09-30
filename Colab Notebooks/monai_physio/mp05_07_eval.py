"""Local, read-only MP-05/07 pilot. No medical arrays or machine paths are saved.

Run with an external data root: python mp05_07_eval.py baseline --data-root ROOT
or: python mp05_07_eval.py pca --data-root ROOT
The root is deliberately a CLI argument, never a tracked default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


SUBJECTS = ("adam", "georg", "nix")
LABEL = "whole_heart"  # TotalSegmentator v2, automatic candidate reference


def image(path: Path) -> nib.Nifti1Image:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required input: {path.name}")
    return nib.load(str(path))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mask(im: nib.Nifti1Image) -> np.ndarray:
    data = np.asarray(im.dataobj)
    if not np.isfinite(data).all() or np.any((data != 0) & (data != 1)):
        raise ValueError("Expected finite binary NIfTI mask")
    return data.astype(bool)


def same_grid(a: nib.Nifti1Image, b: nib.Nifti1Image) -> bool:
    return a.shape == b.shape and np.allclose(a.affine, b.affine, atol=1e-4, rtol=0)


def geometry(binary: np.ndarray, affine: np.ndarray) -> tuple[float, np.ndarray]:
    voxels = int(binary.sum())
    if voxels == 0:
        raise ValueError("Empty heart mask")
    ml = voxels * abs(np.linalg.det(affine[:3, :3])) / 1000
    ijk = np.asarray(ndi.center_of_mass(binary))
    xyz = affine[:3, :3] @ ijk + affine[:3, 3]
    return float(ml), xyz


def compare(pred: np.ndarray, truth: np.ndarray, affine: np.ndarray) -> dict:
    if pred.shape != truth.shape:
        raise ValueError("Masks have different grids")
    vp, cp = geometry(pred, affine)
    vt, ct = geometry(truth, affine)
    intersection = int(np.count_nonzero(pred & truth))
    dice = 2 * intersection / (int(pred.sum()) + int(truth.sum()))
    # Crop both masks together; distance transforms over the full CT FOV waste RAM.
    support = pred | truth
    points = np.where(support)
    sl = tuple(slice(max(0, int(p.min()) - 3), min(n, int(p.max()) + 4))
               for p, n in zip(points, pred.shape))
    a, b = pred[sl], truth[sl]
    structure = ndi.generate_binary_structure(3, 1)
    sa = a & ~ndi.binary_erosion(a, structure=structure)
    sb = b & ~ndi.binary_erosion(b, structure=structure)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    # A sheared grid would invalidate orthogonal EDT distances.
    axes = affine[:3, :3] / spacing
    if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-4):
        raise ValueError("Sheared grid: surface distance requires a different metric")
    da = ndi.distance_transform_edt(~sa, sampling=spacing)[sb]
    db = ndi.distance_transform_edt(~sb, sampling=spacing)[sa]
    distances = np.concatenate((da, db))
    return {
        "dice": round(float(dice), 6),
        "centroid_error_mm": round(float(np.linalg.norm(cp - ct)), 4),
        "pred_volume_ml": round(vp, 4),
        "reference_volume_ml": round(vt, 4),
        "volume_error_ml": round(abs(vp - vt), 4),
        "volume_error_percent_of_reference": round(100 * abs(vp - vt) / vt, 4),
        "assd_mm": round(float(distances.mean()), 4),
        "hd95_mm": round(float(np.percentile(distances, 95)), 4),
    }


def phase_inputs(root: Path, subject: str) -> list[tuple[str, Path, Path]]:
    folder = root / subject
    products = folder / "products_v2"
    phase_dirs = sorted(products.glob("phase_*"))
    result = []
    for d in phase_dirs:
        m = d / f"{LABEL}.nii.gz"
        ct = folder / "phases" / f"{d.name}.nii.gz"
        if m.is_file() and ct.is_file():
            result.append((d.name, m, ct))
        else:
            raise FileNotFoundError(f"Incomplete phase: {subject}/{d.name}")
    if len(result) < 2 or result[0][0] != "phase_00":
        raise ValueError(f"Need phase_00 and other actual phases: {subject}")
    return result


def baseline(root: Path, subjects: tuple[str, ...]) -> dict:
    output = {"method": "fixed_phase_00_mask", "reference": "automatic_TotalSegmentator_v2",
              "status": "exploratory_hypothesis_not_validated", "subjects": {}}
    for subject in subjects:
        phases = phase_inputs(root, subject)
        conversion = json.loads((root / subject / "phases" / "conversion_manifest.json").read_text(encoding="utf-8"))
        phase_values = {row["phase_id"]: row["dicom_phase_value"] for row in conversion["phases"]}
        if set(phase_values) != {phase for phase, _, _ in phases}:
            raise ValueError(f"Phase metadata mismatch: {subject}")
        ref = image(phases[0][1])
        ct0 = image(phases[0][2])
        if not same_grid(ref, ct0):
            raise ValueError(f"CT/mask grid mismatch: {subject}/phase_00")
        stationary = mask(ref)  # Prediction is locked before reading target masks.
        v0, c0 = geometry(stationary, ref.affine)
        rows = []
        for phase, mask_path, ct_path in phases[1:]:
            target_im = image(mask_path)
            ct_im = image(ct_path)
            if not same_grid(ref, target_im) or not same_grid(target_im, ct_im):
                raise ValueError(f"Phase grid mismatch: {subject}/{phase}")
            target = mask(target_im)  # Held out until fixed prediction is set.
            metrics = compare(stationary, target, ref.affine)
            _, ct = geometry(target, ref.affine)
            metrics.update({"phase": phase,
                            "dicom_nominal_rr_percent": phase_values[phase],
                            "mask_sha256": sha256(mask_path),
                            "actual_centroid_displacement_mm": round(float(np.linalg.norm(ct - c0)), 4),
                            "actual_volume_change_from_phase_00_ml": round(metrics["reference_volume_ml"] - v0, 4)})
            rows.append(metrics)
        output["subjects"][subject] = {"phase_count": len(phases), "reference_volume_ml": round(v0, 4),
                                        "reference_nominal_rr_percent": phase_values["phase_00"],
                                        "reference_mask_sha256": sha256(phases[0][1]),
                                        "reference_ct_sha256": sha256(phases[0][2]),
                                        "per_phase": rows, "mean_dice": round(float(np.mean([r["dice"] for r in rows])), 6),
                                        "mean_centroid_error_mm": round(float(np.mean([r["centroid_error_mm"] for r in rows])), 4)}
    return output


def normalized_shape(path: Path, width: int = 64, spacing_mm: float = 3.5) -> np.ndarray:
    im = image(path)
    binary = mask(im)
    _, center = geometry(binary, im.affine)
    axis = (np.arange(width) - (width - 1) / 2) * spacing_mm
    grid = np.meshgrid(axis, axis, axis, indexing="ij")
    world = np.stack([g + center[k] for k, g in enumerate(grid)], axis=0).reshape(3, -1)
    ijk = np.linalg.solve(im.affine[:3, :3], world - im.affine[:3, 3, None])
    # Nearest-neighbour sampling preserves the binary definition of the mask.
    sampled = ndi.map_coordinates(binary.astype(np.uint8), ijk, order=0, mode="constant", cval=0)
    sampled = sampled.reshape((width,) * 3).astype(bool)
    if sampled.sum() < 100 or np.any(sampled[[0, -1], :, :]) or np.any(sampled[:, [0, -1], :]) or np.any(sampled[:, :, [0, -1]]):
        raise ValueError("Centered heart missing or clipped on normalized grid")
    return sampled


def signed_distance(binary: np.ndarray, spacing_mm: float = 3.5) -> np.ndarray:
    return (ndi.distance_transform_edt(~binary) - ndi.distance_transform_edt(binary)).astype(np.float32) * spacing_mm


def pca(root: Path, subjects: tuple[str, ...]) -> dict:
    if len(set(subjects)) != len(subjects) or len(subjects) < 3:
        raise ValueError("Require >=3 distinct subject IDs for leave-one-subject-out pilot")
    # Only phase_00 of each person enters this static shape experiment.
    reference_paths = {s: phase_inputs(root, s)[0][1] for s in subjects}
    shapes = {s: normalized_shape(reference_paths[s]) for s in subjects}
    sdf = {s: signed_distance(shapes[s]).ravel() for s in subjects}
    rows = []
    for heldout in subjects:
        train = [s for s in subjects if s != heldout]
        if heldout in train:
            raise AssertionError("Subject leakage")
        X = np.stack([sdf[s] for s in train])
        mean = X.mean(axis=0)
        _, singular, vt = np.linalg.svd(X - mean, full_matrices=False)
        rank = min(len(train) - 1, int(np.sum(singular > max(float(singular[0]) * 1e-6, 1e-5))))
        target = sdf[heldout]
        for k in (0, min(1, rank)):
            basis = vt[:k]
            reconstruction = mean + basis.T @ (basis @ (target - mean))
            predicted = reconstruction.reshape(shapes[heldout].shape) < 0
            comparison = compare(predicted, shapes[heldout], np.diag([3.5, 3.5, 3.5, 1.0]))
            rows.append({"heldout_subject": heldout, "training_subjects": train,
                         "components": k, "training_rank": rank, **comparison})
    return {"method": "centered_signed_distance_PCA_phase_00_LOSO",
            "grid": "64^3, 3.5 mm isotropic, centered on reference heart",
            "status": "exploratory_hypothesis_not_validated", "rows": rows,
            "reference_mask_sha256": {s: sha256(reference_paths[s]) for s in subjects},
            "interpretation": "Compression of an available mask; projection uses the held-out subject's phase_00 mask. No predicted motion or FEM validation."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("baseline", "pca"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", choices=SUBJECTS, default=list(SUBJECTS))
    args = parser.parse_args()
    subjects = tuple(args.subjects)
    if len(set(subjects)) != len(subjects):
        parser.error("Duplicate subject ID")
    result = baseline(args.data_root, subjects) if args.task == "baseline" else pca(args.data_root, subjects)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
