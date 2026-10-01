"""Compare one model scenario with a held-out public inhale/exhale CT pair.

The stage correspondence is an explicit unverified assumption. This checks
gross lung and heart geometry; it does not turn automatic masks into truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pyvista as pv
from scipy.ndimage import center_of_mass


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def mask_geometry(path: Path) -> dict[str, object]:
    image = nib.load(path)
    mask = np.asarray(image.dataobj) > 0
    if not np.any(mask):
        raise ValueError(f"Empty mask: {path}")
    centroid = nib.affines.apply_affine(image.affine, np.asarray(center_of_mass(mask)))
    touches = any(
        np.any(np.take(mask, index, axis=axis))
        for axis in range(3)
        for index in (0, mask.shape[axis] - 1)
    )
    return {
        "volume_ml": float(np.count_nonzero(mask) * abs(np.linalg.det(image.affine[:3, :3])) / 1000),
        "centroid_ras_mm": [float(value) for value in centroid],
        "touches_image_boundary": bool(touches),
        "shape_xyz": list(image.shape),
        "affine": image.affine.tolist(),
    }


def difference(a: list[float], b: list[float]) -> list[float]:
    return [float(y - x) for x, y in zip(a, b)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lung-mask", type=Path, required=True)
    parser.add_argument("--target-lung-mask", type=Path, required=True)
    parser.add_argument("--source-heart-mask", type=Path, required=True)
    parser.add_argument("--target-heart-mask", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--scenario-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alias", required=True)
    args = parser.parse_args()

    lung_source = mask_geometry(args.source_lung_mask)
    lung_target = mask_geometry(args.target_lung_mask)
    heart_source = mask_geometry(args.source_heart_mask)
    heart_target = mask_geometry(args.target_heart_mask)
    for source, target, organ in ((lung_source, lung_target, "lung"),
                                  (heart_source, heart_target, "heart")):
        if source["shape_xyz"] != target["shape_xyz"] or not np.allclose(
            source["affine"], target["affine"]
        ):
            raise ValueError(f"{organ} pair is not on a common physical grid")
        if source["touches_image_boundary"] or target["touches_image_boundary"]:
            raise ValueError(f"{organ} segmentation touches the acquired boundary")

    scenarios = json.loads((args.scenario_dir / "scenario_report.json").read_text(encoding="utf-8"))
    fit = json.loads((args.fit_dir / "fit_report.json").read_text(encoding="utf-8"))
    pred = {float(row["stage"]): row for row in scenarios["stages"]}
    if not {0.0, 0.7}.issubset(pred):
        raise ValueError("Required model stages 0.0 and 0.7 are absent")
    source_lung_mesh = pv.read(str(args.scenario_dir / "lung_stage_0.70.vtp"))
    target_lung_mesh = pv.read(str(args.scenario_dir / "lung_stage_0.00.vtp"))

    observed_lung_change = lung_target["volume_ml"] - lung_source["volume_ml"]
    observed_heart_shift_ras = difference(
        heart_source["centroid_ras_mm"], heart_target["centroid_ras_mm"]
    )
    modeled_heart_shift_lps = pred[0.0]["heart"]["centroid_shift_lps_mm"]
    modeled_heart_shift_ras = [
        -modeled_heart_shift_lps[0], -modeled_heart_shift_lps[1], modeled_heart_shift_lps[2]
    ]
    report = {
        "schema": "mp10_open_pair_geometry_v1",
        "pair_alias": args.alias,
        "status": "single_pair_exploratory_check_not_model_validation",
        "assumed_stage_mapping": "inspiration=0.7, expiration=0.0; not verified",
        "source_lung_mask_sha256": sha256(args.source_lung_mask),
        "target_lung_mask_sha256": sha256(args.target_lung_mask),
        "source_heart_mask_sha256": sha256(args.source_heart_mask),
        "target_heart_mask_sha256": sha256(args.target_heart_mask),
        "fit_report_sha256": sha256(args.fit_dir / "fit_report.json"),
        "scenario_report_sha256": sha256(args.scenario_dir / "scenario_report.json"),
        "fit_nearest_vertex_mean_mm_stride8": fit["nearest_vertex_mean_mm_stride8"],
        "lung_mask_source": lung_source,
        "lung_mask_target": lung_target,
        "heart_mask_source": heart_source,
        "heart_mask_target": heart_target,
        "observed_lung_volume_change_ml": float(observed_lung_change),
        "lung_mesh_open_edges_source": int(source_lung_mesh.n_open_edges),
        "lung_mesh_open_edges_stage_0": int(target_lung_mesh.n_open_edges),
        "modeled_lung_volume_status": "not_evaluated_non_watertight_mesh",
        "modeled_lung_vertex_displacement_mean_mm": pred[0.0]["lung"]["vertex_displacement_mean_mm"],
        "landmarks_stage_0": pred[0.0].get("landmarks"),
        "observed_heart_centroid_shift_ras_mm": observed_heart_shift_ras,
        "observed_heart_centroid_shift_norm_mm": float(np.linalg.norm(observed_heart_shift_ras)),
        "modeled_heart_centroid_shift_ras_mm": modeled_heart_shift_ras,
        "modeled_heart_centroid_shift_norm_mm": float(np.linalg.norm(modeled_heart_shift_ras)),
        "note": "Provided lung masks and TotalSegmentator heart masks are automatic; no manual heart reference",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in (
        "fit_nearest_vertex_mean_mm_stride8",
        "observed_lung_volume_change_ml", "lung_mesh_open_edges_source",
        "modeled_lung_vertex_displacement_mean_mm", "landmarks_stage_0", "observed_heart_centroid_shift_norm_mm",
        "modeled_heart_centroid_shift_norm_mm")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
