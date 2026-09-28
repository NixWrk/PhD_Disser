"""Test Tikhomirov's 2D circle-fit objectives on sealed historical 4D CT.

This is an explicitly adapted *geometric* experiment. The historical MRI
landmarks were defined by five electrode axes; uniformly sampled CT silhouettes
are not those landmarks. No electrical hodograph is reconstructed here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.measure import find_contours
from skimage.morphology import convex_hull_image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "Tikhomirov" / "cardio-model-py" / "src"))
from cardio_model.sphere_fit import equal_sphere  # noqa: E402

CHAMBERS = ("heart_ventricle_left", "heart_ventricle_right",
            "heart_atrium_left", "heart_atrium_right")
METHODS = (("LS128", "Use_Contour", 128),
           ("Area128", "Use_Contour_MassCenter", 128),
           ("FixedCenter128", "Use_Contour_Radius", 128),
           ("LS7_uniform", "Use_Contour", 7))
PLANES = ("sagittal", "frontal")
SCOPES = ("blood_four_chambers", "ventricles")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_phase(case: dict, phase: dict, status: dict):
    sequence, masks = Path(case["sequence_dir"]), Path(case["masks_dir"])
    pid = phase["phase_id"]
    ct_path = sequence / phase["nifti"]
    if digest(ct_path) != phase["sha256"]:
        raise ValueError(f"CT hash changed: {case['case_id']} {pid}")
    ct = nib.load(str(ct_path))
    if not np.allclose(ct.affine, phase["affine_ras_mm"], atol=1e-5):
        raise ValueError("CT affine changed")
    recorded = {key.replace("\\", "/"): value
                for key, value in status["phase_hashes"][pid].items()}
    arrays = {}
    hashes = {}
    for chamber in CHAMBERS:
        rel = f"automatic_masks_v2/{pid}/heartchambers_highres/{chamber}.nii.gz"
        path = masks / rel
        sha = digest(path)
        if sha != recorded.get(rel):
            raise ValueError(f"Mask hash changed: {rel}")
        image = nib.load(str(path))
        if image.shape != ct.shape or not np.allclose(image.affine, ct.affine, atol=1e-4):
            raise ValueError(f"Mask grid differs from CT: {rel}")
        arrays[chamber] = np.asanyarray(image.dataobj) > 0
        hashes[rel] = sha
    blood = np.logical_or.reduce(list(arrays.values()))
    ventricles = arrays[CHAMBERS[0]] | arrays[CHAMBERS[1]]
    if not ventricles.any():
        raise ValueError("Empty ventricular mask")
    idx = np.where(blood)
    lower = np.array([int(np.min(v)) for v in idx])
    upper = np.array([int(np.max(v)) + 1 for v in idx])
    sl = tuple(slice(lo, hi) for lo, hi in zip(lower, upper))
    affine = ct.affine.copy()
    affine[:3, 3] = ct.affine[:3, :3] @ lower + ct.affine[:3, 3]
    return {"blood_four_chambers": blood[sl].copy(),
            "ventricles": ventricles[sl].copy()}, affine, hashes


def voxel_centroid(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    index = np.argwhere(mask).mean(axis=0)
    return (affine @ np.r_[index, 1.0])[:3]


def planar_centroid(center: np.ndarray, plane: str) -> np.ndarray:
    # Local 2D coordinates for the chosen CT projection. They are not
    # identified with the 3D axis names in the dissertation or RFBR.
    return center[[2, 1]] if plane == "sagittal" else center[[0, 2]]


def silhouette(mask: np.ndarray, affine: np.ndarray, plane: str):
    canonical = nib.as_closest_canonical(nib.Nifti1Image(mask.astype(np.uint8), affine))
    axes = nib.aff2axcodes(canonical.affine)
    if axes != ("R", "A", "S"):
        raise ValueError(f"Unexpected canonical orientation: {axes}")
    volume = np.asanyarray(canonical.dataobj).astype(bool)
    image = volume.any(axis=0 if plane == "sagittal" else 1)
    hull = convex_hull_image(image)
    contours = find_contours(np.pad(hull, 1), 0.5)
    if not contours:
        raise ValueError("No projected outer contour")
    pixels = max(contours, key=len) - 1.0
    if plane == "sagittal":
        ijk = np.c_[np.zeros(len(pixels)), pixels[:, 0], pixels[:, 1]]
    else:
        ijk = np.c_[pixels[:, 0], np.zeros(len(pixels)), pixels[:, 1]]
    world = nib.affines.apply_affine(canonical.affine, ijk)
    contour = planar_centroid(world.T, plane).T
    # A deterministic start controls the otherwise arbitrary 7-point sample.
    start = int(np.argmax(contour[:, 0] if plane == "sagittal" else contour[:, 1]))
    contour = np.roll(contour, -start, axis=0)
    return contour, int(image.sum()), int(hull.sum())


def equiarc(points: np.ndarray, count: int) -> np.ndarray:
    closed = np.vstack([points, points[0]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(lengths)]
    distance = np.linspace(0, arc[-1], count, endpoint=False)
    return np.c_[np.interp(distance, arc, closed[:, 0]),
                 np.interp(distance, arc, closed[:, 1])]


def run(config: Path, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cases = read_json(config)["cases"]
    protocol = {
        "schema": "tikhomirov_ct_contour_adaptation_v1",
        "source_code_sha256": digest(Path(__file__)),
        "port_sha256": digest(ROOT / "Tikhomirov/cardio-model-py/src/cardio_model/sphere_fit.py"),
        "config_sha256": digest(config),
        "target": "TotalSegmentator v2 union of four cardiac chambers, and ventricles separately",
        "contour": "RAS+ sagittal or frontal convex-hull projection; 7 or 128 equally spaced outer-boundary points",
        "sagittal_coordinates": "(RAS superior, RAS anterior), in mm",
        "frontal_coordinates": "(RAS right, RAS superior), in mm",
        "solver": "existing cardio-model-py EqualSphere port, with its original local-coordinate bounds",
        "status": "geometric CT adaptation; not exact electrode-linked MRI landmark procedure",
        "electrical_inverse_run": False,
        "rfbr_subject_pairing_confirmed": False,
        "interpretation": "Differences from 3D mask centroid are between geometric centre definitions, not accuracy against an independent truth",
        "cases": [case["case_id"] for case in cases],
    }
    rows = []
    for case in cases:
        sequence, masks = Path(case["sequence_dir"]), Path(case["masks_dir"])
        manifest = read_json(sequence / "conversion_manifest.json")
        status_path = masks / "status/run_status.json"
        if not status_path.exists():
            status_path = masks / "run_status.json"
        status = read_json(status_path)
        phases = sorted(manifest["phases"], key=lambda p: p["phase_percent"])
        if (status["status"] != "completed" or len(phases) != 20 or
                [p["phase_percent"] for p in phases] != list(range(0, 100, 5))):
            raise ValueError(f"Incomplete or differently phased case: {case['case_id']}")
        ref_centers = {}
        for phase in phases:
            masks_by_scope, affine, hashes = load_phase(case, phase, status)
            for scope in SCOPES:
                mask = masks_by_scope[scope]
                ct_center = voxel_centroid(mask, affine)
                for plane in PLANES:
                    ct2 = planar_centroid(ct_center, plane)
                    ref_centers.setdefault((scope, plane), ct2.copy())
                    origin = ref_centers[(scope, plane)]
                    outer, projection_pixels, hull_pixels = silhouette(mask, affine, plane)
                    for name, method, count in METHODS:
                        pts = equiarc(outer, count) - origin
                        fit_center, radius = equal_sphere(pts, method)
                        center = fit_center + origin
                        rows.append({
                            "case": case["case_id"], "phase": phase["phase_id"],
                            "phase_percent": phase["phase_percent"], "scope": scope,
                            "plane": plane, "method": name, "sample_count": count,
                            "center_1_mm": float(center[0]), "center_2_mm": float(center[1]),
                            "ct_centroid_1_mm": float(ct2[0]), "ct_centroid_2_mm": float(ct2[1]),
                            "radius_mm": float(radius),
                            "projection_pixels": projection_pixels, "hull_pixels": hull_pixels,
                            "hull_area_ratio": float(hull_pixels / projection_pixels),
                            "radius_at_original_upper_bound": bool(method == "Use_Contour" and radius >= 59.99),
                        })
            print(f"{case['case_id']} {phase['phase_id']} done", flush=True)
    with (output / "phase_fits.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    keys = sorted({(r["case"], r["scope"], r["plane"], r["method"]) for r in rows})
    for key in keys:
        group = sorted((r for r in rows if
                        (r["case"], r["scope"], r["plane"], r["method"]) == key),
                       key=lambda r: r["phase_percent"])
        fit = np.array([[r["center_1_mm"], r["center_2_mm"]] for r in group])
        ct = np.array([[r["ct_centroid_1_mm"], r["ct_centroid_2_mm"]] for r in group])
        diff = (fit - fit[0]) - (ct - ct[0])
        summary.append({"case": key[0], "scope": key[1], "plane": key[2], "method": key[3],
                        "rms_displacement_difference_mm": float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))),
                        "max_displacement_difference_mm": float(np.max(np.linalg.norm(diff, axis=1))),
                        "median_center_definition_offset_mm": float(np.median(np.linalg.norm(fit - ct, axis=1))),
                        "fit_max_excursion_mm": float(np.max(np.linalg.norm(fit - fit[0], axis=1))),
                        "ct_max_excursion_mm": float(np.max(np.linalg.norm(ct - ct[0], axis=1))),
                        "median_radius_mm": float(np.median([r["radius_mm"] for r in group])),
                        "radius_range_mm": float(np.ptp([r["radius_mm"] for r in group])),
                        "original_radius_bound_hits": int(sum(r["radius_at_original_upper_bound"] for r in group)),
                        "max_hull_area_ratio": float(max(r["hull_area_ratio"] for r in group)),
                        "phases": len(group)})
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (output / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"phase_fits": len(rows), "summary_rows": len(summary), "output": str(output)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(run(args.config, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
