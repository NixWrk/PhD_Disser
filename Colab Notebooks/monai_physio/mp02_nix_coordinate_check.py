"""Independent-of-heart-mask bone HU check of the existing Nix CT/FEM axes.

This is a cross-modality surface discrepancy, not an independently measured
coordinate-registration error bound. Only aggregate distances enter Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import trimesh
import pydicom


HERE = Path(__file__).resolve().parent
PHASES = (0, 1, 3, 7, 10)
RAS_TO_FEM = np.diag([-1., -1., 1.])


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-root", type=Path, required=True)
    parser.add_argument("--static-bones-stl", type=Path, required=True)
    parser.add_argument("--dicom-root", type=Path, required=True)
    args = parser.parse_args()
    dicom_series = {301: {"z": [], "frame_hashes": set()},
                    307: {"z": [], "frame_hashes": set()}}
    for path in args.dicom_root.rglob("*.dcm"):
        dataset = pydicom.dcmread(str(path), stop_before_pixels=True,
                                 specific_tags=["SeriesNumber", "FrameOfReferenceUID", "ImagePositionPatient"])
        number = int(dataset.get("SeriesNumber", -1))
        if number in dicom_series:
            dicom_series[number]["z"].append(float(dataset.ImagePositionPatient[2]))
            dicom_series[number]["frame_hashes"].add(
                hashlib.sha256(str(dataset.FrameOfReferenceUID).encode()).hexdigest())
    if (len(dicom_series[301]["frame_hashes"]) != 1 or
            dicom_series[301]["frame_hashes"] != dicom_series[307]["frame_hashes"] or
            len(dicom_series[301]["z"]) != 241 or len(dicom_series[307]["z"]) != 3072):
        raise ValueError("Static/dynamic DICOM provenance differs from established Nix series")
    fov = {str(number): {"dicom_instances": len(values["z"]),
                          "unique_slice_centres": len(set(values["z"])),
                          "z_centres_lps_mm_min_max": [min(values["z"]), max(values["z"])]}
           for number, values in dicom_series.items()}
    mesh = trimesh.load(str(args.static_bones_stl), process=False)
    vertices = np.unique(np.asarray(mesh.vertices), axis=0)[::5]
    vertices = vertices[(np.abs(vertices[:, 0]) < 130) &
                        (vertices[:, 2] > -220) & (vertices[:, 2] < -95)]
    if len(vertices) < 1000:
        raise ValueError("Too few static bone surface points")
    rows = []
    for index in PHASES:
        path = args.subject_root / "phases" / f"phase_{index:02d}.nii.gz"
        image = nib.load(str(path))
        if image.shape != (512, 512, 256):
            raise ValueError("Unexpected Nix CT shape")
        stride = np.array([2, 2, 2])
        hu = np.asarray(image.dataobj[::2, ::2, ::2], dtype=np.float32)
        bone = hu > 250
        boundary = bone & ~ndi.binary_erosion(bone)
        idx = np.argwhere(boundary) * stride
        points = nib.affines.apply_affine(image.affine, idx) @ RAS_TO_FEM.T
        keep = ((np.abs(points[:, 0]) < 140) & (points[:, 2] > -225) &
                (points[:, 2] < -90))
        points = points[keep]
        if len(points) < 1000:
            raise ValueError("Too few CT bone HU boundary points")
        distances = cKDTree(points).query(vertices, workers=-1)[0]
        bands = []
        for lower, upper in ((-220, -180), (-180, -140), (-140, -95)):
            subset = distances[(vertices[:, 2] >= lower) & (vertices[:, 2] < upper)]
            bands.append({"z_fem_mm": [lower, upper], "static_samples": int(len(subset)),
                          "distance_p50_p95_mm": np.percentile(subset, [50, 95]).tolist()})
        rows.append({"phase": f"phase_{index:02d}", "ct_sha256": sha(path),
                     "ct_hu_boundary_points": int(len(points)),
                     "static_surface_samples": int(len(vertices)),
                     "distance_p50_p90_p95_mm": np.percentile(distances, [50, 90, 95]).tolist(),
                     "z_bands": bands})
        print(f"phase_{index:02d}: median {np.median(distances):.2f} mm, p95 {np.percentile(distances,95):.2f} mm", flush=True)
    summary = {"schema": "mp02_nix_coordinate_check_v1", "subject_alias": "nix",
               "static_dynamic_same_dicom_frame": True,
               "dicom_fov": fov,
               "static_superior_extent_beyond_dynamic_mm":
                   fov["301"]["z_centres_lps_mm_min_max"][1] - fov["307"]["z_centres_lps_mm_min_max"][1],
               "static_bones_stl_sha256": sha(args.static_bones_stl),
               "fixed_coordinate_map_ras_to_fem_mm": np.diag([-1., -1., 1., 1.]).tolist(),
               "hu_threshold": 250, "display_sampling_stride_ijk": [2, 2, 2],
               "no_heart_mask_used": True, "no_optimized_registration": True,
               "interpretation": "Static bone STL to phase CT high-HU boundary distance; mixes contour, threshold, breathing, and position errors; not an independent registration error bound",
               "rows": rows}
    (HERE / "mp02_nix_coordinate_check.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
