"""Read-only Georg DICOM frame and extracardiac body-alignment audit.

The STL lacks DICOM frame metadata. This script can support a candidate RAS->LPS
map but cannot accept it or validate a moving heart for FEM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
import trimesh
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


def dicom_series_metadata(directory: Path) -> dict:
    files = sorted(directory.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(directory)
    frames: set[str] = set()
    orientations: set[tuple[float, ...]] = set()
    series_numbers: set[int] = set()
    for path in files:
        ds = pydicom.dcmread(path, stop_before_pixels=True,
                            specific_tags=["FrameOfReferenceUID", "SeriesNumber",
                                           "ImageOrientationPatient"])
        frames.add(hashlib.sha256(str(ds.FrameOfReferenceUID).encode()).hexdigest())
        orientations.add(tuple(float(x) for x in ds.ImageOrientationPatient))
        series_numbers.add(int(ds.SeriesNumber))
    return {"file_count": len(files), "frame_uid_sha256": sorted(frames),
            "orientations_lps": [list(o) for o in sorted(orientations)],
            "series_numbers": sorted(series_numbers)}


def body_surface_agreement(ct_path: Path, stl_path: Path, stride: int = 2) -> dict:
    image = nib.load(str(ct_path))
    ct = np.asarray(image.dataobj[::stride, ::stride, ::stride], dtype=np.float32)
    # Simple HU body proxy, used only for a coarse extracardiac coordinate audit.
    body = ct > -350
    labels, number = ndi.label(body)
    if number < 1:
        raise ValueError("No connected body component in thresholded CT")
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    body = labels == int(np.argmax(counts))
    del labels
    surface = body & ~ndi.binary_erosion(body)
    surface_idx = np.argwhere(surface)
    del body, surface, ct
    # The first/last slices and all lateral crop caps are not anatomical body
    # boundaries. Exclude them from both point sets.
    margin_vox = max(3, int(np.ceil(8 / (stride * np.min(image.header.get_zooms()[:3])))))
    shape = np.asarray(image.shape) // stride
    valid_surface = np.all((surface_idx > margin_vox) &
                           (surface_idx < shape - margin_vox), axis=1)
    surface_idx = surface_idx[valid_surface][::3]
    ras = nib.affines.apply_affine(image.affine, surface_idx * stride)
    lps = ras * [-1, -1, 1]
    tree = cKDTree(lps)
    mesh = trimesh.load(str(stl_path), process=False)
    vertices = np.asarray(mesh.vertices)[::5]
    idx_ras = nib.affines.apply_affine(np.linalg.inv(image.affine), vertices * [-1, -1, 1]) / stride
    valid_vertices = np.all((idx_ras > margin_vox) &
                            (idx_ras < shape - margin_vox), axis=1)
    vertices = vertices[valid_vertices]
    if len(vertices) < 100 or len(lps) < 100:
        raise ValueError("Too few overlapping extracardiac surface samples")
    aligned = tree.query(vertices, workers=-1)[0]
    identity = tree.query(vertices * [-1, -1, 1], workers=-1)[0]
    return {"method": "coarse_CT_HU_threshold_body_surface_to_static_STL_vertices",
            "threshold_hu": -350, "downsample_stride": stride,
            "surface_sample_count": int(len(lps)),
            "stl_vertex_sample_count": int(len(vertices)),
            "ras_to_lps_candidate_distance_p50_p90_p95_mm":
                [float(v) for v in np.percentile(aligned, [50, 90, 95])],
            "identity_axis_sanity_distance_p50_p90_p95_mm":
                [float(v) for v in np.percentile(identity, [50, 90, 95])],
            "ct_lps_bounds_mm": [
                [float(v) for v in np.min(nib.affines.apply_affine(image.affine,
                    np.array([[0, 0, 0], np.array(image.shape) - 1])) * [-1, -1, 1], axis=0)],
                [float(v) for v in np.max(nib.affines.apply_affine(image.affine,
                    np.array([[0, 0, 0], np.array(image.shape) - 1])) * [-1, -1, 1], axis=0)]],
            "static_stl_bounds_mm": mesh.bounds.tolist(),
            "limitations": ["CT threshold is not an independent body annotation",
                            "static and 4D acquisitions can differ in breathing, pose and FOV",
                            "STL has no DICOM frame metadata; this is only a candidate map"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dicom-301", type=Path, required=True)
    parser.add_argument("--dicom-309", type=Path, required=True)
    parser.add_argument("--ct-phase-00", type=Path, required=True)
    parser.add_argument("--body-stl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    runs = (Path(__file__).resolve().parent / "runs").resolve()
    if not args.output_json.resolve().is_relative_to(runs):
        parser.error("output-json must be within local ignored runs/")
    series = {"301": dicom_series_metadata(args.dicom_301),
              "309": dicom_series_metadata(args.dicom_309)}
    common = (len(series["301"]["frame_uid_sha256"]) == 1 and
              series["301"]["frame_uid_sha256"] == series["309"]["frame_uid_sha256"])
    result = {"schema": "mp02_georg_coordinate_audit_v1",
              "same_dicom_patient_frame_301_309": common,
              "series": series,
              "candidate_transform_ras_mm_to_static_stl_lps_mm":
                  np.diag([-1.0, -1.0, 1.0, 1.0]).tolist(),
              "body_surface": body_surface_agreement(args.ct_phase_00, args.body_stl),
              "status": "candidate_not_accepted_for_fem"}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"same_dicom_patient_frame_301_309": common,
                      "body_surface": result["body_surface"],
                      "status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
