"""Fit published MONAI Physio lung shape model to one segmented CT.

Input masks must be TotalSegmentator lobe masks on the same CT grid. The CT
itself is not used for PCA fitting after segmentation. All individual meshes,
coefficients and derived labelmaps stay in the supplied local output folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import itk
import nibabel as nib
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from monai_physio import ProcessContours, WorkflowFitStatisticalModelToPatient

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


def nearest_vertex_mean(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)[::8]
    b = np.asarray(b)[::8]
    return float((cKDTree(a).query(b)[0].mean() + cKDTree(b).query(a)[0].mean()) / 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct", type=Path, required=True)
    parser.add_argument("--seg-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alias", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ct = nib.load(args.ct)
    labels = np.zeros(ct.shape, dtype=np.uint8)
    mask_hashes: dict[str, str] = {}
    for label, name in enumerate(LOBES, start=1):
        path = args.seg_dir / f"{name}.nii.gz"
        image = nib.load(path)
        if image.shape != ct.shape or not np.allclose(image.affine, ct.affine, atol=1e-4):
            raise ValueError(f"Lobe {name} does not share the CT physical grid")
        mask = np.asarray(image.dataobj) > 0
        if not np.any(mask):
            raise ValueError(f"Lobe {name} is empty")
        if np.any(labels[mask] != 0):
            raise ValueError(f"Lobe {name} overlaps another lobe")
        labels[mask] = label
        mask_hashes[name] = sha256(path)
    if any(
        np.any(np.take(labels, index, axis=axis) > 0)
        for axis in range(3)
        for index in (0, ct.shape[axis] - 1)
    ):
        raise ValueError("The lung segmentation touches the acquired CT boundary")

    occupied = np.argwhere(labels > 0)
    lower = np.maximum(occupied.min(axis=0) - 12, 0)
    upper = np.minimum(occupied.max(axis=0) + 13, ct.shape)
    source_spacing = np.asarray(nib.affines.voxel_sizes(ct.affine))
    stride = np.maximum(1, np.floor(2.5 / source_spacing).astype(int))
    small = labels[
        lower[0] : upper[0] : stride[0],
        lower[1] : upper[1] : stride[1],
        lower[2] : upper[2] : stride[2],
    ]
    grid_map = np.eye(4)
    grid_map[:3, :3] = np.diag(stride)
    grid_map[:3, 3] = lower
    small_affine = ct.affine @ grid_map
    labelmap_path = args.output_dir / "lung_labels_fit_grid.nii.gz"
    nib.save(nib.Nifti1Image(small, small_affine), labelmap_path)
    labelmap = itk.imread(str(labelmap_path))
    ct_small = np.asarray(ct.dataobj)[
        lower[0] : upper[0] : stride[0],
        lower[1] : upper[1] : stride[1],
        lower[2] : upper[2] : stride[2],
    ]
    image_path = args.output_dir / "ct_fit_grid.nii.gz"
    nib.save(nib.Nifti1Image(ct_small.astype(np.int16), small_affine), image_path)
    patient_image = itk.imread(str(image_path))

    contours = ProcessContours().extract_label_surfaces(
        labelmap,
        isotropic_spacing_mm=2.5,
        distance_sigma_mm=2.5,
        smoothing_iterations=10,
    )
    if set(contours) != set(range(1, len(LOBES) + 1)):
        raise ValueError("A lobe vanished during coarse-grid surface extraction")
    patient_surface = pv.merge([contours[i] for i in range(1, 6)])
    patient_surface_path = args.output_dir / "lung_target_surface.vtp"
    patient_surface.save(str(patient_surface_path))

    mean_path = args.model_dir / "pca_mean_surface.vtp"
    pca_path = args.model_dir / "pca_model.json"
    mean_surface = pv.read(str(mean_path))
    with pca_path.open(encoding="utf-8") as stream:
        pca_model = json.load(stream)
    if mean_surface.n_points != 79978 or len(pca_model["components"]) < 6:
        raise ValueError("Unexpected official lung template topology or PCA dimension")

    # The CT supplies the intensity image required by the registration stage;
    # the labelmap supplies the anatomical objective. No second phase enters fit.
    fit = WorkflowFitStatisticalModelToPatient(
        template_model=mean_surface,
        patient_models=[patient_surface],
        patient_image=patient_image,
        patient_labelmap=labelmap,
    )
    fit.set_use_pca_registration(True, pca_model, 6, use_surface=False)
    fit.set_icp_transform_type("Affine")
    fit.set_mask_dilation_mm(20.0)
    fit.set_distancemap_squared_max(625.0)
    # The optional ICON refinement requires a second, uncached checkpoint.
    # Keep this run fully reproducible with the published lung model alone.
    fit.set_use_labelmap_to_labelmap_registration(False)
    result = fit.process()
    fitted = result["fitted_reference_mesh"]
    if fitted.n_points != mean_surface.n_points:
        raise ValueError("Fitted model lost the template vertex ordering")
    fitted_path = args.output_dir / "lung_fitted_reference.vtp"
    fitted.save(str(fitted_path))
    coeffs = np.asarray(fit.pca_coefficients, dtype=float)
    if coeffs.shape != (6,) or not np.isfinite(coeffs).all():
        raise ValueError("Invalid six-component lung fit")
    coeff_path = args.output_dir / "lung_pca_coefficients.json"
    coeff_path.write_text(json.dumps(coeffs.tolist()) + "\n", encoding="utf-8")

    report = {
        "schema": "mp10_lung_one_ct_fit_v1",
        "subject_alias": args.alias,
        "status": "technical_icp_pca_fit_pending_anatomical_validation",
        "registration_stages": ["affine_icp", "six_component_pca"],
        "source_ct_sha256": sha256(args.ct),
        "lobe_mask_sha256": mask_hashes,
        "pca_model_sha256": sha256(pca_path),
        "pca_mean_surface_sha256": sha256(mean_path),
        "fit_grid_shape_xyz": list(small.shape),
        "fit_grid_spacing_mm": [float(x) for x in nib.affines.voxel_sizes(small_affine)],
        "source_grid_spacing_mm": [float(x) for x in source_spacing],
        "fit_stride_xyz": [int(x) for x in stride],
        "source_lung_volume_ml": float(np.count_nonzero(labels) * np.prod(ct.header.get_zooms()[:3]) / 1000),
        "target_surface_vertices": int(patient_surface.n_points),
        "fitted_surface_vertices": int(fitted.n_points),
        "nearest_vertex_mean_mm_stride8": nearest_vertex_mean(
            patient_surface.points, fitted.points
        ),
        "fitted_surface_sha256": sha256(fitted_path),
        "coefficient_file_sha256": sha256(coeff_path),
        "target_phases_used": [],
        "note": "Surface distance is a discretization-dependent internal fit QC",
    }
    (args.output_dir / "fit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
