"""Retrospective MP-08 resolution holdout on real Nix 4D CT (no patient output in Git).

The native target phase is used to make a blurred 2x lower-resolution input,
discarded before registration, and reloaded only for evaluation. The high-res
reference is a different real phase. This tests information transfer from the
reference, not true super-resolution against a separate acquisition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import date
from importlib.metadata import version
from pathlib import Path

import itk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from monai_physio import RegisterImagesGreedy, WorkflowReconstructHighres4DCT
from scipy import ndimage as ndi


F = np.diag([-1.0, -1.0, 1.0])  # NIfTI RAS -> ITK LPS
ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def itk_image(array_xyz: np.ndarray, affine: np.ndarray) -> itk.Image:
    image = itk.image_from_array(np.ascontiguousarray(array_xyz.transpose(2, 1, 0)))
    axes = F @ affine[:3, :3]
    spacing = np.linalg.norm(axes, axis=0)
    image.SetSpacing(tuple(float(v) for v in spacing))
    image.SetOrigin(tuple(float(v) for v in F @ affine[:3, 3]))
    image.SetDirection(itk.GetMatrixFromArray(axes / spacing))
    return image


def crop_affine(affine: np.ndarray, low: np.ndarray, stride: int = 1) -> np.ndarray:
    out = affine.copy()
    out[:3, 3] += affine[:3, :3] @ low
    out[:3, :3] = affine[:3, :3] * stride
    return out


def read_crop(path: Path, sl: tuple[slice, ...], reference: nib.Nifti1Image) -> np.ndarray:
    image = nib.load(str(path))
    if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-4):
        raise ValueError("CT/mask geometry mismatch")
    return np.asarray(image.dataobj[sl], dtype=np.float32)


def resample_to(image: itk.Image, reference: itk.Image) -> np.ndarray:
    transform = itk.IdentityTransform[itk.D, 3].New()
    result = itk.resample_image_filter(
        image, transform=transform, use_reference_image=True,
        reference_image=reference, default_pixel_value=-1024.0,
    )
    return np.ascontiguousarray(itk.array_view_from_image(result).transpose(2, 1, 0))


def scores(pred: np.ndarray, truth: np.ndarray, roi: np.ndarray) -> dict:
    a = pred[roi].astype(np.float64)
    b = truth[roi].astype(np.float64)
    delta = a - b
    return {"voxels": int(a.size), "mae_hu": float(np.mean(np.abs(delta))),
            "rmse_hu": float(np.sqrt(np.mean(delta ** 2))),
            "bias_hu": float(np.mean(delta)),
            "p95_absolute_error_hu": float(np.percentile(np.abs(delta), 95))}


def plot_review(truth: np.ndarray, baseline: np.ndarray, reconstructed: np.ndarray,
                mask: np.ndarray, path: Path, target: str) -> None:
    loc = np.argwhere(mask).mean(axis=0).astype(int)
    z = int(np.clip(loc[2], 0, truth.shape[2] - 1))
    panels = [truth[:, :, z], baseline[:, :, z], reconstructed[:, :, z],
              reconstructed[:, :, z] - truth[:, :, z]]
    titles = [f"{target}: удержанная КТ", "Линейная интерполяция lowres",
              "MONAI MP-08", "MONAI − КТ, HU"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel.T, origin="lower", cmap="gray" if "HU" not in title else "coolwarm",
                  vmin=-150 if "HU" not in title else -300,
                  vmax=350 if "HU" not in title else 300)
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-dir", type=Path, required=True)
    parser.add_argument("--products-dir", type=Path, required=True)
    parser.add_argument("--target", choices=[f"phase_{i:02d}" for i in range(1, 12)], default="phase_07")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "mp08_nix_real_holdout")
    args = parser.parse_args()
    if not args.output_dir.resolve().is_relative_to((ROOT / "runs").resolve()):
        parser.error("Patient outputs must remain in monai_physio/runs/")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.phase_dir / "conversion_manifest.json").read_text(encoding="utf-8"))
    entries = {e["nifti"]: e for e in manifest["phases"]}
    reference_path = args.phase_dir / "phase_00.nii.gz"
    target_path = args.phase_dir / f"{args.target}.nii.gz"
    mask_path = args.products_dir / "phase_00" / "whole_heart.nii.gz"
    ref = nib.load(str(reference_path))
    if ref.shape != (512, 512, 256) or not np.allclose(
        np.linalg.norm(ref.affine[:3, :3], axis=0),
        entries[reference_path.name]["voxel_spacing_mm"], atol=1e-3,
    ) or entries[target_path.name]["affine_coordinate_system"] != "NIfTI RAS+":
        raise ValueError("Unexpected Nix CT conversion geometry")
    mask_full = nib.load(str(mask_path))
    if mask_full.shape != ref.shape or not np.allclose(mask_full.affine, ref.affine, atol=1e-4):
        raise ValueError("Reference mask geometry mismatch")
    idx = np.argwhere(np.asarray(mask_full.dataobj) > 0)
    spacing = np.linalg.norm(ref.affine[:3, :3], axis=0)
    margin = np.ceil(25.0 / spacing).astype(int)
    low = np.maximum(idx.min(axis=0) - margin, 0)
    high = np.minimum(idx.max(axis=0) + 1 + margin, ref.shape)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
    affine = crop_affine(ref.affine, low)
    ref_array = read_crop(reference_path, sl, ref)
    roi = read_crop(mask_path, sl, ref) > 0
    # Real target is read once for degradation. Its original pixels are not
    # passed to the workflow; it is reloaded only after reconstruction.
    target_native = read_crop(target_path, sl, ref)
    target_low = ndi.gaussian_filter(target_native, sigma=0.8)[::2, ::2, ::2].copy()
    del target_native
    ref_low = ndi.gaussian_filter(ref_array, sigma=0.8)[::2, ::2, ::2].copy()
    affine_low = crop_affine(affine, np.zeros(3, dtype=int), stride=2)
    fixed = itk_image(ref_array, affine)
    input_images = [itk_image(ref_low, affine_low), itk_image(target_low, affine_low)]
    registrar = RegisterImagesGreedy(log_level=logging.ERROR)
    registrar.set_modality("ct")
    registrar.set_metric("CC")
    registrar.set_number_of_iterations([12, 6, 3])
    registrar.deformable_smoothing = "3.0vox 1.0vox"
    workflow = WorkflowReconstructHighres4DCT(
        time_series_images=input_images, reference_image=fixed,
        reference_time_frame=0, register_reference_time_frame_to_reference_image=False,
        registration_method=registrar, log_level=logging.ERROR,
    )
    workflow.set_modality("ct")
    workflow.set_composite_mode("reference")
    output = workflow.process()
    reconstructed = resample_to(output["reconstructed_images"][1], fixed)
    baseline = resample_to(input_images[1], fixed)
    truth = read_crop(target_path, sl, ref)
    stationary = ref_array
    metrics = {name: scores(image, truth, roi) for name, image in
               [("stationary_reference", stationary), ("linear_lowres", baseline),
                ("monai_mp08", reconstructed)]}
    ratio = metrics["monai_mp08"]["mae_hu"] / metrics["linear_lowres"]["mae_hu"]
    review = args.output_dir / f"mp08_{args.target}_review.png"
    plot_review(truth, baseline, reconstructed, roi, review, args.target)
    nii = nib.Nifti1Image(reconstructed.astype(np.float32), affine)
    nii.header.set_xyzt_units("mm")
    output_nifti = args.output_dir / f"mp08_{args.target}_reconstructed.nii.gz"
    nib.save(nii, str(output_nifti))
    report = {
        "schema": "mp08_nix_real_holdout_v1", "status": "real_ct_retrospective_resolution_holdout_completed",
        "run_date": date.today().isoformat(),
        "implementation_sha256": sha256(Path(__file__)),
        "subject_pseudonym": "nix", "reference_phase": "phase_00", "target_phase": args.target,
        "input_sha256": {"reference_ct": sha256(reference_path), "target_ct": sha256(target_path),
                         "reference_mask": sha256(mask_path)},
        "input_kind": "real cardiac 4D CT; target deliberately blurred and decimated 2x for fitting",
        "target_native_used_for_fitting": False,
        "independent_high_resolution_acquisition": False,
        "crop_from": "phase_00 automatic whole_heart mask plus 25 mm; no target mask",
        "crop_start_ijk": low.tolist(), "crop_stop_exclusive_ijk": high.tolist(),
        "native_shape_ijk": list(ref_array.shape), "native_spacing_mm": spacing.tolist(),
        "lowres_shape_ijk": list(target_low.shape), "lowres_spacing_mm": (2 * spacing).tolist(),
        "nifti_header_spatial_units": ref.header.get_xyzt_units()[0],
        "millimetres_basis": "external DICOM conversion manifest",
        "software": {"monai_physio": version("monai-physio"), "itk": version("itk")},
        "parameter_selection_history": "Greedy schedule/smoothing reused from earlier MP-03 Nix work; phase 07 was previously inspected there. No tuning during this MP-08 run, but not a completely blind protocol.",
        "workflow": {"api": "WorkflowReconstructHighres4DCT.process",
                     "registrar": "RegisterImagesGreedy CC deformable",
                     "iterations": [12, 6, 3], "smoothing": "3.0vox 1.0vox",
                     "composite_mode": "reference", "upsample_to_fixed_resolution": True,
                     "output_size_xyz": list(map(int, itk.size(output["reconstructed_images"][1]))),
                     "output_spacing_xyz_mm": list(map(float, itk.spacing(output["reconstructed_images"][1]))),
                     "registration_loss": float(output["losses"][1])},
        "metric_region": "phase_00 whole_heart voxels within fixed 25 mm crop",
        "metrics_hu": metrics, "mae_ratio_monai_over_linear": float(ratio),
        "better_than_linear_on_heart_roi": bool(ratio < 1),
        "local_artifacts": [review.name, output_nifti.name],
        "interpretation": "Retrospective resolution holdout on real CT; no independent highres scan, anatomical ground truth, or clinical validation",
    }
    output_json = ROOT / f"mp08_nix_real_holdout_{args.target}.json"
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "target": args.target,
                      "metrics_hu": metrics, "mae_ratio_monai_over_linear": ratio,
                      "report": output_json.name}, ensure_ascii=False))


if __name__ == "__main__":
    main()
