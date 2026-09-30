"""Local review images for a native-grid MONAI registration candidate.

The target outlines are automatic masks, not an independent anatomical reference.
All image outputs must stay in the ignored runs/ directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


F = np.diag([-1.0, -1.0, 1.0])


def plane_coordinates(shape: tuple[int, int, int], axis: int, index: int) -> np.ndarray:
    varying = [k for k in range(3) if k != axis]
    grid = np.meshgrid(
        np.arange(shape[varying[0]], dtype=np.float32),
        np.arange(shape[varying[1]], dtype=np.float32),
        indexing="ij",
    )
    coords = np.empty((*grid[0].shape, 3), dtype=np.float32)
    coords[..., axis] = index
    coords[..., varying[0]] = grid[0]
    coords[..., varying[1]] = grid[1]
    return coords


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-dir", type=Path, required=True)
    parser.add_argument("--products-dir", type=Path, required=True)
    parser.add_argument("--qc-json", type=Path, required=True)
    parser.add_argument("--field-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    runs = (Path(__file__).resolve().parent / "runs").resolve()
    output = args.output_dir.resolve()
    if not output.is_relative_to(runs) or output == runs:
        parser.error("output-dir must be a child of local ignored runs/")
    report = json.loads(args.qc_json.read_text(encoding="utf-8"))
    with args.field_npz.open("rb") as source:
        field_hash = hashlib.file_digest(source, "sha256").hexdigest()
    if field_hash != report["field_qc"]["local_field_sha256"]:
        raise ValueError("Field hash differs from registration QC")
    if report["grid_qc"]["stride_ijk"] != [1, 1, 1]:
        parser.error("review requires native-grid stride [1,1,1]")
    start = np.asarray(report["grid_qc"]["crop_start_ijk"], dtype=int)
    stop = np.asarray(report["grid_qc"]["crop_stop_ijk_exclusive"], dtype=int)
    slices = tuple(slice(int(a), int(b)) for a, b in zip(start, stop))
    source = report["moving_reference"]
    target = report["fixed_target"]
    source_nii = nib.load(str(args.phase_dir / f"{source}.nii.gz"))
    target_nii = nib.load(str(args.phase_dir / f"{target}.nii.gz"))
    affine = source_nii.affine.copy()
    affine[:3, 3] += affine[:3, :3] @ start
    source_ct = np.asarray(source_nii.dataobj[slices], dtype=np.float32)
    target_ct = np.asarray(target_nii.dataobj[slices], dtype=np.float32)
    source_mask = np.asarray(
        nib.load(str(args.products_dir / source / "whole_heart.nii.gz")).dataobj[slices]
    ) > 0
    target_mask = np.asarray(
        nib.load(str(args.products_dir / target / "whole_heart.nii.gz")).dataobj[slices]
    ) > 0
    with np.load(args.field_npz) as saved:
        field_zyx = saved["displacement_zyx_lps_mm"]
        if not np.allclose(saved["origin_lps_mm"], F @ affine[:3, 3], atol=1e-3):
            raise ValueError("Field origin differs from CT crop")
        if not np.allclose(saved["spacing_mm"], np.linalg.norm(affine[:3, :3], axis=0), atol=1e-3):
            raise ValueError("Field spacing differs from CT crop")
        expected_direction = F @ affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0)
        if not np.allclose(saved["direction_lps"], expected_direction, atol=1e-6):
            raise ValueError("Field direction differs from CT crop")
    field_xyz = field_zyx.transpose(2, 1, 0, 3)
    if field_xyz.shape[:3] != source_ct.shape:
        raise ValueError("Field shape differs from CT crop")
    inverse_axes = np.linalg.inv(F @ affine[:3, :3])
    center = np.rint(np.argwhere(target_mask).mean(axis=0)).astype(int)
    output.mkdir(parents=True, exist_ok=True)
    summary = {"status": "visual_review_only_auto_mask_control",
               "source": source, "target": target, "native_spacing_mm": report["grid_qc"]["evaluation_spacing_mm"],
               "crop_start_ijk": start.tolist(), "slice_indices_within_crop": center.tolist(),
               "images": []}
    for axis, label in enumerate("ijk"):
        index = int(center[axis])
        coords = plane_coordinates(source_ct.shape, axis, index)
        selector = [slice(None)] * 3
        selector[axis] = index
        selector = tuple(selector)
        delta_lps = field_xyz[selector]
        delta_ijk = delta_lps @ inverse_axes.T
        mapped = coords + delta_ijk
        warped_ct = ndi.map_coordinates(source_ct, [mapped[..., k] for k in range(3)],
                                        order=1, mode="constant", cval=-1000.0)
        warped_mask = ndi.map_coordinates(source_mask.astype(np.uint8),
                                          [mapped[..., k] for k in range(3)],
                                          order=0, mode="constant", cval=0) > 0
        fixed_ct_plane = target_ct[selector]
        fixed_mask_plane = target_mask[selector]
        static_mask_plane = source_mask[selector]
        magnitude = np.linalg.norm(delta_lps, axis=-1)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        for panel, candidate, title in zip(
            axes[:2], [static_mask_plane, warped_mask],
            ["Неподвижная маска", "После регистрации"],
        ):
            panel.imshow(fixed_ct_plane.T, origin="lower", cmap="gray", vmin=-200, vmax=300)
            panel.contour(fixed_mask_plane.T, levels=[0.5], colors="lime", linewidths=0.8)
            panel.contour(candidate.T, levels=[0.5], colors="red", linewidths=0.8)
            panel.set_title(title + "\nзелёный: auto цель; красный: источник")
        axes[2].imshow(fixed_ct_plane.T, origin="lower", cmap="gray", vmin=-200, vmax=300)
        overlay = axes[2].imshow(magnitude.T, origin="lower", cmap="magma", alpha=0.65,
                                 vmin=0, vmax=max(1.0, float(np.percentile(magnitude, 99))))
        fig.colorbar(overlay, ax=axes[2], label="Смещение, мм")
        axes[2].set_title("Модуль смещения на целевой сетке")
        for panel in axes:
            panel.set_xlabel("индекс в плоскости")
            panel.set_ylabel("индекс в плоскости")
        fig.suptitle(f"{source} → {target}; ось {label}, срез {index}; автоматические маски")
        name = f"{source}_to_{target}_{label}_{index}_review.png"
        fig.savefig(output / name, dpi=130)
        plt.close(fig)
        summary["images"].append(name)
    (output / "review_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
