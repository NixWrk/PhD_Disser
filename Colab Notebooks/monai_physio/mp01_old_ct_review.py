"""Render local-only OLD CT overlays for MONAI, TS, and candidate contours."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


HERE = Path(__file__).resolve().parent
CHAMBERS = {
    "LV": ("highres_ventricle_left", "heart_ventricle_left.nii.gz", "LV_cavity_envelope.nii.gz"),
    "RV": ("highres_ventricle_right", "heart_ventricle_right.nii.gz", "RV_cavity_envelope.nii.gz"),
    "LA": ("highres_atrium_left", "heart_atrium_left.nii.gz", "LA_cavity_envelope.nii.gz"),
    "RA": ("highres_atrium_right", "heart_atrium_right.nii.gz", "RA_cavity_envelope.nii.gz"),
}


def array(path):
    return np.asanyarray(nib.load(str(path)).dataobj)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "runs" / "mp01_old_review.png")
    parser.add_argument("--nv-prompted", action="store_true",
                        help="Show the separate pretrained NVIDIA model with explicit cardiac prompts")
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to((HERE / "runs").resolve()):
        parser.error("Image must stay inside ignored runs/")
    fig, axes = plt.subplots(2, 4, figsize=(19, 9), constrained_layout=True)
    for row, phase in enumerate(("phase_00", "phase_08")):
        ct = array(args.old_root / "sequence_02" / "phases" / f"{phase}.nii.gz")
        run = HERE / "runs" / f"mp01_old_{phase.replace('phase_', 'phase')}_full"
        nv_run = HERE / "runs" / f"mp01_old_{phase.replace('phase_', 'phase')}_nvsegment_prompted"
        ts_root = args.old_root / "sequence02_ts_comparator_v1" / "automatic_masks_v2" / phase / "heartchambers_highres"
        direct_root = args.old_root / "anatomical_audit_v02" / "direct_v01" / phase
        for col, (name, (monai_name, ts_name, direct_name)) in enumerate(CHAMBERS.items()):
            monai = (array(nv_run / f"nvsegment_{name}.nii.gz") > 0 if args.nv_prompted
                     else array(run / f"mp01_02_{monai_name}.nii.gz") > 0)
            ts = array(ts_root / ts_name) > 0
            direct = array(direct_root / direct_name) > 0
            z = int(np.argmax(direct.sum(axis=(0, 1))))
            union = monai[:, :, z] | ts[:, :, z] | direct[:, :, z]
            xy = np.argwhere(union)
            lo = np.maximum(xy.min(axis=0) - 20, 0)
            hi = np.minimum(xy.max(axis=0) + 21, ct.shape[:2])
            sl = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), z)
            ax = axes[row, col]
            ax.imshow(np.clip(ct[sl].T, 0, 500), cmap="gray", vmin=0, vmax=500, origin="lower")
            for mask, color in ((monai, "#20c7e8"), (ts, "#77e45a"), (direct, "#ff677d")):
                ax.contour(mask[sl].T, levels=[0.5], colors=[color], linewidths=1.2)
            ax.set_title(f"{phase[-2:]} / {name} / z={z}")
            ax.axis("off")
            del monai, ts, direct
        del ct
    first = "pretrained NVIDIA prompted" if args.nv_prompted else "MONAI TS wrapper"
    fig.suptitle(f"OLD CT: {first} cyan / TotalSegmentator green / unblinded direct contour pink")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
