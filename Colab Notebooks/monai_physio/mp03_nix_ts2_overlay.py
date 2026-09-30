"""Render local CT/TSv2/transported-mask panels for Nix phase QC."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


HERE = Path(__file__).resolve().parent
PHASES = (0, 1, 3, 7, 10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-root", type=Path, required=True)
    args = parser.parse_args()
    source = nib.load(str(args.subject_root / "products_v2/phase_00/whole_heart.nii.gz"))
    centre = np.rint(ndi.center_of_mass(np.asarray(source.dataobj) > 0)).astype(int)
    selection = (lambda a: a[:, :, centre[2]],
                 lambda a: a[:, centre[1], :],
                 lambda a: a[centre[0], :, :])
    fig, panels = plt.subplots(5, 3, figsize=(14, 17), layout="constrained")
    for row, index in enumerate(PHASES):
        phase = f"phase_{index:02d}"
        ct = nib.load(str(args.subject_root / "phases" / f"{phase}.nii.gz"))
        ts2 = nib.load(str(args.subject_root / "products_v2" / phase / "whole_heart.nii.gz"))
        if index:
            moved = nib.load(str(HERE / "runs" / "mp02_nix_paired_fem" /
                                 f"{phase}_transported.nii.gz"))
        for col, get_slice in enumerate(selection):
            panel = panels[row, col]
            image = np.asarray(get_slice(ct.dataobj)).T
            target = np.asarray(get_slice(ts2.dataobj)).T
            panel.imshow(image, origin="lower", cmap="gray", vmin=-200, vmax=400)
            panel.contour(target, levels=[0.5], colors=["#65ff66"], linewidths=1.1)
            if index:
                candidate = np.asarray(get_slice(moved.dataobj)).T
                panel.contour(candidate, levels=[0.5], colors=["#ff55d0"], linewidths=1.0)
            panel.set_title(f"{phase} · {('axial k', 'sagittal j', 'coronal i')[col]}"
                            f" = {centre[(2, 1, 0)[col]]}")
            panel.set_xticks([])
            panel.set_yticks([])
    fig.suptitle("Ник · КТ: TSv2 по фазе (зелёный), перенос 00 (розовый) · выборочные срезы")
    path = HERE / "runs" / "mp03_nix_ts2_qc" / "selected_ct_overlays.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
