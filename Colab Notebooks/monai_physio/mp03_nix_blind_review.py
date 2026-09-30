"""Create CT-only phase sheets for independent anatomical review.

The heart mask selects slice locations but is never drawn. Images and exact
slice provenance remain under ignored runs/ and do not constitute annotation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "runs" / "mp03_nix_independent_review"
PHASES = (0, 1, 3, 7, 10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-root", type=Path, required=True)
    args = parser.parse_args()
    reference = nib.load(str(args.subject_root / "products_v2" / "phase_00" / "whole_heart.nii.gz"))
    centre = np.rint(np.argwhere(np.asarray(reference.dataobj) > 0).mean(axis=0)).astype(int)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig, panels = plt.subplots(len(PHASES), 3, figsize=(14, 3.6 * len(PHASES)),
                               layout="constrained")
    for row, phase in enumerate(PHASES):
        image = nib.load(str(args.subject_root / "phases" / f"phase_{phase:02d}.nii.gz"))
        if image.shape != reference.shape or not np.allclose(image.affine, reference.affine):
            raise ValueError("Phase image and reference grid mismatch")
        slices = (np.asarray(image.dataobj[:, :, centre[2]]),
                  np.asarray(image.dataobj[:, centre[1], :]),
                  np.asarray(image.dataobj[centre[0], :, :]))
        for column, (array, label, index) in enumerate(zip(slices,
                                                            ("axial k", "sagittal j", "coronal i"),
                                                            (centre[2], centre[1], centre[0]))):
            panel = panels[row, column]
            panel.imshow(array.T, origin="lower", cmap="gray", vmin=-200, vmax=400,
                         interpolation="nearest")
            panel.set_title(f"Фаза {phase:02d} · {label}={index}")
            panel.set_xlabel("индекс в срезе")
            panel.set_ylabel("индекс в срезе")
    fig.suptitle("Ник · исходная 4D-КТ для независимого просмотра · без наложения сегментации")
    destination = OUTPUT / "ct_only_00_01_03_07_10.png"
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    manifest = {"status": "ct_only_review_material_not_annotation",
                "phases": [f"phase_{i:02d}" for i in PHASES],
                "slice_indices_ijk": centre.tolist(),
                "slice_selection": "centre of automatic phase_00 whole_heart, used only to choose locations",
                "visible_segmentation": False,
                "accepted_independent_manual_reference": False,
                "image": destination.name}
    (OUTPUT / "review_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
