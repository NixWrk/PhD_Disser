#!/usr/bin/env python
"""Run both TotalSegmentator v2 cardiac models on 4D-CT phases.

The script intentionally produces two independent model outputs per phase:

* ``total_heart/heart.nii.gz`` from the open ``total`` task;
* chamber and myocardium masks from licensed ``heartchambers_highres``.

The license is read from the user's TotalSegmentator configuration.  It is
never accepted as a command-line argument and never written to a manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path


WHOLE_HEART_MASK = "heart.nii.gz"
HIGHRES_MASKS = (
    "heart_myocardium.nii.gz",
    "heart_atrium_left.nii.gz",
    "heart_ventricle_left.nii.gz",
    "heart_atrium_right.nii.gz",
    "heart_ventricle_right.nii.gz",
    "aorta.nii.gz",
    "pulmonary_artery.nii.gz",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def model_provenance(dataset_ids: set[str]) -> list[dict[str, object]]:
    """Hash only model files for the tasks used here; never inspect config.json."""
    roots = [
        Path.home() / ".totalsegmentator" / "nnunet" / "results",
        Path(os.environ.get("nnUNet_results", "__not_configured__")),
    ]
    suffixes = {".pth", ".model", ".pkl", ".json"}
    seen: set[Path] = set()
    files = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            resolved = path.resolve()
            tail = "/".join(resolved.parts[-8:])
            if resolved in seen or not any(item in tail for item in dataset_ids):
                continue
            seen.add(resolved)
            files.append(
                {
                    "relative_tail": tail,
                    "bytes": resolved.stat().st_size,
                    "sha256": sha256_file(resolved),
                }
            )
    return sorted(files, key=lambda item: str(item["relative_tail"]))


def assert_same_grid(reference_path: Path, candidates: list[Path]) -> None:
    import nibabel as nib
    import numpy as np

    reference = nib.load(str(reference_path))
    for path in candidates:
        image = nib.load(str(path))
        if reference.shape != image.shape or not np.allclose(
            reference.affine, image.affine, atol=1e-4
        ):
            raise RuntimeError(f"{path}: output grid differs from source CT")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--phase-start", type=int, default=0)
    parser.add_argument("--phase-stop", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    from totalsegmentator.python_api import totalsegmentator

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected environment")

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    phase_paths = sorted(input_dir.glob("phase_*.nii.gz"))
    stop = len(phase_paths) if args.phase_stop is None else args.phase_stop
    phase_paths = phase_paths[args.phase_start : stop]
    if not phase_paths:
        raise FileNotFoundError("No selected phase_*.nii.gz files")
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for phase_path in phase_paths:
        phase_id = phase_path.name.removesuffix(".nii.gz")
        phase_output = output_dir / phase_id
        total_output = phase_output / "total_heart"
        highres_output = phase_output / "heartchambers_highres"
        expected = [total_output / WHOLE_HEART_MASK]
        expected.extend(highres_output / name for name in HIGHRES_MASKS)

        if all(path.is_file() for path in expected) and not args.overwrite:
            assert_same_grid(phase_path, expected)
            runs.append({"phase_id": phase_id, "status": "reused_existing_output"})
            print(phase_id, "reused_existing_output", flush=True)
            continue

        phase_output.mkdir(parents=True, exist_ok=True)
        total_output.mkdir(parents=True, exist_ok=True)
        highres_output.mkdir(parents=True, exist_ok=True)
        started = utc_now()

        # Independent whole-organ mask. roi_subset_robust uses the open v2 total
        # model and a robust 3-mm crop; it is not constructed from chamber masks.
        totalsegmentator(
            phase_path,
            total_output,
            task="total",
            roi_subset_robust=["heart"],
            device="gpu",
            nr_thr_resamp=1,
            nr_thr_saving=1,
            quiet=args.quiet,
            no_derived_masks=True,
        )

        # Licensed high-resolution anatomical task. Great-vessel masks are kept
        # as raw outputs, while downstream blood_pool deliberately uses only the
        # four cardiac chambers.
        totalsegmentator(
            phase_path,
            highres_output,
            task="heartchambers_highres",
            device="gpu",
            nr_thr_resamp=1,
            nr_thr_saving=1,
            quiet=args.quiet,
            robust_crop=True,
            no_derived_masks=True,
        )

        missing = [str(path.relative_to(phase_output)) for path in expected if not path.is_file()]
        if missing:
            raise RuntimeError(f"{phase_id}: missing output masks {missing}")
        assert_same_grid(phase_path, expected)
        runs.append(
            {
                "phase_id": phase_id,
                "status": "automatic_segmentation_pending_manual_review",
                "started_at": started,
                "finished_at": utc_now(),
            }
        )
        print(phase_id, "automatic_segmentation_pending_manual_review", flush=True)

    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "backend": "TotalSegmentator",
        "backend_version": importlib.metadata.version("TotalSegmentator"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "visible_gpu_selector": args.gpu,
        "models": {
            "whole_heart": {
                "task": "total",
                "class": "heart",
                "roi_subset_robust": True,
                "definition": "independent whole-organ mask; not a union of chambers and myocardium",
            },
            "anatomical_components": {
                "task": "heartchambers_highres",
                "classes": [name.removesuffix(".nii.gz") for name in HIGHRES_MASKS],
                "robust_crop": True,
            },
        },
        "license": {
            "source": "local TotalSegmentator user configuration",
            "value_recorded": False,
        },
        "model_files": model_provenance(
            {
                "Dataset291",
                "Dataset292",
                "Dataset293",
                "Dataset294",
                "Dataset295",
                "Dataset297",
                "Dataset301",
            }
        ),
        "runs": runs,
        "status": "automatic_segmentation_pending_manual_review",
    }
    suffix = f"gpu{args.gpu}_p{args.phase_start}_{stop}"
    (output_dir / f"segmentation_manifest_{suffix}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
