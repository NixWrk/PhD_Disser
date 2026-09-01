#!/usr/bin/env python
"""Stable Windows runner for TotalSegmentator 1.5.7 cardiac classes.

The PyPI 1.5.7 console entry point passes a newer argument list to its older
Python API on this Windows environment.  This runner calls that documented API
directly and confines the legacy checkpoint compatibility flag to this process.
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


EXPECTED_MASKS = (
    "heart_myocardium.nii.gz",
    "heart_atrium_left.nii.gz",
    "heart_ventricle_left.nii.gz",
    "heart_atrium_right.nii.gz",
    "heart_ventricle_right.nii.gz",
)
ROI_SUBSET = tuple(name.removesuffix(".nii.gz") for name in EXPECTED_MASKS)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def model_provenance():
    candidates = []
    roots = [
        Path(os.environ.get("nnUNet_results", "")),
        Path.home() / ".totalsegmentator" / "nnunet" / "results",
    ]
    seen = set()
    for root in roots:
        if not str(root) or not root.exists():
            continue
        for path in root.rglob("model_final_checkpoint.model"):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(
                {
                    "relative_tail": "/".join(resolved.parts[-6:]),
                    "bytes": resolved.stat().st_size,
                    "sha256": sha256_file(resolved),
                }
            )
    return sorted(candidates, key=lambda item: item["relative_tail"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--phase-start", type=int, default=0)
    parser.add_argument("--phase-stop", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # Official TotalSegmentator v1 checkpoints predate PyTorch 2.6.  They are
    # downloaded by TotalSegmentator itself; do not set this globally.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    import nibabel as nib
    import numpy as np
    import torch
    from totalsegmentator.python_api import totalsegmentator

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected environment")

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    phases = sorted(input_dir.glob("phase_*.nii.gz"))
    stop = len(phases) if args.phase_stop is None else args.phase_stop
    phases = phases[args.phase_start : stop]
    if not phases:
        raise FileNotFoundError("No selected phase_*.nii.gz files")
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for phase_path in phases:
        phase_id = phase_path.name.removesuffix(".nii.gz")
        phase_output = output_dir / phase_id
        phase_output.mkdir(parents=True, exist_ok=True)
        existing = [phase_output / name for name in EXPECTED_MASKS]
        if all(path.exists() for path in existing) and not args.overwrite:
            status = "reused_existing_output"
        else:
            started = utc_now()
            totalsegmentator(
                phase_path,
                phase_output,
                nr_thr_saving=1,
                roi_subset=list(ROI_SUBSET),
                quiet=args.quiet,
            )
            missing = [name for name in EXPECTED_MASKS if not (phase_output / name).is_file()]
            if missing:
                raise RuntimeError(f"{phase_id}: missing output masks {missing}")
            reference = nib.load(str(phase_path))
            for name in EXPECTED_MASKS:
                candidate = nib.load(str(phase_output / name))
                if reference.shape != candidate.shape or not np.allclose(
                    reference.affine, candidate.affine, atol=1e-4
                ):
                    raise RuntimeError(f"{phase_id}/{name}: output grid differs from CT")
            status = "automatic_segmentation_pending_manual_review"
            runs.append(
                {
                    "phase_id": phase_id,
                    "status": status,
                    "started_at": started,
                    "finished_at": utc_now(),
                }
            )
            print(phase_id, status, flush=True)
            continue
        runs.append({"phase_id": phase_id, "status": status})
        print(phase_id, status, flush=True)

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
        "task": "total_v1_roi_subset",
        "roi_subset": list(ROI_SUBSET),
        "legacy_checkpoint_loading": {
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "scope": "this process only",
            "trust_basis": "weights downloaded by official TotalSegmentator package",
        },
        "model_files": model_provenance(),
        "runs": runs,
        "status": "automatic_segmentation_pending_manual_review",
    }
    suffix = f"gpu{args.gpu}_p{args.phase_start}_{stop}"
    (output_dir / f"segmentation_manifest_{suffix}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

