"""Isolated ConvexAdam 0.2.0 runner.

This script is executed by ``.venv-registration`` and communicates with the
main package only through temporary NPZ/JSON files.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from convexAdam.convex_adam_MIND import convex_adam_pt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parameters: dict[str, Any] = json.loads(args.params.read_text(encoding="utf-8"))
    with np.load(args.input) as payload:
        fixed = np.asarray(payload["fixed"], dtype=np.float32)
        moving = np.asarray(payload["moving"], dtype=np.float32)
        fixed_mask = np.asarray(payload["fixed_mask"], dtype=np.uint8)
        moving_mask = np.asarray(payload["moving_mask"], dtype=np.uint8)
    if not torch.cuda.is_available():
        raise RuntimeError("ConvexAdam benchmark requires a CUDA device")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="convexadam-masks-") as mask_directory:
        mask_root = Path(mask_directory)
        fixed_mask_path = mask_root / "fixed_mask.nii.gz"
        moving_mask_path = mask_root / "moving_mask.nii.gz"
        nib.save(nib.Nifti1Image(fixed_mask, np.eye(4)), fixed_mask_path)
        nib.save(nib.Nifti1Image(moving_mask, np.eye(4)), moving_mask_path)
        displacement = convex_adam_pt(
            img_fixed=fixed,
            img_moving=moving,
            mind_r=int(parameters["mind_r"]),
            mind_d=int(parameters["mind_d"]),
            lambda_weight=float(parameters["lambda_weight"]),
            grid_sp=int(parameters["grid_sp"]),
            disp_hw=int(parameters["disp_hw"]),
            selected_niter=int(parameters["selected_niter"]),
            selected_smooth=int(parameters["selected_smooth"]),
            grid_sp_adam=int(parameters["grid_sp_adam"]),
            ic=bool(parameters["inverse_consistency"]),
            use_mask=bool(parameters["use_mask"]),
            path_fixed_mask=fixed_mask_path,
            path_moving_mask=moving_mask_path,
            dtype=(
                torch.float16
                if parameters["dtype"] == "float16"
                else torch.float32
            ),
            verbose=False,
            device=torch.device("cuda"),
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    metadata = {
        "torch_version": torch.__version__,
        "convexadam_version": importlib.metadata.version("convexAdam"),
        "device": torch.cuda.get_device_name(0),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "elapsed_s": float(elapsed),
    }
    np.savez(
        args.output,
        displacement_voxels=np.asarray(displacement, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata)),
    )


if __name__ == "__main__":
    main()
