"""Evaluate frozen displacement candidates with the exact J1.2 CUDA data terms."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import torch
from convexAdam.convex_adam_utils import MINDSSC
from run_joint_piecewise_svf import (
    _base_grid,
    _numpy_xyz_to_tensor_scalar,
    _numpy_xyz_to_tensor_vector,
    _warp_tensor,
)
from scipy import ndimage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    return parser.parse_args()


def _masked_mse(
    fixed: torch.Tensor,
    warped: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    channels = fixed.shape[1]
    denominator = mask.sum().clamp_min(1.0) * channels
    return float((((fixed - warped).square() * mask).sum() / denominator).item())


def _validate_params(payload: dict[str, Any]) -> None:
    if set(payload) != {"runner_version", "descriptor", "erosion_iterations"}:
        raise ValueError("unexpected identifiability runner parameters")
    if payload["runner_version"] != "joint-svf-data-identifiability-v1":
        raise ValueError("unsupported identifiability runner version")
    descriptor = payload["descriptor"]
    if not isinstance(descriptor, dict) or set(descriptor) != {
        "kind",
        "radius",
        "dilation",
        "source",
    }:
        raise ValueError("invalid descriptor parameters")
    if descriptor["kind"] != "MINDSSC":
        raise ValueError("only MINDSSC is supported")
    if descriptor["source"] != "convexAdam.convex_adam_utils.MINDSSC":
        raise ValueError("unexpected MIND source")
    if int(payload["erosion_iterations"]) != 2:
        raise ValueError("the frozen screen requires two erosion iterations")


def _load_input(path: Path) -> dict[str, np.ndarray[Any, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        expected = {
            "fixed_image",
            "moving_image",
            "fixed_lung_mask",
            "fixed_body_mask",
            "spacing_mm",
            "candidate_names",
            "lung_displacement_mm",
            "body_displacement_mm",
        }
        if set(payload.files) != expected:
            raise ValueError("unexpected identifiability input schema")
        arrays = {key: np.asarray(payload[key]) for key in expected}
    shape = arrays["fixed_image"].shape
    if arrays["moving_image"].shape != shape:
        raise ValueError("phase images must share one grid")
    if (
        arrays["fixed_lung_mask"].shape != shape
        or arrays["fixed_body_mask"].shape != shape
    ):
        raise ValueError("masks must share the image grid")
    candidate_count = len(arrays["candidate_names"])
    expected_field_shape = (candidate_count, *shape, 3)
    if (
        arrays["lung_displacement_mm"].shape != expected_field_shape
        or arrays["body_displacement_mm"].shape != expected_field_shape
    ):
        raise ValueError("candidate displacement arrays have invalid shape")
    if arrays["spacing_mm"].shape != (3,) or np.any(arrays["spacing_mm"] <= 0):
        raise ValueError("spacing must contain three positive values")
    return arrays


def main() -> None:
    args = parse_args()
    parameters: dict[str, Any] = json.loads(args.params.read_text(encoding="utf-8"))
    _validate_params(parameters)
    arrays = _load_input(args.input)
    if not torch.cuda.is_available():
        raise RuntimeError("identifiability runner requires a CUDA device")
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    spacing = tuple(float(value) for value in arrays["spacing_mm"])
    fixed_image = arrays["fixed_image"].astype(np.float32) / 1000.0
    moving_image = arrays["moving_image"].astype(np.float32) / 1000.0
    fixed_lung = arrays["fixed_lung_mask"].astype(bool)
    fixed_wall = arrays["fixed_body_mask"].astype(bool) & ~fixed_lung
    erosion_iterations = int(parameters["erosion_iterations"])
    masks_numpy = {
        "lung_full": fixed_lung,
        "lung_interior": ndimage.binary_erosion(
            fixed_lung,
            iterations=erosion_iterations,
        ),
        "body_full": fixed_wall,
        "body_interior": ndimage.binary_erosion(
            fixed_wall,
            iterations=erosion_iterations,
        ),
    }
    if not all(np.any(mask) for mask in masks_numpy.values()):
        raise ValueError("an identifiability evaluation mask is empty")
    masks = {
        name: _numpy_xyz_to_tensor_scalar(mask.astype(np.float32), device)
        for name, mask in masks_numpy.items()
    }
    fixed_tensor = _numpy_xyz_to_tensor_scalar(fixed_image, device)
    moving_tensor = _numpy_xyz_to_tensor_scalar(moving_image, device)
    shape_dhw = tuple(int(value) for value in fixed_tensor.shape[2:])
    base_grid = _base_grid(shape_dhw, device=device, dtype=torch.float32)
    descriptor = parameters["descriptor"]
    fixed_mind = MINDSSC(
        fixed_tensor,
        radius=int(descriptor["radius"]),
        dilation=int(descriptor["dilation"]),
        device=device,
    ).detach()
    moving_mind = MINDSSC(
        moving_tensor,
        radius=int(descriptor["radius"]),
        dilation=int(descriptor["dilation"]),
        device=device,
    ).detach()
    names = [str(value) for value in arrays["candidate_names"].tolist()]
    records: list[dict[str, str | float]] = []
    for index, name in enumerate(names):
        lung_field = _numpy_xyz_to_tensor_vector(
            arrays["lung_displacement_mm"][index].astype(np.float32),
            device,
        )
        body_field = _numpy_xyz_to_tensor_vector(
            arrays["body_displacement_mm"][index].astype(np.float32),
            device,
        )
        for region, field in (("lung", lung_field), ("body", body_field)):
            warped_mind = _warp_tensor(moving_mind, field, spacing, base_grid)
            warped_intensity = _warp_tensor(
                moving_tensor,
                field,
                spacing,
                base_grid,
            )
            records.append(
                {
                    "candidate": name,
                    "region": region,
                    "mind_mse_full": _masked_mse(
                        fixed_mind,
                        warped_mind,
                        masks[f"{region}_full"],
                    ),
                    "mind_mse_interior": _masked_mse(
                        fixed_mind,
                        warped_mind,
                        masks[f"{region}_interior"],
                    ),
                    "intensity_mse_full": _masked_mse(
                        fixed_tensor,
                        warped_intensity,
                        masks[f"{region}_full"],
                    ),
                    "intensity_mse_interior": _masked_mse(
                        fixed_tensor,
                        warped_intensity,
                        masks[f"{region}_interior"],
                    ),
                }
            )
    result = {
        "runner_version": parameters["runner_version"],
        "torch_version": torch.__version__,
        "convexadam_version": importlib.metadata.version("convexAdam"),
        "mind_source_module": MINDSSC.__module__,
        "mind_source_file_name": Path(inspect.getfile(MINDSSC)).name,
        "scipy_version": scipy.__version__,
        "device": torch.cuda.get_device_name(device),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_s": time.perf_counter() - started,
        "coordinate_conversion": "numpy_XYZ_components_XYZ_to_torch_DHW_channels_XYZ",
        "transform_direction": "fixed-expiration_to_moving-inspiration",
        "intensity_units": "HU_divided_by_1000_squared_MSE",
        "records": records,
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
