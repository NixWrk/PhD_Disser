"""ConvexAdam adapter with an isolated PyTorch execution environment.

The external runner returns a fixed-to-moving displacement in voxels on a
possibly downsampled zero-origin grid.  This module is the only place that
converts it to the project's X/Y/Z RAS+ millimetres and lifts it to the input
grid.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import SimpleITK as sitk

from breathgeom.measure.registration import (
    BoolArray,
    DeformableResult,
    VectorArray,
    warp_image,
    warp_mask,
)
from breathgeom.measure.wall import IntArray

Float32Array = npt.NDArray[np.float32]


@dataclass(frozen=True)
class ConvexAdamParams:
    """Locked parameters passed verbatim to convexAdam 0.2.0."""

    mind_r: int = 1
    mind_d: int = 2
    lambda_weight: float = 1.25
    grid_sp: int = 6
    disp_hw: int = 4
    selected_niter: int = 80
    selected_smooth: int = 0
    grid_sp_adam: int = 2
    inverse_consistency: bool = True
    use_mask: bool = True
    maximum_dimension: int = 208
    dtype: str = "float16"

    def __post_init__(self) -> None:
        if self.mind_r < 0 or self.mind_d < 1:
            raise ValueError("MIND radius must be non-negative and dilation positive")
        if self.lambda_weight < 0:
            raise ValueError("lambda_weight must be non-negative")
        if min(self.grid_sp, self.disp_hw, self.grid_sp_adam) < 1:
            raise ValueError("grid and displacement parameters must be positive")
        if self.selected_niter < 0 or self.selected_smooth < 0:
            raise ValueError("iteration and smoothing counts must be non-negative")
        if self.maximum_dimension < 32:
            raise ValueError("maximum_dimension must be at least 32")
        if self.dtype not in {"float16", "float32"}:
            raise ValueError("dtype must be float16 or float32")


@dataclass(frozen=True)
class ConvexAdamRuntime:
    """External-runner metadata safe to store in provenance."""

    torch_version: str
    convexadam_version: str
    device: str
    peak_gpu_memory_bytes: int
    elapsed_s: float
    preprocessed_shape: tuple[int, int, int]
    preprocessed_spacing_mm: tuple[float, float, float]


def preprocessing_grid(
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    maximum_dimension: int,
) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    """Choose a no-upsampling grid capped per axis and preserving physical extent."""
    if len(shape) != 3 or len(spacing) != 3:
        raise ValueError("shape and spacing must be three-dimensional")
    if any(size < 2 for size in shape) or any(value <= 0 for value in spacing):
        raise ValueError("shape and spacing values must be positive")
    sizes: list[int] = []
    spacings: list[float] = []
    for size, voxel_spacing in zip(shape, spacing, strict=True):
        target_size = min(size, maximum_dimension)
        if target_size % 2:
            target_size -= 1
        target_size = max(target_size, 2)
        extent = (size - 1) * voxel_spacing
        sizes.append(target_size)
        spacings.append(extent / (target_size - 1))
    return cast(tuple[int, int, int], tuple(sizes)), cast(
        tuple[float, float, float], tuple(spacings)
    )


def _to_sitk(
    array_xyz: npt.NDArray[np.generic],
    spacing: tuple[float, float, float],
    pixel_id: int,
) -> sitk.Image:
    image = sitk.GetImageFromArray(np.ascontiguousarray(array_xyz.transpose(2, 1, 0)))
    image.SetSpacing(spacing)
    result: sitk.Image = sitk.Cast(image, pixel_id)
    return result


def _reference(
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    pixel_id: int,
) -> sitk.Image:
    image = sitk.Image([int(value) for value in shape], pixel_id)
    image.SetSpacing(spacing)
    return image


def _resample_scalar(
    values: npt.NDArray[np.generic],
    source_spacing: tuple[float, float, float],
    target_shape: tuple[int, int, int],
    target_spacing: tuple[float, float, float],
    *,
    nearest: bool,
) -> npt.NDArray[np.generic]:
    source = _to_sitk(
        values,
        source_spacing,
        sitk.sitkUInt8 if nearest else sitk.sitkFloat32,
    )
    reference = _reference(
        target_shape,
        target_spacing,
        sitk.sitkUInt8 if nearest else sitk.sitkFloat32,
    )
    result = sitk.Resample(
        source,
        reference,
        sitk.Transform(),
        sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear,
        0.0 if nearest else -1024.0,
        reference.GetPixelID(),
    )
    array = sitk.GetArrayFromImage(result).transpose(2, 1, 0)
    return np.ascontiguousarray(array)


def voxel_displacement_to_mm(
    displacement_voxels: npt.NDArray[np.generic],
    spacing: tuple[float, float, float],
) -> VectorArray:
    """Convert ConvexAdam X/Y/Z voxel offsets to physical RAS+ millimetres."""
    if displacement_voxels.ndim != 4 or displacement_voxels.shape[-1] != 3:
        raise ValueError("displacement must have shape X x Y x Z x 3")
    result = np.asarray(displacement_voxels, dtype=np.float32).copy()
    result *= np.asarray(spacing, dtype=np.float32)
    return np.ascontiguousarray(result)


def resample_displacement_mm(
    field_mm: VectorArray,
    source_spacing: tuple[float, float, float],
    target_shape: tuple[int, int, int],
    target_spacing: tuple[float, float, float],
) -> VectorArray:
    """Lift a physical vector field without scaling its millimetre components."""
    if field_mm.ndim != 4 or field_mm.shape[-1] != 3:
        raise ValueError("field_mm must have shape X x Y x Z x 3")
    reference = _reference(target_shape, target_spacing, sitk.sitkFloat32)
    components: list[Float32Array] = []
    for index in range(3):
        source = _to_sitk(field_mm[..., index], source_spacing, sitk.sitkFloat32)
        resampled = sitk.Resample(
            source,
            reference,
            sitk.Transform(),
            sitk.sitkLinear,
            0.0,
            sitk.sitkFloat32,
        )
        component = (
            sitk.GetArrayFromImage(resampled).transpose(2, 1, 0).astype(np.float32)
        )
        components.append(np.ascontiguousarray(component))
    return np.ascontiguousarray(np.stack(components, axis=-1))


def _jacobian_stats(
    field_mm: VectorArray,
    spacing: tuple[float, float, float],
) -> tuple[float, float, float]:
    image = sitk.GetImageFromArray(field_mm.transpose(2, 1, 0, 3).astype(np.float64))
    image.SetSpacing(spacing)
    values = sitk.GetArrayFromImage(
        sitk.DisplacementFieldJacobianDeterminant(image)
    )
    return (
        float(np.min(values)),
        float(np.percentile(values, 1)),
        float(np.mean(values <= 0.0)),
    )


def _runner_path(repo_root: Path) -> Path:
    path = repo_root / "tools/run_convexadam.py"
    if not path.is_file():
        raise FileNotFoundError(f"ConvexAdam runner is missing: {path}")
    return path


def register_convexadam(
    fixed_ras: IntArray,
    moving_ras: IntArray,
    spacing: tuple[float, float, float],
    fixed_mask: BoolArray,
    moving_mask: BoolArray,
    *,
    python_executable: Path,
    repo_root: Path,
    params: ConvexAdamParams | None = None,
    temporary_root: Path | None = None,
) -> tuple[DeformableResult, ConvexAdamRuntime]:
    """Run ConvexAdam externally and return a project-native dense field."""
    if fixed_ras.shape != moving_ras.shape or fixed_ras.shape != fixed_mask.shape:
        raise ValueError("fixed, moving and masks must share one voxel grid")
    if moving_mask.shape != fixed_mask.shape:
        raise ValueError("fixed and moving masks must share one voxel grid")
    if not python_executable.is_file():
        raise FileNotFoundError(f"registration Python is missing: {python_executable}")
    params = params or ConvexAdamParams()
    target_shape, target_spacing = preprocessing_grid(
        cast(tuple[int, int, int], fixed_ras.shape),
        spacing,
        params.maximum_dimension,
    )
    fixed_pre = _resample_scalar(
        np.clip(fixed_ras, -1024, 500).astype(np.float32),
        spacing,
        target_shape,
        target_spacing,
        nearest=False,
    ).astype(np.float32)
    moving_pre = _resample_scalar(
        np.clip(moving_ras, -1024, 500).astype(np.float32),
        spacing,
        target_shape,
        target_spacing,
        nearest=False,
    ).astype(np.float32)
    fixed_mask_pre = _resample_scalar(
        fixed_mask.astype(np.uint8),
        spacing,
        target_shape,
        target_spacing,
        nearest=True,
    ).astype(np.uint8)
    moving_mask_pre = _resample_scalar(
        moving_mask.astype(np.uint8),
        spacing,
        target_shape,
        target_spacing,
        nearest=True,
    ).astype(np.uint8)

    temporary_root_value = str(temporary_root) if temporary_root is not None else None
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(
        prefix="breathgeom-convexadam-",
        dir=temporary_root_value,
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        input_path = temporary / "input.npz"
        output_path = temporary / "output.npz"
        params_path = temporary / "params.json"
        np.savez(
            input_path,
            fixed=fixed_pre,
            moving=moving_pre,
            fixed_mask=fixed_mask_pre,
            moving_mask=moving_mask_pre,
        )
        params_path.write_text(
            json.dumps(asdict(params), indent=2),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                str(python_executable),
                str(_runner_path(repo_root)),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--params",
                str(params_path),
            ],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"ConvexAdam runner failed with code {completed.returncode}: {message}"
            )
        with np.load(output_path) as payload:
            displacement_voxels = np.asarray(payload["displacement_voxels"])
            metadata: Any = json.loads(str(payload["metadata_json"]))

    coarse_mm = voxel_displacement_to_mm(displacement_voxels, target_spacing)
    full_mm = resample_displacement_mm(
        coarse_mm,
        target_spacing,
        cast(tuple[int, int, int], fixed_ras.shape),
        spacing,
    )
    jacobian_min, jacobian_p01, nonpositive_fraction = _jacobian_stats(full_mm, spacing)
    elapsed = time.perf_counter() - started
    result = DeformableResult(
        warped_moving=warp_image(moving_ras, full_mm, spacing),
        warped_moving_mask=warp_mask(moving_mask, full_mm, spacing),
        displacement_mm=full_mm,
        rigid_report=None,
        jacobian_min=jacobian_min,
        jacobian_p01=jacobian_p01,
        nonpositive_jacobian_fraction=nonpositive_fraction,
        metric=float("nan"),
        elapsed_s=float(elapsed),
    )
    runtime = ConvexAdamRuntime(
        torch_version=str(metadata["torch_version"]),
        convexadam_version=str(metadata["convexadam_version"]),
        device=str(metadata["device"]),
        peak_gpu_memory_bytes=int(metadata["peak_gpu_memory_bytes"]),
        elapsed_s=float(metadata["elapsed_s"]),
        preprocessed_shape=target_shape,
        preprocessed_spacing_mm=target_spacing,
    )
    return result, runtime


__all__ = [
    "ConvexAdamParams",
    "ConvexAdamRuntime",
    "preprocessing_grid",
    "register_convexadam",
    "resample_displacement_mm",
    "voxel_displacement_to_mm",
]
