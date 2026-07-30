"""Optional elastix baseline for paired respiratory CT.

All arrays are in the project's canonical RAS+ basis.  ITK images created here
have zero origin and identity direction, so their numerical physical axes are
RAS even though ITK normally labels medical image coordinates as LPS.  Elastix
returns the resampling transform ``fixed/exhale -> moving/inhale``; the dense
field returned by this module follows that same direction in millimetres.

The method is deliberately not trusted by construction.  A caller must still
evaluate independent expert landmarks and the displacement Jacobian before any
phase-wise anatomical measurement is admitted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import SimpleITK as sitk

from breathgeom.measure.registration import DeformableResult, warp_mask
from breathgeom.measure.wall import IntArray

BoolArray = npt.NDArray[np.bool_]
VectorArray = npt.NDArray[np.float32]


@dataclass(frozen=True)
class ElastixParams:
    """Reproducible rigid-affine-B-spline respiratory registration settings."""

    stages: tuple[str, ...] = ("rigid", "affine", "bspline")
    iterations: tuple[int, ...] = (128, 128, 256)
    number_of_resolutions: int = 4
    spatial_samples: int = 4096
    final_grid_spacing_mm: float = 24.0
    bending_energy_weight: float = 0.1
    random_seed: int = 20260730

    def __post_init__(self) -> None:
        allowed = {"rigid", "affine", "bspline"}
        if not self.stages or any(stage not in allowed for stage in self.stages):
            raise ValueError("stages must contain rigid, affine and/or bspline")
        if len(self.stages) != len(self.iterations):
            raise ValueError("stages and iterations must have equal length")
        if any(value <= 0 for value in self.iterations):
            raise ValueError("iterations must be positive")
        if self.number_of_resolutions <= 0 or self.spatial_samples <= 0:
            raise ValueError("resolutions and spatial samples must be positive")
        if self.final_grid_spacing_mm <= 0:
            raise ValueError("final_grid_spacing_mm must be positive")
        if self.bending_energy_weight < 0:
            raise ValueError("bending_energy_weight must be non-negative")


def _import_itk() -> Any:
    try:
        import itk
    except ImportError as error:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "elastix registration requires `pip install -e .[registration]`"
        ) from error
    return itk


def _to_itk(
    array_ras: npt.NDArray[np.generic],
    spacing: tuple[float, float, float],
) -> Any:
    itk = _import_itk()
    image = itk.image_from_array(np.ascontiguousarray(array_ras.transpose(2, 1, 0)))
    image.SetSpacing(tuple(float(value) for value in spacing))
    image.SetOrigin((0.0, 0.0, 0.0))
    return image


def _parameter_object(params: ElastixParams) -> Any:
    itk = _import_itk()
    parameter_object = itk.ParameterObject.New()
    for stage, iterations in zip(params.stages, params.iterations, strict=True):
        parameter_map = dict(parameter_object.GetDefaultParameterMap(stage))
        parameter_map["MaximumNumberOfIterations"] = (str(iterations),)
        parameter_map["NumberOfResolutions"] = (str(params.number_of_resolutions),)
        parameter_map["NumberOfSpatialSamples"] = (str(params.spatial_samples),)
        parameter_map["DefaultPixelValue"] = ("-1024",)
        parameter_map["RandomSeed"] = (str(params.random_seed),)
        parameter_map["WriteResultImage"] = ("true",)
        if stage == "bspline":
            parameter_map["FinalGridSpacingInPhysicalUnits"] = (
                f"{params.final_grid_spacing_mm:g}",
            )
            parameter_map["Metric0Weight"] = ("1.0",)
            parameter_map["Metric1Weight"] = (f"{params.bending_energy_weight:g}",)
        parameter_object.AddParameterMap(parameter_map)
    return parameter_object


def _dense_field(moving: Any, transform_parameters: Any) -> VectorArray:
    itk = _import_itk()
    transformix = itk.TransformixFilter.New(moving)
    transformix.SetTransformParameterObject(transform_parameters)
    transformix.SetComputeDeformationField(True)
    transformix.LogToConsoleOff()
    transformix.UpdateLargestPossibleRegion()
    field_image = transformix.GetOutputDeformationField()
    field = itk.array_from_image(field_image).transpose(2, 1, 0, 3).astype(np.float32)
    return np.ascontiguousarray(field)


def register_elastix(
    fixed_ras: IntArray,
    moving_ras: IntArray,
    spacing: tuple[float, float, float],
    fixed_mask: BoolArray,
    moving_mask: BoolArray,
    *,
    params: ElastixParams | None = None,
) -> DeformableResult:
    """Register moving inspiration to fixed expiration with optional ITK-elastix."""
    if fixed_ras.shape != moving_ras.shape or fixed_ras.shape != fixed_mask.shape:
        raise ValueError("fixed, moving and masks must share one voxel grid")
    if moving_mask.shape != fixed_mask.shape:
        raise ValueError("fixed and moving masks must share one voxel grid")
    if any(value <= 0 for value in spacing):
        raise ValueError("spacing must be positive")
    params = params or ElastixParams()
    itk = _import_itk()

    fixed = _to_itk(np.clip(fixed_ras, -1024, 500).astype(np.float32), spacing)
    moving = _to_itk(np.clip(moving_ras, -1024, 500).astype(np.float32), spacing)
    fixed_mask_image = _to_itk(fixed_mask.astype(np.uint8), spacing)
    moving_mask_image = _to_itk(moving_mask.astype(np.uint8), spacing)

    started = time.perf_counter()
    registered, transform_parameters = itk.elastix_registration_method(
        fixed,
        moving,
        fixed_mask=fixed_mask_image,
        moving_mask=moving_mask_image,
        parameter_object=_parameter_object(params),
        log_to_console=False,
        log_to_file=False,
    )
    elapsed = time.perf_counter() - started
    displacement = _dense_field(moving, transform_parameters)
    if displacement.shape != fixed_ras.shape + (3,):
        raise RuntimeError(
            f"elastix returned field {displacement.shape}, expected {fixed_ras.shape + (3,)}"
        )

    field_image = sitk.GetImageFromArray(
        displacement.transpose(2, 1, 0, 3).astype(np.float64)
    )
    field_image.SetSpacing(spacing)
    jacobian = sitk.GetArrayFromImage(
        sitk.DisplacementFieldJacobianDeterminant(field_image)
    )
    warped = itk.array_from_image(registered).transpose(2, 1, 0)
    warped_int = np.rint(np.clip(warped, -32768, 32767)).astype(np.int16)

    return DeformableResult(
        warped_moving=np.ascontiguousarray(warped_int),
        warped_moving_mask=warp_mask(moving_mask, displacement, spacing),
        displacement_mm=displacement,
        rigid_report=None,
        jacobian_min=float(np.min(jacobian)),
        jacobian_p01=float(np.percentile(jacobian, 1)),
        nonpositive_jacobian_fraction=float(np.mean(jacobian <= 0.0)),
        # The existing result slot is retained for API compatibility; elapsed
        # time is more useful here because elastix does not expose one final
        # scalar across its chained stages.
        metric=float(elapsed),
    )


__all__ = ["ElastixParams", "register_elastix"]
