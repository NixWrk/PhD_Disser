"""Physical-space deformable registration baseline with explicit QC.

The implementation is intentionally a baseline. Diffeomorphic Demons provides
a dense field and a folding check, but it does not model pleural or fissural
sliding. It must therefore earn its use through independent landmarks rather
than through visual plausibility or intensity similarity alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import SimpleITK as sitk
from scipy import ndimage

from breathgeom.measure.align import AlignmentReport, rigid_align
from breathgeom.measure.segmentation import binary_surface
from breathgeom.measure.wall import IntArray

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]
VectorArray = npt.NDArray[np.float32]


@dataclass(frozen=True)
class RegistrationParams:
    """Multiresolution settings in physical units."""

    level_spacing_mm: tuple[float, ...] = (6.0, 3.0)
    iterations: tuple[int, ...] = (40, 30)
    displacement_smoothing_sigma: float = 1.5
    update_smoothing_sigma: float = 1.0
    maximum_update_step_mm: float = 2.0

    def __post_init__(self) -> None:
        if len(self.level_spacing_mm) != len(self.iterations):
            raise ValueError("level_spacing_mm and iterations must have equal length")
        if not self.level_spacing_mm or any(value <= 0 for value in self.level_spacing_mm):
            raise ValueError("registration levels must have positive spacing")
        if any(value <= 0 for value in self.iterations):
            raise ValueError("registration iterations must be positive")


@dataclass(frozen=True)
class BSplineParams:
    """Free-form deformation settings for the classical comparison baseline."""

    control_point_spacing_mm: float = 40.0
    optimization_spacing_mm: float = 3.0
    iterations: int = 80
    sampling_fraction: float = 0.1
    shrink_factors: tuple[int, ...] = (2, 1)
    smoothing_sigmas_mm: tuple[float, ...] = (2.0, 0.0)
    # LBFGSB keeps a fixed parameter vector; zero disables ITK's per-level mesh
    # adaptor, whose changing parameter count is incompatible with this optimizer.
    mesh_scale_factors: tuple[int, ...] = (0, 0)

    def __post_init__(self) -> None:
        if (
            self.control_point_spacing_mm <= 0
            or self.optimization_spacing_mm <= 0
            or self.iterations <= 0
        ):
            raise ValueError("B-spline spacing and iterations must be positive")
        if not 0 < self.sampling_fraction <= 1:
            raise ValueError("sampling_fraction must lie in (0, 1]")
        levels = len(self.shrink_factors)
        if levels != len(self.smoothing_sigmas_mm) or levels != len(self.mesh_scale_factors):
            raise ValueError("all B-spline multiresolution sequences must have equal length")


@dataclass(frozen=True)
class MaskMetrics:
    dice: float
    surface_mean_mm: float
    surface_p95_mm: float


@dataclass(frozen=True)
class DeformableResult:
    """Moving phase resampled into fixed space and its fixed-to-moving DVF."""

    warped_moving: npt.NDArray[np.int16]
    warped_moving_mask: BoolArray
    displacement_mm: VectorArray
    rigid_report: AlignmentReport | None
    jacobian_min: float
    jacobian_p01: float
    nonpositive_jacobian_fraction: float
    metric: float


def _to_sitk(
    array_ras: npt.NDArray[np.generic],
    spacing: tuple[float, float, float],
    pixel_id: int,
) -> sitk.Image:
    image = sitk.GetImageFromArray(np.ascontiguousarray(array_ras.transpose(2, 1, 0)))
    image.SetSpacing(tuple(float(value) for value in spacing))
    cast: sitk.Image = sitk.Cast(image, pixel_id)
    return cast


def _from_sitk_int(image: sitk.Image) -> npt.NDArray[np.int16]:
    array = sitk.GetArrayFromImage(image).transpose(2, 1, 0).astype(np.int16)
    return np.ascontiguousarray(array)


def _from_sitk_bool(image: sitk.Image) -> BoolArray:
    array: BoolArray = sitk.GetArrayFromImage(image).transpose(2, 1, 0) > 0
    return np.ascontiguousarray(array)


def _from_sitk_vector(image: sitk.Image) -> VectorArray:
    # SimpleITK arrays are (z, y, x, component); project arrays are (x, y, z, component).
    array = sitk.GetArrayFromImage(image).transpose(2, 1, 0, 3).astype(np.float32)
    return np.ascontiguousarray(array)


def _level_reference(image: sitk.Image, target_spacing_mm: float) -> sitk.Image:
    old_size = np.asarray(image.GetSize(), dtype=np.int64)
    old_spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    target = np.maximum(old_spacing, target_spacing_mm)
    extent = np.maximum(old_size - 1, 1) * old_spacing
    new_size = np.maximum(np.rint(extent / target).astype(np.int64) + 1, 2)
    reference = sitk.Image([int(value) for value in new_size], image.GetPixelID())
    reference.SetSpacing(tuple(float(value) for value in target))
    reference.SetOrigin(image.GetOrigin())
    reference.SetDirection(image.GetDirection())
    return reference


def _resample_scalar(image: sitk.Image, reference: sitk.Image) -> sitk.Image:
    return sitk.Resample(
        image, reference, sitk.Transform(), sitk.sitkLinear, -1000.0, sitk.sitkFloat32
    )


def _resample_field(field: sitk.Image, reference: sitk.Image) -> sitk.Image:
    result: sitk.Image = sitk.Resample(
        field,
        reference,
        sitk.Transform(),
        sitk.sitkLinear,
        0.0,
        sitk.sitkVectorFloat64,
    )
    return result


def _rigid_transform(report: AlignmentReport) -> sitk.Euler3DTransform:
    transform = sitk.Euler3DTransform()
    transform.SetCenter(report.centre_mm)
    transform.SetRotation(*(np.radians(value) for value in report.rotation_deg))
    transform.SetTranslation(report.translation_mm)
    return transform


def _initial_field(reference: sitk.Image, transform: sitk.Transform | None) -> sitk.Image:
    transform = transform or sitk.Transform(3, sitk.sitkIdentity)
    field: sitk.Image = sitk.TransformToDisplacementField(
        transform,
        sitk.sitkVectorFloat64,
        reference.GetSize(),
        reference.GetOrigin(),
        reference.GetSpacing(),
        reference.GetDirection(),
    )
    return field


def register_diffeomorphic(
    fixed_ras: IntArray,
    moving_ras: IntArray,
    spacing: tuple[float, float, float],
    fixed_mask: BoolArray,
    moving_mask: BoolArray,
    *,
    params: RegistrationParams | None = None,
    run_rigid: bool = True,
) -> DeformableResult:
    """Register ``moving`` to ``fixed`` and return a fixed-to-moving field."""
    if fixed_ras.shape != moving_ras.shape or fixed_ras.shape != fixed_mask.shape:
        raise ValueError("fixed, moving and masks must share one voxel grid")
    if moving_mask.shape != fixed_mask.shape:
        raise ValueError("fixed and moving masks must share one voxel grid")
    params = params or RegistrationParams()

    rigid_report: AlignmentReport | None = None
    initial_transform: sitk.Transform | None = None
    if run_rigid:
        _, rigid_report = rigid_align(fixed_ras, moving_ras, spacing)
        initial_transform = _rigid_transform(rigid_report)

    fixed = _to_sitk(fixed_ras, spacing, sitk.sitkFloat32)
    moving = _to_sitk(moving_ras, spacing, sitk.sitkFloat32)
    fixed = sitk.Clamp(fixed, lowerBound=-1000.0, upperBound=500.0)
    moving = sitk.Clamp(moving, lowerBound=-1000.0, upperBound=500.0)
    fixed_mask_image = _to_sitk(fixed_mask.astype(np.uint8), spacing, sitk.sitkUInt8)
    moving_mask_image = _to_sitk(moving_mask.astype(np.uint8), spacing, sitk.sitkUInt8)
    fixed = sitk.Mask(fixed, fixed_mask_image, outsideValue=-1000.0)
    moving = sitk.Mask(moving, moving_mask_image, outsideValue=-1000.0)

    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(256)
    matcher.SetNumberOfMatchPoints(10)
    matcher.ThresholdAtMeanIntensityOn()
    moving_matched = matcher.Execute(moving, fixed)

    field: sitk.Image | None = None
    metric = float("nan")
    for target_spacing, iterations in zip(
        params.level_spacing_mm, params.iterations, strict=True
    ):
        reference = _level_reference(fixed, target_spacing)
        fixed_level = _resample_scalar(fixed, reference)
        moving_level = _resample_scalar(moving_matched, reference)
        if field is None:
            level_field = _initial_field(reference, initial_transform)
        else:
            level_field = _resample_field(field, reference)

        demons = sitk.DiffeomorphicDemonsRegistrationFilter()
        demons.SetNumberOfIterations(iterations)
        demons.SetStandardDeviations(params.displacement_smoothing_sigma)
        demons.SetUpdateFieldStandardDeviations(params.update_smoothing_sigma)
        demons.SetMaximumUpdateStepLength(params.maximum_update_step_mm)
        field = demons.Execute(fixed_level, moving_level, level_field)
        metric = float(demons.GetMetric())

    if field is None:
        raise RuntimeError("registration produced no displacement field")
    full_field = _resample_field(field, fixed)
    transform = sitk.DisplacementFieldTransform(sitk.Image(full_field))
    moving_original = _to_sitk(moving_ras, spacing, sitk.sitkInt16)
    warped = sitk.Resample(
        moving_original, fixed, transform, sitk.sitkLinear, -1024.0, sitk.sitkInt16
    )
    warped_mask_image = sitk.Resample(
        moving_mask_image, fixed_mask_image, transform, sitk.sitkNearestNeighbor, 0,
        sitk.sitkUInt8,
    )
    jacobian_image = sitk.DisplacementFieldJacobianDeterminant(full_field)
    jacobian = sitk.GetArrayFromImage(jacobian_image)

    return DeformableResult(
        warped_moving=_from_sitk_int(warped),
        warped_moving_mask=_from_sitk_bool(warped_mask_image),
        displacement_mm=_from_sitk_vector(full_field),
        rigid_report=rigid_report,
        jacobian_min=float(np.min(jacobian)),
        jacobian_p01=float(np.percentile(jacobian, 1)),
        nonpositive_jacobian_fraction=float(np.mean(jacobian <= 0.0)),
        metric=metric,
    )


def register_bspline(
    fixed_ras: IntArray,
    moving_ras: IntArray,
    spacing: tuple[float, float, float],
    fixed_mask: BoolArray,
    moving_mask: BoolArray,
    *,
    params: BSplineParams | None = None,
) -> DeformableResult:
    """Register with a same-modality cubic B-spline free-form deformation."""
    if fixed_ras.shape != moving_ras.shape or fixed_ras.shape != fixed_mask.shape:
        raise ValueError("fixed, moving and masks must share one voxel grid")
    if moving_mask.shape != fixed_mask.shape:
        raise ValueError("fixed and moving masks must share one voxel grid")
    params = params or BSplineParams()

    fixed = _to_sitk(fixed_ras, spacing, sitk.sitkFloat32)
    moving = _to_sitk(moving_ras, spacing, sitk.sitkFloat32)
    fixed = sitk.Clamp(fixed, lowerBound=-1000.0, upperBound=500.0)
    moving = sitk.Clamp(moving, lowerBound=-1000.0, upperBound=500.0)
    fixed_mask_image = _to_sitk(fixed_mask.astype(np.uint8), spacing, sitk.sitkUInt8)
    moving_mask_image = _to_sitk(moving_mask.astype(np.uint8), spacing, sitk.sitkUInt8)
    optimization_reference = _level_reference(fixed, params.optimization_spacing_mm)
    fixed_optimization = _resample_scalar(fixed, optimization_reference)
    moving_optimization = _resample_scalar(moving, optimization_reference)
    fixed_mask_optimization = sitk.Resample(
        fixed_mask_image, optimization_reference, sitk.Transform(), sitk.sitkNearestNeighbor,
        0, sitk.sitkUInt8,
    )
    moving_mask_optimization = sitk.Resample(
        moving_mask_image, optimization_reference, sitk.Transform(),
        sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
    )
    fixed_optimization = sitk.Mask(
        fixed_optimization, fixed_mask_optimization, outsideValue=-1000.0
    )
    moving_optimization = sitk.Mask(
        moving_optimization, moving_mask_optimization, outsideValue=-1000.0
    )

    physical_size = np.asarray(optimization_reference.GetSize()) * np.asarray(
        optimization_reference.GetSpacing()
    )
    mesh_size = np.maximum(
        np.rint(physical_size / params.control_point_spacing_mm).astype(np.int64), 1
    )
    initial = sitk.BSplineTransformInitializer(
        optimization_reference, [int(value) for value in mesh_size], order=3
    )

    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(params.sampling_fraction, seed=20260730)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=params.iterations,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=params.iterations * 20,
        costFunctionConvergenceFactor=1e7,
    )
    method.SetShrinkFactorsPerLevel(list(params.shrink_factors))
    method.SetSmoothingSigmasPerLevel(list(params.smoothing_sigmas_mm))
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransformAsBSpline(
        initial, inPlace=False, scaleFactors=list(params.mesh_scale_factors)
    )
    transform = method.Execute(fixed_optimization, moving_optimization)
    full_field = sitk.TransformToDisplacementField(
        transform,
        sitk.sitkVectorFloat64,
        fixed.GetSize(),
        fixed.GetOrigin(),
        fixed.GetSpacing(),
        fixed.GetDirection(),
    )
    moving_original = _to_sitk(moving_ras, spacing, sitk.sitkInt16)
    warped = sitk.Resample(
        moving_original, fixed, transform, sitk.sitkLinear, -1024.0, sitk.sitkInt16
    )
    warped_mask_image = sitk.Resample(
        moving_mask_image, fixed_mask_image, transform, sitk.sitkNearestNeighbor, 0,
        sitk.sitkUInt8,
    )
    jacobian = sitk.GetArrayFromImage(
        sitk.DisplacementFieldJacobianDeterminant(full_field)
    )

    return DeformableResult(
        warped_moving=_from_sitk_int(warped),
        warped_moving_mask=_from_sitk_bool(warped_mask_image),
        displacement_mm=_from_sitk_vector(full_field),
        rigid_report=None,
        jacobian_min=float(np.min(jacobian)),
        jacobian_p01=float(np.percentile(jacobian, 1)),
        nonpositive_jacobian_fraction=float(np.mean(jacobian <= 0.0)),
        metric=float(method.GetMetricValue()),
    )


def mask_metrics(first: BoolArray, second: BoolArray, spacing: tuple[float, ...]) -> MaskMetrics:
    """Dice and symmetric surface distances in millimetres."""
    total = int(first.sum()) + int(second.sum())
    dice = 0.0 if total == 0 else 2.0 * float(np.logical_and(first, second).sum()) / total
    first_surface = binary_surface(first)
    second_surface = binary_surface(second)
    if not first_surface.any() or not second_surface.any():
        return MaskMetrics(dice=dice, surface_mean_mm=float("nan"), surface_p95_mm=float("nan"))
    to_first = ndimage.distance_transform_edt(~first_surface, sampling=spacing)
    to_second = ndimage.distance_transform_edt(~second_surface, sampling=spacing)
    distances = np.concatenate((to_second[first_surface], to_first[second_surface]))
    return MaskMetrics(
        dice=dice,
        surface_mean_mm=float(np.mean(distances)),
        surface_p95_mm=float(np.percentile(distances, 95)),
    )


def transform_points(
    points_mm: FloatArray,
    displacement_mm: VectorArray,
    spacing: tuple[float, float, float],
) -> FloatArray:
    """Sample a fixed-to-moving DVF at arbitrary physical points."""
    if points_mm.ndim != 2 or points_mm.shape[1] != 3:
        raise ValueError("points_mm must have shape N x 3")
    # The caller supplies physical points; the DVF grid uses a zero origin and
    # unit directions, like every RAS+ array produced by this package.
    field = sitk.GetImageFromArray(displacement_mm.transpose(2, 1, 0, 3).astype(np.float64))
    field.SetSpacing(spacing)
    transform = sitk.DisplacementFieldTransform(field)
    transformed = np.array(
        [transform.TransformPoint(tuple(float(value) for value in point)) for point in points_mm],
        dtype=np.float64,
    )
    return transformed


def landmark_tre(
    fixed_points_mm: FloatArray,
    moving_points_mm: FloatArray,
    displacement_mm: VectorArray | None = None,
    spacing: tuple[float, float, float] | None = None,
) -> FloatArray:
    """Euclidean landmark errors before or after applying a fixed-to-moving DVF."""
    if fixed_points_mm.shape != moving_points_mm.shape:
        raise ValueError("fixed and moving landmarks must have the same shape")
    if displacement_mm is not None and spacing is None:
        raise ValueError("spacing is required when a displacement field is supplied")
    predicted = fixed_points_mm
    if displacement_mm is not None and spacing is not None:
        predicted = transform_points(fixed_points_mm, displacement_mm, spacing)
    errors: FloatArray = np.linalg.norm(predicted - moving_points_mm, axis=1)
    return errors


def warp_mask(
    moving_mask: BoolArray,
    displacement_mm: VectorArray,
    spacing: tuple[float, float, float],
) -> BoolArray:
    """Resample a moving-space mask through a fixed-to-moving displacement field."""
    if displacement_mm.shape != moving_mask.shape + (3,):
        raise ValueError("displacement field and moving mask grids do not match")
    reference = _to_sitk(
        np.zeros(moving_mask.shape, dtype=np.uint8), spacing, sitk.sitkUInt8
    )
    source = _to_sitk(moving_mask.astype(np.uint8), spacing, sitk.sitkUInt8)
    field = sitk.GetImageFromArray(
        displacement_mm.transpose(2, 1, 0, 3).astype(np.float64)
    )
    field.SetSpacing(spacing)
    transform = sitk.DisplacementFieldTransform(field)
    warped = sitk.Resample(
        source,
        reference,
        transform,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    return _from_sitk_bool(warped)


__all__ = [
    "BSplineParams",
    "DeformableResult",
    "MaskMetrics",
    "RegistrationParams",
    "landmark_tre",
    "mask_metrics",
    "register_diffeomorphic",
    "register_bspline",
    "transform_points",
    "warp_mask",
]
