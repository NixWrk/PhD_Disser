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
class FOVAwareMaskMetrics:
    """Mask agreement restricted to anatomy observable in both acquisitions."""

    dice: float
    surface_mean_mm: float
    surface_p95_mm: float
    valid_voxel_fraction: float
    first_surface_coverage: float
    second_surface_coverage: float
    first_surface_count: int
    second_surface_count: int


@dataclass(frozen=True)
class JacobianMetrics:
    """Jacobian determinant summary on an explicit evaluation domain."""

    minimum: float
    p01: float
    nonpositive_fraction: float
    voxel_count: int


@dataclass(frozen=True)
class RoundTripMetrics:
    """Residual of forward followed by inverse displacement in millimetres."""

    voxel_count: int
    mean_mm: float
    p95_mm: float
    maximum_mm: float


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
    elapsed_s: float = float("nan")


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


def acquisition_fov_mask(
    body_mask: BoolArray,
    *,
    axis: int = 2,
    minimum_slice_fraction: float = 0.01,
) -> BoolArray:
    """Infer the acquired axial slab from non-empty body-mask slices.

    Learn2Reg volumes share a padded array grid even when one acquisition
    covers only part of the thorax.  A full-one array would therefore
    incorrectly mark padding as observed anatomy.  The body support provides
    a conservative, image-derived acquisition slab without looking at
    registration landmarks.
    """
    if body_mask.ndim != 3:
        raise ValueError("body_mask must be three-dimensional")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    if not 0.0 <= minimum_slice_fraction < 1.0:
        raise ValueError("minimum_slice_fraction must lie in [0, 1)")

    reduce_axes = tuple(index for index in range(3) if index != axis)
    counts = np.asarray(body_mask.sum(axis=reduce_axes), dtype=np.int64)
    maximum = int(counts.max(initial=0))
    if maximum == 0:
        raise ValueError("cannot infer acquisition FOV from an empty body mask")
    threshold = max(1, int(np.ceil(maximum * minimum_slice_fraction)))
    active = np.flatnonzero(counts >= threshold)
    if active.size == 0:
        raise ValueError("no body slices satisfy the acquisition FOV threshold")

    selector = [slice(None), slice(None), slice(None)]
    selector[axis] = slice(int(active[0]), int(active[-1]) + 1)
    result = np.zeros(body_mask.shape, dtype=bool)
    result[tuple(selector)] = True
    return np.ascontiguousarray(result)


def _safe_fov_domain(
    valid_domain: BoolArray,
    spacing: tuple[float, float, float],
    boundary_margin_mm: float,
) -> BoolArray:
    if boundary_margin_mm < 0:
        raise ValueError("boundary_margin_mm must be non-negative")
    safe = np.asarray(valid_domain, dtype=bool).copy()
    for axis in range(3):
        lower: list[slice | int] = [slice(None), slice(None), slice(None)]
        upper: list[slice | int] = [slice(None), slice(None), slice(None)]
        lower[axis] = 0
        upper[axis] = -1
        safe[tuple(lower)] = False
        safe[tuple(upper)] = False
    if boundary_margin_mm == 0:
        return np.ascontiguousarray(safe)
    distance_to_boundary = ndimage.distance_transform_edt(safe, sampling=spacing)
    return np.ascontiguousarray(distance_to_boundary > boundary_margin_mm)


def fov_aware_mask_metrics(
    first: BoolArray,
    second: BoolArray,
    spacing: tuple[float, float, float],
    *,
    valid_domain: BoolArray,
    boundary_margin_mm: float = 5.0,
) -> FOVAwareMaskMetrics:
    """Dice and surface distance inside a shared observable physical domain.

    Surfaces close to either acquisition boundary are excluded so that a
    padded or truncated scan does not create a false anatomical surface.
    Coverage is reported explicitly; an empty observable surface returns NaN
    distances and must fail a downstream gate.
    """
    if first.shape != second.shape or first.shape != valid_domain.shape:
        raise ValueError("masks and valid_domain must share one voxel grid")
    if len(spacing) != 3 or any(value <= 0 for value in spacing):
        raise ValueError("spacing must contain three positive values")

    domain = np.asarray(valid_domain, dtype=bool)
    first_observed = np.logical_and(first, domain)
    second_observed = np.logical_and(second, domain)
    total = int(first_observed.sum()) + int(second_observed.sum())
    dice = (
        0.0
        if total == 0
        else 2.0 * float(np.logical_and(first_observed, second_observed).sum()) / total
    )

    first_surface_all = binary_surface(first)
    second_surface_all = binary_surface(second)
    safe_domain = _safe_fov_domain(domain, spacing, boundary_margin_mm)
    first_surface = np.logical_and(first_surface_all, safe_domain)
    second_surface = np.logical_and(second_surface_all, safe_domain)
    first_total = int(first_surface_all.sum())
    second_total = int(second_surface_all.sum())
    first_count = int(first_surface.sum())
    second_count = int(second_surface.sum())
    first_coverage = 0.0 if first_total == 0 else first_count / first_total
    second_coverage = 0.0 if second_total == 0 else second_count / second_total

    if first_count == 0 or second_count == 0:
        mean = float("nan")
        p95 = float("nan")
    else:
        to_first = ndimage.distance_transform_edt(~first_surface, sampling=spacing)
        to_second = ndimage.distance_transform_edt(~second_surface, sampling=spacing)
        distances = np.concatenate((to_second[first_surface], to_first[second_surface]))
        mean = float(np.mean(distances))
        p95 = float(np.percentile(distances, 95))

    return FOVAwareMaskMetrics(
        dice=dice,
        surface_mean_mm=mean,
        surface_p95_mm=p95,
        valid_voxel_fraction=float(domain.mean()),
        first_surface_coverage=float(first_coverage),
        second_surface_coverage=float(second_coverage),
        first_surface_count=first_count,
        second_surface_count=second_count,
    )


def jacobian_metrics(
    displacement_mm: VectorArray,
    spacing: tuple[float, float, float],
    *,
    valid_domain: BoolArray,
) -> JacobianMetrics:
    """Evaluate physical field invertibility only where anatomy is observable."""
    if displacement_mm.ndim != 4 or displacement_mm.shape[-1] != 3:
        raise ValueError("displacement_mm must have shape X x Y x Z x 3")
    if displacement_mm.shape[:3] != valid_domain.shape:
        raise ValueError("displacement field and valid_domain grids do not match")
    if not valid_domain.any():
        raise ValueError("Jacobian evaluation domain is empty")
    field = sitk.GetImageFromArray(
        displacement_mm.transpose(2, 1, 0, 3).astype(np.float64)
    )
    field.SetSpacing(spacing)
    determinant = sitk.GetArrayFromImage(
        sitk.DisplacementFieldJacobianDeterminant(field)
    ).transpose(2, 1, 0)
    values = determinant[valid_domain]
    return JacobianMetrics(
        minimum=float(np.min(values)),
        p01=float(np.percentile(values, 1)),
        nonpositive_fraction=float(np.mean(values <= 0.0)),
        voxel_count=int(len(values)),
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


def warp_image(
    moving_ras: IntArray,
    displacement_mm: VectorArray,
    spacing: tuple[float, float, float],
) -> npt.NDArray[np.int16]:
    """Resample a moving CT through a fixed-to-moving displacement field."""
    if displacement_mm.shape != moving_ras.shape + (3,):
        raise ValueError("displacement field and moving image grids do not match")
    reference = _to_sitk(
        np.zeros(moving_ras.shape, dtype=np.int16), spacing, sitk.sitkInt16
    )
    source = _to_sitk(moving_ras, spacing, sitk.sitkInt16)
    field = sitk.GetImageFromArray(
        displacement_mm.transpose(2, 1, 0, 3).astype(np.float64)
    )
    field.SetSpacing(spacing)
    transform = sitk.DisplacementFieldTransform(field)
    warped = sitk.Resample(
        source,
        reference,
        transform,
        sitk.sitkLinear,
        -1024,
        sitk.sitkInt16,
    )
    return _from_sitk_int(warped)


def compose_displacements(
    base_fixed_to_moving_mm: VectorArray,
    residual_fixed_to_intermediate_mm: VectorArray,
    spacing: tuple[float, float, float],
) -> VectorArray:
    """Compose ``base(residual(x))`` fields on one zero-origin physical grid.

    If ``base`` maps fixed coordinates into the original moving image and the
    residual maps fixed coordinates into an intermediate image already warped
    by ``base``, the combined displacement is
    ``residual(x) + base(x + residual(x))``.
    """
    if base_fixed_to_moving_mm.shape != residual_fixed_to_intermediate_mm.shape:
        raise ValueError("displacement fields must share one grid")
    if base_fixed_to_moving_mm.ndim != 4 or base_fixed_to_moving_mm.shape[3] != 3:
        raise ValueError("displacement fields must have shape X x Y x Z x 3")
    shape = base_fixed_to_moving_mm.shape[:3]
    x_grid, y_grid = np.meshgrid(
        np.arange(shape[0], dtype=np.float64),
        np.arange(shape[1], dtype=np.float64),
        indexing="ij",
    )
    combined = np.empty_like(base_fixed_to_moving_mm, dtype=np.float32)
    spacing_array = np.asarray(spacing, dtype=np.float64)
    for z_index in range(shape[2]):
        residual = residual_fixed_to_intermediate_mm[:, :, z_index].astype(np.float64)
        coordinates = np.stack(
            (
                x_grid + residual[..., 0] / spacing_array[0],
                y_grid + residual[..., 1] / spacing_array[1],
                np.full(shape[:2], z_index, dtype=np.float64)
                + residual[..., 2] / spacing_array[2],
            )
        )
        for component in range(3):
            sampled = ndimage.map_coordinates(
                base_fixed_to_moving_mm[..., component],
                coordinates,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
            combined[:, :, z_index, component] = (
                residual[..., component] + sampled
            ).astype(np.float32)
    return np.ascontiguousarray(combined)


def invert_displacement(
    forward_mm: VectorArray,
    spacing: tuple[float, float, float],
    *,
    maximum_iterations: int = 100,
    maximum_error_tolerance_mm: float = 0.01,
    mean_error_tolerance_mm: float = 0.001,
) -> VectorArray:
    """Numerically invert a same-grid physical displacement field.

    If ``forward_mm`` maps fixed coordinates to moving coordinates, the
    returned field maps moving coordinates back to fixed coordinates.
    """
    if forward_mm.ndim != 4 or forward_mm.shape[-1] != 3:
        raise ValueError("forward_mm must have shape X x Y x Z x 3")
    if len(spacing) != 3 or any(value <= 0 for value in spacing):
        raise ValueError("spacing must contain three positive values")
    if maximum_iterations <= 0:
        raise ValueError("maximum_iterations must be positive")
    if maximum_error_tolerance_mm <= 0 or mean_error_tolerance_mm <= 0:
        raise ValueError("inversion tolerances must be positive")

    field = sitk.GetImageFromArray(
        forward_mm.transpose(2, 1, 0, 3).astype(np.float64)
    )
    field.SetSpacing(spacing)
    inverse = sitk.InvertDisplacementField(
        field,
        maximumNumberOfIterations=maximum_iterations,
        maxErrorToleranceThreshold=maximum_error_tolerance_mm,
        meanErrorToleranceThreshold=mean_error_tolerance_mm,
        enforceBoundaryCondition=False,
    )
    return _from_sitk_vector(inverse)


def displacement_round_trip_metrics(
    forward_mm: VectorArray,
    inverse_mm: VectorArray,
    spacing: tuple[float, float, float],
    *,
    valid_domain: BoolArray,
) -> RoundTripMetrics:
    """Measure ``forward(x) + inverse(x + forward(x))`` on an explicit domain."""
    if forward_mm.shape != inverse_mm.shape:
        raise ValueError("forward and inverse fields must share one grid")
    if forward_mm.shape[:-1] != valid_domain.shape:
        raise ValueError("valid_domain must match the displacement grid")
    if not np.any(valid_domain):
        raise ValueError("valid_domain is empty")
    residual = compose_displacements(inverse_mm, forward_mm, spacing)
    error = np.linalg.norm(residual, axis=-1)[valid_domain]
    return RoundTripMetrics(
        voxel_count=len(error),
        mean_mm=float(np.mean(error)),
        p95_mm=float(np.percentile(error, 95)),
        maximum_mm=float(np.max(error)),
    )


__all__ = [
    "BSplineParams",
    "DeformableResult",
    "FOVAwareMaskMetrics",
    "JacobianMetrics",
    "MaskMetrics",
    "RegistrationParams",
    "RoundTripMetrics",
    "acquisition_fov_mask",
    "displacement_round_trip_metrics",
    "fov_aware_mask_metrics",
    "invert_displacement",
    "jacobian_metrics",
    "landmark_tre",
    "mask_metrics",
    "compose_displacements",
    "register_diffeomorphic",
    "register_bspline",
    "transform_points",
    "warp_mask",
    "warp_image",
]
