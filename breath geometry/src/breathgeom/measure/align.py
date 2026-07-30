"""Rigid alignment of two breath-holds on the skeleton.

Two breath-holds of one person differ by two things at once: the person lies
slightly differently, and the person breathed. Measuring the difference without
separating them mixes posture into the answer, and the master plan forbids that
outright (Этап 5).

The metric is restricted to the **vertebral column**. A whole-thorax intensity
metric would be dominated by the lung and the diaphragm, which are exactly what
is supposed to move. Plain bone is not enough either: ribs rotate with every
breath, so a metric that included them would absorb part of the effect into the
transform. The column is isolated by density, by lying posterior, and by staying
near the midline at every level.

Alignment does not merely tidy the comparison, it removes a confound that points
the same way as the effect being looked for: at exhale the diaphragm rises, so a
band anchored to the lung base slides cranially, and that slide correlates with
the depth of the exhalation just as the sought change does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import SimpleITK as sitk
from scipy import ndimage

from breathgeom.measure.wall import (
    SPINE_HU,
    BoolArray,
    IntArray,
    WallParams,
    anatomical_midline,
    body_mask,
)

FloatArray = npt.NDArray[np.float64]

# Half-width of the corridor kept around the midline when isolating the column.
COLUMN_HALF_WIDTH_MM = 45.0
# A pyramid level is only useful while enough masked voxels survive its
# downsampling; below this the joint histogram empties and the metric fails.
MIN_MASKED_VOXELS_PER_LEVEL = 500


@dataclass(frozen=True)
class AlignmentReport:
    """What the rigid fit actually did, so it can be judged rather than trusted."""

    translation_mm: tuple[float, float, float]
    rotation_deg: tuple[float, float, float]
    centre_mm: tuple[float, float, float]
    metric: float
    column_dice_before: float
    column_dice_after: float
    column_voxels_fixed: int
    column_voxels_moving: int

    @property
    def shift_mm(self) -> float:
        return float(np.linalg.norm(self.translation_mm))

    def rotation_matrix(self) -> FloatArray:
        """Euler rotation in the same X-Y-Z order SimpleITK composes it."""
        ax, ay, az = (np.radians(angle) for angle in self.rotation_deg)
        rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
        ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
        rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
        matrix: FloatArray = rz @ ry @ rx
        return matrix

    def to_fixed(self, points_mm: FloatArray) -> FloatArray:
        """Carry points from the moving frame into the fixed one.

        Resampling uses the transform the other way round — for every point of
        the fixed image it asks where to sample the moving one — so bringing
        moving-frame landmarks over requires the inverse. Getting the direction
        wrong roughly doubles the apparent error instead of removing it, which is
        why this is a named method rather than an inline matrix multiply.
        """
        centre = np.asarray(self.centre_mm, dtype=np.float64)
        shift = np.asarray(self.translation_mm, dtype=np.float64)
        rotation = self.rotation_matrix()
        moved = np.asarray(points_mm, dtype=np.float64) - centre - shift
        result: FloatArray = moved @ rotation + centre
        return result

    @property
    def improved(self) -> bool:
        """Did the two vertebral columns actually come into register?

        Overlap of the columns is checked rather than the optimiser's own
        metric: a mutual-information value says how well the fit went by its own
        standard, while Dice says whether the bone that was supposed to coincide
        now does. The latter is the claim being relied upon downstream.
        """
        return self.column_dice_after > self.column_dice_before


def _to_sitk(volume_ras: IntArray, spacing: tuple[float, float, float]) -> sitk.Image:
    """Wrap a RAS+ (x, y, z) numpy array as an image with physical spacing."""
    image = sitk.GetImageFromArray(np.ascontiguousarray(volume_ras.transpose(2, 1, 0)))
    image.SetSpacing(tuple(float(value) for value in spacing))
    return image


def _from_sitk(image: sitk.Image) -> IntArray:
    array: IntArray = sitk.GetArrayFromImage(image).transpose(2, 1, 0).astype(np.int16)
    return np.ascontiguousarray(array)


def pyramid_factors(masked_voxels: int) -> list[int]:
    """Shrink factors whose coarsest level still holds enough masked voxels.

    Each level divides the masked count by the cube of its factor. A level that
    empties the mask does not merely lose precision: Mattes mutual information
    becomes undefined and ITK raises rather than degrades.
    """
    factors = [
        factor
        for factor in (4, 2)
        if masked_voxels / float(factor) ** 3 >= MIN_MASKED_VOXELS_PER_LEVEL
    ]
    factors.append(1)
    return factors


def _dice(first: BoolArray, second: BoolArray) -> float:
    """Overlap of two masks, 0 when they are disjoint and 1 when identical."""
    total = int(first.sum()) + int(second.sum())
    if total == 0:
        return 0.0
    return 2.0 * float(np.logical_and(first, second).sum()) / float(total)


def _column_mask(
    volume_ras: IntArray,
    spacing: tuple[float, float, float],
    params: WallParams,
) -> BoolArray:
    """Vertebral column only: dense, posterior, and near the midline.

    Thresholding on density alone keeps the ribs, and ribs rotate with every
    breath — that is the motion under study. Registering on them would absorb
    part of the effect into the transform and quietly shrink the result. What
    separates the column is that it stays near the midline at every level.
    """
    midline = anatomical_midline(volume_ras, params)
    half_width = COLUMN_HALF_WIDTH_MM / spacing[0]

    mask = np.zeros(volume_ras.shape, dtype=bool)
    rows = np.arange(volume_ras.shape[0], dtype=np.float64)[:, None]
    near_midline = np.abs(rows - midline) <= half_width
    for index in range(volume_ras.shape[2]):
        slice_hu = volume_ras[:, :, index]
        body = body_mask(slice_hu, params)
        if body is None:
            continue
        anterior = np.where(body.any(axis=0))[0]
        # In RAS+ axis 1 points anterior, so the column sits in the low half.
        posterior_limit = int(0.5 * (float(anterior[0]) + float(anterior[-1])))
        bone = body & (slice_hu > SPINE_HU) & near_midline
        bone[:, posterior_limit:] = False
        mask[:, :, index] = bone

    labels, count = ndimage.label(mask)
    if count == 0:
        raise ValueError("No vertebral column found: the phases cannot be aligned")
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    largest: BoolArray = labels == (int(np.argmax(sizes)) + 1)
    return largest


def rigid_align(
    fixed_ras: IntArray,
    moving_ras: IntArray,
    spacing: tuple[float, float, float],
    sampling_fraction: float = 0.2,
    params: WallParams | None = None,
) -> tuple[IntArray, AlignmentReport]:
    """Bring ``moving`` into the frame of ``fixed`` by a rigid transform.

    Returns the resampled moving volume and a report of what was applied. Both
    inputs must already be RAS+ with the same voxel size.
    """
    params = params or WallParams()
    fixed = sitk.Cast(_to_sitk(fixed_ras, spacing), sitk.sitkFloat32)
    moving = sitk.Cast(_to_sitk(moving_ras, spacing), sitk.sitkFloat32)
    fixed_column = _column_mask(fixed_ras, spacing, params)
    moving_column = _column_mask(moving_ras, spacing, params)
    fixed_bone = _to_sitk(fixed_column.astype(np.int16), spacing)
    moving_bone = _to_sitk(moving_column.astype(np.int16), spacing)
    fixed_bone = sitk.Cast(fixed_bone, sitk.sitkUInt8)
    moving_bone = sitk.Cast(moving_bone, sitk.sitkUInt8)

    initial = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    method.SetMetricFixedMask(fixed_bone)
    method.SetMetricMovingMask(moving_bone)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(sampling_fraction, seed=20260728)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-4, numberOfIterations=200,
        gradientMagnitudeTolerance=1e-6,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    # The pyramid is cut to fit the column: each level divides the masked voxel
    # count by the cube of its shrink factor, and an empty mask makes Mattes
    # mutual information undefined rather than merely imprecise.
    factors = pyramid_factors(min(int(fixed_column.sum()), int(moving_column.sum())))
    method.SetShrinkFactorsPerLevel(factors)
    method.SetSmoothingSigmasPerLevel([float(f) / 2.0 for f in factors[:-1]] + [0.0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(initial, inPlace=False)

    transform = method.Execute(fixed, moving)

    # Execute returns a composite when the initial transform is not in place;
    # the rigid part is its first component.
    composite = sitk.CompositeTransform(transform)
    euler = sitk.Euler3DTransform(composite.GetNthTransform(0))
    resampled = sitk.Resample(
        moving, fixed, transform, sitk.sitkLinear, -1024.0, moving.GetPixelID()
    )
    moved_column = sitk.Resample(
        moving_bone, fixed_bone, transform, sitk.sitkNearestNeighbor, 0, moving_bone.GetPixelID()
    )
    moved_column_array = sitk.GetArrayViewFromImage(moved_column).transpose(2, 1, 0) > 0

    report = AlignmentReport(
        translation_mm=tuple(float(value) for value in euler.GetTranslation()),  # type: ignore[arg-type]
        rotation_deg=(
            float(np.degrees(euler.GetAngleX())),
            float(np.degrees(euler.GetAngleY())),
            float(np.degrees(euler.GetAngleZ())),
        ),
        centre_mm=tuple(float(value) for value in euler.GetCenter()),  # type: ignore[arg-type]
        metric=float(method.GetMetricValue()),
        column_dice_before=_dice(fixed_column, moving_column),
        column_dice_after=_dice(fixed_column, moved_column_array),
        column_voxels_fixed=int(fixed_column.sum()),
        column_voxels_moving=int(moving_column.sum()),
    )
    return _from_sitk(resampled), report


__all__ = ["AlignmentReport", "pyramid_factors", "rigid_align"]
