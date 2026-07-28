"""Chest-wall soft-tissue thickness under a surface electrode array.

The quantity is the shortest skin-to-lung path inside a lateral angular sector:
that is what a tetrapolar surface array actually probes, and it is the ``h`` of
the two-layer impedance model. Apparent resistivity is sensitive to lung
resistivity roughly as ``S_rho2 ~ (h/a)^-3``, so an error in ``h`` propagates
into ``rho2`` amplified by the array geometry.

Measured geometrically, this is the whole soft-tissue envelope — muscle together
with subcutaneous fat and skin — not an isolated muscle layer. The fat and
muscle components are reported separately, but only on native (non-contrast)
scans is that split trustworthy: iodine shifts the muscle window.

Volumes are canonicalised to RAS+ before measurement, so the anatomical left and
right never depend on how the source file happened to be stored.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.typing as npt
from scipy import ndimage

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]
BoolArray = npt.NDArray[np.bool_]

# Everything above this is tissue: the outer body boundary.
BODY_HU = -300
# Everything below this inside the body is aerated lung.
AIR_HU = -400
# Standard body-composition windows. Valid on native scans only.
FAT_HU = (-190, -30)
MUSCLE_HU = (-29, 150)


class Side(StrEnum):
    """Which lateral chest wall is measured."""

    RIGHT = "right"
    LEFT = "left"


class Reduction(StrEnum):
    """How the per-slice distribution of skin-to-lung distances is collapsed.

    The distance transform already gives the shortest path to the lung for
    *every* skin pixel in the sector; the question is what to do with that
    vector. ``MIN`` answers "how close does the lung ever come to the skin",
    which is a biased-low order statistic: its bias depends on how many
    candidates there are and on the local curvature, and both change between
    inhale and exhale. ``MEDIAN`` answers "how thick is the wall across the
    sector", which is what an extended electrode array actually senses.
    """

    MIN = "min"
    MEDIAN = "median"


@dataclass(frozen=True)
class WallParams:
    """Tunables of the measurement, all in physical units where applicable."""

    # Half-width of the lateral sector, measured from the pure left-right ray.
    sector_half_angle_deg: float = 35.0
    # Slices whose lung cross-section is smaller than this are skipped: near the
    # costophrenic recess the sector holds no lung and the minimum runs off into
    # the axilla, which is not a chest wall.
    min_lung_area_mm2: float = 2000.0
    # Fraction of the lung's cranio-caudal extent counted as its lower part.
    lower_fraction: float = 1.0 / 3.0
    # Connected components smaller than this are bowel gas or noise, not lung.
    min_lung_component_px: int = 400
    # A body cross-section smaller than this is table, blanket or empty air.
    min_body_area_px: int = 500
    # Skin voxels closer than this to the reconstruction circle count as touching.
    fov_margin_px: float = 2.0
    # A sector holding fewer skin voxels than this is too small to trust.
    min_sector_px: int = 20
    # Step used to sample tissue composition along the measured ray.
    composition_step_mm: float = 0.25
    # How the sector's distance distribution becomes one number per slice.
    # MIN is the default only because it is the closer of the two to the h of
    # the two-layer impedance model; neither is a validated estimator, and the
    # two disagree on the sign of the inhale-to-exhale change.
    reduction: Reduction = Reduction.MIN


@dataclass(frozen=True)
class WallRay:
    """One shortest skin-to-lung measurement in a single axial slice."""

    slice_index: int
    thickness_mm: float
    fat_mm: float
    muscle_mm: float
    fov_truncated: bool


@dataclass(frozen=True)
class WallThickness:
    """Result of measuring one volume, with the provenance needed to judge it."""

    side: Side
    rays: tuple[WallRay, ...]
    lung_slice_first: int | None
    lung_slice_last: int | None
    lung_extent_mm: float | None
    slices_in_lower_part: int
    slices_skipped_small_lung: int
    fov_contact_fraction: float | None

    @property
    def thickness_mm(self) -> FloatArray:
        return np.array([ray.thickness_mm for ray in self.rays], dtype=np.float64)

    @property
    def fat_mm(self) -> FloatArray:
        return np.array([ray.fat_mm for ray in self.rays], dtype=np.float64)

    @property
    def muscle_mm(self) -> FloatArray:
        return np.array([ray.muscle_mm for ray in self.rays], dtype=np.float64)

    @property
    def truncated_ray_count(self) -> int:
        return sum(1 for ray in self.rays if ray.fov_truncated)

    def percentiles(self, values: FloatArray) -> tuple[float, float, float]:
        """Median with the 10th and 90th percentile, or NaN when empty."""
        if values.size == 0:
            return (float("nan"),) * 3
        return (
            float(np.percentile(values, 10)),
            float(np.median(values)),
            float(np.percentile(values, 90)),
        )


def load_ras(path: Path) -> tuple[IntArray, tuple[float, float, float]]:
    """Read a NIfTI volume and return it in RAS+ together with its voxel size.

    Canonicalising here is what keeps :func:`measure_wall` free of any assumption
    about storage order; a silent left-right flip would otherwise measure the
    wrong wall without failing.
    """
    # nibabel ships py.typed but leaves this helper unannotated
    canonical = nib.as_closest_canonical(nib.load(path))  # type: ignore[no-untyped-call]
    volume: IntArray = np.asarray(canonical.dataobj, dtype=np.int16)
    zooms = canonical.header.get_zooms()[:3]
    return volume, (float(zooms[0]), float(zooms[1]), float(zooms[2]))


def body_mask(slice_hu: IntArray, params: WallParams) -> BoolArray | None:
    """Largest tissue component of one slice, holes filled.

    Returns None when the slice holds no plausible body cross-section.
    """
    closed = ndimage.binary_closing(slice_hu > BODY_HU, np.ones((5, 5), dtype=bool))
    labels, count = ndimage.label(closed)
    if count == 0:
        return None
    sizes = ndimage.sum(closed, labels, range(1, count + 1))
    largest: BoolArray = labels == (int(np.argmax(sizes)) + 1)
    filled: BoolArray = ndimage.binary_fill_holes(largest)
    if int(filled.sum()) < params.min_body_area_px:
        return None
    return filled


def lung_mask(
    slice_hu: IntArray,
    body: BoolArray,
    side: Side,
    params: WallParams,
) -> BoolArray | None:
    """Aerated lung on one side of the midline, in a RAS+ slice.

    Axis 0 of a RAS+ volume increases towards the patient's right, so the right
    lung is the aerated component with the *largest* row centroid.
    """
    air = ndimage.binary_opening(body & (slice_hu < AIR_HU), np.ones((3, 3), dtype=bool))
    labels, count = ndimage.label(air)
    if count == 0:
        return None
    sizes = ndimage.sum(air, labels, range(1, count + 1))
    keep = [index + 1 for index in range(count) if sizes[index] > params.min_lung_component_px]
    if not keep:
        return None
    centres = ndimage.center_of_mass(air, labels, keep)
    rows = [float(np.atleast_1d(centre)[0]) for centre in centres]
    chosen = keep[int(np.argmax(rows) if side is Side.RIGHT else np.argmin(rows))]
    mask: BoolArray = np.asarray(labels == chosen, dtype=bool)
    return mask


def _fov_ring(shape: tuple[int, int], params: WallParams) -> BoolArray:
    """Outer band of the reconstruction circle inscribed in the slice."""
    rows, columns = shape
    grid_r, grid_c = np.mgrid[0:rows, 0:columns]
    radius = np.sqrt(
        (grid_r - (rows - 1) / 2.0) ** 2 + (grid_c - (columns - 1) / 2.0) ** 2
    )
    ring: BoolArray = radius > (min(rows, columns) / 2.0 - params.fov_margin_px)
    return ring


def _composition_mm(
    slice_hu: IntArray,
    start: tuple[float, float],
    end: tuple[float, float],
    spacing: tuple[float, float],
    params: WallParams,
) -> tuple[float, float]:
    """Fat and muscle length along the straight skin-to-lung segment."""
    delta = np.array(
        [(end[0] - start[0]) * spacing[0], (end[1] - start[1]) * spacing[1]],
        dtype=np.float64,
    )
    length = float(np.linalg.norm(delta))
    steps = max(int(length / params.composition_step_mm), 1)
    fraction = np.linspace(0.0, 1.0, steps, endpoint=False)
    rows = np.clip(
        np.round(start[0] + fraction * (end[0] - start[0])).astype(int),
        0,
        slice_hu.shape[0] - 1,
    )
    columns = np.clip(
        np.round(start[1] + fraction * (end[1] - start[1])).astype(int),
        0,
        slice_hu.shape[1] - 1,
    )
    samples = slice_hu[rows, columns]
    per_sample = length / steps
    fat = float(np.count_nonzero((samples >= FAT_HU[0]) & (samples <= FAT_HU[1]))) * per_sample
    muscle = (
        float(np.count_nonzero((samples >= MUSCLE_HU[0]) & (samples <= MUSCLE_HU[1])))
        * per_sample
    )
    return fat, muscle


def _lateral_sector(
    skin_rows: IntArray,
    skin_columns: IntArray,
    body_centre: tuple[float, float],
    side: Side,
    params: WallParams,
) -> BoolArray:
    """Skin voxels lying laterally on the requested side of the body centre.

    In RAS+ the row axis increases towards the patient's right, so an angle of 0
    points right and +-180 points left.
    """
    centre_r, centre_c = body_centre
    angle = np.degrees(np.arctan2(skin_columns - centre_c, skin_rows - centre_r))
    if side is Side.RIGHT:
        sector: BoolArray = np.abs(angle) < params.sector_half_angle_deg
    else:
        sector = np.abs(np.abs(angle) - 180.0) < params.sector_half_angle_deg
    return sector


def measure_wall(
    volume_ras: IntArray,
    spacing: tuple[float, float, float],
    side: Side = Side.RIGHT,
    params: WallParams | None = None,
) -> WallThickness:
    """Measure soft-tissue thickness over the lower part of one lung.

    ``volume_ras`` must already be in RAS+ (see :func:`load_ras`): axis 0 towards
    the patient's right, axis 2 towards the head. Slices are taken along axis 2.
    """
    params = params or WallParams()
    sx, sy, sz = spacing
    in_plane = (sx, sy)
    slice_count = volume_ras.shape[2]
    ring = _fov_ring((volume_ras.shape[0], volume_ras.shape[1]), params)
    pixel_area_mm2 = sx * sy

    bodies: dict[int, BoolArray] = {}
    lungs: dict[int, BoolArray] = {}
    areas: dict[int, float] = {}
    touching = 0
    for index in range(slice_count):
        slice_hu = volume_ras[:, :, index]
        body = body_mask(slice_hu, params)
        if body is None:
            continue
        bodies[index] = body
        touching += int(bool((body & ring).any()))
        lung = lung_mask(slice_hu, body, side, params)
        if lung is None:
            continue
        lungs[index] = lung
        areas[index] = float(lung.sum()) * pixel_area_mm2

    if not lungs:
        return WallThickness(
            side=side,
            rays=(),
            lung_slice_first=None,
            lung_slice_last=None,
            lung_extent_mm=None,
            slices_in_lower_part=0,
            slices_skipped_small_lung=0,
            fov_contact_fraction=(touching / len(bodies)) if bodies else None,
        )

    indices = sorted(lungs)
    first, last = indices[0], indices[-1]
    cutoff = first + int((last - first) * params.lower_fraction)
    lower = [index for index in indices if index <= cutoff]
    usable = [index for index in lower if areas[index] >= params.min_lung_area_mm2]

    rays: list[WallRay] = []
    for index in usable:
        ray = _measure_slice(
            volume_ras[:, :, index], bodies[index], lungs[index], index,
            in_plane, ring, side, params,
        )
        if ray is not None:
            rays.append(ray)

    return WallThickness(
        side=side,
        rays=tuple(rays),
        lung_slice_first=first,
        lung_slice_last=last,
        lung_extent_mm=(last - first) * sz,
        slices_in_lower_part=len(lower),
        slices_skipped_small_lung=len(lower) - len(usable),
        fov_contact_fraction=(touching / len(bodies)) if bodies else None,
    )


def _measure_slice(
    slice_hu: IntArray,
    body: BoolArray,
    lung: BoolArray,
    index: int,
    in_plane: tuple[float, float],
    ring: BoolArray,
    side: Side,
    params: WallParams,
) -> WallRay | None:
    """Shortest skin-to-lung path within the lateral sector of one slice."""
    distance, nearest = ndimage.distance_transform_edt(
        ~lung, sampling=in_plane, return_indices=True
    )
    skin: BoolArray = body & ~ndimage.binary_erosion(body, np.ones((3, 3), dtype=bool))
    skin_rows, skin_columns = np.where(skin)
    if skin_rows.size == 0:
        return None

    centre = ndimage.center_of_mass(body)
    sector = _lateral_sector(
        skin_rows, skin_columns, (float(centre[0]), float(centre[1])), side, params
    )
    if int(sector.sum()) < params.min_sector_px:
        return None

    candidate_rows = skin_rows[sector]
    candidate_columns = skin_columns[sector]
    reach = distance[candidate_rows, candidate_columns]
    if params.reduction is Reduction.MIN:
        best = int(np.argmin(reach))
    else:
        # The candidate sitting closest to the median: a real ray, so the
        # composition below is sampled along an actual path through tissue
        # rather than along a synthetic average that crosses nothing.
        best = int(np.argmin(np.abs(reach - np.median(reach))))
    start_r, start_c = int(candidate_rows[best]), int(candidate_columns[best])
    end_r = int(nearest[0, start_r, start_c])
    end_c = int(nearest[1, start_r, start_c])

    fat, muscle = _composition_mm(
        slice_hu, (start_r, start_c), (end_r, end_c), in_plane, params
    )
    return WallRay(
        slice_index=index,
        thickness_mm=float(reach[best]),
        fat_mm=fat,
        muscle_mm=muscle,
        fov_truncated=bool((body & ring).any()),
    )
