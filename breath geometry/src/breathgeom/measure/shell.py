"""Depth of the lung below the skin, as a field on the lung surface.

Both earlier attempts asked the question from the wrong end. Taking the shortest
skin-to-lung distance and then collapsing a sector to one ray made the answer
depend on the order statistic — the minimum and the median disagree on the sign
of the breathing change. Casting rays outwards from the body centroid was worse:
a ray that grazes the lung leaves it at once and then runs on through the axilla,
reporting a hundred millimetres of "wall".

The shortest distance from a point to a surface is perpendicular to that
surface, so a distance field already carries the normals. Nothing has to be
cast, and no sector has to be drawn:

* the depth field is the distance from every body voxel to the nearest voxel
  outside the body, i.e. how deep it lies under the skin;
* it is read on the *lung surface*, so every sample has lung beneath it by
  construction and a grazing geometry cannot arise;
* the lateral aspect is selected by where the shortest path actually points,
  not by an angle measured from a centroid that moves between phases.

The value read at a lung-surface voxel is the ``h`` of the two-layer impedance
model at that spot: the thickness of tissue between the skin and the lung.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from breathgeom.measure.wall import (
    FAT_HU,
    MUSCLE_HU,
    Side,
    WallParams,
    body_mask,
    fov_ring,
    lung_mask,
)

FloatArray = npt.NDArray[np.float64]
# Same alias as breathgeom.measure.wall so masks pass between the two modules.
IntArray = npt.NDArray[np.int_]
BoolArray = npt.NDArray[np.bool_]

# Above this a voxel is rib or vertebra, not soft tissue.
BONE_HU = 200


@dataclass(frozen=True)
class ShellParams:
    """Tunables of the depth measurement."""

    # Height of the measured belt above the lung base, fixed in millimetres so
    # that the same physical extent is compared in both phases. A fraction of
    # lung length would not be: at exhale the lung is shorter.
    band_height_mm: float = 60.0
    # Slices whose lung cross-section is smaller than this carry no wall.
    min_lung_area_mm2: float = 2000.0
    min_lung_component_px: int = 400
    min_body_area_px: int = 500
    fov_margin_px: float = 2.0
    # Keep surface points whose shortest way out is more sideways than it is
    # front-to-back. This is what "lateral wall" means without invoking a
    # centroid, and it survives the body shifting between phases.
    lateral_dominance: float = 1.0
    # Step used to classify tissue along the shortest path to the skin.
    composition_step_mm: float = 0.25
    # Deeper than this is not a chest wall but a path through the mediastinum.
    max_depth_mm: float = 120.0


@dataclass(frozen=True)
class ShellSample:
    """One lung-surface voxel and the tissue column standing above it."""

    slice_index: int
    depth_mm: float
    fat_mm: float
    muscle_mm: float
    bone_mm: float


@dataclass(frozen=True)
class Shell:
    """Depth field over the lateral lung surface, with its provenance."""

    side: Side
    samples: tuple[ShellSample, ...]
    slices_used: int
    surface_voxels_seen: int
    surface_voxels_kept: int
    band_first_slice: int | None
    band_last_slice: int | None
    fov_contact_fraction: float | None

    def _values(self, attribute: str) -> FloatArray:
        return np.array(
            [getattr(item, attribute) for item in self.samples], dtype=np.float64
        )

    @property
    def depth_mm(self) -> FloatArray:
        return self._values("depth_mm")

    @property
    def fat_mm(self) -> FloatArray:
        return self._values("fat_mm")

    @property
    def muscle_mm(self) -> FloatArray:
        return self._values("muscle_mm")

    @property
    def bone_mm(self) -> FloatArray:
        return self._values("bone_mm")

    def spread(self, values: FloatArray) -> tuple[float, float, float]:
        """p10, median and p90, or NaN when nothing was measured."""
        if values.size == 0:
            return (float("nan"),) * 3
        return (
            float(np.percentile(values, 10)),
            float(np.median(values)),
            float(np.percentile(values, 90)),
        )

    @property
    def kept_fraction(self) -> float:
        """Share of silhouette voxels whose way out really points sideways."""
        if self.surface_voxels_seen == 0:
            return float("nan")
        return self.surface_voxels_kept / self.surface_voxels_seen


def _lateral_silhouette(lung: BoolArray, side: Side) -> tuple[IntArray, IntArray]:
    """Outermost lung voxel of each column on the requested flank.

    In RAS+ rows increase towards the patient's right, so the right silhouette
    is the last lung row of every column and the left one is the first.
    """
    columns = np.flatnonzero(lung.any(axis=0))
    if columns.size == 0:
        empty: IntArray = np.empty(0, dtype=np.int_)
        return empty, empty
    if side is Side.RIGHT:
        rows = lung.shape[0] - 1 - np.argmax(lung[::-1, columns], axis=0)
    else:
        rows = np.argmax(lung[:, columns], axis=0)
    return rows.astype(np.int_), columns.astype(np.int_)


def _composition_mm(
    slice_hu: IntArray,
    start: tuple[int, int],
    end: tuple[int, int],
    spacing: tuple[float, float],
    params: ShellParams,
) -> tuple[float, float, float]:
    """Fat, muscle and bone length along the shortest path out of the body."""
    delta = np.array(
        [(end[0] - start[0]) * spacing[0], (end[1] - start[1]) * spacing[1]],
        dtype=np.float64,
    )
    length = float(np.linalg.norm(delta))
    steps = max(int(length / params.composition_step_mm), 1)
    fraction = np.linspace(0.0, 1.0, steps, endpoint=False)
    rows = np.clip(
        np.round(start[0] + fraction * (end[0] - start[0])).astype(int),
        0, slice_hu.shape[0] - 1,
    )
    columns = np.clip(
        np.round(start[1] + fraction * (end[1] - start[1])).astype(int),
        0, slice_hu.shape[1] - 1,
    )
    samples = slice_hu[rows, columns]
    per_sample = length / steps
    fat = float(np.count_nonzero((samples >= FAT_HU[0]) & (samples <= FAT_HU[1])))
    muscle = float(np.count_nonzero((samples >= MUSCLE_HU[0]) & (samples <= MUSCLE_HU[1])))
    bone = float(np.count_nonzero(samples > BONE_HU))
    return fat * per_sample, muscle * per_sample, bone * per_sample


def measure_shell(
    volume_ras: IntArray,
    spacing: tuple[float, float, float],
    side: Side = Side.RIGHT,
    params: ShellParams | None = None,
) -> Shell:
    """Read the skin-to-lung depth over the lateral surface of one lung.

    ``volume_ras`` must be in RAS+: axis 0 to the patient's right, axis 1
    anterior, axis 2 to the head. Slices are taken along axis 2.
    """
    params = params or ShellParams()
    sx, sy, sz = spacing
    in_plane = (sx, sy)
    pixel_area_mm2 = sx * sy
    mask_params = WallParams(
        min_lung_component_px=params.min_lung_component_px,
        min_body_area_px=params.min_body_area_px,
    )
    ring = fov_ring((volume_ras.shape[0], volume_ras.shape[1]), mask_params)
    outward = 1.0 if side is Side.RIGHT else -1.0

    bodies: dict[int, BoolArray] = {}
    lungs: dict[int, BoolArray] = {}
    touching = 0
    for index in range(volume_ras.shape[2]):
        slice_hu = volume_ras[:, :, index]
        body = body_mask(slice_hu, mask_params)
        if body is None:
            continue
        bodies[index] = body
        touching += int(bool((body & ring).any()))
        lung = lung_mask(slice_hu, body, side, mask_params)
        if lung is None or float(lung.sum()) * pixel_area_mm2 < params.min_lung_area_mm2:
            continue
        lungs[index] = lung

    contact = (touching / len(bodies)) if bodies else None
    if not lungs:
        return Shell(side, (), 0, 0, 0, None, None, contact)

    base = min(lungs)
    band = [i for i in sorted(lungs) if base <= i <= base + int(params.band_height_mm / sz)]

    samples: list[ShellSample] = []
    seen = 0
    for index in band:
        slice_hu = volume_ras[:, :, index]
        body, lung = bodies[index], lungs[index]

        # Distance from every body voxel to the nearest voxel outside the body,
        # plus the index of that voxel: the depth field and its normals.
        depth, nearest = ndimage.distance_transform_edt(
            body, sampling=in_plane, return_indices=True
        )
        # Only the lateral silhouette: for each anteroposterior position, the
        # outermost lung voxel on the chosen flank. The medial surface would
        # otherwise be sampled too, and its shortest way out runs straight
        # through the lung, reporting the whole thorax as wall.
        rows, columns = _lateral_silhouette(lung, side)
        if rows.size == 0:
            continue

        exit_rows = nearest[0, rows, columns]
        exit_columns = nearest[1, rows, columns]
        step_row = (exit_rows - rows) * sx
        step_column = (exit_columns - columns) * sy
        # The way out points sideways, towards the requested flank, more than it
        # points front-to-back: that is the lateral wall, defined by the
        # geometry itself rather than by an angle from a moving centroid.
        lateral = (outward * step_row > 0) & (
            np.abs(step_row) > params.lateral_dominance * np.abs(step_column)
        )
        seen += int(rows.size)

        for position in np.flatnonzero(lateral):
            row, column = int(rows[position]), int(columns[position])
            value = float(depth[row, column])
            if value <= 0.0 or value > params.max_depth_mm:
                continue
            fat, muscle, bone = _composition_mm(
                slice_hu, (row, column),
                (int(exit_rows[position]), int(exit_columns[position])),
                in_plane, params,
            )
            samples.append(
                ShellSample(
                    slice_index=index,
                    depth_mm=value,
                    fat_mm=fat,
                    muscle_mm=muscle,
                    bone_mm=bone,
                )
            )

    return Shell(
        side=side,
        samples=tuple(samples),
        slices_used=len(band),
        surface_voxels_seen=seen,
        surface_voxels_kept=len(samples),
        band_first_slice=band[0] if band else None,
        band_last_slice=band[-1] if band else None,
        fov_contact_fraction=contact,
    )


__all__ = ["BONE_HU", "Shell", "ShellParams", "ShellSample", "measure_shell"]
