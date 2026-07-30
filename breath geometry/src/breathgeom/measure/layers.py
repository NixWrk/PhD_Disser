"""Exploratory over-rib thickness of fat and muscle in the chest wall.

This answers a different question from :mod:`breathgeom.measure.shell`, and a
better-posed one. That module reports how deep the lung lies under the skin,
which is the ``h`` of the two-layer impedance model — but it is anchored to the
lung, and the lung is the thing that moves. Its inner boundary is set by a
Hounsfield threshold whose position shifts as the lung densifies, and at exhale
the lung retreats out of any fixed measurement belt altogether.

This module deliberately measures only walks that actually reach bone. It is
therefore an *over-rib* diagnostic, not the full skin-to-lung profile needed by
the bioimpedance model. Intercostal walks and current paths that continue behind
a rib require a separate tissue-profile measurement.

Each layer is read along the inward normal, taken as the gradient of the
distance-to-outside field: the shortest way out of the body is perpendicular to
the skin, so that field already carries the surface normals and nothing has to
be cast from a centroid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from breathgeom.measure.wall import (
    AIR_HU,
    FAT_HU,
    MUSCLE_HU,
    Side,
    WallParams,
    anatomical_midline,
    body_mask,
    fov_ring,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]
BoolArray = npt.NDArray[np.bool_]

# Above this the walk has reached rib or vertebra and the soft wall is over.
BONE_HU = 200


@dataclass(frozen=True)
class LayerParams:
    """Tunables of the layer measurement."""

    # Step along the inward normal.
    step_mm: float = 0.25
    # Skip this much at the start: the outermost voxels are part skin, part air,
    # and their Hounsfield values belong to neither.
    skin_skip_mm: float = 1.0
    # Give up after this depth; beyond it the walk is inside the thorax, not the
    # wall, and whatever it finds is not a chest-wall layer.
    max_walk_mm: float = 60.0
    # A run shorter than this is speckle rather than a layer.
    min_layer_mm: float = 0.75
    # Single stray voxels inside a layer must not end it: partial volume puts
    # occasional muscle-valued voxels inside fat and vice versa.
    bridge_mm: float = 1.0
    # Half-width of the sampled sector. At 180 the whole circumference is taken,
    # which is what angular binning of the wall needs.
    sector_half_angle_deg: float = 35.0
    min_body_area_px: int = 500
    fov_margin_px: float = 2.0


@dataclass(frozen=True)
class LayerSample:
    """One inward walk from a skin voxel."""

    slice_index: int
    angle_deg: float
    fat_mm: float
    muscle_mm: float
    reached_bone: bool
    fov_truncated: bool

    @property
    def soft_mm(self) -> float:
        return self.fat_mm + self.muscle_mm


@dataclass(frozen=True)
class Layers:
    """Fat and muscle thickness over a stretch of chest wall."""

    side: Side
    samples: tuple[LayerSample, ...]
    slices_used: int
    walks_attempted: int
    band_first_slice: int | None
    band_last_slice: int | None
    fov_contact_fraction: float | None

    def _values(self, attribute: str) -> FloatArray:
        return np.array(
            [getattr(item, attribute) for item in self.samples], dtype=np.float64
        )

    @property
    def fat_mm(self) -> FloatArray:
        return self._values("fat_mm")

    @property
    def muscle_mm(self) -> FloatArray:
        return self._values("muscle_mm")

    @property
    def soft_mm(self) -> FloatArray:
        return self._values("soft_mm")

    @property
    def angle_deg(self) -> FloatArray:
        return self._values("angle_deg")

    @property
    def bone_fraction(self) -> float:
        """Share of retained walks that ended on a rib.

        Valid output from this module has fraction 1.0. The property remains in
        the result as a runtime invariant and for backwards-compatible QC.
        """
        if not self.samples:
            return float("nan")
        return sum(1 for item in self.samples if item.reached_bone) / len(self.samples)

    def spread(self, values: FloatArray) -> tuple[float, float, float]:
        """p10, median and p90, or NaN when nothing was measured."""
        if values.size == 0:
            return (float("nan"),) * 3
        return (
            float(np.percentile(values, 10)),
            float(np.median(values)),
            float(np.percentile(values, 90)),
        )


def _runs(labels: npt.NDArray[np.int8], target: int, bridge: int) -> int:
    """Length of the leading run of ``target``, tolerating short interruptions."""
    total = 0
    gap = 0
    for value in labels:
        if value == target:
            total += gap + 1
            gap = 0
        elif value == 0 or gap < bridge:
            gap += 1
            if gap > bridge:
                break
        else:
            break
    return total


def _classify(samples: IntArray) -> npt.NDArray[np.int8]:
    """1 for fat, 2 for muscle, 3 for bone, 0 for anything else."""
    labels = np.zeros(samples.shape, dtype=np.int8)
    labels[(samples >= FAT_HU[0]) & (samples <= FAT_HU[1])] = 1
    labels[(samples >= MUSCLE_HU[0]) & (samples <= MUSCLE_HU[1])] = 2
    labels[samples > BONE_HU] = 3
    return labels


def measure_layers(
    volume_ras: IntArray,
    spacing: tuple[float, float, float],
    side: Side = Side.RIGHT,
    params: LayerParams | None = None,
    band: tuple[int, int] | None = None,
    origin: tuple[float, float] | None = None,
) -> Layers:
    """Walk inward from the skin and measure fat and muscle above a rib.

    ``volume_ras`` must be RAS+. ``band`` fixes the slice range; two phases may
    only share one after being brought to a common frame.

    ``origin`` fixes the point angles are measured from. Comparing two phases
    by angular sector requires it: the body centroid shifts by more than ten
    millimetres between inhale and exhale, so a sector defined per phase names a
    different piece of wall in each. One origin, taken from either phase, keeps
    the sectors meaning the same thing.
    """
    params = params or LayerParams()
    sx, sy, sz = spacing
    mask_params = WallParams(
        min_body_area_px=params.min_body_area_px, fov_margin_px=params.fov_margin_px
    )
    ring = fov_ring((volume_ras.shape[0], volume_ras.shape[1]), mask_params)
    midline = anatomical_midline(volume_ras, mask_params)
    outward = 1.0 if side is Side.RIGHT else -1.0

    steps = int(params.max_walk_mm / params.step_mm)
    skip = int(params.skin_skip_mm / params.step_mm)
    bridge = max(int(params.bridge_mm / params.step_mm), 1)
    minimum = params.min_layer_mm

    first, last = band if band is not None else (0, volume_ras.shape[2] - 1)
    samples: list[LayerSample] = []
    attempted = 0
    used = 0
    touching = 0
    bodies = 0

    for index in range(volume_ras.shape[2]):
        slice_hu = volume_ras[:, :, index]
        body = body_mask(slice_hu, mask_params)
        if body is None:
            continue
        bodies += 1
        truncated = bool((body & ring).any())
        touching += int(truncated)
        if not (first <= index <= last):
            continue
        used += 1

        depth = ndimage.distance_transform_edt(body, sampling=(sx, sy))
        # The gradient of the depth field is the inward surface normal.
        grad_row, grad_column = np.gradient(depth, sx, sy)
        skin = body & ~ndimage.binary_erosion(body, np.ones((3, 3), dtype=bool))
        rows, columns = np.where(skin)
        if rows.size == 0:
            continue

        centre = origin if origin is not None else ndimage.center_of_mass(body)
        angle = np.degrees(np.arctan2(columns - centre[1], rows - centre[0]))
        if params.sector_half_angle_deg >= 180.0:
            sector = np.ones(rows.shape, dtype=bool)
        else:
            if side is Side.RIGHT:
                sector = np.abs(angle) < params.sector_half_angle_deg
            else:
                sector = np.abs(np.abs(angle) - 180.0) < params.sector_half_angle_deg
            sector &= (outward * (rows - midline)) > 0

        for position in np.flatnonzero(sector):
            row, column = int(rows[position]), int(columns[position])
            direction = np.array(
                [grad_row[row, column], grad_column[row, column]], dtype=np.float64
            )
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                continue
            direction /= norm
            attempted += 1

            offsets = np.arange(steps, dtype=np.float64) * params.step_mm
            walk_rows = np.clip(
                np.round(row + offsets * direction[0] / sx).astype(int),
                0, body.shape[0] - 1,
            )
            walk_columns = np.clip(
                np.round(column + offsets * direction[1] / sy).astype(int),
                0, body.shape[1] - 1,
            )
            values = slice_hu[walk_rows, walk_columns][skip:]
            if values.size == 0:
                continue
            # The wall ends where air begins: beyond that is lung, not wall.
            air = np.flatnonzero(values < AIR_HU)
            if air.size:
                values = values[: air[0]]
            labels = _classify(values)
            bone_at = np.flatnonzero(labels == 3)
            # Without a detected rib there is no defined inner boundary for
            # this measurement. Keeping such a walk silently turns liver,
            # intercostal space or the walk budget into a fictitious endpoint.
            if not bone_at.size:
                continue
            labels = labels[: bone_at[0]]

            fat = _runs(labels, 1, bridge) * params.step_mm
            after = int(fat / params.step_mm)
            muscle = _runs(labels[after:], 2, bridge) * params.step_mm
            if fat < minimum and muscle < minimum:
                continue
            samples.append(
                LayerSample(
                    slice_index=index,
                    angle_deg=float(angle[position]),
                    fat_mm=fat if fat >= minimum else 0.0,
                    muscle_mm=muscle if muscle >= minimum else 0.0,
                    reached_bone=True,
                    fov_truncated=truncated,
                )
            )

    return Layers(
        side=side,
        samples=tuple(samples),
        slices_used=used,
        walks_attempted=attempted,
        band_first_slice=first if samples else None,
        band_last_slice=last if samples else None,
        fov_contact_fraction=(touching / bodies) if bodies else None,
    )


__all__ = ["BONE_HU", "LayerParams", "LayerSample", "Layers", "measure_layers"]
