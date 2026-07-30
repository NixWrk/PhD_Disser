"""Whole-lung-surface skin-to-lung tissue profiles in physical 3D space.

Unlike the legacy lateral-wall estimator, this module has no electrode sector.
It samples the entire segmented pleural surface, follows the shortest straight
path to the exterior of the filled body mask, and partitions the path into fat,
muscle, bone and other tissue.  A profile that crosses aerated lung again or
ends at a truncated field of view is rejected rather than silently shortened.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from breathgeom.measure.registration import transform_points
from breathgeom.measure.segmentation import binary_surface
from breathgeom.measure.wall import FAT_HU, MUSCLE_HU, IntArray

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]
VectorArray = npt.NDArray[np.float32]


@dataclass(frozen=True)
class ProfileParams:
    """Sampling and QC thresholds, all distances in millimetres."""

    surface_spacing_mm: float = 8.0
    line_step_mm: float = 0.75
    min_thickness_mm: float = 2.0
    max_thickness_mm: float = 150.0
    max_initial_lung_mm: float = 4.0
    max_lung_reentry_mm: float = 1.5
    mapped_surface_tolerance_mm: float = 6.0
    fov_margin_mm: float = 2.0
    bone_lower_hu: float = 151.0

    def __post_init__(self) -> None:
        positive = (
            self.surface_spacing_mm,
            self.line_step_mm,
            self.min_thickness_mm,
            self.max_thickness_mm,
            self.max_initial_lung_mm,
            self.max_lung_reentry_mm,
            self.mapped_surface_tolerance_mm,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("profile distances must be positive")
        if self.max_thickness_mm <= self.min_thickness_mm:
            raise ValueError("max_thickness_mm must exceed min_thickness_mm")
        if self.fov_margin_mm < 0:
            raise ValueError("fov_margin_mm cannot be negative")


@dataclass(frozen=True)
class TissueProfile:
    """One lung-surface to skin path; invalid rows retain their QC reason."""

    profile_id: int
    valid: bool
    reason: str
    region: str
    lung_x_mm: float
    lung_y_mm: float
    lung_z_mm: float
    skin_x_mm: float
    skin_y_mm: float
    skin_z_mm: float
    thickness_mm: float
    soft_tissue_mm: float
    fat_mm: float
    muscle_mm: float
    bone_mm: float
    other_mm: float


@dataclass(frozen=True)
class ProfileSet:
    coordinate_basis: str
    coverage: str
    profiles: tuple[TissueProfile, ...]

    @property
    def valid_count(self) -> int:
        return sum(profile.valid for profile in self.profiles)


@dataclass(frozen=True)
class PairedProfile:
    profile_id: int
    region: str
    valid: bool
    reason: str
    fixed_thickness_mm: float
    moving_thickness_mm: float
    delta_moving_minus_fixed_mm: float
    fixed_fat_mm: float
    moving_fat_mm: float
    delta_fat_mm: float
    fixed_muscle_mm: float
    moving_muscle_mm: float
    delta_muscle_mm: float
    fixed_bone_mm: float
    moving_bone_mm: float


@dataclass(frozen=True)
class PairedProfileSet:
    transform_direction: str
    profiles: tuple[PairedProfile, ...]

    @property
    def valid_count(self) -> int:
        return sum(profile.valid for profile in self.profiles)


@dataclass(frozen=True)
class ProfileSummary:
    candidate_count: int
    valid_count: int
    invalid_reasons: dict[str, int]
    region_counts: dict[str, int]
    thickness_p10_mm: float
    thickness_median_mm: float
    thickness_p90_mm: float
    fat_median_mm: float
    muscle_median_mm: float
    bone_median_mm: float


def _invalid(profile_id: int, point_mm: FloatArray, reason: str) -> TissueProfile:
    nan = float("nan")
    return TissueProfile(
        profile_id=profile_id,
        valid=False,
        reason=reason,
        region="unknown",
        lung_x_mm=float(point_mm[0]),
        lung_y_mm=float(point_mm[1]),
        lung_z_mm=float(point_mm[2]),
        skin_x_mm=nan,
        skin_y_mm=nan,
        skin_z_mm=nan,
        thickness_mm=nan,
        soft_tissue_mm=nan,
        fat_mm=nan,
        muscle_mm=nan,
        bone_mm=nan,
        other_mm=nan,
    )


def _region(vector_mm: FloatArray) -> str:
    axis = int(np.argmax(np.abs(vector_mm)))
    if axis == 0:
        return "right" if vector_mm[0] > 0 else "left"
    if axis == 1:
        return "anterior" if vector_mm[1] > 0 else "posterior"
    return "superior" if vector_mm[2] > 0 else "inferior"


def _surface_indices(lung_mask: BoolArray, spacing: tuple[float, float, float],
                     surface_spacing_mm: float) -> npt.NDArray[np.int64]:
    indices = np.argwhere(binary_surface(lung_mask)).astype(np.int64)
    if not len(indices):
        return indices
    points_mm = indices * np.asarray(spacing)
    bins = np.floor(points_mm / surface_spacing_mm).astype(np.int64)
    _, inverse = np.unique(bins, axis=0, return_inverse=True)
    bin_centres = (bins + 0.5) * surface_spacing_mm
    score = np.sum((points_mm - bin_centres) ** 2, axis=1)
    # Choosing the first lexicographic voxel systematically biases every bin
    # toward left/posterior/inferior.  The point nearest each physical bin
    # centre gives balanced coverage without random sampling.
    order = np.lexsort((score, inverse))
    _, first = np.unique(inverse[order], return_index=True)
    selected = order[first]
    return indices[np.sort(selected)]


def _nearest_surface_indices(
    points_mm: FloatArray,
    lung_mask: BoolArray,
    spacing: tuple[float, float, float],
    tolerance_mm: float,
) -> tuple[npt.NDArray[np.int64], BoolArray]:
    surface = binary_surface(lung_mask)
    distance, nearest = ndimage.distance_transform_edt(
        ~surface, sampling=spacing, return_indices=True
    )
    voxel = np.rint(points_mm / np.asarray(spacing)).astype(np.int64)
    inside = np.all((voxel >= 0) & (voxel < np.asarray(lung_mask.shape)), axis=1)
    clipped = np.clip(voxel, 0, np.asarray(lung_mask.shape) - 1)
    point_distance = distance[tuple(clipped.T)]
    valid = inside & (point_distance <= tolerance_mm)
    snapped = np.column_stack(
        [nearest[axis][tuple(clipped.T)] for axis in range(3)]
    ).astype(np.int64)
    return snapped, valid


def _measure_one(
    profile_id: int,
    start_index: npt.NDArray[np.int64],
    volume_hu: npt.NDArray[np.float32],
    spacing: tuple[float, float, float],
    body_mask: BoolArray,
    lung_mask: BoolArray,
    distance_to_outside: FloatArray,
    nearest_outside: npt.NDArray[np.int32],
    params: ProfileParams,
) -> TissueProfile:
    point_mm = start_index.astype(np.float64) * np.asarray(spacing)
    key = tuple(int(value) for value in start_index)
    centre_distance = float(distance_to_outside[key])
    if centre_distance < params.min_thickness_mm:
        return _invalid(profile_id, point_mm, "too_thin")
    if centre_distance > params.max_thickness_mm:
        return _invalid(profile_id, point_mm, "too_deep")

    outside_index = nearest_outside[(slice(None),) + key].astype(np.float64)
    outside_mm = outside_index * np.asarray(spacing)
    vector = outside_mm - point_mm
    length = float(np.linalg.norm(vector))
    if length == 0:
        return _invalid(profile_id, point_mm, "no_exterior_path")
    samples = max(int(np.ceil(length / params.line_step_mm)) + 1, 3)
    distances = np.linspace(0.0, length, samples)
    coordinates_mm = point_mm[:, None] + vector[:, None] * (distances / length)
    coordinates = coordinates_mm / np.asarray(spacing)[:, None]
    line_lung = ndimage.map_coordinates(
        lung_mask, coordinates, order=0, mode="constant", cval=False
    ) > 0
    line_body = ndimage.map_coordinates(
        body_mask, coordinates, order=0, mode="constant", cval=False
    ) > 0

    tissue_candidates = np.flatnonzero(~line_lung)
    if not len(tissue_candidates):
        return _invalid(profile_id, point_mm, "never_leaves_lung")
    first_tissue = int(tissue_candidates[0])
    lung_interface_mm = (
        distances[first_tissue] / 2.0
        if first_tissue == 0
        else (distances[first_tissue - 1] + distances[first_tissue]) / 2.0
    )
    if lung_interface_mm > params.max_initial_lung_mm:
        return _invalid(profile_id, point_mm, "crosses_lung_interior")

    exits = np.flatnonzero(~line_body[first_tissue:])
    if not len(exits):
        return _invalid(profile_id, point_mm, "no_skin_exit")
    first_outside = first_tissue + int(exits[0])
    if first_outside <= first_tissue:
        return _invalid(profile_id, point_mm, "no_tissue_segment")
    reentry_mm = float(
        np.sum(line_lung[first_tissue:first_outside])
        * (length / max(samples - 1, 1))
    )
    if reentry_mm > params.max_lung_reentry_mm:
        return _invalid(profile_id, point_mm, "lung_reentry")

    skin_distance_mm = (
        distances[first_outside - 1] + distances[first_outside]
    ) / 2.0
    thickness = float(skin_distance_mm - lung_interface_mm)
    if thickness < params.min_thickness_mm or thickness > params.max_thickness_mm:
        return _invalid(profile_id, point_mm, "thickness_out_of_range")
    skin_mm = point_mm + vector * (skin_distance_mm / length)
    bounds_mm = (np.asarray(volume_hu.shape) - 1) * np.asarray(spacing)
    if np.any(skin_mm <= params.fov_margin_mm) or np.any(
        skin_mm >= bounds_mm - params.fov_margin_mm
    ):
        return _invalid(profile_id, point_mm, "fov_truncated")

    tissue_coordinates = coordinates[:, first_tissue:first_outside]
    hu = ndimage.map_coordinates(
        volume_hu,
        tissue_coordinates,
        order=1,
        mode="nearest",
        prefilter=False,
    )
    fat = (hu >= FAT_HU[0]) & (hu <= FAT_HU[1])
    muscle = (hu >= MUSCLE_HU[0]) & (hu <= MUSCLE_HU[1])
    bone = hu >= params.bone_lower_hu
    other = ~(fat | muscle | bone)
    counts = np.asarray([fat.sum(), muscle.sum(), bone.sum(), other.sum()], dtype=np.float64)
    fractions = counts / counts.sum()
    fat_mm, muscle_mm, bone_mm, other_mm = (fractions * thickness).tolist()
    return TissueProfile(
        profile_id=profile_id,
        valid=True,
        reason="",
        region=_region(vector),
        lung_x_mm=float(point_mm[0]),
        lung_y_mm=float(point_mm[1]),
        lung_z_mm=float(point_mm[2]),
        skin_x_mm=float(skin_mm[0]),
        skin_y_mm=float(skin_mm[1]),
        skin_z_mm=float(skin_mm[2]),
        thickness_mm=thickness,
        soft_tissue_mm=float(fat_mm + muscle_mm + other_mm),
        fat_mm=float(fat_mm),
        muscle_mm=float(muscle_mm),
        bone_mm=float(bone_mm),
        other_mm=float(other_mm),
    )


def measure_profiles_at_points(
    volume_ras: IntArray,
    spacing: tuple[float, float, float],
    body_mask: BoolArray,
    lung_mask: BoolArray,
    points_mm: FloatArray,
    *,
    profile_ids: npt.NDArray[np.int64] | None = None,
    params: ProfileParams | None = None,
) -> ProfileSet:
    """Snap physical points to pleura and measure profiles to the outer skin."""
    params = params or ProfileParams()
    if volume_ras.shape != body_mask.shape or body_mask.shape != lung_mask.shape:
        raise ValueError("volume, body and lung masks must share one grid")
    if points_mm.ndim != 2 or points_mm.shape[1] != 3:
        raise ValueError("points_mm must have shape N x 3")
    ids = profile_ids if profile_ids is not None else np.arange(len(points_mm), dtype=np.int64)
    if len(ids) != len(points_mm):
        raise ValueError("profile_ids and points_mm must have equal length")

    snapped, on_surface = _nearest_surface_indices(
        points_mm, lung_mask, spacing, params.mapped_surface_tolerance_mm
    )
    volume_hu = np.asarray(volume_ras, dtype=np.float32)
    distance, nearest = ndimage.distance_transform_edt(
        body_mask, sampling=spacing, return_indices=True
    )
    profiles: list[TissueProfile] = []
    for offset, profile_id in enumerate(ids):
        if not on_surface[offset]:
            profiles.append(_invalid(int(profile_id), points_mm[offset], "mapped_off_pleura"))
            continue
        profiles.append(
            _measure_one(
                int(profile_id),
                snapped[offset],
                volume_hu,
                spacing,
                body_mask,
                lung_mask,
                distance,
                nearest,
                params,
            )
        )
    return ProfileSet(
        coordinate_basis="local-zero-origin-RAS+_mm",
        coverage="whole-segmented-lung-surface_no-electrode-filter",
        profiles=tuple(profiles),
    )


def extract_whole_body_profiles(
    volume_ras: IntArray,
    spacing: tuple[float, float, float],
    body_mask: BoolArray,
    lung_mask: BoolArray,
    *,
    params: ProfileParams | None = None,
) -> ProfileSet:
    """Uniformly sample profiles over the full segmented lung surface."""
    params = params or ProfileParams()
    indices = _surface_indices(lung_mask, spacing, params.surface_spacing_mm)
    if volume_ras.shape != body_mask.shape or body_mask.shape != lung_mask.shape:
        raise ValueError("volume, body and lung masks must share one grid")
    volume_hu = np.asarray(volume_ras, dtype=np.float32)
    distance, nearest = ndimage.distance_transform_edt(
        body_mask, sampling=spacing, return_indices=True
    )
    profiles = tuple(
        _measure_one(
            profile_id,
            index,
            volume_hu,
            spacing,
            body_mask,
            lung_mask,
            distance,
            nearest,
            params,
        )
        for profile_id, index in enumerate(indices)
    )
    return ProfileSet(
        coordinate_basis="local-zero-origin-RAS+_mm",
        coverage="whole-segmented-lung-surface_no-electrode-filter",
        profiles=profiles,
    )


def pair_whole_body_profiles(
    fixed_volume_ras: IntArray,
    moving_volume_ras: IntArray,
    spacing: tuple[float, float, float],
    fixed_body_mask: BoolArray,
    moving_body_mask: BoolArray,
    fixed_lung_mask: BoolArray,
    moving_lung_mask: BoolArray,
    fixed_to_moving_mm: VectorArray,
    *,
    registration_gate_pass: bool,
    params: ProfileParams | None = None,
) -> tuple[ProfileSet, ProfileSet, PairedProfileSet]:
    """Measure corresponding phase profiles only after the registration gate."""
    if not registration_gate_pass:
        raise ValueError("paired profile comparison is blocked by registration QC")
    params = params or ProfileParams()
    fixed = extract_whole_body_profiles(
        fixed_volume_ras, spacing, fixed_body_mask, fixed_lung_mask, params=params
    )
    valid_fixed = [profile for profile in fixed.profiles if profile.valid]
    fixed_points = np.asarray(
        [[p.lung_x_mm, p.lung_y_mm, p.lung_z_mm] for p in valid_fixed],
        dtype=np.float64,
    )
    ids = np.asarray([p.profile_id for p in valid_fixed], dtype=np.int64)
    moving_points = transform_points(fixed_points, fixed_to_moving_mm, spacing)
    moving = measure_profiles_at_points(
        moving_volume_ras,
        spacing,
        moving_body_mask,
        moving_lung_mask,
        moving_points,
        profile_ids=ids,
        params=params,
    )
    moving_by_id = {profile.profile_id: profile for profile in moving.profiles}
    paired: list[PairedProfile] = []
    for fixed_profile in valid_fixed:
        moving_profile = moving_by_id[fixed_profile.profile_id]
        valid = moving_profile.valid
        reason = "" if valid else moving_profile.reason
        paired.append(
            PairedProfile(
                profile_id=fixed_profile.profile_id,
                region=fixed_profile.region,
                valid=valid,
                reason=reason,
                fixed_thickness_mm=fixed_profile.thickness_mm,
                moving_thickness_mm=moving_profile.thickness_mm,
                delta_moving_minus_fixed_mm=(
                    moving_profile.thickness_mm - fixed_profile.thickness_mm
                ),
                fixed_fat_mm=fixed_profile.fat_mm,
                moving_fat_mm=moving_profile.fat_mm,
                delta_fat_mm=moving_profile.fat_mm - fixed_profile.fat_mm,
                fixed_muscle_mm=fixed_profile.muscle_mm,
                moving_muscle_mm=moving_profile.muscle_mm,
                delta_muscle_mm=moving_profile.muscle_mm - fixed_profile.muscle_mm,
                fixed_bone_mm=fixed_profile.bone_mm,
                moving_bone_mm=moving_profile.bone_mm,
            )
        )
    return (
        fixed,
        moving,
        PairedProfileSet(
            transform_direction="fixed-expiration_to_moving-inspiration",
            profiles=tuple(paired),
        ),
    )


def write_profiles_csv(path: Path, profiles: ProfileSet | PairedProfileSet) -> int:
    """Write profile-level diagnostic data; aggregation remains subject-level."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(profile) for profile in profiles.profiles]
    if not rows:
        raise ValueError("cannot write an empty profile table")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def summarize_profiles(profiles: ProfileSet) -> ProfileSummary:
    """Subject-phase summary; profiles remain diagnostic sampling units."""
    valid = [profile for profile in profiles.profiles if profile.valid]
    if not valid:
        raise ValueError("profile set contains no valid measurements")
    invalid_reasons: dict[str, int] = {}
    region_counts: dict[str, int] = {}
    for profile in profiles.profiles:
        if profile.valid:
            region_counts[profile.region] = region_counts.get(profile.region, 0) + 1
        else:
            invalid_reasons[profile.reason] = invalid_reasons.get(profile.reason, 0) + 1
    thickness = np.asarray([profile.thickness_mm for profile in valid])
    return ProfileSummary(
        candidate_count=len(profiles.profiles),
        valid_count=len(valid),
        invalid_reasons=invalid_reasons,
        region_counts=region_counts,
        thickness_p10_mm=float(np.percentile(thickness, 10)),
        thickness_median_mm=float(np.median(thickness)),
        thickness_p90_mm=float(np.percentile(thickness, 90)),
        fat_median_mm=float(np.median([profile.fat_mm for profile in valid])),
        muscle_median_mm=float(np.median([profile.muscle_mm for profile in valid])),
        bone_median_mm=float(np.median([profile.bone_mm for profile in valid])),
    )


__all__ = [
    "PairedProfile",
    "PairedProfileSet",
    "ProfileParams",
    "ProfileSet",
    "ProfileSummary",
    "TissueProfile",
    "extract_whole_body_profiles",
    "measure_profiles_at_points",
    "pair_whole_body_profiles",
    "summarize_profiles",
    "write_profiles_csv",
]
