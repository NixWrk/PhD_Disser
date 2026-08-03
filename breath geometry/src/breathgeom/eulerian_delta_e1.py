"""E1: the spine-anchored delta-h map, the comparison arm for the rib frame.

This map is Eulerian by construction and therefore wrong in a known, measured
way: the wall slides past a stationary grid, and the measured gradient of h
times the measured material displacement puts that artifact at roughly 1.4 mm
typically and up to 12 mm in steep zones, against a 3 mm effect.  It is
computed anyway because the difference between this map and the rib-anchored
one *is* that artifact, so having both turns the estimate into a measurement.

Protocol frozen in ``configs/wall_eulerian_delta_e1.json``; see
``docs/WALL_EULERIAN_DELTA_E1.md``.  Nothing here may be reported as a
validated respiratory effect.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from breathgeom.measure.profiles import ProfileParams, extract_whole_body_profiles
from breathgeom.measure.segmentation import body_mask_3d, lung_mask_3d
from breathgeom.measure.wall import IntArray, WallParams, anatomical_midline
from breathgeom.spatial_map_w2 import (
    CylindricalFrame,
    MapResolution,
    bin_indices,
    bootstrap_median_se,
)

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class EulerianDeltaConfig:
    map_version: str
    dataset_id: str
    subjects: tuple[str, ...]
    sampling_density_mm: float
    resolution: MapResolution
    min_profiles_per_bin: int
    bootstrap_resamples: int
    bootstrap_seed: int
    resolved_sigma_multiple: float = 2.0


@dataclass(frozen=True)
class DeltaBin:
    subject_id: str
    bin_index: int
    angular_sector: int
    axial_level: int
    fixed_count: int
    moving_count: int
    fixed_median_mm: float
    moving_median_mm: float
    delta_mm: float
    uncertainty_mm: float
    resolved: bool


@dataclass(frozen=True)
class SubjectDelta:
    subject_id: str
    bins_in_both_phases: int
    resolved_bins: int
    resolved_fraction: float
    median_abs_delta_mm: float
    p90_abs_delta_mm: float
    min_delta_mm: float
    max_delta_mm: float
    median_uncertainty_mm: float


def load_eulerian_config(path: Path) -> EulerianDeltaConfig:
    top = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    if top.get("status") != "diagnostic":
        raise ValueError("E1 must declare status = diagnostic; it is not a result")
    measurand = cast(dict[str, object], top.get("measurand", {}))
    if measurand.get("name") != "soft_tissue_mm":
        raise ValueError("E1 measures soft_tissue_mm; separate tissues were merged by D-044")
    resolution = cast(dict[str, int], top["map_resolution"])
    occupancy = cast(dict[str, object], top["occupancy"])
    uncertainty = cast(dict[str, object], top["per_bin_uncertainty"])
    return EulerianDeltaConfig(
        map_version=str(top["map_version"]),
        dataset_id=str(top["dataset_id"]),
        subjects=tuple(str(item) for item in cast(list[object], top["subjects"])),
        sampling_density_mm=float(cast(float, top["sampling_density_mm"])),
        resolution=MapResolution(
            angular_sectors=int(resolution["angular_sectors"]),
            axial_levels=int(resolution["axial_levels"]),
        ),
        min_profiles_per_bin=int(cast(int, occupancy["min_profiles_per_bin"])),
        bootstrap_resamples=int(cast(int, uncertainty["bootstrap_resamples"])),
        bootstrap_seed=int(cast(int, uncertainty["bootstrap_seed"])),
    )


def spine_extent_mm(
    volume_ras: IntArray,
    spacing: tuple[float, float, float],
    wall_params: WallParams | None = None,
    *,
    spine_hu: float = 300.0,
    half_width_mm: float = 25.0,
) -> tuple[float, float]:
    """Axial span of the vertebral column, the one landmark breathing does not move.

    Normalising to the lung extent would anchor the map to the very thing the
    measurement is about, so the spine is used instead.
    """
    wall_params = wall_params or WallParams()
    midline = anatomical_midline(volume_ras, wall_params)
    body = body_mask_3d(volume_ras, wall_params)
    half = half_width_mm / spacing[0]
    column: BoolArray = np.zeros(volume_ras.shape, dtype=bool)
    low, high = max(int(midline - half), 0), int(midline + half)
    column[low:high, :, :] = body[low:high, :, :] & (volume_ras[low:high, :, :] > spine_hu)
    occupied = np.flatnonzero(column.any(axis=(0, 1)))
    if not len(occupied):
        raise ValueError("no vertebral bone found: the axial anchor is undefined")
    return float(occupied[0]) * spacing[2], float(occupied[-1]) * spacing[2]


def build_shared_frame(
    fixed_ras: IntArray,
    spacing: tuple[float, float, float],
    wall_params: WallParams | None = None,
) -> CylindricalFrame:
    """One frame from the fixed phase, applied unchanged to both.

    Applying a single frame to both phases is exactly what makes the comparison
    Eulerian; rebuilding it per phase would silently follow the anatomy.
    """
    wall_params = wall_params or WallParams()
    body = body_mask_3d(fixed_ras, wall_params)
    slices = body.shape[2]
    centroid_x = np.zeros(slices, dtype=np.float64)
    centroid_y = np.zeros(slices, dtype=np.float64)
    occupied = [index for index in range(slices) if body[:, :, index].any()]
    if not occupied:
        raise ValueError("body mask is empty: the cylindrical frame has no origin")
    for index in occupied:
        rows, columns = np.nonzero(body[:, :, index])
        centroid_x[index] = rows.mean() * spacing[0]
        centroid_y[index] = columns.mean() * spacing[1]
    measured = np.asarray(occupied, dtype=np.int64)
    for index in range(slices):
        if not body[:, :, index].any():
            nearest = int(measured[np.argmin(np.abs(measured - index))])
            centroid_x[index] = centroid_x[nearest]
            centroid_y[index] = centroid_y[nearest]

    low, high = spine_extent_mm(fixed_ras, spacing, wall_params)
    return CylindricalFrame(
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        spacing=spacing,
        lung_z_min_mm=low,
        lung_z_max_mm=high,
    )


def _binned_values(
    volume_ras: IntArray,
    spacing: tuple[float, float, float],
    frame: CylindricalFrame,
    config: EulerianDeltaConfig,
) -> dict[int, FloatArray]:
    wall_params = WallParams()
    body = body_mask_3d(volume_ras, wall_params)
    lung = lung_mask_3d(volume_ras, wall_params)
    params = replace(ProfileParams(), surface_spacing_mm=config.sampling_density_mm)
    profiles = extract_whole_body_profiles(volume_ras, spacing, body, lung, params=params)
    valid = tuple(item for item in profiles.profiles if item.valid)
    if not valid:
        raise ValueError("no valid profiles")
    angle, axial = frame.coordinates(valid)
    flat = bin_indices(angle, axial, config.resolution)
    values = np.asarray([item.soft_tissue_mm for item in valid], dtype=np.float64)
    return {
        index: values[flat == index]
        for index in range(config.resolution.bin_count)
        if int(np.count_nonzero(flat == index)) >= config.min_profiles_per_bin
    }


def compute_delta(
    fixed_ras: IntArray,
    moving_ras: IntArray,
    spacing: tuple[float, float, float],
    config: EulerianDeltaConfig,
    subject_id: str,
) -> tuple[SubjectDelta, tuple[DeltaBin, ...]]:
    """Subtract the two phase maps in one shared spine-anchored frame."""
    frame = build_shared_frame(fixed_ras, spacing)
    fixed_bins = _binned_values(fixed_ras, spacing, frame, config)
    moving_bins = _binned_values(moving_ras, spacing, frame, config)

    generator = np.random.default_rng(config.bootstrap_seed)
    rows: list[DeltaBin] = []
    for index in sorted(set(fixed_bins) & set(moving_bins)):
        fixed_values = fixed_bins[index]
        moving_values = moving_bins[index]
        fixed_se = bootstrap_median_se(fixed_values, config.bootstrap_resamples, generator)
        moving_se = bootstrap_median_se(moving_values, config.bootstrap_resamples, generator)
        uncertainty = float(np.hypot(fixed_se, moving_se))
        delta = float(np.median(moving_values) - np.median(fixed_values))
        rows.append(
            DeltaBin(
                subject_id=subject_id,
                bin_index=index,
                angular_sector=index % config.resolution.angular_sectors,
                axial_level=index // config.resolution.angular_sectors,
                fixed_count=len(fixed_values),
                moving_count=len(moving_values),
                fixed_median_mm=float(np.median(fixed_values)),
                moving_median_mm=float(np.median(moving_values)),
                delta_mm=delta,
                uncertainty_mm=uncertainty,
                resolved=abs(delta) > config.resolved_sigma_multiple * uncertainty,
            )
        )

    if not rows:
        raise ValueError(f"{subject_id}: no bin cleared the minimum count in both phases")
    deltas = np.asarray([item.delta_mm for item in rows])
    resolved = [item for item in rows if item.resolved]
    summary = SubjectDelta(
        subject_id=subject_id,
        bins_in_both_phases=len(rows),
        resolved_bins=len(resolved),
        resolved_fraction=len(resolved) / len(rows),
        median_abs_delta_mm=float(np.median(np.abs(deltas))),
        p90_abs_delta_mm=float(np.percentile(np.abs(deltas), 90)),
        min_delta_mm=float(deltas.min()),
        max_delta_mm=float(deltas.max()),
        median_uncertainty_mm=float(
            np.median([item.uncertainty_mm for item in rows])
        ),
    )
    return summary, tuple(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_eulerian_delta(
    output_dir: Path,
    config_path: Path,
    config: EulerianDeltaConfig,
    summaries: tuple[SubjectDelta, ...],
    bins: tuple[DeltaBin, ...],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    bins_path = output_dir / "delta_bins.csv"
    rows = [asdict(item) for item in bins]
    with bins_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    subjects_path = output_dir / "subjects.csv"
    subject_rows = [asdict(item) for item in summaries]
    with subjects_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(subject_rows[0]))
        writer.writeheader()
        writer.writerows(subject_rows)

    summary = {
        "map_version": config.map_version,
        "dataset_id": config.dataset_id,
        "status": "diagnostic",
        "frame": "eulerian, spine-anchored, one frame applied to both phases",
        "carries_frame_artifact": (
            "measured gradient 0.09-0.12 mm/mm median and 0.29 p90 times measured "
            "material displacement 11.3-30.2 mm median gives roughly 1.4 mm typical "
            "and up to 12 mm in steep zones"
        ),
        "not_a_validated_effect": True,
        "subjects": subject_rows,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    manifest = {
        "map_version": config.map_version,
        "config": {"path": str(config_path).replace("\\", "/"), "sha256": _sha256(config_path)},
        "delta_bins_sha256": _sha256(bins_path),
        "subjects_sha256": _sha256(subjects_path),
        "summary_sha256": _sha256(summary_path),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return manifest_path


__all__ = [
    "DeltaBin",
    "EulerianDeltaConfig",
    "SubjectDelta",
    "build_shared_frame",
    "compute_delta",
    "load_eulerian_config",
    "spine_extent_mm",
    "write_eulerian_delta",
]
