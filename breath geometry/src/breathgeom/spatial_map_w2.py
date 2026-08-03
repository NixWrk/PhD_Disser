"""W2: how fine a soft-tissue map over the body surface the data supports.

The reportable result of the minimum programme is the distribution of the
soft-tissue path length over the body, not a subject median -- a median hides
exactly the zones the electrode work needs to see.  W1 measured a per-bin noise
of 2.626 mm, but the diagnosis in D-045 showed that this is sampling error of
the median inside anatomically heterogeneous bins rather than a property of the
measurement: a homogeneous bin reaches 0.17 mm while ``anterior|inferior``
mixes anatomy at sigma 36.94 mm.

So the map is limited by two free parameters, angular resolution and sampling
density, and this module measures which combination the data actually carries.
Binning happens in a spine-anchored cylindrical frame attached to the skin
endpoint, because the map lives on the body surface.  The protocol is frozen in
``configs/wall_spatial_map_w2.json``; see ``docs/WALL_SPATIAL_MAP_W2.md``.
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

from breathgeom.measure.profiles import (
    ProfileParams,
    ProfileSet,
    TissueProfile,
    extract_whole_body_profiles,
)
from breathgeom.measure.segmentation import body_mask_3d, lung_mask_3d
from breathgeom.measure.wall import IntArray, WallParams

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class MapResolution:
    angular_sectors: int
    axial_levels: int

    @property
    def label(self) -> str:
        return f"{self.angular_sectors}x{self.axial_levels}"

    @property
    def bin_count(self) -> int:
        return self.angular_sectors * self.axial_levels

    def finer_than(self, other: MapResolution) -> bool:
        return self.bin_count > other.bin_count


@dataclass(frozen=True)
class SpatialMapConfig:
    map_version: str
    dataset_id: str
    subject_id: str
    phases: tuple[str, ...]
    baseline_wall: WallParams
    baseline_profile: ProfileParams
    sampling_densities_mm: tuple[float, ...]
    resolutions: tuple[MapResolution, ...]
    bootstrap_resamples: int
    bootstrap_seed: int
    min_profiles_per_bin: int
    min_occupied_fraction: float
    feasible_max_mm: float
    marginal_max_mm: float
    target_resolvable_effect_mm: float


@dataclass(frozen=True)
class CylindricalFrame:
    """Per-slice body centroid plus the baseline lung extent along z.

    Frozen from the baseline so a perturbation that moves anatomy cannot
    silently redefine the grid it is being compared across.
    """

    centroid_x: FloatArray
    centroid_y: FloatArray
    spacing: tuple[float, float, float]
    lung_z_min_mm: float
    lung_z_max_mm: float

    def coordinates(self, profiles: tuple[TissueProfile, ...]) -> tuple[FloatArray, FloatArray]:
        """Angle in turns [0, 1) from anterior, and normalised axial position."""
        skin = np.asarray(
            [[item.skin_x_mm, item.skin_y_mm, item.skin_z_mm] for item in profiles],
            dtype=np.float64,
        )
        slice_index = np.clip(
            np.rint(skin[:, 2] / self.spacing[2]).astype(np.int64),
            0,
            len(self.centroid_x) - 1,
        )
        delta_x = skin[:, 0] - self.centroid_x[slice_index]
        delta_y = skin[:, 1] - self.centroid_y[slice_index]
        # Angle zero is anterior (RAS+ axis 1), counter-clockwise from superior.
        angle = np.arctan2(delta_x, delta_y) / (2.0 * np.pi)
        angle = np.mod(angle, 1.0)
        span = self.lung_z_max_mm - self.lung_z_min_mm
        axial = (skin[:, 2] - self.lung_z_min_mm) / span
        return angle, axial


def build_frame(
    body_mask: BoolArray, lung_mask: BoolArray, spacing: tuple[float, float, float]
) -> CylindricalFrame:
    """Radial origin per slice, and the axial span the map is normalised to."""
    slices = body_mask.shape[2]
    centroid_x = np.zeros(slices, dtype=np.float64)
    centroid_y = np.zeros(slices, dtype=np.float64)
    occupied = [index for index in range(slices) if body_mask[:, :, index].any()]
    if not occupied:
        raise ValueError("body mask is empty: the cylindrical frame has no origin")
    for index in occupied:
        rows, columns = np.nonzero(body_mask[:, :, index])
        centroid_x[index] = rows.mean() * spacing[0]
        centroid_y[index] = columns.mean() * spacing[1]
    # Slices without a body cross-section borrow the nearest measured origin.
    measured = np.asarray(occupied, dtype=np.int64)
    for index in range(slices):
        if not body_mask[:, :, index].any():
            nearest = int(measured[np.argmin(np.abs(measured - index))])
            centroid_x[index] = centroid_x[nearest]
            centroid_y[index] = centroid_y[nearest]

    lung_slices = np.flatnonzero(lung_mask.any(axis=(0, 1)))
    if not len(lung_slices):
        raise ValueError("lung mask is empty: the axial span is undefined")
    return CylindricalFrame(
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        spacing=spacing,
        lung_z_min_mm=float(lung_slices[0]) * spacing[2],
        lung_z_max_mm=float(lung_slices[-1]) * spacing[2],
    )


def bin_indices(
    angle: FloatArray, axial: FloatArray, resolution: MapResolution
) -> npt.NDArray[np.int64]:
    """Flat bin index, or -1 for a skin point outside the mapped axial span."""
    sector = np.floor(angle * resolution.angular_sectors).astype(np.int64)
    sector = np.clip(sector, 0, resolution.angular_sectors - 1)
    level = np.floor(axial * resolution.axial_levels).astype(np.int64)
    outside = (level < 0) | (level >= resolution.axial_levels)
    level = np.clip(level, 0, resolution.axial_levels - 1)
    flat = level * resolution.angular_sectors + sector
    flat[outside] = -1
    return flat


@dataclass(frozen=True)
class BinStatistic:
    bin_index: int
    angular_sector: int
    axial_level: int
    count: int
    median_mm: float
    std_mm: float
    bootstrap_se_mm: float


@dataclass(frozen=True)
class MapEvaluation:
    phase: str
    sampling_density_mm: float
    resolution: str
    valid_profiles: int
    mapped_profiles: int
    occupied_bins: int
    total_bins: int
    occupied_fraction: float
    p95_bootstrap_se_mm: float
    max_bootstrap_se_mm: float
    median_bin_count: float
    qualifies: bool


def _soft_tissue(profiles: tuple[TissueProfile, ...]) -> FloatArray:
    return np.asarray([item.soft_tissue_mm for item in profiles], dtype=np.float64)


def bootstrap_median_se(
    values: FloatArray, resamples: int, generator: np.random.Generator
) -> float:
    """Standard error of the median; the primary per-bin noise estimate.

    W1 already showed the observed spatial spread is bounded by median sampling
    error, so resampling the bin is the direct estimator of it.
    """
    if len(values) < 2:
        return float("nan")
    draws = generator.integers(0, len(values), size=(resamples, len(values)))
    medians = np.median(values[draws], axis=1)
    return float(np.std(medians, ddof=1))


def evaluate_map(
    profiles: ProfileSet,
    frame: CylindricalFrame,
    resolution: MapResolution,
    config: SpatialMapConfig,
    *,
    phase: str,
    sampling_density_mm: float,
) -> tuple[MapEvaluation, tuple[BinStatistic, ...]]:
    """Bin one extraction and measure the per-bin noise floor."""
    valid = tuple(item for item in profiles.profiles if item.valid)
    if not valid:
        raise ValueError(f"{phase}: no valid profiles to map")
    angle, axial = frame.coordinates(valid)
    flat = bin_indices(angle, axial, resolution)
    values = _soft_tissue(valid)

    generator = np.random.default_rng(config.bootstrap_seed)
    statistics: list[BinStatistic] = []
    for index in range(resolution.bin_count):
        selected = values[flat == index]
        if len(selected) < config.min_profiles_per_bin:
            continue
        statistics.append(
            BinStatistic(
                bin_index=index,
                angular_sector=index % resolution.angular_sectors,
                axial_level=index // resolution.angular_sectors,
                count=len(selected),
                median_mm=float(np.median(selected)),
                std_mm=float(np.std(selected, ddof=1)),
                bootstrap_se_mm=bootstrap_median_se(
                    selected, config.bootstrap_resamples, generator
                ),
            )
        )

    errors = np.asarray([item.bootstrap_se_mm for item in statistics], dtype=np.float64)
    occupied_fraction = len(statistics) / resolution.bin_count
    p95 = float(np.percentile(errors, 95)) if len(errors) else float("inf")
    evaluation = MapEvaluation(
        phase=phase,
        sampling_density_mm=sampling_density_mm,
        resolution=resolution.label,
        valid_profiles=len(valid),
        mapped_profiles=int(np.count_nonzero(flat >= 0)),
        occupied_bins=len(statistics),
        total_bins=resolution.bin_count,
        occupied_fraction=occupied_fraction,
        p95_bootstrap_se_mm=p95,
        max_bootstrap_se_mm=float(np.max(errors)) if len(errors) else float("inf"),
        median_bin_count=float(np.median([item.count for item in statistics]))
        if statistics
        else 0.0,
        qualifies=(
            p95 <= config.feasible_max_mm
            and occupied_fraction >= config.min_occupied_fraction
        ),
    )
    return evaluation, tuple(statistics)


def select_configuration(
    evaluations: tuple[MapEvaluation, ...], config: SpatialMapConfig
) -> tuple[float, str] | None:
    """Finest qualifying resolution, at the coarsest density that supports it.

    Both phases must qualify; the rule was frozen before any of these numbers
    existed and is applied verbatim.
    """
    by_resolution = {item.label: item for item in config.resolutions}
    qualifying: dict[tuple[float, str], int] = {}
    for evaluation in evaluations:
        key = (evaluation.sampling_density_mm, evaluation.resolution)
        qualifying[key] = qualifying.get(key, 0) + int(evaluation.qualifies)

    complete = [key for key, count in qualifying.items() if count == len(config.phases)]
    if not complete:
        return None
    best_bins = max(by_resolution[label].bin_count for _, label in complete)
    finest = [key for key in complete if by_resolution[key[1]].bin_count == best_bins]
    return max(finest, key=lambda key: key[0])


@dataclass(frozen=True)
class SpatialMapVerdict:
    map_version: str
    dataset_id: str
    subject_id: str
    selected_density_mm: float | None
    selected_resolution: str | None
    selected_p95_noise_mm: float | None
    selection_made: bool
    best_p95_noise_mm: float
    notes: tuple[str, ...]


def decide_map(
    evaluations: tuple[MapEvaluation, ...], config: SpatialMapConfig
) -> SpatialMapVerdict:
    selection = select_configuration(evaluations, config)
    notes = [
        "W2 measures within-phase spatial stability; it compares no phases and "
        "produces no delta map.",
        "The per-bin noise is the bootstrap standard error of the bin median, the "
        "term that dominated the W1 spatial spread.",
        "A bin below the frozen minimum count is unresolved and is not plotted.",
    ]
    best = min(
        (item.p95_bootstrap_se_mm for item in evaluations),
        default=float("inf"),
    )
    if selection is None:
        notes.append(
            "No configuration met both the noise and the occupancy rule, so no "
            "resolution is selected and the map stays unreported."
        )
        return SpatialMapVerdict(
            map_version=config.map_version,
            dataset_id=config.dataset_id,
            subject_id=config.subject_id,
            selected_density_mm=None,
            selected_resolution=None,
            selected_p95_noise_mm=None,
            selection_made=False,
            best_p95_noise_mm=best,
            notes=tuple(notes),
        )
    density, resolution = selection
    chosen = [
        item.p95_bootstrap_se_mm
        for item in evaluations
        if item.sampling_density_mm == density and item.resolution == resolution
    ]
    return SpatialMapVerdict(
        map_version=config.map_version,
        dataset_id=config.dataset_id,
        subject_id=config.subject_id,
        selected_density_mm=density,
        selected_resolution=resolution,
        selected_p95_noise_mm=float(max(chosen)),
        selection_made=True,
        best_p95_noise_mm=best,
        notes=tuple(notes),
    )


def _require(mapping: dict[str, object], keys: tuple[str, ...], where: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{where} is missing required keys: {', '.join(sorted(missing))}")


def load_spatial_map_config(path: Path) -> SpatialMapConfig:
    top = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    _require(
        top,
        (
            "map_version",
            "dataset_id",
            "subject_id",
            "phases",
            "sampling_density_variants_mm",
            "map_resolution_variants",
            "per_bin_noise",
            "occupancy",
            "decision_thresholds",
        ),
        str(path),
    )
    cross_phase = cast(dict[str, object], top.get("cross_phase_comparison", {}))
    if cross_phase.get("performed") is not False:
        raise ValueError("W2 must declare cross_phase_comparison.performed = false")
    measurand = cast(dict[str, object], top.get("measurand", {}))
    if measurand.get("name") != "soft_tissue_mm":
        raise ValueError("W2 measures soft_tissue_mm; separate tissues were merged by D-044")

    noise = cast(dict[str, object], top["per_bin_noise"])
    _require(noise, ("bootstrap_resamples", "bootstrap_seed"), f"{path}: per_bin_noise")
    resamples = int(cast(int, noise["bootstrap_resamples"]))
    seed = int(cast(int, noise["bootstrap_seed"]))
    occupancy = cast(dict[str, object], top["occupancy"])
    thresholds = cast(dict[str, object], top["decision_thresholds"])

    return SpatialMapConfig(
        map_version=str(top["map_version"]),
        dataset_id=str(top["dataset_id"]),
        subject_id=str(top["subject_id"]),
        phases=tuple(str(phase) for phase in cast(list[object], top["phases"])),
        baseline_wall=WallParams(),
        baseline_profile=ProfileParams(),
        sampling_densities_mm=tuple(
            float(cast(float, value))
            for value in cast(list[object], top["sampling_density_variants_mm"])
        ),
        resolutions=tuple(
            MapResolution(
                angular_sectors=int(cast(dict[str, int], entry)["angular_sectors"]),
                axial_levels=int(cast(dict[str, int], entry)["axial_levels"]),
            )
            for entry in cast(list[object], top["map_resolution_variants"])
        ),
        bootstrap_resamples=resamples,
        bootstrap_seed=seed,
        min_profiles_per_bin=int(cast(int, occupancy["min_profiles_per_bin"])),
        min_occupied_fraction=float(cast(float, occupancy["min_occupied_fraction"])),
        feasible_max_mm=float(cast(float, thresholds["feasible_max_mm"])),
        marginal_max_mm=float(cast(float, thresholds["marginal_max_mm"])),
        target_resolvable_effect_mm=float(
            cast(float, thresholds["target_resolvable_effect_mm"])
        ),
    )


def evaluate_phase_map(
    volume_ras: IntArray,
    spacing: tuple[float, float, float],
    config: SpatialMapConfig,
    phase: str,
) -> tuple[tuple[MapEvaluation, ...], dict[tuple[float, str], tuple[BinStatistic, ...]]]:
    """Run the frozen density x resolution scan for one phase."""
    body = body_mask_3d(volume_ras, config.baseline_wall)
    lung = lung_mask_3d(volume_ras, config.baseline_wall)
    frame = build_frame(body, lung, spacing)

    evaluations: list[MapEvaluation] = []
    statistics: dict[tuple[float, str], tuple[BinStatistic, ...]] = {}
    for density in config.sampling_densities_mm:
        params = replace(config.baseline_profile, surface_spacing_mm=density)
        profiles = extract_whole_body_profiles(
            volume_ras, spacing, body, lung, params=params
        )
        for resolution in config.resolutions:
            evaluation, bins = evaluate_map(
                profiles,
                frame,
                resolution,
                config,
                phase=phase,
                sampling_density_mm=density,
            )
            evaluations.append(evaluation)
            statistics[(density, resolution.label)] = bins
    return tuple(evaluations), statistics


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_spatial_map(
    output_dir: Path,
    config_path: Path,
    config: SpatialMapConfig,
    evaluations: tuple[MapEvaluation, ...],
    bins: dict[tuple[str, float, str], tuple[BinStatistic, ...]],
    verdict: SpatialMapVerdict,
) -> Path:
    """Write the scan, the per-bin table of the selected map and provenance."""
    output_dir.mkdir(parents=True, exist_ok=True)

    scan_path = output_dir / "scan.csv"
    rows = [asdict(item) for item in evaluations]
    with scan_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    bin_rows = [
        {"phase": phase, "sampling_density_mm": density, "resolution": resolution, **asdict(item)}
        for (phase, density, resolution), entries in bins.items()
        for item in entries
    ]
    bins_path = output_dir / "bins.csv"
    with bins_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(bin_rows[0]))
        writer.writeheader()
        writer.writerows(bin_rows)

    summary = {
        "map_version": config.map_version,
        "dataset_id": config.dataset_id,
        "subject_id": config.subject_id,
        "measurand": "soft_tissue_mm",
        "cross_phase_comparison": {
            "performed": False,
            "reason": "the delta map stays blocked until Gate 1B",
        },
        "thresholds": {
            "feasible_max_mm": config.feasible_max_mm,
            "marginal_max_mm": config.marginal_max_mm,
            "target_resolvable_effect_mm": config.target_resolvable_effect_mm,
            "min_profiles_per_bin": config.min_profiles_per_bin,
            "min_occupied_fraction": config.min_occupied_fraction,
        },
        "verdict": asdict(verdict),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    manifest = {
        "map_version": config.map_version,
        "config": {"path": str(config_path).replace("\\", "/"), "sha256": _sha256(config_path)},
        "scan_sha256": _sha256(scan_path),
        "bins_sha256": _sha256(bins_path),
        "summary_sha256": _sha256(summary_path),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return manifest_path


__all__ = [
    "BinStatistic",
    "CylindricalFrame",
    "MapEvaluation",
    "MapResolution",
    "SpatialMapConfig",
    "SpatialMapVerdict",
    "bin_indices",
    "bootstrap_median_se",
    "build_frame",
    "decide_map",
    "evaluate_map",
    "evaluate_phase_map",
    "load_spatial_map_config",
    "select_configuration",
    "write_spatial_map",
]
