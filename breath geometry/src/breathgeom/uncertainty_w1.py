"""W1: how much the wall-profile number moves under pipeline choices alone.

The minimum programme wants one number, the change of soft-tissue thickness
between inspiration and expiration.  Before any correspondence work it has to
be known whether such a number can exist at all: if an arbitrary threshold or
sampling offset moves it as much as the expected respiratory effect, the
number measures the pipeline and not the patient.

This module answers only that question.  It never compares the two phases --
paired deltas stay blocked by the registration gate -- and instead measures the
dispersion of the measurand *inside one phase* under perturbations that must
not change anatomy.  The protocol, the perturbation families and the decision
thresholds are frozen in ``configs/measurement_uncertainty_budget_w1.json``
before any of this runs; see ``docs/WALL_MEASUREMENT_UNCERTAINTY_W1.md``.
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
from scipy import ndimage

from breathgeom.measure.profiles import ProfileParams, extract_whole_body_profiles
from breathgeom.measure.segmentation import body_mask_3d, lung_mask_3d
from breathgeom.measure.wall import IntArray, WallParams

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]

MEASURANDS = (
    "median_thickness_mm",
    "median_soft_tissue_mm",
    "median_fat_mm",
    "median_muscle_mm",
    "valid_fraction",
)
# A bin holding fewer valid profiles than this cannot support a median, so it
# is dropped from the spatial term rather than contributing sampling noise.
MIN_BIN_COUNT = 20


@dataclass(frozen=True)
class BudgetThresholds:
    feasible_max_mm: float
    marginal_max_mm: float
    target_resolvable_effect_mm: float


@dataclass(frozen=True)
class BudgetConfig:
    """Frozen protocol; nothing here may be chosen after seeing the numbers."""

    budget_version: str
    dataset_id: str
    subject_id: str
    phases: tuple[str, ...]
    baseline_wall: WallParams
    baseline_profile: ProfileParams
    body_hu_variants: tuple[float, ...]
    air_hu_variants: tuple[float, ...]
    tissue_window_variants: tuple[tuple[float, float], ...]
    sampling_offset_variants: tuple[float, ...]
    grid_jitter_variants: tuple[tuple[float, float, float], ...]
    thresholds: BudgetThresholds


@dataclass(frozen=True)
class VariantMeasurement:
    """One perturbed measurement of a single phase."""

    family: str
    variant_label: str
    is_family_reference: bool
    candidate_count: int
    valid_count: int
    valid_fraction: float
    median_thickness_mm: float
    median_soft_tissue_mm: float
    median_fat_mm: float
    median_muscle_mm: float

    def measurand(self, name: str) -> float:
        return cast(float, getattr(self, name))


@dataclass(frozen=True)
class FamilyBudget:
    family: str
    measurand: str
    minimum: float
    maximum: float
    half_range: float


@dataclass(frozen=True)
class SpatialBudget:
    family: str
    compared_bins: int
    p95_abs_median_shift_mm: float
    max_abs_median_shift_mm: float


@dataclass(frozen=True)
class PhaseBudget:
    phase: str
    measurements: tuple[VariantMeasurement, ...]
    families: tuple[FamilyBudget, ...]
    spatial: tuple[SpatialBudget, ...]
    combined_half_range: dict[str, float]
    worst_family: dict[str, str]

    def combined(self, measurand: str) -> float:
        return self.combined_half_range[measurand]


@dataclass(frozen=True)
class BudgetVerdict:
    budget_version: str
    dataset_id: str
    subject_id: str
    deciding_phase: str
    combined_half_range: dict[str, float]
    thickness_classification: str
    fat_classification: str
    muscle_classification: str
    soft_tissue_classification: str
    separate_fat_muscle_allowed: bool
    spatial_p95_worst_mm: float
    notes: tuple[str, ...]


def _require(mapping: dict[str, object], keys: tuple[str, ...], where: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{where} is missing required keys: {', '.join(sorted(missing))}")


def load_budget_config(path: Path) -> BudgetConfig:
    """Read the frozen protocol; an incomplete config is an error, not a default."""
    top = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    _require(
        top,
        (
            "budget_version",
            "dataset_id",
            "subject_id",
            "phases",
            "baseline_parameters",
            "perturbation_families",
            "decision_thresholds",
        ),
        str(path),
    )
    baseline = cast(dict[str, object], top["baseline_parameters"])
    families = cast(dict[str, object], top["perturbation_families"])
    _require(
        families,
        (
            "body_threshold",
            "lung_threshold",
            "tissue_window",
            "surface_sampling",
            "grid_jitter",
        ),
        f"{path}: perturbation_families",
    )
    thresholds = cast(dict[str, object], top["decision_thresholds"])
    fat = cast(list[float], baseline["fat_hu"])
    muscle = cast(list[float], baseline["muscle_hu"])

    cross_phase = cast(dict[str, object], top.get("cross_phase_comparison", {}))
    if cross_phase.get("performed") is not False:
        raise ValueError("W1 must declare cross_phase_comparison.performed = false")

    def variants(name: str) -> list[object]:
        return cast(list[object], cast(dict[str, object], families[name])["variants"])

    return BudgetConfig(
        budget_version=str(top["budget_version"]),
        dataset_id=str(top["dataset_id"]),
        subject_id=str(top["subject_id"]),
        phases=tuple(str(phase) for phase in cast(list[object], top["phases"])),
        baseline_wall=WallParams(
            body_hu=float(cast(float, baseline["body_hu"])),
            air_hu=float(cast(float, baseline["air_hu"])),
        ),
        baseline_profile=ProfileParams(
            fat_hu=(float(fat[0]), float(fat[1])),
            muscle_hu=(float(muscle[0]), float(muscle[1])),
            bone_lower_hu=float(cast(float, baseline["bone_lower_hu"])),
            surface_spacing_mm=float(cast(float, baseline["surface_spacing_mm"])),
            line_step_mm=float(cast(float, baseline["line_step_mm"])),
        ),
        body_hu_variants=tuple(float(cast(float, value)) for value in variants("body_threshold")),
        air_hu_variants=tuple(float(cast(float, value)) for value in variants("lung_threshold")),
        tissue_window_variants=tuple(
            (float(cast(list[float], pair)[0]), float(cast(list[float], pair)[1]))
            for pair in variants("tissue_window")
        ),
        sampling_offset_variants=tuple(
            float(cast(float, value)) for value in variants("surface_sampling")
        ),
        grid_jitter_variants=tuple(
            (
                float(cast(list[float], shift)[0]),
                float(cast(list[float], shift)[1]),
                float(cast(list[float], shift)[2]),
            )
            for shift in variants("grid_jitter")
        ),
        thresholds=BudgetThresholds(
            feasible_max_mm=float(cast(float, thresholds["feasible_max_mm"])),
            marginal_max_mm=float(cast(float, thresholds["marginal_max_mm"])),
            target_resolvable_effect_mm=float(
                cast(float, thresholds["target_resolvable_effect_mm"])
            ),
        ),
    )


def axial_thirds(lung_mask: BoolArray) -> tuple[float, float]:
    """Two z cuts splitting the lung extent into thirds, in voxel index units.

    The bins are fixed once from the baseline mask and reused by every variant,
    otherwise a perturbation that moves the lung extent would silently
    redefine the bins it is being compared across.
    """
    occupied = np.flatnonzero(lung_mask.any(axis=(0, 1)))
    if not len(occupied):
        raise ValueError("lung mask is empty: axial bins cannot be defined")
    low = float(occupied[0])
    high = float(occupied[-1])
    span = high - low
    return low + span / 3.0, low + 2.0 * span / 3.0


def _bin_label(region: str, z_index: float, cuts: tuple[float, float]) -> str:
    if z_index < cuts[0]:
        third = "inferior_third"
    elif z_index < cuts[1]:
        third = "middle_third"
    else:
        third = "superior_third"
    return f"{region}|{third}"


def measure_variant(
    volume_ras: IntArray,
    spacing: tuple[float, float, float],
    body_mask: BoolArray,
    lung_mask: BoolArray,
    profile_params: ProfileParams,
    *,
    family: str,
    variant_label: str,
    is_family_reference: bool,
    bin_cuts: tuple[float, float],
) -> tuple[VariantMeasurement, dict[str, float]]:
    """Measure one perturbed variant and its per-bin median thickness."""
    profiles = extract_whole_body_profiles(
        volume_ras, spacing, body_mask, lung_mask, params=profile_params
    )
    valid = [profile for profile in profiles.profiles if profile.valid]
    if not valid:
        raise ValueError(f"{family}/{variant_label}: no valid profiles")

    binned: dict[str, list[float]] = {}
    for profile in valid:
        label = _bin_label(profile.region, profile.lung_z_mm / spacing[2], bin_cuts)
        binned.setdefault(label, []).append(profile.thickness_mm)
    bin_medians = {
        label: float(np.median(values))
        for label, values in binned.items()
        if len(values) >= MIN_BIN_COUNT
    }

    soft = np.asarray([profile.soft_tissue_mm for profile in valid])
    measurement = VariantMeasurement(
        family=family,
        variant_label=variant_label,
        is_family_reference=is_family_reference,
        candidate_count=len(profiles.profiles),
        valid_count=len(valid),
        valid_fraction=len(valid) / len(profiles.profiles),
        median_thickness_mm=float(np.median([profile.thickness_mm for profile in valid])),
        median_soft_tissue_mm=float(np.median(soft)),
        median_fat_mm=float(np.median([profile.fat_mm for profile in valid])),
        median_muscle_mm=float(np.median([profile.muscle_mm for profile in valid])),
    )
    return measurement, bin_medians


def _shift_volume(volume_ras: IntArray, shift: tuple[float, float, float]) -> IntArray:
    """Sub-voxel resample; every jitter variant including zero goes through it.

    Running the zero shift through the same interpolator keeps the blur common
    to the family, so the reported spread is jitter and not interpolation.
    """
    shifted = ndimage.shift(
        np.asarray(volume_ras, dtype=np.float32),
        shift,
        order=1,
        mode="nearest",
        prefilter=False,
    )
    return cast(IntArray, np.rint(shifted).astype(volume_ras.dtype))


def _half_range(values: list[float]) -> tuple[float, float, float]:
    low = min(values)
    high = max(values)
    return low, high, (high - low) / 2.0


def evaluate_phase(
    volume_ras: IntArray,
    spacing: tuple[float, float, float],
    config: BudgetConfig,
    phase: str,
) -> PhaseBudget:
    """Run every frozen perturbation family against one phase."""
    baseline_body = body_mask_3d(volume_ras, config.baseline_wall)
    baseline_lung = lung_mask_3d(volume_ras, config.baseline_wall)
    bin_cuts = axial_thirds(baseline_lung)

    mask_cache: dict[tuple[float, float], tuple[BoolArray, BoolArray]] = {
        (config.baseline_wall.body_hu, config.baseline_wall.air_hu): (
            baseline_body,
            baseline_lung,
        )
    }

    def masks(body_hu: float, air_hu: float) -> tuple[BoolArray, BoolArray]:
        key = (body_hu, air_hu)
        if key not in mask_cache:
            wall = replace(config.baseline_wall, body_hu=body_hu, air_hu=air_hu)
            mask_cache[key] = (body_mask_3d(volume_ras, wall), lung_mask_3d(volume_ras, wall))
        return mask_cache[key]

    measurements: list[VariantMeasurement] = []
    bins_by_variant: dict[tuple[str, str], dict[str, float]] = {}
    reference_of: dict[str, str] = {}

    def record(
        family: str,
        label: str,
        is_reference: bool,
        body: BoolArray,
        lung: BoolArray,
        profile_params: ProfileParams,
        volume: IntArray,
    ) -> None:
        measurement, bin_medians = measure_variant(
            volume,
            spacing,
            body,
            lung,
            profile_params,
            family=family,
            variant_label=label,
            is_family_reference=is_reference,
            bin_cuts=bin_cuts,
        )
        measurements.append(measurement)
        bins_by_variant[(family, label)] = bin_medians
        if is_reference:
            reference_of[family] = label

    base_body_hu = config.baseline_wall.body_hu
    base_air_hu = config.baseline_wall.air_hu

    for body_hu in config.body_hu_variants:
        body, lung = masks(body_hu, base_air_hu)
        record(
            "body_threshold",
            f"body_hu={body_hu:g}",
            body_hu == base_body_hu,
            body,
            lung,
            config.baseline_profile,
            volume_ras,
        )

    for air_hu in config.air_hu_variants:
        body, lung = masks(base_body_hu, air_hu)
        record(
            "lung_threshold",
            f"air_hu={air_hu:g}",
            air_hu == base_air_hu,
            body,
            lung,
            config.baseline_profile,
            volume_ras,
        )

    base_fat_upper = config.baseline_profile.fat_hu[1]
    for fat_upper, muscle_lower in config.tissue_window_variants:
        params = replace(
            config.baseline_profile,
            fat_hu=(config.baseline_profile.fat_hu[0], fat_upper),
            muscle_hu=(muscle_lower, config.baseline_profile.muscle_hu[1]),
        )
        record(
            "tissue_window",
            f"fat_upper={fat_upper:g}",
            fat_upper == base_fat_upper,
            baseline_body,
            baseline_lung,
            params,
            volume_ras,
        )

    for offset in config.sampling_offset_variants:
        params = replace(config.baseline_profile, sampling_offset_fraction=offset)
        record(
            "surface_sampling",
            f"offset={offset:.3f}",
            offset == config.baseline_profile.sampling_offset_fraction,
            baseline_body,
            baseline_lung,
            params,
            volume_ras,
        )

    for shift in config.grid_jitter_variants:
        shifted = _shift_volume(volume_ras, shift)
        wall = config.baseline_wall
        record(
            "grid_jitter",
            "shift=" + ",".join(f"{value:g}" for value in shift),
            shift == (0.0, 0.0, 0.0),
            body_mask_3d(shifted, wall),
            lung_mask_3d(shifted, wall),
            config.baseline_profile,
            shifted,
        )

    families = tuple(
        FamilyBudget(family=family, measurand=measurand, minimum=low, maximum=high,
                     half_range=half)
        for family in dict.fromkeys(item.family for item in measurements)
        for measurand in MEASURANDS
        for low, high, half in [
            _half_range(
                [item.measurand(measurand) for item in measurements if item.family == family]
            )
        ]
    )

    combined: dict[str, float] = {}
    worst: dict[str, str] = {}
    for measurand in MEASURANDS:
        rows = [row for row in families if row.measurand == measurand]
        combined[measurand] = float(np.sqrt(sum(row.half_range**2 for row in rows)))
        worst[measurand] = max(rows, key=lambda row: row.half_range).family

    spatial: list[SpatialBudget] = []
    for family in dict.fromkeys(item.family for item in measurements):
        reference = bins_by_variant[(family, reference_of[family])]
        shifts: list[float] = []
        for (item_family, label), bins in bins_by_variant.items():
            if item_family != family or label == reference_of[family]:
                continue
            shifts.extend(
                abs(value - reference[key]) for key, value in bins.items() if key in reference
            )
        spatial.append(
            SpatialBudget(
                family=family,
                compared_bins=len(shifts),
                p95_abs_median_shift_mm=float(np.percentile(shifts, 95)) if shifts else 0.0,
                max_abs_median_shift_mm=float(max(shifts)) if shifts else 0.0,
            )
        )

    return PhaseBudget(
        phase=phase,
        measurements=tuple(measurements),
        families=families,
        spatial=tuple(spatial),
        combined_half_range=combined,
        worst_family=worst,
    )


def classify(half_range_mm: float, thresholds: BudgetThresholds) -> str:
    if half_range_mm <= thresholds.feasible_max_mm:
        return "feasible"
    if half_range_mm <= thresholds.marginal_max_mm:
        return "marginal"
    return "infeasible"


def decide(budgets: tuple[PhaseBudget, ...], config: BudgetConfig) -> BudgetVerdict:
    """Apply the frozen rule; the worse phase decides, as declared."""
    if not budgets:
        raise ValueError("no phase budgets to decide on")
    deciding = max(budgets, key=lambda budget: budget.combined("median_thickness_mm"))
    thresholds = config.thresholds
    thickness = classify(deciding.combined("median_thickness_mm"), thresholds)
    fat = classify(deciding.combined("median_fat_mm"), thresholds)
    muscle = classify(deciding.combined("median_muscle_mm"), thresholds)
    soft = classify(deciding.combined("median_soft_tissue_mm"), thresholds)
    spatial_worst = max(
        (row.p95_abs_median_shift_mm for budget in budgets for row in budget.spatial),
        default=0.0,
    )

    notes: list[str] = [
        "W1 measures pipeline reproducibility inside one phase; it is not accuracy "
        "against anatomical truth and it compares no phases.",
        "Families are combined as independent; correlated families would make the "
        "combined half-range an underestimate.",
        f"target_resolvable_effect_mm={thresholds.target_resolvable_effect_mm:g} is a "
        "planning anchor, not a measured respiratory effect.",
    ]
    tissue_rows = [
        row
        for budget in budgets
        for row in budget.families
        if row.family == "tissue_window" and row.measurand == "median_thickness_mm"
    ]
    if any(row.half_range > 1e-9 for row in tissue_rows):
        notes.append(
            "IMPLEMENTATION WARNING: the fat/muscle window moved total thickness, "
            "which is impossible by construction and indicates a defect."
        )

    return BudgetVerdict(
        budget_version=config.budget_version,
        dataset_id=config.dataset_id,
        subject_id=config.subject_id,
        deciding_phase=deciding.phase,
        combined_half_range=dict(deciding.combined_half_range),
        thickness_classification=thickness,
        fat_classification=fat,
        muscle_classification=muscle,
        soft_tissue_classification=soft,
        separate_fat_muscle_allowed=(
            fat in ("feasible", "marginal") and muscle in ("feasible", "marginal")
        ),
        spatial_p95_worst_mm=spatial_worst,
        notes=tuple(notes),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_budget(
    output_dir: Path,
    config_path: Path,
    config: BudgetConfig,
    budgets: tuple[PhaseBudget, ...],
    verdict: BudgetVerdict,
) -> Path:
    """Write per-variant rows, family budgets, the verdict and provenance."""
    output_dir.mkdir(parents=True, exist_ok=True)

    variants_path = output_dir / "variants.csv"
    rows = [
        {"phase": budget.phase, **asdict(item)}
        for budget in budgets
        for item in budget.measurements
    ]
    with variants_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    families_path = output_dir / "families.csv"
    family_rows = [
        {"phase": budget.phase, **asdict(row)} for budget in budgets for row in budget.families
    ]
    with families_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(family_rows[0]))
        writer.writeheader()
        writer.writerows(family_rows)

    summary = {
        "budget_version": config.budget_version,
        "dataset_id": config.dataset_id,
        "subject_id": config.subject_id,
        "cross_phase_comparison": {
            "performed": False,
            "reason": "paired deltas stay blocked by Gate 1",
        },
        "thresholds": asdict(config.thresholds),
        "phases": [
            {
                "phase": budget.phase,
                "combined_half_range": budget.combined_half_range,
                "worst_family": budget.worst_family,
                "spatial": [asdict(row) for row in budget.spatial],
            }
            for budget in budgets
        ],
        "verdict": asdict(verdict),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    manifest = {
        "budget_version": config.budget_version,
        "config": {"path": str(config_path).replace("\\", "/"), "sha256": _sha256(config_path)},
        "variants_sha256": _sha256(variants_path),
        "families_sha256": _sha256(families_path),
        "summary_sha256": _sha256(summary_path),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return manifest_path


__all__ = [
    "MEASURANDS",
    "BudgetConfig",
    "BudgetThresholds",
    "BudgetVerdict",
    "FamilyBudget",
    "PhaseBudget",
    "SpatialBudget",
    "VariantMeasurement",
    "axial_thirds",
    "classify",
    "decide",
    "evaluate_phase",
    "load_budget_config",
    "measure_variant",
    "write_budget",
]
