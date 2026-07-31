"""Development-only batch and hidden-truth evaluator for joint SVF J1.2."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
from scipy import ndimage

from breathgeom.measure.joint_svf_registration import (
    JointSVFResult,
    JointSVFSearch,
    JointSVFWeights,
    register_joint_piecewise_svf,
)
from breathgeom.measure.registration import FloatArray, VectorArray, transform_points
from breathgeom.measure.sliding_phantom import (
    RegionFieldError,
    SlidingInterfaceMetrics,
    SlidingPhantomPair,
    evaluate_sliding_interface,
    make_sliding_phantom_pair,
    region_field_error,
)
from breathgeom.measure.wall import IntArray as WallIntArray
from breathgeom.synthetic_s1 import SlidingSuite


@dataclass(frozen=True)
class AdvectedSurfaceMetrics:
    """Independent fixed-surface to moving-mask contact QC."""

    surface_point_count: int
    lung_abs_distance_p95_mm: float
    body_abs_distance_p95_mm: float
    abs_distance_p95_max_mm: float
    coverage_min: float
    gap_fraction: float
    collision_fraction: float


@dataclass(frozen=True)
class J12DevelopmentRecord:
    """One development case/variant row; never a human measurement."""

    suite_version: str
    search_version: str
    case_id: str
    variant_id: str
    runner_version: str
    transform_direction: str
    coordinate_basis: str
    expert_landmarks_loaded: bool
    challenge_loaded: bool
    measurement_eligible: bool
    endpoint_p95_limit_mm: float
    lung_field_mean_mm: float
    lung_field_p95_mm: float
    lung_field_maximum_mm: float
    body_field_mean_mm: float
    body_field_p95_mm: float
    body_field_maximum_mm: float
    normal_mismatch_p95_mm: float
    tangential_slip_truth_median_mm: float
    tangential_slip_observed_median_mm: float
    tangential_slip_error_mm: float
    lung_jacobian_p01: float
    body_jacobian_p01: float
    lung_nonpositive_jacobian_fraction: float
    body_nonpositive_jacobian_fraction: float
    advected_surface_point_count: int
    advected_lung_abs_distance_p95_mm: float
    advected_body_abs_distance_p95_mm: float
    advected_target_surface_p95_max_mm: float
    advected_surface_coverage_min: float
    advected_gap_fraction: float
    advected_collision_fraction: float
    objective_initial: float
    objective_final: float
    rejected_topology_updates: int
    final_learning_rate: float
    elapsed_s: float
    peak_gpu_memory_bytes: int
    torch_version: str
    convexadam_version: str
    device: str
    gate_pass: bool
    gate_reasons: tuple[str, ...]

    def as_csv_row(self) -> dict[str, str | float | int | bool]:
        values = asdict(self)
        values["gate_reasons"] = ";".join(self.gate_reasons)
        return cast(dict[str, str | float | int | bool], values)


@dataclass(frozen=True)
class J12DevelopmentRun:
    """In-memory pair, estimated fields and independent record."""

    pair: SlidingPhantomPair
    result: JointSVFResult
    record: J12DevelopmentRecord


@dataclass(frozen=True)
class J12TruthPreflightRecord:
    """Case-level proof that generator truth is or is not eligible for J1.2."""

    suite_version: str
    search_version: str
    case_id: str
    tolerance_mm: float
    required_coverage_min: float
    surface_point_count: int
    lung_abs_distance_p95_mm: float
    body_abs_distance_p95_mm: float
    abs_distance_p95_max_mm: float
    coverage_min: float
    gap_fraction: float
    collision_fraction: float
    truth_gate_pass: bool
    gate_reasons: tuple[str, ...]

    def as_csv_row(self) -> dict[str, str | float | int | bool]:
        values = asdict(self)
        values["gate_reasons"] = ";".join(self.gate_reasons)
        return cast(dict[str, str | float | int | bool], values)


def _surface_mask(mask: np.ndarray[tuple[int, ...], np.dtype[np.bool_]]) -> np.ndarray[
    tuple[int, ...], np.dtype[np.bool_]
]:
    return np.ascontiguousarray(mask & ~ndimage.binary_erosion(mask))


def _signed_surface_distance(
    mask: np.ndarray[tuple[int, ...], np.dtype[np.bool_]],
    spacing_mm: tuple[float, float, float],
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    surface = _surface_mask(mask)
    distance = ndimage.distance_transform_edt(
        ~surface,
        sampling=spacing_mm,
    ).astype(np.float64)
    distance[mask] *= -1.0
    distance[surface] = 0.0
    return np.ascontiguousarray(distance)


def _sample_scalar_at_points(
    values_xyz: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    points_mm: FloatArray,
    spacing_mm: tuple[float, float, float],
) -> FloatArray:
    coordinates = (points_mm / np.asarray(spacing_mm, dtype=np.float64)).T
    sampled = ndimage.map_coordinates(
        values_xyz,
        coordinates,
        order=1,
        mode="nearest",
        prefilter=False,
    )
    return np.asarray(sampled, dtype=np.float64)


def evaluate_advected_surface_fields(
    pair: SlidingPhantomPair,
    lung_displacement_mm: VectorArray,
    body_displacement_mm: VectorArray,
    *,
    tolerance_mm: float,
) -> AdvectedSurfaceMetrics:
    """Evaluate both regional transforms against the observed moving boundary."""
    surface_indices = np.argwhere(_surface_mask(pair.fixed_lung_mask))
    if len(surface_indices) == 0:
        raise ValueError("fixed lung surface is empty")
    fixed_points = (
        surface_indices.astype(np.float64)
        * np.asarray(pair.spacing_mm, dtype=np.float64)
    )
    lung_points = transform_points(
        fixed_points,
        lung_displacement_mm,
        pair.spacing_mm,
    )
    body_points = transform_points(
        fixed_points,
        body_displacement_mm,
        pair.spacing_mm,
    )
    signed = _signed_surface_distance(
        pair.moving_lung_mask,
        pair.spacing_mm,
    )
    lung_distance = _sample_scalar_at_points(
        signed,
        lung_points,
        pair.spacing_mm,
    )
    body_distance = _sample_scalar_at_points(
        signed,
        body_points,
        pair.spacing_mm,
    )
    lung_p95 = float(np.percentile(np.abs(lung_distance), 95))
    body_p95 = float(np.percentile(np.abs(body_distance), 95))
    coverage = min(
        float(np.mean(np.abs(lung_distance) <= tolerance_mm)),
        float(np.mean(np.abs(body_distance) <= tolerance_mm)),
    )
    combined = np.concatenate((lung_distance, body_distance))
    return AdvectedSurfaceMetrics(
        surface_point_count=len(surface_indices),
        lung_abs_distance_p95_mm=lung_p95,
        body_abs_distance_p95_mm=body_p95,
        abs_distance_p95_max_mm=max(lung_p95, body_p95),
        coverage_min=coverage,
        gap_fraction=float(np.mean(combined < -tolerance_mm)),
        collision_fraction=float(np.mean(combined > tolerance_mm)),
    )


def evaluate_advected_surface(
    pair: SlidingPhantomPair,
    result: JointSVFResult,
    *,
    tolerance_mm: float,
) -> AdvectedSurfaceMetrics:
    """Evaluate a runner result with the independent field-level implementation."""
    return evaluate_advected_surface_fields(
        pair,
        result.lung_displacement_mm,
        result.body_displacement_mm,
        tolerance_mm=tolerance_mm,
    )


def validate_j12_suite_truth(
    suite: SlidingSuite,
    search: JointSVFSearch,
) -> dict[str, AdvectedSurfaceMetrics]:
    """Refuse an optimizer batch when the generator's own truth fails contact."""
    failures: dict[str, AdvectedSurfaceMetrics] = {}
    tolerance = search.gate.advected_target_surface_p95_max_mm
    for case in suite.cases:
        pair = make_sliding_phantom_pair(case.params)
        metrics = evaluate_advected_surface_fields(
            pair,
            pair.phantom.lung_displacement_mm,
            pair.phantom.body_displacement_mm,
            tolerance_mm=tolerance,
        )
        if (
            metrics.abs_distance_p95_max_mm > tolerance
            or metrics.coverage_min < search.gate.advected_surface_coverage_min
        ):
            failures[case.case_id] = metrics
    return failures


def evaluate_j12_truth_preflight(
    suite: SlidingSuite,
    search: JointSVFSearch,
) -> tuple[J12TruthPreflightRecord, ...]:
    """Evaluate every exact hidden field before any optimizer process starts."""
    records: list[J12TruthPreflightRecord] = []
    tolerance = search.gate.advected_target_surface_p95_max_mm
    required_coverage = search.gate.advected_surface_coverage_min
    for case in suite.cases:
        pair = make_sliding_phantom_pair(case.params)
        metrics = evaluate_advected_surface_fields(
            pair,
            pair.phantom.lung_displacement_mm,
            pair.phantom.body_displacement_mm,
            tolerance_mm=tolerance,
        )
        reasons: list[str] = []
        if metrics.abs_distance_p95_max_mm > tolerance:
            reasons.append("advected_surface_p95")
        if metrics.coverage_min < required_coverage:
            reasons.append("advected_surface_coverage")
        records.append(
            J12TruthPreflightRecord(
                suite_version=suite.suite_version,
                search_version=search.search_version,
                case_id=case.case_id,
                tolerance_mm=tolerance,
                required_coverage_min=required_coverage,
                surface_point_count=metrics.surface_point_count,
                lung_abs_distance_p95_mm=metrics.lung_abs_distance_p95_mm,
                body_abs_distance_p95_mm=metrics.body_abs_distance_p95_mm,
                abs_distance_p95_max_mm=metrics.abs_distance_p95_max_mm,
                coverage_min=metrics.coverage_min,
                gap_fraction=metrics.gap_fraction,
                collision_fraction=metrics.collision_fraction,
                truth_gate_pass=not reasons,
                gate_reasons=tuple(reasons),
            )
        )
    return tuple(records)


def _gate_reasons(
    *,
    pair: SlidingPhantomPair,
    search: JointSVFSearch,
    result: JointSVFResult,
    lung_error: RegionFieldError,
    body_error: RegionFieldError,
    interface: SlidingInterfaceMetrics,
    truth_interface: SlidingInterfaceMetrics,
    advected: AdvectedSurfaceMetrics,
) -> tuple[str, ...]:
    gate = search.gate
    endpoint_limit = gate.endpoint_p95_voxels_max * max(pair.spacing_mm)
    slip_error = abs(
        interface.tangential_slip_median_mm
        - truth_interface.tangential_slip_median_mm
    )
    reasons: list[str] = []
    if result.transform_direction != pair.transform_direction:
        reasons.append("transform_direction")
    if lung_error.p95_mm > endpoint_limit:
        reasons.append("lung_field_p95")
    if body_error.p95_mm > endpoint_limit:
        reasons.append("body_field_p95")
    if interface.normal_mismatch_p95_mm > gate.normal_mismatch_p95_max_mm:
        reasons.append("normal_mismatch_p95")
    if slip_error > gate.tangential_slip_error_max_mm:
        reasons.append("tangential_slip")
    if interface.lung_jacobian_p01 < gate.jacobian_p01_min:
        reasons.append("lung_jacobian_p01")
    if interface.body_jacobian_p01 < gate.jacobian_p01_min:
        reasons.append("body_jacobian_p01")
    if (
        interface.lung_nonpositive_jacobian_fraction
        > gate.nonpositive_jacobian_fraction_max
    ):
        reasons.append("lung_folding")
    if (
        interface.body_nonpositive_jacobian_fraction
        > gate.nonpositive_jacobian_fraction_max
    ):
        reasons.append("body_folding")
    if (
        advected.abs_distance_p95_max_mm
        > gate.advected_target_surface_p95_max_mm
    ):
        reasons.append("advected_surface_p95")
    if advected.coverage_min < gate.advected_surface_coverage_min:
        reasons.append("advected_surface_coverage")
    if (
        result.runtime.peak_gpu_memory_bytes
        > search.params.maximum_peak_gpu_memory_bytes
    ):
        reasons.append("gpu_memory")
    return tuple(reasons)


def evaluate_j12_development(
    *,
    suite_version: str,
    case_id: str,
    search: JointSVFSearch,
    weights: JointSVFWeights,
    pair: SlidingPhantomPair,
    result: JointSVFResult,
) -> J12DevelopmentRecord:
    """Apply hidden truth and independent geometry only after runner completion."""
    phantom = pair.phantom
    body_wall = phantom.body_mask & ~phantom.lung_mask
    lung_error = region_field_error(
        result.lung_displacement_mm,
        phantom.lung_displacement_mm,
        phantom.lung_mask,
    )
    body_error = region_field_error(
        result.body_displacement_mm,
        phantom.body_displacement_mm,
        body_wall,
    )
    interface = evaluate_sliding_interface(
        phantom,
        lung_displacement_mm=result.lung_displacement_mm,
        body_displacement_mm=result.body_displacement_mm,
    )
    truth_interface = evaluate_sliding_interface(phantom)
    advected = evaluate_advected_surface(
        pair,
        result,
        tolerance_mm=search.gate.advected_target_surface_p95_max_mm,
    )
    reasons = _gate_reasons(
        pair=pair,
        search=search,
        result=result,
        lung_error=lung_error,
        body_error=body_error,
        interface=interface,
        truth_interface=truth_interface,
        advected=advected,
    )
    runtime = result.runtime
    return J12DevelopmentRecord(
        suite_version=suite_version,
        search_version=search.search_version,
        case_id=case_id,
        variant_id=weights.variant_id,
        runner_version=runtime.runner_version,
        transform_direction=result.transform_direction,
        coordinate_basis=phantom.coordinate_basis,
        expert_landmarks_loaded=False,
        challenge_loaded=False,
        measurement_eligible=False,
        endpoint_p95_limit_mm=(
            search.gate.endpoint_p95_voxels_max * max(pair.spacing_mm)
        ),
        lung_field_mean_mm=lung_error.mean_mm,
        lung_field_p95_mm=lung_error.p95_mm,
        lung_field_maximum_mm=lung_error.maximum_mm,
        body_field_mean_mm=body_error.mean_mm,
        body_field_p95_mm=body_error.p95_mm,
        body_field_maximum_mm=body_error.maximum_mm,
        normal_mismatch_p95_mm=interface.normal_mismatch_p95_mm,
        tangential_slip_truth_median_mm=(
            truth_interface.tangential_slip_median_mm
        ),
        tangential_slip_observed_median_mm=(
            interface.tangential_slip_median_mm
        ),
        tangential_slip_error_mm=abs(
            interface.tangential_slip_median_mm
            - truth_interface.tangential_slip_median_mm
        ),
        lung_jacobian_p01=interface.lung_jacobian_p01,
        body_jacobian_p01=interface.body_jacobian_p01,
        lung_nonpositive_jacobian_fraction=(
            interface.lung_nonpositive_jacobian_fraction
        ),
        body_nonpositive_jacobian_fraction=(
            interface.body_nonpositive_jacobian_fraction
        ),
        advected_surface_point_count=advected.surface_point_count,
        advected_lung_abs_distance_p95_mm=(
            advected.lung_abs_distance_p95_mm
        ),
        advected_body_abs_distance_p95_mm=(
            advected.body_abs_distance_p95_mm
        ),
        advected_target_surface_p95_max_mm=(
            advected.abs_distance_p95_max_mm
        ),
        advected_surface_coverage_min=advected.coverage_min,
        advected_gap_fraction=advected.gap_fraction,
        advected_collision_fraction=advected.collision_fraction,
        objective_initial=runtime.objective_initial,
        objective_final=runtime.objective_final,
        rejected_topology_updates=runtime.rejected_topology_updates,
        final_learning_rate=runtime.final_learning_rate,
        elapsed_s=runtime.elapsed_s,
        peak_gpu_memory_bytes=runtime.peak_gpu_memory_bytes,
        torch_version=runtime.torch_version,
        convexadam_version=runtime.convexadam_version,
        device=runtime.device,
        gate_pass=not reasons,
        gate_reasons=reasons,
    )


def run_j12_development(
    suite: SlidingSuite,
    search: JointSVFSearch,
    *,
    registration_python: Path,
    repo_root: Path,
    temporary_root: Path,
) -> tuple[J12DevelopmentRun, ...]:
    """Run the finite 3 x 3 development search without touching challenge data."""
    truth_failures = validate_j12_suite_truth(suite, search)
    if truth_failures:
        details = ", ".join(
            f"{case_id}:p95={metrics.abs_distance_p95_max_mm:.3f},"
            f"coverage={metrics.coverage_min:.3f}"
            for case_id, metrics in truth_failures.items()
        )
        raise ValueError(
            "J1.2 suite truth fails the advected-surface preflight; "
            f"optimizer is blocked ({details})"
        )
    runs: list[J12DevelopmentRun] = []
    for weights in search.variants:
        for case in suite.cases:
            pair = make_sliding_phantom_pair(case.params)
            result = register_joint_piecewise_svf(
                cast(WallIntArray, pair.fixed_image),
                cast(WallIntArray, pair.moving_image),
                pair.fixed_lung_mask,
                pair.moving_lung_mask,
                pair.fixed_body_mask,
                pair.moving_body_mask,
                pair.spacing_mm,
                search=search,
                weights=weights,
                python_executable=registration_python,
                repo_root=repo_root,
                temporary_root=temporary_root,
            )
            record = evaluate_j12_development(
                suite_version=suite.suite_version,
                case_id=case.case_id,
                search=search,
                weights=weights,
                pair=pair,
                result=result,
            )
            runs.append(
                J12DevelopmentRun(
                    pair=pair,
                    result=result,
                    record=record,
                )
            )
    return tuple(runs)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _code_version(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _selected_variant(
    records: list[J12DevelopmentRecord],
    search: JointSVFSearch,
) -> str | None:
    eligible: list[tuple[float, float, int, str]] = []
    for variant in search.variants:
        rows = [row for row in records if row.variant_id == variant.variant_id]
        if len(rows) != 3 or not all(row.gate_pass for row in rows):
            continue
        eligible.append(
            (
                float(np.median([row.lung_field_p95_mm for row in rows])),
                float(
                    np.median(
                        [
                            row.advected_target_surface_p95_max_mm
                            for row in rows
                        ]
                    )
                ),
                max(row.peak_gpu_memory_bytes for row in rows),
                variant.variant_id,
            )
        )
    return min(eligible)[3] if eligible else None


def write_j12_truth_preflight(
    records: tuple[J12TruthPreflightRecord, ...],
    output_dir: Path,
    *,
    suite_path: Path,
    search_path: Path,
    search: JointSVFSearch,
    repo_root: Path,
) -> Path:
    """Write exact-truth eligibility evidence without opening the challenge."""
    if not records:
        raise ValueError("cannot write an empty J1.2 truth preflight")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    rows = [record.as_csv_row() for record in records]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "artifact_type": "piecewise_svf_j12_truth_preflight",
        "usage": "synthetic_generator_QC_ONLY",
        "suite_version": records[0].suite_version,
        "search_version": records[0].search_version,
        "code_version": _code_version(repo_root),
        "suite_config": {
            "path": suite_path.name,
            "sha256": _sha256(suite_path),
        },
        "search_config": {
            "path": search_path.name,
            "sha256": _sha256(search_path),
        },
        "challenge": {
            "path_recorded_but_not_loaded": search.challenge_suite_path,
            "loaded": False,
        },
        "case_count": len(records),
        "truth_pass_count": sum(record.truth_gate_pass for record in records),
        "all_truth_pass": all(record.truth_gate_pass for record in records),
        "optimizer_started": False,
        "development_search_allowed": all(
            record.truth_gate_pass for record in records
        ),
        "summary_sha256": _sha256(summary_path),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return manifest_path


def write_j12_development_batch(
    runs: tuple[J12DevelopmentRun, ...],
    output_dir: Path,
    *,
    suite_path: Path,
    search_path: Path,
    search: JointSVFSearch,
    repo_root: Path,
) -> Path:
    """Write fields, summary, finite-search decision and checksummed provenance."""
    expected_count = len(search.variants) * 3
    if len(runs) != expected_count:
        raise ValueError(f"expected {expected_count} development runs")
    output_dir.mkdir(parents=True, exist_ok=True)
    field_hashes: dict[str, str] = {}
    record_hashes: dict[str, str] = {}
    for run in runs:
        stem = f"{run.record.variant_id}__{run.record.case_id}"
        field_path = output_dir / f"{stem}.npz"
        np.savez_compressed(
            field_path,
            usage=np.asarray("synthetic_development_ONLY"),
            lung_displacement_mm=run.result.lung_displacement_mm,
            body_displacement_mm=run.result.body_displacement_mm,
            lung_velocity_mm=run.result.lung_velocity_mm,
            body_velocity_mm=run.result.body_velocity_mm,
            truth_lung_displacement_mm=(
                run.pair.phantom.lung_displacement_mm
            ),
            truth_body_displacement_mm=(
                run.pair.phantom.body_displacement_mm
            ),
            fixed_lung_mask=run.pair.fixed_lung_mask,
            moving_lung_mask=run.pair.moving_lung_mask,
            spacing_mm=np.asarray(run.pair.spacing_mm),
            transform_direction=np.asarray(run.result.transform_direction),
        )
        field_hashes[field_path.name] = _sha256(field_path)
        record_path = output_dir / f"{stem}.json"
        record_path.write_text(
            json.dumps(
                asdict(run.record),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        record_hashes[record_path.name] = _sha256(record_path)
    records = [run.record for run in runs]
    summary_path = output_dir / "summary.csv"
    rows = [record.as_csv_row() for record in records]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selected = _selected_variant(records, search)
    pass_counts = {
        variant.variant_id: sum(
            record.gate_pass
            for record in records
            if record.variant_id == variant.variant_id
        )
        for variant in search.variants
    }
    manifest = {
        "artifact_type": "piecewise_svf_j12_development_search",
        "usage": "synthetic_development_ONLY",
        "suite_version": records[0].suite_version,
        "search_version": search.search_version,
        "code_version": _code_version(repo_root),
        "suite_config": {
            "path": suite_path.name,
            "sha256": _sha256(suite_path),
        },
        "search_config": {
            "path": search_path.name,
            "sha256": _sha256(search_path),
        },
        "challenge": {
            "path_recorded_but_not_loaded": search.challenge_suite_path,
            "hold_condition": search.challenge_hold_condition,
            "required_result": search.challenge_required_result,
            "loaded": False,
        },
        "run_count": len(records),
        "variant_pass_counts": pass_counts,
        "selected_variant": selected,
        "candidate_freeze_allowed": selected is not None,
        "summary_sha256": _sha256(summary_path),
        "field_sha256": field_hashes,
        "record_sha256": record_hashes,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return manifest_path


__all__ = [
    "AdvectedSurfaceMetrics",
    "J12DevelopmentRecord",
    "J12DevelopmentRun",
    "J12TruthPreflightRecord",
    "evaluate_advected_surface",
    "evaluate_advected_surface_fields",
    "evaluate_j12_development",
    "evaluate_j12_truth_preflight",
    "run_j12_development",
    "validate_j12_suite_truth",
    "write_j12_development_batch",
    "write_j12_truth_preflight",
]
