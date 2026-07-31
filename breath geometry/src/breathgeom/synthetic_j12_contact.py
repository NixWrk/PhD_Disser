"""Exact-truth preflight for the frozen contact-valid J1.2 development suite."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np

from breathgeom.measure.contact_svf_phantom import (
    ContactSVFPhantomPair,
    ContactSVFSuite,
    analytic_target_signed_distance,
    make_contact_svf_phantom_pair,
    sample_fixed_interface_points,
)
from breathgeom.measure.joint_svf_registration import (
    JointSVFSearch,
    register_joint_piecewise_svf,
)
from breathgeom.measure.registration import (
    displacement_round_trip_metrics,
    transform_points,
)
from breathgeom.measure.sliding_phantom import evaluate_sliding_interface
from breathgeom.measure.wall import IntArray as WallIntArray
from breathgeom.synthetic_j12 import (
    AdvectedSurfaceMetrics,
    J12DevelopmentRun,
    evaluate_advected_surface_fields,
    evaluate_j12_development,
)


@dataclass(frozen=True)
class ContactTruthGate:
    """Thresholds frozen in the J1.2 v2 protocol before implementation."""

    analytic_surface_p95_max_mm: float = 0.05
    analytic_surface_coverage_min: float = 0.99
    round_trip_p95_max_mm: float = 0.05
    jacobian_p01_min: float = 0.2
    nonpositive_jacobian_fraction_max: float = 0.0
    tangential_slip_median_min_mm: float = 0.5


@dataclass(frozen=True)
class ContactTruthPreflightRecord:
    """One case-level proof that the hidden truth is eligible for optimization."""

    suite_version: str
    generator_version: str
    search_version: str
    case_id: str
    transform_direction: str
    coordinate_basis: str
    analytic_surface_tolerance_mm: float
    analytic_surface_required_coverage_min: float
    analytic_surface_point_count: int
    analytic_lung_abs_distance_p95_mm: float
    analytic_body_abs_distance_p95_mm: float
    analytic_abs_distance_p95_max_mm: float
    analytic_surface_coverage_min: float
    raster_surface_tolerance_mm: float
    raster_surface_required_coverage_min: float
    raster_surface_point_count: int
    raster_lung_abs_distance_p95_mm: float
    raster_body_abs_distance_p95_mm: float
    raster_abs_distance_p95_max_mm: float
    raster_surface_coverage_min: float
    raster_gap_fraction: float
    raster_collision_fraction: float
    lung_round_trip_p95_mm: float
    body_round_trip_p95_mm: float
    normal_mismatch_p95_mm: float
    tangential_slip_median_mm: float
    lung_jacobian_p01: float
    body_jacobian_p01: float
    lung_nonpositive_jacobian_fraction: float
    body_nonpositive_jacobian_fraction: float
    truth_gate_pass: bool
    gate_reasons: tuple[str, ...]

    def as_csv_row(self) -> dict[str, str | float | int | bool]:
        values = asdict(self)
        values["gate_reasons"] = ";".join(self.gate_reasons)
        return cast(dict[str, str | float | int | bool], values)


@dataclass(frozen=True)
class ContactTruthPreflightRun:
    """Record and synthetic diagnostics kept outside the optimizer process."""

    record: ContactTruthPreflightRecord
    generated: ContactSVFPhantomPair
    fixed_surface_points_mm: np.ndarray
    advected_lung_surface_points_mm: np.ndarray
    advected_body_surface_points_mm: np.ndarray


def _gate_reasons(
    record: ContactTruthPreflightRecord,
    gate: ContactTruthGate,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if record.analytic_abs_distance_p95_max_mm > gate.analytic_surface_p95_max_mm:
        reasons.append("analytic_surface_p95")
    if record.analytic_surface_coverage_min < gate.analytic_surface_coverage_min:
        reasons.append("analytic_surface_coverage")
    if record.raster_abs_distance_p95_max_mm > record.raster_surface_tolerance_mm:
        reasons.append("raster_surface_p95")
    if (
        record.raster_surface_coverage_min
        < record.raster_surface_required_coverage_min
    ):
        reasons.append("raster_surface_coverage")
    if record.lung_round_trip_p95_mm > gate.round_trip_p95_max_mm:
        reasons.append("lung_round_trip")
    if record.body_round_trip_p95_mm > gate.round_trip_p95_max_mm:
        reasons.append("body_round_trip")
    if record.lung_jacobian_p01 < gate.jacobian_p01_min:
        reasons.append("lung_jacobian_p01")
    if record.body_jacobian_p01 < gate.jacobian_p01_min:
        reasons.append("body_jacobian_p01")
    if (
        record.lung_nonpositive_jacobian_fraction
        > gate.nonpositive_jacobian_fraction_max
    ):
        reasons.append("lung_folding")
    if (
        record.body_nonpositive_jacobian_fraction
        > gate.nonpositive_jacobian_fraction_max
    ):
        reasons.append("body_folding")
    if record.tangential_slip_median_mm < gate.tangential_slip_median_min_mm:
        reasons.append("tangential_slip")
    return tuple(reasons)


def evaluate_contact_truth_preflight(
    suite: ContactSVFSuite,
    search: JointSVFSearch,
    *,
    gate: ContactTruthGate | None = None,
) -> tuple[ContactTruthPreflightRun, ...]:
    """Evaluate exact hidden fields without loading or invoking the CUDA runner."""
    gate = gate or ContactTruthGate()
    runs: list[ContactTruthPreflightRun] = []
    raster_tolerance = search.gate.advected_target_surface_p95_max_mm
    raster_coverage = search.gate.advected_surface_coverage_min
    for case in suite.cases:
        generated = make_contact_svf_phantom_pair(case.params)
        pair = generated.pair
        fixed_points = sample_fixed_interface_points(case.params)
        lung_points = transform_points(
            fixed_points,
            pair.phantom.lung_displacement_mm,
            pair.spacing_mm,
        )
        body_points = transform_points(
            fixed_points,
            pair.phantom.body_displacement_mm,
            pair.spacing_mm,
        )
        lung_signed = analytic_target_signed_distance(
            lung_points,
            center_mm=generated.center_mm,
            target_radii_mm=generated.target_lung_radii_mm,
        )
        body_signed = analytic_target_signed_distance(
            body_points,
            center_mm=generated.center_mm,
            target_radii_mm=generated.target_lung_radii_mm,
        )
        analytic_lung_p95 = float(np.percentile(np.abs(lung_signed), 95))
        analytic_body_p95 = float(np.percentile(np.abs(body_signed), 95))
        analytic_coverage = min(
            float(np.mean(np.abs(lung_signed) <= gate.analytic_surface_p95_max_mm)),
            float(np.mean(np.abs(body_signed) <= gate.analytic_surface_p95_max_mm)),
        )
        raster: AdvectedSurfaceMetrics = evaluate_advected_surface_fields(
            pair,
            pair.phantom.lung_displacement_mm,
            pair.phantom.body_displacement_mm,
            tolerance_mm=raster_tolerance,
        )
        body_wall = pair.phantom.body_mask & ~pair.phantom.lung_mask
        lung_round_trip = displacement_round_trip_metrics(
            pair.phantom.lung_displacement_mm,
            pair.lung_moving_to_fixed_mm,
            pair.spacing_mm,
            valid_domain=pair.phantom.lung_mask,
        )
        body_round_trip = displacement_round_trip_metrics(
            pair.phantom.body_displacement_mm,
            pair.body_moving_to_fixed_mm,
            pair.spacing_mm,
            valid_domain=body_wall,
        )
        interface = evaluate_sliding_interface(pair.phantom)
        provisional = ContactTruthPreflightRecord(
            suite_version=suite.suite_version,
            generator_version=suite.generator_version,
            search_version=search.search_version,
            case_id=case.case_id,
            transform_direction=pair.transform_direction,
            coordinate_basis=pair.phantom.coordinate_basis,
            analytic_surface_tolerance_mm=gate.analytic_surface_p95_max_mm,
            analytic_surface_required_coverage_min=(
                gate.analytic_surface_coverage_min
            ),
            analytic_surface_point_count=len(fixed_points),
            analytic_lung_abs_distance_p95_mm=analytic_lung_p95,
            analytic_body_abs_distance_p95_mm=analytic_body_p95,
            analytic_abs_distance_p95_max_mm=max(
                analytic_lung_p95,
                analytic_body_p95,
            ),
            analytic_surface_coverage_min=analytic_coverage,
            raster_surface_tolerance_mm=raster_tolerance,
            raster_surface_required_coverage_min=raster_coverage,
            raster_surface_point_count=raster.surface_point_count,
            raster_lung_abs_distance_p95_mm=(
                raster.lung_abs_distance_p95_mm
            ),
            raster_body_abs_distance_p95_mm=(
                raster.body_abs_distance_p95_mm
            ),
            raster_abs_distance_p95_max_mm=(
                raster.abs_distance_p95_max_mm
            ),
            raster_surface_coverage_min=raster.coverage_min,
            raster_gap_fraction=raster.gap_fraction,
            raster_collision_fraction=raster.collision_fraction,
            lung_round_trip_p95_mm=lung_round_trip.p95_mm,
            body_round_trip_p95_mm=body_round_trip.p95_mm,
            normal_mismatch_p95_mm=interface.normal_mismatch_p95_mm,
            tangential_slip_median_mm=interface.tangential_slip_median_mm,
            lung_jacobian_p01=interface.lung_jacobian_p01,
            body_jacobian_p01=interface.body_jacobian_p01,
            lung_nonpositive_jacobian_fraction=(
                interface.lung_nonpositive_jacobian_fraction
            ),
            body_nonpositive_jacobian_fraction=(
                interface.body_nonpositive_jacobian_fraction
            ),
            truth_gate_pass=False,
            gate_reasons=(),
        )
        reasons = _gate_reasons(provisional, gate)
        record = ContactTruthPreflightRecord(
            **(
                asdict(provisional)
                | {
                    "truth_gate_pass": not reasons,
                    "gate_reasons": reasons,
                }
            )
        )
        runs.append(
            ContactTruthPreflightRun(
                record=record,
                generated=generated,
                fixed_surface_points_mm=fixed_points,
                advected_lung_surface_points_mm=lung_points,
                advected_body_surface_points_mm=body_points,
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


def write_contact_truth_preflight(
    runs: tuple[ContactTruthPreflightRun, ...],
    output_dir: Path,
    *,
    suite_path: Path,
    search_path: Path,
    search: JointSVFSearch,
    repo_root: Path,
    gate: ContactTruthGate | None = None,
) -> Path:
    """Write checksummed exact-truth evidence without opening the challenge."""
    if not runs:
        raise ValueError("cannot write an empty contact truth preflight")
    gate = gate or ContactTruthGate()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [run.record.as_csv_row() for run in runs]
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    diagnostic_hashes: dict[str, str] = {}
    for run in runs:
        diagnostic_path = output_dir / f"{run.record.case_id}__surface.npz"
        np.savez_compressed(
            diagnostic_path,
            usage=np.asarray("synthetic_generator_QC_ONLY"),
            fixed_lung_mask=run.generated.pair.fixed_lung_mask,
            moving_lung_mask=run.generated.pair.moving_lung_mask,
            fixed_surface_points_mm=run.fixed_surface_points_mm,
            advected_lung_surface_points_mm=(
                run.advected_lung_surface_points_mm
            ),
            advected_body_surface_points_mm=(
                run.advected_body_surface_points_mm
            ),
            target_lung_radii_mm=np.asarray(
                run.generated.target_lung_radii_mm,
            ),
            center_mm=np.asarray(run.generated.center_mm),
            spacing_mm=np.asarray(run.generated.pair.spacing_mm),
            transform_direction=np.asarray(run.record.transform_direction),
            coordinate_basis=np.asarray(run.record.coordinate_basis),
        )
        diagnostic_hashes[diagnostic_path.name] = _sha256(diagnostic_path)
    all_pass = all(run.record.truth_gate_pass for run in runs)
    manifest = {
        "artifact_type": "piecewise_svf_j12_contact_truth_preflight",
        "usage": "synthetic_generator_QC_ONLY",
        "suite_version": runs[0].record.suite_version,
        "generator_version": runs[0].record.generator_version,
        "search_version": runs[0].record.search_version,
        "code_version": _code_version(repo_root),
        "suite_config": {
            "path": suite_path.name,
            "sha256": _sha256(suite_path),
        },
        "search_config": {
            "path": search_path.name,
            "sha256": _sha256(search_path),
        },
        "truth_gate": asdict(gate),
        "challenge": {
            "path_recorded_but_not_loaded": search.challenge_suite_path,
            "loaded": False,
        },
        "case_count": len(runs),
        "truth_pass_count": sum(run.record.truth_gate_pass for run in runs),
        "all_truth_pass": all_pass,
        "optimizer_started": False,
        "development_search_allowed": all_pass,
        "summary_sha256": _sha256(summary_path),
        "diagnostic_sha256": diagnostic_hashes,
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


def run_contact_j12_development(
    suite: ContactSVFSuite,
    search: JointSVFSearch,
    *,
    registration_python: Path,
    repo_root: Path,
    temporary_root: Path,
) -> tuple[J12DevelopmentRun, ...]:
    """Run the frozen finite search only after contact truth passes every gate."""
    preflight = evaluate_contact_truth_preflight(suite, search)
    failures = [run.record for run in preflight if not run.record.truth_gate_pass]
    if failures:
        details = ", ".join(
            f"{record.case_id}:{';'.join(record.gate_reasons)}"
            for record in failures
        )
        raise ValueError(
            "contact J1.2 suite truth fails preflight; "
            f"optimizer is blocked ({details})"
        )
    runs: list[J12DevelopmentRun] = []
    for weights in search.variants:
        for case in suite.cases:
            generated = make_contact_svf_phantom_pair(case.params)
            pair = generated.pair
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


__all__ = [
    "ContactTruthGate",
    "ContactTruthPreflightRecord",
    "ContactTruthPreflightRun",
    "evaluate_contact_truth_preflight",
    "run_contact_j12_development",
    "write_contact_truth_preflight",
]
