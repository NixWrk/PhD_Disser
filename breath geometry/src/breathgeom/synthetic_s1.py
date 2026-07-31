"""Batch execution and provenance for the frozen S1 synthetic gate."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, cast

import numpy as np

from breathgeom.measure.registration import VectorArray
from breathgeom.measure.sliding_phantom import (
    SlidingPhantomPair,
    SlidingPhantomParams,
    evaluate_sliding_interface,
    make_sliding_phantom_pair,
    region_field_error,
)
from breathgeom.measure.sliding_registration import (
    SlidingS1Params,
    SlidingS1Result,
    register_sliding_s1,
)
from breathgeom.measure.wall import IntArray as WallIntArray

CASE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class SlidingSuiteCase:
    """One frozen phantom variant."""

    case_id: str
    description: str
    params: SlidingPhantomParams


@dataclass(frozen=True)
class SlidingSuite:
    """Versioned collection evaluated as one algorithmic gate."""

    suite_version: str
    cases: tuple[SlidingSuiteCase, ...]


@dataclass(frozen=True)
class SyntheticS1Gate:
    """Predeclared S1 synthetic acceptance thresholds."""

    endpoint_p95_voxels_max: float = 1.5
    normal_mismatch_p95_max_mm: float = 1.0
    tangential_slip_error_max_mm: float = 1.0
    jacobian_p01_min: float = 0.10
    nonpositive_jacobian_fraction_max: float = 0.0


@dataclass(frozen=True)
class SyntheticS1Record:
    """One subject-like synthetic case result."""

    suite_version: str
    case_id: str
    s1_version: str
    transform_direction: str
    shape: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    normal_motion_truth_mm: float
    tangential_slip_truth_mm: float
    endpoint_p95_limit_mm: float
    lung_field_mean_mm: float
    lung_field_p95_mm: float
    lung_field_maximum_mm: float
    body_field_mean_mm: float
    body_field_p95_mm: float
    body_field_maximum_mm: float
    raw_normal_mismatch_p95_mm: float
    normal_mismatch_p95_mm: float
    raw_tangential_slip_median_mm: float
    tangential_slip_median_mm: float
    tangential_slip_error_mm: float
    lung_jacobian_p01: float
    lung_nonpositive_jacobian_fraction: float
    body_jacobian_p01: float
    body_nonpositive_jacobian_fraction: float
    lung_runtime_s: float
    body_runtime_s: float
    torch_version: str
    convexadam_version: str
    device: str
    gate_pass: bool
    gate_reasons: tuple[str, ...]

    def as_csv_row(self) -> dict[str, str | float | bool]:
        values = asdict(self)
        values["shape"] = "x".join(str(value) for value in self.shape)
        values["spacing_mm"] = "x".join(str(value) for value in self.spacing_mm)
        values["gate_reasons"] = ";".join(self.gate_reasons)
        return cast(dict[str, str | float | bool], values)


@dataclass(frozen=True)
class SyntheticS1Run:
    """In-memory case result used to write diagnostic synthetic artifacts."""

    record: SyntheticS1Record
    pair: SlidingPhantomPair
    result: SlidingS1Result


def _tuple_values(
    payload: dict[str, Any],
    key: str,
    length: int,
    caster: type[int] | type[float],
) -> tuple[Any, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{key} must be a JSON array of length {length}")
    return tuple(caster(item) for item in value)


def load_sliding_suite(path: Path) -> SlidingSuite:
    """Load an exact suite schema and reject path-like case identifiers."""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"suite_version", "cases"}:
        raise ValueError("suite must contain exactly suite_version and cases")
    suite_version = payload["suite_version"]
    case_payloads = payload["cases"]
    if not isinstance(suite_version, str) or not suite_version:
        raise ValueError("suite_version must be a non-empty string")
    if not isinstance(case_payloads, list) or not case_payloads:
        raise ValueError("cases must be a non-empty JSON array")
    allowed_params = {item.name for item in fields(SlidingPhantomParams)}
    cases: list[SlidingSuiteCase] = []
    identifiers: set[str] = set()
    for item in case_payloads:
        if not isinstance(item, dict) or set(item) != {
            "case_id",
            "description",
            "params",
        }:
            raise ValueError("each suite case must contain case_id, description and params")
        case_id = item["case_id"]
        description = item["description"]
        params_payload = item["params"]
        if (
            not isinstance(case_id, str)
            or not CASE_ID_PATTERN.fullmatch(case_id)
            or case_id in identifiers
        ):
            raise ValueError(f"invalid or duplicate case_id: {case_id!r}")
        if not isinstance(description, str) or not description:
            raise ValueError(f"{case_id}: description must be non-empty")
        if not isinstance(params_payload, dict):
            raise ValueError(f"{case_id}: params must be a JSON object")
        unknown = set(params_payload) - allowed_params
        if unknown:
            raise ValueError(f"{case_id}: unknown phantom params {sorted(unknown)}")
        params_values = cast(dict[str, Any], params_payload).copy()
        # Phantom v2.0 predates the explicit ramp-mode field.  Preserve its
        # historical axis-singular truth so the committed config checksum and
        # superseded S1.0 report remain reproducible.
        if (
            suite_version == "sliding-phantom-v2.0"
            and "tangential_ramp_mode" not in params_values
        ):
            params_values["tangential_ramp_mode"] = "ellipsoidal_v1"
        params_values["shape"] = _tuple_values(params_values, "shape", 3, int)
        for key in ("spacing_mm", "lung_radii_mm", "body_radii_mm"):
            params_values[key] = _tuple_values(params_values, key, 3, float)
        cases.append(
            SlidingSuiteCase(
                case_id=case_id,
                description=description,
                params=SlidingPhantomParams(**params_values),
            )
        )
        identifiers.add(case_id)
    return SlidingSuite(suite_version=suite_version, cases=tuple(cases))


def evaluate_synthetic_s1_fields(
    *,
    suite_version: str,
    case: SlidingSuiteCase,
    pair: SlidingPhantomPair,
    s1_version: str,
    lung_displacement_mm: VectorArray,
    body_displacement_mm: VectorArray,
    raw_lung_displacement_mm: VectorArray | None = None,
    raw_body_displacement_mm: VectorArray | None = None,
    gate: SyntheticS1Gate | None = None,
    transform_direction: str = "fixed-expiration_to_moving-inspiration",
    lung_runtime_s: float = float("nan"),
    body_runtime_s: float = float("nan"),
    torch_version: str = "",
    convexadam_version: str = "",
    device: str = "",
) -> SyntheticS1Record:
    """Evaluate fields without exposing them to the registration candidate."""
    gate = gate or SyntheticS1Gate()
    phantom = pair.phantom
    body_wall = phantom.body_mask & ~phantom.lung_mask
    lung_error = region_field_error(
        lung_displacement_mm,
        phantom.lung_displacement_mm,
        phantom.lung_mask,
    )
    body_error = region_field_error(
        body_displacement_mm,
        phantom.body_displacement_mm,
        body_wall,
    )
    metrics = evaluate_sliding_interface(
        phantom,
        lung_displacement_mm=lung_displacement_mm,
        body_displacement_mm=body_displacement_mm,
    )
    raw_metrics = evaluate_sliding_interface(
        phantom,
        lung_displacement_mm=(
            lung_displacement_mm
            if raw_lung_displacement_mm is None
            else raw_lung_displacement_mm
        ),
        body_displacement_mm=(
            body_displacement_mm
            if raw_body_displacement_mm is None
            else raw_body_displacement_mm
        ),
    )
    truth_metrics = evaluate_sliding_interface(phantom)
    endpoint_limit = gate.endpoint_p95_voxels_max * max(pair.spacing_mm)
    slip_error = abs(
        metrics.tangential_slip_median_mm
        - truth_metrics.tangential_slip_median_mm
    )
    reasons: list[str] = []
    if transform_direction != pair.transform_direction:
        reasons.append("transform_direction")
    if lung_error.p95_mm > endpoint_limit:
        reasons.append("lung_field_p95")
    if body_error.p95_mm > endpoint_limit:
        reasons.append("body_field_p95")
    if metrics.normal_mismatch_p95_mm > gate.normal_mismatch_p95_max_mm:
        reasons.append("normal_mismatch_p95")
    if slip_error > gate.tangential_slip_error_max_mm:
        reasons.append("tangential_slip")
    if metrics.lung_jacobian_p01 < gate.jacobian_p01_min:
        reasons.append("lung_jacobian_p01")
    if (
        metrics.lung_nonpositive_jacobian_fraction
        > gate.nonpositive_jacobian_fraction_max
    ):
        reasons.append("lung_folding")
    if metrics.body_jacobian_p01 < gate.jacobian_p01_min:
        reasons.append("body_jacobian_p01")
    if (
        metrics.body_nonpositive_jacobian_fraction
        > gate.nonpositive_jacobian_fraction_max
    ):
        reasons.append("body_folding")

    return SyntheticS1Record(
        suite_version=suite_version,
        case_id=case.case_id,
        s1_version=s1_version,
        transform_direction=transform_direction,
        shape=case.params.shape,
        spacing_mm=case.params.spacing_mm,
        normal_motion_truth_mm=case.params.normal_motion_mm,
        tangential_slip_truth_mm=truth_metrics.tangential_slip_median_mm,
        endpoint_p95_limit_mm=endpoint_limit,
        lung_field_mean_mm=lung_error.mean_mm,
        lung_field_p95_mm=lung_error.p95_mm,
        lung_field_maximum_mm=lung_error.maximum_mm,
        body_field_mean_mm=body_error.mean_mm,
        body_field_p95_mm=body_error.p95_mm,
        body_field_maximum_mm=body_error.maximum_mm,
        raw_normal_mismatch_p95_mm=raw_metrics.normal_mismatch_p95_mm,
        normal_mismatch_p95_mm=metrics.normal_mismatch_p95_mm,
        raw_tangential_slip_median_mm=raw_metrics.tangential_slip_median_mm,
        tangential_slip_median_mm=metrics.tangential_slip_median_mm,
        tangential_slip_error_mm=slip_error,
        lung_jacobian_p01=metrics.lung_jacobian_p01,
        lung_nonpositive_jacobian_fraction=(
            metrics.lung_nonpositive_jacobian_fraction
        ),
        body_jacobian_p01=metrics.body_jacobian_p01,
        body_nonpositive_jacobian_fraction=(
            metrics.body_nonpositive_jacobian_fraction
        ),
        lung_runtime_s=lung_runtime_s,
        body_runtime_s=body_runtime_s,
        torch_version=torch_version,
        convexadam_version=convexadam_version,
        device=device,
        gate_pass=not reasons,
        gate_reasons=tuple(reasons),
    )


def run_synthetic_s1_suite(
    suite: SlidingSuite,
    s1_params: SlidingS1Params,
    *,
    registration_python: Path,
    repo_root: Path,
    temporary_root: Path,
    gate: SyntheticS1Gate | None = None,
) -> tuple[SyntheticS1Run, ...]:
    """Run the frozen suite; truth is used only after each S1 call returns."""
    runs: list[SyntheticS1Run] = []
    for case in suite.cases:
        pair = make_sliding_phantom_pair(case.params)
        result = register_sliding_s1(
            cast(WallIntArray, pair.fixed_image),
            cast(WallIntArray, pair.moving_image),
            pair.spacing_mm,
            pair.fixed_lung_mask,
            pair.moving_lung_mask,
            pair.fixed_body_mask,
            pair.moving_body_mask,
            registration_python=registration_python,
            repo_root=repo_root,
            params=s1_params,
            temporary_root=temporary_root,
        )
        record = evaluate_synthetic_s1_fields(
            suite_version=suite.suite_version,
            case=case,
            pair=pair,
            s1_version=s1_params.version,
            lung_displacement_mm=result.lung_displacement_mm,
            body_displacement_mm=result.body_displacement_mm,
            raw_lung_displacement_mm=result.raw_lung_displacement_mm,
            raw_body_displacement_mm=result.raw_body_displacement_mm,
            gate=gate,
            transform_direction=result.transform_direction,
            lung_runtime_s=result.lung_runtime.elapsed_s,
            body_runtime_s=result.body_elapsed_s,
            torch_version=result.lung_runtime.torch_version,
            convexadam_version=result.lung_runtime.convexadam_version,
            device=result.lung_runtime.device,
        )
        runs.append(SyntheticS1Run(record=record, pair=pair, result=result))
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


def write_synthetic_s1_suite(
    runs: tuple[SyntheticS1Run, ...],
    output_dir: Path,
    *,
    suite_path: Path,
    s1_config_path: Path,
    repo_root: Path,
    gate: SyntheticS1Gate | None = None,
) -> Path:
    """Write local batch artifacts before any notebook visualisation."""
    if not runs:
        raise ValueError("cannot write an empty synthetic suite")
    gate = gate or SyntheticS1Gate()
    output_dir.mkdir(parents=True, exist_ok=True)
    field_hashes: dict[str, str] = {}
    for run in runs:
        case_id = run.record.case_id
        if not CASE_ID_PATTERN.fullmatch(case_id):
            raise ValueError(f"unsafe case_id: {case_id!r}")
        field_path = output_dir / f"{case_id}.npz"
        np.savez_compressed(
            field_path,
            usage=np.asarray("synthetic_diagnostic_ONLY"),
            fixed_image=run.pair.fixed_image,
            moving_image=run.pair.moving_image,
            fixed_lung_mask=run.pair.fixed_lung_mask,
            moving_lung_mask=run.pair.moving_lung_mask,
            fixed_body_mask=run.pair.fixed_body_mask,
            moving_body_mask=run.pair.moving_body_mask,
            truth_lung_displacement_mm=run.pair.phantom.lung_displacement_mm,
            truth_body_displacement_mm=run.pair.phantom.body_displacement_mm,
            estimated_lung_displacement_mm=run.result.lung_displacement_mm,
            estimated_body_displacement_mm=run.result.body_displacement_mm,
            raw_lung_displacement_mm=run.result.raw_lung_displacement_mm,
            raw_body_displacement_mm=run.result.raw_body_displacement_mm,
            spacing_mm=np.asarray(run.pair.spacing_mm),
            transform_direction=np.asarray(run.record.transform_direction),
        )
        field_hashes[field_path.name] = _sha256(field_path)
        (output_dir / f"{case_id}.json").write_text(
            json.dumps(asdict(run.record), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    summary_path = output_dir / "summary.csv"
    rows = [run.record.as_csv_row() for run in runs]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "artifact_type": "synthetic_s1_suite",
        "usage": "synthetic_diagnostic_ONLY",
        "suite_version": runs[0].record.suite_version,
        "s1_version": runs[0].record.s1_version,
        "code_version": _code_version(repo_root),
        "suite_config": {
            "path": suite_path.name,
            "sha256": _sha256(suite_path),
        },
        "s1_config": {
            "path": s1_config_path.name,
            "sha256": _sha256(s1_config_path),
        },
        "gate": asdict(gate),
        "case_count": len(runs),
        "pass_count": sum(run.record.gate_pass for run in runs),
        "all_pass": all(run.record.gate_pass for run in runs),
        "field_sha256": field_hashes,
        "summary_sha256": _sha256(summary_path),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest_path


__all__ = [
    "SlidingSuite",
    "SlidingSuiteCase",
    "SyntheticS1Gate",
    "SyntheticS1Record",
    "SyntheticS1Run",
    "evaluate_synthetic_s1_fields",
    "load_sliding_suite",
    "run_synthetic_s1_suite",
    "write_synthetic_s1_suite",
]
