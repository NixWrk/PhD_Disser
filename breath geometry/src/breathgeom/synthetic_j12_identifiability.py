"""Development-only data-term identifiability screen for joint SVF J1.2."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from breathgeom.measure.contact_svf_phantom import (
    ContactSVFPhantomPair,
    ContactSVFSuite,
    ContactSVFSuiteCase,
    make_contact_svf_phantom_pair,
)
from breathgeom.measure.piecewise_svf import exponentiate_piecewise_svf
from breathgeom.measure.registration import VectorArray

CANDIDATE_NAMES = (
    "truth",
    "estimated_balanced",
    "glued_mean_velocity",
    "common_scale_only",
    "zero",
    "inverse_truth",
)
METRIC_NAMES = (
    "mind_mse_full",
    "mind_mse_interior",
    "intensity_mse_full",
    "intensity_mse_interior",
)


@dataclass(frozen=True)
class IdentifiabilityThresholds:
    """Frozen ratio thresholds for the diagnostic decision tree."""

    truth_to_wrong_direction_ratio_max: float
    truth_to_zero_ratio_max: float
    truth_to_glued_ratio_max: float
    estimated_to_zero_ratio_max: float


@dataclass(frozen=True)
class IdentifiabilityScreen:
    """Strict development-only screen configuration."""

    screen_version: str
    suite_path: str
    development_batch_manifest_path: str
    runner_version: str
    device: str
    dtype: str
    descriptor: dict[str, str | int]
    candidates: tuple[str, ...]
    metrics: tuple[str, ...]
    thresholds: IdentifiabilityThresholds


@dataclass(frozen=True)
class DataTermRecord:
    """One candidate/region objective row returned by the external evaluator."""

    screen_version: str
    suite_version: str
    case_id: str
    candidate: str
    region: str
    mind_mse_full: float
    mind_mse_interior: float
    intensity_mse_full: float
    intensity_mse_interior: float

    def as_csv_row(self) -> dict[str, str | float]:
        return cast(dict[str, str | float], asdict(self))


@dataclass(frozen=True)
class IdentifiabilityDecision:
    """Case-level interpretation using only predeclared ratios."""

    screen_version: str
    suite_version: str
    case_id: str
    warp_direction_pass: bool
    mind_slip_identifiable: bool
    intensity_slip_identifiable: bool
    optimizer_reduced_data_term: bool
    truth_to_inverse_mind_ratio_max: float
    truth_to_inverse_intensity_ratio_max: float
    truth_to_zero_mind_ratio_max: float
    truth_to_zero_intensity_ratio_max: float
    truth_to_glued_mind_ratio_max: float
    truth_to_glued_intensity_ratio_max: float
    estimated_to_zero_mind_ratio_max: float
    balanced_gate_reasons: tuple[str, ...]
    classification: str

    def as_csv_row(self) -> dict[str, str | float | bool]:
        values = asdict(self)
        values["balanced_gate_reasons"] = ";".join(self.balanced_gate_reasons)
        return cast(dict[str, str | float | bool], values)


@dataclass(frozen=True)
class IdentifiabilityRun:
    """All rows, decisions and external runtime payloads for one screen."""

    records: tuple[DataTermRecord, ...]
    decisions: tuple[IdentifiabilityDecision, ...]
    runner_payloads: dict[str, dict[str, Any]]


def _strict_object(
    value: object,
    *,
    expected: set[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} must contain exactly {sorted(expected)}")
    return cast(dict[str, Any], value)


def load_identifiability_screen(path: Path) -> IdentifiabilityScreen:
    """Load the frozen screen without loading its suite or field artifacts."""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    top = _strict_object(
        payload,
        expected={
            "screen_version",
            "suite_path",
            "development_batch_manifest_path",
            "runner_version",
            "device",
            "dtype",
            "descriptor",
            "evaluation_masks",
            "candidates",
            "metrics",
            "decision_thresholds",
            "decision_rule",
            "challenge_policy",
        },
        label="identifiability screen",
    )
    descriptor = _strict_object(
        top["descriptor"],
        expected={"kind", "radius", "dilation", "source"},
        label="descriptor",
    )
    masks = _strict_object(
        top["evaluation_masks"],
        expected={"primary", "diagnostic", "body_region"},
        label="evaluation_masks",
    )
    if masks != {
        "primary": "full_fixed_region",
        "diagnostic": "fixed_region_eroded_2_voxels",
        "body_region": "fixed_body_minus_fixed_lung",
    }:
        raise ValueError("unexpected evaluation mask policy")
    candidates = top["candidates"]
    metrics = top["metrics"]
    if not isinstance(candidates, list) or tuple(candidates) != CANDIDATE_NAMES:
        raise ValueError("unexpected identifiability candidates")
    if not isinstance(metrics, list) or tuple(metrics) != METRIC_NAMES:
        raise ValueError("unexpected identifiability metrics")
    thresholds_payload = _strict_object(
        top["decision_thresholds"],
        expected={
            "truth_to_wrong_direction_ratio_max",
            "truth_to_zero_ratio_max",
            "truth_to_glued_ratio_max",
            "estimated_to_zero_ratio_max",
        },
        label="decision_thresholds",
    )
    thresholds = IdentifiabilityThresholds(
        **{key: float(value) for key, value in thresholds_payload.items()}
    )
    if any(
        value <= 0 or value >= 1
        for value in asdict(thresholds).values()
    ):
        raise ValueError("identifiability ratios must lie in (0, 1)")
    decision = top["decision_rule"]
    if not isinstance(decision, dict) or set(decision) != {
        "warp_direction_pass",
        "mind_slip_identifiable",
        "intensity_slip_identifiable",
        "optimizer_reduced_data_term",
        "descriptor_failure",
        "optimizer_or_parameterization_failure",
        "generator_observability_failure",
        "direction_failure",
    }:
        raise ValueError("unexpected decision rule")
    challenge = top["challenge_policy"]
    if not isinstance(challenge, dict) or challenge.get("loaded") is not False:
        raise ValueError("identifiability screen must keep challenge closed")
    if top["device"] != "cuda" or top["dtype"] != "float32":
        raise ValueError("identifiability screen requires CUDA float32")
    if descriptor["kind"] != "MINDSSC":
        raise ValueError("identifiability screen requires MINDSSC")
    return IdentifiabilityScreen(
        screen_version=str(top["screen_version"]),
        suite_path=str(top["suite_path"]),
        development_batch_manifest_path=str(
            top["development_batch_manifest_path"]
        ),
        runner_version=str(top["runner_version"]),
        device=str(top["device"]),
        dtype=str(top["dtype"]),
        descriptor=cast(dict[str, str | int], descriptor),
        candidates=tuple(candidates),
        metrics=tuple(metrics),
        thresholds=thresholds,
    )


def build_identifiability_candidates(
    case: ContactSVFSuiteCase,
    generated: ContactSVFPhantomPair,
    estimated_lung_mm: VectorArray,
    estimated_body_mm: VectorArray,
) -> dict[str, tuple[VectorArray, VectorArray]]:
    """Build only the six candidates frozen before the screen."""
    expected_shape = generated.pair.fixed_image.shape + (3,)
    if (
        estimated_lung_mm.shape != expected_shape
        or estimated_body_mm.shape != expected_shape
    ):
        raise ValueError("estimated fields do not match the generated case grid")
    lung_velocity = generated.lung_velocity_mm
    body_velocity = generated.body_velocity_mm
    mean_velocity = np.ascontiguousarray(
        (0.5 * (lung_velocity + body_velocity)).astype(np.float32)
    )
    denominator = case.params.body_motion_rad - case.params.lung_motion_rad
    if abs(denominator) <= 1e-12:
        raise ValueError("regional motions cannot define a scale-only velocity")
    scale_velocity = np.ascontiguousarray(
        (
            (
                case.params.body_motion_rad * lung_velocity
                - case.params.lung_motion_rad * body_velocity
            )
            / denominator
        ).astype(np.float32)
    )
    glued = exponentiate_piecewise_svf(
        mean_velocity,
        mean_velocity,
        case.params.spacing_mm,
        squaring_steps=case.params.squaring_steps,
        outside_mode=case.params.outside_mode,
    )
    scale_only = exponentiate_piecewise_svf(
        scale_velocity,
        scale_velocity,
        case.params.spacing_mm,
        squaring_steps=case.params.squaring_steps,
        outside_mode=case.params.outside_mode,
    )
    zeros = np.zeros(expected_shape, dtype=np.float32)
    pair = generated.pair
    return {
        "truth": (
            pair.phantom.lung_displacement_mm,
            pair.phantom.body_displacement_mm,
        ),
        "estimated_balanced": (
            np.ascontiguousarray(estimated_lung_mm.astype(np.float32)),
            np.ascontiguousarray(estimated_body_mm.astype(np.float32)),
        ),
        "glued_mean_velocity": (
            glued.lung_displacement_mm,
            glued.body_displacement_mm,
        ),
        "common_scale_only": (
            scale_only.lung_displacement_mm,
            scale_only.body_displacement_mm,
        ),
        "zero": (zeros, zeros.copy()),
        "inverse_truth": (
            pair.lung_moving_to_fixed_mm,
            pair.body_moving_to_fixed_mm,
        ),
    }


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= np.finfo(np.float64).eps:
        return 0.0 if numerator <= np.finfo(np.float64).eps else float("inf")
    return numerator / denominator


def _max_ratio(
    lookup: dict[tuple[str, str], DataTermRecord],
    numerator: str,
    denominator: str,
    metric: str,
) -> float:
    return max(
        _ratio(
            float(getattr(lookup[(numerator, region)], metric)),
            float(getattr(lookup[(denominator, region)], metric)),
        )
        for region in ("lung", "body")
    )


def decide_identifiability(
    records: tuple[DataTermRecord, ...],
    *,
    thresholds: IdentifiabilityThresholds,
    balanced_gate_reasons: tuple[str, ...],
) -> IdentifiabilityDecision:
    """Apply the frozen ratio decision tree to one case."""
    if len(records) != len(CANDIDATE_NAMES) * 2:
        raise ValueError("one case must contain six candidates and two regions")
    lookup = {(record.candidate, record.region): record for record in records}
    if set(lookup) != {
        (candidate, region)
        for candidate in CANDIDATE_NAMES
        for region in ("lung", "body")
    }:
        raise ValueError("candidate/region records are incomplete")
    first = records[0]
    inverse_mind = _max_ratio(lookup, "truth", "inverse_truth", "mind_mse_full")
    inverse_intensity = _max_ratio(
        lookup,
        "truth",
        "inverse_truth",
        "intensity_mse_full",
    )
    zero_mind = _max_ratio(lookup, "truth", "zero", "mind_mse_full")
    zero_intensity = _max_ratio(
        lookup,
        "truth",
        "zero",
        "intensity_mse_full",
    )
    glued_mind = _max_ratio(
        lookup,
        "truth",
        "glued_mean_velocity",
        "mind_mse_full",
    )
    glued_intensity = _max_ratio(
        lookup,
        "truth",
        "glued_mean_velocity",
        "intensity_mse_full",
    )
    estimated_zero_mind = _max_ratio(
        lookup,
        "estimated_balanced",
        "zero",
        "mind_mse_full",
    )
    direction_pass = (
        inverse_mind <= thresholds.truth_to_wrong_direction_ratio_max
        and inverse_intensity <= thresholds.truth_to_wrong_direction_ratio_max
        and zero_mind <= thresholds.truth_to_zero_ratio_max
        and zero_intensity <= thresholds.truth_to_zero_ratio_max
    )
    mind_identifiable = glued_mind <= thresholds.truth_to_glued_ratio_max
    intensity_identifiable = (
        glued_intensity <= thresholds.truth_to_glued_ratio_max
    )
    optimizer_reduced = (
        estimated_zero_mind <= thresholds.estimated_to_zero_ratio_max
    )
    if not direction_pass:
        classification = "direction_failure"
    elif not intensity_identifiable:
        classification = "generator_observability_failure"
    elif not mind_identifiable:
        classification = "descriptor_failure"
    elif "tangential_slip" in balanced_gate_reasons:
        classification = "optimizer_or_parameterization_failure"
    else:
        classification = "no_failure_detected"
    return IdentifiabilityDecision(
        screen_version=first.screen_version,
        suite_version=first.suite_version,
        case_id=first.case_id,
        warp_direction_pass=direction_pass,
        mind_slip_identifiable=mind_identifiable,
        intensity_slip_identifiable=intensity_identifiable,
        optimizer_reduced_data_term=optimizer_reduced,
        truth_to_inverse_mind_ratio_max=inverse_mind,
        truth_to_inverse_intensity_ratio_max=inverse_intensity,
        truth_to_zero_mind_ratio_max=zero_mind,
        truth_to_zero_intensity_ratio_max=zero_intensity,
        truth_to_glued_mind_ratio_max=glued_mind,
        truth_to_glued_intensity_ratio_max=glued_intensity,
        estimated_to_zero_mind_ratio_max=estimated_zero_mind,
        balanced_gate_reasons=balanced_gate_reasons,
        classification=classification,
    )


def _safe_repo_path(repo_root: Path, relative: str) -> Path:
    candidate = (repo_root / relative).resolve()
    root = repo_root.resolve()
    if root not in candidate.parents:
        raise ValueError("screen paths must stay inside the repository")
    return candidate


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


def run_identifiability_screen(
    screen: IdentifiabilityScreen,
    suite: ContactSVFSuite,
    *,
    registration_python: Path,
    repo_root: Path,
    temporary_root: Path,
) -> IdentifiabilityRun:
    """Run development-only exact data terms without invoking an optimizer."""
    batch_manifest_path = _safe_repo_path(
        repo_root,
        screen.development_batch_manifest_path,
    )
    batch_root = batch_manifest_path.parent
    batch_manifest = json.loads(batch_manifest_path.read_text(encoding="utf-8"))
    if (
        batch_manifest.get("run_count") != 9
        or batch_manifest.get("challenge", {}).get("loaded") is not False
    ):
        raise ValueError("development batch manifest is incomplete or opened challenge")
    runner_path = repo_root / "tools/evaluate_joint_svf_data_term.py"
    if not registration_python.is_file() or not runner_path.is_file():
        raise FileNotFoundError("identifiability runner environment is incomplete")
    all_records: list[DataTermRecord] = []
    decisions: list[IdentifiabilityDecision] = []
    runner_payloads: dict[str, dict[str, Any]] = {}
    temporary_root.mkdir(parents=True, exist_ok=True)
    for case in suite.cases:
        stem = f"balanced__{case.case_id}"
        field_path = batch_root / f"{stem}.npz"
        record_path = batch_root / f"{stem}.json"
        expected_field_hash = batch_manifest["field_sha256"].get(field_path.name)
        expected_record_hash = batch_manifest["record_sha256"].get(record_path.name)
        if (
            expected_field_hash != _sha256(field_path)
            or expected_record_hash != _sha256(record_path)
        ):
            raise ValueError(f"{case.case_id}: development artifact checksum mismatch")
        balanced_record = json.loads(record_path.read_text(encoding="utf-8"))
        with np.load(field_path, allow_pickle=False) as artifact:
            estimated_lung = np.asarray(
                artifact["lung_displacement_mm"],
                dtype=np.float32,
            )
            estimated_body = np.asarray(
                artifact["body_displacement_mm"],
                dtype=np.float32,
            )
            stored_direction = str(artifact["transform_direction"].item())
        generated = make_contact_svf_phantom_pair(case.params)
        if stored_direction != generated.pair.transform_direction:
            raise ValueError(f"{case.case_id}: stored transform direction drifted")
        candidates = build_identifiability_candidates(
            case,
            generated,
            estimated_lung,
            estimated_body,
        )
        with tempfile.TemporaryDirectory(
            prefix=f"identifiability-{case.case_id}-",
            dir=temporary_root,
        ) as temporary_name:
            temporary = Path(temporary_name)
            input_path = temporary / "input.npz"
            params_path = temporary / "params.json"
            output_path = temporary / "output.json"
            np.savez_compressed(
                input_path,
                fixed_image=generated.pair.fixed_image,
                moving_image=generated.pair.moving_image,
                fixed_lung_mask=generated.pair.fixed_lung_mask,
                fixed_body_mask=generated.pair.fixed_body_mask,
                spacing_mm=np.asarray(generated.pair.spacing_mm),
                candidate_names=np.asarray(screen.candidates),
                lung_displacement_mm=np.stack(
                    [candidates[name][0] for name in screen.candidates],
                ),
                body_displacement_mm=np.stack(
                    [candidates[name][1] for name in screen.candidates],
                ),
            )
            params_path.write_text(
                json.dumps(
                    {
                        "runner_version": screen.runner_version,
                        "descriptor": screen.descriptor,
                        "erosion_iterations": 2,
                    },
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    str(registration_python),
                    str(runner_path),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--params",
                    str(params_path),
                ],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        runner_payloads[case.case_id] = payload
        case_records = tuple(
            DataTermRecord(
                screen_version=screen.screen_version,
                suite_version=suite.suite_version,
                case_id=case.case_id,
                candidate=str(item["candidate"]),
                region=str(item["region"]),
                mind_mse_full=float(item["mind_mse_full"]),
                mind_mse_interior=float(item["mind_mse_interior"]),
                intensity_mse_full=float(item["intensity_mse_full"]),
                intensity_mse_interior=float(item["intensity_mse_interior"]),
            )
            for item in payload["records"]
        )
        all_records.extend(case_records)
        decisions.append(
            decide_identifiability(
                case_records,
                thresholds=screen.thresholds,
                balanced_gate_reasons=tuple(balanced_record["gate_reasons"]),
            )
        )
    return IdentifiabilityRun(
        records=tuple(all_records),
        decisions=tuple(decisions),
        runner_payloads=runner_payloads,
    )


def write_identifiability_screen(
    run: IdentifiabilityRun,
    output_dir: Path,
    *,
    screen: IdentifiabilityScreen,
    screen_path: Path,
    suite_path: Path,
    repo_root: Path,
) -> Path:
    """Write checksummed rows, decisions and external runtime provenance."""
    if not run.records or not run.decisions:
        raise ValueError("cannot write an empty identifiability screen")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    rows = [record.as_csv_row() for record in run.records]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    decision_path = output_dir / "decisions.csv"
    decisions = [decision.as_csv_row() for decision in run.decisions]
    with decision_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decisions[0]))
        writer.writeheader()
        writer.writerows(decisions)
    runner_hashes: dict[str, str] = {}
    for case_id, payload in run.runner_payloads.items():
        path = output_dir / f"{case_id}__runner.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        runner_hashes[path.name] = _sha256(path)
    classifications: dict[str, int] = {}
    for decision in run.decisions:
        classifications[decision.classification] = (
            classifications.get(decision.classification, 0) + 1
        )
    batch_manifest_path = _safe_repo_path(
        repo_root,
        screen.development_batch_manifest_path,
    )
    runner_path = repo_root / "tools/evaluate_joint_svf_data_term.py"
    manifest = {
        "artifact_type": "piecewise_svf_j12_identifiability_screen",
        "usage": "synthetic_development_diagnostic_ONLY",
        "screen_version": screen.screen_version,
        "suite_version": run.records[0].suite_version,
        "code_version": _code_version(repo_root),
        "screen_config": {
            "path": screen_path.name,
            "sha256": _sha256(screen_path),
        },
        "suite_config": {
            "path": suite_path.name,
            "sha256": _sha256(suite_path),
        },
        "development_batch_manifest": {
            "path": screen.development_batch_manifest_path,
            "sha256": _sha256(batch_manifest_path),
        },
        "external_runner": {
            "path": runner_path.name,
            "sha256": _sha256(runner_path),
        },
        "challenge": {"loaded": False},
        "record_count": len(run.records),
        "case_count": len(run.decisions),
        "classification_counts": classifications,
        "summary_sha256": _sha256(summary_path),
        "decisions_sha256": _sha256(decision_path),
        "runner_sha256": runner_hashes,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return manifest_path


__all__ = [
    "CANDIDATE_NAMES",
    "DataTermRecord",
    "IdentifiabilityDecision",
    "IdentifiabilityRun",
    "IdentifiabilityScreen",
    "IdentifiabilityThresholds",
    "build_identifiability_candidates",
    "decide_identifiability",
    "load_identifiability_screen",
    "run_identifiability_screen",
    "write_identifiability_screen",
]
