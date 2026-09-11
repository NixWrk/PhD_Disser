"""Restartable direct search for the full TRKG4 inverse FEM model.

The MATLAB/EIDORS process accumulates sparse-memory fragmentation during a long
run on the v5 mesh.  This driver keeps the computational workaround, but treats
resume and failure handling as a strict contract:

* model, data and code fingerprints must match before history is accepted;
* only an explicitly known electrode-geometry error becomes invalid_geometry;
* every other MATLAB failure stops the run without a physical penalty;
* finalisation requires the complete, freshly written diagnostic artifact set.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
LOWER = [2.0, 10.0, -120.0, -120.0, -20.0]
UPPER = [10.0, 32.0, 120.0, 120.0, 20.0]
INITIAL_STEP = [0.75, 3.0, 10.0, 10.0, 5.0]
MINIMUM_STEP = [0.05, 0.25, 0.5, 0.5, 0.5]
PARAMETER_NAMES = [
    "rho_soft_ohm_m",
    "rho_lungs_ohm_m",
    "centre_u_mm",
    "centre_v_mm",
    "phi_deg",
]
HISTORY_FIELDS = [
    "evaluation",
    *PARAMETER_NAMES,
    "objective",
    "fem_slope_ohm_per_mm",
    "h_centre_mm",
    "status",
    "failure_code",
    "engine_version",
    "engine_fingerprint",
]
NUMERIC_HISTORY_FIELDS = {
    "evaluation",
    *PARAMETER_NAMES,
    "objective",
    "fem_slope_ohm_per_mm",
    "h_centre_mm",
}
DRIVER_CONTRACT_VERSION = "restartable-inverse-driver-v2"
INVALID_GEOMETRY_OBJECTIVE = 1.0e12
VALID_STATUSES = {"evaluated", "invalid_geometry"}

# Exact MATLAB identifiers that are allowed to become invalid_geometry.
# Prefix matching is deliberately forbidden: emptyBoundary, input errors,
# noSeed under an unknown builder, and programming errors are technical failures.
KNOWN_GEOMETRY_ERROR_CODES = frozenset({
    "trkg4:electrodeDiagnostics",
    "trkg4:patchOverlap",
    "electrode_faces_by_area:noSeed",
    "electrode_faces_by_area:patchTooSmall",
    "electrode_faces_by_area_legacy:patchTooSmall",
    "electrode_faces_by_area_legacy_fast:patchTooSmall",
})


class RestartableSearchError(RuntimeError):
    """Base class for errors that must stop the restartable search."""


class ResumeIdentityError(RestartableSearchError):
    """A checkpoint does not belong to the requested model and run."""


class EvaluationContractError(RestartableSearchError):
    """MATLAB returned an incomplete or ambiguous evaluation contract."""


class TechnicalEvaluationError(RestartableSearchError):
    """MATLAB failed for a reason that is not an expected geometry outcome."""


class NoValidEvaluationError(RestartableSearchError):
    """No finite FEM evaluation is available for finalisation."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("legacy_fast", "disjoint"), required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--data-selection", default="manifest")
    parser.add_argument(
        "--initial-points",
        type=Path,
        default=None,
        help="Headerless CSV with N rows of admissible 5-parameter starts.",
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def read_first_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return next(csv.DictReader(stream))


def point_from_row(row: dict[str, str]) -> list[float]:
    return [float(row[name]) for name in PARAMETER_NAMES]


def _resolve_initial_points_path(
    value: Path | None, *, root: Path = ROOT
) -> Path | None:
    if value is None:
        return None
    candidate = value if value.is_absolute() else root / value
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Initial-points CSV does not exist: {candidate}")
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(
            "The --initial-points CSV must be inside the repository ROOT."
        ) from exc
    return candidate


def _read_initial_points_csv(path: Path) -> list[list[float]]:
    points: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, row in enumerate(csv.reader(stream), start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != len(PARAMETER_NAMES):
                raise ValueError(
                    f"{path}:{line_number} must contain exactly five numeric columns."
                )
            try:
                point = [float(cell) for cell in row]
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number} contains a non-numeric initial point."
                ) from exc
            if not all(math.isfinite(value) for value in point):
                raise ValueError(
                    f"{path}:{line_number} contains a non-finite initial point."
                )
            if any(
                value < lower or value > upper
                for value, lower, upper in zip(point, LOWER, UPPER)
            ):
                raise ValueError(
                    f"{path}:{line_number} contains a point outside driver bounds."
                )
            points.append(point)
    if not points:
        raise ValueError(f"Initial-points CSV is empty: {path}")
    return unique_points(points)


def initial_points(seed_file: Path | None = None) -> list[list[float]]:
    if seed_file is not None:
        return _read_initial_points_csv(seed_file)
    old = read_first_row(
        OUT
        / "repro_check_20260901_before"
        / "nik_trkg4_inverse_inhale_fit_summary_v5_1mm_local2mm.csv"
    )
    recovered = read_first_row(
        OUT / "nik_trkg4_inverse_inhale_fit_summary_ablation_opt_oldstart_manifest90.csv"
    )
    current = read_first_row(
        OUT / "nik_trkg4_inverse_inhale_fit_summary_v5_1mm_local2mm.csv"
    )
    with (
        OUT / "nik_trkg4_fast_cem_refine_v5_1mm_local2mm.csv"
    ).open("r", encoding="utf-8-sig", newline="") as stream:
        cem_rows = list(csv.DictReader(stream))
    cem_rows.sort(key=lambda row: float(row["fast_CEM_objective"]))
    current_point = point_from_row(current)
    old_point = point_from_row(old)
    points = [old_point, point_from_row(recovered), current_point]
    pose_keys: set[tuple[float, ...]] = set()
    for cem_row in cem_rows:
        pose = [
            float(cem_row["centre_u_mm"]),
            float(cem_row["centre_v_mm"]),
            float(cem_row["phi_deg"]),
        ]
        key = point_key(pose)
        if key in pose_keys:
            continue
        pose_keys.add(key)
        points.append([current_point[0], current_point[1], *pose])
        points.append([old_point[0], old_point[1], *pose])
        if len(pose_keys) == 5:
            break
    return unique_points(points)


def point_key(point: Iterable[float]) -> tuple[float, ...]:
    return tuple(round(float(value), 12) for value in point)


def unique_points(points: Iterable[Iterable[float]]) -> list[list[float]]:
    result: list[list[float]] = []
    seen: set[tuple[float, ...]] = set()
    for raw in points:
        point = [float(value) for value in raw]
        key = point_key(point)
        if key not in seen:
            seen.add(key)
            result.append(point)
    return result


def matlab_string(path_or_text: str | Path) -> str:
    return str(path_or_text).replace("\\", "/").replace("'", "''")


def write_points(path: Path, points: list[list[float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(points)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_records(root: Path, paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted((Path(item) for item in paths), key=lambda item: item.as_posix()):
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = path.as_posix()
        if relative in seen:
            continue
        seen.add(relative)
        if path.is_file():
            records.append({
                "path": relative,
                "exists": True,
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            })
        else:
            records.append({"path": relative, "exists": False})
    return records


def _group_fingerprint(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(_canonical_json(records).encode("utf-8")).hexdigest()


def _identity_file_groups(root: Path) -> dict[str, list[Path]]:
    src_dir = root / "src"
    vendor_dir = root / "vendor_stl_eidors"
    tools_dir = root / "tools"
    code_paths = list(src_dir.rglob("*.m")) if src_dir.is_dir() else []
    if vendor_dir.is_dir():
        code_paths.extend(vendor_dir.rglob("*.m"))
    if tools_dir.is_dir():
        code_paths.extend(tools_dir.rglob("*.m"))
    code_paths.append(tools_dir / "run_restartable_inverse_search.py")
    return {
        "model": [
            root / "output" / "nik_trkg4_right_rib_050mm_full_v5_1mm_local2mm.mat",
            root / "data" / "nik" / "stl" / "lungs_solid_v3_volume_fill.stl",
            root / "data" / "nik" / "electrodes" / "electrodes_4_right_rib_140mm_xyz_mm.csv",
        ],
        "data": [
            root / "data" / "nik" / "experimental" / "nik_breath_hold_levels.csv",
        ],
        "code": code_paths,
    }


def _identity_without_fingerprint(identity: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(identity)
    payload.pop("fingerprint", None)
    return payload


def _complete_identity(identity: dict[str, Any]) -> dict[str, Any]:
    completed = copy.deepcopy(identity)
    completed["fingerprint"] = hashlib.sha256(
        _canonical_json(_identity_without_fingerprint(completed)).encode("utf-8")
    ).hexdigest()
    return completed


def _seed_identity(root: Path, seed_file: Path | None) -> dict[str, Any]:
    if seed_file is None:
        return {"path": None, "sha256": None}
    path = Path(seed_file).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Initial-points CSV does not exist: {path}")
    try:
        relative = path.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            "The --initial-points CSV must be inside the repository ROOT."
        ) from exc
    return {"path": relative, "sha256": _sha256_file(path)}


def build_run_identity(
    *,
    tag: str,
    mode: str,
    data_selection: str,
    root: Path = ROOT,
    engine_version: str | None = None,
    engine_fingerprint: str | None = None,
    initial_points_path: Path | None = None,
) -> dict[str, Any]:
    groups = _identity_file_groups(root)
    files = {name: _file_records(root, paths) for name, paths in groups.items()}
    fingerprints = {
        name: _group_fingerprint(records) for name, records in files.items()
    }
    return _complete_identity({
        "schema": DRIVER_CONTRACT_VERSION,
        "version": DRIVER_CONTRACT_VERSION,
        "run": {
            "tag": tag,
            "patch_builder_mode": mode,
            "data_selection": data_selection,
        },
        "model": {"fingerprint": fingerprints["model"], "files": files["model"]},
        "data": {"fingerprint": fingerprints["data"], "files": files["data"]},
        "code": {"fingerprint": fingerprints["code"], "files": files["code"]},
        "seed": _seed_identity(root, initial_points_path),
        "engine": {
            "version": engine_version,
            "fingerprint": engine_fingerprint,
        },
    })


def _identity_path(tag: str, out: Path | None = None) -> Path:
    return (OUT if out is None else out) / f".restartable_{tag}_identity.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResumeIdentityError(f"Cannot read JSON identity file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResumeIdentityError(f"Identity file {path} must contain a JSON object.")
    return value


def _set_engine_identity(
    identity: dict[str, Any], version: str | None, fingerprint: str | None
) -> dict[str, Any]:
    if not version or not fingerprint:
        raise EvaluationContractError(
            "MATLAB result must provide non-empty version and fingerprint."
        )
    updated = copy.deepcopy(identity)
    engine = updated.setdefault("engine", {})
    if engine.get("version") not in (None, version) or engine.get(
        "fingerprint"
    ) not in (None, fingerprint):
        raise ResumeIdentityError(
            "MATLAB version/fingerprint changed during one restartable run."
        )
    engine["version"] = str(version)
    engine["fingerprint"] = str(fingerprint)
    return _complete_identity(updated)


def _validate_identity(
    saved: dict[str, Any],
    *,
    tag: str,
    mode: str,
    data_selection: str,
    root: Path = ROOT,
    initial_points_path: Path | None = None,
) -> dict[str, Any]:
    engine = saved.get("engine")
    if not isinstance(engine, dict):
        raise ResumeIdentityError("Checkpoint identity has no engine object.")
    expected = build_run_identity(
        tag=tag,
        mode=mode,
        data_selection=data_selection,
        root=root,
        engine_version=engine.get("version"),
        engine_fingerprint=engine.get("fingerprint"),
        initial_points_path=initial_points_path,
    )
    if saved != expected:
        raise ResumeIdentityError(
            "Checkpoint identity does not match the current model, data, code, "
            "run tag, or MATLAB engine fingerprint. Start a new run with a new "
            "tag or use --no-resume."
        )
    return expected


def _contract_path(tag: str, out: Path | None = None) -> Path:
    out = OUT if out is None else out
    return out / f"nik_trkg4_inverse_inhale_contract_{tag}.json"


def _diagnostics_path(tag: str, out: Path | None = None) -> Path:
    out = OUT if out is None else out
    return out / f"nik_trkg4_inverse_inhale_jacobian_diagnostics_{tag}.json"


def _validate_record_list(
    contract: dict[str, Any], field: str, *, allow_empty_path: bool
) -> list[dict[str, Any]]:
    records = contract.get(field)
    if not isinstance(records, list):
        raise EvaluationContractError(
            f"Final MATLAB contract field {field} must be a list of records."
        )
    required_keys = {"name", "path", "sha256"}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != required_keys:
            raise EvaluationContractError(
                f"Final MATLAB contract {field}[{index}] must have exactly "
                "name, path, and sha256."
            )
        if not isinstance(record["name"], str) or not record["name"]:
            raise EvaluationContractError(
                f"Final MATLAB contract {field}[{index}] has no record name."
            )
        if not isinstance(record["path"], str):
            raise EvaluationContractError(
                f"Final MATLAB contract {field}[{index}] has a non-string path."
            )
        if not allow_empty_path and not record["path"]:
            raise EvaluationContractError(
                f"Final MATLAB contract {field}[{index}] has an empty path."
            )
        if record["path"] and Path(record["path"]).is_absolute():
            raise EvaluationContractError(
                f"Final MATLAB contract {field}[{index}] path must be relative to ROOT."
            )
        if record["path"] and Path(record["path"]).name != record["name"]:
            raise EvaluationContractError(
                f"Final MATLAB contract {field}[{index}] name/path mismatch."
            )
        if not isinstance(record["sha256"], str) or not record["sha256"]:
            raise EvaluationContractError(
                f"Final MATLAB contract {field}[{index}] has no SHA256."
            )
    return records


def _validate_contract_identity(
    contract: dict[str, Any], *, tag: str, mode: str, data_selection: str
) -> tuple[str, str]:
    version = contract.get("version")
    fingerprint = contract.get("fingerprint")
    if not isinstance(version, str) or not version:
        raise EvaluationContractError("Final MATLAB contract has no version.")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise EvaluationContractError("Final MATLAB contract has no fingerprint.")
    if contract.get("result_tag") != tag:
        raise EvaluationContractError(
            f"Final MATLAB contract result_tag {contract.get('result_tag')!r} "
            f"does not match {tag!r}."
        )
    if contract.get("patch_builder_mode") != mode:
        raise EvaluationContractError(
            f"Final MATLAB contract patch_builder_mode "
            f"{contract.get('patch_builder_mode')!r} does not match {mode!r}."
        )
    if contract.get("data_selection") != data_selection:
        raise EvaluationContractError(
            f"Final MATLAB contract data_selection "
            f"{contract.get('data_selection')!r} does not match {data_selection!r}."
        )
    if contract.get("compute_jacobian") is not True:
        raise EvaluationContractError(
            "Final MATLAB contract must have compute_jacobian=true."
        )
    _validate_record_list(contract, "inputs", allow_empty_path=True)
    _validate_record_list(contract, "artifacts", allow_empty_path=False)
    return version, fingerprint


def _is_known_geometry_code(code: str) -> bool:
    return code.strip() in KNOWN_GEOMETRY_ERROR_CODES


def _geometry_code_from_log(text: str) -> str | None:
    for code in sorted(KNOWN_GEOMETRY_ERROR_CODES, key=len, reverse=True):
        pattern = rf"(?<![A-Za-z0-9_:]){re.escape(code)}(?![A-Za-z0-9_:])"
        if re.search(pattern, text):
            return code
    return None


def _validate_history_rows(
    history: list[dict[str, Any]], identity: dict[str, Any]
) -> None:
    engine = identity["engine"]
    expected_version = engine.get("version")
    expected_fingerprint = engine.get("fingerprint")
    for row in history:
        status = row.get("status")
        if status not in VALID_STATUSES:
            raise ResumeIdentityError(f"Unknown history row status: {status!r}.")
        row_version = row.get("engine_version") or None
        row_fingerprint = row.get("engine_fingerprint") or None
        if row_version or row_fingerprint:
            if row_version != expected_version or row_fingerprint != expected_fingerprint:
                raise ResumeIdentityError(
                    "A history row has a different MATLAB version/fingerprint."
                )
        if status == "evaluated" and not math.isfinite(float(row["objective"])):
            raise ResumeIdentityError("An evaluated history row has a non-finite objective.")
        if status == "invalid_geometry" and not _is_known_geometry_code(
            str(row.get("failure_code", ""))
        ):
            raise ResumeIdentityError(
                "An invalid_geometry history row lacks an explicit known geometry code."
            )


def load_resume_state(
    history_file: Path,
    summary_file: Path,
    identity_file: Path,
    *,
    tag: str,
    mode: str,
    data_selection: str,
    root: Path = ROOT,
    initial_points_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[float]]:
    """Validate identity before returning a checkpoint to the search."""

    if _search_state_path(tag, history_file.parent).is_file():
        state = _restore_search_state(
            history_file, summary_file, identity_file, tag=tag, mode=mode,
            data_selection=data_selection, root=root, initial_points_path=initial_points_path,
        )
        return state["history"], state["identity"], state["step"].copy()

    if not history_file.is_file():
        identity = build_run_identity(
            tag=tag,
            mode=mode,
            data_selection=data_selection,
            root=root,
            initial_points_path=initial_points_path,
        )
        return [], identity, INITIAL_STEP.copy()
    if not identity_file.is_file():
        raise ResumeIdentityError(
            f"History exists without identity sidecar: {identity_file}. Refusing resume."
        )
    sidecar_identity = _read_json(identity_file)
    saved_identity = sidecar_identity
    summary: dict[str, Any] = {}
    if summary_file.is_file():
        summary = _read_json(summary_file)
        summary_identity = summary.get("identity")
        if not isinstance(summary_identity, dict) or summary_identity != sidecar_identity:
            raise ResumeIdentityError("Summary and identity sidecar do not match.")
        saved_identity = summary_identity
    identity = _validate_identity(
        saved_identity,
        tag=tag,
        mode=mode,
        data_selection=data_selection,
        root=root,
        initial_points_path=initial_points_path,
    )
    contract_file = _contract_path(tag)
    if contract_file.is_file():
        contract = _read_json(contract_file)
        contract_version, contract_fingerprint = _validate_contract_identity(
            contract, tag=tag, mode=mode, data_selection=data_selection
        )
        engine = identity["engine"]
        if engine.get("version") != contract_version or engine.get(
            "fingerprint"
        ) != contract_fingerprint:
            raise ResumeIdentityError(
                "Final MATLAB contract fingerprint does not match checkpoint identity."
            )
        if summary.get("finalization", {}).get("status") == "complete":
            _validate_driver_artifacts(contract, root=root)
    history = load_history(history_file)
    _validate_history_rows(history, identity)
    if contract_file.is_file() and summary.get("finalization", {}).get("status") == "complete":
        _validate_driver_artifacts(contract, root=root)
        _validate_contract_parameters(contract, best_evaluated(history))
    saved_step = summary.get("final_step")
    if not isinstance(saved_step, list) or len(saved_step) != 5:
        saved_step = INITIAL_STEP.copy()
    return history, identity, [float(value) for value in saved_step]


def _engine_fields(row: dict[str, str]) -> tuple[str, str]:
    # The MATLAB batch contract uses the exact result.version and
    # result.fingerprint field names.  History uses engine_* only after parsing.
    return (row.get("version") or "").strip(), (row.get("fingerprint") or "").strip()


def _make_record(
    *,
    evaluation: int,
    point: list[float],
    objective: float,
    slope: float,
    h_centre: float,
    status: str,
    failure_code: str = "",
    engine_version: str = "",
    engine_fingerprint: str = "",
) -> dict[str, Any]:
    return {
        "evaluation": evaluation,
        **dict(zip(PARAMETER_NAMES, point)),
        "objective": objective,
        "fem_slope_ohm_per_mm": slope,
        "h_centre_mm": h_centre,
        "status": status,
        "failure_code": failure_code,
        "engine_version": engine_version,
        "engine_fingerprint": engine_fingerprint,
    }


def _parse_batch_rows(
    path: Path,
    points: list[list[float]],
    *,
    first_evaluation: int,
) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(points):
        raise EvaluationContractError(
            f"MATLAB returned {len(rows)} rows for {len(points)} requested points."
        )
    records: list[dict[str, Any]] = []
    for offset, (point, row) in enumerate(zip(points, rows)):
        try:
            returned_point = [float(row[name]) for name in PARAMETER_NAMES]
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationContractError(
                "MATLAB result is missing a numeric parameter column."
            ) from exc
        if point_key(returned_point) != point_key(point):
            raise EvaluationContractError(
                "MATLAB result row does not correspond to the requested point."
            )
        status = (row.get("status") or "").strip()
        if status not in VALID_STATUSES:
            raise EvaluationContractError(
                f"MATLAB result has unknown or missing status: {status!r}."
            )
        version, fingerprint = _engine_fields(row)
        if not version or not fingerprint:
            raise EvaluationContractError(
                "Every MATLAB result row must contain version and fingerprint."
            )
        failure_code = (row.get("failure_code") or "").strip()
        try:
            slope = float(row.get("fem_slope_ohm_per_mm", "nan"))
            h_centre = float(row.get("h_centre_mm", "nan"))
        except (TypeError, ValueError) as exc:
            raise EvaluationContractError(
                "MATLAB result contains non-numeric diagnostics."
            ) from exc
        if status == "invalid_geometry":
            if not _is_known_geometry_code(failure_code):
                raise EvaluationContractError(
                    "invalid_geometry requires an explicit known geometry error code."
                )
            objective = INVALID_GEOMETRY_OBJECTIVE
        else:
            if failure_code:
                raise EvaluationContractError(
                    "An evaluated MATLAB row must have an empty failure_code."
                )
            try:
                objective = float(row["objective"])
            except (KeyError, TypeError, ValueError) as exc:
                raise EvaluationContractError(
                    "Evaluated MATLAB row has no objective."
                ) from exc
            if not math.isfinite(objective):
                raise EvaluationContractError(
                    "A non-finite objective must be invalid_geometry or a technical failure."
                )
        records.append(_make_record(
            evaluation=first_evaluation + offset,
            point=point,
            objective=objective,
            slope=slope,
            h_centre=h_centre,
            status=status,
            failure_code=failure_code,
            engine_version=version,
            engine_fingerprint=fingerprint,
        ))
    return records


def _merge_result_identity(
    identity: dict[str, Any], records: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    updated = identity
    for record in records:
        version = str(record.get("engine_version", ""))
        fingerprint = str(record.get("engine_fingerprint", ""))
        if version or fingerprint:
            updated = _set_engine_identity(updated, version, fingerprint)
    return updated


def evaluate_points(
    points: list[list[float]],
    *,
    tag: str,
    mode: str,
    data_selection: str,
    first_evaluation: int,
    checkpoint_file: Path | None = None,
    history_prefix: list[dict[str, Any]] | None = None,
    identity: dict[str, Any] | None = None,
    identity_file: Path | None = None,
    runner: Callable[..., Any] | None = None,
    on_identity: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate points with an injectable subprocess runner."""

    evaluated: list[dict[str, Any]] = []
    batch_size = 1 if mode == "disjoint" else 5
    execute = runner or subprocess.run
    current_identity = identity
    for batch_start in range(0, len(points), batch_size):
        batch = points[batch_start: batch_start + batch_size]
        batch_number = (first_evaluation + batch_start - 1) // batch_size + 1
        points_file = OUT / f".restartable_{tag}_{batch_number:03d}_points.csv"
        results_file = OUT / f".restartable_{tag}_{batch_number:03d}_results.csv"
        log_file = OUT / f"restartable_{tag}_{batch_number:03d}.log"
        write_points(points_file, batch)
        # This exact file is the only batch result input accepted below.  Remove
        # any previous copy so a successful no-op runner cannot reuse stale data.
        results_file.unlink(missing_ok=True)
        results_started_ns = time.time_ns()
        expression = (
            f"cd('{matlab_string(ROOT)}'); "
            "addpath('src','vendor_stl_eidors','tools'); "
            "run_trkg4_inverse_evaluation_batch("
            f"'{matlab_string(points_file)}','{matlab_string(results_file)}',"
            f"'{data_selection}','{mode}');"
        )
        with log_file.open("w", encoding="utf-8") as log:
            completed = execute(
                ["matlab", "-batch", expression],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        if completed.returncode != 0:
            log_text = log_file.read_text(encoding="utf-8", errors="replace")
            geometry_code = _geometry_code_from_log(log_text)
            if geometry_code is not None and len(batch) == 1:
                record = _make_record(
                    evaluation=first_evaluation + batch_start,
                    point=batch[0],
                    objective=INVALID_GEOMETRY_OBJECTIVE,
                    slope=math.nan,
                    h_centre=math.nan,
                    status="invalid_geometry",
                    failure_code=geometry_code,
                )
                evaluated.append(record)
                if checkpoint_file is not None:
                    write_history(
                        checkpoint_file, [*(history_prefix or []), *evaluated]
                    )
                print(
                    f"batch {batch_number:03d}: invalid_geometry ({geometry_code}); "
                    f"preserved {log_file}",
                    flush=True,
                )
                continue
            if geometry_code is not None:
                raise TechnicalEvaluationError(
                    f"MATLAB reported geometry failure for an ambiguous batch of "
                    f"{len(batch)} points; inspect {log_file}."
                )
            raise TechnicalEvaluationError(
                f"MATLAB failed with exit code {completed.returncode}; "
                f"inspect {log_file}. No physical penalty was recorded."
            )
        if not _fresh_artifact(results_file, results_started_ns):
            raise TechnicalEvaluationError(
                f"MATLAB exited successfully but did not produce a fresh, non-empty "
                f"{results_file}."
            )
        batch_rows = _parse_batch_rows(
            results_file, batch, first_evaluation=first_evaluation + batch_start
        )
        evaluated.extend(batch_rows)
        if current_identity is not None:
            current_identity = _merge_result_identity(current_identity, batch_rows)
            if identity_file is not None:
                _write_json(identity_file, current_identity)
            if on_identity is not None:
                on_identity(current_identity)
        if checkpoint_file is not None:
            write_history(
                checkpoint_file, [*(history_prefix or []), *evaluated]
            )
        points_file.unlink(missing_ok=True)
        results_file.unlink(missing_ok=True)
        valid = [row["objective"] for row in batch_rows if row["status"] == "evaluated"]
        best_batch = min(valid) if valid else math.inf
        print(
            f"batch {batch_number:03d}: {len(batch_rows)} points, "
            f"best valid J={best_batch:.6g}",
            flush=True,
        )
    return evaluated


def load_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if not set(HISTORY_FIELDS).issubset(fieldnames):
        raise ResumeIdentityError(
            f"History file {path} has an old or incomplete contract header; refusing resume."
        )
    converted: list[dict[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = {}
        for field in HISTORY_FIELDS:
            if field in NUMERIC_HISTORY_FIELDS:
                try:
                    record[field] = float(row[field])
                except (TypeError, ValueError) as exc:
                    raise ResumeIdentityError(
                        f"History field {field} is not numeric in {path}."
                    ) from exc
            else:
                record[field] = row[field]
        converted.append(record)
    return converted


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in history:
            writer.writerow({field: row.get(field, "") for field in HISTORY_FIELDS})


def record_point(record: dict[str, Any]) -> list[float]:
    return [record[name] for name in PARAMETER_NAMES]


def bounded(point: list[float]) -> list[float]:
    return [min(max(value, lo), hi) for value, lo, hi in zip(point, LOWER, UPPER)]


def valid_history(history: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in history
        if row.get("status") == "evaluated"
        and math.isfinite(float(row.get("objective", math.nan)))
    ]


def best_evaluated(history: Iterable[dict[str, Any]]) -> dict[str, Any]:
    valid = valid_history(history)
    if not valid:
        raise NoValidEvaluationError(
            "No finite evaluated FEM point is available; invalid_geometry points "
            "cannot be finalised."
        )
    return min(valid, key=lambda row: row["objective"])


def _final_artifact_paths(tag: str, out: Path | None = None) -> dict[str, Path]:
    out = OUT if out is None else out
    return {
        "mat": out / f"nik_trkg4_inverse_inhale_{tag}.mat",
        "fit_summary": out / f"nik_trkg4_inverse_inhale_fit_summary_{tag}.csv",
        "comparison": out / f"nik_trkg4_inverse_inhale_comparison_{tag}.csv",
        "depth_curve": out / f"nik_trkg4_inverse_inhale_depth_curve_{tag}.csv",
        "electrodes": out / f"nik_trkg4_inverse_inhale_electrodes_{tag}.csv",
        "grid_geometry": out / f"nik_trkg4_inverse_inhale_grid_geometry_{tag}.csv",
        "jacobian": out / f"nik_trkg4_inverse_inhale_jacobian_{tag}.csv",
        "parameter_sensitivity": out / f"nik_trkg4_inverse_inhale_parameter_sensitivity_{tag}.csv",
        "history": out / f"nik_trkg4_inverse_inhale_history_{tag}.csv",
    }


def _fresh_artifact(
    path: Path,
    started_ns: int,
    previous_signature: tuple[int, str] | None = None,
) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if previous_signature is None:
        return True
    previous_mtime_ns, previous_sha256 = previous_signature
    current_mtime_ns = path.stat().st_mtime_ns
    return current_mtime_ns != previous_mtime_ns or _sha256_file(path) != previous_sha256


def _validate_jacobian_diagnostics(diagnostics: dict[str, Any]) -> None:
    if not isinstance(diagnostics, dict) or not diagnostics:
        raise EvaluationContractError("Jacobian diagnostics JSON is empty.")


def _contract_parameter_values(contract: dict[str, Any]) -> list[float]:
    parameters = contract.get("parameters")
    if not isinstance(parameters, dict) or set(parameters) != set(PARAMETER_NAMES):
        raise EvaluationContractError(
            "Final MATLAB contract parameters must contain exactly the five "
            "canonical parameter names."
        )
    try:
        values = [float(parameters[name]) for name in PARAMETER_NAMES]
    except (TypeError, ValueError) as exc:
        raise EvaluationContractError(
            "Final MATLAB contract parameters must be numeric."
        ) from exc
    return values


def _validate_contract_parameters(
    contract: dict[str, Any], best: dict[str, Any]
) -> None:
    actual = _contract_parameter_values(contract)
    expected = record_point(best)
    for name, got, want in zip(PARAMETER_NAMES, actual, expected):
        if not math.isfinite(got) or not math.isclose(
            got, want, rel_tol=1e-10, abs_tol=1e-9
        ):
            raise EvaluationContractError(
                f"Final MATLAB contract parameter {name}={got!r} does not match "
                f"best={want!r}."
            )


def _root_relative_path(path: Path, root: Path) -> str:
    root_resolved = root.resolve()
    path_resolved = path.resolve()
    try:
        relative = path_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise EvaluationContractError(
            f"Driver artifact {path} is outside ROOT {root_resolved}."
        ) from exc
    return relative.as_posix()


def _path_from_root(relative: str, root: Path) -> Path:
    if not relative or Path(relative).is_absolute():
        raise EvaluationContractError(
            "Contract artifact paths must be non-empty and relative to ROOT."
        )
    root_resolved = root.resolve()
    path = (root_resolved / Path(relative)).resolve()
    try:
        path.relative_to(root_resolved)
    except ValueError as exc:
        raise EvaluationContractError(
            f"Contract artifact path escapes ROOT: {relative!r}."
        ) from exc
    return path


def _validate_driver_artifacts(
    contract: dict[str, Any], *, root: Path | None = None
) -> None:
    root = ROOT if root is None else root
    artifacts = _validate_record_list(
        contract, "artifacts", allow_empty_path=False
    )
    tag = contract.get("result_tag")
    if not isinstance(tag, str) or not tag:
        raise EvaluationContractError(
            "Cannot locate driver artifacts without contract result_tag."
        )
    required_names = {
        f"nik_trkg4_inverse_direct_search_history_{tag}.csv",
        f"nik_trkg4_inverse_direct_search_summary_{tag}.json",
    }
    for name in sorted(required_names):
        matches = [record for record in artifacts if record["name"] == name]
        if len(matches) != 1:
            raise EvaluationContractError(
                f"Final MATLAB contract must contain exactly one driver artifact: {name}."
            )
        record = matches[0]
        path = _path_from_root(record["path"], root)
        if path.name != name or not path.is_file():
            raise EvaluationContractError(
                f"Final MATLAB contract driver artifact is missing: {name}."
            )
        if _sha256_file(path) != record["sha256"]:
            raise EvaluationContractError(
                f"Final MATLAB contract driver artifact hash does not match: {name}."
            )


def _attach_driver_artifacts(
    contract_file: Path,
    contract: dict[str, Any],
    driver_artifacts: dict[str, Path],
) -> dict[str, Any]:
    updated = copy.deepcopy(contract)
    artifacts = list(
        _validate_record_list(updated, "artifacts", allow_empty_path=False)
    )
    root = ROOT
    for path in driver_artifacts.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise EvaluationContractError(
                f"Cannot hash missing driver artifact: {path}."
            )
        relative = _root_relative_path(path, root)
        record = {
            "name": path.name,
            "path": relative,
            "sha256": _sha256_file(path),
        }
        artifacts = [
            entry for entry in artifacts if entry["path"] != relative
        ]
        artifacts.append(record)
    updated["artifacts"] = artifacts
    temporary = contract_file.with_name(contract_file.name + ".tmp")
    try:
        _write_json(temporary, updated)
        os.replace(temporary, contract_file)
    finally:
        temporary.unlink(missing_ok=True)
    return _read_json(contract_file)


def finalise_result(
    best: dict[str, Any],
    *,
    tag: str,
    mode: str,
    data_selection: str,
    identity: dict[str, Any] | None = None,
    runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run final MATLAB with Jacobian enabled and validate fresh artifacts."""

    if best.get("status") != "evaluated":
        raise NoValidEvaluationError(
            "Finalisation is forbidden for an invalid_geometry point."
        )
    point_literal = " ".join(f"{value:.17g}" for value in record_point(best))
    log_file = OUT / f"restartable_{tag}_final.log"
    artifacts = _final_artifact_paths(tag)
    contract_path = _contract_path(tag)
    diagnostics_path = _diagnostics_path(tag)
    watched_paths = [*artifacts.values(), contract_path, diagnostics_path]
    previous_signatures = {
        path: (
            (path.stat().st_mtime_ns, _sha256_file(path))
            if path.is_file()
            else None
        )
        for path in watched_paths
    }
    started_ns = time.time_ns()
    expression = (
        f"cd('{matlab_string(ROOT)}'); "
        "addpath('src','vendor_stl_eidors'); "
        f"R=run_trkg4_inverse_inhale(0,[{point_literal}],true,"
        f"'{data_selection}','{tag}','{mode}',false); "
        "fprintf('FINAL J=%.12g RMSE=%.12g\\n',"
        "R.objective,R.fit_summary.rms_residual_ohm);"
    )
    execute = runner or subprocess.run
    with log_file.open("w", encoding="utf-8") as log:
        completed = execute(
            ["matlab", "-batch", expression],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if completed.returncode != 0:
        raise TechnicalEvaluationError(
            f"Final MATLAB evaluation failed with exit code {completed.returncode}; "
            f"inspect {log_file}."
        )
    missing = [
        path
        for path in artifacts.values()
        if not _fresh_artifact(path, started_ns, previous_signatures[path])
    ]
    contract_file = (
        contract_path
        if _fresh_artifact(contract_path, started_ns, previous_signatures[contract_path])
        else None
    )
    diagnostics_file = (
        diagnostics_path
        if _fresh_artifact(
            diagnostics_path, started_ns, previous_signatures[diagnostics_path]
        )
        else None
    )
    if missing or contract_file is None or diagnostics_file is None:
        missing_names = [str(path) for path in missing]
        if contract_file is None:
            missing_names.append(str(contract_path))
        if diagnostics_file is None:
            missing_names.append(str(diagnostics_path))
        raise EvaluationContractError(
            "Final MATLAB call did not freshly produce all required artifacts: "
            + ", ".join(missing_names)
        )
    contract = _read_json(contract_file)
    contract_version, contract_fingerprint = _validate_contract_identity(
        contract, tag=tag, mode=mode, data_selection=data_selection
    )
    diagnostics = _read_json(diagnostics_file)
    _validate_jacobian_diagnostics(diagnostics)
    _validate_contract_parameters(contract, best)
    if identity is not None:
        engine = identity["engine"]
        if engine.get("version") not in (None, contract_version) or engine.get(
            "fingerprint"
        ) not in (None, contract_fingerprint):
            raise ResumeIdentityError(
                "Final MATLAB contract fingerprint differs from evaluated batches."
            )
    return contract



# One atomic state file is authoritative; CSV/sidecar/summary are projections.
SEARCH_STATE_SCHEMA = "restartable-search-state-v1"


def _search_state_path(tag: str, out: Path | None = None) -> Path:
    return (OUT if out is None else out) / f".restartable_{tag}_state.json"


def _json_bytes(value: Any, *, allow_nan: bool = True) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=allow_nan) + "\n").encode("utf-8")


def _state_numbers(state: dict[str, Any], *, encode: bool) -> dict[str, Any]:
    # Only these checkpoint fields can contain non-finite numbers. Quoted
    # "nan"/"inf"/"-inf" round-trip them without changing numeric CSV output.
    state = copy.deepcopy(state)
    fields = [(row, ("objective", "fem_slope_ohm_per_mm", "h_centre_mm"))
              for row in state["history"]]
    if state["pending"] is not None:
        fields.append((state["pending"], ("before",)))
    for record, names in fields:
        for name in names:
            value = record[name]
            if encode and isinstance(value, float) and not math.isfinite(value):
                record[name] = str(value)
            elif not encode and isinstance(value, str):
                if value not in ("nan", "inf", "-inf"):
                    raise ResumeIdentityError(f"Unknown checkpoint numeric token: {value!r}.")
                record[name] = float(value)
    return state


def _reject_state_constant(value: str) -> None:
    raise ValueError(f"Non-standard checkpoint JSON constant: {value}.")


def _history_bytes(rows: list[dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows({name: row.get(name, "") for name in HISTORY_FIELDS} for row in rows)
    return stream.getvalue().encode("utf-8")


def _atomic_bytes(path: Path, content: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _optional_hash(path: Path) -> str | None:
    return _sha256_file(path) if path.is_file() else None


def _state_checksum(state: dict[str, Any]) -> str:
    wire = _state_numbers(state, encode=True)
    wire.pop("checksum", None)
    return hashlib.sha256(json.dumps(
        wire, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def _state_projections(
    state: dict[str, Any], history_file: Path, identity_file: Path
) -> dict[str, tuple[Path, bytes]]:
    return {
        "history": (history_file, _history_bytes(state["history"])),
        "identity": (identity_file, _json_bytes(state["identity"])),
    }


def _project_search_state(
    state: dict[str, Any], history_file: Path, identity_file: Path, summary_file: Path
) -> None:
    projections = _state_projections(state, history_file, identity_file)
    # Check ALL files before writing any. Only the previous committed projection
    # or this transaction's exact projection is recoverable, never arbitrary CSV.
    for name, (path, content) in projections.items():
        expected = hashlib.sha256(content).hexdigest()
        allowed = {expected}
        if state["phase"] == "running":
            allowed.add(state["projection_base"][name])
        if _optional_hash(path) not in allowed:
            raise ResumeIdentityError(f"Unrecognised {name} projection; refusing to mix state.")
    if state["phase"] == "running":
        tag = state["identity"]["run"]["tag"]
        _contract_path(tag, history_file.parent).unlink(missing_ok=True)
        _atomic_bytes(summary_file, _json_bytes({
            **state["identity"]["run"], "identity": state["identity"],
            "run_id": state["run_id"], "revision": state["revision"],
            "function_evaluations": len(state["history"]),
            "final_step": state["step"],
            "finalization": {"status": "running"},
        }))
        for path, content in projections.values():
            if _optional_hash(path) != hashlib.sha256(content).hexdigest():
                _atomic_bytes(path, content)


def _save_search_state(
    state: dict[str, Any], history_file: Path, identity_file: Path, summary_file: Path
) -> None:
    state["revision"] += 1
    state["projection_base"] = {
        name: _optional_hash(path)
        for name, (path, _) in _state_projections(state, history_file, identity_file).items()
    }
    state["checksum"] = _state_checksum(state)
    path = _search_state_path(state["identity"]["run"]["tag"], history_file.parent)
    # Commit first: a crash during projection cannot lose the accepted batch,
    # the current step, or the cursor within a pass.
    _atomic_bytes(path, _json_bytes(_state_numbers(state, encode=True), allow_nan=False))
    _project_search_state(state, history_file, identity_file, summary_file)


def _restore_search_state(
    history_file: Path, summary_file: Path, identity_file: Path, *,
    tag: str, mode: str, data_selection: str, root: Path = ROOT,
    initial_points_path: Path | None = None,
) -> dict[str, Any]:
    try:
        state = json.loads(
            _search_state_path(tag, history_file.parent).read_text(encoding="utf-8"),
            parse_constant=_reject_state_constant,
        )
    except (OSError, ValueError) as exc:
        raise ResumeIdentityError(f"Cannot read checkpoint as strict JSON: {exc}") from exc
    if (not isinstance(state, dict) or state.get("schema") != SEARCH_STATE_SCHEMA
            or state.get("checksum") != _state_checksum(state)):
        raise ResumeIdentityError("Checkpoint schema/checksum is invalid.")
    state = _state_numbers(state, encode=False)
    _validate_identity(
        state["identity"], tag=tag, mode=mode, data_selection=data_selection,
        root=root, initial_points_path=initial_points_path,
    )
    _validate_history_rows(state["history"], state["identity"])
    if (
        state["phase"] not in ("running", "complete")
        or len(state["step"]) != 5
        or any(not math.isfinite(v) or v <= 0 for v in state["step"])
    ):
        raise ResumeIdentityError("Invalid checkpoint phase or step.")
    pending = state["pending"]
    if pending is not None and (
        pending["kind"] not in ("seed", "sweep")
        or not 0 <= pending["next"] <= len(pending["points"])
        or any(len(p) != 5 or not all(math.isfinite(v) for v in p) for p in pending["points"])
    ):
        raise ResumeIdentityError("Invalid checkpoint pass cursor.")
    if state["phase"] == "complete":
        contract_file = _contract_path(tag, history_file.parent)
        if (
            _optional_hash(contract_file) != state["publication"]["contract_sha256"]
            or _optional_hash(summary_file) != state["publication"]["summary_sha256"]
        ):
            raise ResumeIdentityError("Completed checkpoint publication hash mismatch.")
        contract = _read_json(contract_file)
        version, fingerprint = _validate_contract_identity(
            contract, tag=tag, mode=mode, data_selection=data_selection
        )
        if state["identity"]["engine"] != {"version": version, "fingerprint": fingerprint}:
            raise ResumeIdentityError("Completed checkpoint MATLAB identity mismatch.")
        _validate_driver_artifacts(contract, root=root)
        _validate_contract_parameters(contract, best_evaluated(state["history"]))
    _project_search_state(state, history_file, identity_file, summary_file)
    return state


def main() -> None:
    args = parse_args()
    seed_file = _resolve_initial_points_path(args.initial_points)
    if args.budget < 5:
        raise ValueError("The evaluation budget must be at least five.")
    OUT.mkdir(parents=True, exist_ok=True)
    history_file = OUT / f"nik_trkg4_inverse_direct_search_history_{args.tag}.csv"
    summary_file = OUT / f"nik_trkg4_inverse_direct_search_summary_{args.tag}.json"
    identity_file = _identity_path(args.tag)
    resume_args = dict(
        tag=args.tag, mode=args.mode, data_selection=args.data_selection,
        initial_points_path=seed_file,
    )
    if not args.no_resume and _search_state_path(args.tag).is_file():
        state = _restore_search_state(
            history_file, summary_file, identity_file, **resume_args
        )
        if state["phase"] == "complete" and state["pending"] is None:
            # Preserve the old driver's treatment of unused seeds on extension.
            state["seeds_done"] = False
    else:
        if args.no_resume:
            history = []
            run_identity = build_run_identity(**resume_args)
            step = INITIAL_STEP.copy()
        else:
            history, run_identity, step = load_resume_state(
                history_file, summary_file, identity_file, **resume_args
            )
        state = {
            "schema": SEARCH_STATE_SCHEMA, "run_id": uuid.uuid4().hex, "revision": 0,
            "identity": run_identity, "history": history, "step": step,
            "pending": None, "seeds_done": False, "sweep": 0,
        }
    state["phase"] = "running"
    state.pop("publication", None)
    _save_search_state(state, history_file, identity_file, summary_file)

    while True:
        history, step = state["history"], state["step"]
        pending = state["pending"]
        if pending is None:
            evaluated_keys = {point_key(record_point(row)) for row in history}
            if not state["seeds_done"]:
                points = [
                    point for point in initial_points(seed_file)
                    if point_key(point) not in evaluated_keys
                ][: max(0, args.budget - len(history))]
                if not points:
                    state["seeds_done"] = True
                    _save_search_state(state, history_file, identity_file, summary_file)
                    continue
                before = min((r["objective"] for r in valid_history(history)), default=math.inf)
                kind = "seed"
            else:
                best = best_evaluated(history)
                if len(history) >= args.budget or not any(
                    current >= minimum for current, minimum in zip(step, MINIMUM_STEP)
                ):
                    break
                state["sweep"] += 1
                centre = record_point(best)
                points = []
                for parameter in range(5):
                    for direction in (-1.0, 1.0):
                        candidate = centre.copy()
                        candidate[parameter] += direction * step[parameter]
                        candidate = bounded(candidate)
                        if point_key(candidate) not in evaluated_keys:
                            points.append(candidate)
                points = unique_points(points)[: args.budget - len(history)]
                if not points:
                    state["step"] = [value / 2 for value in step]
                    _save_search_state(state, history_file, identity_file, summary_file)
                    continue
                before, kind = best["objective"], "sweep"
            pending = {"kind": kind, "points": points, "next": 0, "before": before}
            state["pending"] = pending
            _save_search_state(state, history_file, identity_file, summary_file)

        if pending["next"] == len(pending["points"]):
            after = best_evaluated(history)["objective"]
            if pending["kind"] == "seed":
                state["seeds_done"] = True
                if after < pending["before"] - 1e-12:
                    state["step"] = INITIAL_STEP.copy()
            elif not after < pending["before"] - 1e-12:
                state["step"] = [value / 2 for value in step]
            state["pending"] = None
            _save_search_state(state, history_file, identity_file, summary_file)
            if pending["kind"] == "sweep":
                print(
                    f"sweep {state['sweep']:02d}: evaluations={len(history)}, "
                    f"best J={after:.6g}, step={state['step']}", flush=True,
                )
            continue
        if len(history) >= args.budget:
            break
        count = min(1 if args.mode == "disjoint" else 5, args.budget - len(history))
        batch = pending["points"][pending["next"]:pending["next"] + count]
        # evaluate_points/finalise_result and the MATLAB interface stay unchanged.
        # They do not own search checkpoints: commit their returned batch here.
        rows = evaluate_points(
            batch, tag=args.tag, mode=args.mode, data_selection=args.data_selection,
            first_evaluation=len(history) + 1, identity=state["identity"],
        )
        state["identity"] = _merge_result_identity(state["identity"], rows)
        history.extend(rows)
        pending["next"] += len(rows)
        _save_search_state(state, history_file, identity_file, summary_file)

    history, step, run_identity = state["history"], state["step"], state["identity"]
    best = best_evaluated(history)
    contract = finalise_result(
        best,
        tag=args.tag,
        mode=args.mode,
        data_selection=args.data_selection,
        identity=run_identity,
    )
    contract_version, contract_fingerprint = _validate_contract_identity(
        contract, tag=args.tag, mode=args.mode, data_selection=args.data_selection
    )
    run_identity = _set_engine_identity(
        run_identity, contract_version, contract_fingerprint
    )
    _atomic_bytes(identity_file, _json_bytes(run_identity))
    contract_file = _contract_path(args.tag)
    diagnostics_file = _diagnostics_path(args.tag)
    summary = {
        "tag": args.tag,
        "patch_builder_mode": args.mode,
        "data_selection": args.data_selection,
        "evaluation_budget": args.budget,
        "function_evaluations": len(history),
        "objective": best["objective"],
        **{name: best[name] for name in PARAMETER_NAMES},
        "fem_slope_ohm_per_mm": best["fem_slope_ohm_per_mm"],
        "h_centre_mm": best["h_centre_mm"],
        "final_step": step,
        "status": (
            "budget_reached"
            if len(history) >= args.budget
            else "step_resolution_reached"
        ),
        "version": contract_version,
        "fingerprint": contract_fingerprint,
        "identity": run_identity,
        "finalization": {
            "status": "complete",
            "contract": str(contract_file),
            "jacobian_diagnostics": str(diagnostics_file),
        },
    }
    _atomic_bytes(summary_file, _json_bytes(summary))
    contract = _attach_driver_artifacts(
        contract_file,
        contract,
        {
            "inverse_direct_search_history": history_file,
            "inverse_direct_search_summary": summary_file,
        },
    )
    _validate_contract_identity(
        contract, tag=args.tag, mode=args.mode, data_selection=args.data_selection
    )
    _validate_contract_parameters(contract, best)
    _validate_driver_artifacts(contract)

    state["identity"] = run_identity
    state["phase"] = "complete"
    state["publication"] = {
        "contract_sha256": _sha256_file(contract_file),
        "summary_sha256": _sha256_file(summary_file),
    }
    _save_search_state(state, history_file, identity_file, summary_file)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

