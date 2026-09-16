"""Merge and analyse the expanded lateral resistivity FEM range.

The module is deliberately independent of the frozen lateral-array analysis
generator.  It reads all declared sources in ``plan.json`` relative to
``MATLAB_TRKG4_real_subjects`` (``PIPE``), checks their SHA-256 values, and
never copies the observation table into the output directory.

The public :func:`load` function is read-only and returns a dictionary with
the following keys: ``plan`` (validated, normalised plan), ``model`` (a
:class:`RangeSurrogate`), ``obs`` (normalised observations), ``best_sets``
(the old physiological selections), ``base_responses`` (the old response
table), ``base_fits`` (the old physiological fits), ``responses`` (the merged
new response table), and ``source_paths``.  The new spline is fitted only to
the new 13 by 17 ``grid`` rows in ``responses.csv``; ``base_responses`` is an
additional out-of-training interpolation check.

Commands::

    python tools/lateral_resistivity_range_analysis.py --out OUT
    python tools/lateral_resistivity_range_analysis.py --out OUT --prepare-direct
    python tools/lateral_resistivity_range_analysis.py --out OUT --check-direct

The commands do not start MATLAB.  ``--prepare-direct`` writes
``direct_checks.csv`` in the form accepted by
``run_lateral_resistivity_range(..., 'RequestsFile', 'direct_checks.csv')``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import brentq, least_squares, linear_sum_assignment


PIPE = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PIPE / "output" / "exploratory" / "lateral_resistivity_range_20260916"
MODEL = "variable_transverse"
SIZES_COUNT = 9
NEW_VALIDATION_COUNT = 16
ANCHOR_COUNT = 2
WORKER_COUNT = 2
CURVE_RHO2_COUNT = 601
CURVE_RHO1_SCAN_COUNT = 401
FIT_GRID_START_COUNT = 8

RESPONSE_COLUMNS = (
    "point_id", "kind", "model", "rho1", "rho2", "L_mm", "Z",
    "dZ_dlogrho1", "dZ_dlogrho2", "relative_residual",
    "reciprocity_abs", "elapsed_seconds",
)
NUMERIC_RESPONSE_COLUMNS = RESPONSE_COLUMNS[3:]
REQUEST_COLUMNS = ("point_id", "kind", "model", "rho1", "rho2", "L_mm")
CURVE_COLUMNS = (
    "state", "L_mm", "rho1", "rho2", "branch_index", "residual_ohm",
    "within_literature_bounds",
)
INTERSECTION_COLUMNS = (
    "state", "L1_mm", "L2_mm", "branch_index_L1", "branch_index_L2",
    "rho1", "rho2", "pair_max_residual_ohm", "all_nine_rmse_ohm",
    "within_literature_bounds", "method",
)


class ContractError(ValueError):
    """Raised when a declared input does not satisfy the range contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it all in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    elif isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    elif isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    elif isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path: Path, value: Any) -> None:
    """Write JSON with strict JSON numbers and a final newline."""

    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2,
                   allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: Any) -> None:
    text = json.dumps(_json_safe(value), ensure_ascii=False, indent=2,
                      allow_nan=False) + "\n"
    _atomic_write_text(path, text)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSON in {path.name}: {exc}") from exc
    _require(isinstance(value, dict), f"{path.name} must contain a JSON object")
    return value


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value.lower()
    )


def _finite_numeric_frame(frame: pd.DataFrame, columns: Iterable[str], label: str) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        _require(
            np.isfinite(result[column].to_numpy(dtype=float)).all(),
            f"{label}: non-finite value in {column}",
        )
    return result


def _size_index(value: float, sizes: np.ndarray, label: str = "L_mm") -> int:
    index = int(np.argmin(np.abs(sizes - value)))
    _require(
        bool(np.isclose(value, sizes[index], rtol=0.0, atol=1e-7)),
        f"{label}={value!r} is not one of the declared sizes",
    )
    return index


def _relative_path(path_value: Any, label: str) -> Path:
    _require(isinstance(path_value, str) and path_value.strip(),
             f"{label}.path must be a non-empty string")
    relative = Path(path_value.replace("\\", "/"))
    _require(not relative.is_absolute(), f"{label}.path must be relative to PIPE")
    return relative


def load_plan(path: Path) -> dict[str, Any]:
    """Read, validate, and hash-check the new range plan and all its sources."""

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"plan file not found: {path}")
    raw = _read_json(path)
    _require(raw.get("finite_electrodes") is True and raw.get("diameter_mm") == 5
             and raw.get("electrode_count_per_measurement") == 4
             and raw.get("geometry_fixed") is True and raw.get("simultaneous_all_sizes") is False,
             "plan must retain fixed geometry and four finite diameter-5mm CEM contacts")
    _require(raw.get("model") == MODEL, "plan.model must be variable_transverse")

    try:
        bounds = np.asarray(raw["bounds_ohm_m"], dtype=float)
        sizes = np.asarray(raw["sizes_mm"], dtype=float)
        grid_rho1 = np.asarray(raw["grid_rho1"], dtype=float)
        grid_rho2 = np.asarray(raw["grid_rho2"], dtype=float)
        literature_bounds = np.asarray(raw["literature_bounds_ohm_m"], dtype=float)
        acceptance = raw["acceptance"]
        sources = raw["sources"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(
            "plan requires bounds_ohm_m, literature_bounds_ohm_m, sizes_mm, "
            "grid_rho1, grid_rho2, acceptance, and sources"
        ) from exc

    _require(bounds.shape == (2, 2) and np.isfinite(bounds).all(),
             "bounds_ohm_m must have shape (2, 2) and finite values")
    _require(np.all(bounds[:, 0] > 0) and np.all(bounds[:, 1] > bounds[:, 0]),
             "resistivity bounds must be positive and increasing")
    _require(sizes.shape == (SIZES_COUNT,) and np.isfinite(sizes).all(),
             "sizes_mm must contain nine finite values")
    _require(np.all(np.diff(sizes) > 0) and len(set(sizes.tolist())) == SIZES_COUNT,
             "sizes_mm must be strictly increasing and unique")
    for name, values, bound_row in (
        ("grid_rho1", grid_rho1, bounds[0]),
        ("grid_rho2", grid_rho2, bounds[1]),
    ):
        _require(values.ndim == 1 and len(values) >= 4 and np.isfinite(values).all(),
                 f"{name} has an invalid length or non-finite value")
        _require(np.all(values > 0) and np.all(np.diff(values) > 0),
                 f"{name} must be positive and increasing")
        _require(np.isclose(values[0], bound_row[0], rtol=1e-10, atol=1e-12)
                 and np.isclose(values[-1], bound_row[1], rtol=1e-10, atol=1e-12),
                 f"{name} must span the declared bounds")
        log_steps = np.diff(np.log(values))
        _require(np.allclose(log_steps, log_steps[0], rtol=3e-7, atol=3e-9),
                 f"{name} must be logarithmically spaced")
    _require(literature_bounds.shape == (2, 2) and np.isfinite(literature_bounds).all(),
             "literature_bounds_ohm_m must have shape (2, 2)")
    _require(np.all(literature_bounds[:, 0] > 0)
             and np.all(literature_bounds[:, 1] > literature_bounds[:, 0]),
             "literature bounds must be positive and increasing")
    _require(isinstance(acceptance, dict), "plan.acceptance must be an object")
    abs_allowance = float(acceptance["interpolation_absolute_allowance_ohm"])
    rel_allowance = float(acceptance["interpolation_relative_allowance"])
    _require(np.isfinite(abs_allowance) and abs_allowance >= 0,
             "interpolation absolute allowance must be finite and non-negative")
    _require(np.isfinite(rel_allowance) and rel_allowance >= 0,
             "interpolation relative allowance must be finite and non-negative")
    _require(isinstance(sources, dict), "plan.sources must be an object")

    required_sources = {
        "base_plan", "prepared_fem", "base_responses", "base_execution",
        "best_sets", "base_fits", "observations",
    }
    _require(required_sources.issubset(sources),
             "plan.sources is missing: "
             + ", ".join(sorted(required_sources - set(sources))))
    source_paths: dict[str, Path] = {}
    source_records: dict[str, dict[str, Any]] = {}
    for name, record in sources.items():
        _require(isinstance(record, dict), f"plan.sources.{name} must be an object")
        relative = _relative_path(record.get("path"), f"sources.{name}")
        declared_sha = str(record.get("sha256", "")).lower()
        _require(_is_sha(declared_sha), f"sources.{name}.sha256 is not SHA-256")
        source = (PIPE / relative).resolve()
        try:
            source.relative_to(PIPE.resolve())
        except ValueError as exc:
            raise ContractError(f"sources.{name}.path escapes PIPE") from exc
        _require(source.is_file(), f"declared source not found: {relative.as_posix()}")
        actual_sha = sha256_file(source)
        _require(actual_sha == declared_sha,
                 f"source SHA mismatch for {name}: {relative.as_posix()}")
        source_paths[name] = source
        source_records[name] = {
            "path": relative.as_posix(), "sha256": declared_sha,
            "actual_sha256": actual_sha,
        }

    plan_sha = sha256_file(path)
    points_sha = str(raw.get("points_sha256", "")).lower()
    _require(_is_sha(points_sha), "plan.points_sha256 is not SHA-256")
    return {
        "raw": raw,
        "path": path,
        "sha256": plan_sha,
        "bounds": bounds,
        "literature_bounds": literature_bounds,
        "sizes": sizes,
        "grid_rho1": grid_rho1,
        "grid_rho2": grid_rho2,
        "acceptance": acceptance,
        "sources": sources,
        "source_paths": source_paths,
        "source_records": source_records,
        "points_sha256": points_sha,
    }


def _read_response_csv(path: Path, label: str, filter_model: bool = False) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, dtype={"point_id": str})
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} not found: {path}") from None
    except (OSError, ValueError, pd.errors.EmptyDataError) as exc:
        raise ContractError(f"cannot read {label}: {path}") from exc
    missing = [column for column in RESPONSE_COLUMNS if column not in frame.columns]
    _require(not missing, f"{label} is missing columns: {', '.join(missing)}")
    _require(len(frame) > 0, f"{label} is empty")
    frame = _finite_numeric_frame(frame, NUMERIC_RESPONSE_COLUMNS, label)
    frame["point_id"] = frame["point_id"].astype(str)
    frame["kind"] = frame["kind"].astype(str)
    frame["model"] = frame["model"].astype(str)
    if filter_model:
        frame = frame[frame["model"] == MODEL].copy()
    _require((frame["model"] == MODEL).all(), f"{label} contains a non-variable_transverse model")
    _require(not frame.duplicated(["point_id", "L_mm"]).any(),
             f"{label} has duplicate point_id/L_mm rows")
    return frame


def _execution_sidecar(path: Path) -> dict[str, Any]:
    sidecar = path.with_name(path.name + ".execution.json")
    execution = _read_json(sidecar)
    _require(execution.get("complete") is True,
             f"{sidecar.name} is not complete; refusing to merge")
    _require(execution.get("responses_sha256") == sha256_file(path),
             f"{sidecar.name} has a response SHA mismatch")
    return execution


def _identity_sidecar(path: Path) -> dict[str, Any]:
    sidecar = path.with_name(path.name + ".identity.json")
    return _read_json(sidecar)


def _validate_worker_pair(
    out: Path,
    request_name: str,
    response_stem: str,
    plan: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Validate two complete MATLAB response partitions and return them."""

    request_path = out / request_name
    try:
        requests = pd.read_csv(request_path, dtype={"point_id": str})
    except FileNotFoundError:
        raise FileNotFoundError(f"request file not found: {request_path}") from None
    except (OSError, ValueError, pd.errors.EmptyDataError) as exc:
        raise ContractError(f"cannot read request file: {request_path}") from exc
    if "model" not in requests.columns:
        requests["model"] = plan["raw"]["model"]
    missing = [column for column in ("point_id", "kind", "model", "rho1", "rho2")
               if column not in requests.columns]
    _require(not missing, f"{request_name} is missing columns: {', '.join(missing)}")
    requests["point_id"] = requests["point_id"].astype(str)
    requests["kind"] = requests["kind"].astype(str)
    requests["model"] = requests["model"].astype(str)
    _require(requests["model"].eq(MODEL).all(), f"{request_name} contains an unsupported model")
    requests = _finite_numeric_frame(requests, ["rho1", "rho2"], request_name)
    if "L_mm" not in requests.columns:
        requests["L_mm"] = np.nan
    else:
        requests["L_mm"] = pd.to_numeric(requests["L_mm"], errors="coerce")
        finite_lengths = requests["L_mm"].notna()
        _require(np.isfinite(requests.loc[finite_lengths, "L_mm"]).all(),
                 f"{request_name} has invalid finite L_mm values")
    _require(len(requests) > 0 and requests["point_id"].is_unique,
             f"{request_name} must contain unique non-empty point_id values")
    request_sha = sha256_file(request_path)
    if request_name == "points.csv":
        _require(request_sha == plan["points_sha256"], "points.csv differs from frozen plan")
    plan_sha = plan["sha256"]
    prepared_sha = plan["source_records"]["prepared_fem"]["sha256"]
    worker_sha = sha256_file(PIPE / "src" / "run_lateral_resistivity_range.m")
    expected_frames: list[pd.DataFrame] = []
    worker_records: list[dict[str, Any]] = []
    expected_ids_all: set[str] = set()

    for worker_index in range(1, WORKER_COUNT + 1):
        response_path = out / f"{response_stem}_w{worker_index:02d}of{WORKER_COUNT:02d}.csv"
        execution = _execution_sidecar(response_path)
        identity = _identity_sidecar(response_path)
        execution_identity = execution.get("identity")
        _require(isinstance(execution_identity, dict),
                 f"{response_path.name}.execution.json has no identity object")
        _require(execution_identity == identity,
                 f"{response_path.name}: execution and identity sidecars differ")
        expected_identity = {
            "plan_sha256": plan_sha,
            "prepared_sha256": prepared_sha,
            "requests_sha256": request_sha,
            "worker_source_sha256": worker_sha,
            "worker_index": worker_index,
            "num_workers": WORKER_COUNT,
        }
        for key, expected in expected_identity.items():
            _require(identity.get(key) == expected,
                     f"{response_path.name}: identity.{key} does not match current inputs")
        _require(identity.get("finite_diameter_mm") == 5,
                 f"{response_path.name}: finite-diameter identity mismatch")
        _require(execution.get("rows") is None
                 or int(execution["rows"]) == len(pd.read_csv(response_path)),
                 f"{response_path.name}: execution row count mismatch")
        frame = _read_response_csv(response_path, response_path.name)
        for column, limit_name in [("relative_residual", "system_relative_residual_max"),
                                   ("reciprocity_abs", "reciprocity_abs_ohm_max")]:
            _require((frame[column] < plan["acceptance"][limit_name]).all(),
                     response_path.name + ": numerical CEM tolerance failed")
        expected_requests = requests.iloc[worker_index - 1::WORKER_COUNT].copy()
        expected_ids = set(expected_requests["point_id"])
        actual_ids = set(frame["point_id"])
        _require(actual_ids == expected_ids,
                 f"{response_path.name}: response point partition does not match {request_name}")
        _require(not (expected_ids_all & actual_ids),
                 f"{response_path.name}: duplicate point ids across workers")
        expected_ids_all.update(actual_ids)
        sizes = plan["sizes"]
        for _, request in expected_requests.iterrows():
            point_id = str(request["point_id"])
            group = frame[frame["point_id"] == point_id]
            selected_length = request["L_mm"]
            expected_length_count = 1 if np.isfinite(selected_length) else SIZES_COUNT
            _require(len(group) == expected_length_count,
                     f"{response_path.name}/{point_id}: unexpected number of size rows")
            _require((group["kind"] == str(request["kind"])).all()
                     and (group["model"] == MODEL).all(),
                     f"{response_path.name}/{point_id}: kind/model mismatch")
            for column in ("rho1", "rho2"):
                _require(np.allclose(group[column].to_numpy(dtype=float), float(request[column]),
                                     rtol=2e-9, atol=2e-11),
                         f"{response_path.name}/{point_id}: {column} mismatch")
            expected_sizes = [float(selected_length)] if np.isfinite(selected_length) else sizes
            observed_sizes = np.sort(group["L_mm"].to_numpy(dtype=float))
            _require(np.allclose(observed_sizes, np.sort(expected_sizes), rtol=0, atol=1e-7),
                     f"{response_path.name}/{point_id}: L_mm set mismatch")
        expected_frames.append(frame)
        worker_records.append({
            "worker_index": worker_index,
            "response_file": response_path.name,
            "response_sha256": sha256_file(response_path),
            "execution_file": response_path.name + ".execution.json",
            "identity_file": response_path.name + ".identity.json",
            "identity": identity,
            "rows": len(frame),
        })

    _require(expected_ids_all == set(requests["point_id"]),
             f"{request_name}: worker partitions do not cover all requests")
    merged = pd.concat(expected_frames, ignore_index=True)
    order = {point_id: index for index, point_id in enumerate(requests["point_id"])}
    merged["_request_order"] = merged["point_id"].map(order)
    merged["_size_order"] = [
        _size_index(value, plan["sizes"]) for value in merged["L_mm"]
    ]
    merged = merged.sort_values(["_request_order", "_size_order"], kind="mergesort")
    merged = merged.drop(columns=["_request_order", "_size_order"]).reset_index(drop=True)
    _require(len(merged) == int(sum(
        1 if np.isfinite(value) else SIZES_COUNT for value in requests["L_mm"]
    )), f"{request_name}: merged row count is inconsistent with requests")
    metadata = {
        "request_file": request_name,
        "request_sha256": request_sha,
        "response_stem": response_stem,
        "plan_sha256": plan_sha,
        "prepared_sha256": prepared_sha,
        "worker_source_sha256": worker_sha,
        "workers": worker_records,
        "request_rows": len(requests),
        "response_rows": len(merged),
    }
    return requests, merged, metadata


def merge_worker_outputs(out: Path) -> pd.DataFrame:
    """Merge complete ``points_w01of02``/``points_w02of02`` output only."""

    out = Path(out).resolve()
    plan = load_plan(out / "plan.json")
    requests, merged, metadata = _validate_worker_pair(
        out, "points.csv", "points", plan
    )
    _require(set(requests["kind"]) == {"grid", "validation", "anchor"},
             "points.csv contains an unsupported kind")
    _require((requests["kind"] == "grid").sum() == len(plan["grid_rho1"]) * len(plan["grid_rho2"]),
             "points.csv grid count differs from the plan")
    _require((requests["kind"] == "validation").sum() == NEW_VALIDATION_COUNT,
             "points.csv must contain 16 validation points")
    _require((requests["kind"] == "anchor").sum() == ANCHOR_COUNT,
             "points.csv must contain two anchor points")
    _require(set(merged["kind"]) == {"grid", "validation", "anchor"},
             "worker responses contain an unsupported kind")

    out.mkdir(parents=True, exist_ok=True)
    response_path = out / "responses.csv"
    response_frame = merged.loc[:, RESPONSE_COLUMNS]
    csv_text = response_frame.to_csv(index=False, float_format="%.17g", lineterminator="\n")
    _atomic_write_text(response_path, csv_text)
    response_sha = sha256_file(response_path)
    identity = {
        "status": "merged_complete",
        "complete": True,
        "model": MODEL,
        "plan_sha256": plan["sha256"],
        "requests_sha256": metadata["request_sha256"],
        "prepared_sha256": metadata["prepared_sha256"],
        "worker_source_sha256": metadata["worker_source_sha256"],
        "workers": metadata["workers"],
        "rows": len(response_frame),
        "points": len(requests),
        "finite_diameter_mm": 5,
        "responses_sha256": response_sha,
    }
    _atomic_write_json(out / "responses.csv.identity.json", identity)
    execution = {
        "complete": True,
        "status": "merged_complete",
        "identity": identity,
        "responses_sha256": response_sha,
        "rows": len(response_frame),
        "points": len(requests),
        "new_direct_CEM_solves": True,
    }
    _atomic_write_json(out / "responses.csv.execution.json", execution)
    return response_frame


def _validate_merged_sidecars(out: Path, frame: pd.DataFrame) -> None:
    execution = _execution_sidecar(out / "responses.csv")
    _require(execution.get("rows") is None or int(execution["rows"]) == len(frame),
             "responses.csv.execution.json row count mismatch")
    identity_path = out / "responses.csv.identity.json"
    if identity_path.is_file():
        identity = _read_json(identity_path)
        _require(identity.get("complete") is True,
                 "responses.csv.identity.json is not complete")
        _require(identity.get("responses_sha256") == sha256_file(out / "responses.csv"),
                 "responses.csv.identity.json response SHA mismatch")


def _group_state(group: pd.DataFrame, sizes: np.ndarray, label: str) -> dict[str, Any]:
    ordered = group.copy()
    ordered["_size_index"] = [
        _size_index(value, sizes, f"{label}.L_mm") for value in ordered["L_mm"]
    ]
    ordered = ordered.sort_values("_size_index", kind="mergesort")
    _require(len(ordered) == len(sizes)
             and np.array_equal(ordered["_size_index"].to_numpy(), np.arange(len(sizes))),
             f"{label} must contain every declared size exactly once")
    for column in ("rho1", "rho2"):
        values = ordered[column].to_numpy(dtype=float)
        _require(np.allclose(values, values[0], rtol=2e-9, atol=2e-11),
                 f"{label}: {column} changes within point_id")
    return {
        "point_id": str(ordered["point_id"].iloc[0]),
        "kind": str(ordered["kind"].iloc[0]),
        "model": MODEL,
        "rho1": float(ordered["rho1"].iloc[0]),
        "rho2": float(ordered["rho2"].iloc[0]),
        "Z": ordered["Z"].to_numpy(dtype=float),
        "J": ordered[["dZ_dlogrho1", "dZ_dlogrho2"]].to_numpy(dtype=float),
    }


def _states(frame: pd.DataFrame, kind: str, sizes: np.ndarray, expected: int,
            label: str) -> list[dict[str, Any]]:
    subset = frame[frame["kind"] == kind]
    _require(subset["point_id"].nunique() == expected,
             f"{label}: expected {expected} point_id states")
    return [
        _group_state(group, sizes, f"{label}/{point_id}")
        for point_id, group in subset.groupby("point_id", sort=False)
    ]


def _read_observations(plan: dict[str, Any]) -> pd.DataFrame:
    path = plan["source_paths"]["observations"]
    try:
        raw = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.EmptyDataError) as exc:
        raise ContractError(f"cannot read observations source: {path}") from exc
    aliases = {
        "L_mm": ("L_mm", "size_mm"),
        "inhale": ("Z_inhale_hold_ohm", "z_inhale_hold_ohm", "Z_inhale_ohm", "z_inhale_ohm"),
        "exhale": ("Z_exhale_hold_ohm", "z_exhale_hold_ohm", "Z_exhale_ohm", "z_exhale_ohm"),
    }
    names = {
        target: next((candidate for candidate in candidates if candidate in raw.columns), None)
        for target, candidates in aliases.items()
    }
    for target, name in names.items():
        _require(name is not None, f"observations source is missing {target}")
    result = raw[[names["L_mm"], names["inhale"], names["exhale"]]].copy()
    result.columns = ["L_mm", "Z_inhale_hold_ohm", "Z_exhale_hold_ohm"]
    result = _finite_numeric_frame(result, result.columns, "observations source")
    result["_size_index"] = [
        _size_index(value, plan["sizes"], "observations.L_mm") for value in result["L_mm"]
    ]
    _require(len(result) == SIZES_COUNT and result["_size_index"].is_unique,
             "observations source must contain one row for every declared size")
    return result.sort_values("_size_index", kind="mergesort").reset_index(drop=True)


def _read_best_sets(plan: dict[str, Any]) -> pd.DataFrame:
    path = plan["source_paths"]["best_sets"]
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.EmptyDataError) as exc:
        raise ContractError(f"cannot read best_sets source: {path}") from exc
    _require({"k", "sizes_mm", "subset_indices_json"}.issubset(frame.columns),
             "best_sets source must contain k, sizes_mm, and subset_indices_json")
    frame["k"] = pd.to_numeric(frame["k"], errors="coerce")
    _require(np.isfinite(frame["k"]).all(), "best_sets.k contains non-finite values")
    frame["k"] = frame["k"].astype(int)
    _require(set(frame["k"]) == set(range(2, 10)) and len(frame) == 8,
             "best_sets must contain exactly one old physiological set for k=2..9")
    rows: list[dict[str, Any]] = []
    for _, row in frame.sort_values("k").iterrows():
        try:
            indices = tuple(int(value) for value in json.loads(row["subset_indices_json"]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ContractError(f"invalid subset_indices_json for k={row['k']}") from exc
        k = int(row["k"])
        _require(len(indices) == k and len(set(indices)) == k
                 and all(0 <= index < SIZES_COUNT for index in indices),
                 f"best_sets.k={k} has invalid subset indices")
        _require(tuple(sorted(indices)) == indices,
                 f"best_sets.k={k} subset indices must be sorted")
        expected_sizes = "|".join(f"{int(plan['sizes'][index]):g}" for index in indices)
        _require(str(row["sizes_mm"]) == expected_sizes,
                 f"best_sets.k={k} does not match declared sizes")
        record = row.to_dict()
        record["_indices"] = indices
        rows.append(record)
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def _read_base_fits(plan: dict[str, Any]) -> pd.DataFrame:
    path = plan["source_paths"]["base_fits"]
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.EmptyDataError) as exc:
        raise ContractError(f"cannot read base_fits source: {path}") from exc
    required = {"state", "k", "rho1_hat", "rho2_hat"}
    _require(required.issubset(frame.columns),
             "base_fits source is missing: " + ", ".join(sorted(required - set(frame.columns))))
    for column in ("k", "rho1_hat", "rho2_hat"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    _require(np.isfinite(frame[["k", "rho1_hat", "rho2_hat"]].to_numpy(dtype=float)).all(),
             "base_fits contains non-finite fit values")
    frame["k"] = frame["k"].astype(int)
    frame["state"] = frame["state"].astype(str)
    keys = {(state, k) for state, k in zip(frame["state"], frame["k"])}
    expected = {(state, k) for state in ("inhale", "exhale") for k in range(2, 10)}
    _require(keys == expected and len(frame) == 16,
             "base_fits must contain exactly 16 old physiological fits")
    return frame


def _read_base_responses(plan: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frame = _read_response_csv(plan["source_paths"]["base_responses"], "base_responses source", filter_model=True)
    grid = frame[(frame["kind"] == "grid") & (frame["model"] == MODEL)]
    states = _states(grid, "grid", plan["sizes"], 49, "base_responses/grid")
    return frame, states


class RangeSurrogate:
    """The bounded new-range surrogate specified by the analysis contract.

    ``grid_z`` has shape ``(len(grid_rho1), len(grid_rho2), 9)``.  The spline
    is built for ``Z / rho1`` in log-resistivity coordinates.  ``predict`` and
    ``jacobian`` accept a two-element ``log_rho`` vector and reject points
    outside the declared rectangular interpolation domain.
    """

    def __init__(self, plan: dict[str, Any], grid_z: np.ndarray) -> None:
        self.rho1_axis = np.asarray(plan["grid_rho1"], dtype=float)
        self.rho2_axis = np.asarray(plan["grid_rho2"], dtype=float)
        self.log_rho1 = np.log(self.rho1_axis)
        self.log_rho2 = np.log(self.rho2_axis)
        values = np.asarray(grid_z, dtype=float)
        expected_shape = (len(self.rho1_axis), len(self.rho2_axis), SIZES_COUNT)
        _require(values.shape == expected_shape and np.isfinite(values).all(),
                 f"grid_z must have shape {expected_shape} and finite values")
        self.grid_z = values
        self.splines = [
            RectBivariateSpline(
                self.log_rho1, self.log_rho2,
                values[:, :, index] / self.rho1_axis[:, None],
                kx=3, ky=3, s=0.0,
            )
            for index in range(SIZES_COUNT)
        ]

    def _check_log_rho(self, log_rho: Sequence[float]) -> tuple[float, float]:
        values = np.asarray(log_rho, dtype=float)
        _require(values.shape == (2,) and np.isfinite(values).all(),
                 "log_rho must be a finite two-element vector")
        x, y = float(values[0]), float(values[1])
        tolerance = 2e-12
        _require(self.log_rho1[0] - tolerance <= x <= self.log_rho1[-1] + tolerance
                 and self.log_rho2[0] - tolerance <= y <= self.log_rho2[-1] + tolerance,
                 "surrogate evaluation outside the declared interpolation bounds")
        return min(max(x, self.log_rho1[0]), self.log_rho1[-1]), min(
            max(y, self.log_rho2[0]), self.log_rho2[-1]
        )

    @staticmethod
    def _scalar_eval(spline: RectBivariateSpline, x: float, y: float,
                     dx: int = 0, dy: int = 0) -> float:
        return float(np.asarray(spline.ev(x, y, dx=dx, dy=dy)).reshape(-1)[0])

    def predict(self, log_rho: Sequence[float], indices: Iterable[int] | None = None) -> np.ndarray:
        x, y = self._check_log_rho(log_rho)
        selected = list(range(SIZES_COUNT) if indices is None else indices)
        _require(all(0 <= int(index) < SIZES_COUNT for index in selected),
                 "surrogate channel index is outside 0..8")
        rho1 = math.exp(x)
        return np.asarray([
            rho1 * self._scalar_eval(self.splines[int(index)], x, y)
            for index in selected
        ], dtype=float)

    def jacobian(self, log_rho: Sequence[float], indices: Iterable[int] | None = None) -> np.ndarray:
        x, y = self._check_log_rho(log_rho)
        selected = list(range(SIZES_COUNT) if indices is None else indices)
        _require(all(0 <= int(index) < SIZES_COUNT for index in selected),
                 "surrogate channel index is outside 0..8")
        rho1 = math.exp(x)
        result = []
        for index in selected:
            spline = self.splines[int(index)]
            value = self._scalar_eval(spline, x, y)
            sx = self._scalar_eval(spline, x, y, dx=1)
            sy = self._scalar_eval(spline, x, y, dy=1)
            result.append((rho1 * (value + sx), rho1 * sy))
        return np.asarray(result, dtype=float)

    def derivative_grid(self, x: np.ndarray, y: np.ndarray, index: int) -> np.ndarray:
        """Evaluate dZ/dlog(rho1) on an already bounded mesh."""

        xx = np.asarray(x, dtype=float)
        yy = np.asarray(y, dtype=float)
        _require(xx.shape == yy.shape, "derivative grid coordinates must have equal shapes")
        spline = self.splines[int(index)]
        return np.exp(xx) * (spline.ev(xx, yy) + spline.ev(xx, yy, dx=1))


def _new_response_tables(plan: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    _require(set(frame["kind"]) == {"grid", "validation", "anchor"},
             "responses.csv must contain grid, validation, and anchor kinds")
    counts = frame.groupby("kind")["point_id"].nunique().to_dict()
    grid_count = len(plan["grid_rho1"]) * len(plan["grid_rho2"])
    _require(counts == {"grid": grid_count, "validation": 16, "anchor": 2},
             f"responses.csv point counts differ from the plan: {counts}")
    grid_states = _states(frame, "grid", plan["sizes"], grid_count, "responses/grid")
    validation_states = _states(frame, "validation", plan["sizes"], 16, "responses/validation")
    anchor_states = _states(frame, "anchor", plan["sizes"], 2, "responses/anchor")
    grid_z = np.full((len(plan["grid_rho1"]), len(plan["grid_rho2"]), SIZES_COUNT), np.nan)
    grid_j = np.full((len(plan["grid_rho1"]), len(plan["grid_rho2"]), SIZES_COUNT, 2), np.nan)
    seen: set[tuple[int, int]] = set()
    for state in grid_states:
        i = int(np.argmin(np.abs(plan["grid_rho1"] - state["rho1"])))
        j = int(np.argmin(np.abs(plan["grid_rho2"] - state["rho2"])))
        _require(np.isclose(state["rho1"], plan["grid_rho1"][i], rtol=2e-9, atol=2e-11)
                 and np.isclose(state["rho2"], plan["grid_rho2"][j], rtol=2e-9, atol=2e-11),
                 f"grid point {state['point_id']} is not on the declared grid")
        _require((i, j) not in seen, f"duplicate grid state {(i, j)}")
        seen.add((i, j))
        grid_z[i, j] = state["Z"]
        grid_j[i, j] = state["J"]
    _require(len(seen) == len(plan["grid_rho1"]) * len(plan["grid_rho2"])
             and np.isfinite(grid_z).all() and np.isfinite(grid_j).all(),
             "responses.csv grid does not cover every declared state")
    for state in validation_states + anchor_states:
        _require(plan["grid_rho1"][0] <= state["rho1"] <= plan["grid_rho1"][-1]
                 and plan["grid_rho2"][0] <= state["rho2"] <= plan["grid_rho2"][-1],
                 f"{state['kind']} point {state['point_id']} lies outside the surrogate domain")
    return {
        "frame": frame,
        "grid_z": grid_z,
        "grid_j": grid_j,
        "grid_states": grid_states,
        "validation_states": validation_states,
        "anchor_states": anchor_states,
        "grid_log_points": np.asarray(list(itertools.product(
            np.log(plan["grid_rho1"]), np.log(plan["grid_rho2"])
        )), dtype=float),
    }


def load(out: Path) -> dict[str, Any]:
    """Load a completed range analysis input set without writing outputs.

    ``load(out)["model"]`` is the new ``RangeSurrogate``.  The ``obs`` table
    is read directly from ``plan.sources.observations`` and the old selections
    are read directly from ``plan.sources.best_sets``.  Calling this function
    does not merge workers; run :func:`merge_worker_outputs` or the main CLI
    first when ``responses.csv`` has not yet been created.
    """

    out = Path(out).resolve()
    plan = load_plan(out / "plan.json")
    response_path = out / "responses.csv"
    frame = _read_response_csv(response_path, "responses.csv")
    _validate_merged_sidecars(out, frame)
    tables = _new_response_tables(plan, frame)
    base_responses, base_grid_states = _read_base_responses(plan)
    best_sets = _read_best_sets(plan)
    base_fits = _read_base_fits(plan)
    obs = _read_observations(plan)
    model = RangeSurrogate(plan, tables["grid_z"])
    return {
        "out": out,
        "plan": plan,
        "model": model,
        "obs": obs,
        "best_sets": best_sets,
        "base_responses": base_responses,
        "base_grid_states": base_grid_states,
        "base_fits": base_fits,
        "responses": frame,
        "tables": tables,
        "source_paths": plan["source_paths"],
    }


def _stats(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float).reshape(-1)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"n": 0, "max_abs": None, "rmse": None, "mean_abs": None,
                "median_abs": None, "p95_abs": None}
    absolute = np.abs(array)
    return {
        "n": int(len(array)),
        "max_abs": float(absolute.max()),
        "rmse": float(np.sqrt(np.mean(array ** 2))),
        "mean_abs": float(absolute.mean()),
        "median_abs": float(np.median(absolute)),
        "p95_abs": float(np.percentile(absolute, 95)),
    }


def _acceptance_result(error: np.ndarray, reference: np.ndarray, plan: dict[str, Any]) -> dict[str, Any]:
    absolute = float(plan["acceptance"]["interpolation_absolute_allowance_ohm"])
    relative = float(plan["acceptance"]["interpolation_relative_allowance"])
    tolerance = absolute + relative * np.abs(reference)
    error = np.asarray(error, dtype=float)
    passed_mask = np.abs(error) <= tolerance
    ratios = np.abs(error) / tolerance
    return {
        "passed": bool(np.isfinite(error).all() and passed_mask.all()),
        "max_abs_error_ohm": float(np.nanmax(np.abs(error))),
        "max_allowed_ohm": float(np.nanmax(tolerance)),
        "max_error_to_allowed_ratio": float(np.nanmax(ratios)),
        "errors_ohm": _stats(error),
        "allowance": "interpolation_absolute_allowance_ohm + "
                     "interpolation_relative_allowance * abs(Zref)",
    }


def _state_interpolation_check(
    states: list[dict[str, Any]], model: RangeSurrogate, plan: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    z_errors: list[np.ndarray] = []
    j_errors: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for state in states:
        log_rho = np.log([state["rho1"], state["rho2"]])
        predicted = model.predict(log_rho)
        predicted_j = model.jacobian(log_rho)
        error = predicted - state["Z"]
        j_error = predicted_j - state["J"]
        z_errors.append(error)
        j_errors.append(j_error)
        allowed = (float(plan["acceptance"]["interpolation_absolute_allowance_ohm"])
                   + float(plan["acceptance"]["interpolation_relative_allowance"]) * np.abs(state["Z"]))
        rows.append({
            "point_id": state["point_id"],
            "rho1": state["rho1"],
            "rho2": state["rho2"],
            "Z_max_abs_error_ohm": float(np.abs(error).max()),
            "Z_max_allowed_ohm": float(allowed.max()),
            "Z_passed": bool(np.all(np.abs(error) <= allowed)),
            "dZ_dlogrho1_max_abs_error_ohm": float(np.abs(j_error[:, 0]).max()),
            "dZ_dlogrho2_max_abs_error_ohm": float(np.abs(j_error[:, 1]).max()),
        })
    error_matrix = np.asarray(z_errors, dtype=float)
    j_matrix = np.asarray(j_errors, dtype=float)
    reference_matrix = np.asarray([state["Z"] for state in states], dtype=float)
    return {
        "label": label,
        "points": len(states),
        "rows": int(len(states) * SIZES_COUNT),
        "Z": _acceptance_result(error_matrix, reference_matrix, plan),
        "dZ_dlogrho1_abs_error_ohm": _stats(j_matrix[:, :, 0]),
        "dZ_dlogrho2_abs_error_ohm": _stats(j_matrix[:, :, 1]),
        "points_summary": rows,
    }


def _anchor_check(ctx: dict[str, Any]) -> dict[str, Any]:
    plan = ctx["plan"]
    anchors = ctx["tables"]["anchor_states"]
    reference = ctx["base_grid_states"]
    errors: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for anchor in anchors:
        matches = [state for state in reference
                   if np.isclose(state["rho1"], anchor["rho1"], rtol=2e-9, atol=2e-11)
                   and np.isclose(state["rho2"], anchor["rho2"], rtol=2e-9, atol=2e-11)]
        if len(matches) != 1:
            rows.append({"point_id": anchor["point_id"], "reference_found": False,
                         "passed": False})
            continue
        error = anchor["Z"] - matches[0]["Z"]
        errors.append(error)
        limit = float(plan["acceptance"].get("anchor_Z_abs_error_ohm_max", 1e-6))
        rows.append({"point_id": anchor["point_id"], "reference_found": True,
                     "max_abs_error_ohm": float(np.abs(error).max()),
                     "allowed_abs_error_ohm": limit,
                     "passed": bool(np.abs(error).max() <= limit)})
    if not errors:
        return {"points": len(anchors), "passed": False, "rows": rows,
                "max_abs_error_ohm": None}
    all_errors = np.asarray(errors)
    limit = float(plan["acceptance"].get("anchor_Z_abs_error_ohm_max", 1e-6))
    return {"points": len(anchors), "passed": bool(np.abs(all_errors).max() <= limit),
            "max_abs_error_ohm": float(np.abs(all_errors).max()),
            "allowed_abs_error_ohm": limit, "rows": rows}


def _monotonicity_qc(model: RangeSurrogate) -> dict[str, Any]:
    x = np.linspace(model.log_rho1[0], model.log_rho1[-1], CURVE_RHO1_SCAN_COUNT)
    y = np.linspace(model.log_rho2[0], model.log_rho2[-1], CURVE_RHO2_COUNT)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    channels = []
    for index in range(SIZES_COUNT):
        derivative = np.asarray(model.derivative_grid(xx, yy, index), dtype=float)
        finite = derivative[np.isfinite(derivative)]
        nonnegative = bool(len(finite) and np.all(finite >= -1e-10))
        nonpositive = bool(len(finite) and np.all(finite <= 1e-10))
        channels.append({
            "L_mm": float(index),
            "min_dZ_dlogrho1": float(finite.min()) if len(finite) else None,
            "max_dZ_dlogrho1": float(finite.max()) if len(finite) else None,
            "numerically_non_decreasing": nonnegative,
            "numerically_non_increasing": nonpositive,
            "has_both_derivative_signs": bool(len(finite)
                                               and finite.min() < -1e-10
                                               and finite.max() > 1e-10),
        })
    return {
        "scan_rho1_points": CURVE_RHO1_SCAN_COUNT,
        "scan_rho2_points": CURVE_RHO2_COUNT,
        "assumed_for_roots": False,
        "channels": channels,
    }


def interpolation_qc(ctx: dict[str, Any]) -> dict[str, Any]:
    """Compare 16 new validation states and 49 old grid states to the spline."""

    plan = ctx["plan"]
    model = ctx["model"]
    new_check = _state_interpolation_check(
        ctx["tables"]["validation_states"], model, plan, "new_validation_16"
    )
    old_check = _state_interpolation_check(
        ctx["base_grid_states"], model, plan, "old_base_grid_49_additional_check"
    )
    knots = []
    for state in ctx["tables"]["grid_states"]:
        error = model.predict(np.log([state["rho1"], state["rho2"]])) - state["Z"]
        knots.append(error)
    knot_error = np.asarray(knots, dtype=float)
    anchor_check = _anchor_check(ctx)
    monotonicity = _monotonicity_qc(model)
    passed = bool(new_check["Z"]["passed"] and old_check["Z"]["passed"]
                  and anchor_check["passed"] and np.abs(knot_error).max() <= 1e-8)
    return {
        "status": "computed_interpolation_qc",
        "scientific_status": "surrogate_qc_only",
        "passed": passed,
        "model": MODEL,
        "training": {
            "source": "merged responses.csv grid only",
            "grid_points": len(plan["grid_rho1"]) * len(plan["grid_rho2"]),
            "channels": SIZES_COUNT,
            "base_responses_used_for_training": False,
        },
        "acceptance": {
            "absolute_allowance_ohm": float(plan["acceptance"]["interpolation_absolute_allowance_ohm"]),
            "relative_allowance": float(plan["acceptance"]["interpolation_relative_allowance"]),
            "formula": "0.02 ohm + 0.001 * abs(Zref) from plan.acceptance",
        },
        "new_validation": new_check,
        "old_base_grid_additional_check": old_check,
        "anchor_identity_check": anchor_check,
        "grid_knot_self_check": {
            "points": len(plan["grid_rho1"]) * len(plan["grid_rho2"]),
            "rows": len(plan["grid_rho1"]) * len(plan["grid_rho2"]) * SIZES_COUNT,
            "max_abs_error_ohm": float(np.abs(knot_error).max()),
            "passed": bool(np.abs(knot_error).max() <= 1e-8),
        },
        "monotonicity": monotonicity,
        "sources": plan["source_records"],
        "observations_source_read_in_place": True,
        "observations_source": plan["source_records"]["observations"]["path"],
        "observations_sha256": plan["source_records"]["observations"]["sha256"],
        "limitations": [
            "Interpolation QC does not establish physical or experimental validity.",
            "Old 49-point responses are a transfer check and are not spline training data.",
            "Numerical monotonicity is reported and is not assumed by root finding.",
        ],
    }


def _best_indices(row: pd.Series) -> tuple[int, ...]:
    value = row.get("_indices")
    if isinstance(value, tuple):
        return value
    return tuple(int(item) for item in json.loads(row["subset_indices_json"]))


def _fit_starts(
    ctx: dict[str, Any], target: np.ndarray, indices: tuple[int, ...],
    old_fit: pd.Series | None,
) -> list[tuple[np.ndarray, str]]:
    grid = ctx["tables"]["grid_z"].reshape(-1, SIZES_COUNT)
    grid_log = ctx["tables"]["grid_log_points"]
    residuals = grid[:, list(indices)] - target[list(indices)]
    ranking = np.argsort(np.sum(residuals ** 2, axis=1), kind="mergesort")
    lower = np.log(ctx["plan"]["bounds"][:, 0])
    upper = np.log(ctx["plan"]["bounds"][:, 1])
    starts: list[tuple[np.ndarray, str]] = []
    for rank, grid_index in enumerate(ranking[:FIT_GRID_START_COUNT]):
        starts.append((np.clip(grid_log[int(grid_index)].copy(), lower, upper),
                       f"best_grid_{rank + 1}"))
    if old_fit is not None:
        old_rho = np.asarray([float(old_fit["rho1_hat"]), float(old_fit["rho2_hat"])])
        if np.isfinite(old_rho).all() and np.all(old_rho > 0):
            starts.append((np.clip(np.log(old_rho), lower, upper), "old_fit"))
    unique: list[tuple[np.ndarray, str]] = []
    for start, label in starts:
        if not any(np.allclose(start, previous, rtol=0, atol=1e-13)
                   for previous, _ in unique):
            unique.append((start, label))
    _require(len(unique) >= 2, "multi-start construction produced fewer than two starts")
    return unique


def _fit_subset(
    ctx: dict[str, Any], target: np.ndarray, indices: tuple[int, ...],
    old_fit: pd.Series | None,
) -> dict[str, Any]:
    model: RangeSurrogate = ctx["model"]
    bounds = ctx["plan"]["bounds"]
    lower, upper = np.log(bounds[:, 0]), np.log(bounds[:, 1])
    starts = _fit_starts(ctx, target, indices, old_fit)
    candidates: list[tuple[Any, str, np.ndarray]] = []
    failures: list[str] = []
    selected = list(indices)
    for start, label in starts:
        try:
            result = least_squares(
                lambda x: model.predict(x, selected) - target[selected],
                start,
                jac=lambda x: model.jacobian(x, selected),
                bounds=(lower, upper), method="trf",
                ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=2000,
            )
            if np.isfinite(result.cost) and np.isfinite(result.x).all():
                candidates.append((result, label, start))
            else:
                failures.append(f"{label}: non-finite optimizer result")
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            failures.append(f"{label}: {exc}")
    _require(candidates, "all real inverse multi-start fits failed: " + "; ".join(failures))
    result, label, start = min(
        candidates,
        key=lambda item: (float(item[0].cost), int(not bool(item[0].success)), item[1]),
    )
    rho = np.exp(result.x)
    residual = model.predict(result.x) - target
    lower_hit = np.isclose(rho, bounds[:, 0], rtol=2e-8, atol=2e-11)
    upper_hit = np.isclose(rho, bounds[:, 1], rtol=2e-8, atol=2e-11)
    return {
        "result": result,
        "start_label": label,
        "start_log_rho": start,
        "rho": rho,
        "residual": residual,
        "lower_hit": lower_hit,
        "upper_hit": upper_hit,
        "n_starts": len(starts),
        "failed_starts": failures,
    }


def _old_fit_for(ctx: dict[str, Any], state: str, k: int) -> pd.Series:
    matches = ctx["base_fits"][(ctx["base_fits"]["state"] == state)
                                & (ctx["base_fits"]["k"] == k)]
    _require(len(matches) == 1, f"base_fits is missing the old fit for {state}, k={k}")
    return matches.iloc[0]


def real_inverse(ctx: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the declared old best sets against the two observed states."""

    plan = ctx["plan"]
    observations = ctx["obs"]
    fit_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for state, target_column in (("inhale", "Z_inhale_hold_ohm"),
                                 ("exhale", "Z_exhale_hold_ohm")):
        target = observations[target_column].to_numpy(dtype=float)
        for _, selected in ctx["best_sets"].sort_values("k").iterrows():
            k = int(selected["k"])
            indices = _best_indices(selected)
            fit = _fit_subset(ctx, target, indices, _old_fit_for(ctx, state, k))
            rho = fit["rho"]
            residual = fit["residual"]
            lit = plan["literature_bounds"]
            within_literature = bool(
                np.all(rho >= lit[:, 0] - 1e-10) and np.all(rho <= lit[:, 1] + 1e-10)
            )
            selected_residual = residual[list(indices)]
            fit_rows.append({
                "state": state,
                "k": k,
                "subset_id": str(selected["subset_id"]),
                "sizes_mm": str(selected["sizes_mm"]),
                "rho1_hat": float(rho[0]),
                "rho2_hat": float(rho[1]),
                "bounds_hit_rho1_lower": bool(fit["lower_hit"][0]),
                "bounds_hit_rho1_upper": bool(fit["upper_hit"][0]),
                "bounds_hit_rho2_lower": bool(fit["lower_hit"][1]),
                "bounds_hit_rho2_upper": bool(fit["upper_hit"][1]),
                "within_literature_bounds": within_literature,
                "fit_resid_rmse_ohm": float(np.sqrt(np.mean(selected_residual ** 2))),
                "fit_resid_norm_ohm": float(np.linalg.norm(selected_residual)),
                "all_size_rmse_ohm": float(np.sqrt(np.mean(residual ** 2))),
                "heldout_rmse_ohm": (
                    float(np.sqrt(np.mean(residual[[i for i in range(SIZES_COUNT)
                                                     if i not in indices]] ** 2)))
                    if k < SIZES_COUNT else np.nan
                ),
                "heldout_count": SIZES_COUNT - k,
                "nfev": int(fit["result"].nfev),
                "optimality": float(fit["result"].optimality),
                "success": bool(fit["result"].success),
                "start_used": fit["start_label"],
                "n_starts": int(fit["n_starts"]),
                "independent_tissue_truth": False,
                "status": "conditional_real_inverse_no_tissue_truth",
            })
            predicted = ctx["model"].predict(np.log(rho))
            for index, size in enumerate(plan["sizes"]):
                prediction_rows.append({
                    "state": state,
                    "k": k,
                    "subset_id": str(selected["subset_id"]),
                    "sizes_mm": str(selected["sizes_mm"]),
                    "L_mm": float(size),
                    "selected": bool(index in indices),
                    "observed_Z_ohm": float(target[index]),
                    "predicted_Z_ohm": float(predicted[index]),
                    "residual_ohm": float(predicted[index] - target[index]),
                    "independent_tissue_truth": False,
                })
    fits = pd.DataFrame(fit_rows)
    predictions = pd.DataFrame(prediction_rows)
    _require(len(fits) == 16 and len(predictions) == 16 * SIZES_COUNT,
             "real inverse did not produce 16 fits and 144 predictions")
    return fits, predictions


def _literature_contains(plan: dict[str, Any], rho1: float, rho2: float) -> bool:
    bounds = plan["literature_bounds"]
    return bool(bounds[0, 0] <= rho1 <= bounds[0, 1]
                and bounds[1, 0] <= rho2 <= bounds[1, 1])


def _find_roots_at(
    model: RangeSurrogate, observed: float, channel: int, rho2: float,
    scan_x: np.ndarray,
) -> list[float]:
    """Find all sign-changing/exact roots in the rho1 interval at one rho2."""

    y = math.log(float(rho2))
    model._check_log_rho((scan_x[0], y))
    model._check_log_rho((scan_x[-1], y))
    values = np.exp(scan_x) * model.splines[channel].ev(scan_x, np.full_like(scan_x, y)) - observed
    roots: list[float] = []
    exact_tolerance = 2e-10 * max(1.0, abs(float(observed)))
    for index, value in enumerate(values):
        if abs(value) <= exact_tolerance:
            roots.append(float(scan_x[index]))
    for index in range(len(scan_x) - 1):
        left, right = float(values[index]), float(values[index + 1])
        if left * right < 0:
            root = brentq(
                lambda x: float(model.predict((float(x), y), [channel])[0] - observed),
                float(scan_x[index]), float(scan_x[index + 1]),
                xtol=1e-13, rtol=1e-14, maxiter=200,
            )
            roots.append(float(root))
    roots.sort()
    unique: list[float] = []
    for root in roots:
        if not unique or abs(root - unique[-1]) > 3e-9:
            unique.append(root)
    return unique


def _track_roots(
    root_lists: list[list[float]], scan_x: np.ndarray,
) -> tuple[dict[int, dict[int, float]], dict[str, Any]]:
    """Track roots between adjacent rho2 samples without blind branch joins."""

    step = float(np.max(np.diff(scan_x)))
    max_jump = max(4.0 * step, 0.05)
    branches: dict[int, dict[int, float]] = {}
    active: dict[int, float] = {}
    next_branch = 0
    ambiguous = 0
    assignments = 0
    births = 0
    deaths = 0
    for sample_index, roots in enumerate(root_lists):
        roots = list(roots)
        current: dict[int, float] = {}
        if active and roots:
            old_ids = list(active)
            cost = np.asarray([[abs(active[branch] - root) for root in roots]
                               for branch in old_ids], dtype=float)
            row_ind, col_ind = linear_sum_assignment(cost)
            used_roots: set[int] = set()
            for row, column in zip(row_ind, col_ind):
                distance = float(cost[row, column])
                if distance <= max_jump:
                    branch = old_ids[int(row)]
                    current[branch] = float(roots[int(column)])
                    used_roots.add(int(column))
                    assignments += 1
                    alternatives = np.sort(cost[row])
                    if len(alternatives) > 1 and alternatives[1] - alternatives[0] < 0.25 * step:
                        ambiguous += 1
            for root_index, root in enumerate(roots):
                if root_index not in used_roots:
                    branch = next_branch
                    next_branch += 1
                    current[branch] = float(root)
                    births += 1
            deaths += max(0, len(active) - sum(branch in current for branch in active))
        else:
            for root in roots:
                branch = next_branch
                next_branch += 1
                current[branch] = float(root)
                births += 1
            if active and not roots:
                deaths += len(active)
        for branch, value in current.items():
            branches.setdefault(branch, {})[sample_index] = value
        active = current
    if not branches:
        branches[0] = {}
    branch_qc = {
        "branch_count": len(branches),
        "multiple_roots_observed": any(len(roots) > 1 for roots in root_lists),
        "max_roots_at_one_rho2": max((len(roots) for roots in root_lists), default=0),
        "samples_without_root": int(sum(not roots for roots in root_lists)),
        "branch_assignment_ambiguities": ambiguous,
        "branch_assignments": assignments,
        "branch_births": births,
        "branch_deaths": deaths,
        "max_allowed_log_rho1_jump": max_jump,
        "tracking_method": "nearest one-to-one assignment with bounded log-rho1 jump",
    }
    return branches, branch_qc


def _curve_for_channel(
    ctx: dict[str, Any], state: str, channel: int, observed: float,
    rho2_axis: np.ndarray, scan_x: np.ndarray,
) -> dict[str, Any]:
    model: RangeSurrogate = ctx["model"]
    root_lists = [
        _find_roots_at(model, observed, channel, rho2, scan_x)
        for rho2 in rho2_axis
    ]
    branches, branch_qc = _track_roots(root_lists, scan_x)
    rows: list[dict[str, Any]] = []
    literature = ctx["plan"]["literature_bounds"]
    for branch_index, values in sorted(branches.items()):
        for sample_index, rho2 in enumerate(rho2_axis):
            log_rho1 = values.get(sample_index, np.nan)
            if np.isfinite(log_rho1):
                rho1 = float(math.exp(log_rho1))
                residual = float(model.predict((log_rho1, math.log(float(rho2))), [channel])[0]
                                 - observed)
                within = bool(literature[0, 0] <= rho1 <= literature[0, 1]
                              and literature[1, 0] <= rho2 <= literature[1, 1])
            else:
                rho1 = np.nan
                residual = np.nan
                within = False
            rows.append({
                "state": state,
                "L_mm": float(ctx["plan"]["sizes"][channel]),
                "rho1": rho1,
                "rho2": float(rho2),
                "branch_index": int(branch_index),
                "residual_ohm": residual,
                "within_literature_bounds": within,
            })
    return {"rows": rows, "root_lists": root_lists, "branches": branches,
            "qc": branch_qc, "channel": channel, "state": state}


def _curve_value_at(
    ctx: dict[str, Any], curve: dict[str, Any], observed: float,
    branch_index: int, rho2: float, left_log: float, right_log: float,
    left_rho2: float, right_rho2: float, scan_x: np.ndarray,
) -> float | None:
    """Select the branch near its locally interpolated log-rho1 position."""

    values = curve["branches"].get(branch_index, {})
    t = 0.0 if right_rho2 == left_rho2 else (
        math.log(rho2) - math.log(left_rho2)
    ) / (math.log(right_rho2) - math.log(left_rho2))
    target = left_log + t * (right_log - left_log)
    roots = _find_roots_at(ctx["model"], observed, curve["channel"], rho2, scan_x)
    if not roots:
        return None
    distances = np.asarray([abs(root - target) for root in roots])
    order = np.argsort(distances)
    if len(order) > 1 and distances[order[1]] - distances[order[0]] < 0.1 * max(
        np.max(np.diff(scan_x)), 1e-8
    ):
        return None
    return float(roots[int(order[0])])


def _curve_intersections(
    ctx: dict[str, Any], curve_results: dict[tuple[str, int], dict[str, Any]],
    rho2_axis: np.ndarray, scan_x: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    intersections: list[dict[str, Any]] = []
    pair_qc: list[dict[str, Any]] = []
    for state in ("inhale", "exhale"):
        observed = ctx["obs"][f"Z_{state}_hold_ohm"].to_numpy(dtype=float)
        for channel_a, channel_b in itertools.combinations(range(SIZES_COUNT), 2):
            curve_a = curve_results[(state, channel_a)]
            curve_b = curve_results[(state, channel_b)]
            candidates: list[tuple[int, int, int]] = []
            for branch_a, values_a in curve_a["branches"].items():
                for branch_b, values_b in curve_b["branches"].items():
                    shared = sorted(set(values_a).intersection(values_b))
                    if not shared:
                        continue
                    differences = np.asarray([
                        values_a[index] - values_b[index] for index in shared
                    ], dtype=float)
                    for local, index in enumerate(shared):
                        if differences[local] == 0:
                            candidates.append((branch_a, branch_b, index))
                        if local + 1 < len(shared):
                            next_index = shared[local + 1]
                            next_difference = differences[local + 1]
                            if differences[local] * next_difference < 0 and next_index == index + 1:
                                candidates.append((branch_a, branch_b, index))
                    pair_qc.append({
                        "state": state,
                        "L1_mm": float(ctx["plan"]["sizes"][channel_a]),
                        "L2_mm": float(ctx["plan"]["sizes"][channel_b]),
                        "branch_index_L1": int(branch_a),
                        "branch_index_L2": int(branch_b),
                        "shared_samples": len(shared),
                        "sampled_sign_changes": int(np.sum(differences[:-1] * differences[1:] < 0)),
                    })
            seen: set[tuple[int, int, int]] = set()
            for branch_a, branch_b, index in candidates:
                key = (int(branch_a), int(branch_b), int(index))
                if key in seen:
                    continue
                seen.add(key)
                values_a = curve_a["branches"][branch_a]
                values_b = curve_b["branches"][branch_b]
                left_index = index
                right_index = index
                difference_left = values_a[index] - values_b[index]
                if index + 1 in values_a and index + 1 in values_b:
                    difference_right = values_a[index + 1] - values_b[index + 1]
                    if difference_left * difference_right < 0:
                        right_index = index + 1
                if right_index == left_index:
                    rho2 = float(rho2_axis[index])
                    log_a = values_a[index]
                    log_b = values_b[index]
                else:
                    left_rho2 = float(rho2_axis[left_index])
                    right_rho2 = float(rho2_axis[right_index])
                    left_log_a = values_a[left_index]
                    right_log_a = values_a[right_index]
                    left_log_b = values_b[left_index]
                    right_log_b = values_b[right_index]

                    def difference(rho2_value: float) -> float:
                        log_a_value = _curve_value_at(
                            ctx, curve_a, observed[channel_a], branch_a, rho2_value,
                            left_log_a, right_log_a, left_rho2, right_rho2, scan_x,
                        )
                        log_b_value = _curve_value_at(
                            ctx, curve_b, observed[channel_b], branch_b, rho2_value,
                            left_log_b, right_log_b, left_rho2, right_rho2, scan_x,
                        )
                        if log_a_value is None or log_b_value is None:
                            raise ValueError("ambiguous or missing branch during intersection refinement")
                        return log_a_value - log_b_value

                    try:
                        rho2 = float(brentq(
                            difference, left_rho2, right_rho2,
                            xtol=1e-11, rtol=1e-13, maxiter=100,
                        ))
                        log_a = float(_curve_value_at(
                            ctx, curve_a, observed[channel_a], branch_a, rho2,
                            left_log_a, right_log_a, left_rho2, right_rho2, scan_x,
                        ))
                        log_b = float(_curve_value_at(
                            ctx, curve_b, observed[channel_b], branch_b, rho2,
                            left_log_b, right_log_b, left_rho2, right_rho2, scan_x,
                        ))
                    except (ValueError, TypeError) as exc:
                        pair_qc.append({
                            "state": state,
                            "L1_mm": float(ctx["plan"]["sizes"][channel_a]),
                            "L2_mm": float(ctx["plan"]["sizes"][channel_b]),
                            "branch_index_L1": int(branch_a),
                            "branch_index_L2": int(branch_b),
                            "intersection_refinement": "ambiguous_or_failed",
                            "message": str(exc),
                        })
                        continue
                rho1 = float(math.exp((log_a + log_b) / 2.0))
                point = np.log([rho1, rho2])
                all_error = ctx["model"].predict(point) - observed
                pair_error = max(abs(float(all_error[channel_a])), abs(float(all_error[channel_b])))
                intersections.append({
                    "state": state,
                    "L1_mm": float(ctx["plan"]["sizes"][channel_a]),
                    "L2_mm": float(ctx["plan"]["sizes"][channel_b]),
                    "branch_index_L1": int(branch_a),
                    "branch_index_L2": int(branch_b),
                    "rho1": rho1,
                    "rho2": rho2,
                    "pair_max_residual_ohm": float(pair_error),
                    "all_nine_rmse_ohm": float(np.sqrt(np.mean(all_error ** 2))),
                    "within_literature_bounds": _literature_contains(ctx["plan"], rho1, rho2),
                    "method": "sampled branch sign change plus Brent refinement",
                })
    frame = pd.DataFrame(intersections, columns=INTERSECTION_COLUMNS)
    return frame, {
        "pair_checks": pair_qc,
        "intersections_found": len(frame),
        "method": "same tracked branches, sign changes on 601 rho2 samples, Brent refinement",
        "multiple_branch_limitation": bool(any(
            item["qc"]["multiple_roots_observed"] for item in curve_results.values()
        )),
    }


def _minimum_spreads(
    ctx: dict[str, Any], curve_results: dict[tuple[str, int], dict[str, Any]],
    rho2_axis: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for state in ("inhale", "exhale"):
        result[state] = {}
        for _, selected in ctx["best_sets"].sort_values("k").iterrows():
            k = int(selected["k"])
            indices = _best_indices(selected)
            spreads: list[float] = []
            valid_rho2: list[float] = []
            branches_used: list[tuple[int, ...]] = []
            for sample_index, rho2 in enumerate(rho2_axis):
                values: list[float] = []
                branches: list[int] = []
                valid = True
                for channel in indices:
                    curve = curve_results[(state, channel)]
                    roots_here = [
                        branch_values[sample_index]
                        for branch_values in curve["branches"].values()
                        if sample_index in branch_values
                    ]
                    if len(roots_here) != 1:
                        valid = False
                        break
                    branch = next(branch for branch, branch_values in curve["branches"].items()
                                  if sample_index in branch_values)
                    values.append(float(math.exp(roots_here[0])))
                    branches.append(int(branch))
                if valid:
                    spreads.append(float(np.ptp(values)))
                    valid_rho2.append(float(rho2))
                    branches_used.append(tuple(branches))
            if spreads:
                minimum_index = int(np.argmin(spreads))
                result[state][str(k)] = {
                    "k": k,
                    "subset_id": str(selected["subset_id"]),
                    "sizes_mm": str(selected["sizes_mm"]),
                    "minimum_rho1_spread_ohm_m": float(spreads[minimum_index]),
                    "rho2_at_minimum": float(valid_rho2[minimum_index]),
                    "valid_samples": len(spreads),
                    "samples_without_unique_roots_excluded": int(len(rho2_axis) - len(spreads)),
                    "branches_at_minimum": list(branches_used[minimum_index]),
                }
            else:
                result[state][str(k)] = {
                    "k": k,
                    "subset_id": str(selected["subset_id"]),
                    "sizes_mm": str(selected["sizes_mm"]),
                    "minimum_rho1_spread_ohm_m": None,
                    "rho2_at_minimum": None,
                    "valid_samples": 0,
                    "samples_without_unique_roots_excluded": len(rho2_axis),
                    "branches_at_minimum": None,
                }
    return result


def solution_curves(ctx: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build branch-aware isoimpedance curves and their pair intersections."""

    plan = ctx["plan"]
    rho2_axis = np.geomspace(plan["bounds"][1, 0], plan["bounds"][1, 1], CURVE_RHO2_COUNT)
    scan_x = np.linspace(np.log(plan["bounds"][0, 0]), np.log(plan["bounds"][0, 1]),
                         CURVE_RHO1_SCAN_COUNT)
    curve_results: dict[tuple[str, int], dict[str, Any]] = {}
    curve_rows: list[dict[str, Any]] = []
    for state in ("inhale", "exhale"):
        observed = ctx["obs"][f"Z_{state}_hold_ohm"].to_numpy(dtype=float)
        for channel in range(SIZES_COUNT):
            result = _curve_for_channel(ctx, state, channel, float(observed[channel]),
                                        rho2_axis, scan_x)
            curve_results[(state, channel)] = result
            curve_rows.extend(result["rows"])
    curves = pd.DataFrame(curve_rows, columns=CURVE_COLUMNS)
    intersections, intersection_qc = _curve_intersections(
        ctx, curve_results, rho2_axis, scan_x
    )
    monotonicity = _monotonicity_qc(ctx["model"])
    minimum_spreads = _minimum_spreads(ctx, curve_results, rho2_axis)
    rootless = [
        {"state": state, "L_mm": float(plan["sizes"][channel]), **result["qc"]}
        for (state, channel), result in curve_results.items()
        if result["qc"]["samples_without_root"] > 0
    ]
    multiple = [
        {"state": state, "L_mm": float(plan["sizes"][channel]), **result["qc"]}
        for (state, channel), result in curve_results.items()
        if result["qc"]["multiple_roots_observed"]
    ]
    finite_curve_residuals = curves["residual_ohm"].to_numpy(dtype=float)
    qc = {
        "status": "computed_solution_curves",
        "scientific_status": "exploratory_hypothesis_not_validated",
        "model": MODEL,
        "bounds_ohm_m": plan["bounds"].tolist(),
        "literature_bounds_ohm_m": plan["literature_bounds"].tolist(),
        "rho2_samples": CURVE_RHO2_COUNT,
        "rho1_root_scan_samples": CURVE_RHO1_SCAN_COUNT,
        "curves": {
            "requested": 2 * SIZES_COUNT,
            "rows": len(curves),
            "rootless_curves": rootless,
            "multiple_root_curves": multiple,
            "max_root_residual_ohm": float(np.nanmax(np.abs(finite_curve_residuals)))
            if np.isfinite(finite_curve_residuals).any() else None,
        },
        "intersections": intersection_qc,
        "minimum_spreads_by_state_and_k": minimum_spreads,
        "monotonicity": monotonicity,
        "sources": plan["source_records"],
        "observations_source_read_in_place": True,
        "observations_source": plan["source_records"]["observations"]["path"],
        "observations_sha256": plan["source_records"]["observations"]["sha256"],
        "best_sets_source": plan["source_records"]["best_sets"],
        "best_sets_reoptimized": False,
        "root_method": "401 log-rho1 samples; every sign-changing interval refined by Brent",
        "no_root_policy": "rho1 and residual are NaN; no boundary substitution",
        "branch_method": "bounded nearest one-to-one continuation in log-rho1",
        "limitations": [
            "Tangential roots without a sampled sign change are not certified by this scan.",
            "Multiple branches are tracked only across bounded local jumps; ambiguous joins are flagged.",
            "Pair intersections use shared tracked branches and sampled sign changes; tangencies and "
            "branch events are not an exhaustive global intersection proof.",
        ],
    }
    return curves, intersections, qc


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    text = frame.to_csv(index=False, float_format="%.17g", lineterminator="\n", na_rep="NaN")
    _atomic_write_text(path, text)


def run_analysis(out: Path) -> dict[str, Any]:
    """Run merge, interpolation QC, real fits, and branch-aware curves."""

    out = Path(out).resolve()
    merge_worker_outputs(out)
    ctx = load(out)
    interpolation = interpolation_qc(ctx)
    _atomic_write_json(out / "interpolation_qc.json", interpolation)
    real_fits, real_predictions = real_inverse(ctx)
    _write_frame(out / "real_fits.csv", real_fits)
    _write_frame(out / "real_predictions.csv", real_predictions)
    curves, intersections, curve_qc = solution_curves(ctx)
    _write_frame(out / "solution_curves.csv", curves)
    _write_frame(out / "solution_curve_intersections.csv", intersections)
    curve_qc["input_sha256"] = {
        "plan.json": sha256_file(out / "plan.json"),
        "responses.csv": sha256_file(out / "responses.csv"),
        "real_fits.csv": sha256_file(out / "real_fits.csv"),
        "real_predictions.csv": sha256_file(out / "real_predictions.csv"),
    }
    curve_qc["code_sha256"] = {Path(__file__).name: sha256_file(Path(__file__))}
    curve_qc["output_sha256"] = {
        "solution_curves.csv": sha256_file(out / "solution_curves.csv"),
        "solution_curve_intersections.csv": sha256_file(out / "solution_curve_intersections.csv"),
    }
    _atomic_write_json(out / "solution_curves_qc.json", curve_qc)
    return {
        "interpolation_qc": interpolation,
        "real_fits": real_fits,
        "real_predictions": real_predictions,
        "solution_curves_qc": curve_qc,
    }


def _curve_root_for_direct(
    ctx: dict[str, Any], state: str, channel: int, rho2: float,
) -> float | None:
    plan = ctx["plan"]
    observed = float(ctx["obs"].loc[ctx["obs"].index[channel],
                                     f"Z_{state}_hold_ohm"])
    scan_x = np.linspace(np.log(plan["bounds"][0, 0]), np.log(plan["bounds"][0, 1]),
                         CURVE_RHO1_SCAN_COUNT)
    roots = _find_roots_at(ctx["model"], observed, channel, rho2, scan_x)
    return float(math.exp(roots[0])) if len(roots) == 1 else None


def prepare_direct(out: Path) -> pd.DataFrame:
    """Write 16 fit requests and the available unique-root curve requests."""

    out = Path(out).resolve()
    # This may merge the completed FEM partitions and recompute only Python
    # diagnostics.  It never starts MATLAB.
    run_analysis(out)
    ctx = load(out)
    fits = pd.read_csv(out / "real_fits.csv")
    rows: list[dict[str, Any]] = []
    for _, fit in fits.sort_values(["state", "k"], kind="mergesort").iterrows():
        rows.append({
            "point_id": f"fit_{fit['state']}_k{int(fit['k'])}",
            "kind": "directcheck_fit",
            "model": MODEL,
            "rho1": float(fit["rho1_hat"]),
            "rho2": float(fit["rho2_hat"]),
            "L_mm": np.nan,
        })
    skipped = []
    rho2_values = [ctx["plan"]["bounds"][1, 0], 1.0,
                   *ctx["plan"]["literature_bounds"][1], ctx["plan"]["bounds"][1, 1]]
    for state in ("inhale", "exhale"):
        for channel, size in enumerate(ctx["plan"]["sizes"]):
            for r2_index, rho2 in enumerate(rho2_values):
                rho1 = _curve_root_for_direct(ctx, state, channel, float(rho2))
                if rho1 is None:
                    skipped.append(dict(state=state, L_mm=float(size), rho2=float(rho2),
                                        reason="no_unique_root_in_declared_rho1_interval"))
                    continue
                rows.append({
                    "point_id": f"curve_{state}_L{int(size)}_r2{r2_index:02d}",
                    "kind": "directcheck_curve",
                    "model": MODEL,
                    "rho1": rho1,
                    "rho2": float(rho2),
                    "L_mm": float(size),
                })
    frame = pd.DataFrame(rows, columns=REQUEST_COLUMNS)
    request_path = out / "direct_checks.csv"
    if request_path.exists():
        previous = pd.read_csv(request_path)
        pd.testing.assert_frame_equal(previous, frame, check_dtype=False, rtol=1e-12, atol=1e-12)
    else:
        _write_frame(request_path, frame)
    _atomic_write_json(out / "direct_requests_qc.json", {
        "fit_points": 16, "curve_points": len(frame)-16, "skipped": skipped,
        "input_sha256": {name: sha256_file(out/name) for name in ["plan.json", "responses.csv", "real_fits.csv"]},
        "requests_sha256": sha256_file(request_path), "source_sha256": sha256_file(Path(__file__)),
    })
    return frame


def _direct_request_metadata(point_id: str, request: pd.Series) -> tuple[str, int | None, int | None]:
    if point_id.startswith("fit_"):
        match = re.fullmatch(r"fit_(inhale|exhale)_k([2-9])", point_id)
        _require(match is not None, f"unrecognised direct fit point_id: {point_id}")
        return "fit", int(match.group(2)), None
    match = re.fullmatch(r"curve_(inhale|exhale)_L([0-9]+)_r2([0-9]{2})", point_id)
    _require(match is not None, f"unrecognised direct curve point_id: {point_id}")
    state = match.group(1)
    channel = _size_index(float(request["L_mm"]), request.attrs["sizes"], "direct L_mm")
    return state, channel, channel


def check_direct(out: Path) -> dict[str, Any]:
    """Validate completed direct-check workers without any inverse optimisation."""

    out = Path(out).resolve()
    ctx = load(out)
    requests, direct, worker_meta = _validate_worker_pair(
        out, "direct_checks.csv", "direct_checks", ctx["plan"]
    )
    request_qc = _read_json(out / "direct_requests_qc.json")
    _require(request_qc["requests_sha256"] == sha256_file(out/"direct_checks.csv"), "direct requests changed")
    for name, digest in request_qc["input_sha256"].items():
        _require(sha256_file(out/name) == digest, "changed input for direct requests: " + name)
    fit_table = pd.read_csv(out/"real_fits.csv").set_index(["state", "k"])
    _require((requests["kind"] == "directcheck_fit").sum() == 16
             and (requests["kind"] == "directcheck_curve").sum() == request_qc["curve_points"],
             "direct request counts differ from preparation QC")
    sizes = ctx["plan"]["sizes"]
    requests.attrs["sizes"] = sizes
    observed_by_state = {
        state: ctx["obs"][f"Z_{state}_hold_ohm"].to_numpy(dtype=float)
        for state in ("inhale", "exhale")
    }
    checks: list[dict[str, Any]] = []
    spline_fem_errors: list[float] = []
    root_spline_errors: list[float] = []
    root_fem_errors: list[float] = []
    fit_observation_errors: list[float] = []
    absolute = float(ctx["plan"]["acceptance"]["interpolation_absolute_allowance_ohm"])
    relative = float(ctx["plan"]["acceptance"]["interpolation_relative_allowance"])

    for _, request in requests.iterrows():
        point_id = str(request["point_id"])
        group = direct[direct["point_id"] == point_id].copy()
        rho = np.asarray([float(request["rho1"]), float(request["rho2"])])
        predicted = ctx["model"].predict(np.log(rho))
        group = group.sort_values("L_mm", kind="mergesort")
        if request["kind"] == "directcheck_fit":
            _require(len(group) == SIZES_COUNT, f"{point_id}: fit direct response must have nine rows")
            z = group["Z"].to_numpy(dtype=float)
            _require(np.allclose(group["L_mm"].to_numpy(dtype=float), sizes, rtol=0, atol=1e-7),
                     f"{point_id}: fit direct sizes mismatch")
            errors = predicted - z
            tolerance = absolute + relative * np.abs(z)
            passed = bool(np.all(np.abs(errors) <= tolerance))
            spline_fem_errors.extend(np.abs(errors).tolist())
            parts = point_id.split("_")
            state = parts[1]
            k = int(parts[2][1:])
            current_fit = fit_table.loc[(state,k)]
            _require(np.allclose(rho, current_fit[["rho1_hat","rho2_hat"]].to_numpy(dtype=float),
                                 rtol=1e-10, atol=1e-10), "direct point differs from current fit")
            selected = tuple(_best_indices(ctx["best_sets"].loc[
                ctx["best_sets"]["k"] == k
            ].iloc[0]))
            fit_observation_errors.extend(np.abs(predicted[list(selected)]
                                                 - observed_by_state[state][list(selected)]).tolist())
            checks.append({
                "point_id": point_id,
                "kind": "fit",
                "rows": len(group),
                "max_spline_vs_fem_Z_error_ohm": float(np.abs(errors).max()),
                "max_allowed_ohm": float(np.max(tolerance)),
                "spline_vs_fem_passed": passed,
                "selected_FEM_RMSE_ohm": float(np.sqrt(np.mean((z[list(selected)] - observed_by_state[state][list(selected)])**2))),
                "selected_spline_RMSE_ohm": float(np.sqrt(np.mean((predicted[list(selected)] - observed_by_state[state][list(selected)])**2))),
                "fit_observation_max_abs_error_ohm": float(
                    np.abs(predicted[list(selected)] - observed_by_state[state][list(selected)]).max()
                ),
            })
        else:
            _require(len(group) == 1, f"{point_id}: curve direct response must have one row")
            size = float(request["L_mm"])
            channel = _size_index(size, sizes, f"{point_id}.L_mm")
            state = point_id.split("_")[1]
            z = float(group["Z"].iloc[0])
            spline_error = float(predicted[channel] - z)
            observed = float(observed_by_state[state][channel])
            root_spline_error = float(predicted[channel] - observed)
            root_fem_error = float(z - observed)
            tolerance_fem = absolute + relative * abs(z)
            tolerance_root = absolute + relative * abs(observed)
            spline_passed = abs(spline_error) <= tolerance_fem
            root_passed = abs(root_spline_error) <= tolerance_root and abs(root_fem_error) <= tolerance_root
            spline_fem_errors.append(abs(spline_error))
            root_spline_errors.append(abs(root_spline_error))
            root_fem_errors.append(abs(root_fem_error))
            checks.append({
                "point_id": point_id,
                "kind": "curve",
                "state": state,
                "L_mm": size,
                "rho1": float(request["rho1"]),
                "rho2": float(request["rho2"]),
                "max_spline_vs_fem_Z_error_ohm": abs(spline_error),
                "root_spline_residual_ohm": root_spline_error,
                "root_fem_residual_ohm": root_fem_error,
                "spline_vs_fem_allowed_ohm": tolerance_fem,
                "root_vs_measurement_allowed_ohm": tolerance_root,
                "spline_vs_fem_passed": bool(spline_passed),
                "root_vs_measurement_passed": bool(root_passed),
            })

    max_spline_fem = max(spline_fem_errors) if spline_fem_errors else None
    max_root_spline = max(root_spline_errors) if root_spline_errors else None
    max_root_fem = max(root_fem_errors) if root_fem_errors else None
    root_passed = bool(all(item.get("root_vs_measurement_passed", True) for item in checks))
    passed = bool(all(item.get("spline_vs_fem_passed", False) for item in checks)
                  and root_passed)
    result = {
        "status": "computed_direct_comparison" if passed else "computed_direct_comparison_qc_failed",
        "complete": True,
        "passed": passed,
        "model": MODEL,
        "points": len(requests),
        "rows": len(direct),
        "fit_points": 16,
        "curve_points": int(request_qc["curve_points"]),
        "max_spline_vs_fem_Z_error_ohm": max_spline_fem,
        "max_root_vs_measurement_spline_error_ohm": max_root_spline,
        "max_root_vs_measurement_fem_error_ohm": max_root_fem,
        "max_selected_rmse_change_ohm": max(abs(c["selected_FEM_RMSE_ohm"]-c["selected_spline_RMSE_ohm"])
                                            for c in checks if c["kind"]=="fit"),
        "max_fit_observation_error_ohm": max(fit_observation_errors)
        if fit_observation_errors else None,
        "spline_vs_fem_passed": bool(all(item.get("spline_vs_fem_passed", False) for item in checks)),
        "root_vs_measurement_passed": root_passed,
        "acceptance": {
            "absolute_allowance_ohm": absolute,
            "relative_allowance": relative,
            "formula": "absolute_allowance + relative_allowance * abs(Zref)",
            "reference_for_spline_vs_fem": "direct FEM Z",
            "reference_for_root_vs_measurement": "observed Z",
        },
        "checks": checks,
        "input_sha256": {
            "plan.json": sha256_file(out / "plan.json"),
            "responses.csv": sha256_file(out / "responses.csv"),
            "direct_checks.csv": sha256_file(out / "direct_checks.csv"),
            "observations": ctx["plan"]["source_records"]["observations"]["sha256"],
            "real_fits.csv": sha256_file(out/"real_fits.csv"),
        },
        "worker_validation": worker_meta,
        "analysis_source_sha256": sha256_file(Path(__file__)),
        "direct_optimization_repeated": False,
        "new_full_FEM_optimization": False,
        "observations_source_read_in_place": True,
        "independent_tissue_truth": False,
    }
    _atomic_write_json(out / "direct_check_qc.json", result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="lateral_resistivity_range_20260916 output directory")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-direct", action="store_true",
                      help="prepare direct_checks.csv; does not start MATLAB")
    mode.add_argument("--check-direct", action="store_true",
                      help="check completed direct-check worker files")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    out = args.out.resolve()
    try:
        if args.prepare_direct:
            frame = prepare_direct(out)
            print(f"prepared {len(frame)} direct checks: {out / 'direct_checks.csv'}")
        elif args.check_direct:
            result = check_direct(out)
            print(json.dumps({key: result[key] for key in ["status", "complete", "passed", "points", "rows",
                "fit_points", "curve_points", "max_spline_vs_fem_Z_error_ohm", "max_selected_rmse_change_ohm"]},
                ensure_ascii=False, indent=2))
        else:
            result = run_analysis(out)
            print(json.dumps({
                "status": "analysis_completed",
                "interpolation_passed": result["interpolation_qc"]["passed"],
                "solution_curve_rows": result["solution_curves_qc"]["curves"]["rows"],
                "intersection_count": result["solution_curves_qc"]["intersections"]["intersections_found"],
            }, ensure_ascii=False, indent=2))
    except (ContractError, FileNotFoundError, OSError, ValueError, np.linalg.LinAlgError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
