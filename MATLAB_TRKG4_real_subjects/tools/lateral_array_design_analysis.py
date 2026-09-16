"""Read-only, scenario-only design analysis for the lateral array sizes.

The input responses are precomputed FEM/model results.  This file does not
run FEM, copy observations, or assign an experimental noise distribution.
The quantities called ``unit`` below are the response to a hypothetical one
ohm independent error and are not CRLBs or confidence intervals.

Usage::

    python tools/lateral_array_design_analysis.py --out OUT plan.json responses.csv

``observations_source`` in the plan is resolved relative to
``MATLAB_TRKG4_real_subjects`` and checked against ``observations_sha256``.
The source is read in place and never copied to ``OUT``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import least_squares


MODELS = ("variable_transverse", "reference")
GRID_MODEL = "variable_transverse"
SIZES_COUNT = 9
GRID_AXIS_COUNT = 7
VALIDATION_COUNT = 16
RESPONSE_COLUMNS = (
    "point_id", "kind", "model", "rho1", "rho2", "L_mm", "Z",
    "dZ_dlogrho1", "dZ_dlogrho2", "relative_residual",
    "reciprocity_abs", "elapsed_seconds",
)
NUMERIC_RESPONSE_COLUMNS = RESPONSE_COLUMNS[3:]
SCENARIO_SIGMAS_OHM = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
SCENARIO_THRESHOLDS_PCT = (1.0, 5.0, 10.0, 20.0)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot encode {type(value)!r}")


def save_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                   default=_json_default) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_numeric_frame(frame: pd.DataFrame, columns: Iterable[str], label: str) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        _require(np.isfinite(result[column].to_numpy(dtype=float)).all(),
                 f"{label}: non-finite value in {column}")
    return result


def _nearest(value: float, axis: np.ndarray, label: str) -> int:
    index = int(np.argmin(np.abs(axis - value)))
    _require(bool(np.isclose(value, axis[index], rtol=2e-7, atol=1e-10)),
             f"{label}={value!r} is not a declared grid value")
    return index


def _size_index(value: float, sizes: np.ndarray) -> int:
    index = int(np.argmin(np.abs(sizes - value)))
    _require(bool(np.isclose(value, sizes[index], rtol=0.0, atol=1e-7)),
             f"L_mm={value!r} is not a declared size")
    return index


def load_plan(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"plan file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid plan JSON: {exc}") from exc
    _require(isinstance(raw, dict), "plan JSON must be an object")
    try:
        bounds = np.asarray(raw["bounds_ohm_m"], dtype=float)
        sizes = np.asarray(raw["sizes_mm"], dtype=float)
        grid_rho1 = np.asarray(raw["grid_rho1"], dtype=float)
        grid_rho2 = np.asarray(raw["grid_rho2"], dtype=float)
        source = str(raw["observations_source"])
        source_sha = str(raw["observations_sha256"]).lower()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "plan requires bounds_ohm_m, sizes_mm, grid_rho1, grid_rho2, "
            "observations_source, and observations_sha256"
        ) from exc

    _require(bounds.shape == (2, 2) and np.isfinite(bounds).all(),
             "bounds_ohm_m must be finite with shape (2, 2)")
    _require((bounds[:, 0] > 0).all() and (bounds[:, 1] > bounds[:, 0]).all(),
             "resistivity bounds must be positive and increasing")
    _require(sizes.shape == (SIZES_COUNT,) and np.isfinite(sizes).all(),
             "sizes_mm must contain nine finite values")
    _require(np.all(np.diff(sizes) > 0) and len(set(sizes.tolist())) == SIZES_COUNT,
             "sizes_mm must be strictly increasing and unique")
    for name, values in (("grid_rho1", grid_rho1), ("grid_rho2", grid_rho2)):
        _require(values.shape == (GRID_AXIS_COUNT,) and np.isfinite(values).all(),
                 f"{name} must contain seven finite values")
        _require((values > 0).all() and np.all(np.diff(values) > 0),
                 f"{name} must be positive and increasing")
        steps = np.diff(np.log(values))
        _require(np.allclose(steps, steps[0], rtol=2e-5, atol=2e-8),
                 f"{name} must be logarithmically spaced")
    _require((grid_rho1 >= bounds[0, 0] * (1 - 2e-7)).all()
             and (grid_rho1 <= bounds[0, 1] * (1 + 2e-7)).all(),
             "grid_rho1 lies outside bounds_ohm_m")
    _require((grid_rho2 >= bounds[1, 0] * (1 - 2e-7)).all()
             and (grid_rho2 <= bounds[1, 1] * (1 + 2e-7)).all(),
             "grid_rho2 lies outside bounds_ohm_m")
    source_path = Path(source)
    _require(source.strip() and not source_path.is_absolute(),
             "observations_source must be a non-empty relative path")
    _require(len(source_sha) == 64 and all(c in "0123456789abcdef" for c in source_sha),
             "observations_sha256 must be a SHA-256 hex string")
    return {
        "raw": raw, "bounds": bounds, "sizes": sizes,
        "grid_rho1": grid_rho1, "grid_rho2": grid_rho2,
        "observations_source": source, "observations_sha256": source_sha,
    }


def read_declared_observations(repo_root: Path, plan: dict[str, Any]) -> tuple[pd.DataFrame, str, str]:
    """Read the declared source in place; never make a local copy."""

    matlab_root = (repo_root / "MATLAB_TRKG4_real_subjects").resolve()
    relative = Path(plan["observations_source"].replace("\\", "/"))
    source = (matlab_root / relative).resolve()
    try:
        source.relative_to(matlab_root)
    except ValueError as exc:
        raise ValueError("observations_source must stay below MATLAB_TRKG4_real_subjects") from exc
    if not source.is_file():
        raise FileNotFoundError(f"observations_source not found: {plan['observations_source']}")
    actual_sha = sha256_file(source)
    _require(actual_sha == plan["observations_sha256"],
             "observations_sha256 does not match the declared source")
    try:
        frame = pd.read_csv(source)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read observations_source: {plan['observations_source']}") from exc
    return frame, plan["observations_source"].replace("\\", "/"), actual_sha


def normalise_observations(frame: pd.DataFrame, sizes: np.ndarray) -> pd.DataFrame:
    aliases = {
        "L_mm": ("L_mm", "size_mm"),
        "inhale": ("Z_inhale_hold_ohm", "z_inhale_hold_ohm", "Z_inhale_ohm", "z_inhale_ohm"),
        "exhale": ("Z_exhale_hold_ohm", "z_exhale_hold_ohm", "Z_exhale_ohm", "z_exhale_ohm"),
    }
    names = {}
    for target, candidates in aliases.items():
        names[target] = next((name for name in candidates if name in frame.columns), None)
        _require(names[target] is not None, f"observations.csv is missing {target}")
    result = frame[[names["L_mm"], names["inhale"], names["exhale"]]].copy()
    result.columns = ["L_mm", "Z_inhale_hold_ohm", "Z_exhale_hold_ohm"]
    result = _finite_numeric_frame(result, result.columns, "observations.csv")
    result["size_index"] = [_size_index(float(x), sizes) for x in result["L_mm"]]
    _require(len(result) == len(sizes) and result["size_index"].is_unique,
             "observations.csv must contain one row for each declared size")
    return result.sort_values("size_index").reset_index(drop=True)


def _execution_sidecar(path: Path) -> str:
    sidecar = path.with_name(path.name + ".execution.json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"response execution sidecar not found: {sidecar.name}")
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid response execution sidecar: {sidecar.name}") from exc
    _require(isinstance(payload, dict) and payload.get("complete") is True,
             f"response execution is not complete: {sidecar.name}")
    _require(payload.get("responses_sha256")==sha256_file(path), "response checksum mismatch")
    return sidecar.name


def _state(group: pd.DataFrame, sizes: np.ndarray, label: str) -> dict[str, Any]:
    ordered = group.sort_values("size_index")
    _require(len(ordered) == len(sizes)
             and np.array_equal(ordered["size_index"].to_numpy(), np.arange(len(sizes))),
             f"{label} must contain all nine sizes once")
    for column in ("rho1", "rho2"):
        values = ordered[column].to_numpy(dtype=float)
        _require(np.allclose(values, values[0], rtol=2e-7, atol=1e-10),
                 f"{label}: {column} changes within point_id")
    return {
        "point_id": str(ordered["point_id"].iloc[0]),
        "rho1": float(ordered["rho1"].iloc[0]), "rho2": float(ordered["rho2"].iloc[0]),
        "Z": ordered["Z"].to_numpy(dtype=float),
        "J": ordered[["dZ_dlogrho1", "dZ_dlogrho2"]].to_numpy(dtype=float),
    }


def load_response_tables(path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"responses file not found: {path}")
    execution_sidecar = _execution_sidecar(path)
    try:
        frame = pd.read_csv(path, dtype={"point_id": str})
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read responses CSV: {path.name}") from exc
    missing = [column for column in RESPONSE_COLUMNS if column not in frame.columns]
    _require(not missing, "responses.csv is missing: " + ", ".join(missing))
    _require(len(frame) > 0, "responses.csv is empty")
    frame = _finite_numeric_frame(frame, NUMERIC_RESPONSE_COLUMNS, "responses.csv")
    frame["point_id"] = frame["point_id"].astype(str)
    _require(set(frame["kind"].unique()) == {"grid", "validation"},
             "responses.kind must contain exactly grid and validation")
    _require(set(frame["model"].unique()).issubset(set(MODELS)),
             "responses.model contains an unsupported value")
    grid_rows = frame[frame["kind"].eq("grid")]
    _require(set(grid_rows["model"].unique()) == {GRID_MODEL},
             "grid rows must use model variable_transverse only")
    frame["size_index"] = [_size_index(float(x), plan["sizes"]) for x in frame["L_mm"]]
    grid_rows = frame[frame["kind"].eq("grid")]
    key = ["kind", "model", "point_id", "size_index"]
    _require(not frame.duplicated(key).any(), "responses has duplicate point/model/size rows")

    grid_z = np.full((7, 7, 9), np.nan)
    grid_j = np.full((7, 7, 9, 2), np.nan)
    grid_states = []
    seen: set[tuple[int, int]] = set()
    _require(grid_rows["point_id"].nunique() == 49,
             "grid must contain exactly 49 point_id states")
    for point_id, group in grid_rows.groupby("point_id", sort=False):
        state = _state(group, plan["sizes"], f"grid point_id={point_id}")
        i = _nearest(state["rho1"], plan["grid_rho1"], "grid rho1")
        j = _nearest(state["rho2"], plan["grid_rho2"], "grid rho2")
        _require((i, j) not in seen, f"duplicate grid state {(i, j)}")
        seen.add((i, j))
        grid_z[i, j] = state["Z"]
        grid_j[i, j] = state["J"]
        grid_states.append(state)
    _require(len(seen) == 49 and np.isfinite(grid_z).all() and np.isfinite(grid_j).all(),
             "grid must contain every 7 by 7 state and nine finite channels")

    validation = {}
    val_rows = frame[frame["kind"].eq("validation")]
    for model in MODELS:
        model_rows = val_rows[val_rows["model"].eq(model)]
        _require(model_rows["point_id"].nunique() == VALIDATION_COUNT,
                 f"validation/{model} must contain 16 point_id states")
        states = [_state(group, plan["sizes"], f"validation/{model}/{point_id}")
                  for point_id, group in model_rows.groupby("point_id", sort=False)]
        validation[model] = states
    return {
        "grid_z": grid_z, "grid_j": grid_j, "grid_states": grid_states,
        "grid_log_points": np.asarray(list(itertools.product(
            np.log(plan["grid_rho1"]), np.log(plan["grid_rho2"]))), dtype=float),
        "validation": validation, "response_file_name": path.name,
        "response_execution_sidecar": execution_sidecar,
    }


class SplineSurrogate:
    def __init__(self, plan: dict[str, Any], grid_z: np.ndarray) -> None:
        x, y = np.log(plan["grid_rho1"]), np.log(plan["grid_rho2"])
        self.log_x, self.log_y = x, y
        self.splines = [RectBivariateSpline(x, y, grid_z[:, :, i], kx=3, ky=3, s=0.0)
                        for i in range(9)]

    @staticmethod
    def _ev(spline: RectBivariateSpline, x: float, y: float, dx: int = 0, dy: int = 0) -> float:
        return float(np.asarray(spline.ev(x, y, dx=dx, dy=dy)).reshape(-1)[0])

    def predict(self, log_rho: np.ndarray, indices: Iterable[int] | None = None) -> np.ndarray:
        selected = range(9) if indices is None else indices
        x, y = float(log_rho[0]), float(log_rho[1])
        return np.asarray([self._ev(self.splines[i], x, y) for i in selected], dtype=float)

    def jacobian(self, log_rho: np.ndarray, indices: Iterable[int] | None = None) -> np.ndarray:
        selected = range(9) if indices is None else indices
        x, y = float(log_rho[0]), float(log_rho[1])
        return np.asarray([[self._ev(self.splines[i], x, y, dx=1),
                            self._ev(self.splines[i], x, y, dy=1)]
                           for i in selected], dtype=float)


def _stats(values: np.ndarray) -> dict[str, Any]:
    values = np.abs(np.asarray(values, dtype=float).reshape(-1))
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "max_abs": None, "rmse": None, "mean_abs": None,
                "median_abs": None, "p95_abs": None}
    return {"n": int(len(values)), "max_abs": float(values.max()),
            "rmse": float(np.sqrt(np.mean(values ** 2))),
            "mean_abs": float(values.mean()), "median_abs": float(np.median(values)),
            "p95_abs": float(np.percentile(values, 95))}


def interpolation_qc(plan: dict[str, Any], tables: dict[str, Any], spline: SplineSurrogate) -> dict[str, Any]:
    z_errors, j1_errors, j2_errors, point_rows = [], [], [], []
    for state in tables["validation"][GRID_MODEL]:
        log_rho = np.log([state["rho1"], state["rho2"]])
        inside = (spline.log_x[0] <= log_rho[0] <= spline.log_x[-1]
                  and spline.log_y[0] <= log_rho[1] <= spline.log_y[-1])
        _require(inside, f"validation point {state['point_id']} is outside spline grid")
        error_z = spline.predict(log_rho) - state["Z"]
        error_j = spline.jacobian(log_rho) - state["J"]
        z_errors.append(error_z); j1_errors.append(error_j[:, 0]); j2_errors.append(error_j[:, 1])
        point_rows.append({
            "point_id": state["point_id"], "rho1": state["rho1"], "rho2": state["rho2"],
            "Z_max_abs_error_ohm": float(np.abs(error_z).max()),
            "dZ_dlogrho1_max_abs_error_ohm": float(np.abs(error_j[:, 0]).max()),
            "dZ_dlogrho2_max_abs_error_ohm": float(np.abs(error_j[:, 1]).max()),
            "J1_relative_l2_error":float(np.linalg.norm(error_j[:,0])/np.linalg.norm(state["J"][:,0])),
            "J2_relative_l2_error":float(np.linalg.norm(error_j[:,1])/np.linalg.norm(state["J"][:,1])),
        })
    z_errors, j1_errors, j2_errors = map(np.asarray, (z_errors, j1_errors, j2_errors))
    knot_errors = [spline.predict(np.log([s["rho1"], s["rho2"]])) - s["Z"]
                   for s in tables["grid_states"]]
    return {
        "status": "computed_no_acceptance_decision", "passed": None,
        "acceptance_thresholds": None, "scientific_status": "surrogate_qc_only",
        "spline": {"class": "scipy.interpolate.RectBivariateSpline", "coordinates": "log(rho1), log(rho2)",
                    "smoothing": 0.0, "channels": 9, "grid_points": 49},
        "validation": {
            "model": GRID_MODEL, "points": 16, "inside_grid": True,
            "Z_abs_error_ohm": _stats(z_errors),
            "dZ_dlogrho1_abs_error_ohm": _stats(j1_errors),
            "dZ_dlogrho2_abs_error_ohm": _stats(j2_errors),
            "per_channel": [
                {"L_mm": float(plan["sizes"][i]), "Z_abs_error_ohm": _stats(z_errors[:, i]),
                 "dZ_dlogrho1_abs_error_ohm": _stats(j1_errors[:, i]),
                 "dZ_dlogrho2_abs_error_ohm": _stats(j2_errors[:, i])}
                for i in range(9)
            ],
            "points_summary": point_rows,
        },
        "grid_knot_self_check": {"Z_abs_error_ohm": _stats(np.asarray(knot_errors))},
        "reference_validation": {
            "model": "reference", "points": 16, "direct_Z_and_J_available": True,
            "surrogate_built": False, "note": "direct CT/FEM target for transfer checks",
        },
        "not_claimed": ["Gaussian uncertainty", "CRLB", "confidence interval", "experimental accuracy"],
    }


def _subset_id(indices: Iterable[int], sizes: np.ndarray) -> str:
    return "-".join(f"{int(sizes[i]):g}" for i in indices)


def _sizes_text(indices: Iterable[int], sizes: np.ndarray) -> str:
    return "|".join(f"{int(sizes[i]):g}" for i in indices)


def _score(indices: tuple[int, ...], grid_j: np.ndarray) -> dict[str, Any]:
    unit_sds, kappas, gains, ranks = [], [], [], []
    for state_j in grid_j.reshape(-1, 9, 2):
        matrix = state_j[list(indices)]
        singular = np.linalg.svd(matrix, compute_uv=False)
        rank = int(np.linalg.matrix_rank(matrix)); ranks.append(rank)
        gains.append(np.linalg.norm(matrix, axis=0))
        if rank < 2 or singular[-1] <= 0:
            unit_sds.append([np.inf, np.inf]); kappas.append(np.inf)
        else:
            unit_sds.append(np.sqrt(np.diag(np.linalg.inv(matrix.T @ matrix))))
            kappas.append(float(singular[0] / singular[-1]))
    unit_sds, gains = np.asarray(unit_sds), np.asarray(gains)
    return {
        "worst_sigma_logrho1_unit_ohm_error": float(unit_sds[:, 0].max()),
        "worst_sigma_logrho2_unit_ohm_error": float(unit_sds[:, 1].max()),
        "worst_max_unit_logrho_error": float(unit_sds.max()),
        "worst_kappa": float(np.max(kappas)), "g1": float(gains[:, 0].min()),
        "g2": float(gains[:, 1].min()), "g1_max": float(gains[:, 0].max()),
        "g2_max": float(gains[:, 1].max()), "rank2_min": int(np.min(ranks)),
    }


def enumerate_subsets(plan: dict[str, Any], tables: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    sizes, all_indices = plan["sizes"], tuple(range(9))
    full = _score(all_indices, tables["grid_j"])
    rows = []
    for k in range(2, 10):
        for indices in itertools.combinations(all_indices, k):
            score = _score(indices, tables["grid_j"])
            rows.append({
                "k": k, "subset_id": _subset_id(indices, sizes),
                "sizes_mm": _sizes_text(indices, sizes),
                "sizes_mm_json": json.dumps([int(sizes[i]) for i in indices]),
                "subset_indices_json": json.dumps(list(indices)), **score,
                "g1_relative_full9": score["g1"] / full["g1"],
                "g2_relative_full9": score["g2"] / full["g2"],
                "attenuation_relative_full9": min(score["g1"] / full["g1"], score["g2"] / full["g2"]),
                "uncertainty_inflation_relative_full9": score["worst_max_unit_logrho_error"] / full["worst_max_unit_logrho_error"],
            })
    _require(len(rows) == 502, f"expected 502 subsets, got {len(rows)}")
    scores = pd.DataFrame(rows).sort_values(
        ["worst_max_unit_logrho_error", "worst_kappa", "g1", "g2", "subset_id"],
        ascending=[True, True, False, False, True], kind="mergesort").reset_index(drop=True)
    scores["global_rank"] = np.arange(1, len(scores) + 1)
    scores["rank_within_count"] = scores.groupby("k", sort=False).cumcount() + 1

    best_rows = []
    for k in range(2, 10):
        group = scores[scores["k"].eq(k)].sort_values(
            ["worst_max_unit_logrho_error", "subset_id"], kind="mergesort")
        best_rows.append(group.iloc[0].copy())
    best = pd.DataFrame(best_rows).sort_values("k").reset_index(drop=True)
    best["worst_score"] = best["worst_max_unit_logrho_error"]
    monotonic = [None]
    for previous, current in zip(best["worst_score"], best["worst_score"].iloc[1:]):
        monotonic.append(bool(current <= previous + 1e-10 * max(1.0, abs(previous))))
    best["worst_score_nonincreasing_vs_previous"] = monotonic
    monotonic_check = bool(all(x is None or x for x in monotonic))
    _require(monotonic_check, "Optimal information must not decrease when adding measurements")
    return scores, best, {"full9": full, "monotonic": monotonic_check,
                          "sequence": [float(x) for x in best["worst_score"]]}


def _nearest_grid_start(target: np.ndarray, indices: tuple[int, ...], grid_z: np.ndarray, grid_log: np.ndarray) -> np.ndarray:
    residual = grid_z.reshape(-1, 9)[:, list(indices)] - target[list(indices)]
    return grid_log[int(np.argmin(np.sum(residual ** 2, axis=1)))].copy()


def _fit(spline: SplineSurrogate, indices: tuple[int, ...], target: np.ndarray,
         bounds: np.ndarray, grid_z: np.ndarray, grid_log: np.ndarray) -> dict[str, Any]:
    lower, upper = np.log(bounds[:, 0]), np.log(bounds[:, 1])
    starts = [(_nearest_grid_start(target, indices, grid_z, grid_log), "best_grid"),
              ((lower + upper) / 2.0, "center")]
    y = target[list(indices)]; fits = []
    for start, label in starts:
        start = np.clip(start, lower, upper)
        result = least_squares(lambda x: spline.predict(x, indices) - y, start,
                               jac=lambda x: spline.jacobian(x, indices),
                               bounds=(lower, upper), method="trf",
                               ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=1000)
        fits.append((result, label, start))
    result, label, start = min(fits, key=lambda x: (float(x[0].cost), int(not x[0].success)))
    rho = np.exp(result.x); residual = spline.predict(result.x, indices) - y
    return {"result": result, "start_label": label, "start_log_rho": start, "rho": rho,
            "residual": residual,
            "lower_hit": np.isclose(rho, bounds[:, 0], rtol=2e-7, atol=1e-10),
            "upper_hit": np.isclose(rho, bounds[:, 1], rtol=2e-7, atol=1e-10)}


def _validation_row(fit: dict[str, Any], selected: pd.Series, target: str, model: str,
                    state: dict[str, Any]) -> dict[str, Any]:
    result, rho = fit["result"], fit["rho"]
    truth = np.asarray([state["rho1"], state["rho2"]])
    error = 100.0 * (rho / truth - 1.0)
    residual = fit["residual"]
    k = int(selected.k)
    return {
        "target": target, "target_model": model, "point_id": state["point_id"], "k": k,
        "subset_id": selected.subset_id, "sizes_mm": selected.sizes_mm,
        "rho1_true_for_validation_only": float(truth[0]), "rho2_true_for_validation_only": float(truth[1]),
        "rho1_hat": float(rho[0]), "rho2_hat": float(rho[1]),
        "rho1_error_pct": float(error[0]), "rho2_error_pct": float(error[1]),
        "rho1_abs_error_pct": float(abs(error[0])), "rho2_abs_error_pct": float(abs(error[1])),
        "bounds_hit_rho1_lower": bool(fit["lower_hit"][0]), "bounds_hit_rho1_upper": bool(fit["upper_hit"][0]),
        "bounds_hit_rho2_lower": bool(fit["lower_hit"][1]), "bounds_hit_rho2_upper": bool(fit["upper_hit"][1]),
        "bounds_hit_any": bool(np.any(fit["lower_hit"] | fit["upper_hit"])),
        "fit_resid_rmse_ohm": float(np.sqrt(np.mean(residual ** 2))),
        "fit_resid_norm_ohm": float(np.linalg.norm(residual)),
        "fit_resid_max_abs_ohm": float(np.abs(residual).max()), "nfev": int(result.nfev),
        "optimality": float(result.optimality), "success": bool(result.success),
        "start_used": fit["start_label"], "zero_residual_is_not_validation": k == 2,
        "validation_status": "two_obs_two_params_not_validation" if k == 2 else "conditional_validation_metric",
        "independent_tissue_truth": False,
    }


def validation_inverse(plan: dict[str, Any], tables: dict[str, Any], spline: SplineSurrogate,
                        best: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for target, model in (("self_model4", GRID_MODEL), ("ct_reference", "reference")):
        for state in tables["validation"][model]:
            for _, selected in best.iterrows():
                indices = tuple(json.loads(selected["subset_indices_json"]))
                log_rho = np.log([state["rho1"], state["rho2"]])
                _require(spline.log_x[0] <= log_rho[0] <= spline.log_x[-1]
                         and spline.log_y[0] <= log_rho[1] <= spline.log_y[-1],
                         f"validation point {state['point_id']} is outside spline grid")
                rows.append(_validation_row(
                    _fit(spline, indices, state["Z"], plan["bounds"], tables["grid_z"], tables["grid_log_points"]),
                    selected, target, model, state))
    inverse = pd.DataFrame(rows)
    summaries = []
    for (target, k), group in inverse.groupby(["target", "k"], sort=True):
        first = group.iloc[0]
        summaries.append({
            "target": target, "target_model": first["target_model"], "k": int(k),
            "subset_id": first["subset_id"], "sizes_mm": first["sizes_mm"],
            "n_validation_points": int(len(group)),
            "mean_signed_rho1_error_pct": float(group["rho1_error_pct"].mean()),
            "mean_signed_rho2_error_pct": float(group["rho2_error_pct"].mean()),
            "mean_abs_rho1_error_pct": float(group["rho1_abs_error_pct"].mean()),
            "mean_abs_rho2_error_pct": float(group["rho2_abs_error_pct"].mean()),
            "max_abs_rho1_error_pct": float(group["rho1_abs_error_pct"].max()),
            "max_abs_rho2_error_pct": float(group["rho2_abs_error_pct"].max()),
            "mean_fit_resid_rmse_ohm": float(group["fit_resid_rmse_ohm"].mean()),
            "max_fit_resid_rmse_ohm": float(group["fit_resid_rmse_ohm"].max()),
            "bounds_hit_count": int(group["bounds_hit_any"].sum()),
            "all_fits_success": bool(group["success"].all()),
            "zero_residual_is_not_validation": int(k) == 2,
            "status": "conditional_validation_metrics_only", "independent_tissue_truth": False,
        })
    return inverse, pd.DataFrame(summaries).sort_values(["target", "k"]).reset_index(drop=True)


def real_inverse(plan: dict[str, Any], tables: dict[str, Any], spline: SplineSurrogate,
                 best: pd.DataFrame, observations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fits, predictions, all_indices = [], [], tuple(range(9))
    for state_name, column in (("inhale", "Z_inhale_hold_ohm"), ("exhale", "Z_exhale_hold_ohm")):
        target = observations[column].to_numpy(dtype=float)
        for _, selected in best.iterrows():
            indices = tuple(json.loads(selected["subset_indices_json"]))
            fit = _fit(spline, indices, target, plan["bounds"], tables["grid_z"], tables["grid_log_points"])
            predicted = spline.predict(np.log(fit["rho"])); residual = predicted - target
            heldout = [i for i in all_indices if i not in indices]
            heldout_rmse: float | str = float(np.sqrt(np.mean(residual[heldout] ** 2))) if int(selected.k) < 9 else "None"
            fits.append({
                "state": state_name, "k": int(selected.k), "subset_id": selected.subset_id,
                "sizes_mm": selected.sizes_mm, "rho1_hat": float(fit["rho"][0]), "rho2_hat": float(fit["rho"][1]),
                "bounds_hit_rho1_lower": bool(fit["lower_hit"][0]), "bounds_hit_rho1_upper": bool(fit["upper_hit"][0]),
                "bounds_hit_rho2_lower": bool(fit["lower_hit"][1]), "bounds_hit_rho2_upper": bool(fit["upper_hit"][1]),
                "fit_resid_rmse_ohm": float(np.sqrt(np.mean(residual[list(indices)] ** 2))),
                "fit_resid_norm_ohm": float(np.linalg.norm(residual[list(indices)])),
                "all_size_rmse_ohm": float(np.sqrt(np.mean(residual ** 2))),
                "heldout_rmse_ohm": heldout_rmse, "heldout_count": len(heldout),
                "nfev": int(fit["result"].nfev), "success": bool(fit["result"].success),
                "start_used": fit["start_label"], "independent_tissue_truth": False,
                "status": "conditional_real_inverse_no_tissue_truth",
            })
            for i, size in enumerate(plan["sizes"]):
                predictions.append({
                    "state": state_name, "k": int(selected.k), "subset_id": selected.subset_id,
                    "L_mm": float(size), "selected": bool(i in indices),
                    "observed_Z_ohm": float(target[i]), "predicted_Z_ohm": float(predicted[i]),
                    "residual_ohm": float(residual[i]), "independent_tissue_truth": False,
                })
    return pd.DataFrame(fits), pd.DataFrame(predictions)


def minimum_k_table(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sigma in SCENARIO_SIGMAS_OHM:
        for threshold in SCENARIO_THRESHOLDS_PCT:
            chosen = None
            for k in range(2, 10):
                candidates = scores[scores["k"].eq(k)].sort_values(
                    ["worst_max_unit_logrho_error", "subset_id"], kind="mergesort")
                allowed = threshold / (100.0 * sigma)
                passing = candidates[np.isfinite(candidates["worst_max_unit_logrho_error"])
                                     & candidates["worst_max_unit_logrho_error"].le(allowed)]
                if len(passing):
                    chosen = passing.iloc[0]; chosen_k = k; break
            if chosen is None:
                rows.append({"sigma_ohm": sigma, "threshold_pct": threshold, "minimum_k": "None",
                             "selected_subset_id": "None", "selected_sizes_mm": "None",
                             "scenario_linearized_log_parameter_change_pct": "None",
                             "worst_max_unit_logrho_error": "None", "worst_kappa": "None",
                             "g1": "None", "g2": "None", "attenuation_relative_full9": "None",
                             "meets_scenario_threshold": False})
            else:
                score = float(chosen["worst_max_unit_logrho_error"])
                rows.append({"sigma_ohm": sigma, "threshold_pct": threshold, "minimum_k": chosen_k,
                             "selected_subset_id": chosen["subset_id"], "selected_sizes_mm": chosen["sizes_mm"],
                             "scenario_linearized_log_parameter_change_pct": 100.0 * sigma * score,
                             "worst_max_unit_logrho_error": score, "worst_kappa": chosen["worst_kappa"],
                             "g1": chosen["g1"], "g2": chosen["g2"],
                             "attenuation_relative_full9": chosen["attenuation_relative_full9"],
                             "meets_scenario_threshold": True})
    return pd.DataFrame(rows)


def run_analysis(out: Path, plan_path: Path, responses_path: Path, repo_root: Path) -> list[str]:
    # All validation, including the response sidecar and observations hash,
    # happens before the output directory is created.
    plan = load_plan(plan_path)
    tables = load_response_tables(responses_path, plan)
    raw_observations, observations_rel, observations_sha = read_declared_observations(repo_root, plan)
    observations = normalise_observations(raw_observations, plan["sizes"])
    out.mkdir(parents=True, exist_ok=True)

    spline = SplineSurrogate(plan, tables["grid_z"])
    qc = interpolation_qc(plan, tables, spline)
    scores, best, subset_meta = enumerate_subsets(plan, tables)
    inverse, summary = validation_inverse(plan, tables, spline, best)
    real_fits, real_predictions = real_inverse(plan, tables, spline, best, observations)
    min_counts = minimum_k_table(scores)

    files = {
        "subset_scores.csv": scores, "best_by_count.csv": best,
        "inverse_validation.csv": inverse, "summary_by_count.csv": summary,
        "real_fits.csv": real_fits, "real_predictions.csv": real_predictions,
        "minimum_k_by_threshold.csv": min_counts,
    }
    for name, frame in files.items():
        frame.to_csv(out / name, index=False)
    save_json(out / "interpolation_qc.json", qc)
    save_json(out / "real_inversion_status.json", {
        "status": "conditional_real_inverse_completed", "model": GRID_MODEL,
        "states": ["inhale", "exhale"], "observations_source": observations_rel,
        "observations_sha256": observations_sha, "observations_sha256_verified": True,
        "observations_copy_created": False, "measurement_design": "sequential_four_contact_montages",
        "simultaneous_all_sizes": False, "heldout_rmse_only_for_k_lt_9": True,
        "independent_tissue_truth": False, "data_scope": "one anatomy and fixed CT/montage",
    })
    save_json(out / "analysis_manifest.json", {
        "status": "exploratory_scenario_analysis",
        "scientific_status": "conditional_design_sensitivity_not_validation",
        "plan_file": plan_path.name, "responses_file": responses_path.name,
        "input_sha256": {"plan.json":sha256_file(plan_path), "responses.csv":sha256_file(responses_path)},
        "analysis_source_sha256":sha256_file(Path(__file__)),
        "response_execution_sidecar": tables["response_execution_sidecar"],
        "observations_source": observations_rel, "observations_sha256_verified": True,
        "observations_copy_created": False,
        "grid": {"model": GRID_MODEL, "points": 49, "channels": 9},
        "validation": {"points_per_model": 16, "models": list(MODELS)},
        "subsets": {"counts": "k=2..9", "total": 502},
        "measurement_design": {"current_experiment": "sequential_four_contact_montages",
                               "simultaneous_all_sizes": False},
        "best_by_count": {
            "rule": "strictly minimum worst local relative-noise-amplification",
            "score": "max over 49 grid points of max sqrt(diag(inv(Jlog.T @ Jlog)))",
            "Jlog": "dZ_dlogrho1 and dZ_dlogrho2; unit hypothetical impedance error in ohms",
            "worst_score_sequence": subset_meta["sequence"],
            "worst_score_nonincreasing_checked": subset_meta["monotonic"],
            "not_selected_by_nonlinear_CT_error": True,
        },
        "additional_subset_metrics": {
            "kappa": "worst singular-value ratio over the same 49 states",
            "g1_g2": "minimum over states of the Jlog column 2-norms",
            "attenuation_relative_full9": "min(g1/full9_g1, g2/full9_g2)",
            "rank2_min": "minimum numerical rank over the 49 states; local check only",
        },
        "identifiability": {"rank2_local_check": True, "global_injectivity_proven": False},
        "surrogate": "RectBivariateSpline in log(rho1), log(rho2), fitted to grid Z only",
        "inverse_validation": {
            "starts": ["best grid point", "log-bound center"],
            "solver": "scipy.optimize.least_squares in log resistivities",
            "self_target": GRID_MODEL, "ct_target": "reference",
            "k2_zero_residual_is_not_validation": True, "independent_tissue_truth": False,
        },
        "real_inverse": {"source_read_in_place": True, "independent_tissue_truth": False,
                         "heldout_rmse_only_for_k_lt_9": True},
        "scenario_sigmas": {"values_ohm": list(SCENARIO_SIGMAS_OHM),
                             "empirical_noise_estimates": False,
                             "confidence_or_crlb": False,
                             "criterion": "linearized sigma times worst unit log-parameter sensitivity"},
        "not_claimed": ["Gaussian errors", "CRLB or confidence intervals", "experimental tissue accuracy",
                         "global injectivity", "multi-anatomy portability", "necessity or sufficiency of any k"],
        "outputs": list(files) + ["interpolation_qc.json", "real_inversion_status.json", "analysis_manifest.json"],
    })
    return list(files) + ["interpolation_qc.json", "real_inversion_status.json", "analysis_manifest.json"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1]/"output/exploratory/lateral_array_design_20260916", help="output directory")
    parser.add_argument("plan_json", type=Path, nargs="?", help="plan.json")
    parser.add_argument("responses_csv", type=Path, nargs="?", help="precomputed responses.csv")
    args = parser.parse_args()
    try:
        names = run_analysis(args.out, args.plan_json or args.out/"plan.json", args.responses_csv or args.out/"responses.csv",
                             Path(__file__).resolve().parents[2])
    except (FileNotFoundError, OSError, ValueError, np.linalg.LinAlgError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print("analysis completed: " + ", ".join(names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
