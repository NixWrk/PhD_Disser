"""Compare two completed electrical runs made on the same heart geometry.

The command is a provenance and numerical report, not a convergence decision.
The runs must use the same original geometric source, material model, motion
parameterisation, frequency, and physical contact.  A fine run may use a
different electrical mesh and quadrature count.  Only states with checked
completions in both directories are compared.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import heart_fractional_electrical_pilot as fractional
import heart_individual_electrical_stage as individual
import heart_replacement_electrical_pilot as core


read, sha, require = core.read, core.sha, core.require

_DROP_PROVENANCE_KEYS = {
    "path",
    "sha256",
    "contact_sha256",
    "prepared_sha256",
    "prepared_fem_sha256",
    "source_prepared_sha256",
}
_QC_KEYS = (
    "max_relative_residual",
    "reciprocity_absolute_ohm",
    "max_current_error_A",
    "max_ground_residual_A",
    "matrix_asymmetry_relative",
)


def _fail(condition: Any, message: str) -> None:
    require(bool(condition), message)


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    _fail(math.isfinite(result), f"{label} is not finite")
    return result


def _q(value: Any, label: str) -> int:
    number = _finite(value, label)
    _fail(number > 0 and number.is_integer(), f"{label} is invalid")
    return int(number)


def _canonical(value: Any) -> Any:
    """Make JSON-loaded values comparable without depending on dict order."""
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    # Test doubles can use numpy scalar values even though real plans are JSON.
    if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return _canonical(value.item())
        except ValueError:
            pass
    if isinstance(value, float) and value == 0.0:
        return 0.0
    return value


def _same(left: Any, right: Any, label: str) -> None:
    _fail(_canonical(left) == _canonical(right), f"{label} differs")


def _resolve(stage: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (stage / path).resolve()


def _checked_hash(path: Path, expected: Any, label: str) -> None:
    _fail(path.is_file(), f"Missing {label}")
    _fail(isinstance(expected, str) and len(expected) == 64, f"Missing {label} hash")
    _same(sha(path), expected, f"{label} hash")


def _strip_provenance(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): _strip_provenance(v)
            for k, v in value.items()
            if str(k) not in _DROP_PROVENANCE_KEYS
        }
    if isinstance(value, list):
        return [_strip_provenance(v) for v in value]
    return _canonical(value)


def _contact_signature(contact: dict[str, Any]) -> dict[str, Any]:
    _fail(isinstance(contact, dict), "Missing contact metadata")
    _fail(isinstance(contact.get("id"), str) and contact["id"], "Contact id is missing")
    areas = contact.get("areas_mm2", contact.get("electrode_areas_mm2"))
    centres = contact.get(
        "actual_centres_xyz_mm",
        contact.get("centres_xyz_mm", contact.get("centers_xyz_mm")),
    )
    _fail(areas is not None, "Contact physical areas are missing")
    _fail(centres is not None, "Contact physical centres are missing")
    _fail(len(areas) > 0 and len(centres) == len(areas), "Contact area/centre counts differ")
    for i, area in enumerate(areas):
        _finite(area, f"Contact area {i}")
    for i, centre in enumerate(centres):
        _fail(len(centre) == 3, f"Contact centre {i} is not 3D")
        for axis, coordinate in enumerate(centre):
            _finite(coordinate, f"Contact centre {i}[{axis}]")
    signature = _strip_provenance(contact)
    signature["id"] = contact["id"]
    signature["areas_mm2"] = _canonical(areas)
    signature["actual_centres_xyz_mm"] = _canonical(centres)
    signature.pop("electrode_areas_mm2", None)
    signature.pop("centres_xyz_mm", None)
    signature.pop("centers_xyz_mm", None)
    return signature


def _state_design(state: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Return state identity while deliberately excluding id and quadrature."""
    _fail(isinstance(state, dict), "Invalid state metadata")
    geometry = state.get("geometry", plan.get("geometry"))
    _fail(isinstance(geometry, str) and geometry, "State geometry is missing")
    volume_fraction = _finite(state.get("volume_fraction", 0.0), "State volume_fraction")
    parameter = state.get("parameter")
    if parameter is None:
        parameter = "baseline" if volume_fraction == 0.0 else "volume_fraction"
    _fail(isinstance(parameter, str) and parameter, "State parameter is missing")
    default_step = 0.0 if parameter == "baseline" else volume_fraction
    signed_step = _finite(state.get("signed_step", default_step), "State signed_step")
    translation = state.get("translation_m", [0.0, 0.0, 0.0])
    _fail(len(translation) == 3, "State translation is not 3D")
    translation = [_finite(v, f"State translation[{i}]") for i, v in enumerate(translation)]
    analytic = _finite(state.get("analytic_volume_ml"), "State analytic volume")
    return {
        "geometry": geometry,
        "parameter": parameter,
        "signed_step": signed_step,
        "volume_fraction": volume_fraction,
        "translation_m": translation,
        "analytic_volume_ml": analytic,
    }


def _design_key(design: dict[str, Any], *, include_analytic_volume: bool = True) -> str:
    value = dict(design)
    if not include_analytic_volume:
        value.pop("analytic_volume_ml", None)
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _display_design(design: dict[str, Any]) -> str:
    if design["parameter"] == "baseline":
        return "baseline"
    return f'{design["parameter"]}:{design["signed_step"]:+g}'


def _state_sort_key(design: dict[str, Any]) -> tuple[Any, ...]:
    return (
        0 if design["parameter"] == "baseline" else 1,
        design["parameter"],
        design["signed_step"],
        tuple(design["translation_m"]),
    )


def _find_refinement_record(stage: Path, plan: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for item in plan.get("inputs", []):
        if not isinstance(item, dict) or "path" not in item:
            continue
        path = _resolve(stage, item["path"])
        if path.name == "heart_refinement_export.json":
            matches.append((path, item))
    _fail(len(matches) == 1, "Fine plan must bind one heart_refinement_export.json")
    path, item = matches[0]
    _checked_hash(path, item.get("sha256"), "heart_refinement_export.json")
    record = read(path)
    _fail(isinstance(record, dict), "Invalid heart_refinement_export.json")
    return path, record


def _record_artifact_path(record_path: Path, value: Any) -> Path:
    _fail(value is not None, "Fine refinement artifact path is missing")
    path = Path(str(value))
    return path if path.is_absolute() else (record_path.parent / path).resolve()


def _validate_fine_provenance(
    coarse: dict[str, Any], fine_dir: Path, fine: dict[str, Any]
) -> dict[str, Any]:
    record_path, record = _find_refinement_record(fine_dir, fine)
    _same(record.get("status"), "passed_heart_refinement_export", "Fine refinement status")
    _same(
        record.get("source_prepared_sha256"),
        coarse["source_prepared_sha256"],
        "Fine refinement source prepared hash",
    )
    _same(
        record.get("original_contact_sha256"),
        coarse["contact"]["sha256"],
        "Fine refinement original contact hash",
    )
    for key in ("original_nodes_preserved", "outer_boundary_preserved", "contact_faces_preserved"):
        _fail(record.get(key) is True, f"Fine refinement did not preserve {key}")
    embedding = _finite(record.get("contact_embedding_relative_error"), "Fine contact embedding error")
    _fail(embedding >= 0.0 and embedding < 1e-12,
          "Fine contact embedding error must be nonnegative and below 1e-12")
    _fail("matrix_relative_error" in record, "Fine export matrix error is missing")
    matrix_error = _finite(record["matrix_relative_error"], "Fine export matrix error")
    _fail(matrix_error >= 0.0 and matrix_error < 1e-12,
          "Fine export matrix error must be nonnegative and below 1e-12")

    fine_prepared = _resolve(fine_dir, fine.get("source_prepared"))
    _checked_hash(fine_prepared, fine.get("source_prepared_sha256"), "fine prepared mesh")
    prepared = _record_artifact_path(record_path, record.get("prepared_file"))
    _same(prepared.resolve(), fine_prepared.resolve(), "Fine prepared artifact")
    _same(record.get("prepared_sha256"), fine["source_prepared_sha256"], "Fine prepared artifact hash")
    _same(record.get("prepared_sha256"), sha(prepared), "Fine prepared artifact content")

    fine_contact = fine.get("contact")
    _fail(isinstance(fine_contact, dict), "Fine contact metadata is missing")
    fine_contact_path = _resolve(fine_dir, fine_contact.get("path"))
    _checked_hash(fine_contact_path, fine_contact.get("sha256"), "fine contact")
    contact = _record_artifact_path(record_path, record.get("contact_file"))
    _same(contact.resolve(), fine_contact_path.resolve(), "Fine contact artifact")
    _same(record.get("contact_sha256"), fine_contact.get("sha256"), "Fine contact artifact hash")
    _same(record.get("contact_sha256"), sha(contact), "Fine contact artifact content")

    return {
        "status": record["status"],
        "source_prepared_sha256": record["source_prepared_sha256"],
        "prepared_sha256": record["prepared_sha256"],
        "contact_sha256": record["contact_sha256"],
        "prepared_file": prepared.name,
        "contact_file": contact.name,
        "original_nodes_preserved": True,
        "outer_boundary_preserved": True,
        "contact_faces_preserved": True,
        "contact_embedding_relative_error": embedding,
        "matrix_relative_error": matrix_error,
        "hashes_passed": True,
        "source_matches_coarse_prepared": True,
        "surface_and_node_prefix_preserved": True,
        "coarse_prepared_sha256": coarse["source_prepared_sha256"],
    }


def _validate_plan_identity(coarse: dict[str, Any], fine: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "geometry",
        "parameter",
        "deformation",
        "frequency_hz",
        "rho_ohm_m",
        "conductivity_by_tissue_id",
        "electrode_model",
        "ground_zero_based",
    )
    for field in fields:
        _fail(field in coarse and field in fine, f"Missing plan field {field}")
        _same(coarse[field], fine[field], f"Plan field {field}")
    for label, plan in (("coarse", coarse), ("fine", fine)):
        source_hash = plan.get("original_geometric_source_sha256")
        _fail(isinstance(source_hash, str) and source_hash, f"{label} original geometric source hash is missing")
        model = plan["electrode_model"]
        _fail(isinstance(model, dict), f"{label} electrode model is missing")
        _fail(_finite(model.get("diameter_mm"), f"{label} electrode diameter") == 5.0,
              f"{label} electrode model is not 5 mm")
    _same(coarse["original_geometric_source_sha256"], fine["original_geometric_source_sha256"],
          "Original geometric source")
    _fail(isinstance(coarse.get("contact"), dict) and isinstance(fine.get("contact"), dict),
          "Both plans must contain contact metadata")
    coarse_contact = _contact_signature(coarse["contact"])
    fine_contact = _contact_signature(fine["contact"])
    _same(coarse_contact, fine_contact, "Physical contact geometry")
    return {
        "same_original_geometric_source": True,
        "original_geometric_source_sha256": coarse["original_geometric_source_sha256"],
        "same_geometry": True,
        "same_parameter": True,
        "same_deformation": True,
        "same_frequency_hz": True,
        "same_rho_ohm_m": True,
        "same_sigma": True,
        "same_electrode_model": True,
        "electrode_diameter_mm": 5.0,
        "same_ground_zero_based": True,
        "contact_id": coarse["contact"]["id"],
        "contact_physical_metadata_equal": True,
        "coarse_prepared_sha256": coarse["source_prepared_sha256"],
        "fine_prepared_sha256": fine["source_prepared_sha256"],
        "coarse_contact_sha256": coarse["contact"]["sha256"],
        "fine_contact_sha256": fine["contact"]["sha256"],
    }


def _validate_state_designs(
    coarse: dict[str, Any], fine: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    maps: list[dict[str, dict[str, Any]]] = []
    for label, plan in (("coarse", coarse), ("fine", fine)):
        _fail(isinstance(plan.get("states"), list) and plan["states"], f"{label} states are missing")
        states: dict[str, dict[str, Any]] = {}
        for state in plan["states"]:
            design = _state_design(state, plan)
            key = _design_key(design)
            _fail(key not in states, f"{label} has duplicate state design {_display_design(design)}")
            states[key] = design
        maps.append(states)
    coarse_states, fine_states = maps
    _same(set(coarse_states), set(fine_states), "Planned state designs")
    for key in coarse_states:
        _same(coarse_states[key], fine_states[key], "State design and analytic volume")
    return coarse_states, fine_states


def _collect_completed(
    stage: Path, plan: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    # summarize() is the canonical stage inventory.  checked_result() is also
    # called explicitly so an unfinished state can never enter this report.
    summary = individual.summarize(stage, plan)
    _fail(isinstance(summary, dict), "Stage summarize did not return an object")
    completed: dict[str, dict[str, Any]] = {}
    for state in plan["states"]:
        row = fractional.checked_result(stage, plan, state)
        if row is None:
            continue
        design = _state_design(state, plan)
        key = _design_key(design)
        _fail(key not in completed, f"Duplicate completed state {_display_design(design)}")
        _fail("Z_ohm" in row, f"Completed state {_display_design(design)} has no Z_ohm")
        _finite(row["Z_ohm"], f"Z_ohm for {_display_design(design)}")
        q = _q(row.get("points_per_tet", state.get("points_per_tet")), "points_per_tet")
        merged = dict(row)
        merged.setdefault("id", state.get("id"))
        merged.setdefault("points_per_tet", q)
        merged.setdefault("analytic_volume_ml", design["analytic_volume_ml"])
        if "material_volume_ml" not in merged and "material_volume_ml" in state:
            merged["material_volume_ml"] = state["material_volume_ml"]
        for field in ("geometry", "parameter", "signed_step", "volume_fraction", "translation_m"):
            if field in row:
                _same(row[field], design[field], f"Completed state {field}")
        completed[key] = {"design": design, "row": merged, "state_id": state.get("id")}
    return summary, completed


def _paired_design(design: dict[str, Any]) -> dict[str, Any]:
    paired = dict(design)
    paired["signed_step"] = -design["signed_step"]
    paired["volume_fraction"] = -design["volume_fraction"]
    paired["translation_m"] = [-v for v in design["translation_m"]]
    return paired


def _derivatives(completed: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = next((item for item in completed.values()
                     if item["design"]["parameter"] == "baseline"), None)
    if baseline is None:
        return []
    baseline_volume = _finite(baseline["design"]["analytic_volume_ml"], "Baseline analytic volume")
    baseline_z = _finite(baseline["row"]["Z_ohm"], "Baseline Z")
    pair_lookup = {
        _design_key(item["design"], include_analytic_volume=False): item
        for item in completed.values()
    }
    derivatives: list[dict[str, Any]] = []
    for item in completed.values():
        design = item["design"]
        step = design["signed_step"]
        if design["parameter"] == "baseline" or step <= 0:
            continue
        minus = pair_lookup.get(_design_key(_paired_design(design), include_analytic_volume=False))
        if minus is None:
            continue
        plus_row, minus_row = item["row"], minus["row"]
        if design["parameter"] == "volume_fraction":
            denominator = 2.0 * step * baseline_volume
            one_sided_denominator = step * baseline_volume
            units, expected = "ohm/ml", 1.0
        else:
            denominator = 2.0 * step
            one_sided_denominator = step
            units, expected = "ohm/mm", 0.0
        _fail(denominator != 0 and math.isfinite(denominator), "Invalid derivative denominator")
        plus_z = _finite(plus_row["Z_ohm"], "Positive-state Z")
        minus_z = _finite(minus_row["Z_ohm"], "Negative-state Z")
        derivative = (plus_z - minus_z) / denominator
        # These are deliberately retained beside the central difference.  A
        # stable central slope can coexist with visible +/- asymmetry.
        sminus = (baseline_z - minus_z) / one_sided_denominator
        splus = (plus_z - baseline_z) / one_sided_denominator
        normalized_denominator = (abs(sminus) + abs(splus)) / 2.0
        normalized_difference = None if normalized_denominator == 0 else (
            (splus - sminus) / normalized_denominator
        )
        material_derivative = None
        if plus_row.get("material_volume_ml") is not None and minus_row.get("material_volume_ml") is not None:
            material_derivative = (
                _finite(plus_row["material_volume_ml"], "Positive material volume")
                - _finite(minus_row["material_volume_ml"], "Negative material volume")
            ) / denominator
        derivatives.append({
            "parameter": design["parameter"],
            "step": step,
            "units": units,
            "q_plus": _q(plus_row["points_per_tet"], "Positive q"),
            "q_minus": _q(minus_row["points_per_tet"], "Negative q"),
            "analytic_denominator": denominator,
            "baseline_analytic_volume_ml": baseline_volume,
            "Z_baseline_ohm": baseline_z,
            "dV_one_sided": one_sided_denominator,
            "Z_minus_ohm": minus_z,
            "Z_plus_ohm": plus_z,
            "Sminus": sminus,
            "Splus": splus,
            "one_sided_slope_difference_Splus_minus_Sminus": splus - sminus,
            "one_sided_normalized_difference": normalized_difference,
            "derivative": derivative,
            "material_volume_derivative": material_derivative,
            "expected_material_volume_derivative": expected,
        })
    return sorted(derivatives, key=lambda d: (d["parameter"], d["step"]))


def _step_refinement(derivatives: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for parameter in sorted({d["parameter"] for d in derivatives}):
        selected = sorted((d for d in derivatives if d["parameter"] == parameter), key=lambda d: d["step"])
        for fine, coarse in zip(selected, selected[1:]):
            difference = coarse["derivative"] - fine["derivative"]
            result.append({
                "parameter": parameter,
                "fine_step": fine["step"],
                "coarse_step": coarse["step"],
                "fine_derivative": fine["derivative"],
                "coarse_derivative": coarse["derivative"],
                "derivative_difference_coarse_minus_fine": difference,
                "relative_difference": None if fine["derivative"] == 0 else abs(difference / fine["derivative"]),
                "fine_one_sided_normalized_difference": fine["one_sided_normalized_difference"],
                "coarse_one_sided_normalized_difference": coarse["one_sided_normalized_difference"],
            })
    return result


def _state_comparisons(
    coarse: dict[str, dict[str, Any]], fine: dict[str, dict[str, Any]], all_designs: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    common = sorted(set(coarse) & set(fine), key=lambda key: _state_sort_key(all_designs[key]))
    result: list[dict[str, Any]] = []
    for key in common:
        design = all_designs[key]
        coarse_item, fine_item = coarse[key], fine[key]
        coarse_z = _finite(coarse_item["row"]["Z_ohm"], "Coarse Z")
        fine_z = _finite(fine_item["row"]["Z_ohm"], "Fine Z")
        difference = fine_z - coarse_z
        result.append({
            "design": design,
            "label": _display_design(design),
            "state_id_coarse": coarse_item["state_id"],
            "state_id_fine": fine_item["state_id"],
            "q_coarse": _q(coarse_item["row"]["points_per_tet"], "Coarse q"),
            "q_fine": _q(fine_item["row"]["points_per_tet"], "Fine q"),
            "Z_coarse_ohm": coarse_z,
            "Z_fine_ohm": fine_z,
            "Z_fine_minus_coarse_ohm": difference,
            "Z_difference_relative_to_coarse": None if coarse_z == 0 else difference / abs(coarse_z),
        })
    return result


def _baseline_shift(state_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    baseline = next((item for item in state_rows if item["design"]["parameter"] == "baseline"), None)
    if baseline is None:
        return None
    return {
        "q_coarse": baseline["q_coarse"],
        "q_fine": baseline["q_fine"],
        "Z_coarse_ohm": baseline["Z_coarse_ohm"],
        "Z_fine_ohm": baseline["Z_fine_ohm"],
        "Z_fine_minus_coarse_ohm": baseline["Z_fine_minus_coarse_ohm"],
        "relative_to_coarse": baseline["Z_difference_relative_to_coarse"],
    }


def _material_slope_comparison(
    coarse: list[dict[str, Any]], fine: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    c = {(d["parameter"], d["step"]): d for d in coarse}
    f = {(d["parameter"], d["step"]): d for d in fine}
    result: list[dict[str, Any]] = []
    for key in sorted(set(c) & set(f)):
        c_value, f_value = c[key]["material_volume_derivative"], f[key]["material_volume_derivative"]
        result.append({
            "parameter": key[0],
            "step": key[1],
            "q_coarse_plus_minus": [c[key]["q_minus"], c[key]["q_plus"]],
            "q_fine_plus_minus": [f[key]["q_minus"], f[key]["q_plus"]],
            "material_dV_slope_coarse": c_value,
            "material_dV_slope_fine": f_value,
            "material_dV_slope_fine_minus_coarse": None
            if c_value is None or f_value is None else f_value - c_value,
            "expected_material_dV_slope": c[key]["expected_material_volume_derivative"],
        })
    return result


def _numerical_qc(completed: dict[str, dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    maxima: dict[str, float] = {}
    for item in sorted(completed.values(), key=lambda v: _state_sort_key(v["design"])):
        row = item["row"]
        qc: dict[str, Any] = {}
        for key in _QC_KEYS:
            if key in row:
                value = _finite(row[key], f"Numerical QC {key}")
                qc[key] = value
                maxima[key] = max(maxima.get(key, -math.inf), value)
        rows.append({
            "label": _display_design(item["design"]),
            "state_id": item["state_id"],
            "q": _q(row["points_per_tet"], "QC q"),
            "checked_result": True,
            "checks": qc,
        })
    return {
        "summary_status": summary.get("status"),
        "summary_completed": summary.get("completed", len(completed)),
        "summary_required": summary.get("required"),
        "completed_rows": rows,
        "maxima": maxima,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "label", "parameter", "signed_step", "q_coarse", "q_fine",
        "Z_coarse_ohm", "Z_fine_ohm", "Z_fine_minus_coarse_ohm",
        "Z_difference_relative_to_coarse",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in rows:
            design = item["design"]
            writer.writerow({
                "label": item["label"],
                "parameter": design["parameter"],
                "signed_step": design["signed_step"],
                "q_coarse": item["q_coarse"],
                "q_fine": item["q_fine"],
                "Z_coarse_ohm": item["Z_coarse_ohm"],
                "Z_fine_ohm": item["Z_fine_ohm"],
                "Z_fine_minus_coarse_ohm": item["Z_fine_minus_coarse_ohm"],
                "Z_difference_relative_to_coarse": item["Z_difference_relative_to_coarse"],
            })


def compare(coarse_dir: Path | str, fine_dir: Path | str, output: Path | str) -> dict[str, Any]:
    """Validate and compare two stage directories, writing JSON and CSV."""
    coarse_dir = Path(coarse_dir).resolve()
    fine_dir = Path(fine_dir).resolve()
    output = Path(output).resolve()
    _fail(coarse_dir.is_dir(), "Coarse stage directory is missing")
    _fail(fine_dir.is_dir(), "Fine stage directory is missing")
    _fail(coarse_dir != fine_dir, "Coarse and fine directories must differ")

    coarse = fractional.load_plan(coarse_dir)
    fine = fractional.load_plan(fine_dir)
    provenance = _validate_plan_identity(coarse, fine)
    _checked_hash(_resolve(coarse_dir, coarse.get("source_prepared")), coarse.get("source_prepared_sha256"),
                  "coarse prepared mesh")
    coarse_contact = coarse["contact"]
    _checked_hash(_resolve(coarse_dir, coarse_contact.get("path")), coarse_contact.get("sha256"), "coarse contact")
    refinement = _validate_fine_provenance(coarse, fine_dir, fine)
    coarse_designs, fine_designs = _validate_state_designs(coarse, fine)
    coarse_summary, coarse_completed = _collect_completed(coarse_dir, coarse)
    fine_summary, fine_completed = _collect_completed(fine_dir, fine)

    all_designs = dict(coarse_designs)
    common = set(coarse_completed) & set(fine_completed)
    _fail(common, "No common completed states")
    state_rows = _state_comparisons(coarse_completed, fine_completed, all_designs)
    coarse_derivatives = _derivatives(coarse_completed)
    fine_derivatives = _derivatives(fine_completed)
    c_derivatives = {(d["parameter"], d["step"]): d for d in coarse_derivatives}
    f_derivatives = {(d["parameter"], d["step"]): d for d in fine_derivatives}
    derivative_difference: list[dict[str, Any]] = []
    for key in sorted(set(c_derivatives) & set(f_derivatives)):
        c, f = c_derivatives[key], f_derivatives[key]
        derivative_difference.append({
            "parameter": key[0],
            "step": key[1],
            "q_coarse_plus_minus": [c["q_minus"], c["q_plus"]],
            "q_fine_plus_minus": [f["q_minus"], f["q_plus"]],
            "derivative_coarse": c["derivative"],
            "derivative_fine": f["derivative"],
            "derivative_fine_minus_coarse": f["derivative"] - c["derivative"],
            "Sminus_coarse": c["Sminus"],
            "Splus_coarse": c["Splus"],
            "Sminus_fine": f["Sminus"],
            "Splus_fine": f["Splus"],
            "one_sided_normalized_difference_coarse": c["one_sided_normalized_difference"],
            "one_sided_normalized_difference_fine": f["one_sided_normalized_difference"],
            "material_dV_slope_coarse": c["material_volume_derivative"],
            "material_dV_slope_fine": f["material_volume_derivative"],
        })

    output.mkdir(parents=True, exist_ok=True)
    coarse_plan_sha256 = sha(coarse_dir / "plan.json")
    fine_plan_sha256 = sha(fine_dir / "plan.json")
    comparison_implementation_sha256 = sha(Path(__file__).resolve())
    coarse_implementation_sha256 = [
        item["sha256"] for item in coarse.get("implementation", [])
        if isinstance(item, dict) and isinstance(item.get("sha256"), str)
    ]
    fine_implementation_sha256 = [
        item["sha256"] for item in fine.get("implementation", [])
        if isinstance(item, dict) and isinstance(item.get("sha256"), str)
    ]
    report = {
        "schema": "heart_mesh_level_comparison_v1",
        "status": "complete" if len(common) == len(all_designs) else "partial",
        "coarse_stage": coarse_dir.name,
        "fine_stage": fine_dir.name,
        "coarse_plan_sha256": coarse_plan_sha256,
        "fine_plan_sha256": fine_plan_sha256,
        "implementation_sha256": comparison_implementation_sha256,
        "stage_implementation_sha256": {
            "coarse": coarse_implementation_sha256,
            "fine": fine_implementation_sha256,
        },
        "spatial_convergence": False,
        "spatial_convergence_reason": "One coarse/fine mesh pair is a comparison and cannot certify spatial convergence.",
        "physical_validation": False,
        "one_sided_asymmetry_reported": True,
        "one_sided_asymmetry_note": "Sminus and Splus are reported separately; no cause is assigned without separating tests.",
        "provenance": dict(
            provenance,
            coarse_plan_sha256=coarse_plan_sha256,
            fine_plan_sha256=fine_plan_sha256,
            implementation_sha256=comparison_implementation_sha256,
            coarse_implementation_sha256=coarse_implementation_sha256,
            fine_implementation_sha256=fine_implementation_sha256,
        ),
        "fine_refinement_provenance": refinement,
        "state_designs": {
            "planned": len(all_designs),
            "coarse_completed": len(coarse_completed),
            "fine_completed": len(fine_completed),
            "compared_completed": len(common),
            "coarse_only_completed": [_display_design(coarse_completed[key]["design"])
                                      for key in sorted(set(coarse_completed) - common,
                                                         key=lambda k: _state_sort_key(coarse_completed[k]["design"]))],
            "fine_only_completed": [_display_design(fine_completed[key]["design"])
                                    for key in sorted(set(fine_completed) - common,
                                                       key=lambda k: _state_sort_key(fine_completed[k]["design"]))],
        },
        "quadrature": {
            "coarse_points_per_tet": sorted({_q(v["row"]["points_per_tet"], "Coarse q")
                                               for v in coarse_completed.values()}),
            "fine_points_per_tet": sorted({_q(v["row"]["points_per_tet"], "Fine q")
                                             for v in fine_completed.values()}),
        },
        "baseline_Z_shift": _baseline_shift(state_rows),
        "state_comparisons": state_rows,
        "derivatives": {"coarse": coarse_derivatives, "fine": fine_derivatives},
        "derivative_comparison": derivative_difference,
        "material_dV_slopes": _material_slope_comparison(coarse_derivatives, fine_derivatives),
        "step_refinement": {
            "coarse": _step_refinement(coarse_derivatives),
            "fine": _step_refinement(fine_derivatives),
            "stage_summaries": {
                "coarse": coarse_summary.get("step_checks", []),
                "fine": fine_summary.get("step_checks", []),
            },
        },
        "numerical_qc": {
            "coarse": _numerical_qc(coarse_completed, coarse_summary),
            "fine": _numerical_qc(fine_completed, fine_summary),
        },
    }
    (output / "comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_csv(output / "comparison.csv", state_rows)
    return report


def run(args: Any) -> dict[str, Any]:
    """argparse-compatible entry point retained for the repository's tool style."""
    return compare(args.coarse, args.fine, args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse", type=Path, required=True, help="coarse electrical stage directory")
    parser.add_argument("--fine", type=Path, required=True, help="fine electrical stage directory")
    parser.add_argument("--output", type=Path, required=True, help="report directory")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "status": report["status"],
        "compared_completed": report["state_designs"]["compared_completed"],
        "spatial_convergence": report["spatial_convergence"],
        "output": args.output.name,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
