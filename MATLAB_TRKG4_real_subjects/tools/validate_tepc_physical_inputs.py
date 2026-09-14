"""Validate sealed P1-P3 input files before TEPC physical-validation analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path

MONTAGES = {f"tepc_{number}" for number in range(2, 8)}
PARAMS = ("soft", "heart", "lung")
SHA256 = re.compile(r"[0-9a-f]{64}")
INSTRUMENT_COLUMNS = {
    "run_id", "session_id", "load_id", "repeat_index", "connection_repeat",
    "frequency_hz", "current_rms_A", "reference_Z_real_ohm",
    "reference_Z_imag_ohm", "reference_standard_id",
    "reference_standard_uncertainty_ohm", "temperature_C",
    "measured_Z_real_ohm", "measured_Z_imag_ohm",
    "operator_anonymous_id", "notes",
}
PHANTOM_COLUMNS = {
    "run_id", "dataset_role", "montage", "state_id", "repeat_index",
    "reapplication_index", "changed_compartment", "frequency_hz",
    "current_rms_A", "rho_soft_ohm_m", "u_rho_soft_ohm_m",
    "rho_heart_ohm_m", "u_rho_heart_ohm_m", "rho_lung_ohm_m",
    "u_rho_lung_ohm_m", "rho_bone_ohm_m", "u_rho_bone_ohm_m",
    "temperature_C", "contact_impedance_ohm_m2", "measured_Z_real_ohm",
    "measured_Z_imag_ohm", "u_measured_Z_real_ohm",
    "u_measured_Z_imag_ohm", "operator_anonymous_id", "notes",
}
PREDICTION_COLUMNS = {
    "run_id", "montage", "state_id", "model_Z_real_ohm",
    "model_Z_imag_ohm", "u_model_Z_real_ohm", "u_model_Z_imag_ohm",
    "mesh_level", "model_identity_sha256",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def finite(row: dict, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric field {name}") from error
    require(math.isfinite(value), f"Nonfinite field {name}")
    if positive:
        require(value > 0, f"{name} must be positive")
    if nonnegative:
        require(value >= 0, f"{name} must be nonnegative")
    return value


def relative_file(base: Path, value: str) -> Path:
    require(isinstance(value, str) and value and "\x00" not in value, "A relative file path is required")
    require(not Path(value).is_absolute() and not Path(value).drive, "Absolute paths are not allowed")
    path = (base / value).resolve()
    require(path.is_relative_to(base.resolve()), "Path escapes manifest directory")
    require(path.is_file(), "Referenced file is missing: " + value)
    return path


def checked_file(base: Path, section: dict, path_name: str, hash_name: str) -> Path:
    path = relative_file(base, section[path_name])
    expected = str(section[hash_name]).lower()
    require(SHA256.fullmatch(expected) is not None, "Invalid SHA-256 field: " + hash_name)
    require(digest(path) == expected, "SHA-256 mismatch: " + path_name)
    return path


def rows(path: Path, required: set[str]) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None and set(reader.fieldnames) == required,
                "Unexpected CSV columns: " + path.name)
        result = list(reader)
    require(result, "CSV contains no measurements: " + path.name)
    return result


def validate_instrument(items: list[dict], frequency: float, current: float) -> dict:
    identities = set()
    loads = set()
    sessions = set()
    for item in items:
        identity = (
            item["run_id"], item["session_id"], item["load_id"],
            item["repeat_index"], item["connection_repeat"],
        )
        require(identity not in identities, "Duplicate P1 measurement identity")
        identities.add(identity)
        loads.add(item["load_id"])
        sessions.add(item["session_id"])
        require(math.isclose(finite(item, "frequency_hz", positive=True), frequency,
                             rel_tol=0, abs_tol=1e-9), "P1 frequency mismatch")
        require(math.isclose(finite(item, "current_rms_A", positive=True), current,
                             rel_tol=1e-9, abs_tol=0), "P1 current mismatch")
        for name in ("reference_Z_real_ohm", "reference_Z_imag_ohm",
                     "temperature_C", "measured_Z_real_ohm", "measured_Z_imag_ohm"):
            finite(item, name)
        finite(item, "reference_standard_uncertainty_ohm", nonnegative=True)
        require(item["reference_standard_id"].strip(), "Reference standard ID is required")
        require(item["operator_anonymous_id"].strip(), "Anonymous operator ID is required")
    require(len(loads) >= 2, "P1 must cover at least two reference loads")
    require(len(sessions) >= 2, "P1 must contain at least two sessions")
    return {"rows": len(items), "loads": len(loads), "sessions": len(sessions)}


def validate_phantom(
    items: list[dict],
    frequency: float,
    current: float,
    contact: float,
    split: dict,
) -> tuple[dict, set[tuple[str, str, str]]]:
    calibration = set(split["calibration_run_ids"])
    validation = set(split["validation_run_ids"])
    require(calibration and validation and calibration.isdisjoint(validation),
            "Calibration and validation run IDs must be nonempty and disjoint")
    require(split["fixed_before_validation_results_were_viewed"] is True,
            "The calibration/validation split was not predeclared")
    identities = set()
    conditions = set()
    observed_runs = set()
    for item in items:
        role = item["dataset_role"]
        require(role in {"calibration", "validation"}, "Unexpected dataset_role")
        run_id = item["run_id"]
        require(run_id in (calibration if role == "calibration" else validation),
                "Run ID is inconsistent with the sealed split")
        observed_runs.add(run_id)
        require(item["montage"] in MONTAGES, "Unexpected TEPC montage")
        require(item["changed_compartment"] in {*PARAMS, "none"}, "Unexpected changed compartment")
        identity = (
            run_id, item["montage"], item["state_id"],
            item["repeat_index"], item["reapplication_index"],
        )
        require(identity not in identities, "Duplicate phantom measurement identity")
        identities.add(identity)
        require(math.isclose(finite(item, "frequency_hz", positive=True), frequency,
                             rel_tol=0, abs_tol=1e-9), "Phantom frequency mismatch")
        require(math.isclose(finite(item, "current_rms_A", positive=True), current,
                             rel_tol=1e-9, abs_tol=0), "Phantom current mismatch")
        require(math.isclose(finite(item, "contact_impedance_ohm_m2", positive=True), contact,
                             rel_tol=1e-9, abs_tol=0), "Contact impedance differs from manifest")
        for name in ("rho_soft_ohm_m", "rho_heart_ohm_m", "rho_lung_ohm_m",
                     "rho_bone_ohm_m"):
            finite(item, name, positive=True)
        for name in ("u_rho_soft_ohm_m", "u_rho_heart_ohm_m", "u_rho_lung_ohm_m",
                     "u_rho_bone_ohm_m", "u_measured_Z_real_ohm",
                     "u_measured_Z_imag_ohm"):
            finite(item, name, nonnegative=True)
        for name in ("temperature_C", "measured_Z_real_ohm", "measured_Z_imag_ohm"):
            finite(item, name)
        require(item["operator_anonymous_id"].strip(), "Anonymous operator ID is required")
        if role == "validation":
            conditions.add((run_id, item["montage"], item["state_id"]))
    require(observed_runs == calibration | validation, "Manifest split includes absent run IDs")
    return {
        "rows": len(items),
        "calibration_runs": len(calibration),
        "validation_runs": len(validation),
        "validation_conditions": len(conditions),
    }, conditions


def validate_predictions(items: list[dict], wanted: set[tuple[str, str, str]]) -> dict:
    conditions = set()
    for item in items:
        condition = (item["run_id"], item["montage"], item["state_id"])
        require(condition not in conditions, "Duplicate model prediction")
        conditions.add(condition)
        require(item["montage"] in MONTAGES, "Unexpected prediction montage")
        for name in ("model_Z_real_ohm", "model_Z_imag_ohm"):
            finite(item, name)
        for name in ("u_model_Z_real_ohm", "u_model_Z_imag_ohm"):
            finite(item, name, nonnegative=True)
        require(item["mesh_level"].strip(), "Mesh level is required")
        require(SHA256.fullmatch(item["model_identity_sha256"].lower()) is not None,
                "Invalid model identity SHA-256")
    require(conditions == wanted, "Predictions do not match all and only validation conditions")
    return {"rows": len(items), "validation_conditions": len(conditions)}


def validate(args) -> dict:
    manifest_path = args.manifest.resolve()
    manifest = read_json(manifest_path)
    require(manifest.get("schema") == "tepc_physical_validation_v1", "Unexpected schema")
    require(manifest.get("status") == "recorded_inputs_sealed", "Manifest is a template or is not sealed")
    frequency = float(manifest["frequency_hz"])
    require(math.isclose(frequency, 50000, rel_tol=0, abs_tol=1e-9), "Expected 50 kHz")
    current = float(manifest["instrument"]["current_rms_A"])
    require(math.isfinite(current) and current > 0, "Positive current_rms_A is required")
    require(math.isclose(float(manifest["contacts"]["diameter_mm"]), 5.0,
                         rel_tol=0, abs_tol=1e-12), "Expected 5 mm contacts")
    contact = float(manifest["contacts"]["specific_contact_impedance_ohm_m2"])
    require(math.isfinite(contact) and contact > 0, "Measured/controlled contact impedance is required")
    base = manifest_path.parent
    checked_file(base, manifest["instrument"], "calibration_record", "calibration_record_sha256")
    checked_file(base, manifest["phantom"], "geometry_file", "geometry_file_sha256")
    data = manifest["data"]
    instrument_path = checked_file(base, data, "instrument_measurements_csv",
                                   "instrument_measurements_sha256")
    phantom_path = checked_file(base, data, "phantom_measurements_csv",
                                "phantom_measurements_sha256")
    prediction_path = checked_file(base, data, "model_predictions_csv",
                                   "model_predictions_sha256")
    acceptance = manifest["acceptance"]
    require(acceptance["fixed_before_validation_results_were_viewed"] is True,
            "Acceptance criteria were not predeclared")
    for name in ("Z_relative_error_max", "delta_Z_normalized_error_max",
                 "sensitivity_normalized_error_max"):
        value = float(acceptance[name])
        require(math.isfinite(value) and value > 0, "Invalid acceptance threshold: " + name)
    p1 = validate_instrument(rows(instrument_path, INSTRUMENT_COLUMNS), frequency, current)
    phantom, wanted = validate_phantom(
        rows(phantom_path, PHANTOM_COLUMNS), frequency, current, contact, manifest["split"]
    )
    prediction = validate_predictions(rows(prediction_path, PREDICTION_COLUMNS), wanted)
    return {
        "schema": "tepc_physical_validation_input_receipt_v1",
        "status": "sealed_inputs_passed_contract_validation",
        "manifest": os.path.relpath(manifest_path, Path.cwd()).replace("\\", "/"),
        "manifest_sha256": digest(manifest_path),
        "frequency_hz": frequency,
        "contact_diameter_mm": 5.0,
        "instrument": p1,
        "phantom": phantom,
        "predictions": prediction,
        "physical_validation": False,
        "interpretation": "Input integrity passed; agreement between model and measurement has not yet been analysed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        require(not output.exists(), "Refusing to overwrite an existing receipt")
        output.write_text(text, encoding="utf-8")
        output.with_suffix(output.suffix + ".sha256").write_text(digest(output) + "\n", encoding="ascii")
    print(text, end="")


if __name__ == "__main__":
    main()