import csv
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import sys

root = Path(__file__).resolve().parent / "physical_contract_fixture"
if root.exists():
    shutil.rmtree(root)
root.mkdir()
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import validate_tepc_physical_inputs as contract

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

(root / "calibration.txt").write_text("synthetic self-test only\n", encoding="utf-8")
(root / "phantom.stl").write_text("solid test\nendsolid test\n", encoding="ascii")

p1_columns = sorted(contract.INSTRUMENT_COLUMNS)
with (root / "p1.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=p1_columns)
    writer.writeheader()
    for session in ("s1", "s2"):
        for index, load in enumerate(("r1", "r2"), 1):
            row = {name: "" for name in p1_columns}
            row.update(run_id=f"p1_{session}_{load}", session_id=session, load_id=load,
                       repeat_index="1", connection_repeat="1", frequency_hz="50000",
                       current_rms_A="0.001", reference_Z_real_ohm=str(index * 10),
                       reference_Z_imag_ohm="0", reference_standard_id=load,
                       reference_standard_uncertainty_ohm="0.01", temperature_C="22",
                       measured_Z_real_ohm=str(index * 10.001), measured_Z_imag_ohm="0",
                       operator_anonymous_id="op1", notes="")
            writer.writerow(row)

phantom_columns = sorted(contract.PHANTOM_COLUMNS)
phantom_rows = []
for run_id, role, state in (("cal1", "calibration", "state0"),
                            ("val1", "validation", "state1")):
    row = {name: "" for name in phantom_columns}
    row.update(run_id=run_id, dataset_role=role, montage="tepc_2", state_id=state,
               repeat_index="1", reapplication_index="1", changed_compartment="soft",
               frequency_hz="50000", current_rms_A="0.001",
               rho_soft_ohm_m="5", u_rho_soft_ohm_m="0.05",
               rho_heart_ohm_m="2", u_rho_heart_ohm_m="0.02",
               rho_lung_ohm_m="6", u_rho_lung_ohm_m="0.06",
               rho_bone_ohm_m="48", u_rho_bone_ohm_m="0.48",
               temperature_C="22", contact_impedance_ohm_m2="0.000159",
               measured_Z_real_ohm="20", measured_Z_imag_ohm="0",
               u_measured_Z_real_ohm="0.02", u_measured_Z_imag_ohm="0.02",
               operator_anonymous_id="op1", notes="")
    phantom_rows.append(row)
with (root / "phantom.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=phantom_columns)
    writer.writeheader()
    writer.writerows(phantom_rows)

prediction_columns = sorted(contract.PREDICTION_COLUMNS)
row = {name: "" for name in prediction_columns}
row.update(run_id="val1", montage="tepc_2", state_id="state1",
           model_Z_real_ohm="20.01", model_Z_imag_ohm="0",
           u_model_Z_real_ohm="0.03", u_model_Z_imag_ohm="0.03",
           mesh_level="L06", model_identity_sha256="a" * 64)
with (root / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=prediction_columns)
    writer.writeheader()
    writer.writerow(row)

manifest = {
    "schema": "tepc_physical_validation_v1",
    "status": "recorded_inputs_sealed",
    "frequency_hz": 50000,
    "instrument": {
        "current_rms_A": 0.001,
        "calibration_record": "calibration.txt",
        "calibration_record_sha256": sha(root / "calibration.txt"),
    },
    "contacts": {
        "diameter_mm": 5.0,
        "specific_contact_impedance_ohm_m2": 0.000159,
    },
    "phantom": {
        "geometry_file": "phantom.stl",
        "geometry_file_sha256": sha(root / "phantom.stl"),
    },
    "data": {
        "instrument_measurements_csv": "p1.csv",
        "instrument_measurements_sha256": sha(root / "p1.csv"),
        "phantom_measurements_csv": "phantom.csv",
        "phantom_measurements_sha256": sha(root / "phantom.csv"),
        "model_predictions_csv": "predictions.csv",
        "model_predictions_sha256": sha(root / "predictions.csv"),
    },
    "split": {
        "calibration_run_ids": ["cal1"],
        "validation_run_ids": ["val1"],
        "fixed_before_validation_results_were_viewed": True,
    },
    "acceptance": {
        "fixed_before_validation_results_were_viewed": True,
        "Z_relative_error_max": 0.05,
        "delta_Z_normalized_error_max": 2.0,
        "sensitivity_normalized_error_max": 2.0,
    },
}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
report = contract.validate(SimpleNamespace(manifest=root / "manifest.json"))
assert report["status"] == "sealed_inputs_passed_contract_validation"
assert report["instrument"] == {"rows": 4, "loads": 2, "sessions": 2}
assert report["phantom"]["validation_conditions"] == 1
print(json.dumps(report, ensure_ascii=False, indent=2))