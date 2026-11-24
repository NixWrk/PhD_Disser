"""Local distinguishability of three independent material changes in the 3D FEM.

The previous split-tissue 60-mm calibration supplies a single operating point
per subject and posture, with lung resistivity fixed at its starting value.
Use native FEM partial derivatives before tying fat to soft tissue. Each row
of the resulting matrix represents one posterior size and one relative voltage
change at fixed injected current. 200-mm measurements are held out.
"""
import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SIZES = (60, 100, 140)
MATERIALS = ("soft", "fat", "lung")


def evaluate():
    with (ROOT.parent / "fem60_fields.csv").open(encoding="utf-8-sig") as src:
        data = [r for r in csv.DictReader(src) if r["scenario"] == "common_multiplier"
                and int(r["size_mm"]) in SIZES]

    results = []
    for participant in ("U1", "U2"):
        for posture in ("seated", "lying"):
            group = sorted((r for r in data if r["participant"] == participant
                            and r["posture"] == posture), key=lambda r: int(r["size_mm"]))
            if tuple(int(r["size_mm"]) for r in group) != SIZES:
                raise ValueError(f"Incomplete FEM data for {participant}, {posture}")
            rho = {m: float(group[0][f"rho_{m}_ohm_m"]) for m in MATERIALS}
            if any(float(r[f"rho_{m}_ohm_m"]) != rho[m] for r in group for m in MATERIALS):
                raise ValueError("Three different sizes must share one physical operating point")
            if any(r["prediction_is_full_FEM"] != "True" or r["placement"] != "back_right"
                   or r["pair"] != "back_arm" or int(r["frequency_hz"]) != 100000 for r in group):
                raise ValueError("Expected right posterior full 3D FEM fields at 100 kHz")

            z = np.array([float(r["predicted_Z_ohm"]) for r in group])
            jacobian = np.array([[float(r[f"j_{m}_ohm"]) for m in MATERIALS] for r in group])
            relative = jacobian / z[:, None]
            _, singular_values, directions = np.linalg.svd(relative)
            weak = directions[-1].copy()
            if weak[0] < 0:
                weak *= -1  # Orient the weakest direction consistently across participants.
            np.testing.assert_allclose(np.linalg.norm(relative @ weak), singular_values[-1])
            results.append(dict(
                participant=participant, posture=posture, stage="split_calibrated_60_initial_lung",
                sizes_mm="60;100;140", **{f"rho_{m}_ohm_m": rho[m] for m in MATERIALS},
                **{f"singular_value_{i}": sv for i, sv in enumerate(singular_values, start=1)},
                condition_number=singular_values[0] / singular_values[-1],
                **{f"weak_direction_dln_{m}": w for m, w in zip(MATERIALS, weak)},
                weak_direction_predicted_dlnZ_norm=np.linalg.norm(relative @ weak),
            ))
    return results


if __name__ == "__main__":
    rows = evaluate()
    with (ROOT / "tissue_separation.csv").open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
