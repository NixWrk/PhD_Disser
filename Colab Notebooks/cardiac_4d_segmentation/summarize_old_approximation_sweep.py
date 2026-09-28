"""Aggregate sealed 4D CT approximation fits for scientific comparison.

No individual medical geometry or source CT is exported by this script.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_reduced_heart_geometry import digest, write_json


METHODS = [
    "T0", "I1", "I2", "I3", "G1", "G2", "G3a", "G3b", "G3c", "G4a", "G4b", "G5",
    "L1", "L2", "L3a", "L3b", "L3c", "L3d", "L4a", "L4b",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def run(args):
    cores, library = [Path(x) for x in args.core], Path(args.library)
    library_status = load(library / "status.json")
    if library_status["status"] != "completed" or library_status["completed"] != 540:
        raise ValueError(f"MATLAB library scoring incomplete: {library_status}")
    cases = sorted(path.name for path in library.glob("historical_*") if path.is_dir())
    if len(cases) != 3 or len(cores) != len(cases):
        raise ValueError("Provide one core output root per CT case in sorted order")
    records = {}
    core_protocols, core_statuses = [], []
    for case, root in zip(cases, cores):
        protocol, status = load(root / "protocol.json"), load(root / "status.json")
        core_protocols.append(protocol); core_statuses.append(status)
        if protocol["methods"] != METHODS[:20]:
            raise ValueError(f"Incomplete project method registry: {case}")
        if (protocol["code_sha256"] != core_protocols[0]["code_sha256"]
                or protocol["settings"] != core_protocols[0]["settings"]):
            raise ValueError(f"Core partitions use different code or settings: {case}")
        paths = list((root / case).glob("*.json"))
        if len(paths) != 400:
            raise ValueError(f"Incomplete project partition {case}: {len(paths)}/400")
        for path in paths:
            row = load(path)
            if row["case_id"] != case:
                raise ValueError(f"Wrong case in {path}")
            key = (case, row["phase_id"], row["method_id"])
            if key in records:
                raise ValueError(f"Duplicate result {key}")
            row["family"] = "project"
            records[key] = row
    for path in library.glob("historical_*/*.json"):
        row = load(path)
        key = (row["case_id"], row["phase_id"], row["method_id"])
        if key in records:
            raise ValueError(f"Duplicate result {key}")
        row["family"] = "matlab"
        records[key] = row
    for case in cases:
        for phase in (f"phase_{i:02d}" for i in range(20)):
            target_hashes = records[(case, phase, "T0")]["input_hashes"]
            if any(records[(case, phase, method)]["input_hashes"] != target_hashes
                   for method in METHODS):
                raise ValueError(f"Different CT masks across methods: {case} {phase}")
    phases = sorted({x[1] for x in records})
    if len(cases) != 3 or len(phases) != 20:
        raise ValueError("Expected three complete twenty-phase CT series")
    expected = {(c, p, m) for c in cases for p in phases for m in METHODS}
    if set(records) != expected:
        raise ValueError(f"Incomplete method matrix: missing {len(expected-set(records))}, extra {len(set(records)-expected)}")
    by_case_method = defaultdict(list)
    failures = []
    for key, row in sorted(records.items()):
        if row.get("metrics") is None:
            failures.append({"case": key[0], "phase": key[1], "method": key[2],
                             "status": row["status"], "reason": row.get("reason")})
        else:
            by_case_method[(key[0], key[2])].append(row)
    summary = []
    for case in cases:
        for method in METHODS:
            rows = by_case_method[(case, method)]
            values = lambda field: np.array([r["metrics"][field] for r in rows], float)
            if rows:
                dice, volume, center = values("dice"), values("volume_error_percent"), values("center_shift_mm")
                summary.append(dict(case=case, method=method, family=rows[0]["family"],
                    successful_phases=len(rows), median_dice=float(np.median(dice)),
                    min_dice=float(np.min(dice)), median_absolute_volume_error_percent=float(np.median(np.abs(volume))),
                    max_absolute_volume_error_percent=float(np.max(np.abs(volume))),
                    median_center_shift_mm=float(np.median(center)),
                    max_center_shift_mm=float(np.max(center))))
            else:
                summary.append(dict(case=case, method=method, successful_phases=0))
    # Compare changes relative to the same phase-zero anchor, never absolute scanner origins.
    trajectories = []
    for case in cases:
        for method in METHODS:
            anchor = records[(case, "phase_00", method)]
            origin = (np.asarray(anchor["parameters"]["center_mm"], float)
                      if anchor.get("parameters") else np.asarray(anchor["target_center_ras_mm"], float))
            for phase in phases:
                row = records[(case, phase, method)]
                parameters = row.get("parameters") or {}
                center = np.asarray(parameters.get("center_mm", row["target_center_ras_mm"]), float)
                delta = center - origin
                trajectories.append(dict(case=case, phase=phase, method=method,
                    phase_percent=row["phase_percent"], dx_ras_mm=float(delta[0]),
                    dy_ras_mm=float(delta[1]), dz_ras_mm=float(delta[2]),
                    frontal_path_from_phase0_mm=float(np.linalg.norm(delta[[0, 2]])),
                    status=row["status"]))
    by_trajectory = {(r["case"], r["phase"], r["method"]): r for r in trajectories}
    for row in summary:
        case, method = row["case"], row["method"]
        distances = []
        for phase in phases:
            actual = by_trajectory[(case, phase, method)]
            target = by_trajectory[(case, phase, "T0")]
            distances.append(np.hypot(actual["dx_ras_mm"] - target["dx_ras_mm"],
                                      actual["dz_ras_mm"] - target["dz_ras_mm"]))
        row["rms_frontal_displacement_error_mm"] = float(np.sqrt(np.mean(np.square(distances))))
        row["max_frontal_displacement_error_mm"] = float(np.max(distances))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("method_summary.csv", summary), ("trajectories.csv", trajectories)):
        with (output / filename).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
            writer.writeheader(); writer.writerows(rows)
    write_json(output / "summary.json", dict(schema="old_approximation_summary_v1",
        core_statuses=core_statuses, library_status=library_status,
        core_protocol_sha256=[digest(core / "protocol.json") for core in cores],
        library_protocol_sha256=digest(library / "protocol.json"),
        cases=cases, phases=phases, methods=METHODS, failures=failures,
        table=summary, rfbr_direct_numerical_comparison=False,
        target="unfilled_union_of_four_automatic_blood_chamber_masks",
        note="T0 is a self-comparison; historical RFBR subject and segmentation equivalence not verified"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", required=True, nargs="+")
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
