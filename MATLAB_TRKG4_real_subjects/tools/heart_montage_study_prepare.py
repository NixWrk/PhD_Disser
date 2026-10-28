"""Matched smaller-step and combined cardiac probes on the existing CT mesh.

Local input/output paths are explicit. Geometry and material integration reuse
the existing STL adapter; electrical solves remain under run_pipeline.
No measured physiological range, noise distribution or tissue truth is assumed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np


SHAPES = ("STL", "G1", "L1")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def number_id(value):
    return ("p" if value > 0 else "m") + f"{abs(value):g}".replace(".", "d")


def geometry(data_root, output):
    b, out = Path(data_root).resolve(), Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    target = out / "geometry_probes.local.json"
    if target.exists():
        raise FileExistsError(target)
    study_path = b / "shape_electrical_study_v1.local.json"
    study = read(study_path)
    for path, digest in study["sources_sha256"].items():
        if sha(path) != digest:
            raise ValueError("Changed preceding study source: " + path)
    cfg_path = b / "shape_materials_v1.local.json"
    cfg = read(cfg_path)
    candidates = read(cfg["candidates_file"])
    merged_path = Path(study["material_plan"])
    merged = read(merged_path)
    old = {r["id"]: r for r in merged["variants"]}
    reference = float(candidates["STL"]["volume_ml"])
    probes, jobs = [], []
    old_records = []

    def add(shape, kind, axis, step, factor, shift, q, old_id=None):
        shift = np.asarray(shift, dtype=float)
        if kind == "base":
            stem = shape + "_base"
        elif kind == "joint":
            stem = shape + "_joint_" + number_id(step)
        else:
            stem = shape + "_" + axis + number_id(step)
        var = f"{stem}_q{q}"
        method = stem
        candidate = copy.deepcopy(candidates[shape])
        candidate["center_mm"] = (np.asarray(candidate["center_mm"]) + shift).tolist()
        candidate["method_id"] = method
        candidate["numerical_probe_not_physiology"] = True
        if candidate["sampling_family"] == "stl":
            matrix = np.eye(4)
            matrix[:3, :3] *= candidate["scale"]
            matrix[:3, 3] = np.asarray(candidate["center_mm"]) - candidate["scale"] * np.asarray(candidate["reference_centroid_mm"])
            candidate["transform_world"] = matrix.tolist()
        candidates[method] = candidate
        row = dict(id=var, shape=shape, kind=kind, axis=axis, signed_step=step,
                   volume_factor=factor, nominal_delta_volume_ml=reference*(factor-1),
                   center_shift_mm=shift.tolist(), quadrature=q,
                   material_variant=old_id or var, material_reused=old_id is not None)
        if old_id:
            material = copy.deepcopy(old[old_id])
            material["material_file"] = str((merged_path.parent / material["material_file"]).resolve())
            if sha(material["material_file"]) != material["material_sha256"]:
                raise ValueError("Changed reused map: " + old_id)
            old_records.append(material)
        else:
            jobs.append(dict(id=var, method=method, factor=factor, q=q))
        probes.append(row)

    for shape in SHAPES:
        for q in (64, 256):
            add(shape, "base", "", 0., 1., [0, 0, 0], q,
                f"{shape}_v100_q64" if q == 64 else None)
            for percent in ((1., 2.5, 5.) if q == 64 else (1.,)):
                for sign in (-1, 1):
                    fraction = sign*percent/100
                    old_id = f"{shape}_v{95 if sign < 0 else 105:03d}_q64" if q == 64 and percent == 5 else None
                    add(shape, "volume", "V", sign*percent, 1+fraction, [0, 0, 0], q, old_id)
            for axis in "xyz":
                for distance in ((1., 2.5, 5.) if q == 64 else (1.,)):
                    for sign in (-1, 1):
                        shift = np.zeros(3); shift["xyz".index(axis)] = sign*distance
                        old_id = f"{shape}_{axis}{'m' if sign < 0 else 'p'}5_v100_q64" if q == 64 and distance == 5 else None
                        add(shape, "translation", axis, sign*distance, 1., shift, q, old_id)
            if q == 64:
                for sign in (-1, 1):
                    add(shape, "joint", "Vxyzrho", sign, 1+sign*.01,
                        np.array([1., -1., 1.])*sign, q)
    if len(probes) != 108 or len(jobs) != 81 or len(old_records) != 27:
        raise ValueError("Unexpected matched geometric grid")
    controls = []
    for shape in SHAPES:
        control_id = shape + "_base_control_q64"
        jobs.append(dict(id=control_id, method=shape+"_base", factor=1., q=64))
        controls.append(dict(shape=shape, prepared_variant=control_id,
                             reused_variant=shape+"_v100_q64"))
    candidates_path = out / "geometry_candidates.local.json"
    material_cfg_path = out / "materials.local.json"
    cfg["candidates_file"] = str(candidates_path)
    cfg["output_dir"] = str(out / "materials")
    cfg["variants"] = jobs
    # Select one envelope for all q levels, including reused extreme probes.
    # Keep the entire OLD integration envelope as well: the candidate-only
    # envelope can be narrower in one axis despite having more total elements.
    old_selection = copy.deepcopy(cfg["selection_variants"])
    cfg["selection_variants"] = old_selection + [dict(id=p["id"], method=p["id"].rsplit("_q", 1)[0],
                                           factor=p["volume_factor"], q=p["quadrature"]) for p in probes]
    write_new(candidates_path, candidates)
    write_new(material_cfg_path, cfg)
    source_files = [study_path, cfg_path, merged_path, candidates_path, material_cfg_path, Path(__file__)]
    write_new(target, dict(schema="heart_montage_geometry_v1", shapes=list(SHAPES),
        reference_volume_ml=reference, probes=probes, reused_materials=old_records,
        baseline_controls=controls,
        material_config=str(material_cfg_path), material_plan=str(out / "materials/materials_plan.local.json"),
        scenarios=study["scenarios"], frequency_Hz=study["frequency_Hz"], current_A=study["current_A"],
        reference_electrical_config=study["new_config"],
        source_sha256={str(p): sha(p) for p in source_files},
        rho_origin="Two conditional common scenarios retained from prior STL/EIT study; not independent tissue measurements",
        limits=["Computational probes, not measured motion or cardiac amplitudes",
                "q256 checks only base and smallest geometry steps; no global mesh convergence",
                "Frozen background convention retained; q64/q256 improve local heart AND background integration",
                "Joint geometry sign also determines +/-1 percent heart rho in subsequent electrical jobs",
                "Alternative electrode models are not simultaneous measurements"] ))
    print(json.dumps(dict(status="geometry_ready", probes=len(probes), new_materials=len(jobs), reused=len(old_records))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["geometry"])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    geometry(args.data_root, args.output)
