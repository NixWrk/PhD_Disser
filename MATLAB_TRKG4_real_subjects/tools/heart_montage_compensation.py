"""Finite FEM controls of a volume--heart-rho cancellation predicted locally.

Preparation uses completed q256 small-step probes. The cancellation is chosen
for one declared montage and evaluated unchanged in every montage. No noise
floor, physiological range or exact nonlinear cancellation is assumed.
"""
from __future__ import annotations

import argparse
import copy
from functools import cache
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from heart_montage_study_prepare import read, sha, write_new
from heart_shape_electrical_comparison import Sources, write_csv, write_json, require


@cache
def sensitivity_source(root):
    found = list(Path(root).glob("**/cr_compartment_sensitivity.m"))
    require(len(found) == 1, "Ambiguous/missing sensitivity source")
    return found[0]


def load_state(record, sources):
    path, cp = Path(record["path"]), Path(record["config"])
    sources.add(path, record.get("mat_sha256"))
    d = loadmat(path, simplify_cells=True)
    identity, row = d["identity"], d["row"]
    sources.add(cp, identity["config_sha256"])
    cfg = read(cp)
    sources.add(cfg["material_plan"], identity["plan_sha256"])
    for name in ("ttrkg", "eit"):
        sources.add(cfg[name + "_model_file"], identity[name + "_model_sha256"])
    sources.add(Path(__file__).with_name("heart_stl_electrical_check.m"), identity["implementation_sha256"])
    require("sensitivity_sha256" in identity, "Missing sensitivity implementation identity")
    sources.add(sensitivity_source(cfg["pipeline_root"]), identity["sensitivity_sha256"])
    for key in ("native_forward_relative_error", "reciprocity_relative_error"):
        if key in row:
            require(np.isfinite(row[key]) and row[key] < 1e-8, "Saved solver QC failed: " + key)
    require(row["material_sha256"] == record["material_sha256"], "Different material")
    require(np.array_equal(row["rho_ohm_m"], record["rho_ohm_m"]), "Different rho")
    require(np.allclose(d["sigma"], 1 / np.array(record["rho_ohm_m"]), rtol=1e-14), "Sigma mismatch")
    z = np.asarray(d["Ztt"])
    k = np.asarray(d["Ktt"]).reshape(5)
    require(np.isrealobj(z) and z.size == 1 and np.isrealobj(k), "Real signed scalar TT required")
    require(np.isfinite(np.r_[z.ravel(), k]).all(), "Nonfinite result")
    require(float(z) == float(row["Ztt_ohm"]), "Saved Z mismatch")
    return dict(z=float(z), v=float(row["volume_ml"][3]),
                jr=-k / np.array(record["rho_ohm_m"])**2, row=row, cfg=cfg)


def prepare(manifest, output, target_montage):
    mp, out = Path(manifest).resolve(), Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = Sources(); source.add(mp); source.add(__file__)
    study = read(mp)
    require(study["schema"] == "heart_montage_study_v1", "Unknown study")
    for p, h in study["source_sha256"].items():
        source.add(p, h)
    names = [m["id"] for m in study["montages"]]
    require(target_montage in names, "Select an existing target montage")
    rows = {(r["montage"], r["scenario"], r["probe"]["id"]): r for r in study["records"]}
    configs = {}; records = []
    for scenario, rho in study["scenarios"].items():
        for shape in study["shapes"]:
            states = {}
            for name in names:
                family = []
                for stem in ("base", "Vm1", "Vp1"):
                    r = rows[name, scenario, f"{shape}_{stem}_q256"]
                    family.append((r, load_state(r, source) if name == target_montage else None))
                states[name] = family
                if name not in configs:
                    cfg = copy.deepcopy(read(family[0][0]["config"]))
                    cfg.update(output_dir=str(out/name), jobs=[])
                    configs[name] = cfg
            target_base, tm, tp = [x[1] for x in states[target_montage]]
            hm, hp = target_base["v"]-tm["v"], tp["v"]-target_base["v"]
            require(hm > 0 and hp > 0, "Wrong volume ordering")
            kt = (hp*(target_base["z"]-tm["z"])/hm + hm*(tp["z"]-target_base["z"])/hp)/(hm+hp)
            kh = target_base["jr"][3]
            require(kh != 0, "Cannot compensate using a zero heart-rho derivative")
            for index, sign in ((1, "m"), (2, "p")):
                dv = states[target_montage][index][1]["v"] - target_base["v"]
                drho = -kt * dv / kh
                values = list(rho); values[3] += drho
                require(values[3] > 0, "Predicted compensated rho is nonpositive")
                for name in names:
                    base_r, base = states[name][0]
                    changed_r, changed = states[name][index]
                    require(base_r["material_sha256"] == states[target_montage][0][0]["material_sha256"] and
                            changed_r["material_sha256"] == states[target_montage][index][0]["material_sha256"],
                            "Montages use different material maps")
                    job_id = f"{scenario}_{shape}_V{sign}1_rho_compensated_q256"
                    configs[name]["jobs"].append(dict(id=job_id, variant=changed_r["probe"]["material_variant"],
                        scenario=scenario, rho_ohm_m=values, eit=False))
                    records.append(dict(montage=name, scenario=scenario, shape=shape, sign=sign,
                        path=str(out/name/(job_id+".mat")), config=str(out/(name+".local.json")),
                        material_sha256=changed_r["material_sha256"], rho_ohm_m=values,
                        base_record=base_r, volume_only_record=changed_r,
                        derivative_records=[x[0] for x in states[name]],
                        compensation_target_montage=target_montage, delta_volume_ml=dv,
                        delta_rho_heart_ohm_m=drho, relative_delta_rho_heart=drho/rho[3],
                        outside_previously_probed_rho_fraction=abs(drho/rho[3]) > .05,
                        target_K_V_ohm_per_ml=kt, target_K_rhoH_ohm_per_ohm_m=kh,
                        target_predicted_linear_delta_ohm=kt*dv+kh*drho))
    configurations = []
    for name, cfg in configs.items():
        cp = out/(name+".local.json"); write_new(cp, cfg); source.add(cp)
        configurations.append(dict(montage=name, path=str(cp), jobs=len(cfg["jobs"])))
    source.unchanged(rehash=True)
    write_json(out/"study.local.json", dict(schema="heart_montage_compensation_v1", records=records,
        source_study=str(mp), configurations=configurations, source_hashes=source.records,
        status="prepared_not_solved", stack_kind="alternative_montages",
        scope="local_linear_cancellation_tested_by_saved_finite_FEM_not_physiological_inference",
        measured_noise=None, clinical_validation=False))
    print(f"Prepared {len(records)} finite compensated states; target {target_montage}")


def analyze(manifest, output):
    mp, out = Path(manifest).resolve(), Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    study = read(mp); require(study["schema"] == "heart_montage_compensation_v1", "Unknown compensation study")
    source = Sources(); source.add(mp); source.add(__file__)
    for p, h in study["source_hashes"].items(): source.add(p, h["sha256"])
    rows = []
    for r in study["records"]:
        base = load_state(r["base_record"], source)
        volume = load_state(r["volume_only_record"], source)
        comp = load_state(r, source)
        dv = comp["v"]-base["v"]; require(dv == r["delta_volume_ml"], "Compensated geometry differs")
        dz, dz_v = comp["z"]-base["z"], volume["z"]-base["z"]
        _, minus, plus = [load_state(x, source) for x in r["derivative_records"]]
        hm, hp = base["v"]-minus["v"], plus["v"]-base["v"]
        require(hm > 0 and hp > 0, "Wrong volume derivative ordering")
        kv = (hp*(base["z"]-minus["z"])/hm + hm*(plus["z"]-base["z"])/hp)/(hm+hp)
        predicted = kv*dv + base["jr"][3]*r["delta_rho_heart_ohm_m"]
        rows.append(dict(montage=r["montage"], scenario=r["scenario"], shape=r["shape"], sign=r["sign"],
            target_montage=r["compensation_target_montage"], delta_volume_ml=dv,
            delta_rho_heart_ohm_m=r["delta_rho_heart_ohm_m"],
            relative_delta_rho_heart=r["relative_delta_rho_heart"],
            outside_previously_probed_rho_fraction=r["outside_previously_probed_rho_fraction"],
            uncompensated_delta_ohm=dz_v, compensated_delta_ohm=dz,
            predicted_linear_delta_ohm=predicted,
            nonlinear_remainder_ohm=dz-predicted,
            absolute_response_ratio=None if dz_v==0 else abs(dz/dz_v),
            apparent_delta_volume_ml=None if kv==0 else dz/kv,
            apparent_volume_error_ml=None if kv==0 else dz/kv-dv,
            K_V_ohm_per_ml=kv, base_Z_ohm=base["z"], compensated_Z_ohm=comp["z"],
            one_Newton_rho_update_ohm_m=None if comp["jr"][3]==0 else -dz/comp["jr"][3],
            exact_nonlinear_cancellation_claimed=False))
    source.unchanged(rehash=True)
    write_csv(out/"finite_compensation.csv", rows)
    write_json(out/"analysis.local.json", dict(schema="heart_montage_compensation_analysis_v1", rows=rows,
        source_hashes=source.records, status="finite_compensation_evaluated", measured_noise=None,
        source_content_rehashed_after_analysis=True,
        limits=["Alternative montages are not simultaneous measurements", "No measured detection limit",
                "Finite remainder includes numerical integration and nonlinearity", "Not SV/EF validation"]))
    print(f"Analyzed {len(rows)} compensated finite FEM states")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "analyze"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-montage", default="top_gap100")
    args = parser.parse_args()
    if args.stage == "prepare": prepare(args.manifest, args.output, args.target_montage)
    else: analyze(args.manifest, args.output)
