"""Supplementary L1 base/volume quadrature control on the same FEM operator."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from heart_montage_study_prepare import read, sha, write_new
from heart_montage_compensation import load_state
from heart_shape_electrical_comparison import Sources, write_json, write_csv, require


def prepare(study_path, plan_path, output):
    sp, pp, out = map(lambda p: Path(p).resolve(), (study_path, plan_path, output))
    study, plan = read(sp), read(pp)
    require(study["schema"] == "heart_montage_study_v1", "Unknown primary study")
    require(plan["schema"] == "heart_stl_materials_v1", "Unknown material plan")
    require(len(plan["variants"]) == 3 and all(r["quadrature"] == 1024 for r in plan["variants"]), "Require three q1024 states")
    require({r["id"] for r in plan["variants"]} == {"L1_base_q1024", "L1_Vm1_q1024", "L1_Vp1_q1024"}, "Unexpected refinement grid")
    out.mkdir(parents=True, exist_ok=True)
    sources = Sources(); sources.add(sp); sources.add(pp); sources.add(__file__)
    for p, h in study["source_sha256"].items(): sources.add(p, h)
    for p, h in plan["sources"].items(): sources.add(p, h)
    for r in plan["variants"]: sources.add(pp.parent/r["material_file"], r["material_sha256"])
    configs, records = [], []
    for m in study["montages"]:
        template = next(r for r in study["records"] if r["montage"] == m["id"] and r["probe"]["quadrature"] == 256)
        cfg = copy.deepcopy(read(template["config"]))
        sources.add(cfg["mesh_path"], plan["mesh_sha256"])
        sources.add(cfg["stl_path"], plan["stl_sha256"])
        require(plan["names"] == ["soft_background", "lung", "bone", "heart", "fat"], "Material order")
        cfg.update(output_dir=str(out/m["id"]), material_plan=str(pp), jobs=[])
        cp = out/(m["id"]+".local.json")
        for scenario, rho in study["scenarios"].items():
            for r in plan["variants"]:
                job_id = scenario+"_"+r["id"]
                cfg["jobs"].append(dict(id=job_id, variant=r["id"], scenario=scenario, rho_ohm_m=rho, eit=False))
                old_id = r["id"].replace("_q1024", "_q256")
                previous = next(x for x in study["records"] if x["montage"] == m["id"] and
                                x["scenario"] == scenario and x["probe"]["id"] == old_id)
                records.append(dict(montage=m["id"], scenario=scenario, variant=r["id"],
                    path=str(out/m["id"]/(job_id+".mat")), config=str(cp), rho_ohm_m=rho,
                    material_sha256=r["material_sha256"], previous_record=previous))
        write_new(cp, cfg); sources.add(cp)
        configs.append(dict(montage=m["id"], path=str(cp), jobs=len(cfg["jobs"])))
    sources.unchanged(rehash=True)
    write_json(out/"study.local.json", dict(schema="heart_montage_volume_refinement_v1", records=records,
        configurations=configs, source_hashes=sources.records, status="prepared_not_solved",
        scope="L1_base_and_one_percent_volume_only_no_global_mesh_convergence"))
    print("Prepared", len(records), "L1 q1024 states")


def analyze(study_path, output):
    sp, out = Path(study_path).resolve(), Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    s = read(sp); require(s["schema"] == "heart_montage_volume_refinement_v1", "Unknown refinement")
    sources = Sources(); sources.add(sp); sources.add(__file__)
    for p, h in s["source_hashes"].items(): sources.add(p, h["sha256"])
    rows = []
    for montage, scenario in sorted({(r["montage"], r["scenario"]) for r in s["records"]}):
        rr = {r["variant"]:r for r in s["records"] if r["montage"]==montage and r["scenario"]==scenario}
        values = {}
        for q in (256, 1024):
            states = [load_state(rr["L1_"+stem+"_q1024"] if q==1024 else
                       rr["L1_"+stem+"_q1024"]["previous_record"], sources) for stem in ("base", "Vm1", "Vp1")]
            base, minus, plus = states
            hm, hp = base["v"]-minus["v"], plus["v"]-base["v"]
            require(hm>0 and hp>0, "Wrong volume steps")
            left, right = (base["z"]-minus["z"])/hm, (plus["z"]-base["z"])/hp
            values[q] = (hp*left+hm*right)/(hm+hp)
            rows.append(dict(montage=montage, scenario=scenario, shape="L1", quadrature=q,
                K_V_ohm_per_ml=values[q], left=left, right=right,
                baseline_volume_ml=base["v"], baseline_Z_ohm=base["z"],
                minus_step_ml=hm, plus_step_ml=hp, global_mesh_convergence_proven=False))
        rows[-1]["relative_change_from_q256"] = None if values[1024]==0 else (values[1024]-values[256])/abs(values[1024])
    sources.unchanged(rehash=True)
    write_csv(out/"volume_refinement.csv", rows)
    write_json(out/"analysis.local.json", dict(schema="heart_montage_volume_refinement_analysis_v1", rows=rows,
        source_hashes=sources.records, scope=s["scope"], converged=None, clinical_validation=False))
    print("Analyzed", len(rows), "volume derivatives")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "analyze"])
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--plan", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.stage == "prepare":
        require(a.plan is not None, "--plan is required")
        prepare(a.study, a.plan, a.output)
    else: analyze(a.study, a.output)
