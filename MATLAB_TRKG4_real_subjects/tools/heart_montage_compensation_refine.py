"""One Newton correction of the target montage's finite compensation residual.

This deliberately constructs ambiguous states; it is not volume validation.
The same corrected rho is evaluated unchanged in all alternative montages.
FEM remains in run_pipeline/heart_stl_electrical_check.
"""
from __future__ import annotations
import argparse
import copy
from pathlib import Path

from heart_montage_study_prepare import read, write_new
from heart_montage_compensation import load_state
from heart_shape_electrical_comparison import Sources, write_json, require


def prepare(manifest, output):
    mp, out = Path(manifest).resolve(), Path(output).resolve()
    old = read(mp); require(old["schema"] == "heart_montage_compensation_v1", "Unknown input")
    require(not old.get("newton_iteration"), "This adapter performs exactly one correction")
    source = Sources(); source.add(mp); source.add(__file__)
    for p, h in old["source_hashes"].items(): source.add(p, h["sha256"])
    targets = {r["compensation_target_montage"] for r in old["records"]}
    require(len(targets) == 1, "One target montage required")
    target = targets.pop(); changes = {}
    for r in old["records"]:
        if r["montage"] != target: continue
        base, current = load_state(r["base_record"], source), load_state(r, source)
        residual, derivative = current["z"]-base["z"], current["jr"][3]
        require(derivative != 0, "Cannot refine with zero derivative")
        changes[r["scenario"], r["shape"], r["sign"]] = dict(
            update_ohm_m=-residual/derivative, old_residual_ohm=residual,
            current_KrhoH=derivative, target_state=r["path"])
    require(len(changes) == 12, "All 12 target states required")
    out.mkdir(parents=True, exist_ok=True)
    configs, records = {}, []
    for previous in old["records"]:
        r = copy.deepcopy(previous); name = r["montage"]
        change = changes[r["scenario"], r["shape"], r["sign"]]
        if name not in configs:
            cfg = copy.deepcopy(read(r["config"])); cfg.update(output_dir=str(out/name), jobs=[])
            configs[name] = cfg
        r["rho_ohm_m"][3] += change["update_ohm_m"]
        require(r["rho_ohm_m"][3] > 0, "Corrected rho is nonpositive")
        rho0 = r["base_record"]["rho_ohm_m"][3]
        r["delta_rho_heart_ohm_m"] = r["rho_ohm_m"][3]-rho0
        r["relative_delta_rho_heart"] = r["delta_rho_heart_ohm_m"]/rho0
        r["outside_previously_probed_rho_fraction"] = abs(r["relative_delta_rho_heart"]) > .05
        r["target_predicted_linear_delta_ohm"] = (
            r["target_K_V_ohm_per_ml"]*r["delta_volume_ml"] +
            r["target_K_rhoH_ohm_per_ohm_m"]*r["delta_rho_heart_ohm_m"])
        job_id = Path(previous["path"]).stem+"_newton1"
        r.update(path=str(out/name/(job_id+".mat")), config=str(out/(name+".local.json")),
                 newton_iteration=1, refinement=change, prior_compensation_record=previous)
        variant = r["volume_only_record"]["probe"]["material_variant"]
        configs[name]["jobs"].append(dict(id=job_id, variant=variant,
            scenario=r["scenario"], rho_ohm_m=r["rho_ohm_m"], eit=False))
        records.append(r)
    configurations = []
    for name, cfg in configs.items():
        cp = out/(name+".local.json"); write_new(cp, cfg); source.add(cp)
        configurations.append(dict(montage=name, path=str(cp), jobs=len(cfg["jobs"])))
    source.unchanged(rehash=True)
    write_json(out/"study.local.json", dict(schema="heart_montage_compensation_v1", records=records,
        source_study=str(mp), configurations=configurations, source_hashes=source.records,
        status="prepared_not_solved", newton_iteration=1, stack_kind="alternative_montages",
        scope="one_Newton_correction_chosen_to_cancel_target_montage_not_volume_validation",
        measured_noise=None, clinical_validation=False))
    print("Prepared", len(records), "finite one-Newton compensation states")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); prepare(a.manifest, a.output)
