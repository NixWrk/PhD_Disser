"""Bind completed material/contact QC to the established FEM profile.

This is an input adapter, not an electrical solver. All paths are supplied by
local manifests. Previously solved states are reused only for the unchanged
original montage and with matching material, scenario and operator hashes.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from heart_montage_study_prepare import read, sha, write_new, number_id


def require(value, message):
    if not value:
        raise ValueError(message)


def prepare(geometry_path, contact_path, output, threads, quadrature=None, material_snapshot=None):
    gp, cp, out = map(lambda p: Path(p).resolve(), (geometry_path, contact_path, output))
    out.mkdir(parents=True, exist_ok=True)
    require(not (out / "study.local.json").exists(), "Study already exists; do not overwrite")
    g, contacts = read(gp), read(cp)
    require(g["schema"] == "heart_montage_geometry_v1", "Unknown geometry schema")
    require(contacts["schema"] == "heart_montage_contacts_v1" and contacts["status"] == "ready", "Contacts not ready")
    require(contacts["mesh"]["exact_EIT_TTRKG_raw_mesh_identity"], "Different meshes")
    source_hashes = dict(g["source_sha256"])
    for p, h in source_hashes.items():
        require(sha(p) == h, "Changed geometry source: " + p)
    for p in (gp, cp, Path(__file__).resolve()):
        source_hashes[str(p)] = sha(p)
    if quadrature is not None:
        g["probes"] = [p for p in g["probes"] if p["quadrature"] == quadrature]
    material_path = Path(material_snapshot or g["material_plan"]).resolve()
    plan = read(material_path)
    source_hashes[str(material_path)] = sha(material_path)
    cfg0 = read(g["reference_electrical_config"])
    for key in ("frequency_Hz", "current_A"):
        require(g[key] == cfg0[key], "Geometry/electrical physical context differs: " + key)
    require(plan["names"] == ["soft_background", "lung", "bone", "heart", "fat"], "Material order")
    for field, key in (("mesh_path", "mesh_sha256"), ("stl_path", "stl_sha256")):
        require(sha(cfg0[field]) == plan[key], "Source mesh/STL changed")
        source_hashes[cfg0[field]] = plan[key]
    require(contacts["provenance"]["sources"]["mesh_file"]["sha256"] == plan["mesh_sha256"], "Contact mesh differs")
    require(contacts["provenance"]["sources"]["eit_model_file"]["sha256"] == sha(cfg0["eit_model_file"]), "EIT identity differs")
    for p, h in plan["sources"].items():
        require(sha(p) == h, "Changed material source: " + p)
        source_hashes[p] = h
    material_rows = {}
    for r in [*g["reused_materials"], *plan["variants"]]:
        r = copy.deepcopy(r)
        p = Path(r["material_file"])
        if not p.is_absolute():
            p = material_path.parent / p
        p = p.resolve()
        require(sha(p) == r["material_sha256"], "Material hash differs: " + r["id"])
        source_hashes[str(p)] = r["material_sha256"]
        r["material_file"] = os.path.relpath(p, out)
        require(r["id"] not in material_rows, "Duplicate material id")
        material_rows[r["id"]] = r
    controls = []
    for c in g["baseline_controls"]:
        a, b = (loadmat(out / material_rows[c[k]]["material_file"], variable_names=["fractions"])["fractions"]
                for k in ("prepared_variant", "reused_variant"))
        require(np.array_equal(a, b), "Background/baseline changed: " + c["shape"])
        controls.append(dict(**c, exact_fraction_array_equality=True, array_shape=list(a.shape)))
        del a, b
    used = {p["material_variant"] for p in g["probes"]}
    require(len(used) == len(g["probes"]), "Duplicate geometry states")
    merged = copy.deepcopy(plan)
    merged["variants"] = [material_rows[k] for k in sorted(used)]
    merged["source_geometry_manifest"] = str(gp)
    merged["baseline_controls"] = controls
    merged_path = out / "materials.local.json"
    write_new(merged_path, merged)
    source_hashes[str(merged_path)] = sha(merged_path)
    old_study_path = Path(g["reference_electrical_config"]).with_name("shape_electrical_study_v1.local.json")
    old = read(old_study_path)
    # Old source manifests must remain exactly the versions verified when the
    # geometry study was prepared; saved MAT hashes bind reused results too.
    require(sha(old_study_path) == g["source_sha256"][str(old_study_path)], "Old study changed")
    source_hashes[str(old_study_path)] = sha(old_study_path)
    old_records = {(r["scenario"], r["variant"]): r for r in old["probes"]}
    previous_audit_path = old_study_path.parent / "shape_electrical_comparison_v1/analysis.local.json"
    previous_audit = read(previous_audit_path)
    source_hashes[str(previous_audit_path)] = sha(previous_audit_path)
    reference_model_hash = sha(cfg0["ttrkg_model_file"])
    entries, configurations, montage_info = [], [], []
    for montage in contacts["montages"]:
        model = (cp.parent / montage["ttrkg_model_file"]).resolve()
        require(sha(model) == montage["model_sha256"], "Changed contact model")
        source_hashes[str(model)] = montage["model_sha256"]
        require(montage["current_A"] == g["current_A"], "Contact current differs")
        reuse = model == Path(cfg0["ttrkg_model_file"]).resolve() and sha(model) == reference_model_hash
        montage_id = montage["id"]
        cfg = copy.deepcopy(cfg0)
        cfg.pop("shape_comparison", None)
        cfg.update(threads=threads, output_dir=str(out / montage_id),
                   material_plan=str(merged_path), ttrkg_model_file=str(model),
                   montage_basis=montage["basis"] + "; computational alternative, not simultaneous acquired data")
        cfg["jobs"] = []
        config_path = out / (montage_id + ".local.json")
        for scenario, rho in g["scenarios"].items():
            probes = copy.deepcopy(g["probes"])
            for shape in (g["shapes"] if quadrature != 256 else []):
                base = next(p for p in probes if p["shape"] == shape and p["kind"] == "base" and p["quadrature"] == 64)
                for percent in (-5., -1., 1., 5.):
                    p = copy.deepcopy(base)
                    p.update(id=f"{shape}_rhoH{number_id(percent)}_q64", kind="rho", axis="rho_heart", signed_step=percent)
                    probes.append(p)
            # Small-step controls first; each form/scenario remains explicitly tagged.
            probes.sort(key=lambda p: (p["shape"], p["quadrature"], p["kind"] != "base", abs(p["signed_step"]), p["id"]))
            for p in probes:
                values = list(rho)
                if p["kind"] == "rho":
                    values[3] *= 1 + p["signed_step"] / 100
                elif p["kind"] == "joint":
                    values[3] *= 1 + .01 * p["signed_step"]
                row = dict(montage=montage_id, scenario=scenario, probe=p, rho_ohm_m=values,
                           material_sha256=material_rows[p["material_variant"]]["material_sha256"],
                           reused=False, format="electrical", probe_index=None)
                previous = old_records.get((scenario, p["material_variant"]))
                if reuse and previous and p["kind"] not in ("rho", "joint"):
                    saved_path = Path(previous["path"])
                    old_hash = previous_audit["source_hashes"][str(saved_path.resolve())]
                    require(old_hash["sha256"] == old_hash["sha256_after_analysis"], "Prior audit source changed")
                    require(sha(saved_path) == old_hash["sha256"], "Previously solved MAT changed")
                    if previous.get("mat_sha256"):
                        require(previous["mat_sha256"] == old_hash["sha256"], "Prior study/audit hashes differ")
                    d = loadmat(saved_path, simplify_cells=True)
                    require(previous["format"] == "electrical", "Reuse requires explicit electrical state")
                    require(d["identity"]["ttrkg_model_sha256"] == reference_model_hash, "Reused operator differs")
                    require(d["row"]["material_sha256"] == row["material_sha256"], "Reused material differs")
                    require(np.allclose(d["row"]["rho_ohm_m"], values, rtol=1e-13, atol=0), "Reused rho differs")
                    pc = Path(previous["config"])
                    require(sha(pc) == d["identity"]["config_sha256"], "Old config changed")
                    oldcfg = read(pc)
                    for key in ("frequency_Hz", "current_A", "breath_interval_status"):
                        require(oldcfg[key] == cfg[key], "Reused physical context differs: " + key)
                    row.update(path=str(saved_path), config=str(pc), reused=True, mat_sha256=sha(saved_path))
                    source_hashes[str(saved_path)] = row["mat_sha256"]
                    source_hashes[str(pc)] = sha(pc)
                else:
                    job_id = scenario + "_" + p["id"]
                    cfg["jobs"].append(dict(id=job_id, variant=p["material_variant"], scenario=scenario,
                                            rho_ohm_m=values, eit=False))
                    row.update(path=str(Path(cfg["output_dir"]) / (job_id + ".mat")), config=str(config_path))
                entries.append(row)
        write_new(config_path, cfg)
        source_hashes[str(config_path)] = sha(config_path)
        configurations.append(dict(montage=montage_id, path=str(config_path), jobs=len(cfg["jobs"]),
                                   output_dir=cfg["output_dir"], threads=threads))
        montage_info.append(dict(id=montage_id, model_path=str(model), model_sha256=sha(model), basis=cfg["montage_basis"]))
    expected = {None: (720, 54), 64: (558, 54), 256: (162, 0)}[quadrature]
    require((len(entries), sum(r["reused"] for r in entries)) == expected, "Unexpected job count")
    study = dict(schema="heart_montage_study_v1", geometry_manifest=str(gp), contact_manifest=str(cp),
                 material_plan=str(merged_path), baseline_controls=controls,
                 shapes=g["shapes"], scenarios=g["scenarios"], reference_volume_ml=g["reference_volume_ml"],
                 frequency_Hz=g["frequency_Hz"], current_A=g["current_A"],
                 breath_interval_status=cfg0["breath_interval_status"], montages=montage_info,
                 configurations=configurations, records=entries, source_sha256=source_hashes,
                 stack_kind="alternative_montages", measured_noise=None,
                 status="prepared_not_solved", quadrature_levels=sorted({p["quadrature"] for p in g["probes"]}), limits=g["limits"])
    write_new(out / "study.local.json", study)
    print(json.dumps(dict(status=study["status"], states=len(entries), reused=expected[1], configs=configurations), ensure_ascii=True))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--geometry", type=Path, required=True)
    p.add_argument("--contacts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--quadrature", type=int, choices=[64, 256])
    p.add_argument("--material-snapshot", type=Path,
                   help="Immutable copy of completed progress, beside the original materials")
    a = p.parse_args()
    require(a.threads > 0, "Positive thread count required")
    prepare(a.geometry, a.contacts, a.output, a.threads, a.quadrature, a.material_snapshot)
