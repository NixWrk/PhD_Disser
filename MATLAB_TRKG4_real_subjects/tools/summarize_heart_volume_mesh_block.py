"""Checked matched-quadrature comparison of three heart forms on two meshes."""
from pathlib import Path
import argparse
import json
import hashlib
import csv
import compare_heart_mesh_levels as mesh_compare
import heart_fractional_electrical_pilot as fractional

SHAPES = ("individual", "sphere", "ellipsoid")
METRICS = ("Sminus", "derivative", "Splus")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative_difference(value, reference):
    return None if reference == 0 else (value - reference) / abs(reference)


def shape_errors(reports):
    rows = []
    for level in ("coarse", "fine"):
        lookup = {g: {d["step"]: d for d in reports[g]["derivatives"][level]} for g in SHAPES}
        if any(set(lookup[g]) != {.005, .01} for g in SHAPES):
            raise ValueError("Expected the same two volume steps for every shape")
        for step in sorted(lookup["individual"]):
            for metric in METRICS:
                reference = lookup["individual"][step][metric]
                errors = {g: relative_difference(lookup[g][step][metric], reference) for g in SHAPES[1:]}
                absolute = {g: None if errors[g] is None else abs(errors[g]) for g in errors}
                margin = None if reference == 0 else absolute["sphere"] - absolute["ellipsoid"]
                preference = "undefined_zero_reference" if margin is None else (
                    "tie" if margin == 0 else "ellipsoid" if margin > 0 else "sphere")
                rows.append(dict(mesh=level, step=step, metric=metric,
                    individual_ohm_per_ml=reference,
                    sphere_ohm_per_ml=lookup["sphere"][step][metric],
                    ellipsoid_ohm_per_ml=lookup["ellipsoid"][step][metric],
                    sphere_absolute_relative_error=absolute["sphere"],
                    ellipsoid_absolute_relative_error=absolute["ellipsoid"],
                    preference_margin_sphere_minus_ellipsoid=margin, preferred=preference))
    return rows


def quadrature_history(reports, plans, old_individual, old_analytic, candidate_hash):
    rows, provenance = [], []
    for folder, geometries in ((Path(old_individual), ("individual",)), (Path(old_analytic), SHAPES[1:])):
        plan = fractional.load_plan(folder)
        reference = plans["individual", "coarse"]
        for key in ("source_prepared_sha256", "frequency_hz", "rho_ohm_m", "conductivity_by_tissue_id"):
            if plan[key] != reference[key]:
                raise ValueError("Quadrature-history physical mismatch: " + key)
        if plan["contact"]["sha256"] != reference["contact"]["sha256"]:
            raise ValueError("Quadrature-history contact mismatch")
        if geometries == ("individual",):
            if plan["original_geometric_source_sha256"] != reference["original_geometric_source_sha256"]:
                raise ValueError("Individual geometric source changed")
        else:
            identity_entry = next(x for x in plan["inputs"] if Path(x["path"]).name == "identity.json")
            identity = read(identity_entry["path"])
            if identity["candidate_parameters_sha256"] != candidate_hash:
                raise ValueError("Analytic quadrature history uses different candidate parameters")
        provenance.append(dict(plan_sha256=sha(folder / "plan.json"), points_per_tet=4096, geometries=geometries))
        for geometry in geometries:
            completed = {}
            for state in plan["states"]:
                if state["geometry"] == geometry and state["points_per_tet"] == 4096:
                    result = fractional.checked_result(folder, plan, state)
                    if result is None:
                        raise ValueError("Incomplete quadrature-history state")
                    completed[state["volume_fraction"]] = result
            if set(completed) != {0., -.005, .005, -.01, .01}:
                raise ValueError("Unexpected quadrature-history volume design")
            baseline = completed[0.]
            for refined in reports[geometry]["derivatives"]["coarse"]:
                step = refined["step"];delta = step * baseline["analytic_volume_ml"]
                if baseline["analytic_volume_ml"] != refined["baseline_analytic_volume_ml"]:
                    raise ValueError("Quadrature-history analytic volume changed")
                zm,z0,zp = (completed[h]["Z_ohm"] for h in (-step,0.,step))
                values = {"Sminus":(z0-zm)/delta,"derivative":(zp-zm)/(2*delta),"Splus":(zp-z0)/delta}
                for metric,value in values.items():
                    new=refined[metric]
                    rows.append(dict(geometry=geometry,step=step,metric=metric,
                        value_q4096=value,value_q16384=new,
                        absolute_relative_change_to_q16384=None if new==0 else abs(new-value)/abs(new)))
    return dict(mesh="coarse", rows=rows, provenance=provenance,
        fine_mesh_electrical_quadrature_refinement_tested=False)


def mesh_motion_scales(plans):
    import numpy as np
    reference=plans["individual", "coarse"]
    runtime=fractional.core.configure(Path(reference["runtime"]["deps"]),reference["threads"])
    edges={}
    for level in ("coarse","fine"):
        plan=plans["individual",level]
        with runtime.h5py.File(plan["source_prepared"]) as f:
            nodes=f["fmdl_m/nodes"][:].T
            elems=f["fmdl_m/elems"][:].T.astype(np.int64)-1
            labels=f["tissue_id"][:].ravel()
        heart=elems[labels==3]
        pairs=np.concatenate([heart[:,[i,j]] for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))])
        pairs=np.unique(np.sort(pairs,axis=1),axis=0)
        lengths=1000*np.linalg.norm(nodes[pairs[:,0]]-nodes[pairs[:,1]],axis=1)
        if not np.all(np.isfinite(lengths)) or np.any(lengths<=0):
            raise ValueError("Invalid cardiac-element edge lengths")
        edges[level]=dict(unique_edges=len(pairs),heart_tetrahedra=len(heart),
            minimum_mm=float(np.min(lengths)),median_mm=float(np.median(lengths)),
            percentile95_mm=float(np.percentile(lengths,95)),maximum_mm=float(np.max(lengths)))
    radius_mm=(3*reference["nominal_volume_ml"]*1000/(4*np.pi))**(1/3)
    motions=[dict(step=h,radius_increase_mm=radius_mm*((1+h)**(1/3)-1),
                  radius_decrease_magnitude_mm=radius_mm*(1-(1-h)**(1/3))) for h in (.005,.01)]
    return dict(edge_scope="unique edges incident to original-heart-material elements; not an error bound",edges=edges,
        equal_volume_sphere_radius_mm=radius_mm,sphere_radial_displacements=motions)


def run(block, reference_individual, output, previous_individual=None, previous_analytic=None):
    block, reference_individual, output = map(lambda p: Path(p).resolve(), (block, reference_individual, output))
    output.mkdir(parents=True, exist_ok=True)
    reports, plans, provenance = {}, {}, []
    for geometry in SHAPES:
        coarse = reference_individual if geometry == "individual" else block / ("L0_" + geometry + "_q16384")
        fine = block / ("L1_" + geometry + "_q16384")
        report = mesh_compare.compare(coarse, fine, output / geometry)
        if report["status"] != "complete" or report["state_designs"]["compared_completed"] != 5:
            raise ValueError("Incomplete matched mesh comparison: " + geometry)
        if report["quadrature"] != {"coarse_points_per_tet": [16384], "fine_points_per_tet": [16384]}:
            raise ValueError("Mixed quadrature would confound the mesh comparison")
        reports[geometry] = report
        for label, folder in (("coarse", coarse), ("fine", fine)):
            plan = read(folder / "plan.json")
            if sha(folder / "plan.json") != report[label + "_plan_sha256"]:
                raise ValueError("Plan changed after checked mesh comparison")
            plans[geometry, label] = plan
            provenance.append(dict(geometry=geometry, mesh=label, plan_sha256=sha(folder / "plan.json"),
                geometry_qc_sha256=sha(folder / "geometry_qc.json"), summary_sha256=sha(folder / "summary.json")))
    # Pairwise mesh checks above do not by themselves establish equal inputs across shapes.
    for level in ("coarse", "fine"):
        reference = plans["individual", level]
        for geometry in SHAPES[1:]:
            candidate = plans[geometry, level]
            for field in ("source_prepared_sha256", "original_geometric_source_sha256", "frequency_hz",
                          "rho_ohm_m", "conductivity_by_tissue_id", "contact", "electrode_model", "nominal_volume_ml"):
                if candidate[field] != reference[field]:
                    raise ValueError("Cross-shape mismatch: " + field)
    candidate_hashes = {}
    for geometry in SHAPES[1:]:
        for level in ("coarse", "fine"):
            values = [entry["sha256"] for entry in plans[geometry, level]["inputs"]
                      if Path(entry["path"]).name == "candidate_parameters.json"]
            if len(values) != 1:
                raise ValueError("Exactly one sealed candidate parameter file required")
            candidate_hashes[geometry + "_" + level] = values[0]
    if len(set(candidate_hashes.values())) != 1:
        raise ValueError("Candidate geometry differs between matched runs")
    errors = shape_errors(reports)
    stability = []
    for coarse in (r for r in errors if r["mesh"] == "coarse"):
        fine = next(r for r in errors if r["mesh"] == "fine" and r["step"] == coarse["step"] and r["metric"] == coarse["metric"])
        stability.append(dict(step=coarse["step"], metric=coarse["metric"],
            preferred_coarse=coarse["preferred"], preferred_fine=fine["preferred"],
            preference_unchanged=coarse["preferred"] == fine["preferred"],
            margin_coarse=coarse["preference_margin_sphere_minus_ellipsoid"],
            margin_fine=fine["preference_margin_sphere_minus_ellipsoid"]))
    result = dict(status="complete_matched_comparison", new_solutions=25,
        total_states_compared=30, montage="TEPC-2", points_per_tet=16384,
        units="ohm/ml", physical_settings={k: plans["individual", "coarse"][k] for k in ("frequency_hz", "rho_ohm_m", "nominal_volume_ml", "electrode_model", "criteria")},
        meshes={level: {k: plans["individual", level][k] for k in ("node_count", "element_count", "source_prepared_sha256")} for level in ("coarse", "fine")},
        baseline={g: reports[g]["baseline_Z_shift"] for g in SHAPES},
        derivatives={g: reports[g]["derivatives"] for g in SHAPES},
        derivative_mesh_changes={g: reports[g]["derivative_comparison"] for g in SHAPES},
        shape_errors=errors, preference_stability=stability,
        geometry_qc={g: {level: read((reference_individual if g == "individual" and level == "coarse" else block / (("L0_" if level == "coarse" else "L1_") + g + "_q16384")) / "geometry_qc.json")["rows"] for level in ("coarse", "fine")} for g in SHAPES},
        numerical_qc={g: reports[g]["numerical_qc"] for g in SHAPES},
        provenance=provenance, candidate_parameters_sha256=next(iter(candidate_hashes.values())), implementation_sha256=sha(__file__),
        spatial_convergence=False, physical_validation=False,
        acceptance_tolerance=None,
        limitations=["Only two interior mesh levels; unchanged outer surface and contact discretization",
            "Equal quadrature count is not proof of equal or sufficient quadrature accuracy",
            "Whole effective heart with fixed background; not ventricular stroke volume",
            "No registration, translations, other montage, conductivity state or subject in this block"])
    if (previous_individual is None) != (previous_analytic is None):
        raise ValueError("Provide both historical quadrature references")
    result["actual_contact_areas_mm2"] = {level: plans["individual", level]["contact"]["areas_mm2"] for level in ("coarse", "fine")}
    result["mesh_motion_scales"] = mesh_motion_scales(plans)
    result["electrical_quadrature_history"] = None if previous_individual is None else quadrature_history(
        reports, plans, previous_individual, previous_analytic, result["candidate_parameters_sha256"])
    (output / "summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    with (output / "shape_errors.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(errors[0]));writer.writeheader();writer.writerows(errors)
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--block",type=Path,required=True)
    p.add_argument("--reference-individual",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--previous-individual",type=Path)
    p.add_argument("--previous-analytic",type=Path)
    a=p.parse_args();r=run(a.block,a.reference_individual,a.output,a.previous_individual,a.previous_analytic)
    print(json.dumps({"status":r["status"],"preference_stability":r["preference_stability"]}))
