"""Compare original-heart and analytic-shape volume probes on identical inputs."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import heart_replacement_electrical_pilot as core
import heart_fractional_electrical_pilot as frac
import heart_individual_electrical_stage as individual

read, write, sha, require = core.read, core.write, core.sha, core.require


def run(a):
    source=a.individual.resolve(); previous=a.previous.resolve(); out=a.output.resolve()
    p=frac.load_plan(source); q=frac.load_plan(previous)
    require(p["parameter"]=="volume","Expected volume stage")
    for key in ("source_prepared_sha256","rho_ohm_m","frequency_hz","ground_zero_based"):
        require(p[key]==q[key],"Incompatible comparison: "+key)
    require(p["contact"]["sha256"]==q["contact"]["sha256"],"Contact mismatch")
    own=individual.summarize(source,p)
    require(own["status"]=="complete","Individual run incomplete")
    oldrows=[frac.checked_result(previous,q,s) for s in q["states"]]
    require(all(r is not None for r in oldrows),"Analytic-shape run incomplete")
    old=read(previous/"summary.json")
    require(old["completed"]==len(oldrows),"Stale previous summary")
    for r in old["states"]:
        matched=next(v for v in oldrows if v["id"]==r["id"])
        require(matched==r,"Previous reader summary differs from checked results")
    reference_dir = a.baseline.resolve()
    reference_plan = core.load_plan(reference_dir)
    require(reference_plan["rho_ohm_m"] == p["rho_ohm_m"] and
            reference_plan["frequency_hz"] == p["frequency_hz"], "Static reference physical state differs")
    require(reference_plan["source_prepared_sha256"] == p["source_prepared_sha256"],
            "Static identity reference uses a different mesh")
    reference_contact=next(c for c in reference_plan["contacts"] if c["id"]==p["contact"]["id"])
    require(reference_contact==p["contact"], "Static reference contact metadata differs")
    for key in ("criteria", "ground_zero_based", "node_count", "element_count", "conductivity_by_tissue_id"):
        require(reference_plan[key]==p[key], "Static reference differs: "+key)
    for state in p["states"]:
        require(np.isclose(state["analytic_volume_ml"],p["nominal_volume_ml"]*(1+state["volume_fraction"]),rtol=1e-13),
                "Analytic volume does not follow the declared scaling law")
    nominal = core.completed(reference_dir, reference_plan, "nominal", "tepc_2")
    require(nominal is not None, "Original static reference incomplete")
    baseline = next(r for r in own["rows"] if r["parameter"] == "baseline")
    identity_error = abs(baseline["Z_ohm"] - nominal["Z_ohm"])
    tolerance = p["criteria"]["reciprocity_atol_ohm"] + p["criteria"]["reciprocity_rtol"] * abs(nominal["Z_ohm"])
    require(identity_error <= tolerance, "Fractional identity does not reproduce original impedance")
    analytic_level=max(r["points_per_tet"] for r in oldrows)
    allrows=[dict(r,geometry="individual") for r in own["rows"]]+[
        r for r in oldrows if r["points_per_tet"]==analytic_level]
    lookup={(r["geometry"],r["volume_fraction"]):r for r in allrows}
    steps=sorted({r["signed_step"] for r in own["rows"] if r["signed_step"]>0})
    result=[]
    for step in steps:
        reference=next(d for d in own["derivatives"] if d["step"]==step)["derivative"]
        for geometry in ("individual","sphere","ellipsoid"):
            baseline=lookup[geometry,0.]; plus=lookup[geometry,step]; minus=lookup[geometry,-step]
            volume=baseline["analytic_volume_ml"]
            require(abs(volume-lookup["individual",0]["analytic_volume_ml"])<1e-6,
                    "Figures do not share analytic volume")
            derivative=(plus["Z_ohm"]-minus["Z_ohm"])/(2*step*volume)
            error=derivative-reference
            minus_slope=(baseline["Z_ohm"]-minus["Z_ohm"])/(step*volume)
            plus_slope=(plus["Z_ohm"]-baseline["Z_ohm"])/(step*volume)
            result.append(dict(geometry=geometry,points_per_tet=baseline["points_per_tet"],step_fraction=step,volume_ml=volume,
                Z_base_ohm=baseline["Z_ohm"],dZ_dV_ohm_per_ml=derivative,
                derivative_error_ohm_per_ml=error,
                decrease_slope_ohm_per_ml=minus_slope,increase_slope_ohm_per_ml=plus_slope,
                one_sided_gap_relative_to_central=None if abs(derivative)<1e-14 else abs((plus_slope-minus_slope)/derivative),
                derivative_relative_error=None if abs(reference)<1e-14 else abs(error/reference),
                elasticity=volume*derivative/baseline["Z_ohm"],
                material_volume_slope=(plus["material_volume_ml"]-minus["material_volume_ml"])/(2*step*volume)))
    require(sha(source/"geometry_qc.json") == p["geometry_qc_sha256"], "Geometry QC changed")
    geometry=read(source/"geometry_qc.json")
    material=[]
    for n in geometry["quadrature_levels"]:
        selected={r["signed_step"]:r for r in geometry["rows"] if r["points_per_tet"]==n}
        for step in steps:
            slope=(selected[step]["material_volume_ml"]-selected[-step]["material_volume_ml"])/(2*step*geometry["original_volume_ml"])
            material.append(dict(points_per_tet=n,step_fraction=step,
                                 dV_material_dV_analytic=slope,error_from_one=slope-1))
    out.mkdir(parents=True,exist_ok=True)
    with (out/"volume_derivatives.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(result[0]));writer.writeheader();writer.writerows(result)
    repo = Path(__file__).resolve().parents[2]
    inputs={str(path.relative_to(repo)).replace("\\", "/"):sha(path) for path in (
        source/"plan.json",source/"summary.json",source/"geometry_qc.json",
        previous/"plan.json",previous/"summary.json",
        reference_dir/"nominal/tepc_2/result.json")}
    write(out/"comparison.json",dict(
        status="calculated_on_original_mesh",derivatives=result,
        individual_material_derivative_checks=material,states=allrows,
        checked_inputs=inputs,implementation_sha256=sha(__file__),
        montage="TEPC-2",frequency_hz=p["frequency_hz"],rho_ohm_m=p["rho_ohm_m"],
        individual_quadrature=p["states"][0]["points_per_tet"],analytic_shape_quadrature=analytic_level,
        original_identity_impedance_error_ohm=identity_error,
        original_identity_numerical_tolerance_ohm=tolerance,
        electrode_diameter_mm=5,physical_validation=False,spatial_convergence=False,
        notes=[
            "Individual tetrahedral union is a computational geometric comparator, not independent anatomical truth.",
            "Identical CEM mesh/contact/material state; original nonlinear shape differences remain.",
            "Individual and analytic-shape quadrature resolutions are recorded separately; equal counts are not assumed.",
            "Relative derivative error concerns this uniform volume-scaling law only.",
            "Central differences are finite-step estimates; one-sided slope agreement is checked separately.",
            "No acceptance threshold fixed: observed differences do not certify convergence."
        ]))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--individual",type=Path,required=True)
    parser.add_argument("--previous",type=Path,required=True)
    parser.add_argument("--baseline",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args())
