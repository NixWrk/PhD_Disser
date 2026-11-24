"""Sequential full-CEM lung fit after common_absolute right-60 calibration.

Run from any directory: python run_lung.py prepare; python run_lung.py run
Export/resume inspection: python run_lung.py export
Only this directory and its ignored output are written. No README edits.

Four separate states, U1/U2 lying/seated. The exact fem60 common_absolute
rho is fixed in both the soft_background and fat masks. Bone/heart retain
gabriel100_proxy; one electrical lung is the only free parameter. The real
BEI1 medians and frozen 8-s windows are inherited from hash-verified fem60
sources. Objective = (Z100-y100)^2+(Z140-y140)^2, in ohm^2, equal weights.
The 200-mm observation never enters the objective or parameter selection.
Z60 is diagnostic.

Profile starts at 0.1,2,9.315252924705511,12,100 ohm m. The 2/12 nodes reflect
the author's earlier COMSOL working range, not biological bounds. A boundary
minimum triggers decade diagnostics to 0.001 or 10000. A verified finite
stationary minimum outside 0.1..100 can be retained with an explicit status.
If the profile still decreases outward, the primary boundary is reported as
a bounded calculation point, NOT an identified finite positive tissue rho.
Both the bounded solution and external diagnostic information remain in CSV.
Sampled gradient brackets do not establish global uniqueness/monotonicity.

All knots and accepted fields are full canonical trkg4_lateral_cem_response
solutions or exact prior fem60 solutions with the same rho/mesh/contacts.
One assembled tissue K is shared by the 100/140 pair; five material operators
are loaded once per mesh. Analytical adjoint J is used without extra FD runs.
Small mesh: 60/100/140; extended mesh: 200, with no global convergence claim.
J fields retain signs, units ohm, derivatives with respect to ln(rho).
The soft+fat derivative is j_soft_ohm+j_fat_ohm for their common log change.

Material source: unchanged tissue_literature_v2.json, ITIS V5.0,
DOI 10.13099/VIP21000-05-0; Gabriel dispersion at 100000 Hz, resistive proxy.
Fixed CT, contact model, constructed arms in soft mask and the imperfect
right-60 baseline remain assumptions/limitations inherited from fem60.
Baseline rho is transferred between separately recorded sizes within posture;
no identical pulse across recordings is assumed. No pulse inverse is done.

CSV fields key: participant,posture,size_mm; measured_Z_ohm,predicted_Z_ohm,
rho_lung_ohm_m,rho_soft_ohm_m,rho_fat_ohm_m,j_lung_ohm,j_soft_fat_ohm,
the other signed tissue J, status and provenance. Partial tables are emitted
while MATLAB runs; result_complete=False until the 200-mm field is present.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sp = importlib.util.spec_from_file_location("fem60_source", HERE.parent/"fem60.py")
fm = importlib.util.module_from_spec(sp)
sp.loader.exec_module(fm)
tf, ROOT, PIPE = fm.tf, fm.ROOT, fm.PIPE
WORK = PIPE/"output/exploratory/back_measurement_20261123/inspiration_rho/sequential_lung"
SCENARIO = "sequential_lung_fixed_common60"


def prepare():
    if (WORK/"manifest.local.json").exists():
        raise FileExistsError("Existing manifest: use run/export without overwriting")
    assert subprocess.run(["git","check-ignore",str(WORK/"probe.mat")],cwd=ROOT,capture_output=True).returncode == 0
    tf.verify_frequency()
    old = tf.read(fm.WORK/"manifest.local.json")
    receipt = tf.read(fm.WORK/"export_receipt.local.json")
    assert tf.sha(HERE.parent/"fem60_fits.csv") == receipt["fit_table_sha256"]
    assert tf.sha(HERE.parent/"fem60_fields.csv") == receipt["field_table_sha256"]
    table = fm.rows(HERE.parent/"fem60_fits.csv")
    jobs = []
    for sourcejob in old["jobs"]:
        u = sourcejob["participant"]
        previous = tf.read(fm.WORK/u/"common_absolute.local.json")
        assert previous["manifest_sha256"] == tf.sha(fm.WORK/"manifest.local.json")
        assert previous["extended_complete"]
        states, imports = [], []
        for fit in previous["fits"]:
            assert fit["status"] == "converged_full_FEM"
            rho = fit["rho_ohm_m"]
            assert rho[0] == rho[4] and rho[1:4] == old["rho_proxy"][1:4]
            pub = next(r for r in table if (r["participant"],r["posture"],r["scenario"]) == (u,fit["posture"],"common_absolute"))
            assert float(pub["rho_soft_ohm_m"]) == rho[0]
            obs = [o for o in sourcejob["observations"] if o["posture"] == fit["posture"]]
            for pred in fit["predictions"]:
                assert tf.sha(pred["source"]) == pred["source_sha256"]
                imports.append(pred)
            states.append(dict(posture=fit["posture"], rho_fixed=rho, observations=obs,
                target_Z_ohm=[next(o["measured_Z_ohm"] for o in obs if o["size_mm"] == k) for k in (100,140)],
                baseline=fit["predictions"]))
        jobs.append(dict(participant=u,sources=sourcejob["sources"],states=states,imports=imports))
    manifest = dict(root=str(ROOT),work=str(WORK),jobs=jobs,scenario=SCENARIO,
        rho_proxy=old["rho_proxy"],registry_sha256=old["registry_sha256"],kernel_sha256=old["kernel_sha256"],
        matlab_sha256=tf.sha(HERE/"solve_lung.m"),python_sha256=tf.sha(__file__),
        fem60_manifest_sha256=tf.sha(fm.WORK/"manifest.local.json"),fem60_fits_sha256=receipt["fit_table_sha256"],
        fem60_fields_sha256=receipt["field_table_sha256"],windows_sha256=old["windows_sha256"],
        primary_interval_ohm_m=[.1,100.],profile_grid_ohm_m=[.1,2.,old["rho_proxy"][1],12.,100.],
        lower_diagnostics_ohm_m=[.01,.001],upper_diagnostics_ohm_m=[1000.,10000.],
        gradient_tolerance_ohm2=1e-4,log_parameter_tolerance=1e-5,maximum_refinements=12,
        objective="equal_weight_sum_squared_100_140_ohm2",global_minimum_proven=False,
        global_mesh_convergence_proven=False,physiological_validation=False)
    tf.write(WORK/"manifest.local.json",manifest)
    print("Prepared four states; immutable common rho:",[(j["participant"],s["posture"],s["rho_fixed"][0]) for j in jobs for s in j["states"]],flush=True)


def csv_atomic(name, rows):
    if not rows:
        return
    temp = WORK/(name+".writing")
    tf.csv_write(temp,rows)
    os.replace(temp,HERE/name)


def export(require_complete=False):
    m = tf.read(WORK/"manifest.local.json")
    fits, fields, profiles, diagnostic_fields = [], [], [], []
    for job in m["jobs"]:
        for state in job["states"]:
            u, pose = job["participant"], state["posture"]
            folder = WORK/u
            pf = folder/(pose+"_profile.local.json")
            if pf.exists():
                pdata=tf.read(pf)
                assert pdata["manifest_sha256"] == tf.sha(WORK/"manifest.local.json")
                points=pdata["points"]
                if isinstance(points,dict):points=[points]
                for q in points:
                    profiles.append(dict(participant=u,posture=pose,scenario=SCENARIO,
                        rho_common_ohm_m=state["rho_fixed"][0],rho_lung_ohm_m=q["rho_lung_ohm_m"],
                        objective_ohm2=q["objective_ohm2"],gradient_logrho_ohm2=q["gradient_logrho_ohm2"],
                        Z100_ohm=q["Z_ohm"][0],Z140_ohm=q["Z_ohm"][1],
                        residual100_ohm=q["residual_ohm"][0],residual140_ohm=q["residual_ohm"][1],
                        j_lung100_ohm=q["j_lung_ohm"][0],j_lung140_ohm=q["j_lung_ohm"][1],
                        j_soft_fat100_ohm=q["j_soft_fat_ohm"][0],j_soft_fat140_ohm=q["j_soft_fat_ohm"][1],
                        purpose=q["purpose"],outside_primary_interval=q["rho_lung_ohm_m"]<.1-1e-12 or q["rho_lung_ohm_m"]>100+1e-10,
                        full_FEM=True,frequency_hz=100000,material_base="gabriel100_proxy",
                        solve100_sha256=q["fields"][0]["source_sha256"],solve140_sha256=q["fields"][1]["source_sha256"],
                        global_monotonicity_proven=False))
            rf = folder/(pose+"_result.local.json")
            if not rf.exists():
                assert not require_complete
                continue
            r=tf.read(rf)
            assert r["manifest_sha256"] == tf.sha(WORK/"manifest.local.json")
            if require_complete:assert r["complete"]
            chosen=r["selected"]; baseline=r["baseline"]
            meta=dict(participant=u,posture=pose,scenario=SCENARIO,fit_status=r["status"],
                result_complete=r["complete"],finite_optimum_identified=r["finite_optimum_identified"],
                rho_lung_ohm_m=chosen["rho_lung_ohm_m"],rho_soft_ohm_m=state["rho_fixed"][0],
                rho_lung_estimate_ohm_m=chosen["rho_lung_ohm_m"] if r["finite_optimum_identified"] else None,
                rho_common_ohm_m=state["rho_fixed"][0],
                rho_fat_ohm_m=state["rho_fixed"][4],rho_bone_ohm_m=state["rho_fixed"][2],rho_heart_ohm_m=state["rho_fixed"][3],
                rho_lung_is_boundary_calculation=not r["finite_optimum_identified"],
                frequency_hz=100000,material_base="gabriel100_proxy",pair="back_arm",placement="back_right",
                fem60_fits_sha256=m["fem60_fits_sha256"],fem60_fields_sha256=m["fem60_fields_sha256"],
                kernel_sha256=m["kernel_sha256"],registry_sha256=m["registry_sha256"],
                manifest_sha256=tf.sha(WORK/"manifest.local.json"),matlab_sha256=m["matlab_sha256"],
                fit_sizes_mm="100;140",holdout200_used_for_selection=False,
                global_minimum_proven=False,global_mesh_convergence_proven=False,
                physiological_validation=False,baseline60_artifact_present=True,
                baseline_transfer_between_recordings="same_rho_assumption",fixed_ct_geometry=True)
            fits.append(dict(meta,objective_ohm2=chosen["objective_ohm2"],baseline_objective_ohm2=baseline["objective_ohm2"],
                residual100_ohm=chosen["residual_ohm"][0],residual140_ohm=chosen["residual_ohm"][1],
                gradient_logrho_ohm2=chosen["gradient_logrho_ohm2"],
                lung_sensitivity_l2_ohm=sum(v*v for v in chosen["j_lung_ohm"])**.5,
                primary_rho_lung_ohm_m=r["primary_selected"]["rho_lung_ohm_m"],primary_objective_ohm2=r["primary_selected"]["objective_ohm2"],
                checked_rho_min_ohm_m=r["checked_min"],checked_rho_max_ohm_m=r["checked_max"],
                best_checked_rho_lung_ohm_m=r["best_checked"]["rho_lung_ohm_m"],best_checked_objective_ohm2=r["best_checked"]["objective_ohm2"],
                sampled_stationary_brackets=r["sampled_stationary_brackets"],profile_nodes=r["profile_nodes"],
                gradient_tolerance_ohm2=m["gradient_tolerance_ohm2"],
                low_or_high_boundary=r["boundary"],outside_primary_stationary_minimum=r["outside_primary_stationary_minimum"]))
            pending=[(p,False) for p in r["predictions"]]+[(p,True) for p in r.get("diagnostic_predictions",[])]
            for pred,is_diagnostic in pending:
                k=pred["size_mm"];obs=next(o for o in state["observations"] if o["size_mm"]==k)
                old=next(o for o in state["baseline"] if o["size_mm"]==k)
                fam="extended" if k==200 else "small";src=job["sources"][fam];j=pred["J_logrho_ohm"]
                assert pred["rho_ohm_m"][0] == pred["rho_ohm_m"][4] == state["rho_fixed"][0]
                assert tf.sha(pred["source"]) == pred["source_sha256"]
                fieldmeta=dict(meta,rho_lung_ohm_m=pred["rho_ohm_m"][1],
                    solution_role="external_boundary_diagnostic" if is_diagnostic else "primary_result")
                destination=diagnostic_fields if is_diagnostic else fields
                destination.append(dict(fieldmeta,size_mm=k,role="holdout" if k==200 else "diagnostic_recheck" if k==60 else "fit",
                    measured_Z_ohm=obs["measured_Z_ohm"],predicted_Z_ohm=pred["Z_ohm"],
                    prediction_error_ohm=pred["Z_ohm"]-obs["measured_Z_ohm"],baseline_Z_ohm=old["Z_ohm"],
                    j_soft_ohm=j[0],j_lung_ohm=j[1],j_bone_ohm=j[2],j_heart_ohm=j[3],j_fat_ohm=j[4],j_soft_fat_ohm=j[0]+j[4],
                    baseline_j_lung_ohm=old["J_logrho_ohm"][1],mesh_family=fam,mesh_sha256=src["mesh_sha256"],
                    shared_sha256=src["shared_sha256"],contacts_sha256=src["contacts_sha256"],z_contact_ohm_m2=src["z_contact_ohm_m2"],
                    nominal_contact_diameter_mm=5,material_fraction_sampling="q4096",cross_mesh_from_fit=k==200,
                    window_start_s=obs["start_s"],window_stop_s=obs["stop_s"],window_evidence=obs["window_evidence"],
                    source_array_sha256=obs["array_sha256"],solve_sha256=pred["source_sha256"],
                    system_relative_residual=pred["qc"]["system_relative_residual"],reciprocity_ohm=pred["qc"]["reciprocity_ohm"],
                    prediction_is_full_FEM=True))
    if require_complete:
        assert len(fits)==4 and len(fields)==16
        assert len({(r["participant"],r["posture"],r["size_mm"]) for r in fields})==16
        assert tf.sha(HERE/"solve_lung.m") == m["matlab_sha256"]
        assert tf.sha(PIPE/"src/trkg4_lateral_cem_response.m") == m["kernel_sha256"]
        assert tf.sha(HERE.parent/"fem60_fits.csv") == m["fem60_fits_sha256"]
        assert tf.sha(HERE.parent/"fem60_fields.csv") == m["fem60_fields_sha256"]
    fits.sort(key=lambda r:(r["participant"],r["posture"]))
    fields.sort(key=lambda r:(r["participant"],r["posture"],r["size_mm"]))
    diagnostic_fields.sort(key=lambda r:(r["participant"],r["posture"],r["size_mm"]))
    profiles.sort(key=lambda r:(r["participant"],r["posture"],r["rho_lung_ohm_m"]))
    csv_atomic("lung_profile.csv",profiles);csv_atomic("lung_fits.csv",fits);csv_atomic("lung_fields.csv",fields)
    csv_atomic("lung_diagnostic_fields.csv",diagnostic_fields)
    if require_complete:
        tf.write(WORK/"export_receipt.local.json",dict(fits=len(fits),fields=len(fields),profile_nodes=len(profiles),diagnostic_fields=len(diagnostic_fields),
            preparation_python_sha256=m["python_sha256"],exporter_python_sha256=tf.sha(__file__),
            solver_matlab_sha256=m["matlab_sha256"],manifest_sha256=tf.sha(WORK/"manifest.local.json"),
            exported_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
            **{name+"_sha256":tf.sha(HERE/(name+".csv")) for name in ("lung_fits","lung_fields","lung_profile","lung_diagnostic_fields") if (HERE/(name+".csv")).exists()}))
    return len(fits),len(fields),len(profiles),len(diagnostic_fields)


def run():
    m=tf.read(WORK/"manifest.local.json");assert tf.sha(HERE/"solve_lung.m")==m["matlab_sha256"]
    for name in ("preferences","temp"):(WORK/name).mkdir(exist_ok=True)
    env=os.environ.copy();env.update(MATLAB_PREFDIR=str(WORK/"preferences"),TEMP=str(WORK/"temp"),TMP=str(WORK/"temp"))
    quote=lambda v:"'"+str(v).replace("'","''")+"'"
    command=f"addpath({quote(HERE)});solve_lung({quote(WORK/'manifest.local.json')});"
    attempt=len(list(WORK.glob("matlab_*.log")))+1;start=time.monotonic()
    with (WORK/f"launcher_{attempt}.log").open("w",encoding="utf-8") as log:
        proc=subprocess.Popen([shutil.which("matlab"),"-wait","-logfile",str(WORK/f"matlab_{attempt}.log"),"-batch",command],
            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        tf.write(WORK/"process.local.json",dict(pid=proc.pid,attempt=attempt,command=command))
        print("Single MATLAB started PID",proc.pid,flush=True)
        previous=None
        while proc.poll() is None:
            time.sleep(10)
            counts=export()
            if counts!=previous:print("Partial fits/fields/profile:",counts,flush=True);previous=counts
        code=proc.returncode
    tf.write(WORK/f"exit_{attempt}.local.json",dict(exit_code=code,seconds=time.monotonic()-start))
    if code:raise RuntimeError(f"MATLAB exit {code}, see private matlab_{attempt}.log")
    print("Final export",export(require_complete=True),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("action",choices=("prepare","run","export"));args=ap.parse_args()
    if args.action=="prepare":prepare()
    elif args.action=="run":run()
    else:print(export(require_complete=all((WORK/u/"complete.local.json").exists() for u in ("U1","U2"))))
