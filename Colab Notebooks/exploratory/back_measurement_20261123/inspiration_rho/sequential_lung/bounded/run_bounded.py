"""Author-bounded full-CEM lung experiment, independent of historical runs.

python run_bounded.py prepare
python run_bounded.py run
python run_bounded.py export

Fixed w is the exact fem60 common_absolute right-60 solution at the original
Gabriel 100-kHz lung rho r0. For each state compare [w,12], [w,17] (primary),
[w,30] (limiting check). These are AUTHOR CONDITIONS, not confidence intervals
or independently measured tissue bounds. Only 100/140 enter equal-weight
sum squared errors in ohm^2. Full 60 is a back-check; 200 is held out.

Every field request, including a cache hit, is guarded by w <= r <= 30 and
exact equality of all four fixed material properties. Eligible historical
solutions at r0/12 are hash verified and reused. Five required nodes plus
one midpoint of the largest logarithmic gap check the signed profile.
Negative-to-positive gradient brackets and near-zero nodes trigger scalar
refinement with full FEM pairs. The lowest SSE among endpoints, all available
nodes and stationary candidates is retained. Interior conditional estimates
and author-imposed boundaries are distinct statuses; neither is independent
physiological identification. Sampled evidence does not prove global optimality.

The MATLAB file is a small batch dispatcher to the unchanged canonical
trkg4_lateral_cem_response, using original q4096 material operators and CEM
contacts. Historical drivers are only read/hashed, not copied or modified.
Shared K is used for 100/140. The same rho/size/mesh is solved only once.

Fits key: participant,posture,lung_upper_ohm_m (12/17/30).
Fields add size_mm (60/100/140/200), measured_Z_ohm,predicted_Z_ohm,
signed five tissue J=dZ/dlnrho, and j_soft_fat_ohm=sum of the two J.
Profile adds rho_lung_ohm_m and preserves the prior Z100_ohm,Z140_ohm,
residual100_ohm,residual140_ohm,gradient_logrho_ohm2 names. Shared profile
nodes appear under each interval containing them, with identical solve hashes.

Sources/settings: frozen sequential_lung/fem60 manifests, original 19 Nov
BEI1 8-s medians, tissue_literature_v2.json (ITIS V5.0/Gabriel 100000 Hz,
resistive proxy). Fixed CT and masks, original bone/heart properties, posterior
5-mm contacts, small mesh 60/100/140 and extended mesh 200 are unchanged.
The first-stage 60 artifact, stationarity of baseline properties between
separate recordings, and lack of global mesh/physiological validation remain.
No pulse inverse, new waveform, geometry change, or extra finite differences.
Only this directory and its own ignored output are written; no README edits.
"""
import argparse
import importlib.util
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("historical_lung",HERE.parent/"run_lung.py")
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
tf,ROOT,PIPE=old.tf,old.ROOT,old.PIPE
WORK=old.WORK/"bounded"
UPPERS=(12.,17.,30.)


def prepare():
    assert not (WORK/"manifest.local.json").exists(),"Existing run: use run/export"
    assert subprocess.run(["git","check-ignore",str(WORK/"probe.mat")],cwd=ROOT,capture_output=True).returncode==0
    tf.verify_frequency()
    source=tf.read(old.WORK/"manifest.local.json")
    assert tf.sha(HERE.parent/"solve_lung.m")==source["matlab_sha256"]
    assert tf.sha(HERE.parent.parent/"fem60_fields.csv")==source["fem60_fields_sha256"]
    jobs=[]
    for job in source["jobs"]:
        previous=tf.read(old.WORK/job["participant"]/"cache.local.json")
        assert previous["manifest_sha256"]==tf.sha(old.WORK/"manifest.local.json")
        states=[];imports=[]
        for st in job["states"]:
            rho=st["rho_fixed"];w=rho[0];assert w==rho[4] and 0<w<rho[1]<12
            grid=sorted([w,rho[1],12.,17.,30.])
            gap=max(range(len(grid)-1),key=lambda i:math.log(grid[i+1]/grid[i]))
            grid.append(math.sqrt(grid[gap]*grid[gap+1]));grid.sort()
            states.append(dict(st,grid_ohm_m=grid))
            for r in previous["rows"]:
                rr=r["rho_ohm_m"]
                if r["size_mm"] not in (60,100,140,200) or not w<=rr[1]<=30:continue
                if [rr[k] for k in (0,2,3,4)] != [rho[k] for k in (0,2,3,4)]:continue
                assert tf.sha(r["source"])==r["source_sha256"]
                imports.append(r)
        jobs.append(dict(participant=job["participant"],sources=job["sources"],states=states,imports=imports))
    manifest=dict(root=str(ROOT),work=str(WORK),jobs=jobs,upper_bounds_ohm_m=list(UPPERS),
        kernel_sha256=source["kernel_sha256"],registry_sha256=source["registry_sha256"],
        historical_manifest_sha256=tf.sha(old.WORK/"manifest.local.json"),
        historical_matlab_sha256=source["matlab_sha256"],historical_python_sha256=tf.sha(HERE.parent/"run_lung.py"),
        fem60_fields_sha256=source["fem60_fields_sha256"],fem60_fits_sha256=source["fem60_fits_sha256"],
        windows_sha256=source["windows_sha256"],rho_proxy=source["rho_proxy"],
        matlab_sha256=tf.sha(HERE/"solve_bounded.m"),python_sha256=tf.sha(__file__),
        gradient_sign_tolerance_ohm2=1e-6,gradient_zero_tolerance_ohm2=1e-4,maximum_stationary_iterations=24,
        primary_upper_ohm_m=17.,hard_upper_ohm_m=30.,
        objective="equal_weight_SSE100_140_ohm2",bounds_origin="author_given_conditions",
        global_minimum_proven=False,global_mesh_convergence_proven=False)
    tf.write(WORK/"manifest.local.json",manifest)
    print("Prepared",[(j["participant"],len(j["imports"])) for j in jobs],"eligible cached fields",flush=True)


def save_csv(name,rows):
    if rows:
        temp=WORK/(name+".writing");tf.csv_write(temp,rows);os.replace(temp,HERE/name)


def export(complete=False):
    m=tf.read(WORK/"manifest.local.json");fits=[];fields=[];profiles=[]
    for job in m["jobs"]:
        for st in job["states"]:
            part=job["participant"];pose=st["posture"];w=st["rho_fixed"][0]
            path=WORK/part/(pose+".local.json")
            if not path.exists():assert not complete;continue
            data=tf.read(path);assert data["manifest_sha256"]==tf.sha(WORK/"manifest.local.json")
            nodes=data["profile"];nodes=[nodes] if isinstance(nodes,dict) else nodes
            for upper in UPPERS:
                subset=[n for n in nodes if w<=n["rho_lung_ohm_m"]<=upper]
                for n in subset:
                    profiles.append(dict(participant=part,posture=pose,lung_upper_ohm_m=upper,lung_lower_ohm_m=w,
                        rho_common_ohm_m=w,**{k:n[k] for k in ("rho_lung_ohm_m","Z100_ohm","Z140_ohm","residual100_ohm","residual140_ohm","objective_ohm2","gradient_logrho_ohm2")},
                        gradient_sign=1 if n["gradient_logrho_ohm2"]>0 else -1 if n["gradient_logrho_ohm2"]<0 else 0,
                        j_lung100_ohm=n["fields"][0]["J_logrho_ohm"][1],j_lung140_ohm=n["fields"][1]["J_logrho_ohm"][1],
                        solve100_sha256=n["fields"][0]["source_sha256"],solve140_sha256=n["fields"][1]["source_sha256"],
                        all_requests_within_w_30=True,global_monotonicity_proven=False))
            results=data["results"];results=[results] if isinstance(results,dict) else results
            for r in results:
                if not r:continue
                upper=r["lung_upper_ohm_m"];chosen=r["chosen"];q=chosen["rho_lung_ohm_m"]
                assert w<=q<=upper<=30
                baseline=next(n for n in nodes if n["rho_lung_ohm_m"]==st["rho_fixed"][1])
                common=dict(participant=part,posture=pose,lung_upper_ohm_m=upper,lung_lower_ohm_m=w,
                    interval_role="primary" if upper==17 else "tighter" if upper==12 else "limiting_check",
                    rho_lung_ohm_m=q,rho_common_ohm_m=w,rho_soft_ohm_m=w,rho_fat_ohm_m=w,
                    rho_bone_ohm_m=st["rho_fixed"][2],rho_heart_ohm_m=st["rho_fixed"][3],
                    baseline_lung_rho_ohm_m=st["rho_fixed"][1],fit_status=r["status"],boundary=r["boundary"],
                    boundary_is_given_condition=r["boundary"]!="none",interior_conditional_estimate=r["interior_conditional_estimate"],
                    conditional_rho_estimate_ohm_m=q if r["interior_conditional_estimate"] else None,
                    independent_physiological_identification=False,result_complete=data["complete"],
                    all_requests_within_w_30=True,frequency_hz=100000,material_base="gabriel100_proxy",pair="back_arm",placement="back_right",
                    fit_sizes_mm="100;140",holdout200_used_for_selection=False,backcheck60_used_for_selection=False,
                    fixed_parameters_identical=True,measurements_identical_to_fem60=True,original_contact_operators=True,
                    bounds_origin="author_given_conditions",global_minimum_proven=False,global_mesh_convergence_proven=False,
                    physiological_validation=False,baseline60_artifact_present=True,baseline_transfer_between_recordings="same_rho_assumption",
                    manifest_sha256=tf.sha(WORK/"manifest.local.json"),matlab_sha256=m["matlab_sha256"],kernel_sha256=m["kernel_sha256"],
                    registry_sha256=m["registry_sha256"],fem60_fields_sha256=m["fem60_fields_sha256"],windows_sha256=m["windows_sha256"])
                fits.append(dict(common,**{k:chosen[k] for k in ("Z100_ohm","Z140_ohm","residual100_ohm","residual140_ohm","objective_ohm2","gradient_logrho_ohm2")},
                    baseline_objective_ohm2=baseline["objective_ohm2"],min_sampled_gradient_ohm2=r["min_gradient"],max_sampled_gradient_ohm2=r["max_gradient"],
                    checked_profile_nodes=r["checked_nodes"],sampled_gradient_sign_consistent=r["gradient_sign_consistent"]))
                for pred in r["predictions"]:
                    k=pred["size_mm"];fam="extended" if k==200 else "small";s=job["sources"][fam]
                    obs=next(o for o in st["observations"] if o["size_mm"]==k);base=next(o for o in st["baseline"] if o["size_mm"]==k)
                    assert pred["rho_ohm_m"]==[w,q,st["rho_fixed"][2],st["rho_fixed"][3],w]
                    assert tf.sha(pred["source"])==pred["source_sha256"]
                    jj=pred["J_logrho_ohm"]
                    fields.append(dict(common,size_mm=k,role="holdout" if k==200 else "backcheck" if k==60 else "fit",
                        measured_Z_ohm=obs["measured_Z_ohm"],predicted_Z_ohm=pred["Z_ohm"],prediction_error_ohm=pred["Z_ohm"]-obs["measured_Z_ohm"],
                        baseline_Z_ohm=base["Z_ohm"],delta_Z_from_r0_ohm=pred["Z_ohm"]-base["Z_ohm"],
                        j_soft_ohm=jj[0],j_lung_ohm=jj[1],j_bone_ohm=jj[2],j_heart_ohm=jj[3],j_fat_ohm=jj[4],j_soft_fat_ohm=jj[0]+jj[4],
                        baseline_j_lung_ohm=base["J_logrho_ohm"][1],mesh_family=fam,mesh_sha256=s["mesh_sha256"],contacts_sha256=s["contacts_sha256"],
                        shared_sha256=s["shared_sha256"],z_contact_ohm_m2=s["z_contact_ohm_m2"],nominal_contact_diameter_mm=5,material_fraction_sampling="q4096",
                        cross_mesh_from_fit=k==200,source_array_sha256=obs["array_sha256"],window_start_s=obs["start_s"],window_stop_s=obs["stop_s"],
                        window_evidence=obs["window_evidence"],system_relative_residual=pred["qc"]["system_relative_residual"],
                        reciprocity_ohm=pred["qc"]["reciprocity_ohm"],solve_sha256=pred["source_sha256"],field_is_exact_FEM=True))
            if complete:assert data["complete"] and len(results)==3
    if complete:
        assert len(fits)==12 and len(fields)==48
        assert len({(r["participant"],r["posture"],r["lung_upper_ohm_m"],r["size_mm"]) for r in fields})==48
        assert tf.sha(HERE/"solve_bounded.m")==m["matlab_sha256"]
        assert tf.sha(HERE.parent/"solve_lung.m")==m["historical_matlab_sha256"]
        assert tf.sha(old.WORK/"manifest.local.json")==m["historical_manifest_sha256"]
        assert tf.sha(PIPE/"src/trkg4_lateral_cem_response.m")==m["kernel_sha256"]
        assert tf.sha(HERE.parent.parent/"fem60_fields.csv")==m["fem60_fields_sha256"]
    fits.sort(key=lambda x:(x["participant"],x["posture"],x["lung_upper_ohm_m"]))
    fields.sort(key=lambda x:(x["participant"],x["posture"],x["lung_upper_ohm_m"],x["size_mm"]))
    profiles.sort(key=lambda x:(x["participant"],x["posture"],x["lung_upper_ohm_m"],x["rho_lung_ohm_m"]))
    for name,rows in (("bounded_fits",fits),("bounded_fields",fields),("bounded_profile",profiles)):save_csv(name+".csv",rows)
    if complete:tf.write(WORK/"export_receipt.local.json",dict(fits=len(fits),fields=len(fields),profile_rows=len(profiles),
        preparation_python_sha256=m["python_sha256"],exporter_python_sha256=tf.sha(__file__),solver_matlab_sha256=m["matlab_sha256"],
        manifest_sha256=tf.sha(WORK/"manifest.local.json"),**{n+"_sha256":tf.sha(HERE/(n+".csv")) for n in ("bounded_fits","bounded_fields","bounded_profile")}))
    return len(fits),len(fields),len(profiles)


def run():
    m=tf.read(WORK/"manifest.local.json");assert tf.sha(HERE/"solve_bounded.m")==m["matlab_sha256"]
    for name in ("preferences","temp"):(WORK/name).mkdir(exist_ok=True)
    env=os.environ.copy();env.update(MATLAB_PREFDIR=str(WORK/"preferences"),TEMP=str(WORK/"temp"),TMP=str(WORK/"temp"))
    quote=lambda p:"'"+str(p).replace("'","''")+"'"
    command=f"addpath({quote(HERE)});solve_bounded({quote(WORK/'manifest.local.json')});"
    attempt=len(list(WORK.glob("matlab_*.log")))+1;start=time.monotonic()
    with (WORK/f"launcher_{attempt}.log").open("w",encoding="utf-8") as log:
        proc=subprocess.Popen([shutil.which("matlab"),"-wait","-logfile",str(WORK/f"matlab_{attempt}.log"),"-batch",command],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        tf.write(WORK/"process.local.json",dict(pid=proc.pid,attempt=attempt,command=command));print("Single MATLAB started PID",proc.pid,flush=True)
        previous=None
        while proc.poll() is None:
            time.sleep(10)
            try:
                counts=export()
                if counts!=previous:print("Partial fits/fields/profile",counts,flush=True);previous=counts
            except PermissionError as exc:print("Table temporarily locked; retry on next update:",type(exc).__name__,flush=True)
        code=proc.returncode
    tf.write(WORK/f"exit_{attempt}.local.json",dict(exit_code=code,seconds=time.monotonic()-start))
    if code:raise RuntimeError(f"MATLAB exit {code}; inspect private log")
    print("Final export",export(True),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("action",choices=("prepare","run","export"));arg=ap.parse_args()
    if arg.action=="prepare":prepare()
    elif arg.action=="run":run()
    else:print(export(all((WORK/u/"complete.local.json").exists() for u in ("U1","U2"))))
