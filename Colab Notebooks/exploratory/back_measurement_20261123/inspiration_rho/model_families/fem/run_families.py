"""Three FEM families; only this directory and its ignored output are written.

python run_families.py prepare
python run_families.py run
python run_families.py export

pure_fem_common: exact canonical bounded CSV replay, no new solves.
pure_fem_separate_q: exact fem60 common_multiplier pair, retaining distinct
soft_background/fat operators. The first mask is residual soft tissue excluding
the separately segmented fat, NOT pure muscle. q is the full Gabriel ratio,
not the rounded label 8.3372. One scale was fitted in the historical FEM60.
mixed_flat60_fem: exact analytical Z60/(DiskModel.base/pi), canonical finite
5-mm disks, 4x16 quadrature, 512 terms. Reconstruct full precision and verify
the published 12-significant-digit halfspace60 CSV. Assign this one value to
both FEM material masks without FEM60 recalibration. Flat math is read-only.

For new families only lung rho varies, over [rho_soft,12/17/30]; 17 is primary.
Equal-weight SSE of raw measured Z100/Z140 in ohm^2. Full baseline FEM at r0,
final backcheck60 and holdout200 are computed; neither 60 nor 200 selects lung.
Analytical signed tissue derivatives remain dZ/dlnrho in ohm. j_soft_fat is
their sum, i.e. a common log scale of the two masks, also when their rho differ.

No copied public FEM solver. A private runnable derivative of the hash-pinned
solve_bounded.m is prepared by exact, count-checked substitutions. The patch
receipt records every before/after string and both hashes. Changes generalize
the fixed-pair guard, add baseline fields, and use a cubic proposal inside an
existing gradient bracket. Every proposed point still invokes full CEM or an
identical verified cache entry. The canonical CEM kernel is unchanged.

One MATLAB process; parts/contact arrays reused per participant/mesh. Every
request, including cache hits, is guarded by rho_soft <= rho_lung <= 30.
The residual soft mask includes existing artificial arms. Original CT/heart,
one lung, Gabriel 100000-Hz resistive bone/heart proxies, 5-mm posterior CEM
contacts and q4096 material fractions remain. Small mesh 60/100/140; extended
200 without a global convergence claim. Baseline60 artifact, separate-record
stationarity and unvalidated flat-to-FEM material transfer remain explicit.

API: fem_fits.csv key model_family,participant,posture,lung_upper_ohm_m;
fem_fields.csv adds size_mm; fem_profile.csv adds rho_lung_ohm_m;
fem_inputs.csv has one row per family/participant/posture. Partial CSVs mark
result_complete=False. Bounds are author conditions, not biological estimates.
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
INSP=HERE.parents[1]
BOUNDED=INSP/"sequential_lung/bounded"
spec=importlib.util.spec_from_file_location("bounded_source",BOUNDED/"run_bounded.py")
bf=importlib.util.module_from_spec(spec);spec.loader.exec_module(bf)
tf,ROOT,PIPE=bf.tf,bf.ROOT,bf.PIPE
FM=bf.old.fm
WORK=PIPE/"output/exploratory/back_measurement_20261123/inspiration_rho/model_families/fem"
FAMILIES=("pure_fem_common","pure_fem_separate_q","mixed_flat60_fem")
readcsv=FM.rows


def prepare_driver():
    original=BOUNDED/"solve_bounded.m"
    original_text=original.read_text(encoding="utf-8-sig")
    expected=tf.read(bf.WORK/"manifest.local.json")["matlab_sha256"]
    assert tf.sha(original)==expected
    text=original_text;patches=[]
    def replace(before,after,why):
        nonlocal text
        assert text.count(before)==1,(why,text.count(before))
        text=text.replace(before,after)
        patches.append(dict(before=before,after=after,expected_matches=1,reason=why))
    replace("function solve_bounded(manifestfile)","function solve_families(manifestfile)","Private function name")
    replace("which('solve_bounded')","which('solve_families')","Validate generated driver hash")
    replace("assert(st.rho_fixed(1)==st.rho_fixed(5)&&st.rho_fixed(1)>0);",
        "assert(all(isfinite(st.rho_fixed))&&all(st.rho_fixed>0));\nassert(st.rho_fixed(1)==st.expected_soft_ohm_m&&st.rho_fixed(5)==st.expected_fat_ohm_m);",
        "Preserve the exact fixed pair; equality is required by mixed inputs, not imposed on separate tissues")
    replace("  data.results=struct([]);",
        "  [base60,cache]=field(ctx,cache,st,st.rho_fixed(2),60,[]);\n  basepoint=data.profile([data.profile.rho_lung_ohm_m]==st.rho_fixed(2));assert(isscalar(basepoint));\n  data.baseline=[base60;basepoint.fields(:)];save_state(ctx,st,data);\n  data.results=struct([]);",
        "Full FEM baseline60 at r0; cached for exact separate calibration, new for mixed transfer")
    replace("  for vi=1:numel(data.results)",
        "  [base200,cache]=field(ctx,cache,st,st.rho_fixed(2),200,[]);\n  if ~any([data.baseline.size_mm]==200),data.baseline=[data.baseline(:);base200];end\n  for vi=1:numel(data.results)",
        "Full r0 holdout baseline, never used for lung selection")
    replace("row.provenance='new_bounded_canonical_CEM';","row.provenance='model_family_full_CEM';","Separate receipt provenance")
    replace(" x=(xa*b.gradient_logrho_ohm2-xb*a.gradient_logrho_ohm2)/(b.gradient_logrho_ohm2-a.gradient_logrho_ohm2);\n if x<=xa+1e-4*(xb-xa)||x>=xb-1e-4*(xb-xa),x=(xa+xb)/2;end",
        " dx=xb-xa;fun=@(t)(2*t.^3-3*t.^2+1)*a.objective_ohm2+(t.^3-2*t.^2+t)*dx*a.gradient_logrho_ohm2+(-2*t.^3+3*t.^2)*b.objective_ohm2+(t.^3-t.^2)*dx*b.gradient_logrho_ohm2;\n t=fminbnd(fun,0,1,optimset('TolX',1e-10,'Display','off'));\n if t<=1e-6||t>=1-1e-6,t=.5;end\n x=xa+t*dx;",
        "Cheap cubic proposal using full FEM SSE/gradient endpoints; accepted values remain full FEM")
    folder=WORK/"generated";folder.mkdir(parents=True,exist_ok=True)
    generated=folder/"solve_families.m";generated.write_text(text,encoding="utf-8")
    tf.write(WORK/"driver_patch.local.json",dict(source=str(original),source_sha256=expected,generated=str(generated),
        generated_sha256=tf.sha(generated),patches=patches,canonical_kernel_changed=False))
    return original,generated


def prepare():
    assert not (WORK/"manifest.local.json").exists(),"Use run/export on the prepared run"
    assert subprocess.run(["git","check-ignore",str(WORK/"probe.mat")],cwd=ROOT,capture_output=True).returncode==0
    tf.verify_frequency()
    bounded=tf.read(bf.WORK/"manifest.local.json");receipt=tf.read(bf.WORK/"export_receipt.local.json")
    for stem in ("bounded_fits","bounded_fields","bounded_profile"):assert tf.sha(BOUNDED/(stem+".csv"))==receipt[stem+"_sha256"]
    fem_manifest=tf.read(FM.WORK/"manifest.local.json")
    fem_fits=readcsv(INSP/"fem60_fits.csv");fem_fields=readcsv(INSP/"fem60_fields.csv")
    flat_fits=readcsv(INSP/"flat_fits.csv");flat_fields=readcsv(INSP/"flat_fields.csv")
    sys.path.insert(0,str(INSP.parent.parents[1]/"scorpion_v2"))
    # Resolve the canonical DiskModel from the path recorded by flat_rho.py.
    diskfile=ROOT/"Colab Notebooks/scorpion_v2/lateral_pilot.py"
    sys.path.insert(0,str(diskfile.parent))
    from lateral_pilot import DiskModel
    inputs=[];jobs=[];proxy=fem_manifest["rho_proxy"]
    qref=proxy[4]/proxy[0]
    for oldjob in bounded["jobs"]:
        u=oldjob["participant"];states=[];imports=[]
        separate=tf.read(FM.WORK/u/"common_multiplier.local.json")
        assert separate["manifest_sha256"]==tf.sha(FM.WORK/"manifest.local.json")
        for family in FAMILIES:
            for oldstate in oldjob["states"]:
                pose=oldstate["posture"];obs=oldstate["observations"]
                z60=next(o["measured_Z_ohm"] for o in obs if o["size_mm"]==60)
                provenance=dict(model_family=family,participant=u,posture=pose,measured_Z60_ohm=z60,
                    rho_lung_reference_ohm_m=proxy[1],soft_material_definition="residual_soft_tissue_excluding_separate_fat",
                    soft_is_pure_muscle=False,final_forward_model="3D_CEM",FEM60_recalibrated=False)
                if family=="pure_fem_common":
                    rho=oldstate["rho_fixed"];baseline=oldstate["baseline"]
                    provenance.update(calibration_source="fem60_common_absolute",calibration_forward_model="3D_CEM",source_values_full_precision=True)
                elif family=="pure_fem_separate_q":
                    fit=next(r for r in separate["fits"] if r["posture"]==pose)
                    pub=next(r for r in fem_fits if (r["participant"],r["posture"],r["scenario"])==(u,pose,"common_multiplier"))
                    rho=fit["rho_ohm_m"];baseline=fit["predictions"]
                    assert rho[0]==float(pub["rho_soft_ohm_m"]) and rho[4]==float(pub["rho_fat_ohm_m"])
                    assert abs(rho[4]/rho[0]-qref)<1e-13 and rho[0]!=rho[4]
                    for p in baseline:
                        rr=next(r for r in fem_fields if (r["participant"],r["posture"],r["scenario"],int(r["size_mm"]))==(u,pose,"common_multiplier",p["size_mm"]))
                        assert p["Z_ohm"]==float(rr["predicted_Z_ohm"])
                        assert p["J_logrho_ohm"]==[float(rr[k]) for k in ("j_soft_ohm","j_lung_ohm","j_bone_ohm","j_heart_ohm","j_fat_ohm")]
                        assert tf.sha(p["source"])==p["source_sha256"]
                        imports.append(p)
                    provenance.update(calibration_source="fem60_common_multiplier",calibration_forward_model="3D_CEM",
                        fixed_Gabriel_fat_soft_ratio=qref,calibration_multiplier=float(pub["parameter"]),source_values_full_precision=True)
                else:
                    fit=next(r for r in flat_fits if (r["participant"],r["posture"],r["model"])==(u,pose,"halfspace60"))
                    fr=next(r for r in flat_fields if (r["participant"],r["posture"],r["model"],int(r["size_mm"]))==(u,pose,"halfspace60",60))
                    assert tf.sha(diskfile)==fit["disk_code_sha256"] and tf.sha(INSP/"flat_rho.py")==fit["code_sha256"]
                    for ob in obs:
                        rr=next(r for r in flat_fields if (r["participant"],r["posture"],r["model"],int(r["size_mm"]))==(u,pose,"halfspace60",ob["size_mm"]))
                        assert float(rr["measured_Z_ohm"])==ob["measured_Z_ohm"]
                    disk=DiskModel([60],float(fr["h_ct_median_mm"]),diameter_mm=5,radial=4,angular=16,terms=512)
                    K=float(disk.base[0]/math.pi);value=z60/K
                    assert format(value,".12g")==fit["rho_eff_from_Z60_ohm_m"]
                    check=float(disk.evaluate([value,value])[0][0]);assert abs(check-z60)<1e-10
                    rho=proxy.copy();rho[0]=rho[4]=value;baseline=[]
                    provenance.update(calibration_source="flat_halfspace60_Z_over_K",calibration_forward_model="finite_disk_homogeneous_halfspace",
                        K60_per_m=K,flat_rho_published_12sig=float(fit["rho_eff_from_Z60_ohm_m"]),flat_rho_full_precision=value,
                        flat_rho_recalc_minus_published=value-float(fit["rho_eff_from_Z60_ohm_m"]),flat60_reproduction_error_ohm=check-z60,
                        flat_quadrature_radial=4,flat_quadrature_angular=16,flat_image_terms=512,source_values_full_precision=True,
                        flat_to_FEM_material_transfer_validated=False)
                assert rho[1:4]==proxy[1:4] and 0<rho[0]<proxy[1]<12
                provenance.update(rho_soft_ohm_m=rho[0],rho_fat_ohm_m=rho[4],rho_bone_ohm_m=rho[2],rho_heart_ohm_m=rho[3],
                    rho_common_ohm_m=rho[0] if rho[0]==rho[4] else None,fat_to_soft_ratio=rho[4]/rho[0],lung_lower_ohm_m=rho[0])
                if baseline:
                    b60=next(p for p in baseline if p["size_mm"]==60)
                    provenance.update(fem60_at_r0_ohm=b60["Z_ohm"],fem60_at_r0_error_ohm=b60["Z_ohm"]-z60)
                inputs.append(provenance)
                if family=="pure_fem_common":continue
                grid=sorted([rho[0],proxy[1],12.,17.,30.]);i=max(range(4),key=lambda k:math.log(grid[k+1]/grid[k]))
                grid.append(math.sqrt(grid[i]*grid[i+1]));grid.sort()
                states.append(dict(posture=family+"__"+pose,recording_posture=pose,model_family=family,rho_fixed=rho,
                    expected_soft_ohm_m=rho[0],expected_fat_ohm_m=rho[4],observations=obs,baseline=baseline,
                    grid_ohm_m=grid,target_Z_ohm=[next(o["measured_Z_ohm"] for o in obs if o["size_mm"]==k) for k in (100,140)]))
        jobs.append(dict(participant=u,sources=oldjob["sources"],states=states,imports=imports))
    original,generated=prepare_driver()
    paths=[INSP/"fem60_fits.csv",INSP/"fem60_fields.csv",INSP/"flat_fits.csv",INSP/"flat_fields.csv",INSP/"flat_rho.py",diskfile,
           BOUNDED/"run_bounded.py",original]+[BOUNDED/(n+".csv") for n in ("bounded_fits","bounded_fields","bounded_profile")]
    manifest=dict(root=str(ROOT),work=str(WORK),jobs=jobs,inputs=inputs,upper_bounds_ohm_m=[12.,17.,30.],
        kernel_sha256=bounded["kernel_sha256"],registry_sha256=bounded["registry_sha256"],windows_sha256=bounded["windows_sha256"],
        rho_proxy=proxy,historical_sources=[dict(path=str(p),sha256=tf.sha(p)) for p in paths],
        common_csv_sha256={n:tf.sha(BOUNDED/(n+".csv")) for n in ("bounded_fits","bounded_fields","bounded_profile")},
        template_path=str(original),template_sha256=tf.sha(original),generated_driver=str(generated),matlab_sha256=tf.sha(generated),
        wrapper_matlab_sha256=tf.sha(HERE/"run_families.m"),python_sha256=tf.sha(__file__),
        gradient_sign_tolerance_ohm2=1e-6,gradient_zero_tolerance_ohm2=1e-4,maximum_stationary_iterations=24,
        primary_upper_ohm_m=17.,hard_upper_ohm_m=30.,global_mesh_convergence_proven=False)
    tf.write(WORK/"manifest.local.json",manifest)
    print("Prepared 8 new states, 4 canonical replay states; exact inputs:",flush=True)
    for r in inputs:print(r["model_family"],r["participant"],r["posture"],r["rho_soft_ohm_m"],r["rho_fat_ohm_m"],flush=True)
    print("Initial export",export(),flush=True)


def atomic_csv(stem,rows):
    if not rows:return
    path=WORK/(stem+".writing");tf.csv_write(path,rows);os.replace(path,HERE/(stem+".csv"))


def metadata(inp):
    return {k:inp[k] for k in ("model_family","participant","posture","calibration_source","calibration_forward_model",
        "rho_soft_ohm_m","rho_fat_ohm_m","rho_common_ohm_m","fat_to_soft_ratio","soft_material_definition","soft_is_pure_muscle",
        "FEM60_recalibrated","final_forward_model")}


def export(complete=False):
    m=tf.read(WORK/"manifest.local.json");inputs=[dict(x) for x in m["inputs"]]
    lookup={(x["model_family"],x["participant"],x["posture"]):x for x in inputs}
    fits=[];fields=[];profiles=[]
    common_fields=readcsv(BOUNDED/"bounded_fields.csv")
    for stem,target in (("bounded_fits",fits),("bounded_fields",fields),("bounded_profile",profiles)):
        assert tf.sha(BOUNDED/(stem+".csv"))==m["common_csv_sha256"][stem]
        for original in readcsv(BOUNDED/(stem+".csv")):
            row=dict(original);inp=lookup[FAMILIES[0],row["participant"],row["posture"]]
            for key,value in metadata(inp).items():row.setdefault(key,value)
            row.update(canonical_replay=True,source_canonical_csv_sha256=m["common_csv_sha256"][stem])
            if stem=="bounded_fits":
                final=next(r for r in common_fields if (r["participant"],r["posture"],r["lung_upper_ohm_m"],r["size_mm"])==(row["participant"],row["posture"],row["lung_upper_ohm_m"],"60"))
                row.update(fem60_at_r0_ohm=inp["fem60_at_r0_ohm"],fem60_at_r0_error_ohm=inp["fem60_at_r0_error_ohm"],
                    fem60_after_lung_fit_ohm=final["predicted_Z_ohm"],fem60_after_lung_fit_error_ohm=final["prediction_error_ohm"])
            target.append(row)
    for job in m["jobs"]:
        for st in job["states"]:
            family,u,pose=st["model_family"],job["participant"],st["recording_posture"]
            inp=lookup[family,u,pose];path=WORK/u/(st["posture"]+".local.json")
            if not path.exists():assert not complete;continue
            data=tf.read(path);assert data["manifest_sha256"]==tf.sha(WORK/"manifest.local.json")
            points=data["profile"];points=[points] if isinstance(points,dict) else points
            baseline=data.get("baseline",st["baseline"])
            if isinstance(baseline,dict):baseline=[baseline]
            b60=next((p for p in baseline if p["size_mm"]==60),None)
            if b60:inp.update(fem60_at_r0_ohm=b60["Z_ohm"],fem60_at_r0_error_ohm=b60["Z_ohm"]-inp["measured_Z60_ohm"],baseline60_solve_sha256=b60["source_sha256"])
            for upper in (12.,17.,30.):
                for n in points:
                    if n["rho_lung_ohm_m"]>upper:continue
                    profiles.append(dict(metadata(inp),lung_upper_ohm_m=upper,lung_lower_ohm_m=st["rho_fixed"][0],
                        **{k:n[k] for k in ("rho_lung_ohm_m","Z100_ohm","Z140_ohm","residual100_ohm","residual140_ohm","objective_ohm2","gradient_logrho_ohm2")},
                        gradient_sign=1 if n["gradient_logrho_ohm2"]>0 else -1 if n["gradient_logrho_ohm2"]<0 else 0,
                        j_lung100_ohm=n["fields"][0]["J_logrho_ohm"][1],j_lung140_ohm=n["fields"][1]["J_logrho_ohm"][1],
                        solve100_sha256=n["fields"][0]["source_sha256"],solve140_sha256=n["fields"][1]["source_sha256"],
                        all_requests_within_lower_30=True,global_monotonicity_proven=False,canonical_replay=False))
            results=data["results"];results=[results] if isinstance(results,dict) else results
            for result in results:
                n=result["chosen"];q=n["rho_lung_ohm_m"];upper=result["lung_upper_ohm_m"]
                assert st["rho_fixed"][0]<=q<=upper<=30
                meta=dict(metadata(inp),lung_upper_ohm_m=upper,lung_lower_ohm_m=st["rho_fixed"][0],
                    interval_role="primary" if upper==17 else "tighter" if upper==12 else "limiting_check",
                    rho_lung_ohm_m=q,rho_bone_ohm_m=st["rho_fixed"][2],rho_heart_ohm_m=st["rho_fixed"][3],baseline_lung_rho_ohm_m=st["rho_fixed"][1],
                    fit_status=result["status"],boundary=result["boundary"],boundary_is_given_condition=result["boundary"]!="none",
                    interior_conditional_estimate=result["interior_conditional_estimate"],conditional_rho_estimate_ohm_m=q if result["interior_conditional_estimate"] else None,
                    independent_physiological_identification=False,result_complete=data["complete"],canonical_replay=False,
                    all_requests_within_lower_30=True,frequency_hz=100000,material_base="gabriel100_proxy",pair="back_arm",placement="back_right",
                    fit_sizes_mm="100;140",holdout200_used_for_selection=False,backcheck60_used_for_selection=False,
                    fixed_parameters_identical=True,measurements_identical_to_fem60=True,original_contact_operators=True,
                    bounds_origin="author_given_conditions",global_minimum_proven=False,global_mesh_convergence_proven=False,physiological_validation=False,
                    baseline60_artifact_present=True,baseline_transfer_between_recordings="same_rho_assumption",manifest_sha256=tf.sha(WORK/"manifest.local.json"),
                    matlab_sha256=m["matlab_sha256"],kernel_sha256=m["kernel_sha256"],registry_sha256=m["registry_sha256"],windows_sha256=m["windows_sha256"])
                nominal=next(p for p in points if p["rho_lung_ohm_m"]==st["rho_fixed"][1])
                last60=next(p for p in result["predictions"] if p["size_mm"]==60)
                fits.append(dict(meta,**{k:n[k] for k in ("Z100_ohm","Z140_ohm","residual100_ohm","residual140_ohm","objective_ohm2","gradient_logrho_ohm2")},
                    baseline_objective_ohm2=nominal["objective_ohm2"],min_sampled_gradient_ohm2=result["min_gradient"],max_sampled_gradient_ohm2=result["max_gradient"],
                    checked_profile_nodes=result["checked_nodes"],sampled_gradient_sign_consistent=result["gradient_sign_consistent"],
                    fem60_at_r0_ohm=inp["fem60_at_r0_ohm"],fem60_at_r0_error_ohm=inp["fem60_at_r0_error_ohm"],
                    fem60_after_lung_fit_ohm=last60["Z_ohm"],fem60_after_lung_fit_error_ohm=last60["Z_ohm"]-inp["measured_Z60_ohm"]))
                for p in result["predictions"]:
                    k=p["size_mm"];fam="extended" if k==200 else "small";s=job["sources"][fam]
                    obs=next(o for o in st["observations"] if o["size_mm"]==k);base=next((b for b in baseline if b["size_mm"]==k),None)
                    assert p["rho_ohm_m"]==[st["rho_fixed"][0],q,st["rho_fixed"][2],st["rho_fixed"][3],st["rho_fixed"][4]]
                    assert tf.sha(p["source"])==p["source_sha256"]
                    j=p["J_logrho_ohm"]
                    fields.append(dict(meta,size_mm=k,role="holdout" if k==200 else "backcheck" if k==60 else "fit",
                        measured_Z_ohm=obs["measured_Z_ohm"],predicted_Z_ohm=p["Z_ohm"],prediction_error_ohm=p["Z_ohm"]-obs["measured_Z_ohm"],
                        baseline_Z_ohm=base["Z_ohm"] if base else None,delta_Z_from_r0_ohm=p["Z_ohm"]-base["Z_ohm"] if base else None,
                        baseline_solve_sha256=base["source_sha256"] if base else None,j_soft_ohm=j[0],j_lung_ohm=j[1],j_bone_ohm=j[2],j_heart_ohm=j[3],j_fat_ohm=j[4],
                        j_soft_fat_ohm=j[0]+j[4],soft_fat_derivative_mode="common_log_scale_fixed_ratio",
                        baseline_j_lung_ohm=base["J_logrho_ohm"][1] if base else None,mesh_family=fam,mesh_sha256=s["mesh_sha256"],contacts_sha256=s["contacts_sha256"],
                        shared_sha256=s["shared_sha256"],z_contact_ohm_m2=s["z_contact_ohm_m2"],nominal_contact_diameter_mm=5,material_fraction_sampling="q4096",
                        cross_mesh_from_fit=k==200,source_array_sha256=obs["array_sha256"],window_start_s=obs["start_s"],window_stop_s=obs["stop_s"],window_evidence=obs["window_evidence"],
                        system_relative_residual=p["qc"]["system_relative_residual"],reciprocity_ohm=p["qc"]["reciprocity_ohm"],solve_sha256=p["source_sha256"],field_is_exact_FEM=True))
            if complete:assert data["complete"] and len(results)==3 and len(baseline)==4
    if complete:
        assert len(fits)==36 and len(fields)==144 and len(inputs)==12
        assert len({(r["model_family"],r["participant"],r["posture"],float(r["lung_upper_ohm_m"]),int(r["size_mm"])) for r in fields})==144
        for src in m["historical_sources"]:assert tf.sha(src["path"])==src["sha256"]
        assert tf.sha(m["generated_driver"])==m["matlab_sha256"]
    for stem,rows in (("fem_inputs",inputs),("fem_fits",fits),("fem_fields",fields),("fem_profile",profiles)):
        rows.sort(key=lambda r:(r["model_family"],r["participant"],r["posture"],float(r.get("lung_upper_ohm_m",0)),float(r.get("size_mm",r.get("rho_lung_ohm_m",0)))))
        atomic_csv(stem,rows)
    if complete:tf.write(WORK/"export_receipt.local.json",dict(fits=len(fits),fields=len(fields),inputs=len(inputs),profile_rows=len(profiles),
        preparation_python_sha256=m["python_sha256"],exporter_python_sha256=tf.sha(__file__),generated_matlab_sha256=m["matlab_sha256"],
        wrapper_matlab_sha256=tf.sha(HERE/"run_families.m"),manifest_sha256=tf.sha(WORK/"manifest.local.json"),
        **{n+"_sha256":tf.sha(HERE/(n+".csv")) for n in ("fem_inputs","fem_fits","fem_fields","fem_profile")}))
    return len(fits),len(fields),len(profiles)


def run():
    m=tf.read(WORK/"manifest.local.json");assert tf.sha(HERE/"run_families.m")==m["wrapper_matlab_sha256"]
    for n in ("preferences","temp"):(WORK/n).mkdir(exist_ok=True)
    env=os.environ.copy();env.update(MATLAB_PREFDIR=str(WORK/"preferences"),TEMP=str(WORK/"temp"),TMP=str(WORK/"temp"))
    quote=lambda p:"'"+str(p).replace("'","''")+"'"
    command=f"addpath({quote(HERE)});run_families({quote(WORK/'manifest.local.json')});"
    attempt=len(list(WORK.glob("matlab_*.log")))+1;start=time.monotonic()
    with (WORK/f"launcher_{attempt}.log").open("w",encoding="utf-8") as log:
        proc=subprocess.Popen([shutil.which("matlab"),"-wait","-logfile",str(WORK/f"matlab_{attempt}.log"),"-batch",command],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        tf.write(WORK/"process.local.json",dict(pid=proc.pid,attempt=attempt,command=command));print("Single MATLAB PID",proc.pid,flush=True)
        previous=None
        while proc.poll() is None:
            time.sleep(10)
            try:
                counts=export()
                if counts!=previous:print("Partial fits/fields/profile",counts,flush=True);previous=counts
            except PermissionError:print("CSV locked; export retries next poll",flush=True)
        code=proc.returncode
    tf.write(WORK/f"exit_{attempt}.local.json",dict(exit_code=code,seconds=time.monotonic()-start))
    if code:raise RuntimeError(f"MATLAB exit {code}; see private log")
    print("Final",export(True),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("action",choices=("prepare","run","export"));arg=ap.parse_args()
    if arg.action=="prepare":prepare()
    elif arg.action=="run":run()
    else:print(export(all((WORK/u/"complete.local.json").exists() for u in ("U1","U2"))))
