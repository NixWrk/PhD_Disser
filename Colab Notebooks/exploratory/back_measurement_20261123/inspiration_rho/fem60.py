"""Four scalar full-FEM inverse hypotheses, calibrated on right posterior 60 mm.

python fem60.py prepare --archive <original ZIP>
python fem60.py run
python fem60.py export

No pulse inverse is performed. Raw medians use the frozen 8-s inspiration
windows, BEI1, back_arm, separately for U1/U2 and seated/lying. Pair-mean
medians and the alternating-sample defect remain explicit controls. Fitting
a defective baseline does not establish a reliable 60-mm pulse or tissue rho.

Scenarios: soft_only, fat_only, common_absolute, common_multiplier. The first
two vary one absolute rho; common_absolute sets rho_soft=rho_fat BEFORE the
operator is assembled; common_multiplier multiplies their distinct Gabriel
100-kHz bases by one positive factor and preserves their original ratio.
Lung, bone and original heart properties/geometry remain fixed. One lung.

Adaptive interpolation is used ONLY to propose the next scalar parameter.
Its knots and log derivatives come from full CEM solutions; accepted roots
and every final field are full canonical solutions or hash-verified exact
prior solutions with identical geometry, contacts and five rho. Two poses
share this static forward curve but are fitted independently. No equality
of physiological changes at different recording times is assumed.

The technical envelope is 0.1..100 ohm m for each variable absolute rho;
the multiplier envelope is their intersection. It is not a physiological
prior. Only the actually evaluated parameter span is called checked. Positive
derivatives at sampled states do not prove global monotonicity or exclude
roots outside the checked interval. Multiple sign-change brackets are kept
as an ambiguity, not silently reduced to one fitted value. No endpoint is
reported as a tissue estimate when the target was not attained.

Fields CSV is WIDE: one row per participant/posture/scenario/size_mm, with
signed j_soft_ohm, j_fat_ohm, j_lung_ohm, j_bone_ohm, j_heart_ohm, all dZ/dlnrho.
Sizes 60/100/140 use the same small mesh; 200 uses the existing extended mesh.
The 60 row is calibration, 100/140/200 independent baseline predictions.
Each observation has its own frozen window. These are conditional properties
on one CT geometry, not measured posture-specific tissue truth. Resistive
proxy, contact assumptions and lack of global mesh convergence remain.
Holdout comparisons transfer the fitted baseline rho between separate
recordings of sizes in one posture; this is an explicit stationarity assumption,
not an assumption that their pulses or instantaneous changes were identical.

Material reference: tissue_literature_v2.json, ITIS V5.0,
DOI 10.13099/VIP21000-05-0; Gabriel dispersion evaluated at 100000 Hz.
The assigned resistive proxies are Muscle / Lung Inflated / Cortical Bone /
Heart Muscle / Fat (average infiltrated), in the five unchanged CT masks.
Only rho=1/Re(sigma) is used; permittivity is not included. The mask called
soft_background also includes the existing artificial arm extensions.
There is no individual mixture rule or physiological validation of fitted rho.

Public writes: this file, fem60.m, fem60_fits.csv, fem60_fields.csv and the
compact fem60_control.csv of actually evaluated right-60 states only.
All raw-source references, plans, solve receipts and logs stay in the ignored
MATLAB_TRKG4_real_subjects/output/exploratory/back_measurement_20261123/
inspiration_rho/fem60 directory. Existing sources are read-only.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
spec = importlib.util.spec_from_file_location("posterior_fields", STUDY/"small_arrays/tissue_fields.py")
tf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tf)
ROOT, PIPE = tf.ROOT, tf.PIPE
WORK = PIPE/"output/exploratory/back_measurement_20261123/inspiration_rho/fem60"
SCENARIOS = ("soft_only", "fat_only", "common_absolute", "common_multiplier")


def rows(path):
    return list(csv.DictReader(Path(path).open(encoding="utf-8-sig", newline="")))


def observations(archive):
    sys.path.insert(0, str(STUDY))
    from analyze_recordings_20261119 import iter_exports, ARCHIVE_SHA256
    windows = {(r["participant"], r["posture"], r["pair"], int(r["size_mm"])): r
               for r in rows(STUDY/"inspiration_windows_20261119.csv")}
    prior = {(r["participant"], r["posture"], r["pair"], int(r["size_mm"])): r
             for r in rows(STUDY/"pulse_levels_20261119.csv") if r["window"] == "inspiration_candidate"}
    previous = {(r["participant"], r["posture"], int(r["size_mm"])): r for r in
                rows(STUDY/"small_arrays/fused_observations.csv") if r["pair"] == "back_arm"}
    out = []
    for key, data in iter_exports(archive):
        if key[2] != "back_arm":
            continue
        w = windows[key]
        a, b = int(float(w["start_s"])*1000), int(float(w["stop_s"])*1000)
        assert b-a == 8000
        z = data[a:b, 3]
        value = float(np.median(z))
        assert abs(value-float(prior[key]["ch1_median_Z_ohm"])) < 1e-6
        if key[3] != 200:
            assert abs(value-float(previous[key[0], key[1], key[3]]["measured_Z_ohm"])) < 1e-9
        out.append(dict(participant=key[0], posture=key[1], pair=key[2], size_mm=key[3],
            measured_Z_ohm=value, start_s=a/1000, stop_s=b/1000,
            q25_Z_ohm=float(np.quantile(z, .25)), q75_Z_ohm=float(np.quantile(z, .75)),
            pair_mean_median_Z_ohm=float(np.median(z.reshape(-1, 2).mean(1))),
            alternating_step_mohm=float(1000*np.median(np.diff(z)*(1-2*(data[a+1:b, 0] % 2)))),
            window_evidence=w["interpretation"], baseline_artifact_flag=key[3] == 60,
            archive_sha256=ARCHIVE_SHA256,
            array_sha256=hashlib.sha256(data.astype("<f8").tobytes()).hexdigest()))
    assert len(out) == 16
    return out


def prepare(archive):
    if (WORK/"manifest.local.json").exists():
        raise FileExistsError("Prepared run exists; use run/export without overwriting")
    assert subprocess.run(["git", "check-ignore", str(WORK/"probe.mat")], cwd=ROOT, capture_output=True).returncode == 0
    reg, _ = tf.verify_frequency()
    proxy = [reg["tissues"][t]["gabriel_at_frequency"]["100000"]["rho_ohm_m"] for t in tf.PROXIES]
    obs = observations(archive)
    prior = tf.read(tf.WORK/"manifest.local.json")["jobs"]
    fused_root = tf.WORK.parent/"fused_wall"
    fused = tf.read(fused_root/"manifest.local.json")
    assert fused["kernel_sha256"] == tf.KERNEL_HASH
    jobs = []
    for participant in ("U1", "U2"):
        sources, imports = {}, []
        for family in ("small", "extended"):
            sj = next(j for j in prior if j["participant"] == participant and j["family"] == family)
            source = Path(sj["source_plan"])
            p = tf.read(source)
            assert tf.sha(source) == tf.PLAN_HASHES[participant, family]
            assert tf.sha(p["mesh_path"]) == p["mesh_sha256"]
            assert tf.read(source.parent/"contact_qc.local.json")["passed"]
            shared_hash = tf.sha(p["shared_file"])
            cfile = source.parent/"contacts.mat"
            assert tf.sha(cfile) == sj["source_contacts_sha256"]
            sources[family] = dict(plan=str(source), plan_sha256=tf.sha(source),
                mesh_sha256=p["mesh_sha256"], shared_file=p["shared_file"], shared_sha256=shared_hash,
                contacts_file=str(cfile), contacts_sha256=tf.sha(cfile), z_contact_ohm_m2=p["z_contact_ohm_m2"])
            sizes = (60, 100, 140) if family == "small" else (200,)
            for size in sizes:
                f = Path(sj["plan"]).parent/f"gabriel100_proxy_right_L{size}.local.json"
                r = tf.read(f)
                assert r["plan_sha256"] == tf.sha(sj["plan"])
                np.testing.assert_array_equal(r["rho_ohm_m"], proxy)
                imports.append(dict(family=family, id=r["id"], rho_ohm_m=r["rho_ohm_m"],
                    Z_ohm=r["Z_ohm"], J_logrho_ohm=r["J_logrho_ohm"], qc=r["qc"],
                    source=str(f), source_sha256=tf.sha(f), provenance="previous_exact_Gabriel_FEM"))
        fj = next(j for j in fused["jobs"] if j["participant"] == participant)
        assert fj["source_plan_sha256"] == sources["small"]["plan_sha256"]
        assert fj["contacts_sha256"] == sources["small"]["contacts_sha256"]
        for f in (fused_root/participant).glob("solve_*.local.json"):
            r = tf.read(f)
            if r["id"] not in ("right_L60", "right_L100", "right_L140"):
                continue
            assert r["manifest_sha256"] == tf.sha(fused_root/"manifest.local.json")
            np.testing.assert_array_equal(np.asarray(r["rho_ohm_m"])[1:4], np.asarray(proxy)[1:4])
            imports.append(dict(family="small", id=r["id"], rho_ohm_m=r["rho_ohm_m"],
                Z_ohm=r["Z_ohm"], J_logrho_ohm=r["J_logrho_ohm"], qc=r["qc"],
                source=str(f), source_sha256=tf.sha(f), provenance="previous_exact_common_rho_FEM"))
        job = dict(participant=participant, sources=sources, imports=imports,
            observations=[o for o in obs if o["participant"] == participant],
            targets=sorted([o for o in obs if o["participant"] == participant and o["size_mm"] == 60], key=lambda o:o["measured_Z_ohm"]))
        jobs.append(job)
    m = dict(root=str(ROOT), work=str(WORK), jobs=jobs, rho_proxy=proxy, scenarios=list(SCENARIOS),
        kernel_sha256=tf.KERNEL_HASH, registry_sha256=tf.REGISTRY_HASH,
        matlab_sha256=tf.sha(HERE/"fem60.m"), python_sha256=tf.sha(__file__),
        observations=obs, windows_sha256=tf.sha(STUDY/"inspiration_windows_20261119.csv"),
        parser_sha256=tf.sha(STUDY/"analyze_recordings_20261119.py"),
        root_tolerance_ohm=1e-5, fd_log_step=.001, parameter_absolute_envelope_ohm_m=[.1,100.],
        root_iterations_max=10, expansion_evaluations_max=10, equal_log_step_limit=float(np.log(3)),
        condition_screen=1e4, derivative_screen_ohm=1e-4,
        inverse_search_envelope_is_physiological=False, global_root_uniqueness_proven=False,
        global_mesh_convergence_proven=False, physiological_validation=False)
    tf.write(WORK/"manifest.local.json", m)
    print("Prepared 16 conditional fits to 4 targets, 16 baseline observations; prior exact states:", [len(j["imports"]) for j in jobs], flush=True)
    for j in jobs:
        print(j["participant"], [(x["posture"], x["measured_Z_ohm"]) for x in j["targets"]], flush=True)


def run():
    m = tf.read(WORK/"manifest.local.json")
    assert tf.sha(HERE/"fem60.m") == m["matlab_sha256"]
    for d in ("preferences", "temp"):
        (WORK/d).mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(MATLAB_PREFDIR=str(WORK/"preferences"), TEMP=str(WORK/"temp"), TMP=str(WORK/"temp"))
    q = lambda x:"'"+str(x).replace("'", "''")+"'"
    attempt = len(list(WORK.glob("matlab_*.log")))+1
    command = f"addpath({q(HERE)});fem60({q(WORK/'manifest.local.json')});"
    start = time.monotonic()
    with (WORK/f"launcher_{attempt}.log").open("w", encoding="utf-8") as log:
        proc = subprocess.Popen([shutil.which("matlab"), "-wait", "-logfile", str(WORK/f"matlab_{attempt}.log"), "-batch", command],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        tf.write(WORK/"process.local.json", dict(pid=proc.pid, attempt=attempt, command=command))
        print("Single heavy MATLAB PID", proc.pid, flush=True)
        code = proc.wait()
    tf.write(WORK/f"exit_{attempt}.local.json", dict(exit_code=code, seconds=time.monotonic()-start))
    if code:
        raise RuntimeError(f"MATLAB exit {code}; inspect private matlab_{attempt}.log")
    print("All FEM60 jobs complete", flush=True)


def export():
    m = tf.read(WORK/"manifest.local.json")
    tf.verify_frequency()
    assert tf.sha(HERE/"fem60.m") == m["matlab_sha256"]
    assert tf.sha(PIPE/"src/trkg4_lateral_cem_response.m") == m["kernel_sha256"]
    fits, fields, controls = [], [], []
    for job in m["jobs"]:
        part = job["participant"]
        assert tf.read(WORK/part/"complete.local.json")["completed"]
        cache = tf.read(WORK/part/"cache.local.json")["rows"]
        for source in job["sources"].values():
            p = tf.read(source["plan"])
            for path, digest in ((source["plan"], source["plan_sha256"]), (p["mesh_path"], source["mesh_sha256"]),
                                 (source["shared_file"], source["shared_sha256"]), (source["contacts_file"], source["contacts_sha256"])):
                assert tf.sha(path) == digest
        for scenario in SCENARIOS:
            study = tf.read(WORK/part/(scenario+".local.json"))
            assert study["manifest_sha256"] == tf.sha(WORK/"manifest.local.json")
            assert study["small_complete"] and study["extended_complete"]
            for node, (logq, z, derivative) in enumerate(study["curve_logparameter_Z_J"]):
                q = float(np.exp(logq))
                rho = np.array(m["rho_proxy"])
                if scenario == "soft_only": rho[0] = q
                elif scenario == "fat_only": rho[4] = q
                elif scenario == "common_absolute": rho[[0,4]] = q
                else: rho[[0,4]] *= q
                state = next(s for s in reversed(cache) if s["family"] == "small" and s["id"] == "right_L60"
                             and np.max(np.abs(np.asarray(s["rho_ohm_m"])-rho)) < 1e-10)
                assert abs(state["Z_ohm"]-z) < 1e-10
                assert tf.sha(state["source"]) == state["source_sha256"]
                jj = state["J_logrho_ohm"]
                controls.append(dict(participant=part, scenario=scenario, node=node, parameter=q,
                    parameter_units="dimensionless_multiplier" if scenario == "common_multiplier" else "ohm_m",
                    rho_soft_ohm_m=rho[0], rho_fat_ohm_m=rho[4], predicted_Z_ohm=z,
                    j_scenario_ohm=derivative, j_soft_ohm=jj[0], j_fat_ohm=jj[4],
                    j_lung_ohm=jj[1], j_bone_ohm=jj[2], j_heart_ohm=jj[3],
                    provenance=state["provenance"], solve_sha256=state["source_sha256"],
                    mesh_sha256=job["sources"]["small"]["mesh_sha256"],
                    contacts_sha256=job["sources"]["small"]["contacts_sha256"],
                    material_base="gabriel100_proxy", frequency_hz=100000,
                    actual_full_FEM_node=True, interval_monotonicity_proven=False))
            rr = study["fits"] if isinstance(study["fits"], list) else [study["fits"]]
            for r in rr:
                observation = next(o for o in job["targets"] if o["posture"] == r["posture"])
                ok = r["status"] == "converged_full_FEM"
                rho = r["rho_ohm_m"] if ok else [None]*5
                nearest = min(study["curve_logparameter_Z_J"], key=lambda n:abs(n[1]-observation["measured_Z_ohm"]))
                common = dict(participant=part, posture=r["posture"], scenario=scenario,
                    frequency_hz=100000, material_base="gabriel100_proxy", placement="back_right", pair="back_arm",
                    fit_status=r["status"], root_converged=ok, ill_conditioned=r.get("ill_conditioned"),
                    parameter=r.get("parameter") if ok else None,
                    parameter_units="dimensionless_multiplier" if scenario == "common_multiplier" else "ohm_m",
                    rho_soft_ohm_m=rho[0], rho_fat_ohm_m=rho[4], rho_lung_ohm_m=m["rho_proxy"][1],
                    rho_bone_ohm_m=m["rho_proxy"][2], rho_heart_ohm_m=m["rho_proxy"][3],
                    calibration_size_mm=60, measured_Z60_ohm=observation["measured_Z_ohm"],
                    pair_mean_Z60_ohm=observation["pair_mean_median_Z_ohm"],
                    alternating_step60_mohm=observation["alternating_step_mohm"],
                    checked_parameter_min=study["checked_parameter_min"], checked_parameter_max=study["checked_parameter_max"],
                    checked_Z_min_ohm=study["checked_Z_min_ohm"], checked_Z_max_ohm=study["checked_Z_max_ohm"],
                    sampled_J_parameter_min_ohm=study["sampled_J_parameter_min_ohm"],
                    sampled_J_parameter_max_ohm=study["sampled_J_parameter_max_ohm"],
                    sampled_crossing_count=r["sampled_crossing_count"],
                    uniqueness_status=r["uniqueness_status"], global_uniqueness_proven=False,
                    numeric_condition=r.get("relative_condition") if ok else None,
                    condition_definition="abs(Z60/J_parameter)=abs(dlnparameter/dlnZ60)",
                    condition_screen=m["condition_screen"], derivative_screen_ohm=m["derivative_screen_ohm"],
                    J_parameter_ohm=r.get("J_parameter_ohm") if ok else None,
                    root_residual_ohm=r.get("residual_ohm") if ok else None,
                    root_tolerance_ohm=m["root_tolerance_ohm"],
                    FD_relative_error=r.get("FD_relative_error"),
                    FD_scope="right60_small_first_successful_posture_per_participant_scenario_only",
                    registry_sha256=m["registry_sha256"], kernel_sha256=m["kernel_sha256"],
                    manifest_sha256=tf.sha(WORK/"manifest.local.json"),
                    matlab_sha256=m["matlab_sha256"], windows_sha256=m["windows_sha256"],
                    archive_sha256=observation["archive_sha256"],
                    global_mesh_convergence_proven=False, conditional_effective_properties=True,
                    physiological_validation=False, fixed_ct_geometry=True, material_fraction_sampling="q4096",
                    baseline_transfer_between_recordings="same_fitted_rho_assumption",
                    baseline60_artifact_present=True)
                fits.append(dict(common, root_iterations=r.get("root_iterations",0),
                    checked_nodes=study["checked_nodes"], checked_derivative_is_sampled_not_interval_proof=True,
                    closest_checked_parameter=float(np.exp(nearest[0])) if not ok else None,
                    closest_checked_Z60_ohm=nearest[1] if not ok else None,
                    calibration_window_start_s=observation["start_s"], calibration_window_stop_s=observation["stop_s"],
                    calibration_q25_Z_ohm=observation["q25_Z_ohm"], calibration_q75_Z_ohm=observation["q75_Z_ohm"],
                    source_array_sha256=observation["array_sha256"], no_extrapolated_fit=True))
                predictions = r.get("predictions", [])
                if isinstance(predictions, dict):
                    predictions = [predictions]
                if ok:
                    assert sorted(x["size_mm"] for x in predictions) == [60,100,140,200]
                    assert abs(r["residual_ohm"]) <= m["root_tolerance_ohm"]
                    np.testing.assert_array_equal(np.asarray(rho)[1:4], np.asarray(m["rho_proxy"])[1:4])
                for size in (60,100,140,200):
                    fam = "extended" if size == 200 else "small"
                    obs = next(o for o in job["observations"] if o["posture"] == r["posture"] and o["size_mm"] == size)
                    pred = next((x for x in predictions if x["size_mm"] == size), None)
                    if pred:
                        assert tf.sha(pred["source"]) == pred["source_sha256"]
                        np.testing.assert_array_equal(pred["rho_ohm_m"], rho)
                        assert np.isfinite([pred["Z_ohm"], *pred["J_logrho_ohm"]]).all()
                        if size == 60:
                            assert abs(pred["Z_ohm"]-obs["measured_Z_ohm"]-r["residual_ohm"]) < 1e-10
                    jj = pred["J_logrho_ohm"] if pred else [None]*5
                    z = pred["Z_ohm"] if pred else None
                    fields.append(dict(common, size_mm=size, role="calibration" if size == 60 else "independent_prediction",
                        predicted_Z_ohm=z, measured_Z_ohm=obs["measured_Z_ohm"],
                        prediction_error_ohm=z-obs["measured_Z_ohm"] if pred else None,
                        j_soft_ohm=jj[0], j_fat_ohm=jj[4], j_lung_ohm=jj[1], j_bone_ohm=jj[2], j_heart_ohm=jj[3],
                        j_scenario_ohm=(jj[0] if scenario == "soft_only" else jj[4] if scenario == "fat_only" else jj[0]+jj[4]) if pred else None,
                        prediction_is_full_FEM=bool(pred), mesh_family=fam,
                        solve_sha256=pred["source_sha256"] if pred else None,
                        mesh_sha256=job["sources"][fam]["mesh_sha256"], shared_sha256=job["sources"][fam]["shared_sha256"],
                        contacts_sha256=job["sources"][fam]["contacts_sha256"],
                        z_contact_ohm_m2=job["sources"][fam]["z_contact_ohm_m2"], nominal_contact_diameter_mm=5,
                        cross_mesh_from_calibration=size == 200, own_observation_window_start_s=obs["start_s"],
                        own_observation_window_stop_s=obs["stop_s"], window_evidence=obs["window_evidence"],
                        pair_mean_measured_Z_ohm=obs["pair_mean_median_Z_ohm"],
                        system_relative_residual=pred["qc"]["system_relative_residual"] if pred else None,
                        reciprocity_ohm=pred["qc"]["reciprocity_ohm"] if pred else None))
    assert len(fits)==16 and len(fields)==64
    assert len({(r["participant"],r["posture"],r["scenario"]) for r in fits})==16
    assert len({(r["participant"],r["posture"],r["scenario"],r["size_mm"]) for r in fields})==64
    tf.csv_write(HERE/"fem60_fits.csv", fits)
    tf.csv_write(HERE/"fem60_fields.csv", fields)
    tf.csv_write(HERE/"fem60_control.csv", controls)
    tf.write(WORK/"export_receipt.local.json", dict(fits=len(fits), fields=len(fields),
        fit_table_sha256=tf.sha(HERE/"fem60_fits.csv"), field_table_sha256=tf.sha(HERE/"fem60_fields.csv"),
        control_table_sha256=tf.sha(HERE/"fem60_control.csv"), control_rows=len(controls),
        python_sha256=tf.sha(__file__), matlab_sha256=tf.sha(HERE/"fem60.m")))
    print("Exported 16 fit rows and 64 wide field rows", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("prepare","run","export"))
    ap.add_argument("--archive", type=Path)
    a = ap.parse_args()
    if a.action == "prepare":
        if a.archive is None:ap.error("--archive is required for source verification")
        prepare(a.archive)
    elif a.action == "run":run()
    else:export()
