"""Four-solve spatial control of the existing soft-tissue mask, not all arms.

Run prepare, run, export. Only this script, spatial_control.m and
spatial_control.csv are public. Private selection arrays, operators, manifests
and logs stay in ignored output/exploratory/back_measurement_20261123/
ttrkg_transfer/spatial_control. Existing FEM inputs/results are read only.

Use ACTUAL cylinder axes from geometry/surface/surface_qc.local.json, not the
requested pre-join centres (these differ). On each anatomical side, all four
tetrahedron vertices must be radially <= R-2 mm, at least 40 mm outward from
the more distal of the original skin bound/crop/join, and >=10 mm short of the
far cylinder cap. The region is a conservative whole-tetrahedron subset of
the two artificial distal cylinder cores; it is not all arms, segmented arm
anatomy, or independent measurement. Boundary-crossing tetrahedra remain in
the remainder. Retain each element's ORIGINAL q4096 soft-tissue fraction.

Canonical system_mat_fields is evaluated in 100000-element blocks with local
node renumbering and unchanged coordinates/connectivity. Weighted element
matrices reconstruct the WHOLE original soft operator and separately its right
and left distal subsets. Reconstruction must agree with saved soft K; no rows
of global K are cropped. Remainder = original soft K - right subset - left.
The forward K stays the original five-tissue Gabriel100 matrix. Extra adjoint
columns evaluate the two subsets and remainder in the SAME four solves used
to reproduce original Z and all five J. No extra finite-difference solves.

J=dZ/d ln rho [ohm], signed. Ratios to J_soft describe field sensitivity only,
not fractions of a measured pulse. Same q4096 CT mesh for arm and back right
100; same proxy properties, original heart, one lung. Contact impedances remain
different (.01 vs .00015915494309189532 ohm m2); four contacts per own-drive
solve, no simultaneous eight-contact cross-drive model. Electrode-to-BEI
polarity/complex mapping, global convergence and physiological validity are
not established. Outside-CT soft assignment is a modelling assumption.
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
import numpy as np
import pandas as pd
import psutil
from scipy.io import loadmat, savemat

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("spatial_arm_dependency", HERE/"arm_fields.py")
af = importlib.util.module_from_spec(spec);spec.loader.exec_module(af)
tf = af.tf
ROOT, PIPE = tf.ROOT, tf.PIPE
WORK = PIPE/"output/exploratory/back_measurement_20261123/ttrkg_transfer/spatial_control"


def prepare():
    if (WORK/"manifest.local.json").exists():
        raise FileExistsError("Prepared manifest exists; use run/export")
    assert subprocess.run(["git", "check-ignore", "-q", str(WORK/"probe.mat")], cwd=ROOT).returncode == 0
    reg, _ = tf.verify_frequency()
    rho = [reg["tissues"][k]["gabriel_at_frequency"]["100000"]["rho_ohm_m"] for k in tf.PROXIES]
    arm = tf.read(af.WORK/"manifest.local.json")
    posterior = tf.read(tf.WORK/"manifest.local.json")
    jobs = []
    for aj in arm["jobs"]:
        u = aj["participant"]
        sj = next(j for j in posterior["jobs"] if j["participant"] == u and j["family"] == "small")
        ap, bp = tf.read(aj["plan"]), tf.read(sj["plan"])
        original = tf.read(sj["source_plan"])
        assert ap["mesh_sha256"] == bp["mesh_sha256"] and ap["shared_sha256"] == bp["shared_sha256"]
        assert tf.sha(bp["mesh_path"]) == bp["mesh_sha256"]
        assert tf.sha(bp["shared_file"]) == bp["shared_sha256"]
        fraction = next(f for f in original["fraction_files"] if f["points"] == 4096)
        assert tf.sha(fraction["path"]) == fraction["sha256"]
        src = tf.read(Path(aj["paths"]["configuration"]).parent/"model_sources.local.json")
        surface = Path(src["source_mesh_directory"]).parent/"surface"
        geomfile, prepfile = surface/"surface_qc.local.json", surface/"preparation.local.json"
        geometry, prep = tf.read(geomfile), tf.read(prepfile)
        assert tf.sha(geomfile) == prep["products"][geomfile.name] and geometry["passed"]
        assert geometry["supplied_source_stl_sha256"] == original["native_body_sha256"]
        mesh = loadmat(bp["mesh_path"], variable_names=["nodes_mm", "elems"])
        nodes, elems = mesh["nodes_mm"], mesh["elems"].astype(np.int64)-1
        labels = np.zeros(len(elems), np.uint8)
        definitions = []
        for cy in geometry["synthetic_cylinders"]:
            d = cy["definition"]["direction"]
            assert cy["definition"]["axis"] == "x" and d in (-1, 1)
            side = 1 if d == -1 else 2
            native_bound = geometry["skin_bounds_mm"][0 if d == -1 else 1][0]
            low = max(d*native_bound, d*cy["definition"]["cut_x_mm"], d*cy["x_mm"])+40
            high, radius = d*cy["far_x_mm"]-10, cy["radius_mm"]-2
            radial = np.sum((nodes[:, 1:]-np.asarray(cy["centre_yz_mm"]))**2, axis=1)
            inside = (d*nodes[:, 0] >= low) & (d*nodes[:, 0] <= high) & (radial <= radius**2)
            selected = np.all(inside[elems], axis=1)
            assert selected.any() and not labels[selected].any()
            labels[selected] = side
            definitions.append(dict(anatomical_side="right" if d == -1 else "left", direction=d,
                outward_coordinate_min_mm=low, outward_coordinate_max_mm=high,
                actual_join_x_mm=cy["x_mm"], original_crop_x_mm=cy["definition"]["cut_x_mm"],
                actual_center_yz_mm=cy["centre_yz_mm"], core_radius_mm=radius,
                tetrahedra=int(selected.sum()), whole_tetrahedra=True))
        fractions = loadmat(fraction["path"], variable_names=["fractions", "outside_ct_fraction"])
        soft = fractions["fractions"][:, 0]
        assert len(soft) == len(elems) and np.all((soft >= 0) & (soft <= 1))
        for definition in definitions:
            ix = np.flatnonzero(labels == (1 if definition["direction"] == -1 else 2))
            xyz = nodes[elems[ix]]
            vol = abs(np.linalg.det(xyz[:, 1:]-xyz[:, :1]))/6000  # mm3 -> mL
            definition.update(geometric_volume_ml=float(vol.sum()),
                soft_weighted_volume_ml=float(vol@soft[ix]), soft_fraction_min=float(soft[ix].min()),
                soft_fraction_max=float(soft[ix].max()),
                outside_CT_weighted_fraction=float(vol@fractions["outside_ct_fraction"].ravel()[ix]/vol.sum()))
        folder = WORK/u;folder.mkdir(parents=True, exist_ok=True)
        selection = folder/"selection.local.mat"
        savemat(selection, {"region_label": labels[:, None]}, do_compression=True)
        sources = dict(mesh=bp["mesh_path"], shared=bp["shared_file"], fractions=fraction["path"],
            posterior_plan=sj["plan"], arm_plan=aj["plan"],
            posterior_contacts=str(Path(sj["plan"]).parent/"contacts.mat"),
            arm_contacts=str(Path(aj["plan"]).parent/"contacts.mat"),
            posterior_baseline=str(Path(sj["plan"]).parent/"gabriel100_proxy_right_L100.local.json"),
            arm_baseline=str(Path(aj["plan"]).parent/"gabriel100_proxy_arm_arm_top_1x1_step100.local.json"),
            geometry=str(geomfile), geometry_preparation=str(prepfile), selection=str(selection))
        for k in ("arm", "posterior"):
            np.testing.assert_array_equal(tf.read(sources[k+"_baseline"])["rho_ohm_m"], rho)
        jobs.append(dict(participant=u, paths=sources, hashes={k: tf.sha(v) for k, v in sources.items()},
            original_plan_sha256=sj["source_plan_sha256"], definitions=definitions,
            nodes=len(nodes), elements=len(elems)))
        print(u, [(d["anatomical_side"], d["tetrahedra"], round(d["soft_weighted_volume_ml"], 2))
                  for d in definitions], flush=True)
    manifest = dict(root=str(ROOT), work=str(WORK), jobs=jobs, rho=rho,
        scenario="gabriel100_proxy", frequency_hz=100000, element_block_size=100000,
        forward_solution_cap=4, kernel_sha256=tf.KERNEL_HASH, registry_sha256=tf.REGISTRY_HASH,
        matlab_code_sha256=tf.sha(HERE/"spatial_control.m"), python_code_sha256=tf.sha(__file__),
        arm_code_sha256=tf.sha(HERE/"arm_fields.py"), tissue_code_sha256=tf.sha(tf.__file__),
        region_definition=__doc__)
    tf.write(WORK/"manifest.local.json", manifest)


def run():
    receipt = sorted(af.WORK.glob("exit_*.local.json"))
    assert receipt and tf.read(receipt[-1])["exit_code"] == 0, "Arm run is not complete"
    if psutil.virtual_memory().available < 20*1024**3:
        raise RuntimeError("Less than 20 GiB available; no second large MATLAB permitted")
    for p in psutil.process_iter(["name"]):
        if "matlab" in (p.info["name"] or "").lower():
            raise RuntimeError("Visible MATLAB is active; coordinate first")
    m = tf.read(WORK/"manifest.local.json")
    assert tf.sha(HERE/"spatial_control.m") == m["matlab_code_sha256"]
    env = os.environ.copy()
    for name in ("preferences", "temp"):(WORK/name).mkdir(exist_ok=True)
    env.update(MATLAB_PREFDIR=str(WORK/"preferences"), TEMP=str(WORK/"temp"), TMP=str(WORK/"temp"))
    q = lambda p: "'"+str(p).replace("'", "''")+"'"
    command = f"addpath({q(HERE)});spatial_control({q(WORK/'manifest.local.json')});"
    attempt = len(list(WORK.glob("launcher_*.log")))+1;start=time.monotonic()
    with (WORK/f"launcher_{attempt}.log").open("w", encoding="utf-8") as logfile:
        p = subprocess.Popen([shutil.which("matlab"), "-wait", "-logfile", str(WORK/f"matlab_{attempt}.log"),
            "-batch", command], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", creationflags=subprocess.CREATE_NO_WINDOW)
        tf.write(WORK/"process.local.json", dict(pid=p.pid, attempt=attempt))
        print("Single MATLAB", p.pid, flush=True)
        for line in p.stdout:logfile.write(line);logfile.flush();print(line.rstrip(), flush=True)
        code=p.wait()
    tf.write(WORK/f"exit_{attempt}.local.json", dict(exit_code=code, elapsed_s=time.monotonic()-start))
    if code:raise RuntimeError(f"MATLAB failed with {code}; existing solve receipts retained")


def export():
    m=tf.read(WORK/"manifest.local.json");rows=[]
    for job in m["jobs"]:
        assembly=tf.read(WORK/job["participant"]/"assembly.local.json")
        for lead in ("arm", "posterior"):
            file=WORK/job["participant"]/f"{lead}.local.json";r=tf.read(file)
            assert r["reproduction_passed"] and r["J_partition_abs_error_ohm"] < 1e-8
            rows.append(dict(participant=job["participant"], placement="arm_arm_top_1x1_step100" if lead=="arm" else "back_right_100",
                scenario="gabriel100_proxy", frequency_hz=100000, rho_soft_ohm_m=m["rho"][0],
                Z_ohm=r["Z_ohm"], J_soft_ohm=r["J_soft_ohm"], J_distal_right_ohm=r["J_distal_right_ohm"],
                J_distal_left_ohm=r["J_distal_left_ohm"], J_distal_union_ohm=r["J_distal_union_ohm"],
                J_remainder_ohm=r["J_remainder_ohm"],
                signed_distal_to_soft_ratio=r["J_distal_union_ohm"]/r["J_soft_ohm"],
                deltaZ_distal_mohm_for_delta_ln_rho_0p001=r["J_distal_union_ohm"],
                original_Z_error_ohm=r["original_Z_error_ohm"], original_max_J_error_ohm=r["original_max_J_error_ohm"],
                J_partition_abs_error_ohm=r["J_partition_abs_error_ohm"], seconds=r["seconds"],
                **r["qc"], **assembly,
                region="whole_tetrahedra_in_two_distal_artificial_cylinder_cores_not_all_arms",
                region_right_soft_ml=job["definitions"][0]["soft_weighted_volume_ml"],
                region_left_soft_ml=job["definitions"][1]["soft_weighted_volume_ml"],
                native_skin_margin_mm=40, radial_margin_mm=2, distal_cap_margin_mm=10,
                source_mesh_sha256=job["hashes"]["mesh"], fractions_sha256=job["hashes"]["fractions"],
                geometry_sha256=job["hashes"]["geometry"], contacts_sha256=job["hashes"][lead+"_contacts"],
                result_sha256=tf.sha(file), manifest_sha256=tf.sha(WORK/"manifest.local.json"),
                kernel_sha256=m["kernel_sha256"], code_sha256=tf.sha(__file__), matlab_sha256=tf.sha(HERE/"spatial_control.m"),
                z_contact_ohm_m2=.01 if lead=="arm" else .00015915494309189532,
                independent_measurement=False, all_arm_anatomy=False, global_mesh_convergence_proven=False))
    assert len(rows)==4
    pd.DataFrame(rows).to_csv(HERE/"spatial_control.csv", index=False, float_format="%.12g")
    print(pd.DataFrame(rows)[["participant","placement","J_soft_ohm","J_distal_union_ohm","J_remainder_ohm","signed_distal_to_soft_ratio"]].to_string(index=False))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=("prepare","run","export"))
    globals()[p.parse_args().action]()
