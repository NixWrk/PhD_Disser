"""Bounded same-mesh CEM check; run on 2026-11-23 despite historical filename.

Reproduce the native-surface perturbations from placement_20261119.csv.
Preserve historical posterior FEM contact centres exactly at zero shift.
Apply the SAME fixed +/-5 mm vector and native-surface projection as in the
geometry check, transfer that realized displacement to the nominal FEM
centres, then project onto the FEM surface. Thus the historical zero-shift
offset between surfaces is held fixed rather than silently changing nominal
contacts. Compare actual centre motion and all four positions with the native
check. Call the unchanged canonical driver.
The two surface projections are not asserted to be identical. No remeshing, tissue
reassignment, inverse fit or modification of historical results is permitted.
Only the five-tissue base scenario at 50 kHz and nominal length 140 mm is used.
All private inputs, MATLAB preferences/logs and results stay in the supplied
ignored work directory. The public CSV contains aliases and aggregates only.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import trimesh
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PIPE = ROOT / "MATLAB_TRKG4_real_subjects"
WORK = PIPE / "output/exploratory/back_measurement_20261123/electrical_shift/matched_geometry_and_nominal"
GEOMETRY_CODE = HERE / "placement_20261119.py"
GEOMETRY_CSV = GEOMETRY_CODE.with_suffix(".csv")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def prepare(alias, extended_path):
    extended_path = extended_path.resolve()
    p = read(extended_path)
    folder = WORK / alias
    if folder.exists():
        raise FileExistsError("Private participant output already exists; use --export to inspect it")
    original_path = extended_path.parent.parent / "plan.local.json"
    original = read(original_path)
    reference = read(original_path.parent / "depth_maps.local.json")
    geometry = None
    for source in original["sources"]:
        if Path(source["path"]).name == "geometry.local.json":
            item = read(source["path"])
            if "placements" in item:
                assert sha(source["path"]) == source["sha256"]
                geometry = item
    assert geometry is not None
    with GEOMETRY_CSV.open(encoding="utf-8") as stream:
        rows = [r for r in csv.DictReader(stream) if r["participant"] == alias]
    assert len(rows) == 15
    for row in rows:
        assert row["producer_sha256"] == sha(GEOMETRY_CODE)
        assert row["plan_sha256"] == sha(original_path)
        assert row["surface_sha256"] == sha(p["native_body"])
    assert sha(p["mesh_path"]) == p["mesh_sha256"]
    assert sha(PIPE / "src/trkg4_lateral_cem_response.m") == p["canonical_kernel_sha256"]
    for source in p["fraction_files"]:
        assert sha(source["path"]) == source["sha256"]
    assert p["fraction_files"][1]["points"] == 4096
    assert p["frequency_hz"] == 50000 and p["nominal_diameter_mm"] == 5
    assert p["material_order"] == ["soft_background", "lung", "bone", "heart", "fat"]
    assert p["contact_area_relative_error_max"] == .1
    assert p["contact_centroid_error_mm_max"] == .75 and p["contact_extent_mm_max"] == 3.5
    native = trimesh.load_mesh(p["native_body"], process=True)
    mesh_data = loadmat(p["mesh_path"], variable_names=["nodes_mm", "boundary"])
    mesh = trimesh.Trimesh(mesh_data["nodes_mm"], mesh_data["boundary"]-1, process=False)
    del mesh_data
    contacts = []
    for row in rows:
        name, direction = row["placement"], row["direction"]
        amount = int(row["nominal_shift_mm"])
        rays = (geometry["placements"][name]["rays"] if name == "scorpion_lateral"
                else reference["line_profiles"][name.removeprefix("back_")]["rays"])
        xyz = np.array([r["origin_lps_mm"] for r in rays])
        axis = "transverse" if direction == "transverse" else "longitudinal"
        vector = np.array([float(row[axis+"_unit_LPS_"+d]) for d in "xyz"])
        assert abs(np.linalg.norm(vector)-1) < 1e-12
        # Exactly the same input points and vector as the geometry experiment.
        moved = xyz if amount == 0 else native.nearest.on_surface(xyz+amount*vector)[0]
        native_delta = moved[28]-xyz[28]
        np.testing.assert_allclose(np.linalg.norm(native_delta), float(row["realized_centre_shift_mm"]), atol=1e-9, rtol=0)
        np.testing.assert_allclose(np.dot(native_delta, vector), float(row["realized_centre_along_requested_axis_mm"]), atol=1e-9, rtol=0)
        indices = [0, 14, 42, 56]
        nominal = mesh.nearest.on_surface(xyz[indices])[0]
        if name.startswith("back_"):
            old_contact = next(c for c in p["contacts"] if c["size_mm"] == 140 and c["side"] == name.removeprefix("back_"))
            nominal = np.array(old_contact["centres_lps_mm"])
        requested_centres = nominal + moved[indices] - xyz[indices]
        centres = nominal if amount == 0 else mesh.nearest.on_surface(requested_centres)[0]
        centre_zero = mesh.nearest.on_surface(xyz[[28]])[0][0]
        centre_pair = np.array([centre_zero, centre_zero]) if amount == 0 else np.array([
            centre_zero, mesh.nearest.on_surface((centre_zero+native_delta)[None,:])[0][0]])
        actual_delta = centre_pair[1]-centre_pair[0]
        assert amount == 0 or amount*np.dot(actual_delta, vector) > 0
        historical_difference = None
        if amount == 0 and name.startswith("back_"):
            historical_difference = float(np.linalg.norm(centres-np.array(old_contact["centres_lps_mm"]), axis=1).max())
        suffix = "nominal" if amount == 0 else direction + ("_m5" if amount < 0 else "_p5")
        contacts.append(dict(id=name+"_"+suffix, side=name, size_mm=140,
            centres_lps_mm=centres.tolist(), roles=["I+", "U+", "U-", "I-"],
            native_centres_lps_mm=moved[indices].tolist(), direction=direction, nominal_shift_mm=amount,
            native_centre_shift_mm=float(np.linalg.norm(native_delta)),
            FEM_centre_shift_mm=float(np.linalg.norm(actual_delta)),
            FEM_centre_along_requested_axis_mm=float(np.dot(actual_delta, vector)),
            FEM_vs_native_centre_displacement_error_mm=float(np.linalg.norm(actual_delta-native_delta)),
            requested_unit_vector_LPS=vector.tolist(),
            max_native_vs_FEM_contact_position_difference_mm=float(np.linalg.norm(centres-moved[indices],axis=1).max()),
            max_FEM_projection_after_native_displacement_mm=float(np.linalg.norm(centres-requested_centres,axis=1).max()),
            historical_nominal_max_centre_difference_mm=historical_difference,
            geometry_direction_and_displacement_reproduced=True,
            FEM_centre_shift_sign_matches_geometry=True,
            geometry_centre_orientation_change_deg=float(row["centre_orientation_change_deg"])))
    # Reuse material matrices only after the MATLAB wrapper verifies their
    # original plan/mesh identity. The new driver plan must not claim they were
    # assembled for its newly defined contact list.
    p.pop("fraction_files")
    p.update(work=str(folder / "all"), root=str(ROOT), contacts=contacts,
             shared_sha256=sha(p["shared_file"]),
             scenarios=[s for s in p["scenarios"] if s["name"] == "base"])
    assert len(p["scenarios"]) == 1
    folder.mkdir(parents=True)
    for sub in ("all", "accepted", "preferences", "temp"):
        (folder/sub).mkdir()
    write(folder/"all/plan.local.json", p)
    metadata = dict(alias=alias, original_extended_plan=str(extended_path),
        original_extended_plan_sha256=sha(extended_path),
        original_contact_cache=str(extended_path.parent/"contacts.mat"),
        original_contact_cache_sha256=sha(extended_path.parent/"contacts.mat"),
        original_plan_contacts=read(extended_path)["contacts"],
        all_plan=str(folder/"all/plan.local.json"),
        geometry_csv_sha256=sha(GEOMETRY_CSV), geometry_code_sha256=sha(GEOMETRY_CODE),
        launcher_sha256=sha(__file__), wrapper_sha256=sha(HERE/"electrical_shift_20261119.m"),
        driver_sha256=sha(PIPE/"src/trkg4_back_comparison.m"))
    write(folder/"metadata.local.json", metadata)
    print(alias, "15 contact cases prepared; geometry centre perturbations reproduced", flush=True)


def launch(alias):
    folder = WORK / alias
    env = os.environ.copy()
    env.update(MATLAB_PREFDIR=str(folder/"preferences"), TEMP=str(folder/"temp"), TMP=str(folder/"temp"))
    quote = lambda value: "'"+str(value).replace("'", "''")+"'"
    command = f"addpath({quote(HERE)}); electrical_shift_20261119({quote(folder/'metadata.local.json')});"
    started = time.monotonic()
    with (folder/"launcher.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([shutil.which("matlab"), "-wait", "-logfile", str(folder/"matlab.log"), "-batch", command],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        print(alias, "MATLAB started", flush=True)
        code = process.wait()
    write(folder/"exit.local.json", dict(exit_code=code, seconds=time.monotonic()-started))
    print(alias, "MATLAB exit", code, flush=True)
    if code:
        raise RuntimeError("MATLAB failed; inspect private participant log")


def export():
    rows = []
    for alias in ("U1", "U2"):
        folder = WORK/alias
        meta = read(folder/"metadata.local.json")
        p = read(meta["all_plan"])
        qc = {r["id"]:r for r in read(folder/"all/contact_qc.local.json")["rows"]}
        selection = read(folder/"selection.local.json")
        allowed = selection["accepted_ids"]
        if isinstance(allowed, str):
            allowed = [allowed]
        nominal_status = {r["placement"]:r for r in selection["nominals"]}
        result = {}
        for c in p["contacts"]:
            f = Path(selection["result_work"])/("base_"+c["id"]+".local.json")
            if c["id"] in allowed and f.exists():
                result[c["id"]] = read(f)
        for c in p["contacts"]:
            q = qc[c["id"]]
            checks = q.get("contacts", [])
            if isinstance(checks, dict): checks = [checks]
            failures = []
            for e in checks:
                for metric, threshold in (("area_relative_error", .1), ("centroid_error_mm", .75), ("extent_mm", 3.5)):
                    if e[metric] > threshold:
                        failures.append(e["role"]+":"+metric)
            ns = nominal_status[c["side"]]
            row = dict(participant=alias, placement=c["side"], nominal_size_mm=140,
                direction=c["direction"], nominal_shift_mm=c["nominal_shift_mm"],
                contact_qc_passed=q["passed"], failed_contact_metrics=";".join(failures),
                patch_builder_failed=bool(q.get("failure")), nominal_qc_passed=ns["qc_passed"],
                historical_nominal_operator_reproduced=ns["operator_reproduced"] if c["side"].startswith("back_") else None,
                status="solved" if c["id"] in result else ("contact_qc_failed" if not q["passed"] else "nominal_control_failed"),
                maximum_area_relative_error=max((e["area_relative_error"] for e in checks), default=None),
                maximum_centroid_error_mm=max((e["centroid_error_mm"] for e in checks), default=None),
                maximum_extent_mm=max((e["extent_mm"] for e in checks), default=None),
                Z_ohm=None, J_lung_logrho_ohm=None, delta_Z_ohm=None, delta_Z_pct=None,
                delta_J_lung_ohm=None, delta_J_lung_pct=None, system_relative_residual=None, reciprocity_ohm=None,
                historical_nominal_delta_Z_ohm=None, historical_nominal_delta_J_lung_ohm=None)
            row.update({k:c[k] for k in ("native_centre_shift_mm", "FEM_centre_shift_mm", "FEM_centre_along_requested_axis_mm",
                "FEM_vs_native_centre_displacement_error_mm", "max_native_vs_FEM_contact_position_difference_mm",
                "max_FEM_projection_after_native_displacement_mm", "FEM_centre_shift_sign_matches_geometry",
                "historical_nominal_max_centre_difference_mm", "geometry_direction_and_displacement_reproduced",
                "geometry_centre_orientation_change_deg")})
            row.update(actual_contact_mean_shift_mm=None, actual_contact_mean_along_axis_mm=None,
                       actual_contact_mean_shift_sign_matches_geometry=None, actual_contact_axis_orientation_change_deg=None)
            nominal_checks=qc[c["side"]+"_nominal"].get("contacts", [])
            if len(checks)==4 and len(nominal_checks)==4:
                a=np.array([e["actual_centroid_lps_mm"] for e in checks])
                b=np.array([e["actual_centroid_lps_mm"] for e in nominal_checks])
                mean_delta=(a-b).mean(axis=0)
                along=float(np.dot(mean_delta,c["requested_unit_vector_LPS"]))
                correct=c["nominal_shift_mm"]==0 or c["nominal_shift_mm"]*along>0
                u=a[-1]-a[0];v=b[-1]-b[0]
                orientation=float(np.degrees(np.arccos(np.clip(np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v),-1,1))))
                row.update(actual_contact_mean_shift_mm=float(np.linalg.norm(mean_delta)),
                           actual_contact_mean_along_axis_mm=along,
                           actual_contact_mean_shift_sign_matches_geometry=correct,
                           actual_contact_axis_orientation_change_deg=orientation)
                if c["id"] in result: assert correct
            if c["id"] in result:
                z = result[c["id"]]
                base = result[c["side"]+"_nominal"]
                assert z["rho_ohm_m"] == base["rho_ohm_m"] == p["scenarios"][0]["rho_ohm_m"]
                row.update(Z_ohm=z["Z_ohm"], J_lung_logrho_ohm=z["J_logrho_ohm"][1],
                    delta_Z_ohm=z["Z_ohm"]-base["Z_ohm"], delta_Z_pct=100*(z["Z_ohm"]/base["Z_ohm"]-1),
                    delta_J_lung_ohm=z["J_logrho_ohm"][1]-base["J_logrho_ohm"][1],
                    delta_J_lung_pct=100*(z["J_logrho_ohm"][1]/base["J_logrho_ohm"][1]-1),
                    system_relative_residual=z["qc"]["system_relative_residual"], reciprocity_ohm=z["qc"]["reciprocity_ohm"])
                if c["direction"] == "none" and c["side"].startswith("back_"):
                    old = read(Path(meta["original_extended_plan"]).parent/("base_"+c["side"].removeprefix("back_")+"_L140.local.json"))
                    dz=z["Z_ohm"]-old["Z_ohm"]; dj=z["J_logrho_ohm"][1]-old["J_logrho_ohm"][1]
                    assert abs(dz) < 1e-8+1e-8*abs(old["Z_ohm"])
                    assert abs(dj) < 1e-8+1e-8*abs(old["J_logrho_ohm"][1])
                    row.update(historical_nominal_delta_Z_ohm=dz, historical_nominal_delta_J_lung_ohm=dj)
            row.update(frequency_hz=50000, nominal_contact_diameter_mm=5, nominal_contact_area_mm2=p["nominal_area_mm2"],
                z_contact_ohm_m2=p["z_contact_ohm_m2"], mesh_sha256=p["mesh_sha256"], shared_sha256=p["shared_sha256"],
                original_plan_sha256=meta["original_extended_plan_sha256"],
                geometry_csv_sha256=meta["geometry_csv_sha256"], canonical_kernel_sha256=p["canonical_kernel_sha256"],
                driver_sha256=meta["driver_sha256"], launcher_sha256=meta["launcher_sha256"], wrapper_sha256=meta["wrapper_sha256"])
            rows.append(row)
    with Path(__file__).with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
        writer=csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader();writer.writerows(rows)
    print("Exported",len(rows),"cases; solved",sum(r["status"]=="solved" for r in rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--u1-plan", type=Path)
    parser.add_argument("--u2-plan", type=Path)
    parser.add_argument("--export", action="store_true")
    args=parser.parse_args()
    if args.export:
        export();return
    if not args.u1_plan or not args.u2_plan or not shutil.which("matlab"):
        parser.error("Both original extended plans and the existing MATLAB executable are required")
    ignored=subprocess.run(["git", "check-ignore", str(WORK/"probe.local.json")], cwd=ROOT, capture_output=True)
    if ignored.returncode:
        raise ValueError("Private work directory must already be ignored")
    WORK.mkdir(parents=True, exist_ok=True)
    prepare("U1",args.u1_plan);prepare("U2",args.u2_plan)
    launch("U1");launch("U2");export()


if __name__ == "__main__":
    main()
