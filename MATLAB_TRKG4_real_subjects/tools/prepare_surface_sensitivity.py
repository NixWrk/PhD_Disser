"""Import surface JSONs, then seal runnable CEM sensitivity studies after MATLAB QC.

Import preserves exact bytes. No forward solve or electrode-role correction occurs.
"""
from pathlib import Path
import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import os
import re
import zipfile

from electrode_sensitivity_contract import validate_study, load_study

ROOT = Path(__file__).resolve().parents[1]
PARAMS = ["soft", "heart", "lung"]
ROLES = ["I_plus", "V_plus", "V_minus", "I_minus"]


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def rel(path, base):
    return os.path.relpath(Path(path).resolve(), Path(base).resolve()).replace("\\", "/")


def validate_landmarks(value):
    expected = dict(schema="trkg4_ttrkg_surface_v3", modality="TTRKG",
        montage="four_electrode_surface", placement_mode="explicit_points",
        coordinate_system="surface_path_v1", units="mm", symmetry=None,
        reference_definition="user_selected_centres_on_external_body_surface")
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise ValueError(f"Unsupported surface JSON {key}: {value.get(key)!r}")
    if value.get("electrode_order") != ROLES:
        raise ValueError("Unexpected electrode order")
    centres = value.get("preliminary_electrode_centers_xyz_mm")
    if not isinstance(centres, dict) or set(centres) != set(ROLES):
        raise ValueError("Four named electrode centres are required")
    for key in ROLES:
        point = centres[key]
        if not isinstance(point, list) or len(point) != 3 or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in point):
            raise ValueError("Each centre must be a finite XYZ vector")
    if len({tuple(centres[k]) for k in ROLES}) != 4:
        raise ValueError("Coincident electrode centres")
    name = value.get("source_basename", "")
    if not name or Path(name).name != name or any(c in name for c in "/\\"):
        raise ValueError("Source basename required")
    return centres


def import_archive(args):
    base = args.output.resolve()
    if (base / "manifest.json").exists() or (base / "raw_json").exists():
        raise FileExistsError("Use a new preparation directory")
    template = load_study(args.template)
    profile = read(args.rho_profile)
    prepared = (args.template.parent / template["source"]["prepared_fem"]).resolve()
    if sha(prepared) != template["source"]["prepared_sha256"]:
        raise ValueError("Prepared FEM hash mismatch")
    if template["source"]["frequency_hz"] != profile["frequency_hz"]:
        raise ValueError("Frequency/profile mismatch")
    if not all(len(profile["bounds"][k]) == 2 and 0 < profile["bounds"][k][0] < profile["bounds"][k][1] for k in PARAMS):
        raise ValueError("Positive ordered rho bounds required")
    acceptance = read(args.mesh_acceptance)
    if not all(acceptance["checks"].values()) or sha(args.mesh) != acceptance["accepted_mesh_sha256"]:
        raise ValueError("Mesh acceptance mismatch")
    identity = read(args.model_identity)
    if identity["prepared_fem_sha256"] != sha(prepared) or identity["mesh_sha256"] != sha(args.mesh):
        raise ValueError("Prepared model is not bound to the accepted mesh")
    if not math.isfinite(args.diameter_mm) or args.diameter_mm <= 0 or not math.isfinite(args.z_contact) or args.z_contact <= 0:
        raise ValueError("Positive contact diameter and specific impedance required")
    items = []
    with zipfile.ZipFile(args.archive) as z:
        names = z.namelist()
        if len(names) != len(set(n.casefold() for n in names)):
            raise ValueError("Duplicate archive entries")
        for name in sorted(names):
            if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}\.json", name):
                raise ValueError("Only flat, named JSON files are supported")
            if z.getinfo(name).file_size > 1_000_000:
                raise ValueError("Oversized landmark file")
            raw = z.read(name)
            value = json.loads(raw.decode("utf-8-sig"))
            centres = validate_landmarks(value)
            if value["source_basename"] != args.surface.name:
                raise ValueError("Picker source does not match the supplied surface")
            items.append((name, raw, centres))
    if not items:
        raise ValueError("Empty archive")
    base.mkdir(parents=True, exist_ok=True)
    (base / "raw_json").mkdir()
    (base / "source_archive.zip").write_bytes(args.archive.read_bytes())
    (base / "rho_profile.json").write_bytes(args.rho_profile.read_bytes())
    manifest = dict(schema="surface_sensitivity_preparation_v1",
        status="exploratory_hypothesis_not_validated", source=dict(
            prepared_fem=rel(prepared, base), prepared_sha256=sha(prepared),
            mesh=rel(args.mesh, base), mesh_sha256=sha(args.mesh),
            mesh_acceptance=rel(args.mesh_acceptance, base), mesh_acceptance_sha256=sha(args.mesh_acceptance),
            model_identity=rel(args.model_identity, base), model_identity_sha256=sha(args.model_identity),
            surface=rel(args.surface, base), surface_basename=args.surface.name, surface_sha256=sha(args.surface),
            prepared_format=args.prepared_format),
        archive_sha256=sha(args.archive), profile_sha256=sha(base / "rho_profile.json"),
        frequency_hz=profile["frequency_hz"],
        contact=dict(diameter_mm=args.diameter_mm, area_mm2=math.pi*args.diameter_mm**2/4,
            z_contact_ohm_m2=args.z_contact, area_fraction=1.0,
            origin="User selected 5 mm surface discs; fixed specific contact impedance inherited from prior project comparison, not measured."),
        qc=dict(max_surface_distance_mm=5.0, max_centroid_offset_mm=args.diameter_mm/2,
            min_area_fraction=.95, max_area_ratio=1.5),
        parameters=template["parameters"], inputs=[],
        source_article="https://hal.science/hal-03482302v1",
        scope="User supplied surface centres on fixed CT geometry; source article motivates montage topology. No pulse, sphere displacement or local tissue probes.")
    coords = []
    for name, raw, centres in items:
        path = base / "raw_json" / name
        path.write_bytes(raw)
        manifest["inputs"].append(dict(id=Path(name).stem, json=rel(path,base), sha256=sha(path)))
        for role in ROLES:
            coords.append(dict(montage=Path(name).stem,role=role,**dict(zip(["x_mm","y_mm","z_mm"],centres[role]))))
    with (base / "requested_centres.csv").open("x", encoding="utf-8", newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(coords[0]));w.writeheader();w.writerows(coords)
    write(base / "manifest.json", manifest)
    (base / "manifest.sha256").write_text(sha(base / "manifest.json")+"\n",encoding="ascii")
    return manifest


def verify_manifest(base):
    manifest = read(base / "manifest.json")
    if sha(base / "manifest.json") != (base / "manifest.sha256").read_text().strip():
        raise ValueError("Manifest hash mismatch")
    s = manifest["source"]
    for name, digest in [("prepared_fem","prepared_sha256"),("mesh","mesh_sha256"),
            ("surface","surface_sha256"),("model_identity","model_identity_sha256"),
            ("mesh_acceptance","mesh_acceptance_sha256")]:
        if sha(base / s[name]) != s[digest]:
            raise ValueError("Changed source: "+name)
    for name in ["refinement_qc","refinement_export","parent_manifest"]:
        if name in s and sha(base/s[name])!=s[name+"_sha256"]:
            raise ValueError("Changed refinement provenance: "+name)
    if sha(base / "rho_profile.json") != manifest["profile_sha256"] or sha(base / "source_archive.zip") != manifest["archive_sha256"]:
        raise ValueError("Changed profile or archive")
    for item in manifest["inputs"]:
        if sha(base / item["json"]) != item["sha256"]:
            raise ValueError("Changed landmark JSON")
    return manifest


def make_states(profile):
    axes = {k:[lo+(hi-lo)*i/4 for i in range(5)] for k,(lo,hi) in profile["bounds"].items()}
    def state(sid,rho):
        return dict(id=sid,conductivity={k:1/rho[k] for k in PARAMS}|{"bone":1/profile["bone_fixed_rho_ohm_m"]})
    grid = []
    for number,idx in enumerate(itertools.product(range(5),repeat=3)):
        rho={k:axes[k][j] for k,j in zip(PARAMS,idx)}
        grid.append(("grid3" if all(i%2==0 for i in idx) else "grid5_added",state(f"rho_grid_n5_{number:06d}",rho),rho))
    for number,idx in enumerate(itertools.product(range(4),repeat=3)):
        rho={k:(axes[k][j]+axes[k][j+1])/2 for k,j in zip(PARAMS,idx)}
        grid.append(("check5",state(f"rho5_check_{number:06d}",rho),rho))
    mid={k:axes[k][2] for k in PARAMS}
    pilot=[state("pilot_base",mid)]
    for k in PARAMS:
        h=(axes[k][-1]-axes[k][0])*.001
        for step,mult in [("full",1),("half",.5)]:
            for direction,sign in [("plus",1),("minus",-1)]:
                pilot.append(state(f"pilot_{k}_{direction}_{step}",mid|{k:mid[k]+sign*h*mult}))
    return axes,grid,pilot


def seal(base):
    from resistivity_grid_convergence import DEFAULT_CRITERIA
    base=base.resolve()
    if (base/"surface_plan.json").exists() or (base/"studies").exists():
        raise FileExistsError("Use a new preparation; existing sealed studies are immutable")
    manifest=verify_manifest(base)
    prep=read(base/"contacts/preparation.json")
    if prep["status"] != "passed" or prep["manifest_sha256"] != sha(base/"manifest.json") or prep["prepared_sha256"] != manifest["source"]["prepared_sha256"]:
        raise ValueError("MATLAB geometry/CEM preparation has not passed")
    contacts={x["id"]:x for x in prep["montages"]}
    if len(contacts)!=len(prep["montages"]) or set(contacts)!={x["id"] for x in manifest["inputs"]}:
        raise ValueError("Prepared montages do not match archive")
    for item in manifest["inputs"]:
        c=contacts[item["id"]]
        if c["status"] != "passed" or c["source_json_sha256"] != item["sha256"] or sha(base/c["contact_matrix"]) != c["contact_sha256"] or not math.isfinite(c["matrix_error"]) or not 0 <= c["matrix_error"] < 1e-12:
            raise ValueError("Unverified contact model")
        for artifact in ["diagnostics", "geometry"]:
            if sha(base/c[artifact+"_file"]) != c[artifact+"_sha256"]:
                raise ValueError("Changed contact geometry/QC artifact")
    profile=read(base/"rho_profile.json")
    axes,states,pilot=make_states(profile)
    source=manifest["source"]
    plan=dict(manifest_sha256=sha(base/"manifest.json"), preparation_sha256=sha(base/"contacts/preparation.json"),
        profile_sha256=sha(base/"rho_profile.json"), prepared_sha256=source["prepared_sha256"],
        frequency_hz=manifest["frequency_hz"],montages=list(contacts),axes_rho=axes,
        criteria=dict(DEFAULT_CRITERIA),pilot_h_fraction=.001,states=[],batches=[],pilots=[],
        total_required_states=len(contacts)*189,new_fem_states=len(contacts)*189,reused_states=0,
        grid_states_per_montage=125,independent_centres_per_montage=64,
        status="prepared_not_computed",spatial_convergence_evaluated=False,physical_validation=False)
    def emit(mid,role,subset,index,pilot_run=False):
        c=contacts[mid]
        name=f"{mid}__{role}_{index:02d}"
        path=base/"studies"/(name+".json")
        study=dict(schema_version="electrode_sensitivity_v1",status="exploratory_hypothesis_not_validated",
            source=dict(prepared_fem=rel(base/source["prepared_fem"],path.parent),prepared_sha256=source["prepared_sha256"],frequency_hz=manifest["frequency_hz"]),
            parameters=copy.deepcopy(manifest["parameters"]),states=subset,montages=[dict(
                id=mid,contact_matrix=rel(base/c["contact_matrix"],path.parent),contact_sha256=c["contact_sha256"],
                electrodes=["E1","E2","E3","E4"],contact_geometry_provenance="User surface JSON; four equal 5 mm contacts; requested/realised geometry and source hashes in ../contacts/preparation.json",
                channels=[dict(id="standard",drive=[1,0,0,-1],measure=[0,1,-1,0])])],
            analysis=dict(map_states=[],save_fields=False))
        for p in study["parameters"]:
            p["nominal_sigma"]=1/profile["bone_fixed_rho_ohm_m"] if p["id"]=="bone" else 1/axes[p["id"]][2]
        write(path,validate_study(study))
        return dict(id=name,montage=mid,role=role,study=rel(path,base),study_sha256=sha(path),output=("pilot/" if pilot_run else "fem/")+name,states=len(subset))
    for mid in contacts:
        plan["pilots"].append(emit(mid,"pilot",pilot,0,True))
        for role,state,rho in states:
            plan["states"].append(dict(montage=mid,state=state["id"],role=role,rho=rho))
    for role in ["grid3","grid5_added","check5"]:
        subset=[s for r,s,_ in states if r==role]
        for off in range(0,len(subset),8):
            for mid in contacts:
                plan["batches"].append(emit(mid,role,subset[off:off+8],off//8))
    names=["run_electrode_sensitivity.py","electrode_sensitivity_contract.py","compartment_sensitivity.py",
        "resistivity_grid_convergence.py","resistivity_sensitivity_metrics.py","analyze_resistivity_coverage.py",
        "analyze_resistivity_refinement.py","run_surface_sensitivity.py","prepare_surface_sensitivity.py"]
    plan["sources_sha256"]={n:sha(ROOT/"tools"/n) for n in names}
    write(base/"surface_plan.json",plan)
    (base/"surface_plan.sha256").write_text(sha(base/"surface_plan.json")+"\n",encoding="ascii")
    return {k:plan[k] for k in ["status","total_required_states","montages"]}


def adopt_refinement(base):
    """Adopt a checked geometry-preserving refinement, retaining rejected contacts."""
    base=base.resolve()
    if (base/"surface_plan.json").exists():
        raise FileExistsError("A sealed study cannot change its mesh")
    m=verify_manifest(base)
    qc=read(base/"refinement/refinement.json")
    export=read(base/"refinement/export/refinement_export.json")
    mesh=base/"refinement/refined_mesh.mat"
    prepared=base/"refinement/export/prepared_fem.mat"
    if qc["status"]!="passed_geometry_preserving_local_refinement" or not all(qc["checks"].values()) or qc["refined_mesh_sha256"]!=sha(mesh) or qc["parent_prepared_sha256"]!=m["source"]["prepared_sha256"] or qc["manifest_sha256"]!=sha(base/"manifest.json"):
        raise ValueError("Unverified mesh refinement")
    if export["status"]!="passed" or export["matrix_error"]>=1e-12 or export["parent_prepared_sha256"]!=m["source"]["prepared_sha256"] or export["refined_mesh_sha256"]!=sha(mesh) or export["new_prepared_sha256"]!=sha(prepared) or export["refinement_qc_sha256"]!=sha(base/"refinement/refinement.json"):
        raise ValueError("Unverified refined FEM export")
    contacts=base/"contacts";archived=base/"contacts_before_refinement"
    if read(contacts/"preparation.json")["status"]!="failed" or archived.exists():
        raise ValueError("Expected a retained failed geometry attempt")
    if not contacts.resolve().is_relative_to(base) or not archived.resolve().is_relative_to(base):
        raise ValueError("Archive path escaped preparation directory")
    old_manifest=base/"manifest_before_refinement.json"
    if old_manifest.exists():raise FileExistsError(old_manifest)
    old_manifest.write_bytes((base/"manifest.json").read_bytes())
    contacts.rename(archived)
    acceptance=dict(status="accepted_for_surface_contact_preparation_not_solution_convergence",
        accepted_mesh_sha256=sha(mesh),checks=qc["checks"]|dict(eidors_volume_contact_matrix_agreement=True),
        parent_mesh_sha256=m["source"]["mesh_sha256"],refinement_qc_sha256=sha(base/"refinement/refinement.json"),
        export_sha256=sha(base/"refinement/export/refinement_export.json"),physical_validation=False,spatial_solution_convergence=False)
    write(base/"refinement/acceptance.json",acceptance)
    identity=dict(prepared_fem_sha256=sha(prepared),mesh_sha256=sha(mesh),parent_identity_sha256=m["source"]["model_identity_sha256"],
        parent_prepared_sha256=m["source"]["prepared_sha256"],export_sha256=sha(base/"refinement/export/refinement_export.json"),
        sources=[dict(path="src/trkg4_export_refined_surface_model.m",sha256=sha(ROOT/"src/trkg4_export_refined_surface_model.m"))],
        inputs=[dict(name="refined_mesh.mat",sha256=sha(mesh))],status="geometry_preserving_refinement_algebraic_check_passed")
    write(base/"refinement/identity.json",identity)
    m["source"].update(prepared_fem=rel(prepared,base),prepared_sha256=sha(prepared),
        mesh=rel(mesh,base),mesh_sha256=sha(mesh),prepared_format="si_geometry_si_blocks",
        mesh_acceptance="refinement/acceptance.json",mesh_acceptance_sha256=sha(base/"refinement/acceptance.json"),
        model_identity="refinement/identity.json",model_identity_sha256=sha(base/"refinement/identity.json"),
        refinement_qc="refinement/refinement.json",refinement_qc_sha256=sha(base/"refinement/refinement.json"),
        refinement_export="refinement/export/refinement_export.json",refinement_export_sha256=sha(base/"refinement/export/refinement_export.json"),
        parent_manifest="manifest_before_refinement.json",parent_manifest_sha256=sha(old_manifest))
    (base/"manifest.json").write_text(json.dumps(m,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (base/"manifest.sha256").write_text(sha(base/"manifest.json")+"\n",encoding="ascii")
    return dict(status="refined_volume_model_adopted_contacts_require_new_qc",counts=qc["counts"])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    q=sub.add_parser("import")
    for n in ["archive","template","rho-profile","mesh","mesh-acceptance","model-identity","surface","output"]:
        q.add_argument("--"+n,type=Path,required=True)
    q.add_argument("--prepared-format", choices=["legacy_40_05_mm_geometry_si_blocks", "si_geometry_si_blocks"], default="legacy_40_05_mm_geometry_si_blocks")
    q.add_argument("--diameter-mm",type=float,required=True)
    q.add_argument("--z-contact",type=float,required=True)
    q=sub.add_parser("seal");q.add_argument("--base",type=Path,required=True)
    q=sub.add_parser("adopt-refinement");q.add_argument("--base",type=Path,required=True)
    a=p.parse_args()
    out=import_archive(a) if a.command=="import" else (seal(a.base) if a.command=="seal" else adopt_refinement(a.base))
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
