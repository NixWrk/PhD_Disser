"""Read-only, path-free readiness audit for a 4D heart mask -> existing FEM.

All source paths are supplied at runtime. This tool neither registers images,
changes tissue labels, creates a mesh, nor computes electrical quantities.
"""

import argparse
import json
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def check(args):
    import nibabel as nib
    import numpy as np

    phase_root = Path(args.phase_dir)
    product_root = Path(args.products_dir)
    fem_root = Path(args.fem_root)
    names = (args.reference, args.target)
    mask_paths = [product_root / name / "whole_heart.nii.gz" for name in names]
    ct_paths = [phase_root / f"{name}.nii.gz" for name in names]
    manifest_file = phase_root / "conversion_manifest.json"
    mesh_file = fem_root / "output" / f"{args.subject}_body_arm_parameter_mesh_1mm_candidate.msh"
    mesh_qc_file = fem_root / "output" / f"{args.subject}_body_arm_parameter_mesh_1mm_qc.json"
    contact_file = fem_root / "output" / f"{args.subject}_trkg4_arm_electrode_model_tests_contract.json"
    stl_root = fem_root / "data" / args.subject / "stl"
    stl_names = ("body_arm_parameter_surface_1mm.stl", "lungs.stl", "heart.stl", "bones.stl")
    files = {"phase_manifest": manifest_file.is_file(), "mesh": mesh_file.is_file(),
             "mesh_qc": mesh_qc_file.is_file(), "contact_contract": contact_file.is_file(),
             "ct_pair": all(p.is_file() for p in ct_paths),
             "mask_pair": all(p.is_file() for p in mask_paths),
             "stl_set": all((stl_root / n).is_file() for n in stl_names)}
    out = {"schema": "mp02_fem_transfer_preflight_v1", "subject": args.subject,
           "phase_pair": list(names), "available": files,
           "status": "blocked", "reasons": []}
    if not all(files.values()):
        out["reasons"].append("Some required same-subject CT/FEM inputs are absent")
    if files["phase_manifest"] and files["ct_pair"] and files["mask_pair"]:
        manifest = read_json(manifest_file)
        scans = [nib.load(str(p)) for p in ct_paths]
        masks = [nib.load(str(p)) for p in mask_paths]
        same_grid = all(scans[0].shape == im.shape and
                        np.allclose(scans[0].affine, im.affine, atol=1e-5)
                        for im in scans[1:] + masks)
        out["ct_mask_common_grid"] = bool(same_grid)
        out["ct_shape"] = list(scans[0].shape)
        out["ct_spacing_mm"] = [round(float(x), 6) for x in scans[0].header.get_zooms()[:3]]
        out["phase_manifest_count"] = len(manifest.get("phases", []))
        if not same_grid:
            out["reasons"].append("CT and masks do not share one physical grid")
    if files["mesh_qc"]:
        q = read_json(mesh_qc_file)
        out["mesh_qc"] = {k: q.get(k) for k in
                          ("nodes", "tetrahedra", "negative_tetrahedra", "degenerate_tetrahedra")}
        if q.get("negative_tetrahedra") != 0 or q.get("degenerate_tetrahedra") != 0:
            out["reasons"].append("Existing FEM mesh has invalid tetrahedra")
    if files["contact_contract"]:
        c = read_json(contact_file)
        out["existing_contact_test"] = {
            "status": c.get("status"), "frequency_hz": c.get("frequency_hz"),
            "model_kinds": c.get("electrode_kinds"),
            "all_reciprocity_pass": c.get("all_reciprocity_pass")}
    # An explicit, independently checked CT-to-FEM transform is mandatory.
    # The Nix transform, a guessed axis flip, and existing STL bounds do not qualify.
    out["accepted_subject_specific_ct_to_fem_transform"] = False
    out["independent_anatomical_registration_qc"] = False
    out["accepted_full_resolution_registration_qc"] = False
    out["same_cem_material_and_mesh_contract_for_both_arms"] = False
    out["reasons"].extend([
        "No accepted subject-specific RAS-mm to FEM-mm transform supplied and checked",
        "No independent anatomical validation of the registered heart supplied",
        "Full-resolution deformation and FEM equivalence gates remain unverified",
    ])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", choices=("georg",), required=True)
    parser.add_argument("--phase-dir", required=True)
    parser.add_argument("--products-dir", required=True)
    parser.add_argument("--fem-root", required=True)
    parser.add_argument("--reference", default="phase_00")
    parser.add_argument("--target", default="phase_01")
    args = parser.parse_args()
    print(json.dumps(check(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
