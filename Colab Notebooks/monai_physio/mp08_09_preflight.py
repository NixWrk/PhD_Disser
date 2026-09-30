"""Read-only MP-08/09 preflight for an existing external 4D CT derivative.

No pixel arrays, registration, patient mesh, or raw path is written. The JSON
report contains aggregate geometry and readiness checks only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import date
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh


def package_state() -> dict:
    names = {"nibabel": "nibabel", "trimesh": "trimesh", "pyvista": "pyvista",
             "itk": "itk", "monai_physio": "monai-physio", "pxr": "usd-core"}
    state = {}
    for module, distribution in names.items():
        available = importlib.util.find_spec(module) is not None
        try:
            installed_version = version(distribution) if available else None
        except PackageNotFoundError:
            installed_version = None
        state[module] = {"available": available, "version": installed_version}
    return state


def inspect(derived_root: Path, subject: str) -> dict:
    if subject not in {"adam", "georg", "nix"}:
        raise ValueError("Use a supported pseudonym: adam, georg, or nix")
    subject_dir = derived_root / subject
    phase_dir = subject_dir / "phases"
    manifest = json.loads((phase_dir / "conversion_manifest.json").read_text(encoding="utf-8"))
    entries = manifest["phases"]
    if len(entries) != manifest["phase_count"]:
        raise ValueError("Manifest phase_count disagrees with phase entries")

    frames = []
    affines = []
    for entry in entries:
        filename = entry["nifti"]
        if Path(filename).name != filename:
            raise ValueError("Manifest contains a non-local phase filename")
        path = phase_dir / filename
        image = nib.load(str(path))  # NIfTI header only; no voxel array is loaded.
        if len(image.shape) != 3 or not np.isfinite(image.affine).all():
            raise ValueError(f"Invalid 3D geometry for {entry['phase_id']}")
        affine = image.affine
        spacing = np.linalg.norm(affine[:3, :3], axis=0)
        sform, scode = image.header.get_sform(coded=True)
        qform, qcode = image.header.get_qform(coded=True)
        frames.append(
            {
                "phase": entry["phase_id"],
                "shape": list(image.shape),
                "spacing_mm_from_affine": np.round(spacing, 6).tolist(),
                "axis_codes": "".join(nib.aff2axcodes(affine)),
                "spatial_unit_header": image.header.get_xyzt_units()[0],
                "sform_code": int(scode),
                "qform_code": int(qcode),
                "sform_qform_agree": bool(
                    scode > 0 and qcode > 0 and np.allclose(sform, qform, atol=1e-3)
                ),
                "manifest_geometry_agrees": bool(
                    list(image.shape) == entry["shape"]
                    and np.allclose(spacing, entry["voxel_spacing_mm"], atol=1e-3)
                ),
            }
        )
        affines.append(affine)

    reference = frames[0]
    same_grid = all(
        f["shape"] == reference["shape"]
        and np.allclose(a, affines[0], atol=1e-4)
        for f, a in zip(frames, affines)
    )
    source = subject_dir / "products_v2" / "phase_00" / "whole_heart.stl"
    mesh_check = {"available": source.is_file(), "readable": False, "within_ct_fov": None}
    if source.is_file():
        mesh = trimesh.load(str(source), process=False)
        if isinstance(mesh, trimesh.Trimesh) and len(mesh.vertices) > 0:
            corners = np.array(np.meshgrid(*[(0, n - 1) for n in reference["shape"]])).T.reshape(-1, 3)
            world = nib.affines.apply_affine(affines[0], corners)
            low, high = world.min(axis=0), world.max(axis=0)
            mesh_check = {
                "available": True,
                "readable": True,
                "triangles": int(len(mesh.faces)),
                "within_ct_fov": bool(
                    np.all(mesh.bounds[0] >= low - 1)
                    and np.all(mesh.bounds[1] <= high + 1)
                ),
            }

    return {
        "schema": "mp08_09_preflight_v1",
        "run_date": date.today().isoformat(),
        "subject_pseudonym": subject,
        "package_state": package_state(),
        "basis": "external conversion manifest, NIfTI headers, one existing STL; no image pixels read",
        "conversion_status": manifest["status"],
        "phase_tag": manifest["phase_keyword"],
        "phase_count": len(frames),
        "phase_ids": [f["phase"] for f in frames],
        "phase_tag_values": [e.get("dicom_phase_value") for e in entries],
        "same_grid": same_grid,
        "all_manifest_geometry_agrees": all(f["manifest_geometry_agrees"] for f in frames),
        "all_header_units_mm": all(f["spatial_unit_header"] == "mm" for f in frames),
        "header_spatial_units": sorted({f["spatial_unit_header"] for f in frames}),
        "any_sform_qform_disagreement": any(not f["sform_qform_agree"] for f in frames),
        "reference_geometry": reference,
        "mesh_preflight": mesh_check,
        "distinct_high_resolution_reference_supplied": False,
        "registration_qc_accepted": False,
        "mp08_reconstruction_ready": False,
        "mp09_patient_animation_ready": False,
        "interpretation": "Header/mesh preflight only; no motion, image quality, registration, or anatomical validation",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--subject", choices=["adam", "georg", "nix"], default="nix")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = inspect(args.derived_root, args.subject)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        if not args.report.name.startswith("mp08_09_"):
            parser.error("Report filename must start with mp08_09_")
        args.report.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
