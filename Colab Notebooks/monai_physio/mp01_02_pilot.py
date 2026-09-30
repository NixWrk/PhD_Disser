"""MP-01/02: local mask comparison and bounded mask-to-tetrahedra QC.

All outputs must go into a local ignored directory (normally runs/). No input
paths, image headers, masks, meshes or hashes are copied into Git by this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage import measure
import trimesh
from scipy.spatial import cKDTree


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mask(path: Path, assume_mm: bool) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError("Expected a 3D NIfTI mask")
    spatial_unit = image.header.get_xyzt_units()[0]
    if spatial_unit != "mm" and not assume_mm:
        raise ValueError(f"NIfTI spatial unit is {spatial_unit!r}; use --assume-mm only after checking provenance")
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-9:
        raise ValueError("Invalid or singular affine")
    values = np.asanyarray(image.dataobj)
    if not np.isfinite(values).all() or not np.isin(np.unique(values), [0, 1]).all():
        raise ValueError("Expected a finite binary 0/1 NIfTI; extract the intended label explicitly")
    mask = values.astype(bool)
    if not mask.any():
        raise ValueError("Empty mask")
    return image, mask


def grid_signature(image: nib.Nifti1Image) -> dict:
    return {"shape": list(image.shape), "affine_ras": image.affine.tolist(),
            "spatial_unit_header": image.header.get_xyzt_units()[0]}


def require_same_grid(a: nib.Nifti1Image, b: nib.Nifti1Image) -> None:
    if a.shape != b.shape or not np.allclose(a.affine, b.affine, rtol=0, atol=1e-4):
        raise ValueError("Shape/affine mismatch. Register or resample explicitly, then document interpolation.")


def mask_summary(mask: np.ndarray, affine: np.ndarray) -> dict:
    voxel_mm3 = abs(float(np.linalg.det(affine[:3, :3])))
    _, components = ndimage.label(mask)
    return {"voxels": int(mask.sum()), "volume_ml": float(mask.sum() * voxel_mm3 / 1000),
            "components_6_connected": int(components)}


def compare(args: argparse.Namespace) -> dict:
    baseline_image, baseline = load_mask(args.baseline, args.assume_mm)
    candidate_image, candidate = load_mask(args.candidate, args.assume_mm)
    require_same_grid(baseline_image, candidate_image)
    if args.ct is not None:
        ct = nib.load(str(args.ct))
        require_same_grid(baseline_image, ct)
    intersection = int(np.count_nonzero(baseline & candidate))
    union = int(np.count_nonzero(baseline | candidate))
    voxel_ml = abs(float(np.linalg.det(baseline_image.affine[:3, :3]))) / 1000
    return {
        "status": "automatic_mask_agreement_only",
        "label": args.label,
        "baseline_method": args.baseline_method,
        "candidate_method": args.candidate_method,
        "baseline_model_version": args.baseline_version,
        "candidate_model_version": args.candidate_version,
        "same_model_version": args.baseline_version == args.candidate_version,
        "same_ct_claim_requires_external_manifest_check": True,
        "input_sha256": {"baseline_mask": file_sha256(args.baseline),
                         "candidate_mask": file_sha256(args.candidate),
                         "ct": file_sha256(args.ct) if args.ct else None},
        "grid": grid_signature(baseline_image),
        "baseline": mask_summary(baseline, baseline_image.affine),
        "candidate": mask_summary(candidate, candidate_image.affine),
        "dice": 2 * intersection / (int(baseline.sum()) + int(candidate.sum())),
        "jaccard": intersection / union,
        "different_voxels": int(np.count_nonzero(baseline ^ candidate)),
        "symmetric_difference_ml": float(np.count_nonzero(baseline ^ candidate) * voxel_ml),
        "signed_volume_difference_ml": float((int(candidate.sum()) - int(baseline.sum())) * voxel_ml),
        "warning": "Agreement with TotalSegmentator is not independent anatomical validation.",
    }


def surface(mask: np.ndarray, affine: np.ndarray) -> trimesh.Trimesh:
    padded = np.pad(mask.astype(np.uint8), 1)
    vertices, faces, _, _ = measure.marching_cubes(padded, level=0.5)
    vertices -= 1
    vertices = nib.affines.apply_affine(affine, vertices)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def surface_qc(mesh: trimesh.Trimesh, reference_volume_ml: float) -> dict:
    volume_ml = abs(float(mesh.volume)) / 1000 if mesh.is_watertight else None
    return {"vertices": int(len(mesh.vertices)), "triangles": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight), "winding_consistent": bool(mesh.is_winding_consistent),
            "surface_components": int(len(mesh.split(only_watertight=False))),
            "euler_number": int(mesh.euler_number), "surface_area_mm2": float(mesh.area),
            "enclosed_volume_ml": volume_ml,
            "relative_volume_error_vs_voxels": abs(volume_ml / reference_volume_ml - 1) if volume_ml is not None else None}


def compare_surfaces(baseline: trimesh.Trimesh, candidate: trimesh.Trimesh, samples: int = 10000) -> dict:
    """Approximate symmetric distances on reproducibly sampled triangle points."""
    points_a = trimesh.sample.sample_surface(baseline, samples, seed=0)[0]
    points_b = trimesh.sample.sample_surface(candidate, samples, seed=0)[0]
    a_to_b = cKDTree(points_b).query(points_a, workers=1)[0]
    b_to_a = cKDTree(points_a).query(points_b, workers=1)[0]
    distances = np.r_[a_to_b, b_to_a]
    return {"method": "sampled nearest-neighbour approximation; not exact triangle ASSD/HD95",
            "samples_per_surface": samples, "sampled_assd_mm": float(np.mean(distances)),
            "sampled_hd95_mm": float(np.quantile(distances, 0.95)),
            "sampled_hausdorff_mm": float(np.max(distances)),
            "surface_area_difference_percent": float(100 * (candidate.area / baseline.area - 1)),
            "enclosed_volume_difference_ml": (
                float((abs(candidate.volume) - abs(baseline.volume)) / 1000)
                if baseline.is_watertight and candidate.is_watertight else None)}


def tetrahedralize(stl_path: Path, msh_path: Path, mesh_size_mm: float) -> dict:
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_mm)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_mm)
        gmsh.model.add("mp02_mask_geometry")
        gmsh.merge(str(stl_path))
        gmsh.model.mesh.classifySurfaces(np.pi / 3, True, True, np.pi)
        gmsh.model.mesh.createGeometry()
        surfaces = gmsh.model.getEntities(2)
        if not surfaces:
            raise RuntimeError("Gmsh did not identify a closed surface")
        loop = gmsh.model.geo.addSurfaceLoop([tag for _, tag in surfaces])
        gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)
        types, tags, nodes = gmsh.model.mesh.getElements(3)
        tetra_tags = np.concatenate([np.asarray(t) for typ, t in zip(types, tags) if typ in (4, 11)]) if types else np.array([])
        if len(tetra_tags) == 0:
            raise RuntimeError("No tetrahedra generated")
        qualities = gmsh.model.mesh.getElementQualities(tetra_tags.tolist(), "minSICN")
        gmsh.write(str(msh_path))
        return {"tetrahedra": int(len(tetra_tags)), "gmsh_surfaces": len(surfaces),
                "min_scaled_jacobian": float(np.min(qualities)),
                "p05_scaled_jacobian": float(np.quantile(qualities, 0.05)),
                "nonpositive_scaled_jacobian": int(np.count_nonzero(np.asarray(qualities) <= 0)),
                "gmsh_version": getattr(gmsh, "__version__", "unknown")}
    finally:
        gmsh.finalize()


def geometry(args: argparse.Namespace) -> dict:
    image, original = load_mask(args.mask, args.assume_mm)
    stride = args.stride
    if stride < 1:
        raise ValueError("--stride must be >=1")
    coarse = original[::stride, ::stride, ::stride]
    affine = image.affine @ np.diag([stride, stride, stride, 1.0])
    before_fill = mask_summary(coarse, affine)
    if args.closing_iterations < 0:
        raise ValueError("--closing-iterations must be nonnegative")
    coarse_unmodified = coarse.copy()
    if args.closing_iterations:
        coarse = ndimage.binary_closing(coarse, iterations=args.closing_iterations)
    if args.fill_holes:
        coarse = ndimage.binary_fill_holes(coarse)
    voxel = mask_summary(coarse, affine)
    mesh = surface(coarse, affine)
    qc = surface_qc(mesh, voxel["volume_ml"])
    result = {"status": "technical_geometry_qc_only", "mask_sha256": file_sha256(args.mask),
              "grid": grid_signature(image), "assumed_mm": bool(args.assume_mm),
              "stride": stride, "fill_holes": bool(args.fill_holes),
              "closing_iterations": args.closing_iterations,
              "changed_voxel_count_signed": voxel["voxels"] - before_fill["voxels"],
              "changed_voxels_symmetric": int(np.count_nonzero(coarse ^ coarse_unmodified)),
              "working_mask_before_fill": before_fill,
              "original_mask": mask_summary(original, image.affine),
              "working_mask": voxel, "surface": qc,
              "limitations": ["Stride subsampling changes geometry and volume; compare FEM only on equivalent input masks.",
                              "A closed surface alone does not validate anatomy, electrode placement or electrical response."]}
    if args.reference_surface:
        # STL stores triangle soup without shared vertex indices. Weld equal
        # coordinates before evaluating connectivity and watertightness.
        reference = trimesh.load_mesh(args.reference_surface, process=True)
        if not isinstance(reference, trimesh.Trimesh):
            raise ValueError("Reference surface must contain exactly one Trimesh")
        result["reference_surface_sha256"] = file_sha256(args.reference_surface)
        result["reference_surface"] = surface_qc(reference, result["original_mask"]["volume_ml"])
        result["surface_comparison"] = compare_surfaces(reference, mesh)
    if args.tetra:
        if not mesh.is_watertight or len(mesh.split(only_watertight=False)) != 1:
            result["tetra"] = {"blocked": "Surface must be one watertight connected component"}
        else:
            stl_path = args.output.with_name(args.output.stem + "_surface.stl")
            msh_path = args.output.with_name(args.output.stem + "_volume.msh")
            mesh.export(stl_path)
            try:
                result["tetra"] = tetrahedralize(stl_path, msh_path, args.mesh_size_mm)
            except Exception as exc:
                # A surface can be watertight yet fail Gmsh parametrization.
                # Preserve that negative result without claiming a FEM-ready mesh.
                result["tetra"] = {"blocked": type(exc).__name__, "reason": str(exc)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cmp = sub.add_parser("compare", help="MP-01: compare same-label binary NIfTI on one grid")
    cmp.add_argument("--baseline", type=Path, required=True)
    cmp.add_argument("--candidate", type=Path, required=True)
    cmp.add_argument("--ct", type=Path, help="Optional source CT NIfTI, checked for grid and hashed")
    cmp.add_argument("--label", required=True)
    cmp.add_argument("--baseline-method", default="TotalSegmentator")
    cmp.add_argument("--candidate-method", default="MONAI Physio TotalSegmentator wrapper")
    cmp.add_argument("--baseline-version", required=True)
    cmp.add_argument("--candidate-version", required=True)
    geo = sub.add_parser("geometry", help="MP-02: surface and optional Gmsh tetra QC")
    geo.add_argument("--mask", type=Path, required=True)
    geo.add_argument("--stride", type=int, default=1)
    geo.add_argument("--fill-holes", action="store_true", help="Explicit geometry alteration; record changed volume")
    geo.add_argument("--closing-iterations", type=int, default=0, help="Optional binary closing before surface extraction")
    geo.add_argument("--tetra", action="store_true")
    geo.add_argument("--mesh-size-mm", type=float, default=5)
    geo.add_argument("--reference-surface", type=Path, help="Existing STL on the same mask/phase, in RAS mm")
    for p in (cmp, geo):
        p.add_argument("--assume-mm", action="store_true")
        p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    allowed_root = (Path(__file__).resolve().parent / "runs").resolve()
    args.output = args.output.resolve()
    if args.output.parent != allowed_root:
        parser.error("--output must be a JSON directly inside this module's ignored runs/ directory")
    if args.output.exists():
        parser.error("Refusing to overwrite an existing report")
    if args.command == "compare" and (not args.baseline_version or not args.candidate_version):
        parser.error("Both model versions are required")
    result = compare(args) if args.command == "compare" else geometry(args)
    result["software_versions"] = {name: importlib.metadata.version(name)
                                    for name in ("numpy", "nibabel", "scipy", "scikit-image", "trimesh")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "report": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
