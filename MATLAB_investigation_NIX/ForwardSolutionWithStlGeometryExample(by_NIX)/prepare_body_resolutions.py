"""Prepare external BODY STL surfaces at several anatomical resolutions."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import pymeshfix
import pymeshlab


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "stl" / "raw"
INPUT = RAW / "YAROSLAV_BODY.stl"
if not INPUT.is_file():
    INPUT = ROOT / "YAROSLAV_BODY.stl"
OUT = ROOT / "output" / "prepared"


def diagnose(path: Path) -> tuple[int, int, int, int, int]:
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(path))
    mesh = ms.current_mesh()
    counts: list[int] = []
    for filter_name in [
        "compute_selection_by_self_intersections_per_face",
        "compute_selection_from_mesh_border",
        "compute_selection_by_non_manifold_edges_per_face",
    ]:
        ms.set_selection_none()
        ms.apply_filter(filter_name)
        counts.append(mesh.selected_face_number())
    return mesh.vertex_number(), mesh.face_number(), counts[0], counts[1], counts[2]


def prepare_resolution(cell_size_mm: float, overwrite: bool) -> Path:
    tag = f"{cell_size_mm:g}mm"
    resampled = OUT / f"YAROSLAV_BODY_resampled_{tag}_ascii.stl"
    meshfixed = OUT / f"YAROSLAV_BODY_resampled_{tag}_meshfix.stl"

    if meshfixed.exists() and not overwrite:
        print(f"Skipping existing {meshfixed.name}")
        return meshfixed

    started = perf_counter()
    print(f"\n=== BODY {tag} ===")
    print(f"Loading: {INPUT}")
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(INPUT))
    mesh = ms.current_mesh()
    print(f"Input mesh: {mesh.vertex_number()} vertices, {mesh.face_number()} faces")

    print(f"Generating {cell_size_mm:g} mm uniform remesh...")
    ms.apply_filter(
        "generate_resampled_uniform_mesh",
        cellsize=pymeshlab.PureValue(cell_size_mm),
        mergeclosevert=True,
        multisample=True,
    )
    ms.apply_filter("meshing_remove_duplicate_faces")
    ms.apply_filter("meshing_remove_null_faces")
    ms.apply_filter("meshing_remove_unreferenced_vertices")
    mesh = ms.current_mesh()
    print(f"Resampled mesh: {mesh.vertex_number()} vertices, {mesh.face_number()} faces")
    ms.save_current_mesh(str(resampled), binary=False)

    print("Running MeshFix watertight repair...")
    pymeshfix.clean_from_file(str(resampled), str(meshfixed), verbose=True, joincomp=True)

    vertices, faces, self_intersections, borders, non_manifold = diagnose(meshfixed)
    print(
        f"Diagnostics: {vertices} vertices, {faces} faces, "
        f"self_intersections={self_intersections}, borders={borders}, non_manifold={non_manifold}"
    )
    print(f"Saved: {meshfixed}")
    print(f"Done in {perf_counter() - started:.1f} s")
    return meshfixed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=float, default=[6.0, 4.0, 3.0])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for size in args.sizes:
        prepare_resolution(size, args.overwrite)


if __name__ == "__main__":
    main()
