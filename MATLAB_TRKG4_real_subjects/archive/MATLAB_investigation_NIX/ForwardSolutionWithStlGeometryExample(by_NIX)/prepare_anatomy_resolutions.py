"""Prepare BODY, lungs, bones, and blood STL surfaces at matched resolutions."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import pymeshfix
import pymeshlab


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "stl" / "raw"
OUT = ROOT / "output" / "prepared"

ANATOMY = {
    "BODY": "YAROSLAV_BODY.stl",
    "LUNGS": "YAROSLAV_LUNGS.stl",
    "BONES": "YAROSLAV_BONES.stl",
    "BLOOD": "YAROSLAV_BLOOD.stl",
}


def input_path(filename: str) -> Path:
    for candidate in [RAW / filename, ROOT / filename]:
        if candidate.is_file():
            return candidate
    return RAW / filename


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


def prepare_surface(name: str, source: Path, cell_size_mm: float, overwrite: bool) -> Path:
    tag = f"{cell_size_mm:g}mm"
    resampled = OUT / f"YAROSLAV_{name}_resampled_{tag}_ascii.stl"
    meshfixed = OUT / f"YAROSLAV_{name}_resampled_{tag}_meshfix.stl"

    if meshfixed.exists() and not overwrite:
        print(f"Skipping existing {meshfixed.name}")
        return meshfixed

    started = perf_counter()
    print(f"\n=== {name} {tag} ===")
    print(f"Loading: {source}")
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(source))
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
    parser.add_argument("--sizes", nargs="+", type=float, default=[6.0, 4.0, 3.0, 2.0])
    parser.add_argument("--parts", nargs="+", choices=ANATOMY.keys(), default=list(ANATOMY.keys()))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for size in args.sizes:
        for part in args.parts:
            prepare_surface(part, input_path(ANATOMY[part]), size, args.overwrite)


if __name__ == "__main__":
    main()
