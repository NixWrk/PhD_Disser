"""Prepare the external YAROSLAV body STL for reference FEM meshing.

This pipeline is intended for an external body surface, not for soft tissue
with organs subtracted. It keeps the body anatomy much closer than the
convex-hull smoke-test while producing a watertight manifold surface.
"""

from __future__ import annotations

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
RESAMPLED = OUT / "YAROSLAV_BODY_resampled_3mm_ascii.stl"
MESHFIXED = OUT / "YAROSLAV_BODY_resampled_3mm_meshfix.stl"


def diagnose(path: Path) -> None:
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(path))
    mesh = ms.current_mesh()
    print(f"Diagnostics for {path.name}: {mesh.vertex_number()} vertices, {mesh.face_number()} faces")
    for filter_name in [
        "compute_selection_by_self_intersections_per_face",
        "compute_selection_from_mesh_border",
        "compute_selection_by_non_manifold_edges_per_face",
    ]:
        ms.set_selection_none()
        ms.apply_filter(filter_name)
        print(f"  {filter_name}: {mesh.selected_face_number()}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    ms = pymeshlab.MeshSet()
    print(f"Loading: {INPUT}")
    ms.load_new_mesh(str(INPUT))
    mesh = ms.current_mesh()
    print(f"Input mesh: {mesh.vertex_number()} vertices, {mesh.face_number()} faces")

    print("Generating 3 mm uniform remesh...")
    ms.apply_filter(
        "generate_resampled_uniform_mesh",
        cellsize=pymeshlab.PureValue(3),
        mergeclosevert=True,
        multisample=True,
    )
    ms.apply_filter("meshing_remove_duplicate_faces")
    ms.apply_filter("meshing_remove_null_faces")
    ms.apply_filter("meshing_remove_unreferenced_vertices")
    mesh = ms.current_mesh()
    print(f"Resampled mesh: {mesh.vertex_number()} vertices, {mesh.face_number()} faces")
    ms.save_current_mesh(str(RESAMPLED), binary=False)

    print("Running MeshFix watertight repair...")
    pymeshfix.clean_from_file(str(RESAMPLED), str(MESHFIXED), verbose=True, joincomp=True)

    diagnose(MESHFIXED)
    print(f"Saved: {MESHFIXED}")
    print(f"Done in {perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
