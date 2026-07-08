"""Prepare STL surfaces for MATLAB/EIDORS Netgen meshing with PyMeshLab.

The script repairs common segmentation-surface issues, optionally decimates
large meshes, and writes an ASCII STL. It is intentionally conservative:
always inspect important anatomy after aggressive hole closing or decimation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pymeshlab


def mesh_stats(ms: pymeshlab.MeshSet) -> tuple[int, int]:
    mesh = ms.current_mesh()
    return mesh.vertex_number(), mesh.face_number()


def apply_filter(ms: pymeshlab.MeshSet, name: str, **kwargs) -> None:
    try:
        ms.apply_filter(name, **kwargs)
    except Exception as exc:  # noqa: BLE001 - keep batch repair moving.
        print(f"WARNING: {name} skipped: {exc}")


def prepare_stl(
    input_file: Path,
    output_file: Path,
    target_faces: int,
    close_holes: int,
    min_component_faces: int,
    merge_threshold: float | None,
    preserve_topology: bool = True,
    cluster_threshold: float = 0.0,
) -> None:
    ms = pymeshlab.MeshSet()
    print(f"Loading: {input_file}")
    ms.load_new_mesh(str(input_file))
    v0, f0 = mesh_stats(ms)
    print(f"Input mesh: {v0} vertices, {f0} faces")

    apply_filter(ms, "meshing_remove_duplicate_vertices")
    apply_filter(ms, "meshing_remove_duplicate_faces")
    apply_filter(ms, "meshing_remove_null_faces")
    apply_filter(ms, "meshing_remove_unreferenced_vertices")

    if merge_threshold is not None and merge_threshold > 0:
        apply_filter(
            ms,
            "meshing_merge_close_vertices",
            threshold=pymeshlab.PureValue(merge_threshold),
        )

    apply_filter(ms, "meshing_repair_non_manifold_vertices")
    apply_filter(ms, "meshing_repair_non_manifold_edges", method=0)
    apply_filter(ms, "meshing_remove_folded_faces")
    apply_filter(ms, "meshing_remove_unreferenced_vertices")

    if close_holes > 0:
        apply_filter(
            ms,
            "meshing_close_holes",
            maxholesize=close_holes,
            selected=False,
            newfaceselected=False,
            selfintersection=True,
            refinehole=False,
        )

    if min_component_faces > 0:
        apply_filter(
            ms,
            "meshing_remove_connected_component_by_face_number",
            mincomponentsize=min_component_faces,
            removeunref=True,
        )

    _, current_faces = mesh_stats(ms)
    if cluster_threshold > 0 and target_faces > 0 and current_faces > target_faces:
        apply_filter(
            ms,
            "meshing_decimation_clustering",
            threshold=pymeshlab.PureValue(cluster_threshold),
        )

    _, current_faces = mesh_stats(ms)
    if target_faces > 0 and current_faces > target_faces:
        apply_filter(
            ms,
            "meshing_decimation_quadric_edge_collapse",
            targetfacenum=target_faces,
            preservenormal=True,
            preservetopology=preserve_topology,
            optimalplacement=True,
            autoclean=True,
        )

    apply_filter(ms, "meshing_remove_duplicate_faces")
    apply_filter(ms, "meshing_remove_null_faces")
    apply_filter(ms, "meshing_remove_unreferenced_vertices")

    v1, f1 = mesh_stats(ms)
    print(f"Output mesh: {v1} vertices, {f1} faces")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving ASCII STL: {output_file}")
    try:
        ms.save_current_mesh(str(output_file), binary=False)
    except TypeError:
        ms.save_current_mesh(str(output_file))
        print("WARNING: PyMeshLab did not accept binary=False; verify STL output type.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target-faces", type=int, default=200_000)
    parser.add_argument("--close-holes", type=int, default=100)
    parser.add_argument("--min-component-faces", type=int, default=100)
    parser.add_argument("--merge-threshold", type=float, default=0.0)
    parser.add_argument("--allow-topology-change", action="store_true")
    parser.add_argument("--cluster-threshold", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_stl(
        input_file=args.input,
        output_file=args.output,
        target_faces=args.target_faces,
        close_holes=args.close_holes,
        min_component_faces=args.min_component_faces,
        merge_threshold=args.merge_threshold,
        preserve_topology=not args.allow_topology_change,
        cluster_threshold=args.cluster_threshold,
    )


if __name__ == "__main__":
    main()
