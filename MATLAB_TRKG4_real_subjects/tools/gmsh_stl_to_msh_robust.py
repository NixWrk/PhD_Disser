"""Generate a tetrahedral Gmsh MSH 2.2 mesh without GUI/optimizer crashes."""

from __future__ import annotations

import argparse
from pathlib import Path

import gmsh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--mesh-size-mm", type=float, default=25.0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--algorithm-3d", type=int, default=4)
    parser.add_argument("--refine-axis-origin-mm", type=float, nargs=3)
    parser.add_argument("--refine-axis-vector-mm", type=float, nargs=3)
    parser.add_argument("--refine-radii-mm", type=float, nargs=3, default=(20.0, 40.0, 65.0))
    parser.add_argument("--refine-sizes-mm", type=float, nargs=3, default=(5.0, 8.0, 12.0))
    args = parser.parse_args()
    args.target.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.option.setNumber("General.NumThreads", args.threads)
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", args.threads)
        gmsh.option.setNumber("Mesh.MeshSizeMax", args.mesh_size_mm)
        gmsh.option.setNumber("Mesh.Algorithm3D", args.algorithm_3d)
        gmsh.option.setNumber("Mesh.Optimize", 0)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

        gmsh.model.add("trkg4_body")
        gmsh.merge(str(args.source.resolve()))
        surfaces = [tag for _dim, tag in gmsh.model.getEntities(2)]
        if not surfaces:
            raise RuntimeError("No surface entities were imported from STL")
        loop = gmsh.model.geo.addSurfaceLoop(surfaces)
        volume = gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(3, [volume], 1)
        if (args.refine_axis_origin_mm is None) != (args.refine_axis_vector_mm is None):
            raise ValueError("Both refinement axis origin and vector are required")
        if args.refine_axis_origin_mm is not None:
            cylinder_fields = []
            origin = args.refine_axis_origin_mm
            axis = args.refine_axis_vector_mm
            for radius, local_size in zip(args.refine_radii_mm, args.refine_sizes_mm):
                field = gmsh.model.mesh.field.add("Cylinder")
                gmsh.model.mesh.field.setNumber(field, "XCenter", origin[0])
                gmsh.model.mesh.field.setNumber(field, "YCenter", origin[1])
                gmsh.model.mesh.field.setNumber(field, "ZCenter", origin[2])
                gmsh.model.mesh.field.setNumber(field, "XAxis", axis[0])
                gmsh.model.mesh.field.setNumber(field, "YAxis", axis[1])
                gmsh.model.mesh.field.setNumber(field, "ZAxis", axis[2])
                gmsh.model.mesh.field.setNumber(field, "Radius", radius)
                gmsh.model.mesh.field.setNumber(field, "VIn", local_size)
                gmsh.model.mesh.field.setNumber(field, "VOut", args.mesh_size_mm)
                cylinder_fields.append(field)
            minimum = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", cylinder_fields)
            gmsh.model.mesh.field.setAsBackgroundMesh(minimum)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(args.target.resolve()))

        node_tags, _coords, _parametric = gmsh.model.mesh.getNodes()
        element_types, element_tags, _element_nodes = gmsh.model.mesh.getElements(3)
        tetrahedra = sum(len(tags) for tags in element_tags)
        print(f"ROBUST_GMSH source={args.source.resolve()} target={args.target.resolve()}")
        print(f"  surfaces={len(surfaces)} nodes={len(node_tags):,} volume_elements={tetrahedra:,}")
        print(f"  element_types={element_types}")
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
