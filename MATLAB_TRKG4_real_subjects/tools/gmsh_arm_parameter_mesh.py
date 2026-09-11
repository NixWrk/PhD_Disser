"""Build an adaptive torso/arm tetrahedral mesh for electrode tests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import gmsh


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("preparation_report", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("build_report", type=Path)
    parser.add_argument("--arm-size-mm", type=float, default=1.0)
    parser.add_argument("--electrode-size-mm", type=float, default=None)
    parser.add_argument("--electrode-inner-distance-mm", type=float, default=20.0)
    parser.add_argument("--electrode-outer-distance-mm", type=float, default=80.0)
    parser.add_argument("--electrode-refinement-half-width-mm", type=float, default=12.0)
    parser.add_argument("--trunk-size-mm", type=float, default=12.0)
    parser.add_argument("--transition-mm", type=float, default=5.0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--algorithm-3d", type=int, default=1)
    parser.add_argument("--largest-electrode-half-width-mm", type=float, default=7.5)
    args = parser.parse_args()
    if args.electrode_size_mm is None:
        args.electrode_size_mm = args.arm_size_mm
    if args.arm_size_mm <= 0 or args.trunk_size_mm <= 0:
        raise ValueError("Mesh sizes must be positive")
    if args.electrode_size_mm <= 0:
        raise ValueError("Electrode mesh size must be positive")
    if args.electrode_size_mm > args.arm_size_mm:
        raise ValueError("Electrode mesh size must not exceed arm size")
    if args.arm_size_mm >= args.trunk_size_mm:
        raise ValueError("Arm size must be smaller than trunk size")
    if args.electrode_inner_distance_mm <= 0 or args.electrode_outer_distance_mm <= 0:
        raise ValueError("Electrode distances must be positive")
    if args.electrode_refinement_half_width_mm <= 0:
        raise ValueError("Electrode refinement half-width must be positive")

    records = json.loads(args.preparation_report.read_text(encoding="utf-8"))
    if isinstance(records, dict):
        if "arm_extension" not in records:
            raise ValueError(
                "preparation report object has no arm_extension geometry"
            )
        body = records
    elif isinstance(records, list):
        body = next(
            (record for record in records
             if isinstance(record, dict) and record.get("name") == "body"),
            None,
        )
        if body is None:
            raise ValueError("legacy preparation report has no body record")
    else:
        raise ValueError("preparation report must be a JSON object or array")
    arms = body["arm_extension"]
    args.target.parent.mkdir(parents=True, exist_ok=True)
    args.build_report.parent.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.Verbosity", 3)
        gmsh.option.setNumber("General.NumThreads", args.threads)
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", args.threads)
        gmsh.option.setNumber("Mesh.MeshSizeMin", args.electrode_size_mm)
        gmsh.option.setNumber("Mesh.MeshSizeMax", args.trunk_size_mm)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.Algorithm3D", args.algorithm_3d)
        gmsh.option.setNumber("Mesh.Optimize", 0)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)

        gmsh.model.add("trkg4_arm_parameter_mesh")
        gmsh.merge(str(args.source.resolve()))
        surfaces = [tag for _dim, tag in gmsh.model.getEntities(2)]
        if not surfaces:
            raise RuntimeError("No surface entities were imported from STL")
        loop = gmsh.model.geo.addSurfaceLoop(surfaces)
        volume = gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(3, [volume], 1)

        fields: list[int] = []
        boxes: dict[str, dict[str, float]] = {}
        electrode_boxes: dict[str, dict[str, float]] = {}
        for side, sign in (("left", -1.0), ("right", 1.0)):
            arm = arms[side]
            first_edge_x = arm["cut_x"] + sign * (
                20.0 - args.largest_electrode_half_width_mm
            )
            if side == "left":
                x_min, x_max = arm["cylinder_x_min"], first_edge_x
            else:
                x_min, x_max = first_edge_x, arm["cylinder_x_max"]
            margin = arm["radius"] + args.arm_size_mm
            box = {
                "x_min": float(x_min),
                "x_max": float(x_max),
                "y_min": float(arm["center_y"] - margin),
                "y_max": float(arm["center_y"] + margin),
                "z_min": float(arm["center_z"] - margin),
                "z_max": float(arm["center_z"] + margin),
            }
            field = gmsh.model.mesh.field.add("Box")
            gmsh.model.mesh.field.setNumber(field, "VIn", args.arm_size_mm)
            gmsh.model.mesh.field.setNumber(field, "VOut", args.trunk_size_mm)
            gmsh.model.mesh.field.setNumber(field, "XMin", box["x_min"])
            gmsh.model.mesh.field.setNumber(field, "XMax", box["x_max"])
            gmsh.model.mesh.field.setNumber(field, "YMin", box["y_min"])
            gmsh.model.mesh.field.setNumber(field, "YMax", box["y_max"])
            gmsh.model.mesh.field.setNumber(field, "ZMin", box["z_min"])
            gmsh.model.mesh.field.setNumber(field, "ZMax", box["z_max"])
            gmsh.model.mesh.field.setNumber(field, "Thickness", args.transition_mm)
            fields.append(field)
            boxes[side] = box

            if args.electrode_size_mm < args.arm_size_mm:
                inner_x = arm["cut_x"] + sign * args.electrode_inner_distance_mm
                outer_x = inner_x + sign * args.electrode_outer_distance_mm
                for position, centre_x in (("inner", inner_x), ("outer", outer_x)):
                    slab = {
                        "x_min": float(centre_x - args.electrode_refinement_half_width_mm),
                        "x_max": float(centre_x + args.electrode_refinement_half_width_mm),
                        "y_min": float(arm["center_y"] - margin),
                        "y_max": float(arm["center_y"] + margin),
                        "z_min": float(arm["center_z"] - margin),
                        "z_max": float(arm["center_z"] + margin),
                    }
                    slab_field = gmsh.model.mesh.field.add("Box")
                    gmsh.model.mesh.field.setNumber(
                        slab_field, "VIn", args.electrode_size_mm
                    )
                    gmsh.model.mesh.field.setNumber(
                        slab_field, "VOut", args.trunk_size_mm
                    )
                    gmsh.model.mesh.field.setNumber(slab_field, "XMin", slab["x_min"])
                    gmsh.model.mesh.field.setNumber(slab_field, "XMax", slab["x_max"])
                    gmsh.model.mesh.field.setNumber(slab_field, "YMin", slab["y_min"])
                    gmsh.model.mesh.field.setNumber(slab_field, "YMax", slab["y_max"])
                    gmsh.model.mesh.field.setNumber(slab_field, "ZMin", slab["z_min"])
                    gmsh.model.mesh.field.setNumber(slab_field, "ZMax", slab["z_max"])
                    gmsh.model.mesh.field.setNumber(
                        slab_field, "Thickness", args.transition_mm
                    )
                    fields.append(slab_field)
                    electrode_boxes[f"{side}_{position}"] = slab
        minimum = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(minimum)

        gmsh.model.mesh.generate(3)
        gmsh.write(str(args.target.resolve()))
        node_tags, _coords, _parametric = gmsh.model.mesh.getNodes()
        element_types, element_tags, _element_nodes = gmsh.model.mesh.getElements(3)
        volume_elements = sum(len(tags) for tags in element_tags)
        report = {
            "status": "candidate_mesh_requires_external_quality_and_matlab_qc",
            "source": str(args.source.resolve()),
            "source_sha256": sha256(args.source),
            "target": str(args.target.resolve()),
            "arm_size_mm": args.arm_size_mm,
            "electrode_size_mm": args.electrode_size_mm,
            "trunk_size_mm": args.trunk_size_mm,
            "transition_mm": args.transition_mm,
            "electrode_inner_distance_mm": args.electrode_inner_distance_mm,
            "electrode_outer_distance_mm": args.electrode_outer_distance_mm,
            "electrode_refinement_half_width_mm": args.electrode_refinement_half_width_mm,
            "algorithm_3d": args.algorithm_3d,
            "threads": args.threads,
            "surface_entities": len(surfaces),
            "nodes": int(len(node_tags)),
            "volume_elements": int(volume_elements),
            "element_types": [int(item) for item in element_types],
            "arm_refinement_boxes_mm": boxes,
            "electrode_refinement_boxes_mm": electrode_boxes,
            "elapsed_seconds": perf_counter() - started,
        }
        args.build_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
