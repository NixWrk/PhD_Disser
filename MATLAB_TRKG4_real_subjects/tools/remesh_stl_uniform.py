"""Uniformly remesh one STL with the historical PyMeshLab operation.

This is the active, path-independent replacement for the old subject-specific
prepare_*_resolutions.py scripts. MeshFix is optional because it can alter
anatomical volume; a repaired result is rejected when the configured volume
drift is exceeded.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

import pymeshlab


def _ascii_safe(path: Path) -> bool:
    try:
        str(path).encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


def _measure(mesh_set: pymeshlab.MeshSet) -> dict[str, Any]:
    mesh = mesh_set.current_mesh()
    result: dict[str, Any] = {
        "vertices": int(mesh.vertex_number()),
        "faces": int(mesh.face_number()),
    }
    try:
        measures = mesh_set.get_geometric_measures()
        volume = measures.get("mesh_volume")
        result["volume_mm3"] = None if volume is None else float(volume)
    except Exception as exc:  # diagnostics must not corrupt the geometry
        result["volume_mm3"] = None
        result["volume_error"] = str(exc)

    filters = {
        "self_intersection_faces": "compute_selection_by_self_intersections_per_face",
        "border_faces": "compute_selection_from_mesh_border",
        "non_manifold_edge_faces": "compute_selection_by_non_manifold_edges_per_face",
    }
    for key, filter_name in filters.items():
        try:
            mesh_set.set_selection_none()
            mesh_set.apply_filter(filter_name)
            result[key] = int(mesh.selected_face_number())
        except Exception as exc:
            result[key] = None
            result[f"{key}_error"] = str(exc)
    mesh_set.set_selection_none()
    return result


def _load(path: Path) -> pymeshlab.MeshSet:
    mesh_set = pymeshlab.MeshSet()
    mesh_set.load_new_mesh(str(path))
    return mesh_set


def _volume_change_percent(before: dict[str, Any], after: dict[str, Any]) -> float | None:
    v0 = before.get("volume_mm3")
    v1 = after.get("volume_mm3")
    if v0 in (None, 0) or v1 is None:
        return None
    return 100.0 * (float(v1) - float(v0)) / float(v0)


def remesh(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.resolve()
    target = args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if target.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; use --overwrite: {target}")

    started = perf_counter()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kalmykov_stl_") as temp_name:
        temp_dir = Path(temp_name)
        work_source = source
        if not _ascii_safe(source):
            work_source = temp_dir / "input.stl"
            shutil.copy2(source, work_source)
        raw_output = temp_dir / "uniform_remesh_ascii.stl"

        mesh_set = _load(work_source)
        input_stats = _measure(mesh_set)
        mesh_set.apply_filter(
            "generate_resampled_uniform_mesh",
            cellsize=pymeshlab.PureValue(args.cell_size_mm),
            mergeclosevert=True,
            multisample=True,
        )
        for filter_name in (
            "meshing_remove_duplicate_vertices",
            "meshing_remove_duplicate_faces",
            "meshing_remove_null_faces",
            "meshing_remove_unreferenced_vertices",
        ):
            mesh_set.apply_filter(filter_name)
        raw_stats = _measure(mesh_set)
        mesh_set.save_current_mesh(str(raw_output), binary=False)

        selected_output = raw_output
        final_stats = raw_stats
        repair_stats: dict[str, Any] | None = None
        if args.repair == "meshfix":
            import pymeshfix

            repaired_output = temp_dir / "meshfix_repaired.stl"
            pymeshfix.clean_from_file(
                str(raw_output), str(repaired_output), verbose=True, joincomp=True
            )
            repaired_set = _load(repaired_output)
            repair_stats = _measure(repaired_set)
            drift = _volume_change_percent(input_stats, repair_stats)
            repair_stats["volume_change_percent_from_input"] = drift
            if drift is not None and abs(drift) > args.max_volume_change_percent:
                raise RuntimeError(
                    "MeshFix result rejected: volume changed by "
                    f"{drift:.6g}% (limit {args.max_volume_change_percent:g}%)."
                )
            selected_output = repaired_output
            final_stats = repair_stats

        shutil.copy2(selected_output, target)

    report = {
        "status": "method_development_not_validated",
        "source": str(source),
        "output": str(target),
        "cell_size_mm": args.cell_size_mm,
        "repair": args.repair,
        "input": input_stats,
        "raw_uniform_remesh": raw_stats,
        "repair_result": repair_stats,
        "final": final_stats,
        "volume_change_percent_from_input": _volume_change_percent(input_stats, final_stats),
        "elapsed_seconds": perf_counter() - started,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cell-size-mm", type=float, required=True)
    parser.add_argument("--repair", choices=("none", "meshfix"), default="none")
    parser.add_argument("--max-volume-change-percent", type=float, default=0.5)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = remesh(args)
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    print(encoded)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
