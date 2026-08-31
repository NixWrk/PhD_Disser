"""Compute volume and basic tetrahedral defects for an ASCII/Binary Gmsh mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gmsh
import numpy as np


def inspect_mesh(path: Path, degenerate_tolerance_mm3: float) -> dict[str, Any]:
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        node_tags, flat_coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(flat_coords, dtype=np.float64).reshape(-1, 3)
        node_tags = np.asarray(node_tags, dtype=np.int64)

        sequential = (
            node_tags.size == coords.shape[0]
            and node_tags.size > 0
            and node_tags[0] == 1
            and node_tags[-1] == node_tags.size
            and np.all(np.diff(node_tags) == 1)
        )
        tag_to_row = None
        if not sequential:
            tag_to_row = np.full(int(node_tags.max()) + 1, -1, dtype=np.int64)
            tag_to_row[node_tags] = np.arange(node_tags.size, dtype=np.int64)

        element_types, _, element_nodes = gmsh.model.mesh.getElements(3)
        total_abs = 0.0
        total_signed = 0.0
        negative = 0
        weak_negative = 0
        degenerate = 0
        tetra_count = 0
        min_signed = np.inf
        min_abs = np.inf
        max_abs = 0.0
        chunk_size = 250_000

        for element_type, flat_nodes in zip(element_types, element_nodes):
            name, _, _, nodes_per_element, *_ = gmsh.model.mesh.getElementProperties(
                int(element_type)
            )
            if int(element_type) != 4 or int(nodes_per_element) != 4:
                raise RuntimeError(
                    f"Unsupported 3D element type {element_type} ({name}); "
                    "expected linear tetrahedra."
                )
            tetra = np.asarray(flat_nodes, dtype=np.int64).reshape(-1, 4)
            tetra_count += tetra.shape[0]
            for start in range(0, tetra.shape[0], chunk_size):
                tags = tetra[start : start + chunk_size]
                rows = tags - 1 if tag_to_row is None else tag_to_row[tags]
                a = coords[rows[:, 0]]
                b = coords[rows[:, 1]]
                c = coords[rows[:, 2]]
                d = coords[rows[:, 3]]
                signed = np.einsum(
                    "ij,ij->i", b - a, np.cross(c - a, d - a)
                ) / 6.0
                absolute = np.abs(signed)
                total_abs += float(absolute.sum())
                total_signed += float(signed.sum())
                weak_negative += int(np.count_nonzero(signed < 0.0))
                negative += int(np.count_nonzero(signed < -degenerate_tolerance_mm3))
                degenerate += int(
                    np.count_nonzero(absolute <= degenerate_tolerance_mm3)
                )
                min_signed = min(min_signed, float(signed.min()))
                min_abs = min(min_abs, float(absolute.min()))
                max_abs = max(max_abs, float(absolute.max()))

        return {
            "status": "mesh_qc_not_physical_validation",
            "mesh": str(path.resolve()),
            "nodes": int(coords.shape[0]),
            "tetrahedra": int(tetra_count),
            "bounds_min_mm": coords.min(axis=0).tolist(),
            "bounds_max_mm": coords.max(axis=0).tolist(),
            "absolute_volume_mm3": total_abs,
            "absolute_volume_l": total_abs / 1e6,
            "signed_volume_mm3": total_signed,
            "negative_tetrahedra": negative,
            "weak_negative_tetrahedra": weak_negative,
            "degenerate_tetrahedra": degenerate,
            "degenerate_tolerance_mm3": degenerate_tolerance_mm3,
            "min_signed_tetra_volume_mm3": min_signed,
            "min_abs_tetra_volume_mm3": min_abs,
            "max_abs_tetra_volume_mm3": max_abs,
        }
    finally:
        gmsh.finalize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--degenerate-tolerance-mm3", type=float, default=1e-12)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--allow-defects", action="store_true")
    args = parser.parse_args()

    report = inspect_mesh(args.mesh, args.degenerate_tolerance_mm3)
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    print(encoded)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded + "\n", encoding="utf-8")
    if not args.allow_defects and (
        report["negative_tetrahedra"] > 0 or report["degenerate_tetrahedra"] > 0
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
