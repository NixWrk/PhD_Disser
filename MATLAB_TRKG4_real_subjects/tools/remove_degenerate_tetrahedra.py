"""Remove zero-volume linear tetrahedra from a Gmsh mesh copy.

This is a narrow repair step for meshes that preserve the source volume but
contain a few exactly degenerate tetrahedra.  Negative-volume tetrahedra are
never repaired silently: their presence aborts the command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import gmsh
import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repair_mesh(source: Path, target: Path, tolerance_mm3: float) -> dict[str, Any]:
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(source))

        node_tags, flat_coords, _ = gmsh.model.mesh.getNodes()
        node_tags = np.asarray(node_tags, dtype=np.int64)
        coords = np.asarray(flat_coords, dtype=np.float64).reshape(-1, 3)
        tag_to_row = np.full(int(node_tags.max()) + 1, -1, dtype=np.int64)
        tag_to_row[node_tags] = np.arange(node_tags.size, dtype=np.int64)

        removed_tags: list[int] = []
        negative_count = 0
        input_tetrahedra = 0

        for _, entity_tag in gmsh.model.getEntities(3):
            element_types, element_tags, element_nodes = (
                gmsh.model.mesh.getElements(3, entity_tag)
            )
            entity_removed: list[int] = []
            for element_type, tags, flat_nodes in zip(
                element_types, element_tags, element_nodes
            ):
                name, _, _, nodes_per_element, *_ = (
                    gmsh.model.mesh.getElementProperties(int(element_type))
                )
                if int(element_type) != 4 or int(nodes_per_element) != 4:
                    raise RuntimeError(
                        f"Unsupported 3D element type {element_type} ({name}); "
                        "expected linear tetrahedra."
                    )
                tetra = np.asarray(flat_nodes, dtype=np.int64).reshape(-1, 4)
                tags = np.asarray(tags, dtype=np.int64)
                input_tetrahedra += tetra.shape[0]
                points = coords[tag_to_row[tetra]]
                signed = np.einsum(
                    "ij,ij->i",
                    points[:, 1] - points[:, 0],
                    np.cross(
                        points[:, 2] - points[:, 0],
                        points[:, 3] - points[:, 0],
                    ),
                ) / 6.0
                negative_count += int(np.count_nonzero(signed < -tolerance_mm3))
                entity_removed.extend(
                    tags[np.abs(signed) <= tolerance_mm3].astype(int).tolist()
                )

            if negative_count:
                raise RuntimeError(
                    "Negative-volume tetrahedra were found; automatic removal "
                    "is intentionally disabled."
                )
            if entity_removed:
                gmsh.model.mesh.removeElements(3, entity_tag, entity_removed)
                removed_tags.extend(entity_removed)

        target.parent.mkdir(parents=True, exist_ok=True)
        # EIDORS 3.12 reads the original project's text MSH 2.2 files
        # reliably.  A binary MSH 4.1 rewrite of the same discrete mesh is
        # valid in Gmsh but fails in EIDORS' gmsh_read_mesh parser.
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(target))

        output_tetrahedra = sum(
            len(tags)
            for tags in gmsh.model.mesh.getElements(3)[1]
        )
        return {
            "status": "zero_volume_tetrahedra_removed_not_physical_validation",
            "source": str(source.resolve()),
            "source_sha256": file_sha256(source),
            "target": str(target.resolve()),
            "target_sha256": file_sha256(target),
            "degenerate_tolerance_mm3": tolerance_mm3,
            "input_tetrahedra": int(input_tetrahedra),
            "removed_tetrahedra": len(removed_tags),
            "removed_element_tags": removed_tags,
            "output_tetrahedra": int(output_tetrahedra),
            "negative_tetrahedra": negative_count,
        }
    finally:
        gmsh.finalize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--degenerate-tolerance-mm3", type=float, default=1e-12)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report = repair_mesh(
        args.source, args.target, args.degenerate_tolerance_mm3
    )
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    print(encoded)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
