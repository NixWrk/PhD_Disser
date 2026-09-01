"""Inventory existing reference STL files without changing source medical data.

This script deliberately reports geometry only.  It does not infer which cardiac
phase was segmented or whether a mesh represents blood, myocardium, or the whole
heart; those meanings must come from the segmentation protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_mesh(label: str, path: Path) -> dict[str, object]:
    # STL stores triangles independently; processing merges duplicate vertices so
    # watertightness and connected-component counts describe the actual surface.
    loaded = trimesh.load_mesh(path, process=True)
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.to_geometry()
    else:
        mesh = loaded

    bounds = np.asarray(mesh.bounds, dtype=float)
    return {
        "label": label,
        "source_filename": path.name,
        "sha256": sha256(path),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "components": int(len(mesh.split(only_watertight=False))),
        "bounds_mesh_units": bounds.tolist(),
        "extent_mesh_units": (bounds[1] - bounds[0]).tolist(),
        "signed_volume_mesh_units_cubed": float(mesh.volume),
        "notes": [
            "STL does not encode physical units or an anatomical coordinate system.",
            "Volume is meaningful only when the source export units and mesh closure are known.",
            "Cardiac phase and anatomical label are not inferred from the filename.",
        ],
    }


def parse_item(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH")
    return label, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--item", action="append", type=parse_item, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = {
        "status": "geometric_inventory_only",
        "items": [inspect_mesh(label, path) for label, path in args.item],
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
