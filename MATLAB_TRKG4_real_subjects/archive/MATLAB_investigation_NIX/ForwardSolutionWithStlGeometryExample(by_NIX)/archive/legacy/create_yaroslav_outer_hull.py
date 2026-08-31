"""Create a robust convex-hull outer body STL for the YAROSLAV dataset."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pymeshlab


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "stl" / "raw"
PREFERRED_INPUT = ROOT / "output" / "prepared" / "YAROSLAV_SOFT_cluster2_250k_ascii.stl"
FALLBACK_INPUT = RAW / "YAROSLAV_SOFT.stl"
if not FALLBACK_INPUT.is_file():
    FALLBACK_INPUT = ROOT / "YAROSLAV_SOFT.stl"
OUTPUT = ROOT / "output" / "prepared" / "YAROSLAV_SOFT_hull_ascii.stl"


def main() -> None:
    input_file = PREFERRED_INPUT if PREFERRED_INPUT.is_file() else FALLBACK_INPUT
    started = perf_counter()
    ms = pymeshlab.MeshSet()
    print(f"Loading: {input_file}")
    ms.load_new_mesh(str(input_file))
    print(
        f"Input mesh: {ms.current_mesh().vertex_number()} vertices, "
        f"{ms.current_mesh().face_number()} faces"
    )
    ms.apply_filter("generate_convex_hull")
    ms.apply_filter("meshing_re_orient_faces_coherently")
    print(
        f"Hull mesh: {ms.current_mesh().vertex_number()} vertices, "
        f"{ms.current_mesh().face_number()} faces"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ms.save_current_mesh(str(OUTPUT), binary=False)
    print(f"Saved: {OUTPUT}")
    print(f"Done in {perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
