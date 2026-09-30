"""Local-only USD visual-QC export of the 12 observed Nix auto surfaces.

The source STL files are separately segmented phases, not a corresponding
deformed mesh. Phase indices are labels, not measured ECG timing. The USD and
all medical geometry stay in ignored runs/; only aggregate QC is tracked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh
from monai_physio import WorkflowConvertVTKToUSD
from pxr import Usd, UsdGeom


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_usd(points_mm: np.ndarray) -> np.ndarray:
    return points_mm[:, [0, 2, 1]] * np.array([0.001, 0.001, -0.001])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-root", type=Path, required=True)
    args = parser.parse_args()
    source_root = args.derived_root / "nix" / "products_v2"
    source_files = [source_root / f"phase_{i:02d}" / "whole_heart.stl" for i in range(12)]
    if not all(path.is_file() for path in source_files):
        raise FileNotFoundError("Expected all 12 Nix whole-heart phase STL files")
    # pyvista_stl currently fails on this Windows Unicode path; trimesh reads it.
    meshes = []
    for path in source_files:
        loaded = trimesh.load(str(path), process=False)
        if not isinstance(loaded, trimesh.Trimesh):
            raise ValueError("STL did not decode to a single triangular mesh")
        faces = np.column_stack((np.full(len(loaded.faces), 3), loaded.faces)).ravel()
        meshes.append(pv.PolyData(np.asarray(loaded.vertices), faces))
    if not all(isinstance(mesh, pv.PolyData) and mesh.n_points and mesh.n_cells for mesh in meshes):
        raise ValueError("A source STL is empty or not PolyData")
    # Each surface was extracted independently, so the converter must sample
    # both points and topology. Interpolation between phases is not interpreted.
    name = "mp09_nix_observed_auto_phases.usdc"
    workflow = WorkflowConvertVTKToUSD(
        input_meshes=meshes,
        usd_project_name=name,
        output_directory=RUNS,
        separate_by_connectivity=False,
        extract_surface=False,
        static_merge=False,
        time_codes=[float(i) for i in range(12)],
        frames_per_second=1.0,
        appearance="solid",
        solid_color=(0.78, 0.25, 0.25),
        log_level=logging.ERROR,
    )
    output = Path(workflow.process()["usd_file"])
    if output.resolve() != (RUNS / name).resolve():
        raise ValueError("USD output left ignored runs/")
    stage = Usd.Stage.Open(str(output))
    if stage is None:
        raise RuntimeError("OpenUSD cannot reopen the scene")
    mesh_prims = [UsdGeom.Mesh(prim) for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    if len(mesh_prims) != 1:
        raise ValueError(f"Expected one phase-sampled mesh, got {len(mesh_prims)}")
    mesh_prim = mesh_prims[0]
    times = list(mesh_prim.GetPointsAttr().GetTimeSamples())
    if times != [float(i) for i in range(12)]:
        raise ValueError("USD phase time samples are incomplete")
    units = UsdGeom.GetStageMetersPerUnit(stage)
    axis = UsdGeom.GetStageUpAxis(stage)
    if not (units == 1.0 and axis == UsdGeom.Tokens.y):
        raise ValueError(f"USD metadata mismatch: units={units}, axis={axis}")
    rows = []
    for index, (path, source) in enumerate(zip(source_files, meshes)):
        saved_points = np.asarray(mesh_prim.GetPointsAttr().Get(float(index)), dtype=np.float64)
        saved_faces = np.asarray(mesh_prim.GetFaceVertexCountsAttr().Get(float(index)))
        expected = to_usd(np.asarray(source.points, dtype=np.float64))
        if len(saved_points) != len(expected) or len(saved_faces) != source.n_cells:
            raise ValueError(f"USD vertex/face count mismatch for phase {index:02d}")
        centroid_error = float(np.linalg.norm(saved_points.mean(axis=0) - expected.mean(axis=0)))
        bounds_error = float(np.max(np.abs(np.stack((saved_points.min(axis=0), saved_points.max(axis=0))) -
                                           np.stack((expected.min(axis=0), expected.max(axis=0))))))
        if max(centroid_error, bounds_error) > 1e-5:
            raise ValueError(f"USD coordinates mismatch for phase {index:02d}")
        rows.append({
            "phase": f"phase_{index:02d}",
            "source_stl_sha256": sha256(path),
            "vertices": int(source.n_points),
            "faces": int(source.n_cells),
            "source_bounds_mm": np.round(np.asarray(source.bounds).reshape(3, 2), 3).tolist(),
            "usd_centroid_error_m": centroid_error,
            "usd_bounds_error_m": bounds_error,
        })
        print(f"phase_{index:02d}: USD readback okay", flush=True)
    report = {
        "schema": "mp09_nix_usd_visual_qc_v1",
        "subject_alias": "nix",
        "source": "existing independently segmented whole-heart STL in products_v2",
        "usd_relative_path": "runs/" + name,
        "usd_sha256": sha256(output),
        "phase_count": len(rows),
        "phase_indices_are_not_ecg_timing": True,
        "coordinate_conversion": "source RAS xyz mm -> USD (x,z,-y) m, Y-up",
        "between_phase_interpolation_validated": False,
        "topology_correspondence_between_phases": False,
        "independent_anatomical_validation": False,
        "registration_field_used": False,
        "omniverse_render_verified": False,
        "interpretation": "Observed auto-surface visual-QC artifact; no validated movement or FEM admission",
        "phases": rows,
    }
    (HERE / "mp09_nix_usd_visual_qc.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
