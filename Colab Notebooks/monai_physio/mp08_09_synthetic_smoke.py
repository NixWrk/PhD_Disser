"""Small synthetic MONAI Physio MP-08/09 API smoke; never reads patient data.

Writes only mp08_09_* results beside this script. Registration is deliberately
limited to two tiny images and a short rigid Greedy schedule.
"""

from __future__ import annotations

import json
import logging
from importlib.metadata import version
from pathlib import Path

import itk
import numpy as np
import pyvista as pv
from monai_physio import RegisterImagesGreedy, WorkflowConvertVTKToUSD, WorkflowReconstructHighres4DCT
from pxr import Usd, UsdGeom


ROOT = Path(__file__).resolve().parent


def synthetic_ct(shape: tuple[int, int, int], spacing: float, shift_xyz: tuple[float, float, float]):
    z, y, x = np.indices(shape, dtype=np.float32)
    x = x * spacing - shift_xyz[0]
    y = y * spacing - shift_xyz[1]
    z = z * spacing - shift_xyz[2]
    primary = np.exp(-(((x - 16) / 5) ** 2 + ((y - 17) / 4) ** 2 + ((z - 15) / 5) ** 2) / 2)
    landmark = np.exp(-(((x - 21) / 2) ** 2 + ((y - 12) / 2.5) ** 2 + ((z - 18) / 2) ** 2) / 2)
    array = (-1000 + 900 * primary + 450 * landmark).astype(np.float32)
    image = itk.image_from_array(array)
    image.SetSpacing((spacing, spacing, spacing))
    return image


def reconstruction_smoke() -> dict:
    shift = (2.0, -1.0, 0.0)  # known motion in physical xyz millimetres
    baseline = synthetic_ct((32, 32, 32), 1.0, (0.0, 0.0, 0.0))
    moved = synthetic_ct((32, 32, 32), 1.0, shift)
    highres = synthetic_ct((48, 48, 48), 0.75, (0.0, 0.0, 0.0))
    registrar = RegisterImagesGreedy(log_level=logging.ERROR)
    registrar.set_number_of_iterations([8, 4, 2])
    registrar.set_transform_type("Rigid")
    workflow = WorkflowReconstructHighres4DCT(
        time_series_images=[baseline, moved], reference_image=highres,
        reference_time_frame=0, register_reference_time_frame_to_reference_image=False,
        registration_method=registrar, log_level=logging.ERROR,
    )
    workflow.set_modality("ct")
    result = workflow.process()
    images = result["reconstructed_images"]
    fixed_to_moving = result["fixed_to_moving_transforms"][1]
    center = np.array([16.0, 16.0, 16.0])
    recovered = np.asarray(fixed_to_moving.TransformPoint(tuple(center))) - center
    return {
        "status": "completed_synthetic_registration_smoke",
        "input_lowres_shape_zyx": [32, 32, 32],
        "input_lowres_spacing_mm": 1.0,
        "reference_shape_zyx": [48, 48, 48],
        "reference_spacing_mm": 0.75,
        "known_shift_xyz_mm": list(shift),
        "recovered_fixed_to_moving_at_center_xyz_mm": np.round(recovered, 4).tolist(),
        "shift_error_norm_mm": round(float(np.linalg.norm(recovered - shift)), 4),
        "output_count": len(images),
        "output_sizes_xyz": [list(map(int, itk.size(image))) for image in images],
        "output_spacings_xyz_mm": [list(map(float, itk.spacing(image))) for image in images],
        "losses": [float(value) for value in result["losses"]],
        "registration_backend": "RegisterImagesGreedy rigid, iterations [8,4,2]",
        "interpretation": "Synthetic API exercise only; not patient motion or high-resolution information validation",
    }


def usd_smoke() -> dict:
    base = pv.Sphere(radius=4.0, theta_resolution=12, phi_resolution=8)
    offsets = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 1.5)]
    meshes = [base.translate(offset, inplace=False) for offset in offsets]
    workflow = WorkflowConvertVTKToUSD(
        input_meshes=meshes,
        usd_project_name="mp08_09_synthetic_motion.usda",
        output_directory=ROOT,
        separate_by_connectivity=False,
        extract_surface=False,
        static_merge=False,
        time_codes=[0.0, 1.0, 2.0],
        frames_per_second=2.0,
        appearance="solid",
        solid_color=(0.2, 0.7, 0.9),
        log_level=logging.ERROR,
    )
    output = Path(workflow.process()["usd_file"])
    stage = Usd.Stage.Open(str(output))
    if stage is None:
        raise RuntimeError("OpenUSD could not reopen the synthetic scene")
    mesh_prims = [UsdGeom.Mesh(prim) for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    if len(mesh_prims) != 1:
        raise RuntimeError(f"Expected one animated Mesh, got {len(mesh_prims)}")
    points_attr = mesh_prims[0].GetPointsAttr()
    codes = list(points_attr.GetTimeSamples())
    centroids = [np.asarray(points_attr.Get(t), dtype=float).mean(axis=0) for t in codes]
    displacement = np.asarray(centroids) - centroids[0]
    # MONAI's VTK converter writes metre coordinates in a Y-up USD stage.
    expected = np.array([(x, z, -y) for x, y, z in offsets]) * 0.001
    readback_matches = bool(np.allclose(displacement, expected, atol=5e-6))
    point_counts = [len(points_attr.Get(t)) for t in codes]
    if (codes != [0.0, 1.0, 2.0] or not readback_matches
            or len(set(point_counts)) != 1
            or not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0)
            or UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.y):
        raise RuntimeError("Synthetic USD time samples or coordinate conversion disagree")
    return {
        "status": "completed_synthetic_usd_smoke",
        "usd_file": output.name,
        "mesh_prim_count": len(mesh_prims),
        "time_codes": codes,
        "frames_per_second": stage.GetFramesPerSecond(),
        "time_codes_per_second": stage.GetTimeCodesPerSecond(),
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "up_axis": UsdGeom.GetStageUpAxis(stage),
        "points_per_frame": point_counts,
        "centroid_displacements_usd_xyz_m": np.round(displacement, 6).tolist(),
        "known_offsets_vtk_xyz_mm": [list(x) for x in offsets],
        "coordinate_readback_matches_known_offsets": readback_matches,
        "coordinate_conversion_observed": "VTK xyz millimetres -> USD (x,z,-y) metres, Y-up",
        "interpretation": "Synthetic USD readback verifies file/time samples/scale only; no patient geometry or Omniverse rendering",
    }


def main() -> None:
    report = {"schema": "mp08_09_synthetic_smoke_v1", "monai_physio_version": version("monai-physio")}
    for name, func in [("mp08", reconstruction_smoke), ("mp09", usd_smoke)]:
        try:
            report[name] = func()
        except Exception as exc:
            report[name] = {"status": "blocked", "exception_type": type(exc).__name__,
                            "message": str(exc).splitlines()[0][:300]}
    output = ROOT / "mp08_09_synthetic_results.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
