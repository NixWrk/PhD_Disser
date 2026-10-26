"""Small optional integration check of the actual TetGen preparation backend.

Uses only synthetic CT and masks. No bulk FEM or inverse solve is executed.
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ct_fem_core as core


def test_actual_tetgen_boundary_and_material_provenance(tmp_path):
    pytest.importorskip("tetgen")
    cfg = core.read(ROOT / "configs/ct_fem_preparation.example.json")
    cfg["subject_alias"] = "synthetic_TetGen_integration"
    cfg["inputs"]["ct_provenance"] = {"kind": "synthetic_cube", "breath_state": "not_applicable"}
    cfg["inputs"]["mask_provenance"] = {"method": "synthetic_disjoint_boxes", "independently_validated": False}
    affine = np.eye(4); affine[:3, 3] = -12
    hu = np.full((24, 24, 24), -1000, np.int16); hu[4:20, 4:20, 4:20] = 0
    for name in ("static_ct", "lung", "bone", "heart", "fat"):
        data = hu if name == "static_ct" else np.zeros(hu.shape, np.uint8)
        if name != "static_ct":
            x = 6 + 2 * ("lung", "bone", "heart", "fat").index(name)
            data[x:x+2, 8:10, 8:10] = 1
        image = nib.Nifti1Image(data, affine); image.header.set_xyzt_units("mm")
        path = tmp_path / (name + ".nii.gz"); nib.save(image, path)
        if name == "static_ct": cfg["inputs"]["ct"] = str(path)
        else: cfg["inputs"]["masks"][name] = str(path)
    cfg["contacts"]["physical_centres_lps_mm"] = {
        name: dict(raw_lps_mm=[8.51, y, z], nominal_area_mm2=2., provenance="synthetic face")
        for name, (y, z) in zip(("I_plus", "V_plus", "V_minus", "I_minus"),
                              ((-3, -3), (3, -3), (-3, 3), (3, 3)))}
    cfg["surface_qc"]["sample_count"] = 256
    cfg["mesh"]["size_mm"] = 5.
    cfg["mesh"]["steiner_limit"] = 10000
    config = tmp_path / "config.local.json"; core.write_new(config, cfg)
    surface = tmp_path / "surface"; mesh = tmp_path / "mesh"
    core.prepare("surface", config, surface)
    result = core.prepare("mesh", config, mesh, surface)
    qc = core.read(mesh / "mesh_quality.local.json")
    assert qc["actual_boundary_checked"] and qc["tetgen_version"]
    assert qc["actual_surface_qc"]["passed"] and qc["cumulative_geometry_checked"]
    assert qc["relative_volume_error"] <= cfg["mesh"]["maximum_relative_volume_error"]
    assert max(x["max_mm"] for x in qc["boundary_distances"].values()) <= cfg["mesh"]["maximum_sampled_surface_error_mm"]
    assert result["completed_geometry_settings"]["mesh_generation"]["size_mm"] == 5.
    assert not result["FEM_was_run"] and not result["quantitative_result_accepted"]
    assert core.sha(mesh / "contacts.local.json") == core.sha(surface / "contacts.local.json")
    assert core.verify_artifact(mesh)["stage"] == "mesh"


def test_actual_tiny_one_command_pipeline(tmp_path):
    pytest.importorskip("tetgen")
    from test_ct_fem_preparation import fixture_config
    path, cfg = fixture_config(tmp_path)
    cfg["mesh"]["size_mm"] = 5.
    cfg["mesh"]["steiner_limit"] = 10000
    # A closed planar fixture avoids testing the separate HU corner-
    # approximation error with a tolerance chosen for full-sized anatomy.
    import trimesh
    solid = trimesh.creation.box(extents=[30., 30., 30.])
    for _ in range(3):
        solid = solid.subdivide()
    source = tmp_path / "supplied_box.stl"; solid.export(source)
    cfg["body"] = dict(strategy="author_STL_LPS", source_stl=str(source),
        source_stl_sha256=core.sha(source), units="mm", frame="LPS",
        provenance="Synthetic supplied planar solid; no anatomical claim",
        maximum_outside_body_voxels={k: 0 for k in ("lung", "bone", "heart", "fat")},
        coverage_limits_basis="All synthetic tissue is enclosed by the supplied solid")
    cfg["surface_qc"]["manual_STL_export_maximum_sampled_distance_mm"] = .001
    for row in cfg["contacts"]["physical_centres_lps_mm"].values():
        row["raw_lps_mm"][0] = 15.1
    core.write_new(tmp_path / "pipeline_config.json", cfg)
    out = tmp_path / "pipeline"
    result = core.prepare_mesh_pipeline(tmp_path / "pipeline_config.json", out)
    assert result["geometry_and_shape_QC_passed"]
    assert [stage["stage"] for stage in result["completed"]] == ["surface", "remesh", "mesh"]
    assert not result["FEM_was_run"] and not result["CEM_contacts_built"]
    assert core.verify_artifact(out / "mesh")["stage"] == "mesh"
    assert core.read(out / "mesh/mesh_quality.local.json")["surface_recovery_rounds"] == 0
