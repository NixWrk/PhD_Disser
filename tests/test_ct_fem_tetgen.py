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
    assert qc["relative_volume_error"] <= cfg["mesh"]["maximum_relative_volume_error"]
    assert max(x["max_mm"] for x in qc["boundary_distances"].values()) <= cfg["mesh"]["maximum_sampled_surface_error_mm"]
    assert result["completed_geometry_settings"]["mesh_generation"]["size_mm"] == 5.
    assert not result["FEM_was_run"] and not result["quantitative_result_accepted"]
    assert core.sha(mesh / "contacts.local.json") == core.sha(surface / "contacts.local.json")
    assert core.verify_artifact(mesh)["stage"] == "mesh"
