"""Synthetic geometry and signed scenario checks; never start MATLAB/FEM."""
import copy
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pytest
from scipy.io import loadmat, savemat
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MATLAB_TRKG4_real_subjects/tools"))
import scorpion_fem_heart_pilot as pilot


def test_rule_matches_canonical_ct_prefix_exactly():
    cuts = np.sort(qmc.Sobol(3, scramble=True, seed=20261008).random_base2(12), axis=1)
    expected = np.diff(np.c_[np.zeros(4096), cuts, np.ones(4096)], axis=1)
    np.testing.assert_array_equal(pilot.nested_barycentric(), expected)
    assert np.all(expected >= 0)
    np.testing.assert_allclose(expected.sum(1), 1, atol=1e-15)


def test_sphere_uses_entire_voxel_mask_volume_and_lps_affine():
    mask = np.zeros((8, 7, 6), bool)
    mask[1:3, 2:5, 1:3] = True
    affine = np.diag([2., 3., 4., 1.])
    affine[:3, 3] = [10, -5, 20]
    s = pilot.mask_sphere(mask, affine)
    assert s["volume_ml"] == pytest.approx(12*24/1000)
    np.testing.assert_allclose(s["centre_lps_mm"], [-13, -4, 26])
    assert 4*np.pi*s["radius_mm"]**3/3 == pytest.approx(12*24)


def test_background_replaces_only_old_heart_and_is_immutable():
    f = np.eye(5)
    f.flags.writeable = False
    bg = pilot.frozen_background(f)
    np.testing.assert_array_equal(bg[[1, 2, 4]], f[[1, 2, 4]])
    np.testing.assert_array_equal(bg[3], [1, 0, 0, 0, 0])
    np.testing.assert_array_equal(f, np.eye(5))
    with pytest.raises(ValueError, match="five-material"):
        pilot.frozen_background(np.ones((2, 4))/4)


def test_joint_overlay_preserves_spatial_overlap_and_restores_background():
    # A voxel plane separates bone from lung. A small sphere covers the bone
    # side preferentially: multiplying independent tissue/sphere fractions fails.
    labels = np.ones((20, 20, 20), np.uint8)
    labels[:10] = 2
    labels[4:6, 4:6, 4:6] = 3
    affine = np.eye(4)
    vertices = np.array([[[-2., -2, 2], [-14, -2, 2], [-2, -14, 2], [-2, -2, 14]]])
    centre = np.array([-5., -5, 5.])
    bary = pilot.nested_barycentric()
    new, original, overlap, _ = pilot.sample_overlay(vertices, labels, affine, centre, 4., bary)
    assert new[4096][0, 3] > 0
    for q in pilot.LEVELS:
        np.testing.assert_array_equal(new[q].sum(1), [1])
        assert overlap[q][0, 2] > overlap[q][0, 1]
    # The sphere has priority over bone; this is allowed and counted.
    assert overlap[4096][0, 2] > 0
    product = pilot.frozen_background(original[4096]) * (1-new[4096][:, 3, None])
    assert not np.allclose(product[:, 2], new[4096][:, 2])
    far, _, _, _ = pilot.sample_overlay(vertices, labels, affine, centre+100, 1., bary)
    np.testing.assert_array_equal(far[4096], pilot.frozen_background(original[4096]))
    covered, _, _, _ = pilot.sample_overlay(vertices, labels, affine, centre, 100., bary)
    np.testing.assert_array_equal(covered[4096], [[0, 0, 0, 1, 0]])


def test_tiny_intersection_is_not_discarded_by_a_centroid_filter():
    vertices = np.array([[[0., 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]]])
    assert pilot.possible_sphere_tets(vertices, np.full(3, .1), .05)[0]
    assert not pilot.possible_sphere_tets(vertices, np.full(3, 50), .05)[0]


def fixture_forward():
    j = np.zeros((4, 4, 4))
    j[1, 0] = [1, -2, 3, -4]
    j[3, 2] = [-2, 1, -1, 2]
    rhoi = np.array([2., 8, 15, 1.5, 20])
    rhoe = rhoi * [1.01, .9, 1, 1, .99]
    t = np.eye(4)
    te = t + np.einsum("ijk,k->ij", j, np.log(rhoe[pilot.ACTIVE]/rhoi[pilot.ACTIVE]))
    a = dict(case_id="L140_TT2x2", state="inhale", rule=4096,
             rho_ohm_m=rhoi.tolist(), transfer_matrix_ohm=t.tolist(), jacobian_logrho=j.tolist())
    b = dict(a, state="exhale", rho_ohm_m=rhoe.tolist(), transfer_matrix_ohm=te.tolist())
    return dict(subject_alias="synthetic", frequency_hz=100000, FEM_was_run=True,
        derivative_checks=dict(passed=True, active_material_indices=[1, 2, 4, 5]),
        system_checks=[dict(system_relative_residual=1e-12, reciprocity_ohm=1e-12)],
        answers=[a, b], integration_passed=False)


def fixture_scenarios():
    return dict(subject_alias="synthetic", amplitude_status="scenario_not_measured",
                pulses=[dict(id="signed", delta_logrho=[.01, -.02, .03, -.04])])


def test_signed_material_predictions_and_conditional_breath_contrast():
    result = pilot.predict(fixture_forward(), fixture_scenarios())
    row = result["rows"][0]
    assert row["pulses"][0]["delta_transfer_ohm"][1][0] == pytest.approx(.30)
    np.testing.assert_allclose(row["nonlinear_minus_linear_ohm"], 0, atol=1e-15)
    components = np.asarray(row["linear_components_ohm"])
    np.testing.assert_allclose(components.sum(2), row["linear_delta_ohm"])
    np.testing.assert_allclose(row["linear_background_only_ohm"], components[:, :, [0, 3]].sum(2))
    assert result["baseline_integration_passed"] is False
    assert result["quantitative_result_accepted"] is False
    assert result["stroke_volume_estimated"] is False


@pytest.mark.parametrize("mutation,match", [
    (lambda f: f.update(subject_alias="another_subject"), "Cross-subject"),
    (lambda f: f["derivative_checks"].update(passed=False), "not checked"),
    (lambda f: f["derivative_checks"].update(active_material_indices=[1, 2, 3, 4]), "order"),
    (lambda f: f["system_checks"][0].update(system_relative_residual=1), "residual"),
    (lambda f: f["answers"][1]["rho_ohm_m"].__setitem__(2, 16), "Bone"),
    (lambda f: f["answers"].append(copy.deepcopy(f["answers"][0])), "Duplicate"),
])
def test_prediction_refuses_bad_baseline(mutation, match):
    f = fixture_forward()
    mutation(f)
    with pytest.raises(ValueError, match=match):
        pilot.predict(f, fixture_scenarios())


def test_no_pair_is_explicit_and_amplitudes_cannot_be_called_measured():
    f = fixture_forward()
    f["answers"].pop()
    result = pilot.predict(f, fixture_scenarios())
    assert not result["rows"] and len(result["skipped"]) == 1
    s = fixture_scenarios()
    s["amplitude_status"] = "measured"
    with pytest.raises(ValueError, match="scenario_not_measured"):
        pilot.predict(f, s)


def synthetic_preparation(tmp_path):
    cfg = pilot.core.read(ROOT / "configs/ct_fem_preparation.example.json")
    shape = (12, 12, 12)
    affine = np.eye(4); affine[:3, 3] = -5.5
    for name in ("ct", "lung", "bone", "heart", "fat"):
        data = np.zeros(shape, np.int16 if name == "ct" else np.uint8)
        if name == "heart":
            data[5:7, 5:7, 5:7] = 1
        elif name == "lung":
            data[:3] = 1
        elif name == "bone":
            data[3:4] = 1
        elif name == "fat":
            data[-2:] = 1
        image = nib.Nifti1Image(data, affine); image.header.set_xyzt_units("mm")
        nib.save(image, tmp_path / (name + ".nii.gz"))
    cfg["subject_alias"] = "nik"
    cfg["inputs"].update(ct="ct.nii.gz", ct_provenance={"kind": "synthetic"},
        masks={k: k+".nii.gz" for k in ("lung", "bone", "heart", "fat")}, mask_provenance={"kind": "synthetic"})
    cfg["body"].update(strategy="HU_largest_component_per_slice", closing_iterations=1, closing_connectivity=1,
        surface_level=.49, maximum_changed_relative=.001, maximum_outside_body_voxels={k: 0 for k in ("lung", "bone", "heart", "fat")})
    cfg["electrical_context"]["frequency_hz"] = 100000
    geometry_config = tmp_path / "geometry.json"
    pilot.write(geometry_config, cfg)
    canonical = pilot.core.load_config(geometry_config)
    ct, _, _, labels, _ = pilot.core.load_inputs(canonical)
    nodes = np.array([[-4., -4, -4], [4, -4, -4], [4, 4, -4], [-4, 4, -4],
                      [-4, -4, 4], [4, -4, 4], [4, 4, 4], [-4, 4, 4]])
    tets = np.array([[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6],
                     [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]], np.int32)
    k = pilot.core.kernels()
    tets = k["orient"](nodes, tets)
    fractions, outside, inventory = pilot.core.integrate_materials(nodes, tets, labels, ct.affine)
    materials = tmp_path / "materials"; materials.mkdir()
    savemat(materials / "mesh.mat", dict(nodes_mm=nodes, elems=tets+1, boundary=k["boundary_faces"](tets)+1))
    pilot.core.save_fractions(materials, fractions, outside, inventory, pilot.core.material_identity(canonical))
    products = {p.name: pilot.sha(p) for p in materials.iterdir()}
    pilot.write(materials / "preparation.local.json", dict(schema="ct_fem_preparation_artifact_v1",
        status="prepared_not_validated", stage="materials", subject_alias="nik", products=products, sources={}))
    pilot.write(tmp_path / "reference.json", {"synthetic_fixture": True})
    config = dict(schema="scorpion_heart_pilot_config_v1", subject_alias="nik", geometry_config="geometry.json",
        materials_dir="materials", cem_mat="not_created_yet.mat", baseline_forward="not_created_yet.json",
        baseline_prepared="not_created_yet_prepared.mat", reference_materials="reference.json", case_ids=["L140_TT2x2"])
    pilot.write(tmp_path / "pilot.json", config)
    return tmp_path / "pilot.json"


def test_end_to_end_own_ct_materials_no_fem(tmp_path):
    config = synthetic_preparation(tmp_path)
    output = tmp_path / "pilot"
    plan = pilot.prepare(config, output, chunk=2)
    assert len(plan["variants"]) == 8 and not plan["FEM_was_run"]
    assert plan["geometry"]["sphere"]["volume_ml"] == pytest.approx(.008)
    assert plan["geometry"]["boundary_clearance_mm"] > 0
    assert not plan["geometry"]["centroid_filter_used"]
    for variant in plan["variants"]:
        data = loadmat(variant["file"])
        np.testing.assert_array_equal(data["fractions"].sum(1), np.ones(6))
        np.testing.assert_array_equal(data["element_ids"].ravel(), np.arange(1, 7))
    for q in pilot.LEVELS:
        original = loadmat(tmp_path / "materials" / f"fractions_q{q}.mat")["fractions"]
        copied = loadmat(output / f"wholeheart_ct_q{q}.mat")["fractions"]
        np.testing.assert_array_equal(copied, original)
        v = [loadmat(output / f"sphere_v{suffix}_q{q}.mat")["fractions"][:, 3] for suffix in ("095", "100", "105")]
        assert np.all(v[0] <= v[1]) and np.all(v[1] <= v[2])
    result = pilot.summarize(output)
    assert result["status"] == "incomplete" and len(result["incomplete_or_failed"]) == 8
    with pytest.raises(ValueError, match="fresh"):
        pilot.prepare(config, output)


def test_real_pilot_has_no_second_physical_solver():
    source = (ROOT / "MATLAB_TRKG4_real_subjects/src/run_scorpion_heart_pilot.m").read_text()
    for kernel in ("system_mat_fields", "trkg4_scale_fmdl_to_si", "trkg4_lateral_cem_response", "calc_system_mat"):
        assert kernel in source
    assert "decomposition(" not in source and "chol(" not in source
