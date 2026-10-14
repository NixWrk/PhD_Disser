"""Small numerical/geometry checks only; no FEM and no TetGen invocation."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest
from scipy.io import loadmat
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ct_fem_core as core


def write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def fixture_config(tmp_path):
    cfg = core.read(ROOT / "configs/ct_fem_preparation.example.json")
    shape = (24, 24, 24)
    affine = np.eye(4); affine[:3, 3] = -12
    hu = np.full(shape, -1000, np.int16); hu[4:20, 4:20, 4:20] = 0
    for name in ["static_ct", "lung", "bone", "heart", "fat"]:
        data = hu if name == "static_ct" else np.zeros(shape, np.uint8)
        if name == "fat":
            data[5:8, 5:8, 5:8] = 1
        image = nib.Nifti1Image(data, affine); image.header.set_xyzt_units("mm")
        nib.save(image, tmp_path / (name + ".nii.gz"))
    cfg["inputs"]["ct"] = "static_ct.nii.gz"
    cfg["inputs"]["ct_provenance"] = {"kind": "synthetic", "breath_state": "not_applicable"}
    cfg["inputs"]["masks"] = {n: n+".nii.gz" for n in ["lung", "bone", "heart", "fat"]}
    cfg["inputs"]["mask_provenance"] = {"method": "synthetic", "independently_validated": False}
    cfg["surface_qc"]["sample_count"] = 64
    cfg["surface_qc"]["topology_correction"]["maximum_sampled_distance_mm"] = 2
    cfg["contacts"]["physical_centres_lps_mm"] = {
        name: dict(raw_lps_mm=[8.51, y, z], nominal_area_mm2=2., provenance="synthetic")
        for name, (y, z) in zip(["I_plus", "V_plus", "V_minus", "I_minus"], [(0, 0), (2, 0), (0, 2), (2, 2)])}
    cfg["refinement"]["guides_lps_mm"] = [[-3, -3, 3]]
    cfg["refinement"]["radius_mm"] = .2
    path = tmp_path / "config.json"; write(path, cfg)
    return path, cfg


def tiny_mesh():
    nodes = np.array([[2, 2, 2], [4, 2, 2], [2, 4, 2], [2, 2, 4], [4, 4, 4]], float)
    nodes *= [-1, -1, 1]
    kernel = core.kernels()
    tets = kernel["orient"](nodes, np.array([[0, 1, 2, 3], [4, 1, 3, 2]], np.int32))
    return nodes, tets, kernel["boundary_faces"](tets)


def test_empty_slices_and_air_do_not_turn_into_body():
    hu = np.full((20, 20, 7), -1000)
    hu[4:16, 4:16, 2:5] = 0
    mask, _ = core.kernels()["body_mask"](hu)
    assert not mask[:, :, [0, 1, 5, 6]].any()
    assert mask[:, :, 3].any()
    with pytest.raises(ValueError, match="no above-threshold"):
        core.kernels()["body_mask"](np.full((8, 8, 8), -1000))


def test_affine_mismatch_and_unknown_units_are_rejected(tmp_path):
    path, _ = fixture_config(tmp_path)
    cfg = core.load_config(path)
    file = Path(cfg["inputs"]["masks"]["lung"]); old = nib.load(file)
    affine = old.affine.copy(); affine[0, 3] += .001
    image = nib.Nifti1Image(np.zeros(old.shape, np.uint8), affine); image.header.set_xyzt_units("mm")
    nib.save(image, file)
    with pytest.raises(ValueError, match="affine mismatch"):
        core.load_inputs(cfg)
    affine = nib.load(cfg["inputs"]["ct"]).affine
    nib.save(nib.Nifti1Image(np.zeros(old.shape, np.uint8), affine), file)
    with pytest.raises(ValueError, match="Mask units"):
        core.load_inputs(cfg)


def test_rotated_anisotropic_ras_lps_roundtrip_and_surface():
    affine = np.array([[0, -2, 0, 10], [3, 0, 0, -20], [0, 0, 4, 30], [0, 0, 0, 1.]])
    vox = np.array([[2, 3, 4], [1, 2, 3.5]])
    ras = nib.affines.apply_affine(affine, vox)
    lps = ras * [-1, -1, 1]
    np.testing.assert_allclose(core.lps_to_voxel(lps, affine), vox, atol=1e-12)
    mask = np.zeros((7, 7, 7), bool); mask[2:5, 2:5, 2:5] = True
    mesh = core.kernels()["surface_from_mask"](mask, affine, level=.49)
    physical_vertices = core.lps_to_voxel(mesh.vertices, affine)
    np.testing.assert_allclose(physical_vertices.min(0), [1.49]*3, atol=1e-6)
    np.testing.assert_allclose(physical_vertices.max(0), [4.51]*3, atol=1e-6)
    assert mesh.is_volume


def test_material_priority_is_explicit(tmp_path):
    path, _ = fixture_config(tmp_path); cfg = core.load_config(path)
    for n in ["lung", "bone", "heart", "fat"]:
        image = nib.load(cfg["inputs"]["masks"][n]); data = np.zeros(image.shape, np.uint8)
        data[10, 10, 10] = 1
        fresh = nib.Nifti1Image(data, image.affine); fresh.header.set_xyzt_units("mm")
        nib.save(fresh, cfg["inputs"]["masks"][n])
    _, _, _, labels, report = core.load_inputs(cfg)
    assert labels[10, 10, 10] == 1 and report["overlap_voxels"] == 1


@pytest.mark.parametrize("material", range(5))
def test_known_homogeneous_materials_and_outside_ct(material):
    nodes, tets, _ = tiny_mesh()
    labels = np.full((8, 8, 8), material, np.uint8)
    f, outside, _ = core.integrate_materials(nodes, tets, labels, np.eye(4))
    for q in core.POINTS:
        np.testing.assert_array_equal(f[q], np.tile(np.eye(5)[material], (2, 1)))
        np.testing.assert_array_equal(outside[q], [0, 0])
    f, outside, report = core.integrate_materials(nodes+[-100, -100, 100], tets, labels, np.eye(4))
    for q in core.POINTS:
        np.testing.assert_array_equal(f[q][:, 0], [1, 1])
        np.testing.assert_array_equal(outside[q], [1, 1])
    assert report["potentially_outside_CT_cells"] == 2


def test_unchanged_parent_fraction_reuse_and_nested_energy():
    nodes, tets, boundary = tiny_mesh(); kernel = core.kernels()
    labels = np.zeros((8, 8, 8), np.uint8); labels[:3] = 1; labels[3:] = 4
    pf, po, _ = core.integrate_materials(nodes, tets, labels, np.eye(4))
    result = kernel["refine_local"](nodes, tets, boundary, np.array([False, False]))
    fn, ft, _, ep, _, _, _, _ = result
    f, _, report = core.integrate_materials(fn, ft, labels, np.eye(4), parent=(nodes, tets, ep, pf, po))
    assert report["unchanged_parent_elements_reused"] == 2
    for q in core.POINTS:
        np.testing.assert_array_equal(f[q], pf[q])
    result = kernel["refine_local"](nodes, tets, boundary, np.array([True, False]))
    fn, ft, fb, ep, bp, edges, cp, weights = result
    core.mesh_qc(fn, ft, fb)
    nested = kernel["verify_nested_geometry"](nodes, tets, boundary, result,
                                             dict(nodes_mm=nodes, boundary=boundary+1), bp+1)
    energy = kernel["inherited_field_energy_check"](nodes, tets, fn, ft, edges, ep, np.array([.2, .7]), cp, weights)
    assert energy["passed"] and nested["original_nodes_unchanged"]
    np.testing.assert_array_equal(pf[4096][ep], np.take(pf[4096], ep, axis=0))
    changed = fn.copy(); changed[0, 0] += .1
    with pytest.raises(AssertionError):
        core.integrate_materials(changed, ft, labels, np.eye(4), parent=(nodes, tets, ep, pf, po))


def test_circle_full_section_matches_immutable_37cm_original():
    source = core.LEGACY / "scorpion_georg_geometry_revision.py"
    sys.path.insert(0, str(core.LEGACY))
    spec = importlib.util.spec_from_file_location("historical_join_for_test", source)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    surface = trimesh.creation.box(extents=[240, 160, 160])
    old_arm = dict(direction="negative_axis", center_y=20., center_z=0., cut_coordinate=-120.)
    reference = old.embedded_join(surface, old_arm, inward_grid=np.arange(.5, 3., 1.), clearance_mm=2.)
    arm = dict(axis="x", direction=-1, center_yz_mm=[20., 0.], cut_x_mm=-120.,
               radius_mm=370/(2*np.pi), clearance_mm=2., inset_search_mm=[.5, 2.5, 1.])
    actual = core.full_section_join(surface, arm)
    for key in ["x_mm", "centre_yz_mm", "score_mm2", "cross_section_covered_fraction", "candidate_count"]:
        np.testing.assert_allclose(actual[key], reference[key], rtol=0, atol=1e-12)
    arm["radius_mm"] = 100.
    with pytest.raises(ValueError, match="No full circular"):
        core.full_section_join(surface, arm)


def test_mesh_quality_and_node_limits_cannot_be_relaxed(tmp_path):
    nodes, tets, boundary = tiny_mesh()
    with pytest.raises(ValueError, match="shape/resource"):
        core.mesh_qc(nodes, tets, max_nodes=4)
    flat = nodes.copy(); flat[:, 2] *= 1e-7
    with pytest.raises(ValueError, match="shape/resource"):
        core.mesh_qc(flat, tets)
    wrong = boundary.copy(); wrong[0] = wrong[1]
    with pytest.raises(ValueError, match="boundary differs"):
        core.mesh_qc(nodes, tets, wrong)
    path, cfg = fixture_config(tmp_path)
    cfg["mesh"]["quality_limits"]["minimum_mean_ratio"] = .001
    write(path, cfg)
    with pytest.raises(ValueError, match="cannot be relaxed"):
        core.load_config(path)


def test_real_small_surface_cli_artifacts_and_integrity(tmp_path):
    path, _ = fixture_config(tmp_path); out = tmp_path / "surface"
    run = subprocess.run([sys.executable, "-B", str(ROOT / "tools/prepare_ct_fem.py"), "surface",
                          "--config", str(path), "--out", str(out)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    manifest = core.verify_artifact(out)
    assert manifest["stage"] == "surface" and not manifest["FEM_was_run"]
    contacts = core.read(out / "contacts.local.json")
    assert contacts["physical_contact_count"] == 4 and not contacts["CEM_adapter_ready"]
    assert contacts["projection_surface_sha256"] == core.sha(out / "exterior.stl")
    assert contacts["surface_face_is_not_FEM_boundary_id"]
    assert "raw_lps_mm" in contacts["physical_contacts"][0]
    assert "projected_lps_mm" in contacts["physical_contacts"][0]
    with pytest.raises(ValueError, match="fresh output"):
        core.prepare("surface", path, out)
    with (out / "exterior.stl").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        core.verify_artifact(out)


def mesh_artifact(tmp_path, path):
    cfg = core.load_config(path); source = tmp_path / "tiny_mesh"; source.mkdir()
    nodes, tets, boundary = tiny_mesh(); _, qc = core.mesh_qc(nodes, tets, boundary)
    core.save_mesh(source, nodes, tets, boundary, qc)
    # A distinct explicit contact reference: metadata only, not FEM patch QC.
    contact_reference = trimesh.creation.box(extents=[17.02, 20., 20.])
    core.write_new(source / "contacts.local.json", core.contact_metadata(contact_reference, cfg["contacts"]))
    core.publish(source, "mesh", cfg, {}, core.artifact_sources(cfg))
    return source


def test_real_tiny_materials_refinement_and_source_rejection(tmp_path):
    path, _ = fixture_config(tmp_path); source = mesh_artifact(tmp_path, path)
    materials = tmp_path / "materials"; core.prepare("materials", path, materials, source)
    refined = tmp_path / "refined"; core.prepare("refine", path, refined, materials)
    core.verify_artifact(refined)
    qc = core.read(refined / "refinement_qc.local.json")
    assert qc["energy_check"]["passed"] and not qc["FEM_was_run"]
    inherited = loadmat(refined / "inherited/fractions_q4096.mat")["fractions"]
    parent = loadmat(materials / "fractions_q4096.mat")["fractions"]
    ep = loadmat(refined / "parent_maps.mat")["element_parent"].ravel()-1
    np.testing.assert_array_equal(inherited, parent[ep])
    a = loadmat(refined / "inherited/mesh.mat"); b = loadmat(refined / "remapped/mesh.mat")
    assert core.sha(refined / "inherited/mesh.mat") == core.sha(refined / "remapped/mesh.mat")
    assert core.sha(source / "mesh.mat") == core.sha(materials / "mesh.mat")
    for k in ["nodes_mm", "elems", "boundary"]:
        np.testing.assert_array_equal(a[k], b[k])
    core.verify_artifact(refined / "remapped")
    again = tmp_path / "refined_again"
    core.prepare("refine", path, again, refined / "remapped")
    assert core.read(again / "refinement_qc.local.json")["energy_check"]["passed"]
    with pytest.raises(ValueError, match="supported parent"):
        core.prepare("refine", path, tmp_path / "invalid_parent", refined / "inherited")
    with Path(core.load_config(path)["inputs"]["ct"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="source changed"):
        core.prepare("refine", path, tmp_path / "rejected", materials)
    assert not (tmp_path / "rejected").exists()


def test_contacts_require_explicit_order_and_unique_membership():
    surface = trimesh.creation.box(extents=[10, 10, 10])
    spec = dict(maximum_projection_mm=1., physical_centres_lps_mm={
        "a": dict(raw_lps_mm=[5, 0, 0], nominal_area_mm2=1., provenance="synthetic"),
        "b": dict(raw_lps_mm=[5, 2, 0], nominal_area_mm2=1., provenance="synthetic")},
        scenarios=[dict(id="s", groups={"g1": ["a"], "g2": ["b"]}, drive_groups=["g1", "g2"],
                        measure_groups=["g2", "g1"], inactive_contact_model="individual_floating_CEM_assumption")])
    result = core.contact_metadata(surface, spec)
    assert result["scenarios"][0]["measure_groups"] == ["g2", "g1"]
    spec["scenarios"][0]["groups"]["g2"] = ["a"]
    with pytest.raises(ValueError, match="membership"):
        core.contact_metadata(surface, spec)


def test_tiny_cylinder_union_checks_full_section_and_yz_extents(tmp_path):
    path, cfg = fixture_config(tmp_path)
    cfg["synthetic_cylinders"] = [dict(axis="x", direction=1, cut_x_mm=8.51,
        center_yz_mm=[0., 0.], radius_mm=2., clearance_mm=.1, inset_search_mm=[1., 3., 1.],
        outward_length_mm=3., inner_overlap_mm=.5, segments=64)]
    for name, (y, z) in zip(cfg["contacts"]["physical_centres_lps_mm"], [(0, 0), (.2, 0), (0, .2), (.2, .2)]):
        cfg["contacts"]["physical_centres_lps_mm"][name]["raw_lps_mm"] = [11.51, y, z]
    write(path, cfg)
    out = tmp_path / "cylinders"; core.prepare("surface", path, out)
    qc = core.read(out / "surface_qc.local.json")
    assert qc["synthetic_cylinders"][0]["cross_section_covered_fraction"] > 1-1e-10
    assert not qc["pre_union_decimation"] and qc["self_intersecting_faces"] == 0
    skin = trimesh.load(out / "skin_regularized.stl", force="mesh")
    union = trimesh.load(out / "exterior.stl", force="mesh")
    np.testing.assert_allclose(union.bounds[:, 1:], skin.bounds[:, 1:], atol=.02)
    assert union.bounds[1, 0] > skin.bounds[1, 0]+2


def test_mixed_material_remap_changes_split_elements_and_reuses_only_untouched():
    nodes, tets, boundary = tiny_mesh(); kernel = core.kernels()
    # A third independent tetrahedron tests actual untouched-element reuse.
    extra = np.array([[-5, -2, 2], [-6, -2, 2], [-5, -3, 2], [-5, -2, 3]], float)
    nodes = np.vstack([nodes, extra])
    tets = kernel["orient"](nodes, np.vstack([tets, np.array([[5, 6, 7, 8]], np.int32)]))
    boundary = kernel["boundary_faces"](tets)
    labels = np.zeros((9, 9, 9), np.uint8); labels[:3] = 1; labels[3:] = 4
    pf, po, _ = core.integrate_materials(nodes, tets, labels, np.eye(4))
    result = kernel["refine_local"](nodes, tets, boundary, np.array([True, False, False]))
    fn, ft, _, ep, *_ = result
    mapped, _, report = core.integrate_materials(fn, ft, labels, np.eye(4), parent=(nodes, tets, ep, pf, po))
    assert report["unchanged_parent_elements_reused"] == 1
    for q in core.POINTS:
        np.testing.assert_array_equal(mapped[q][ep == 2], pf[q][[2]])
        assert np.any(mapped[q][ep != 2] != pf[q][ep[ep != 2]])


def test_subject_and_ct_identity_cannot_silently_change(tmp_path):
    path, cfg = fixture_config(tmp_path); source = mesh_artifact(tmp_path, path)
    other = tmp_path / "other.json"
    cfg["subject_alias"] = "other_subject"; write(other, cfg)
    with pytest.raises(ValueError, match="subject alias"):
        core.prepare("materials", other, tmp_path / "wrong_subject", source)
    cfg["subject_alias"] = core.read(path)["subject_alias"]
    image = nib.load(tmp_path / "static_ct.nii.gz"); data = np.asanyarray(image.dataobj).copy()
    data[10, 10, 10] = 7
    alternate = nib.Nifti1Image(data, image.affine); alternate.header.set_xyzt_units("mm")
    nib.save(alternate, tmp_path / "another_ct.nii.gz")
    cfg["inputs"]["ct"] = "another_ct.nii.gz"; write(other, cfg)
    with pytest.raises(ValueError, match="CT identity differ"):
        core.prepare("materials", other, tmp_path / "wrong_ct", source)
    assert not (tmp_path / "wrong_subject").exists() and not (tmp_path / "wrong_ct").exists()


def test_change_during_array_loading_leaves_no_completed_artifact(tmp_path, monkeypatch):
    path, _ = fixture_config(tmp_path); original = core.load_inputs
    def changed(cfg):
        result = original(cfg)
        with Path(cfg["inputs"]["ct"]).open("ab") as stream:
            stream.write(b"changed_during_read")
        return result
    monkeypatch.setattr(core, "load_inputs", changed)
    out = tmp_path / "interrupted"
    with pytest.raises(ValueError, match="source changed"):
        core.prepare("validate", path, out)
    assert not (out / "preparation.local.json").exists()
    assert core.read(out / "preparation_failed.local.json")["status"] == "failed"


def test_validate_stage_serializes_header_and_preserves_unvalidated_status(tmp_path):
    path, _ = fixture_config(tmp_path); out = tmp_path / "validation"
    result = core.prepare("validate", path, out)
    assert result["stage"] == "validate" and result["status"] == "prepared_not_validated"
    assert core.read(out / "input_qc.local.json")["spacing_mm"] == [1., 1., 1.]
    assert core.verify_artifact(out)["ct_provenance"]["kind"] == "synthetic"


def test_serialized_boundary_order_preserved_across_refinement_levels(tmp_path):
    path, _ = fixture_config(tmp_path); cfg = core.load_config(path)
    source = tmp_path / "permuted_mesh"; source.mkdir()
    nodes, tets, boundary = tiny_mesh()
    boundary = np.roll(boundary[::-1], 2, axis=0)
    checked, qc = core.mesh_qc(nodes, tets, boundary)
    np.testing.assert_array_equal(checked, boundary)
    core.save_mesh(source, nodes, tets, boundary, qc)
    contact_reference = trimesh.creation.box(extents=[17.02, 20., 20.])
    core.write_new(source / "contacts.local.json", core.contact_metadata(contact_reference, cfg["contacts"]))
    core.publish(source, "mesh", cfg, {}, core.artifact_sources(cfg))
    materials = tmp_path / "material_parent"; core.prepare("materials", path, materials, source)
    parent = materials
    for level in range(2):
        out = tmp_path / f"refine_boundary_{level}"; core.prepare("refine", path, out, parent)
        saved_parent = loadmat(parent / "mesh.mat")
        saved_child = loadmat(out / "remapped/mesh.mat")
        maps = loadmat(out / "parent_maps.mat")
        check = core.kernels()["boundary_identity"](
            saved_parent["nodes_mm"], saved_parent["boundary"]-1,
            saved_child["nodes_mm"], saved_child["boundary"]-1,
            maps["boundary_parent"].ravel()-1)
        assert check["passed"]
        parent = out / "remapped"


def test_inward_boundary_orientation_is_rejected():
    nodes, tets, boundary = tiny_mesh()
    boundary = boundary.copy(); boundary[0] = boundary[0, ::-1]
    with pytest.raises(ValueError, match="orientation"):
        core.mesh_qc(nodes, tets, boundary)


def test_changed_parent_dependency_cannot_be_repinned_after_verification(tmp_path, monkeypatch):
    path, _ = fixture_config(tmp_path); source = mesh_artifact(tmp_path, path)
    original = core.verify_artifact
    def changed(directory):
        result = original(directory)
        with Path(result["configuration"]["inputs"]["masks"]["fat"]).open("ab") as stream:
            stream.write(b"changed_after_verification")
        return result
    monkeypatch.setattr(core, "verify_artifact", changed)
    out = tmp_path / "repin_rejected"
    with pytest.raises(ValueError, match="Parent dependency changed"):
        core.prepare("materials", path, out, source)
    assert not out.exists()


def test_refinement_branch_cannot_continue_when_whole_stage_failed(tmp_path, monkeypatch):
    path, _ = fixture_config(tmp_path); source = mesh_artifact(tmp_path, path)
    materials = tmp_path / "materials"; core.prepare("materials", path, materials, source)
    original = core.publish
    def interrupted(out, stage, *args):
        if stage == "refine":
            raise RuntimeError("interrupted_before_root_manifest")
        return original(out, stage, *args)
    monkeypatch.setattr(core, "publish", interrupted)
    out = tmp_path / "incomplete_refinement"
    with pytest.raises(RuntimeError, match="interrupted_before_root"):
        core.prepare("refine", path, out, materials)
    assert (out / "remapped/preparation.local.json").exists()
    assert core.read(out / "preparation_failed.local.json")["status"] == "failed"
    with pytest.raises(ValueError, match="no completed parent"):
        core.verify_artifact(out / "remapped")


def remesh_surface_artifact(tmp_path):
    path, cfg = fixture_config(tmp_path)
    cfg["remeshing"]["sample_count"] = 256
    cfg["remeshing"]["max_faces"] = 10000
    cfg["remeshing"]["max_vertices"] = 10000
    cfg["remeshing"]["selection"]["guides_lps_mm"] = [[-15., 0., 0.]]
    for name, (y, z) in zip(cfg["contacts"]["physical_centres_lps_mm"], [(0, 0), (3, 0), (0, 3), (3, 3)]):
        cfg["contacts"]["physical_centres_lps_mm"][name]["raw_lps_mm"] = [15.1, y, z]
    write(path, cfg); loaded = core.load_config(path)
    source = tmp_path / "synthetic_surface"; source.mkdir()
    surface = trimesh.creation.box(extents=[30., 30., 30.])
    for _ in range(3):
        surface = surface.subdivide()
    surface.export(source / "exterior.stl")
    surface = trimesh.load(source / "exterior.stl", force="mesh")
    qc = core.surface_qc(surface); qc.update(synthetic_cylinders=[], skin_bounds_mm=surface.bounds.tolist())
    core.write_new(source / "surface_qc.local.json", qc)
    contacts = core.contact_metadata(surface, loaded["contacts"])
    contacts.update(projection_surface_path=str(source / "exterior.stl"),
                    projection_surface_sha256=core.sha(source / "exterior.stl"))
    core.write_new(source / "contacts.local.json", contacts)
    sources = core.artifact_sources(loaded); sources[str(Path(__file__).resolve())] = core.sha(__file__)
    core.publish(source, "surface", loaded, {"fixture": "subdivided planar box, not a CT-derived validated anatomy"}, sources)
    return path, source


def test_remesh_region_selection_uses_explicit_spheres_and_aabbs():
    import ct_fem_remesh as remesh
    vertices = np.array([[-1, -1, 0], [1, -1, 0], [0, 2, 0],
                         [9, -1, 0], [11, -1, 0], [10, 2, 0]], float)
    surface = trimesh.Trimesh(vertices, [[0, 1, 2], [3, 4, 5]], process=False)
    np.testing.assert_array_equal(remesh.select_region(surface, [[0, 0, 0]], 1., []), [True, False])
    np.testing.assert_array_equal(remesh.select_region(surface, [], 1., [[[9, -1, -1], [11, 1, 1]]]), [False, True])


def test_remesh_rejects_vertex_smoothing_and_invalid_regions(tmp_path):
    import ct_fem_remesh as remesh
    path, cfg = fixture_config(tmp_path)
    cfg["remeshing"]["smooth_vertices"] = True
    with pytest.raises(ValueError, match="no vertex smoothing"):
        remesh.settings(cfg)
    cfg["remeshing"]["smooth_vertices"] = False
    cfg["remeshing"]["selection"]["buffer_aabbs_lps_mm"] = [[[1, 1, 1], [0, 0, 0]]]
    with pytest.raises(ValueError, match="AABBs"):
        remesh.settings(cfg)


def test_real_tiny_remesh_cli_preserves_shape_and_contact_provenance(tmp_path):
    path, source = remesh_surface_artifact(tmp_path); out = tmp_path / "remeshed"
    run = subprocess.run([sys.executable, "-B", str(ROOT / "tools/prepare_ct_fem.py"), "remesh",
        "--config", str(path), "--input", str(source), "--out", str(out)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout+run.stderr
    assert core.verify_artifact(out)["stage"] == "remesh"
    qc = core.read(out / "surface_qc.local.json")
    assert qc["passed"] and not qc["vertex_smoothing"] and not qc["automatic_repair"]
    assert [r["target_edge_mm"] for r in qc["operations"]] == [5., 2.8, 1.4]
    assert all(r["selected_faces"] > 0 for r in qc["operations"])
    assert qc["relative_volume_error"] <= .003
    assert max(d["max_mm"] for d in qc["sampled_distances"].values()) <= .75
    for op in qc["operations"]:
        assert max(d["max_mm"] for d in op["sampled_operation_distances"].values()) <= .2
    assert core.sha(out / "exterior.stl") == core.sha(out / "remesh_02_contact.stl")
    assert core.sha(out / "source_contacts.local.json") == core.sha(source / "contacts.local.json")
    before = core.read(source / "contacts.local.json")["physical_contacts"]
    after = core.read(out / "contacts.local.json")["physical_contacts"]
    for a, b in zip(before, after):
        assert b["raw_lps_mm"] == a["raw_lps_mm"]
        assert b["source_projected_lps_mm"] == a["projected_lps_mm"]
        assert b["original_projected_lps_mm"] == a["projected_lps_mm"]
        assert b["source_to_current_displacement_mm"] <= 1.
        assert b["displacement_mm"] <= 1.
    metadata = core.read(out / "contacts.local.json")
    assert metadata["original_projection_surface_sha256"] == core.sha(source / "exterior.stl")
    assert metadata["projection_surface_sha256"] == core.sha(out / "exterior.stl")
    assert not metadata["CEM_adapter_ready"] and not qc["canonical_CEM_patch_QC_performed"]
    import ct_fem_remesh as remesh
    unchanged_surface = trimesh.load(out / "exterior.stl", force="mesh")
    again = remesh.update_contacts(unchanged_surface, metadata, core.load_config(path), remesh.settings(core.load_config(path)))
    assert again["original_projection_surface_sha256"] == metadata["original_projection_surface_sha256"]
    assert again["physical_contacts"][0]["original_projected_lps_mm"] == after[0]["original_projected_lps_mm"]


def test_remesh_yz_bounds_and_operation_distance_are_enforced(tmp_path):
    import ct_fem_remesh as remesh
    surface = trimesh.creation.box(extents=[20., 20., 20.])
    moved = surface.copy(); moved.apply_translation([0., .1, 0.])
    with pytest.raises(ValueError, match="Y/Z bounds"):
        remesh.check_yz_bounds(moved, surface.bounds, .02)
    path, cfg = fixture_config(tmp_path); value = remesh.settings(cfg)
    moved = surface.copy(); moved.apply_translation([1., 0., 0.])
    with pytest.raises(ValueError, match="per-operation distance"):
        remesh.checked_export(moved, tmp_path / "too_far.stl", surface, value, core.kernels())


def test_mesh_accepts_completed_remesh_before_backend_without_running_tetgen(tmp_path, monkeypatch):
    path, source = remesh_surface_artifact(tmp_path)
    remeshed = tmp_path / "remesh"; core.prepare("remesh", path, remeshed, source)
    seen = []
    def stopped(cfg, src, out):
        seen.append(src)
        raise RuntimeError("test_stopped_before_Tetgen")
    monkeypatch.setattr(core, "build_mesh", stopped)
    for i, parent in enumerate([source, remeshed]):
        with pytest.raises(RuntimeError, match="test_stopped_before_Tetgen"):
            core.prepare("mesh", path, tmp_path / f"mesh_stopped_{i}", parent)
    assert seen == [source, remeshed]


@pytest.mark.parametrize("change", ["centre", "area", "readout", "group_order", "body", "coverage", "cylinders", "surface_qc", "mesh_generation"])
def test_continuation_rejects_unexecuted_changes_before_output(tmp_path, change):
    path, cfg = fixture_config(tmp_path); source = mesh_artifact(tmp_path, path)
    before = core.sha(source / "preparation.local.json")
    if change == "centre":
        cfg["contacts"]["physical_centres_lps_mm"]["I_plus"]["raw_lps_mm"][1] += .1
    elif change == "area":
        cfg["contacts"]["physical_centres_lps_mm"]["I_plus"]["nominal_area_mm2"] += .1
    elif change == "readout":
        cfg["contacts"]["scenarios"][0]["measure_groups"].reverse()
    elif change == "group_order":
        groups = cfg["contacts"]["scenarios"][0]["groups"]
        cfg["contacts"]["scenarios"][0]["groups"] = dict(reversed(list(groups.items()))) if isinstance(groups, dict) else list(reversed(groups))
    elif change == "body":
        cfg["body"]["hu_threshold"] -= 1
    elif change == "coverage":
        cfg["body"]["maximum_outside_body_voxels"]["lung"] = 1
    elif change == "cylinders":
        cfg["synthetic_cylinders"] = [{"unexecuted": True}]
    elif change == "surface_qc":
        cfg["surface_qc"]["sample_count"] += 1
    else:
        cfg["mesh"]["size_mm"] += 1
    changed = tmp_path / "changed.json"; write(changed, cfg)
    out = tmp_path / "rejected"
    with pytest.raises(ValueError, match="changed"):
        core.prepare("materials", changed, out, source)
    assert not out.exists() and core.sha(source / "preparation.local.json") == before
    assert core.verify_artifact(source)["status"] == "prepared_not_validated"


@pytest.mark.parametrize("name", ["lung", "bone", "heart", "fat"])
def test_changed_mask_cannot_bypass_zero_outside_body_bound(tmp_path, name):
    path, cfg = fixture_config(tmp_path); source = mesh_artifact(tmp_path, path)
    old = nib.load(tmp_path / (name+".nii.gz")); data = np.asarray(old.dataobj).copy()
    data[0, 0, 0] = 1  # Outside the source synthetic body, even with the same affine.
    image = nib.Nifti1Image(data, old.affine); image.header.set_xyzt_units("mm")
    nib.save(image, tmp_path / "different_mask.nii.gz")
    cfg["inputs"]["masks"][name] = "different_mask.nii.gz"
    changed = tmp_path / "changed.json"; write(changed, cfg)
    out = tmp_path / "rejected"
    with pytest.raises(ValueError, match="Mask identity changed"):
        core.prepare("materials", changed, out, source)
    assert not out.exists() and core.verify_artifact(source)["configuration"]["body"]["maximum_outside_body_voxels"][name] == 0


def test_parent_actual_contacts_must_agree_with_parent_passport(tmp_path):
    path, cfg = fixture_config(tmp_path); source = mesh_artifact(tmp_path, path)
    cfg["contacts"]["scenarios"][0]["measure_groups"].reverse()
    changed = tmp_path / "misdescribed.json"; write(changed, cfg)
    # Simulate a completed artifact produced by the former metadata-echo bug.
    (source / "preparation.local.json").unlink()
    loaded = core.load_config(changed)
    core.publish(source, "mesh", loaded, {}, core.artifact_sources(loaded))
    with pytest.raises(ValueError, match="Parent actual contacts disagree"):
        core.prepare("materials", changed, tmp_path / "rejected", source)
    assert not (tmp_path / "rejected").exists()


def test_completed_geometry_retained_but_new_refinement_settings_applied(tmp_path):
    path, cfg = fixture_config(tmp_path); source = mesh_artifact(tmp_path, path)
    materials = tmp_path / "materials"; core.prepare("materials", path, materials, source)
    expected = core.read(materials / "preparation.local.json")["completed_geometry_settings"]
    cfg["refinement"]["radius_mm"] = .3
    cfg["mesh"]["max_nodes"] = 1000
    changed = tmp_path / "new_refinement.json"; write(changed, cfg)
    out = tmp_path / "refined"; core.prepare("refine", changed, out, materials)
    for parent in [out, out / "inherited", out / "remapped"]:
        result = core.verify_artifact(parent)
        assert result["completed_geometry_settings"]["mesh_generation"] == expected["mesh_generation"]
        assert result["completed_geometry_settings"]["refinement"]["radius_mm"] == .3
        assert result["continuation_contract"]["coverage_bounds_preserved_without_relaxation"]


def cem_parameters(cfg):
    for row in cfg["cem_contacts"]["physical_parameters"]:
        row["patch_qc"] = dict(maximum_relative_area_error=.12, maximum_centroid_offset_mm=1.,
            maximum_extent_mm=3.7, extent_justification="Synthetic 2mm2 fixture case; not anatomical validation")


def test_cem_input_cli_binds_mesh_arrays_and_retains_original_string_ids(tmp_path):
    import ct_fem_contacts as cem
    path, cfg = fixture_config(tmp_path)
    names = ["01 I+", "U-плюс", "U-minus", "I minus"]
    cfg["contacts"]["physical_centres_lps_mm"] = dict(zip(names, cfg["contacts"]["physical_centres_lps_mm"].values()))
    cfg["contacts"]["physical_centres_lps_mm"]["unused candidate"] = dict(
        raw_lps_mm=[8.51, 4, 4], nominal_area_mm2=3., provenance="synthetic catalog only")
    group_ids = ["1-port", "U+ русский", "U minus", "I- port"]
    s = cfg["contacts"]["scenarios"][0]
    s.update(present_contact_ids=names, groups=[dict(id=g, contact_ids=[i]) for g, i in zip(group_ids, names)],
             drive_groups=[group_ids[0], group_ids[3]], measure_groups=[group_ids[1], group_ids[2]])
    for value, name in zip(cfg["cem_contacts"]["physical_parameters"], names):
        value["id"] = name
    cem_parameters(cfg); write(path, cfg)
    source = mesh_artifact(tmp_path, path); out = tmp_path / "cem_input"
    run = subprocess.run([sys.executable, "-B", str(ROOT / "tools/prepare_ct_fem.py"), "cem_contacts_input",
        "--config", str(path), "--input", str(source), "--out", str(out)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout+run.stderr
    manifest = core.verify_artifact(out); contract = core.read(out / "cem_contacts_input.local.json")
    assert manifest["stage"] == "cem_contacts_input" and not manifest["FEM_was_run"]
    assert not contract["CEM_contacts_QC_passed"] and not contract["quantitative_result_accepted"]
    assert core.sha(out / "mesh.mat") == core.sha(source / "mesh.mat")
    assert core.sha(out / "source_contacts.local.json") == core.sha(source / "contacts.local.json")
    assert [r["id"] for r in contract["contact_catalog"]] == names  # Absent candidate excluded entirely.
    assert [g["id"] for g in contract["scenarios"][0]["groups"]] == group_ids
    assert contract["scenarios"][0]["floating_contact_ids"] == []
    m = loadmat(out / "mesh.mat")
    for name, dtype in [("nodes_mm", "<f8"), ("elems", "<i8"), ("boundary", "<i8")]:
        assert contract["mesh_arrays"][name] == cem.array_identity(m[name], dtype)
        swapped = m[name].copy(); swapped[[0, 1]] = swapped[[1, 0]]
        assert cem.array_identity(swapped, dtype)["sha256"] != contract["mesh_arrays"][name]["sha256"]
    assert all(Path(f["path"]).is_absolute() and core.sha(f["path"]) == f["sha256"] for f in contract["source_files"])
    assert any(f["path"] == str(out / "cem_contacts_input.local.json") for f in manifest["product_files"])


def test_cem_present_floating_mixed_areas_and_z_are_explicit(tmp_path):
    import ct_fem_contacts as cem
    path, cfg = fixture_config(tmp_path); cem_parameters(cfg)
    s = cfg["contacts"]["scenarios"][0]
    s["groups"] = {"drive+": ["I_plus", "V_plus"], "drive-": ["I_minus"]}
    s["drive_groups"] = ["drive+", "drive-"]; s["measure_groups"] = ["drive-", "drive+"]
    cfg["contacts"]["physical_centres_lps_mm"]["I_plus"]["nominal_area_mm2"] = 5.
    cfg["cem_contacts"]["physical_parameters"][0]["z_contact_ohm_m2"] = .02
    actual = core.contact_metadata(trimesh.creation.box(extents=[17.02, 20, 20]), cfg["contacts"])
    rows, scenarios = cem.normalize(cfg, actual)
    assert scenarios[0]["floating_contact_ids"] == ["V_minus"]
    assert scenarios[0]["present_contact_ids"] == ["I_plus", "V_plus", "V_minus", "I_minus"]
    assert rows[0]["nominal_area_mm2"] == 5. and rows[1]["nominal_area_mm2"] == 2.
    assert rows[0]["z_contact_ohm_m2"] == .02 and rows[1]["z_contact_ohm_m2"] == .01
    assert scenarios[0]["groups"][0]["contact_ids"] == ["I_plus", "V_plus"]


@pytest.mark.parametrize("problem", ["presence_missing", "absent_assigned", "negative_z", "missing_z_source", "unknown_z_id", "nonstandard_extent", "relaxed_area", "duplicate_group"])
def test_cem_input_rejects_invalid_contacts_before_output(tmp_path, problem):
    import ct_fem_contacts as cem
    path, cfg = fixture_config(tmp_path); cem_parameters(cfg)
    if problem == "presence_missing":
        del cfg["contacts"]["scenarios"][0]["present_contact_ids"]
    elif problem == "absent_assigned":
        cfg["contacts"]["scenarios"][0]["present_contact_ids"].remove("V_minus")
        # Build a former-style metadata artifact, then make its scenario explicit
        # at preparation time below; both checks must reject physical absence.
        with pytest.raises(ValueError, match="explicitly present"):
            core.contact_metadata(trimesh.creation.box(extents=[17.02, 20, 20]), cfg["contacts"])
        return
    elif problem == "negative_z":
        cfg["cem_contacts"]["physical_parameters"][0]["z_contact_ohm_m2"] = -1
    elif problem == "missing_z_source":
        cfg["cem_contacts"]["physical_parameters"][0]["z_contact_source"] = ""
    elif problem == "unknown_z_id":
        cfg["cem_contacts"]["physical_parameters"][0]["id"] = "absent_from_catalog"
    elif problem == "nonstandard_extent":
        del cfg["cem_contacts"]["physical_parameters"][0]["patch_qc"]
    elif problem == "relaxed_area":
        cfg["cem_contacts"]["physical_parameters"][0]["patch_qc"]["maximum_relative_area_error"] = .13
    elif problem == "duplicate_group":
        s = cfg["contacts"]["scenarios"][0]
        s["groups"] = [{"id": "g", "contact_ids": ["I_plus"]}, {"id": "g", "contact_ids": ["V_plus"]}]
        with pytest.raises(ValueError, match="distinct"):
            core.contact_metadata(trimesh.creation.box(extents=[17.02, 20, 20]), cfg["contacts"])
        return
    write(path, cfg); source = mesh_artifact(tmp_path, path); out = tmp_path / "rejected_cem"
    with pytest.raises(ValueError):
        cem.prepare(path, out, source)
    assert not out.exists() and core.verify_artifact(source)["stage"] == "mesh"


def test_5mm_patch_limits_fixed_and_other_extent_requires_case_justification():
    import ct_fem_contacts as cem
    area = np.pi*2.5**2
    assert cem.patch_limits(area, None)["maximum_extent_mm"] == 3.7
    with pytest.raises(ValueError, match="3.7"):
        cem.patch_limits(area, dict(maximum_relative_area_error=.12, maximum_centroid_offset_mm=1.,
            maximum_extent_mm=3.8, extent_justification="not permitted"))
    with pytest.raises(ValueError, match="Non-5mm"):
        cem.patch_limits(2., None)


def test_adapter_source_uses_canonical_preparation_only_no_solve():
    import ct_fem_contacts as cem
    text = (cem.MATLAB / "src/run_ct_fem_contacts_prepare.m").read_text(encoding="utf-8")
    for helper in ("trkg4_local_exact_patches", "trkg4_local_patch_context", "trkg4_local_triangle_distance",
                   "trkg4_cem_contact_matrix", "trkg4_scale_fmdl_to_si"):
        assert helper+"(" in text
    assert "Kgrouped=T'*Kphysical*T" in text
    assert all(forbidden not in text for forbidden in ("fwd_solve(", "inv_solve(", "run_pipeline(", "calc_system_mat("))
    assert "strings_list(s.present_contact_ids)" in text and "assignment==0" in text


def test_changed_executed_remesh_is_rejected_before_mesh_backend(tmp_path, monkeypatch):
    path, source = remesh_surface_artifact(tmp_path)
    parent = tmp_path / "remesh"; core.prepare("remesh", path, parent, source)
    cfg = core.read(path); cfg["remeshing"]["contact_edge_mm"] = 1.2
    changed = tmp_path / "new.json"; write(changed, cfg)
    def forbidden(*args):
        pytest.fail("TetGen backend must not be reached")
    monkeypatch.setattr(core, "build_mesh", forbidden)
    with pytest.raises(ValueError, match="Executed remeshing"):
        core.prepare("mesh", changed, tmp_path / "rejected_mesh", parent)
    assert not (tmp_path / "rejected_mesh").exists()
    complete = core.verify_artifact(parent)["completed_geometry_settings"]
    assert complete["body"] == cfg["body"] and complete["synthetic_cylinders"] == []
    assert complete["remeshing"]["contact_edge_mm"] == 1.4


@pytest.mark.parametrize("problem", ["failed", "units", "unbound_sha"])
def test_cem_rejects_failed_or_misbound_source_mesh_qc(tmp_path, problem):
    path, cfg = fixture_config(tmp_path); cem_parameters(cfg); write(path, cfg)
    source = mesh_artifact(tmp_path, path)
    quality = core.read(source / "mesh_quality.local.json")
    if problem == "failed":
        quality["passed"] = False
    elif problem == "units":
        quality["units"] = "m"
    else:
        quality["mesh_sha256"] = "0"*64
    write(source / "mesh_quality.local.json", quality)
    (source / "preparation.local.json").unlink()
    loaded = core.load_config(path); core.publish(source, "mesh", loaded, {}, core.artifact_sources(loaded))
    with pytest.raises(ValueError, match="mesh QC"):
        core.prepare("cem_contacts_input", path, tmp_path / "rejected", source)
    assert not (tmp_path / "rejected").exists()


def test_grouped_pads_preserve_contact_energy_with_distinct_z_and_floating_pad():
    # Independent analytic triangle-integral identity, not a second production
    # assembler and not MATLAB execution. Final adapter equivalence is pending.
    mass = np.array([[2, 1, 1, -4], [1, 2, 1, -4], [1, 1, 2, -4], [-4, -4, -4, 12]])/12
    area = np.array([2., 5., 3.]); z = np.array([.01, .02, .04])
    k = np.zeros((12, 12))
    for pad in range(3):
        ix = [3*pad, 3*pad+1, 3*pad+2, 9+pad]
        k[np.ix_(ix, ix)] += area[pad]/z[pad]*mass
    tie = np.zeros((12, 11)); tie[:9, :9] = np.eye(9)
    tie[[9, 10], 9] = 1  # Group first two pads with different area and z.
    tie[11, 10] = 1      # Third pad remains a PRESENT floating port.
    grouped = tie.T@k@tie
    potentials = np.r_[np.arange(9)/7, [.3, -.4]]
    physical = tie@potentials
    direct_energy = sum(area[i]/z[i] * (physical[[3*i, 3*i+1, 3*i+2, 9+i]] @ mass @
        physical[[3*i, 3*i+1, 3*i+2, 9+i]]) for i in range(3))
    assert potentials@grouped@potentials == pytest.approx(direct_energy, abs=1e-12)
    np.testing.assert_allclose(grouped@np.ones(11), 0, atol=1e-13)
    assert grouped[10, 10] == pytest.approx(area[2]/z[2])
    assert np.linalg.eigvalsh(grouped).min() >= -1e-12


def test_duplicate_candidate_position_allowed_only_in_exclusive_scenarios(tmp_path):
    import copy
    import ct_fem_contacts as cem
    path, cfg = fixture_config(tmp_path); cem_parameters(cfg)
    cfg["contacts"]["physical_centres_lps_mm"]["alternative I+"] = copy.deepcopy(
        cfg["contacts"]["physical_centres_lps_mm"]["I_plus"])
    p = copy.deepcopy(cfg["cem_contacts"]["physical_parameters"][0]); p["id"] = "alternative I+"
    cfg["cem_contacts"]["physical_parameters"].append(p)
    s = copy.deepcopy(cfg["contacts"]["scenarios"][0]); s["id"] = "exclusive second size"
    s["present_contact_ids"][0] = "alternative I+"; s["groups"]["g1"] = ["alternative I+"]
    cfg["contacts"]["scenarios"].append(s)
    surface = trimesh.creation.box(extents=[17.02, 20, 20])
    actual = core.contact_metadata(surface, cfg["contacts"])
    _, scenarios = cem.normalize(cfg, actual)
    assert "alternative I+" not in scenarios[0]["present_contact_ids"]
    assert "I_plus" not in scenarios[1]["present_contact_ids"]
    assert all(not s["floating_contact_ids"] for s in scenarios)
    import ct_fem_remesh as remesh
    surface.export(tmp_path / "catalog_reference.stl")
    actual.update(projection_surface_path=str(tmp_path / "catalog_reference.stl"),
                  projection_surface_sha256=core.sha(tmp_path / "catalog_reference.stl"))
    projected = remesh.update_contacts(surface, actual, cfg, remesh.settings(cfg))
    assert len(projected["physical_contacts"]) == 5
    _, remesh_scenarios = cem.normalize(cfg, projected)
    assert remesh_scenarios == scenarios
    cfg["contacts"]["scenarios"][0]["present_contact_ids"].append("alternative I+")
    with pytest.raises(ValueError, match="duplicate raw"):
        core.contact_metadata(surface, cfg["contacts"])


def test_one_voxel_closing_is_not_a_boundary_distance_bound(tmp_path):
    # A thin through-channel costs <.001 body voxels to close, but deletes
    # boundary deep inside the body. Kernel support is not a Hausdorff bound.
    original = np.zeros((40, 40, 40), bool); original[2:38, 2:38, 2:38] = True
    original[20, 20, 2:38] = False
    closed = core.close_body_once(original)
    assert np.count_nonzero(original != closed)/original.sum() < .001
    assert not np.any(original & ~closed)
    kernel = core.kernels()
    raw = kernel["surface_from_mask"](original, np.eye(4), level=.49)
    corrected = kernel["surface_from_mask"](closed, np.eye(4), level=.49)
    interior_wall, _, _ = raw.nearest.on_surface([[-20., -20., 20.]])
    _, distance, _ = corrected.nearest.on_surface(interior_wall)
    assert distance[0] > 10.  # One closing iteration, despite a one-voxel structuring radius.
    corrected.export(tmp_path / "corrected.stl")
    exported = trimesh.load(tmp_path / "corrected.stl", force="mesh")
    fidelity = kernel["sampled_distance"](corrected, exported, 64)
    assert fidelity["max_mm"] < 1e-4


def test_surface_distance_contract_requires_named_endpoints_and_separate_budgets(tmp_path):
    path, cfg = fixture_config(tmp_path)
    settings = core.surface_distance_settings(cfg)
    topology = {"reference_to_candidate": {"max_mm": 2.918}, "candidate_to_reference": {"max_mm": 1.225}}
    exported = {"reference_to_candidate": {"max_mm": 1.4e-5}, "candidate_to_reference": {"max_mm": 1.4e-5}}
    settings["topology_correction"]["maximum_sampled_distance_mm"] = 1.25
    report = core.surface_distance_qc(topology, exported, settings)
    assert not report["passed"] and not report["topology_correction"]["passed"]
    assert report["export_fidelity"]["passed"] and not report["automatic_threshold_relaxation"]
    cfg["surface_qc"]["export_fidelity"]["reference"] = "HU_body_native_level_0.49"
    with pytest.raises(ValueError, match="endpoints"):
        core.surface_distance_settings(cfg)
    cfg = core.read(path); cfg["surface_qc"]["maximum_sampled_correction_mm"] = 1.25
    write(path, cfg)
    with pytest.raises(ValueError, match="Ambiguous legacy"):
        core.prepare("surface", path, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_failed_export_fidelity_retains_separate_qc_evidence(tmp_path):
    path, cfg = fixture_config(tmp_path)
    for file in [tmp_path / "static_ct.nii.gz", *(tmp_path / (n+".nii.gz") for n in ("lung", "bone", "heart", "fat"))]:
        image = nib.load(file); data = np.asanyarray(image.dataobj).copy(); affine = image.affine.copy()
        affine[:3, :3] = np.diag([.73, 1.11, 1.37])
        changed = nib.Nifti1Image(data, affine); changed.header.set_xyzt_units("mm"); nib.save(changed, file)
    cfg["surface_qc"]["export_fidelity"]["maximum_sampled_distance_mm"] = 1e-9
    cfg["surface_qc"]["export_fidelity"]["bound_basis"] = "Deliberately strict synthetic negative control"
    write(path, cfg); out = tmp_path / "export_rejected"
    with pytest.raises(ValueError, match="exported-STL fidelity"):
        core.prepare("surface", path, out)
    qc = core.read(out / "body_correction_qc.local.json")
    assert qc["topology_correction"]["passed"] and not qc["export_fidelity"]["passed"]
    assert not (out / "preparation.local.json").exists()
    assert (out / "body_hu_mask.nii.gz").exists() and (out / "skin_regularized.stl").exists()


def test_saved_body_diagnostic_component_connectivity_and_worst_location():
    import ct_fem_body_diagnostic as diagnostic
    voxels = np.array([[0, 0, 0], [1, 1, 1], [2, 1, 1], [7, 7, 7]])
    added = np.ones(4, bool)
    c6, _ = diagnostic.components(voxels, voxels*2., added, 8., 6)
    c26, labels = diagnostic.components(voxels, voxels*2., added, 8., 26)
    assert len(c6) == 3 and len(c26) == 2 and c26[0]["voxels"] == 3
    assert c26[0]["volume_ml"] == .024
    a = trimesh.creation.box(extents=[4., 4., 4.]); b = a.copy(); b.apply_translation([.7, 0, 0])
    report, _ = diagnostic.distances(a, b, 128, .4, diagnostic.cKDTree(voxels*2.), voxels, labels)
    worst = report["worst_locations"][0]
    assert report["max_mm"] == pytest.approx(.7)
    assert np.linalg.norm(np.array(worst["source_lps_mm"])-worst["closest_target_lps_mm"]) == pytest.approx(.7)
    assert report["maximum_is_sampled_not_certified"]
