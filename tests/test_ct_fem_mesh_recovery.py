"""Bounded feedback, automatic guide selection and unchanged geometry gates."""
from pathlib import Path
import sys

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import ct_fem_core as core
import ct_fem_mesh_recovery as recovery
from test_ct_fem_preparation import remesh_surface_artifact, fixture_config


def tetrahedra(near_boundary=True):
    x = 14.99 if near_boundary else 0.
    bad = np.array([[x, -2., 0.], [x, 2., 0.], [x, 0., 2.], [x-1e-7, 0., .5]])
    good = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.], [0., 0., 2.]])
    tets = np.array([[0, 1, 2, 3]], np.int32)
    return bad, good, tets


@pytest.mark.parametrize("eventual_success", [True, False])
def test_shape_feedback_is_bounded_and_never_publishes_a_mesh_before_final_QC(tmp_path, monkeypatch, eventual_success):
    path, source = remesh_surface_artifact(tmp_path); cfg = core.load_config(path)
    cfg["mesh"]["quality_recovery"]["maximum_rounds"] = 1
    bad, good, tets = tetrahedra(); calls = []
    def generate(file, out, settings):
        calls.append(file)
        return (good if len(calls) > 1 and eventual_success else bad), tets, "synthetic_test"
    monkeypatch.setattr(recovery, "generate", generate)
    out = tmp_path / "attempts"; out.mkdir()
    surface = trimesh.load(source / "exterior.stl", force="mesh")
    if eventual_success:
        _, _, _, _, contacts, rounds = recovery.generate_screened(cfg, source, out, surface)
        assert rounds == 1 and contacts["physical_contacts"]
        assert core.read(out / "mesh_attempts.local.json")["selected_attempt"] == 1
    else:
        with pytest.raises(ValueError, match="bounded local recovery"):
            recovery.generate_screened(cfg, source, out, surface)
        assert core.read(out / "mesh_attempts.local.json")["selected_attempt"] is None
    assert len(calls) == 2
    assert not (out / "mesh.mat").exists() and not (out / "preparation.local.json").exists()
    assert core.read(out / "recovery_01/guides.local.json")["element_ids_zero_based"] == [0]
    measured = core.read(out / "recovery_01/surface_qc.local.json")
    assert measured["passed"] and measured["original_exterior_is_reference"]


def test_interior_failure_does_not_trigger_unrelated_surface_changes(tmp_path, monkeypatch):
    path, source = remesh_surface_artifact(tmp_path); cfg = core.load_config(path)
    bad, _, tets = tetrahedra(near_boundary=False)
    monkeypatch.setattr(recovery, "generate", lambda *args: (bad, tets, "synthetic_test"))
    out = tmp_path / "interior"; out.mkdir()
    with pytest.raises(ValueError, match="boundary neighbourhood"):
        recovery.generate_screened(cfg, source, out, trimesh.load(source / "exterior.stl", force="mesh"))
    assert not (out / "recovery_01/exterior.stl").exists()
    assert core.read(out / "mesh_attempts.local.json")["failed_conditioning_round"] == 1


def test_curved_boundary_tetrahedron_is_local_even_when_its_centroid_is_deep(tmp_path, monkeypatch):
    cfg = core.read(core.ROOT/'configs/ct_fem_preparation.example.json')
    sphere = trimesh.creation.icosphere(subdivisions=4,radius=10.)
    file = tmp_path/'sphere.stl';sphere.export(file)
    nodes = np.array([[-8.,0.,6.],[8.,0.,6.],[0.,8.,6.],[0.,2.5,6.000001]])
    tets = np.array([[0,1,2,3]],np.int32)
    assert sphere.nearest.on_surface(nodes.mean(axis=0)[None,:])[1][0] > 2.
    class ReachedBoundarySelection(Exception): pass
    def reached(surface,guides,*args):
        assert len(guides) >= 4
        raise ReachedBoundarySelection()
    monkeypatch.setattr(recovery,'select_region',reached)
    with pytest.raises(ReachedBoundarySelection):
        recovery.conditioned_surface(cfg,tmp_path,file,sphere,{},nodes,tets,tmp_path,core.kernels())
    evidence = core.read(tmp_path/'guides.local.json')
    assert min(evidence['vertex_boundary_distances_mm'][0]) < .02


def test_local_conditioning_cannot_change_the_original_geometry_budget(tmp_path):
    path, source = remesh_surface_artifact(tmp_path); cfg = core.load_config(path)
    original = trimesh.load(source / "exterior.stl", force="mesh")
    original.apply_translation([2., 0., 0.])
    bad, _, tets = tetrahedra(); tets = core.kernels()["orient"](bad, tets)
    out = tmp_path / "budget"; out.mkdir()
    with pytest.raises(ValueError, match="original geometry budget"):
        recovery.conditioned_surface(cfg, source, source / "exterior.stl", original,
            core.read(source / "contacts.local.json"), bad, tets, out, core.kernels())
    measured = core.read(out / "surface_qc.local.json")
    assert not measured["passed"]
    assert max(d["max_mm"] for d in measured["sampled_distances"].values()) > .75
    assert not (out / "contacts.local.json").exists()


def test_pipeline_failure_preserves_completed_stages_and_stops(tmp_path, monkeypatch):
    path, _ = fixture_config(tmp_path); seen = []
    def failed(stage, config, out, parent):
        seen.append(stage)
        raise ValueError("synthetic stage refusal")
    monkeypatch.setattr(core, "prepare", failed)
    out = tmp_path / "pipeline"
    with pytest.raises(ValueError, match="stage refusal"):
        core.prepare_mesh_pipeline(path, out)
    assert seen == ["surface"]
    assert not (out / "pipeline.local.json").exists()
    assert core.read(out / "pipeline_failed.local.json")["completed"] == []


@pytest.mark.parametrize("field,value", [("maximum_rounds", -.1), ("iterations", 0),
    ("maximum_boundary_distance_mm", float("nan")), ("maximum_bad_elements", 1.2)])
def test_recovery_policy_rejects_invalid_resource_and_geometry_settings(field, value):
    cfg = core.read(core.ROOT / "configs/ct_fem_preparation.example.json")
    cfg["mesh"]["quality_recovery"][field] = value
    with pytest.raises(ValueError):
        recovery.validate_settings(cfg["mesh"]["quality_recovery"])
