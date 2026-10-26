"""Native failure must leave evidence and never return an accepted mesh."""
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import ct_fem_core as core
import ct_fem_tetgen as backend


@pytest.mark.parametrize("failure", ["timeout", "native_error"])
def test_native_failure_preserves_request_log_and_exit_status(tmp_path, monkeypatch, failure):
    source = tmp_path / "input.stl"; source.write_bytes(b"source bytes")
    out = tmp_path / "mesh"; out.mkdir()
    monkeypatch.setitem(sys.modules, "tetgen", SimpleNamespace(
        __file__=str(tmp_path / "runtime/tetgen/__init__.py"), __version__="test_runtime"))
    calls = []

    class Child:
        returncode = -9

        def __init__(self, command, **kwargs):
            calls.append(command)
            kwargs["stdout"].write("native diagnostic\n")

        def wait(self, timeout=None):
            if timeout is not None and failure == "timeout":
                raise subprocess.TimeoutExpired(calls[0], timeout)
            return 1 if failure == "native_error" else self.returncode

        def kill(self):
            calls.append("killed_own_child")

    monkeypatch.setattr(backend.subprocess, "Popen", Child)
    settings = core.read(Path(core.ROOT) / "configs/ct_fem_preparation.example.json")["mesh"]
    settings["maximum_seconds"] = 2.
    with pytest.raises(TimeoutError if failure == "timeout" else RuntimeError):
        backend.generate(source, out, settings)
    request = core.read(out / "tetgen_request.local.json")
    assert request["surface_sha256"] == core.sha(source)
    assert request["options"]["quality"] is True
    assert request["options"]["minratio"] == 1.5 and request["options"]["mindihedral"] == 8.
    assert request["raw_output_is_accepted_mesh"] is False
    status = core.read(out / "tetgen_process.local.json")
    assert status["timed_out"] == (failure == "timeout")
    assert ("killed_own_child" in calls) == (failure == "timeout")
    assert (out / "tetgen_stdout.local.log").read_text(encoding="utf8") == "native diagnostic\n"
    assert not (out / "mesh.mat").exists()


@pytest.mark.parametrize("name,value", [("epsilon", float("nan")), ("optmaxdihedral", 180),
    ("opt_scheme", 8), ("nomergefacet", 1), ("opt_iterations", -1), ("smooth_maxiter", .5)])
def test_invalid_native_options_are_rejected(name, value):
    settings = core.read(core.ROOT / "configs/ct_fem_preparation.example.json")["mesh"]
    settings["tetgen_options"][name] = value
    with pytest.raises(ValueError):
        backend.validate_options(settings)


def test_two_allowed_stage_displacements_cannot_double_the_original_budget(tmp_path, monkeypatch):
    cfg = core.read(core.ROOT / "configs/ct_fem_preparation.example.json")
    cfg["surface_qc"]["sample_count"] = 256
    nodes = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.], [0., 0., 2.]])
    tets = np.array([[0, 1, 2, 3]], np.int32)
    faces = core.kernels()["boundary_faces"](tets)
    original = tmp_path / "original.stl"
    trimesh.Trimesh(nodes, faces, process=False).export(original)
    source = tmp_path / "remesh"; source.mkdir()
    trimesh.Trimesh(nodes+[.4, 0., 0.], faces, process=False).export(source / "exterior.stl")
    core.write_new(source / "surface_qc.local.json", dict(synthetic_cylinders=[]))
    core.write_new(source / "contacts.local.json", dict(
        projection_surface_path=str(source / "exterior.stl"),
        projection_surface_sha256=core.sha(source / "exterior.stl"),
        original_projection_surface_path=str(original), original_projection_surface_sha256=core.sha(original)))
    monkeypatch.setattr(backend, "generate", lambda *args: (nodes+[.8, 0., 0.], tets, "test"))
    out = tmp_path / "mesh"; out.mkdir()
    with pytest.raises(ValueError, match="boundary exceeds"):
        core.build_mesh(cfg, source, out)
    qc = core.read(out / "tetgen_boundary_qc.local.json")
    assert qc["passed"] is False
    assert max(d["max_mm"] for d in qc["boundary_distances"].values()) <= .75
    assert max(d["max_mm"] for d in qc["original_boundary_distances"].values()) > .75
    assert not (out / "mesh.mat").exists()
