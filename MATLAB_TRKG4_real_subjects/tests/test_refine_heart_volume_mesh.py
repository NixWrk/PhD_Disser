from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import refine_heart_volume_mesh as mesh


def _orient_positive(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    result = tets.copy()
    signed = mesh._signed_tetra_volumes(nodes, result)
    negative = np.flatnonzero(signed < 0)
    result[negative, 2], result[negative, 3] = (
        result[negative, 3].copy(),
        result[negative, 2].copy(),
    )
    return result


def _free_boundary(tets: np.ndarray) -> np.ndarray:
    faces = np.concatenate(
        [
            tets[:, [1, 2, 3]],
            tets[:, [0, 2, 3]],
            tets[:, [0, 1, 3]],
            tets[:, [0, 1, 2]],
        ]
    )
    canonical = np.sort(faces, axis=1)
    unique, counts = np.unique(canonical, axis=0, return_counts=True)
    return unique[counts == 1]


def _ring_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nodes = np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [-0.5, np.sqrt(3.0) / 2.0, 0.0],
            [-0.5, -np.sqrt(3.0) / 2.0, 0.0],
        ]
    )
    tets = _orient_positive(
        nodes,
        np.array([[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 2]], dtype=np.int64),
    )
    return nodes, tets, _free_boundary(tets), np.array([3, 1, 4], dtype=np.int64)


def _shape_parameters():
    shape = {
        "centre_m": [0.0, 0.0, 0.0],
        "semiaxes_m": [0.2, 0.2, 0.2],
        "axes_columns": np.eye(3).tolist(),
    }
    return {
        "original_heart": {"centre_m": [0.0, 0.0, 0.0]},
        "candidates": {"sphere": shape, "ellipsoid": shape},
    }


def test_actual_candidate_artifact_layout_is_loaded():
    path = ROOT / "output" / "exploratory" / "static_heart_replacement_20260915" / "candidate_parameters.json"
    if not path.exists():
        pytest.skip("local candidate artifact is not present")
    candidates = mesh._read_candidates(path)
    assert candidates["originalheart"]["centre_m"] == candidates["original_heart"]["centre_m"]
    assert set(("sphere", "ellipsoid")).issubset(candidates)


def test_candidate_fixture_merges_top_level_original_heart_with_nested_shapes(tmp_path):
    path = tmp_path / "candidate_parameters.json"
    path.write_text(json.dumps(_shape_parameters()), encoding="utf-8")
    candidates = mesh._read_candidates(path)
    assert candidates["originalheart"]["centre_m"] == [0.0, 0.0, 0.0]
    assert candidates["sphere"]["semiaxes_m"] == [0.2, 0.2, 0.2]
    assert candidates["ellipsoid"]["axes_columns"] == np.eye(3).tolist()


def test_full_edge_bisection_has_eight_positive_children_and_preserves_volume():
    nodes = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    tets = np.array([[0, 1, 2, 3]], dtype=np.int64)
    edges = np.array(list(itertools.combinations(range(4), 2)), dtype=np.int64)
    children, parents = mesh.split_simplices(tets, edges, len(nodes))
    fine_nodes = np.vstack([nodes, (nodes[edges[:, 0]] + nodes[edges[:, 1]]) / 2.0])
    volumes = mesh._signed_tetra_volumes(fine_nodes, children)
    assert len(children) == 8
    assert np.array_equal(parents, np.zeros(8, dtype=np.int64))
    assert np.all(volumes > 0)
    assert np.isclose(volumes.sum(), mesh._signed_tetra_volumes(nodes, tets)[0])


def test_shared_face_split_closure_and_material_parent_inheritance():
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    tets = _orient_positive(
        nodes, np.array([[0, 1, 2, 3], [0, 1, 2, 4]], dtype=np.int64)
    )
    edges = np.array([[0, 1]], dtype=np.int64)
    children, parents = mesh.split_simplices(tets, edges, len(nodes))
    fine_nodes = np.vstack([nodes, (nodes[0] + nodes[1]) / 2.0])
    assert len(children) == 4
    assert np.array_equal(np.bincount(parents, minlength=2), np.array([2, 2]))
    assert np.all(mesh._signed_tetra_volumes(fine_nodes, children) > 0)
    faces = np.concatenate(
        [
            children[:, [1, 2, 3]],
            children[:, [0, 2, 3]],
            children[:, [0, 1, 3]],
            children[:, [0, 1, 2]],
        ]
    )
    faces.sort(axis=1)
    unique, counts = np.unique(faces, axis=0, return_counts=True)
    assert counts[np.all(unique == [0, 2, 5], axis=1)][0] == 2
    assert counts[np.all(unique == [1, 2, 5], axis=1)][0] == 2


def test_interior_marked_edge_preserves_boundary_labels_and_parent_volumes():
    nodes, tets, boundary, labels = _ring_mesh()
    result = mesh.refine_marked_edges(
        nodes, tets, boundary, labels, np.array([[0, 1]], dtype=np.int64)
    )
    new_nodes, new_tets, new_boundary, new_labels, parents, qc = result
    assert len(new_nodes) == len(nodes) + 1
    assert len(new_tets) == 2 * len(tets)
    assert np.array_equal(new_nodes[: len(nodes)], nodes)
    assert np.array_equal(new_boundary, boundary)
    assert np.array_equal(new_labels, labels[parents - 1])
    assert np.array_equal(np.unique(parents), np.arange(1, len(tets) + 1))
    assert all(qc["checks"].values())
    assert qc["metrics"]["original_heart_volume_m3"]["absolute_error"] < 1e-15
    assert "quality_distribution" in qc["metrics"]
    assert "local_edge_length_mm" in qc["metrics"]
    assert np.all(mesh._signed_tetra_volumes(new_nodes, new_tets) > 0)


def test_boundary_marked_edge_is_rejected_without_shrinking_the_request():
    nodes = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    tets = np.array([[0, 1, 2, 3]], dtype=np.int64)
    boundary = _free_boundary(tets)
    labels = np.array([3], dtype=np.int64)
    with pytest.raises(mesh.BoundaryRefinementError, match="smaller collar"):
        mesh.refine_marked_edges(
            nodes, tets, boundary, labels, np.array([[0, 1]], dtype=np.int64)
        )


def test_prepared_mesh_requires_coordinate_units_m(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "prepared_fem.mat"
    nodes = np.array(
        [[0.0, 0.001, 0.0, 0.0], [0.0, 0.0, 0.001, 0.0], [0.0, 0.0, 0.0, 0.001]]
    )
    elems = np.array([[1], [2], [3], [4]], dtype=np.int64)
    boundary = np.array([[2, 2, 1, 1], [3, 4, 3, 4], [4, 3, 4, 3]], dtype=np.int64)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("fmdl_m")
        group.create_dataset("nodes", data=nodes)
        group.create_dataset("elems", data=elems)
        group.create_dataset("boundary", data=boundary)
        handle.create_dataset("tissue_id", data=np.array([[3]], dtype=np.uint8))
        handle.create_dataset("coordinate_units", data=np.array([ord("m")], dtype=np.uint16))
    loaded_nodes, loaded_tets, loaded_boundary, loaded_labels = mesh.load_prepared_mesh(path)
    assert loaded_nodes.shape == (4, 3)
    assert loaded_tets.shape == (1, 4)
    assert loaded_boundary.shape == (4, 3)
    assert loaded_labels.tolist() == [3]

    bad = tmp_path / "prepared_mm.mat"
    with h5py.File(bad, "w") as handle:
        group = handle.create_group("fmdl_m")
        group.create_dataset("nodes", data=nodes)
        group.create_dataset("elems", data=elems)
        group.create_dataset("boundary", data=boundary)
        handle.create_dataset("tissue_id", data=np.array([[3]], dtype=np.uint8))
        handle.create_dataset("coordinate_units", data=np.array([ord(c) for c in "mm"], dtype=np.uint16))
    with pytest.raises(mesh.RefinementError, match="coordinate_units"):
        mesh.load_prepared_mesh(bad)


def test_failed_run_writes_explicit_failed_report(tmp_path):
    output = tmp_path / "failed"
    report = mesh.run(tmp_path / "missing.mat", tmp_path / "missing.json", output)
    assert report["status"] == "failed_heart_volume_refinement"
    assert report["refined_mesh_sha256"] is None
    assert all(value is False for value in report["checks"].values())
    saved = json.loads((output / "refinement_report.json").read_text(encoding="utf-8"))
    assert saved["status"] == report["status"]
