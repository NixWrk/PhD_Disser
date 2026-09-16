from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import refine_heart_interior_mesh as interior
import refine_heart_volume_mesh as base


def _orient_positive(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    result = tets.copy()
    signed = base._signed_tetra_volumes(nodes, result)
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


def _ring_mesh():
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
        np.array(
            [[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 2]],
            dtype=np.int64,
        ),
    )
    boundary = _free_boundary(tets)
    labels = np.array([3, 1, 4], dtype=np.int64)
    return nodes, tets, boundary, labels


def _candidates(nodes: np.ndarray, tets: np.ndarray, labels: np.ndarray) -> dict:
    heart = np.flatnonzero(labels == 3)
    volumes = base._signed_tetra_volumes(nodes, tets[heart])
    centre = np.sum(volumes[:, None] * nodes[tets[heart]].mean(axis=1), axis=0)
    centre /= volumes.sum()
    shape = {
        "centre_m": centre.tolist(),
        "semiaxes_m": [2.0, 2.0, 2.0],
        "axes_columns": np.eye(3).tolist(),
    }
    return {
        "originalheart": {"centre_m": centre.tolist()},
        "sphere": shape,
        "ellipsoid": shape,
    }


def test_boundary_edges_are_excluded_without_changing_the_candidate_region():
    nodes, tets, boundary, _ = _ring_mesh()
    partition = interior.select_interior_marked_edges(
        tets,
        boundary,
        np.arange(len(tets), dtype=np.int64),
        len(nodes),
    )
    assert partition["selected_tets"] == 3
    assert partition["total_candidate_edges"] == 10
    assert partition["excluded_outer_edges_count"] == 9
    assert partition["included_interior_edges_count"] == 1
    assert np.array_equal(
        partition["included_interior_edges"], np.array([[0, 1]], dtype=np.int64)
    )
    assert partition["selected_tets_with_zero_remaining_marked_edges"] == 0


def test_interior_refinement_keeps_surface_and_records_local_qc():
    nodes, tets, boundary, labels = _ring_mesh()
    result = interior.refine_heart_mesh(
        nodes,
        tets,
        boundary,
        labels,
        _candidates(nodes, tets, labels),
        collar_mm=0.0,
    )
    new_nodes, new_tets, new_boundary, new_labels, parents, audit = result
    assert np.array_equal(new_nodes[: len(nodes)], nodes)
    assert np.array_equal(new_boundary, boundary)
    assert np.array_equal(new_labels, labels[parents - 1])
    assert np.all(base._signed_tetra_volumes(new_nodes, new_tets) > 0)
    assert audit["metrics"]["selection"]["collar_mm"] == 0.0
    assert audit["metrics"]["selection"]["excluded_outer_edges"] == 9
    assert audit["metrics"]["selection"]["included_interior_edges"] == 1
    assert audit["metrics"]["heart_plus_affected_region"]["volume_by_tissue_id_m3"]
    assert audit["metrics"]["heart_plus_affected_region"]["edge_length_mm"]
    assert len(audit["candidate_boxes"]) == 9
    assert all(bool(value) for value in audit["checks"].values())


def test_run_report_is_exporter_compatible_and_declares_interior_only_scope(
    tmp_path, monkeypatch
):
    scipy_io = pytest.importorskip(
        "scipy.io", reason="scipy is required for MAT output"
    )
    nodes, tets, boundary, labels = _ring_mesh()
    source = tmp_path / "prepared_fem.mat"
    source.write_bytes(b"prepared mesh fixture")
    normalized = _candidates(nodes, tets, labels)
    candidate_data = {
        "original_heart": normalized["originalheart"],
        "candidates": {
            "sphere": normalized["sphere"],
            "ellipsoid": normalized["ellipsoid"],
        },
    }
    candidate_path = tmp_path / "candidate_parameters.json"
    candidate_path.write_text(
        json.dumps(candidate_data),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        interior._base,
        "load_prepared_mesh",
        lambda path: (nodes, tets, boundary, labels),
    )

    output = tmp_path / "L1_interior_collar0"
    report = interior.run(
        source,
        candidate_path,
        output,
        collar_mm=0.0,
    )
    saved = json.loads(
        (output / "refinement_report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "passed_heart_volume_refinement"
    assert saved["status"] == "passed_heart_volume_refinement"
    assert all(value is True for value in saved["checks"].values())
    assert saved["selection"]["collar_mm"] == 0.0
    assert saved["selection"]["excluded_outer_edges"] == 9
    assert saved["selection"]["included_interior_edges"] == 1
    assert len(saved["candidate_boxes"]) == 9
    assert all(
        isinstance(box["lower_m"], list) and isinstance(box["upper_m"], list)
        for box in saved["candidate_boxes"]
    )
    assert saved["refinement_scope"]["kind"] == "interior_only"
    assert saved["refinement_scope"]["surface_contact_mesh_unchanged"] is True
    assert (
        saved["refinement_scope"]["complete_global_spatial_convergence_proof"]
        is False
    )
    assert len(saved["provenance"]["wrapper_implementation"]["sha256"]) == 64
    assert len(saved["provenance"]["split_simplices_source"]["sha256"]) == 64
    report_bytes = (output / "refinement_report.json").read_bytes()
    assert report_bytes.endswith(b"\n")
    assert not report_bytes.endswith(b"\\n")

    refined = scipy_io.loadmat(output / "refined_mesh.mat")
    for field in ("nodes_m", "elems", "boundary", "tissue_id", "parent_tet_id"):
        assert field in refined
    assert np.array_equal(refined["boundary"], boundary + 1)
    assert np.min(refined["elems"]) >= 1
    assert np.min(refined["boundary"]) >= 1
    assert np.min(refined["parent_tet_id"]) >= 1
