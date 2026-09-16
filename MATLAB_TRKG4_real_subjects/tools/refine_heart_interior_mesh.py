"""Interior-only bounded heart refinement with an unchanged exterior surface.

The candidate tetrahedra use the frozen volume module's candidate boxes and
vertex-AABB selector. Every candidate edge is considered. Edges occurring in
the original boundary triangles are excluded explicitly; the remaining edges
are sent globally to the frozen midpoint splitter, so every incident
tetrahedron is refined consistently. The unchanged boundary is intentional:
this produces an interior-field refinement and is not a complete global
spatial-convergence proof.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import refine_heart_volume_mesh as _base
except ModuleNotFoundError:
    from tools import refine_heart_volume_mesh as _base


TET_EDGE_PAIRS = tuple(itertools.combinations(range(4), 2))
SELECTION_POLICY_NAME = "vertex_aabb_union_exclude_boundary_edges"


def _serialize_candidate_boxes(boxes: list[dict]) -> list[dict]:
    return [
        {
            **box,
            "lower_m": np.asarray(box["lower_m"], dtype=np.float64).tolist(),
            "upper_m": np.asarray(box["upper_m"], dtype=np.float64).tolist(),
            "signed_volume_fraction": float(box["signed_volume_fraction"]),
            "scale_about_volume_centroid": float(
                box["scale_about_volume_centroid"]
            ),
        }
        for box in boxes
    ]


def _selection_policy(collar_mm: float) -> dict:
    return {
        "name": SELECTION_POLICY_NAME,
        "candidate_selection": "vertex AABB overlap with the union of originalheart, sphere, and ellipsoid boxes",
        "collar_mm": float(collar_mm),
        "boundary_edge_action": "exclude every candidate edge present in the original boundary triangles",
        "collar_action": "keep the requested collar unchanged; never shrink it to avoid a boundary edge",
        "interior_refinement_action": "refine every remaining interior edge globally, including all incident tetrahedra",
        "surface_contact_action": "copy the original boundary array exactly, preserving node IDs, order, and orientation",
    }


def provenance() -> dict:
    """Return hashes for this wrapper and every reused refinement primitive."""
    implementation = Path(__file__).resolve()
    base_provenance = _base.provenance()
    return {
        "wrapper_implementation": {
            "path": str(implementation),
            "sha256": _base.sha256(implementation),
        },
        "base_volume_refinement_implementation": base_provenance["implementation"],
        "split_simplices_source": base_provenance["split_simplices_source"],
        "selection_policy": {
            "function": "select_interior_marked_edges",
            "name": SELECTION_POLICY_NAME,
        },
    }


def select_interior_marked_edges(
    tets_zero_based: np.ndarray,
    boundary_zero_based: np.ndarray,
    selected_tet_indices: np.ndarray,
    node_count: int,
) -> dict:
    """Partition candidate edges into excluded boundary and interior sets.

    The returned interior set is global: it is passed unchanged to
    refine_marked_edges and therefore refines all incident input tets,
    including incident tets outside the AABB-selected set.
    """
    tets = np.asarray(tets_zero_based, dtype=np.int64)
    boundary = np.asarray(boundary_zero_based, dtype=np.int64)
    selected = np.asarray(selected_tet_indices, dtype=np.int64).reshape(-1)
    _base._require(
        selected.ndim == 1 and len(selected) > 0,
        "Candidate AABB selection returned no tetrahedra",
    )
    _base._require(
        np.all((selected >= 0) & (selected < len(tets))),
        "Selected tetrahedron index is outside the input range",
    )

    candidate_edges = _base._unique_edges_from_tets(tets[selected])
    original_boundary_edges = _base._boundary_edges(boundary)
    candidate_codes = _base._edge_codes(candidate_edges, node_count)
    boundary_codes = _base._edge_codes(original_boundary_edges, node_count)
    excluded_mask = np.isin(candidate_codes, boundary_codes)
    excluded_edges = candidate_edges[excluded_mask]
    included_edges = candidate_edges[~excluded_mask]

    edge_pairs = np.asarray(TET_EDGE_PAIRS, dtype=np.int64)
    selected_edges = np.sort(tets[selected][:, edge_pairs], axis=2)
    selected_edge_codes = _base._edge_codes(
        selected_edges.reshape(-1, 2), node_count
    ).reshape(len(selected), len(TET_EDGE_PAIRS))
    remaining_counts = np.count_nonzero(
        np.isin(
            selected_edge_codes,
            _base._edge_codes(included_edges, node_count),
        ),
        axis=1,
    )

    return {
        "candidate_edges": candidate_edges,
        "excluded_outer_edges": excluded_edges,
        "included_interior_edges": included_edges,
        "selected_tet_remaining_edge_counts": remaining_counts,
        "selected_tets": int(len(selected)),
        "total_candidate_edges": int(len(candidate_edges)),
        "excluded_outer_edges_count": int(len(excluded_edges)),
        "included_interior_edges_count": int(len(included_edges)),
        "selected_tets_with_zero_remaining_marked_edges": int(
            np.count_nonzero(remaining_counts == 0)
        ),
    }


def _selection_record(partition: dict, selected: np.ndarray, collar_mm: float) -> dict:
    selected = np.asarray(selected, dtype=np.int64).reshape(-1)
    return {
        "collar_mm": float(collar_mm),
        "selected_tets": int(partition["selected_tets"]),
        "selected_tet_id_range_1based": [
            int(selected.min() + 1),
            int(selected.max() + 1),
        ],
        "selected_tet_id_sample_1based": (selected[:16] + 1).tolist(),
        "total_candidate_edges": int(partition["total_candidate_edges"]),
        "excluded_outer_edges": int(partition["excluded_outer_edges_count"]),
        "included_interior_edges": int(partition["included_interior_edges_count"]),
        "selected_tets_with_zero_remaining_marked_edges": int(
            partition["selected_tets_with_zero_remaining_marked_edges"]
        ),
    }


def _heart_plus_affected_region_metrics(
    old_nodes: np.ndarray,
    old_tets: np.ndarray,
    old_labels: np.ndarray,
    new_nodes: np.ndarray,
    new_tets: np.ndarray,
    new_labels: np.ndarray,
    parent_tet_id: np.ndarray,
    included_edges: np.ndarray,
) -> dict:
    affected = _base._incident_tet_mask(old_tets, included_edges, len(old_nodes))
    region = affected | (old_labels == 3)
    region_parent_ids = np.flatnonzero(region)
    new_region = np.isin(
        np.asarray(parent_tet_id, dtype=np.int64) - 1, region_parent_ids
    )

    old_region_tets = old_tets[region]
    new_region_tets = new_tets[new_region]
    old_region_volumes = _base._signed_tetra_volumes(old_nodes, old_region_tets)
    new_region_volumes = _base._signed_tetra_volumes(new_nodes, new_region_tets)
    return {
        "parent_tetrahedra": int(np.count_nonzero(region)),
        "child_tetrahedra": int(np.count_nonzero(new_region)),
        "volume_by_tissue_id_m3": {
            "before": _base._volume_by_label(
                old_region_volumes, old_labels[region]
            ),
            "after": _base._volume_by_label(
                new_region_volumes, new_labels[new_region]
            ),
        },
        "tetra_quality_distribution": {
            "before": _base._distribution(
                _base._tetra_quality(old_nodes, old_region_tets)
            ),
            "after": _base._distribution(
                _base._tetra_quality(new_nodes, new_region_tets)
            ),
        },
        "edge_length_mm": {
            "before": _base._edge_length_distribution(old_nodes, old_region_tets),
            "after": _base._edge_length_distribution(new_nodes, new_region_tets),
        },
    }


def refine_heart_mesh(
    nodes_m: np.ndarray,
    tets_zero_based: np.ndarray,
    boundary_zero_based: np.ndarray,
    tissue_id: np.ndarray,
    candidates: dict,
    collar_mm: float = 15.0,
    max_volume_fraction: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Refine all interior candidate edges while preserving the original surface."""
    nodes = np.asarray(nodes_m, dtype=np.float64)
    tets = np.asarray(tets_zero_based, dtype=np.int64)
    boundary = np.asarray(boundary_zero_based, dtype=np.int64)
    labels = np.asarray(tissue_id).reshape(-1).astype(np.int64, copy=False)
    _base._validate_mesh_arrays(nodes, tets, boundary, labels)

    boxes = _base.build_candidate_boxes(
        nodes, tets, labels, candidates, max_volume_fraction
    )
    selected = _base.select_tetrahedra_by_vertex_aabb(
        nodes, tets, boxes, collar_mm
    )
    partition = select_interior_marked_edges(
        tets, boundary, selected, len(nodes)
    )
    _base._require(
        partition["included_interior_edges_count"] > 0,
        "No interior candidate edges remain after explicit boundary-edge exclusion",
    )

    (
        new_nodes,
        new_tets,
        new_boundary,
        new_labels,
        parent_tet_id,
        qc,
    ) = _base.refine_marked_edges(
        nodes,
        tets,
        boundary,
        labels,
        partition["included_interior_edges"],
    )

    checks = dict(qc["checks"])
    checks.update(
        {
            "heart_volume_centroid_scaling_explicit": True,
            "vertex_aabb_selection_used": True,
            "marked_edges_are_interior": True,
            "quality_and_local_edge_qc_recorded": True,
            "boundary_edges_excluded_by_policy": True,
            "all_included_interior_edges_refined_globally": True,
        }
    )
    selection = _selection_record(partition, selected, collar_mm)
    metrics = dict(qc["metrics"])
    metrics["selection"] = selection
    metrics["selection_policy"] = _selection_policy(collar_mm)
    metrics["heart_plus_affected_region"] = _heart_plus_affected_region_metrics(
        nodes,
        tets,
        labels,
        new_nodes,
        new_tets,
        new_labels,
        parent_tet_id,
        partition["included_interior_edges"],
    )
    metrics["scope"] = {
        "kind": "interior_only",
        "surface_contact_mesh_unchanged": True,
        "complete_global_spatial_convergence_proof": False,
    }
    return (
        new_nodes,
        new_tets,
        new_boundary,
        new_labels,
        parent_tet_id,
        {
            "candidate_boxes": _serialize_candidate_boxes(boxes),
            "checks": checks,
            "metrics": metrics,
            "selection": selection,
        },
    )


def _decorate_report(
    output_dir: Path,
    report: dict,
    collar_mm: float,
    selection: dict | None = None,
) -> dict:
    result = dict(report)
    result["provenance"] = provenance()
    result["selection_policy"] = _selection_policy(collar_mm)
    if selection is not None:
        result["selection"] = selection
    result["refinement_scope"] = {
        "kind": "interior_only",
        "surface_contact_mesh_unchanged": True,
        "complete_global_spatial_convergence_proof": False,
        "statement": (
            "This refines the interior field only. The unchanged surface/contact "
            "mesh does not constitute a complete global spatial-convergence proof."
        ),
    }
    report_path = output_dir / "refinement_report.json"
    report_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + chr(10),
        encoding="utf-8",
    )
    return result


def run(
    source_prepared: Path,
    candidate_parameters: Path,
    output_dir: Path,
    collar_mm: float = 15.0,
    max_volume_fraction: float = 0.01,
) -> dict:
    """Run the bounded interior refinement and write compatible artifacts."""
    source = Path(source_prepared).resolve()
    candidate_path = Path(candidate_parameters).resolve()
    output = Path(output_dir).resolve()
    _base._require(
        not output.exists(),
        f"Refusing to overwrite existing output directory: {output}",
    )
    try:
        _base._require(source.is_file(), f"Missing source-prepared file: {source}")
        _base._require(
            candidate_path.is_file(),
            f"Missing candidate-parameters file: {candidate_path}",
        )
        nodes, tets, boundary, labels = _base.load_prepared_mesh(source)
        candidates = _base._read_candidates(candidate_path)
        (
            new_nodes,
            new_tets,
            new_boundary,
            new_labels,
            parent_tet_id,
            audit,
        ) = refine_heart_mesh(
            nodes,
            tets,
            boundary,
            labels,
            candidates,
            collar_mm=collar_mm,
            max_volume_fraction=max_volume_fraction,
        )
        report = _base._write_output(
            output,
            source,
            candidate_path,
            new_nodes,
            new_tets,
            new_boundary,
            new_labels,
            parent_tet_id,
            audit,
        )
        return _decorate_report(
            output,
            report,
            collar_mm,
            selection=audit["selection"],
        )
    except Exception as error:
        report = _base._write_failure_report(
            output, source, candidate_path, error
        )
        return _decorate_report(output, report, collar_mm)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-prepared", required=True, type=Path)
    parser.add_argument("--candidate-parameters", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--collar-mm", type=float, default=15.0)
    parser.add_argument("--max-volume-fraction", type=float, default=0.01)
    args = parser.parse_args(argv)
    report = run(
        args.source_prepared,
        args.candidate_parameters,
        args.output_dir,
        collar_mm=args.collar_mm,
        max_volume_fraction=args.max_volume_fraction,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if report["status"] != "passed_heart_volume_refinement":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
