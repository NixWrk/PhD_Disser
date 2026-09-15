"""Static TEPC heart replacement diagnostics; never assemble or solve a FEM model.

Example (from MATLAB_TRKG4_real_subjects, using an existing dependency directory)::

    python -B tools/prepare_static_heart_replacement_pilot.py \
        --manifest output/exploratory/tepc_preparation_20260911/manifest.json \
        --output-dir output/exploratory/static_heart_replacement_20260915 \
        --deps output/exploratory/arm_sigma_20260908/python_solver_deps

Only the current four-tissue, five-block, SI MATLAB export is accepted. Candidate
labels are separate, element-ordered NumPy arrays; existing blocks are NOT valid
for a replacement. Centroid rasterisation is diagnostic, not boundary validation.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys


RULE_ID = "static_background_heart_overlay_v1"
TISSUES = ["soft_tissue", "lungs", "heart", "bones"]
BLOCKS = {"soft": [0, 1], "heart": [2], "lung": [3], "bone": [4]}
MEMBERSHIP_TOL = 1e-12


def initialize(deps=None):
    # Limits apply only to this process; do not compete with ongoing FEM jobs.
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[key] = "1"
    if deps:
        sys.path.insert(0, str(Path(deps).resolve()))
    global np, h5py
    import numpy as np
    import h5py


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def integer_array(values, low, high, name):
    values = np.asarray(values)
    if (not np.isfinite(values).all() or np.any(values != np.floor(values))
            or np.any(values < low) or np.any(values > high)):
        raise ValueError(f"Invalid integer range in {name}")
    return values.astype(np.int64)


def validate_labels(values, names, parameters, block_count):
    if list(names) != TISSUES:
        raise ValueError("Expected ordered soft_tissue, lungs, heart, bones contract")
    mapping = {p["id"]: p["block_indices"] for p in parameters}
    if len(parameters) != 4 or mapping != BLOCKS or block_count != 5:
        raise ValueError("TEPC five-block mapping does not match four tissue IDs")
    labels = integer_array(values, 1, len(names), "tissue_id").ravel().astype(np.uint8)
    if set(np.unique(labels)) != set(range(1, 5)):
        raise ValueError("All four source tissues must be present")
    return labels, {name: names.index(name) + 1 for name in TISSUES}


def make_background(original, ids):
    background = original.copy()
    background[original == ids["heart"]] = ids["soft_tissue"]
    background.flags.writeable = False
    return background


def overlay(background, heart_mask, heart_id):
    if heart_mask.dtype != np.bool_ or heart_mask.shape != background.shape:
        raise ValueError("Heart mask must be boolean and element-aligned")
    labels = background.copy()
    labels[heart_mask] = heart_id
    return labels


def tetra_geometry(vertices):
    edges = vertices[:, 1:, :] - vertices[:, :1, :]
    determinant = np.einsum("ni,ni->n", np.cross(edges[:, 0], edges[:, 1]), edges[:, 2])
    scale = np.prod(np.linalg.norm(edges, axis=2), axis=1)
    return determinant, scale, vertices.mean(axis=1)


def exact_moments(vertices):
    """Return volume, centre and covariance of disjoint uniform positive tets."""
    determinant, scale, centres = tetra_geometry(vertices)
    volumes = determinant / 6
    if not np.isfinite(volumes).all() or np.any(volumes <= 128 * np.finfo(float).eps * scale):
        raise ValueError("Nonpositive or numerically degenerate tetrahedra")
    total = volumes.sum()
    anchor = centres[0]
    centre = anchor + np.sum(volumes[:, None] * (centres - anchor), axis=0) / total
    delta = vertices - centres[:, None, :]
    within = np.einsum("nvi,nvj->nij", delta, delta) / 20
    between = centres - centre
    covariance = (np.einsum("n,nij->ij", volumes, within)
                  + np.einsum("n,ni,nj->ij", volumes, between, between)) / total
    return float(total), centre, covariance


def merge_moments(left, right):
    if left is None:
        return right
    v1, c1, cov1 = left
    v2, c2, cov2 = right
    volume = v1 + v2
    delta = c2 - c1
    centre = c1 + delta * (v2 / volume)
    covariance = ((v1 * cov1 + v2 * cov2) / volume
                  + np.outer(delta, delta) * (v1 / volume) * (v2 / volume))
    return volume, centre, covariance


def candidates_from_moments(moments):
    volume, centre, covariance = moments
    eigenvalues, frame = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, frame = eigenvalues[order], frame[:, order]
    if np.any(eigenvalues <= 0):
        raise ValueError("Heart covariance is not positive definite")
    if np.linalg.det(frame) < 0:
        frame[:, -1] *= -1
    raw_axes = np.sqrt(5 * eigenvalues)
    raw_volume = 4 * np.pi * np.prod(raw_axes) / 3
    factor = (volume / raw_volume) ** (1 / 3)
    radius = (3 * volume / (4 * np.pi)) ** (1 / 3)
    common = {"centre_m": centre.tolist(), "analytic_volume_ml": volume * 1e6}
    return {
        "sphere": {**common, "semiaxes_m": [radius] * 3, "axes_columns": np.eye(3).tolist(),
                   "radius_m": radius, "definition": "equal volume; volume centroid"},
        "ellipsoid": {**common, "semiaxes_m": (raw_axes * factor).tolist(),
                      "axes_columns": frame.tolist(), "covariance_eigenvalues_m2": eigenvalues.tolist(),
                      "raw_moment_semiaxes_m": raw_axes.tolist(),
                      "raw_moment_volume_ml": float(raw_volume * 1e6),
                      "equal_volume_scale": float(factor),
                      "definition": "sqrt(5*eigenvalues(C)), then uniform equal-volume rescale",
                      "covariance_preserved_after_rescale": bool(abs(factor - 1) < 1e-12)},
    }


def normalised(points, candidate):
    return ((points - np.array(candidate["centre_m"])) @ np.array(candidate["axes_columns"])
            / np.array(candidate["semiaxes_m"]))


def contains(points, candidate):
    q = normalised(points, candidate)
    return np.sum(q * q, axis=-1) <= 1 + MEMBERSHIP_TOL


def triangle_origin_distance_squared(triangles):
    """Exact Euclidean distance to filled triangles (edge and face interiors)."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    distance = np.full(len(a), np.inf)
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        denom = np.sum(edge * edge, axis=1)
        ratio = np.divide(-np.sum(start * edge, axis=1), denom,
                          out=np.zeros_like(denom), where=denom > 0)
        closest = start + np.clip(ratio, 0, 1)[:, None] * edge
        distance = np.minimum(distance, np.sum(closest * closest, axis=1))
    e1, e2 = b - a, c - a
    n = np.cross(e1, e2)
    nn = np.sum(n * n, axis=1)
    if np.any(nn == 0) or not np.isfinite(nn).all():
        raise ValueError("Degenerate declared boundary triangle")
    na = np.sum(n * a, axis=1)
    projection = n * (na / nn)[:, None]
    relative = projection - a
    s = np.einsum("ni,ni->n", np.cross(relative, e2), n) / nn
    t = np.einsum("ni,ni->n", np.cross(e1, relative), n) / nn
    face = (s >= -MEMBERSHIP_TOL) & (t >= -MEMBERSHIP_TOL) & (s + t <= 1 + MEMBERSHIP_TOL)
    distance[face] = np.minimum(distance[face], na[face] ** 2 / nn[face])
    return distance


def point_in_tets(point, vertices):
    selected = np.all(point >= vertices.min(axis=1) - 1e-12, axis=1)
    selected &= np.all(point <= vertices.max(axis=1) + 1e-12, axis=1)
    v = vertices[selected]
    if not len(v):
        return False
    matrix = np.swapaxes(v[:, 1:] - v[:, :1], 1, 2)
    rhs = np.broadcast_to(point, v[:, 0].shape) - v[:, 0]
    bary = np.linalg.solve(matrix, rhs[..., None])[..., 0]
    return bool(np.any(np.all(bary >= -MEMBERSHIP_TOL, axis=1)
                       & (bary.sum(axis=1) <= 1 + MEMBERSHIP_TOL)))


def matlab_class(obj, expected):
    actual = obj.attrs.get("MATLAB_class", b"")
    if isinstance(actual, bytes):
        actual = actual.decode("ascii")
    if actual != expected:
        raise ValueError(f"MATLAB_class mismatch for {obj.name}: expected {expected}")


def matlab_text(obj):
    matlab_class(obj, "char")
    return "".join(chr(int(x)) for x in obj[:].ravel())


def read_metadata(f, manifest, export):
    model = f["fmdl_m"]
    matlab_class(model, "struct")
    for name in ("nodes", "elems", "boundary"):
        matlab_class(model[name], "double")
    for name in ("tissue_names", "blocks"):
        matlab_class(f[name], "cell")
    matlab_class(f["tissue_id"], "double")
    if matlab_text(f["coordinate_units"]) != "m":
        raise ValueError("Expected SI geometry in metres")
    names = [matlab_text(f[ref]) for ref in f["tissue_names"][:].ravel()]
    labels, ids = validate_labels(f["tissue_id"][:], names, manifest["parameters"], f["blocks"].size)
    shapes = {name: tuple(model[name].shape) for name in ("nodes", "elems", "boundary")}
    for name, width in (("nodes", 3), ("elems", 4), ("boundary", 3)):
        if len(shapes[name]) != 2 or shapes[name][0] != width or shapes[name][1] == 0:
            raise ValueError(f"Unexpected MATLAB array shape: {name}")
    counts = {"nodes": shapes["nodes"][1], "elements": shapes["elems"][1],
              "boundary_faces": shapes["boundary"][1]}
    if len(labels) != counts["elements"] or export["counts"] != counts:
        raise ValueError("Export counts do not match geometry and labels")
    groups = integer_array(export["group_counts"], 1, counts["elements"], "group_counts")
    if groups.shape != (5,):
        raise ValueError("Expected five group counts")
    tissue_counts = np.bincount(labels, minlength=5)[1:]
    if not np.array_equal(tissue_counts, [groups[0] + groups[1], groups[3], groups[2], groups[4]]):
        raise ValueError("Tissue counts disagree with exported material block counts")
    electrodes = model["electrode"]
    matlab_class(electrodes, "struct")
    if electrodes["nodes"].size != 4 or electrodes["z_contact"].size != 4:
        raise ValueError("Expected four TEPC CEM electrodes")
    ground = integer_array(model["gnd_node"][:], 1, counts["nodes"], "gnd_node")
    if ground.size != 1:
        raise ValueError("Expected one ground node")
    # CEM blocks include four electrode potential DOFs, before ground reduction.
    system_size = counts["nodes"] + 4
    for ref in f["blocks"][:].ravel():
        block = f[ref]
        matlab_class(block, "double")
        if (not isinstance(block, h5py.Group) or int(block.attrs.get("MATLAB_sparse", -1)) != system_size
                or block["jc"].size != system_size + 1):
            raise ValueError("Unexpected sparse block dimensions")
    nodes = model["nodes"][:].T
    if not np.isfinite(nodes).all():
        raise ValueError("Nonfinite nodes")
    return nodes, labels, ids, {"counts": counts, "tissue_names": names,
                                "tissue_ids": ids, "block_indices_zero_based": BLOCKS,
                                "matlab_class_tags_checked": True, "coordinate_units": "m",
                                "cem_electrodes": 4, "cem_system_size_before_ground_reduction": system_size,
                                "ground_node_matlab_one_based": int(ground.ravel()[0])}


def chunks(dataset, width, node_count, chunk_size):
    for start in range(0, dataset.shape[1], chunk_size):
        stop = min(start + chunk_size, dataset.shape[1])
        values = integer_array(dataset[:, start:stop].T, 1, node_count, dataset.name) - 1
        if values.shape != (stop - start, width):
            raise ValueError("Connectivity shape mismatch")
        yield slice(start, stop), values


def overlay_audit(background, original, masks, ids):
    states = {name: overlay(background, mask, ids["heart"]) for name, mask in masks.items()}
    nominal_ok = np.array_equal(states["nominal"], original)
    transitions = []
    for before in masks:
        for after in masks:
            # Recompute from the same immutable background, never from states[before].
            result = overlay(background, masks[after], ids["heart"])
            freed = masks[before] & ~masks[after]
            restored = np.array_equal(result[freed], background[freed])
            transitions.append({"from": before, "to": after,
                                "restored_to_background": bool(restored),
                                "order_independent": bool(np.array_equal(result, states[after])),
                                "freed_tet_counts_by_background": {
                                    name: int(np.count_nonzero(freed & (background == code)))
                                    for name, code in ids.items() if name != "heart"}})
    return states, {"nominal_reconstructs_original_labels": bool(nominal_ok),
                    "background_is_readonly": not background.flags.writeable,
                    "background_nonheart_labels_preserved": bool(np.array_equal(
                        background[original != ids["heart"]], original[original != ids["heart"]])),
                    "transitions": transitions,
                    "passed": bool(nominal_ok and all(t["restored_to_background"] and
                                                      t["order_independent"] for t in transitions))}


def run(manifest_path, output_dir, chunk_size=65536):
    manifest_path, output_dir = Path(manifest_path).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError("Output directory already exists; refusing to overwrite")
    if chunk_size < 1:
        raise ValueError("chunk-size must be positive")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    source = manifest["source"]
    if (manifest["schema"] != "surface_sensitivity_preparation_v1"
            or source["prepared_format"] != "si_geometry_si_blocks"):
        raise ValueError("Unsupported TEPC preparation contract")
    paths = {"manifest": manifest_path}
    hashes = {"manifest": sha256(manifest_path)}
    for key, hash_key in (("prepared_fem", "prepared_sha256"),
                          ("mesh_acceptance", "mesh_acceptance_sha256"),
                          ("refinement_export", "refinement_export_sha256")):
        paths[key] = (manifest_path.parent / source[key]).resolve()
        hashes[key] = sha256(paths[key])
        if hashes[key] != source[hash_key]:
            raise ValueError(f"Manifest hash mismatch: {key}")
    export = json.loads(paths["refinement_export"].read_text(encoding="utf-8-sig"))
    if export["prepared_sha256"] != hashes["prepared_fem"] or export["status"] != "passed":
        raise ValueError("Export receipt is not bound to the prepared FEM")
    acceptance = json.loads(paths["mesh_acceptance"].read_text(encoding="utf-8-sig"))
    print("Reading frozen geometry, signed Jacobians and exact heart moments", flush=True)
    with h5py.File(paths["prepared_fem"], "r") as f:
        nodes, original, ids, metadata = read_metadata(f, manifest, export)
        model = f["fmdl_m"]
        n = len(original)
        volumes = np.empty(n)
        moments = None
        qc = {"tetrahedra": n, "negative": 0, "zero": 0, "numerically_degenerate_positive": 0,
              "nonfinite": 0, "min_signed_jacobian_m3": float("inf"),
              "max_signed_jacobian_m3": float("-inf"),
              "criterion": "signed_volume=det(J)/6 > 128*eps*product(norm(v_i-v_0)), i=1..3",
              "orientation_repaired": False}
        for sl, connectivity in chunks(model["elems"], 4, len(nodes), chunk_size):
            vertices = nodes[connectivity]
            det, scale, _ = tetra_geometry(vertices)
            v = det / 6
            volumes[sl] = v
            qc["negative"] += int(np.count_nonzero(det < 0))
            qc["zero"] += int(np.count_nonzero(det == 0))
            qc["nonfinite"] += int(np.count_nonzero(~np.isfinite(det)))
            qc["numerically_degenerate_positive"] += int(np.count_nonzero(
                (v > 0) & (v <= 128 * np.finfo(float).eps * scale)))
            qc["min_signed_jacobian_m3"] = min(qc["min_signed_jacobian_m3"], float(np.min(det)))
            qc["max_signed_jacobian_m3"] = max(qc["max_signed_jacobian_m3"], float(np.max(det)))
            heart = original[sl] == ids["heart"]
            if heart.any() and np.all(v[heart] > 128 * np.finfo(float).eps * scale[heart]):
                moments = merge_moments(moments, exact_moments(vertices[heart]))
        qc["passed"] = not any(qc[k] for k in ("negative", "zero", "nonfinite", "numerically_degenerate_positive"))
        if not qc["passed"]:
            output_dir.mkdir(parents=True, exist_ok=False)
            write_json(output_dir / "input_jacobian_qc.json", qc)
            raise ValueError("Input Jacobian QC failed; no candidate labels prepared")
        candidates = candidates_from_moments(moments)
        masks = {"nominal": original == ids["heart"],
                 **{name: np.zeros(n, dtype=bool) for name in candidates}}
        centre_inside = False
        print("Rasterising candidates on existing tetrahedron centroids", flush=True)
        for sl, connectivity in chunks(model["elems"], 4, len(nodes), chunk_size):
            vertices = nodes[connectivity]
            centres = vertices.mean(axis=1)
            for name, candidate in candidates.items():
                masks[name][sl] = contains(centres, candidate)
            if not centre_inside:
                centre_inside = point_in_tets(moments[1], vertices)
        boundary_qc = {name: {"intersected_or_touched_declared_boundary_faces": 0,
                              "min_normalised_boundary_distance": float("inf"),
                              "centre_inside_tet_domain": centre_inside} for name in candidates}
        for _, connectivity in chunks(model["boundary"], 3, len(nodes), chunk_size):
            triangles = nodes[connectivity]
            for name, candidate in candidates.items():
                distance = triangle_origin_distance_squared(normalised(triangles, candidate))
                q = boundary_qc[name]
                q["intersected_or_touched_declared_boundary_faces"] += int(np.count_nonzero(distance <= 1 + MEMBERSHIP_TOL))
                q["min_normalised_boundary_distance"] = min(q["min_normalised_boundary_distance"], float(np.sqrt(distance.min())))
        for q in boundary_qc.values():
            q["external_boundary_conflict"] = bool(not centre_inside or q["intersected_or_touched_declared_boundary_faces"])
            q["scope"] = "Analytic candidate vs declared mesh boundary triangles; boundary completeness inherited, not rebuilt"
            q["outside_volume_measured"] = False
    background = make_background(original, ids)
    states, audit = overlay_audit(background, original, masks, ids)
    if not audit["passed"]:
        raise ValueError("Immutable background overlay audit failed")
    unchanged = all(sha256(path) == hashes[key] for key, path in paths.items())
    if not unchanged:
        raise ValueError("A source changed during preflight; no candidate labels saved")
    output_dir.mkdir(parents=True, exist_ok=False)
    np.save(output_dir / "background_labels.npy", background, allow_pickle=False)
    background_hash = sha256(output_dir / "background_labels.npy")
    source_records = {key: {"path": Path(os.path.relpath(path, output_dir)).as_posix(), "sha256": hashes[key]}
                      for key, path in paths.items()}
    rule = {"id": RULE_ID, "origin": "explicit_user_decision_2026-09-15",
            "background_variant": "preserve_nonheart_materials",
            "definition": "b=original.copy(); b[original==heart]=soft; labels=b.copy(); labels[H]=heart",
            "bone_policy": "preserve", "heart_priority_over_all_background": True,
            "lung_overlap_allowed": True, "bone_overlap_allowed": True,
            "vacated_region_restored_from": "immutable_background_not_previous_state",
            "original_heart_background_assumption": "soft; no latent lung/bone mask inferred under original heart",
            "model_type": "material overlay, not mechanical deformation",
            "lung_geometry_moved": False,
            "electrical_lung_volume": "background lung volume minus volume selected by heart overlay"}
    parameters = {"original_heart": {"volume_ml": moments[0] * 1e6, "centre_m": moments[1].tolist(),
                                    "covariance_m2": moments[2].tolist(), "tetrahedra": int(masks["nominal"].sum()),
                                    "definition": "whole effective heart region in static FEM, not blood chambers or 4D masks",
                                    "within_tet_covariance": "sum_i outer(v_i-c_tet, v_i-c_tet)/20"},
                  "candidates": candidates}
    occupancy, balance, material_volumes = [], [], []
    contracts = {}
    original_volume = float(volumes[masks["nominal"]].sum() * 1e6)
    for name, labels in states.items():
        mask = masks[name]
        occupied = float(volumes[mask].sum() * 1e6)
        for basis, basis_labels in (("original", original), ("immutable_background", background)):
            for tissue, code in ids.items():
                compartment = basis_labels == code
                selected = compartment & mask
                occupancy.append({"candidate": name, "basis": basis, "compartment": tissue,
                                  "compartment_volume_ml": float(volumes[compartment].sum() * 1e6),
                                  "occupied_tets": int(selected.sum()),
                                  "occupied_volume_ml": float(volumes[selected].sum() * 1e6)})
        for tissue, code in ids.items():
            material_volumes.append({"candidate": name, "tissue": tissue,
                                     "tetrahedra": int(np.count_nonzero(labels == code)),
                                     "electrical_material_volume_ml": float(volumes[labels == code].sum() * 1e6)})
        retained = float(volumes[mask & masks["nominal"]].sum() * 1e6)
        balance.append({"candidate": name, "original_heart_volume_ml": original_volume,
                        "analytic_target_volume_ml": original_volume, "centroid_selected_volume_ml": occupied,
                        "retained_original_heart_ml": retained,
                        "removed_original_heart_ml": float(volumes[~mask & masks["nominal"]].sum() * 1e6),
                        "newly_occupied_nonheart_ml": float(volumes[mask & ~masks["nominal"]].sum() * 1e6),
                        "raster_minus_analytic_ml_not_outside_volume": occupied - original_volume})
        state_dir = output_dir / name
        state_dir.mkdir()
        np.save(state_dir / "labels.npy", labels, allow_pickle=False)
        contract = {"schema": "static_heart_candidate_labels_v1", "candidate": name,
                    "rule": rule, "labels": "labels.npy", "labels_sha256": sha256(state_dir / "labels.npy"),
                    "background": "../background_labels.npy", "background_sha256": background_hash,
                    "source_prepared_sha256": hashes["prepared_fem"], "source_manifest_sha256": hashes["manifest"],
                    "element_order": "fmdl_m.elems MATLAB rows, unchanged; labels.npy is 1D in that order",
                    "shape": list(labels.shape), "dtype": str(labels.dtype), "tissue_ids": ids,
                    "candidate_parameters": "../candidate_parameters.json", "mask_definition": (
                        "original tissue_id == heart" if name == "nominal" else "candidate contains existing tetrahedron centroid"),
                    "membership_squared_radius_tolerance": MEMBERSHIP_TOL,
                    "geometry_qc": boundary_qc.get(name, {"scope": "Nominal labelled subset of unchanged input mesh"}),
                    "lung_coverage_is_rejection": False, "bone_coverage_is_rejection": False,
                    "analytic_candidate_clipped": False, "labels_exist_only_on_input_tetrahedra": True,
                    "source_material_blocks_modified": False, "reuse_source_blocks_for_replacement": False,
                    "solver_ready": False, "required_before_solver": [
                        "assemble material blocks for these labels on the bound unchanged mesh",
                        "retain and verify TEPC contact/ground contracts separately",
                        "resolve any external boundary QC conflict; quantify numerical discretisation"],
                    "anatomical_validation": False, "electrical_equivalence_tested": False}
        write_json(state_dir / "contract.json", contract)
        contracts[name] = {"path": f"{name}/contract.json", "sha256": sha256(state_dir / "contract.json")}
    write_json(output_dir / "candidate_parameters.json", parameters)
    write_json(output_dir / "input_jacobian_qc.json", qc)
    write_csv(output_dir / "compartment_occupancy.csv", occupancy)
    write_csv(output_dir / "heart_replacement_balance.csv", balance)
    write_csv(output_dir / "material_volumes.csv", material_volumes)
    write_json(output_dir / "overlay_audit.json", audit)
    result = {"schema": "static_heart_replacement_preflight_v1", "status": "diagnostic_labels_prepared_not_validated",
              "rule": rule, "sources": source_records, "sources_unchanged_after_read": unchanged,
              "implementation_sha256": sha256(__file__), "numpy_version": np.__version__, "h5py_version": h5py.__version__,
              "metadata": metadata, "background": {"path": "background_labels.npy", "sha256": background_hash},
              "candidate_contracts": contracts, "input_jacobian_qc_passed": qc["passed"],
              "overlay_audit_passed": audit["passed"], "boundary_qc": boundary_qc,
              "source_acceptance_status": acceptance.get("status"),
              "computed_background_variants": ["preserve_nonheart_materials"],
              "not_computed": ["4D registration or dynamics", "new mesh", "material matrices", "FEM solve", "Z or delta Z"],
              "solver_ready": False, "physical_validation": False, "anatomical_validation": False,
              "limitations": [
                  "Uniform-tetrahedron moments are exact for the labelled static mesh, not for continuous anatomy.",
                  "Whole-heart effective material is not a chamber blood-volume target.",
                  "Full tet volumes selected at centroids form a stepped diagnostic boundary; partial intersections are not integrated.",
                  "Analytic equal volume does not imply equal rasterised volume; the difference is not measured outside-body volume.",
                  "Lung and bone coverage changes electrical material volume without mechanical displacement.",
                  "Boundary triangle intersection and centre containment use the declared boundary; topology is not revalidated.",
                  "The frozen 20260911 preparation is an explicit baseline, not the running volume convergence levels."]}
    write_json(output_dir / "preflight.json", result)
    print(json.dumps({"status": result["status"], "heart_volume_ml": original_volume,
                      "overlay_audit_passed": audit["passed"], "boundary_qc": boundary_qc}, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="Must not exist; never overwritten")
    parser.add_argument("--deps", type=Path, help="Optional existing numpy/h5py dependency directory")
    parser.add_argument("--chunk-size", type=int, default=65536)
    args = parser.parse_args()
    initialize(args.deps)
    run(args.manifest, args.output_dir, args.chunk_size)


if __name__ == "__main__":
    main()
