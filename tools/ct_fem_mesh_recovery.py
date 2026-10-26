"""Bounded mesh-quality feedback to local surface conditioning.

Only boundary-local failing tetrahedra guide the next surface pass. The
original geometry/contact budgets and the final volume-mesh QC stay fixed.
No subject names, fitted coordinates, tissue parameters or mesh promotion.
"""
from pathlib import Path

import numpy as np
import pymeshlab
import trimesh

import ct_fem_core as core
from ct_fem_remesh import check_yz_bounds, select_region, update_contacts
from ct_fem_tetgen import generate


def validate_settings(value):
    names = {"maximum_rounds", "radius_mm", "target_edge_mm", "iterations",
             "maximum_bad_elements", "maximum_selected_faces", "maximum_boundary_distance_mm"}
    if set(value) != names:
        raise ValueError("Declare the complete bounded surface-recovery policy")
    rounds = float(value["maximum_rounds"])
    if not np.isfinite(rounds) or rounds < 0 or rounds != int(rounds):
        raise ValueError("Recovery rounds must be a nonnegative integer")
    for name in names-{"maximum_rounds"}:
        (core.natural if name in {"iterations", "maximum_bad_elements", "maximum_selected_faces"}
         else core.positive)(value[name], "quality_recovery."+name)
    if value["radius_mm"] < value["target_edge_mm"]:
        raise ValueError("Recovery radius must cover at least one target edge")
    return value


def failing_elements(nodes, tets, kernel):
    failed = []
    for start in range(0, len(tets), 100000):
        points = nodes[tets[start:start+100000]]
        ratio = kernel["mean_ratio"](points)
        condition = np.linalg.cond(points[:, 1:]-points[:, :1])
        mask = ((ratio < core.QUALITY_LIMITS["minimum_mean_ratio"]) |
                (condition > core.QUALITY_LIMITS["maximum_edge_condition"]))
        failed.extend((start+np.flatnonzero(mask)).tolist())
    return np.asarray(failed, dtype=np.int64)


def conditioned_surface(cfg, source, current_file, original, contacts, nodes, tets, target, kernel):
    value = validate_settings(cfg["mesh"]["quality_recovery"])
    surface = trimesh.load(current_file, force="mesh")
    bad = failing_elements(nodes, tets, kernel)
    if not len(bad) or len(bad) > value["maximum_bad_elements"]:
        raise ValueError("Failing-element count exceeds the declared local-recovery scope")
    centres = nodes[tets[bad]].mean(axis=1)
    guides, distance, faces = surface.nearest.on_surface(centres)
    vertex_guides, vertex_distance, _ = surface.nearest.on_surface(nodes[tets[bad]].reshape(-1,3))
    vertex_distance = vertex_distance.reshape(-1,4)
    boundary_local = np.min(vertex_distance,axis=1) <= value["maximum_boundary_distance_mm"]
    core.write_new(target / "guides.local.json", dict(element_ids_zero_based=bad.tolist(),
        centres_lps_mm=centres.tolist(), projected_guides_lps_mm=guides.tolist(),
        boundary_distance_mm=distance.tolist(), surface_face_ids_zero_based=faces.tolist(),
        vertex_boundary_distances_mm=vertex_distance.tolist(),
        locality_rule="At least one tetrahedron vertex lies in the declared boundary neighbourhood; centroid alone is insufficient for curved boundary faces"))
    if not boundary_local.all():
        raise ValueError("Failing tetrahedra are not confined to the permitted boundary neighbourhood")
    near_vertices = vertex_guides[vertex_distance.ravel() <= value["maximum_boundary_distance_mm"]]
    guides = np.unique(np.vstack([guides,near_vertices]),axis=0)
    selected = select_region(surface, guides, value["radius_mm"], [])
    if not selected.any() or selected.sum() > value["maximum_selected_faces"]:
        raise ValueError("Surface-recovery selection exceeds its declared resource scope")
    options = dict(iterations=value["iterations"], adaptive=False, selectedonly=True,
        featuredeg=cfg["remeshing"]["feature_angle_deg"], checksurfdist=True,
        splitflag=True, collapseflag=True, swapflag=True, smoothflag=True, reprojectflag=True)
    core.write_new(target / "conditioning_settings.local.json", dict(policy=value, options=options,
        target_edge_mm=value["target_edge_mm"],
        local_operation_distance_mm=cfg["remeshing"]["per_operation_distance_mm"],
        source_surface_sha256=core.sha(current_file), selected_faces=int(selected.sum()),
        selection_rule="triangle centroids within declared radius of projected failing-tetrahedron centres",
        source_configuration_is_unchanged=True, automatic_tolerance_relaxation=False))
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(surface.vertices, surface.faces.astype(np.int32),
                              f_scalar_array=selected.astype(float)))
    ms.compute_selection_by_condition_per_face(condselect="fq > 0.5")
    ms.meshing_isotropic_explicit_remeshing(**options,
        targetlen=pymeshlab.PureValue(value["target_edge_mm"]),
        maxsurfdist=pymeshlab.PureValue(cfg["remeshing"]["per_operation_distance_mm"]))
    mesh = ms.current_mesh()
    candidate = trimesh.Trimesh(mesh.vertex_matrix(), mesh.face_matrix(), process=False)
    file = target / "exterior.stl"; candidate.export(file)
    candidate = trimesh.load(file, force="mesh")
    qc = core.surface_qc(candidate)
    if candidate.euler_number != original.euler_number:
        raise ValueError("Local conditioning changed the original surface topology")
    if len(candidate.faces) > cfg["remeshing"]["max_faces"] or len(candidate.vertices) > cfg["remeshing"]["max_vertices"]:
        raise ValueError("Local conditioning exceeded the declared surface resource limits")
    body_qc = core.read(source / "surface_qc.local.json")
    if body_qc.get("synthetic_cylinders"):
        check_yz_bounds(candidate, body_qc["skin_bounds_mm"], cfg["remeshing"]["maximum_yz_extension_mm"])
    samples = cfg["surface_qc"]["sample_count"]
    distances = dict(original_to_candidate=kernel["sampled_distance"](original, candidate, samples),
                     candidate_to_original=kernel["sampled_distance"](candidate, original, samples))
    error = float(abs(candidate.volume/original.volume-1))
    passed = (error <= cfg["mesh"]["maximum_relative_volume_error"] and
              max(d["max_mm"] for d in distances.values()) <= cfg["mesh"]["maximum_sampled_surface_error_mm"])
    qc.update(passed=bool(passed), sampled_distances=distances, relative_volume_error=error,
        original_exterior_is_reference=True, source_surface_sha256=core.sha(current_file),
        exported_surface_sha256=core.sha(file), smoothing_is_local_and_reprojected=True,
        sampled_distance_is_not_certified_Hausdorff_bound=True)
    core.write_new(target / "surface_qc.local.json", qc)
    if not passed:
        raise ValueError("Locally conditioned surface exceeds the original geometry budget")
    # Compare each round to the initial contact metadata, never accumulate
    # a fresh one-millimetre allowance per correction.
    projected = update_contacts(candidate, contacts, cfg, cfg["remeshing"])
    projected.update(source_projection_surface_path=contacts["projection_surface_path"],
        source_projection_surface_sha256=contacts["projection_surface_sha256"],
        projection_surface_path=str(file), projection_surface_sha256=core.sha(file),
        projection_surface_role="quality_recovery_computational_exterior")
    core.write_new(target / "contacts.local.json", projected)
    return file, candidate, projected


def generate_screened(cfg, source, out, surface):
    value = validate_settings(cfg["mesh"]["quality_recovery"])
    kernel = core.kernels(); contacts = core.read(source / "contacts.local.json")
    original_file = Path(contacts.get("original_projection_surface_path", contacts["projection_surface_path"]))
    digest = contacts.get("original_projection_surface_sha256", contacts["projection_surface_sha256"])
    if core.sha(original_file) != digest:
        raise ValueError("Original recovery reference identity changed")
    original = trimesh.load(original_file, force="mesh")
    file = source / "exterior.stl"; current_out = out; projected = None; attempts = []
    for index in range(int(value["maximum_rounds"])+1):
        nodes, tets, version = generate(file, current_out, cfg["mesh"])
        tets = kernel["orient"](nodes, tets)
        qc = kernel["quality"](nodes, tets)
        within = len(nodes) <= cfg["mesh"]["max_nodes"] and len(tets) <= cfg["mesh"]["max_tetrahedra"]
        qc.update(within_resource_limits=bool(within), passed=bool(qc["passed"] and within))
        core.write_new(current_out / "attempt_shape_qc.local.json", qc)
        attempts.append(dict(attempt=index, input_surface_sha256=core.sha(file), shape_qc=qc,
            native_output_directory=str(current_out), tetgen_version=version))
        if qc["passed"]:
            core.write_new(out / "mesh_attempts.local.json", dict(attempts=attempts,
                selected_attempt=index, recovery_rounds=index,
                scope="shape selection before final bulk and boundary checks",
                original_exterior_sha256=digest, quality_limits=core.QUALITY_LIMITS))
            return nodes, tets, version, surface, projected, index
        core.write_new(current_out / "tetgen_rejected_shape_qc.local.json", qc)
        if not within or not qc["all_positive"] or index == value["maximum_rounds"]:
            core.write_new(out / "mesh_attempts.local.json", dict(attempts=attempts,
                selected_attempt=None, recovery_rounds=index, original_exterior_sha256=digest))
            raise ValueError("Mesh fails fixed shape/resource limits after bounded local recovery")
        current_out = out / f"recovery_{index+1:02d}"; current_out.mkdir()
        try:
            file, surface, projected = conditioned_surface(cfg, source, file, original, contacts,
                nodes, tets, current_out, kernel)
        except Exception as error:
            core.write_new(out / "mesh_attempts.local.json", dict(attempts=attempts,
                selected_attempt=None, recovery_rounds=index, failed_conditioning_round=index+1,
                error=str(error), original_exterior_sha256=digest))
            raise
