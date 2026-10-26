"""Controlled surface conditioning, using the historical PyMeshLab recipe.

Regions and numerical settings are explicit. No FEM, CEM patch construction,
automatic repair, vertex smoothing, or subject-specific coordinates.
"""
from __future__ import annotations

import copy
from pathlib import Path
import shutil

import numpy as np
import pymeshlab
import trimesh

import ct_fem_core as core


def settings(cfg):
    value = cfg["remeshing"]
    for name in ["global_edge_mm", "buffer_edge_mm", "contact_edge_mm",
                 "per_operation_distance_mm", "maximum_stage_sampled_distance_mm", "maximum_sampled_distance_mm",
                 "maximum_relative_volume_error", "maximum_projection_shift_mm",
                 "maximum_yz_extension_mm"]:
        core.positive(value[name], name)
    for name in ["iterations", "sample_count", "max_faces", "max_vertices"]:
        core.natural(value[name], name)
    if not value["global_edge_mm"] >= value["buffer_edge_mm"] >= value["contact_edge_mm"]:
        raise ValueError("Remesh targets must progress from coarse to fine")
    if value["smooth_vertices"] is not False or value["adaptive"] is not False:
        raise ValueError("This recipe requires no vertex smoothing and no adaptive remeshing")
    if not 0 < float(value["feature_angle_deg"]) <= 180:
        raise ValueError("Invalid remesh feature angle")
    for name in ["split_edges", "collapse_edges", "swap_edges", "reproject"]:
        if value[name] is not True:
            raise ValueError("This version uses the declared historical split/collapse/swap/reproject recipe")
    selection = value["selection"]
    ids = selection["physical_contact_ids"]
    if not isinstance(ids, list) or len(ids) != len(set(ids)):
        raise ValueError("Explicit unique physical contact IDs are required for remesh selection")
    guides = np.asarray(selection["guides_lps_mm"], float)
    if guides.size and (guides.ndim != 2 or guides.shape[1] != 3 or not np.isfinite(guides).all()):
        raise ValueError("Remesh guides must be finite LPS/mm points")
    for name in ["buffer_radius_mm", "contact_radius_mm"]:
        core.positive(selection[name], name)
    if selection["buffer_radius_mm"] < selection["contact_radius_mm"]:
        raise ValueError("Buffer selection radius cannot be smaller than contact radius")
    for name in ["buffer_aabbs_lps_mm", "contact_aabbs_lps_mm"]:
        boxes = np.asarray(selection[name], float)
        if boxes.size and (boxes.ndim != 3 or boxes.shape[1:] != (2, 3) or
                not np.isfinite(boxes).all() or np.any(boxes[:, 0] >= boxes[:, 1])):
            raise ValueError("AABBs must be explicit finite [minimum, maximum] LPS/mm corners")
    return value


def select_region(surface, centres, radius, aabbs):
    """Historical triangle-centroid rule, with explicit spheres/AABBs."""
    points = surface.triangles_center
    selected = np.zeros(len(points), bool)
    for centre in centres:
        selected |= np.linalg.norm(points-centre, axis=1) < radius
    for low, high in aabbs:
        selected |= np.all((points >= low) & (points <= high), axis=1)
    return selected


def check_yz_bounds(surface, skin_bounds, tolerance):
    bounds = np.asarray(skin_bounds, float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError("Synthetic cylinders require explicit corrected-skin bounds")
    if (np.any(surface.bounds[0, 1:] < bounds[0, 1:]-tolerance) or
            np.any(surface.bounds[1, 1:] > bounds[1, 1:]+tolerance)):
        raise ValueError("Remeshed synthetic-cylinder exterior exceeds corrected-skin Y/Z bounds")


def checked_export(mesh, file, reference, value, kernel, skin_bounds=None):
    if len(mesh.faces) > value["max_faces"] or len(mesh.vertices) > value["max_vertices"]:
        raise ValueError("Remesh surface resource limit exceeded")
    mesh.export(file)
    exported = trimesh.load(file, force="mesh")
    qc = dict(exported_surface_sha256=core.sha(file),
              local_filter_maxsurfdist_mm=value["per_operation_distance_mm"],
              maximum_stage_sampled_distance_mm=value["maximum_stage_sampled_distance_mm"],
              sampled_distance_is_not_certified_Hausdorff_bound=True,
              yz_check_applied=skin_bounds is not None)
    try:
        qc.update(core.surface_qc(exported))
        if exported.euler_number != reference.euler_number:
            raise ValueError("Remesh changed exterior topology")
        if skin_bounds is not None:
            check_yz_bounds(exported, skin_bounds, value["maximum_yz_extension_mm"])
        distances = dict(previous_to_exported=kernel["sampled_distance"](reference, exported, value["sample_count"]),
                         exported_to_previous=kernel["sampled_distance"](exported, reference, value["sample_count"]))
        qc.update(sampled_operation_distances=distances,
                  relative_volume_error_to_previous=float(abs(exported.volume/reference.volume-1)))
        # MeshLab's maxsurfdist controls local edits inside six iterations.
        # A whole exported stage is a different geometric comparison.
        if max(d["max_mm"] for d in distances.values()) > value["maximum_stage_sampled_distance_mm"]:
            raise ValueError("Exported remesh stage exceeds its sampled stage-distance limit")
    except Exception as error:
        qc.update(passed=False, error_type=type(error).__name__, error=str(error))
        core.write_new(file.with_name(file.stem+"_qc.local.json"), qc)
        raise
    return exported, qc


def update_contacts(candidate, source_contacts, cfg, value):
    result = copy.deepcopy(source_contacts)
    records = result["physical_contacts"]
    spec = cfg["contacts"]["physical_centres_lps_mm"]
    if {r["id"] for r in records} != set(spec) or result["scenarios"] != cfg["contacts"]["scenarios"]:
        raise ValueError("Remesh must preserve the declared physical inventory and electrical groups")
    for row in records:
        initial = spec[row["id"]]
        if row["raw_lps_mm"] != initial["raw_lps_mm"] or row["nominal_area_mm2"] != initial["nominal_area_mm2"]:
            raise ValueError("Remesh cannot silently change raw contact coordinates or nominal area")
        previous = np.asarray(row["projected_lps_mm"], float)
        raw = np.asarray(row["raw_lps_mm"], float)
        current, distance, face = candidate.nearest.on_surface(previous[None])
        if distance[0] > value["maximum_projection_shift_mm"]:
            raise ValueError("Remesh contact projection shift exceeds its configured limit")
        raw_distance = float(np.linalg.norm(current[0]-raw))
        if raw_distance > cfg["contacts"]["maximum_projection_mm"]:
            raise ValueError("Raw-to-current contact displacement exceeds the configured limit")
        row["source_projected_lps_mm"] = previous.tolist()
        row.setdefault("original_projected_lps_mm", previous.tolist())
        row.update(projected_lps_mm=current[0].tolist(), surface_face=int(face[0]),
                   source_to_current_displacement_mm=float(distance[0]), displacement_mm=raw_distance,
                   projection_method="nearest point from previous surface-projected centre to exported remesh")
    for scenario in result["scenarios"]:
        present = scenario.get("present_contact_ids", [r["id"] for r in records])
        points = [r["projected_lps_mm"] for r in records if r["id"] in present]
        if len(np.unique(points, axis=0)) != len(points):
            raise ValueError("Remesh collapsed co-present physical contact centres")
    result.setdefault("original_projection_surface_path", source_contacts["projection_surface_path"])
    result.setdefault("original_projection_surface_sha256", source_contacts["projection_surface_sha256"])
    result.update(contact_faces_constructed=False, finite_area_metadata_only=True,
                  contact_overlap_on_mesh_checked=False, CEM_adapter_ready=False,
                  projection_surface_role="remeshed_computational_exterior", **core.FLAGS)
    return result


def build(cfg, source, out):
    value = settings(cfg); kernel = core.kernels()
    original = trimesh.load(source / "exterior.stl", force="mesh")
    core.surface_qc(original)
    if len(original.faces) > value["max_faces"] or len(original.vertices) > value["max_vertices"]:
        raise ValueError("Input surface exceeds the remesh resource limits")
    source_qc = core.read(source / "surface_qc.local.json")
    cylinders = source_qc.get("synthetic_cylinders", [])
    skin_bounds = source_qc.get("skin_bounds_mm") if cylinders else None
    if cylinders and skin_bounds is None:
        skin_file = source / "skin_regularized.stl"
        if not skin_file.exists():
            raise ValueError("Synthetic-cylinder remesh requires a corrected-skin reference")
        skin_bounds = trimesh.load(skin_file, force="mesh").bounds.tolist()
    if skin_bounds is not None:
        check_yz_bounds(original, skin_bounds, value["maximum_yz_extension_mm"])
    contacts = core.read(source / "contacts.local.json")
    by_id = {r["id"]: r for r in contacts["physical_contacts"]}
    selection = value["selection"]
    if not set(selection["physical_contact_ids"]) <= set(by_id):
        raise ValueError("Unknown physical contact ID in remesh selection")
    centres = [by_id[name]["projected_lps_mm"] for name in selection["physical_contact_ids"]]
    centres.extend(selection["guides_lps_mm"])
    core.write_new(out / "remesh_settings.local.json", value)
    current = original; operations = []
    for index, label in enumerate(["global", "buffer", "contact"]):
        target = value[label+"_edge_mm"]
        selected = None if label == "global" else select_region(current, centres,
            selection[label+"_radius_mm"], selection[label+"_aabbs_lps_mm"])
        if selected is not None and not selected.any():
            raise ValueError(f"Declared remesh {label} region selects no faces")
        ms = pymeshlab.MeshSet()
        if selected is None:
            ms.add_mesh(pymeshlab.Mesh(current.vertices, current.faces.astype(np.int32)))
        else:
            ms.add_mesh(pymeshlab.Mesh(current.vertices, current.faces.astype(np.int32),
                                      f_scalar_array=selected.astype(float)))
            ms.compute_selection_by_condition_per_face(condselect="fq > 0.5")
        ms.meshing_isotropic_explicit_remeshing(iterations=value["iterations"], adaptive=value["adaptive"],
            targetlen=pymeshlab.PureValue(target), selectedonly=selected is not None,
            featuredeg=value["feature_angle_deg"], checksurfdist=True,
            maxsurfdist=pymeshlab.PureValue(value["per_operation_distance_mm"]),
            splitflag=value["split_edges"], collapseflag=value["collapse_edges"], swapflag=value["swap_edges"],
            smoothflag=value["smooth_vertices"], reprojectflag=value["reproject"])
        mesh = ms.current_mesh()
        trial = trimesh.Trimesh(mesh.vertex_matrix(), mesh.face_matrix(), process=False)
        file = out / f"remesh_{index:02d}_{label}.stl"
        previous = current
        current, qc = checked_export(trial, file, previous, value, kernel, skin_bounds)
        qc.update(operation=label, target_edge_mm=target, iterations=value["iterations"],
                  source_faces=len(previous.faces), source_vertices=len(previous.vertices),
                  selected_faces=len(previous.faces) if selected is None else int(selected.sum()),
                  selected_region_rule="all faces" if selected is None else "triangle centroids inside explicit spheres or AABBs",
                  source_surface_sha256=core.sha(source / "exterior.stl") if index == 0 else operations[-1]["exported_surface_sha256"])
        core.write_new(out / f"remesh_{index:02d}_{label}_qc.local.json", qc)
        operations.append(qc)
    # Preserve the exact bytes that passed the final operation checks.
    shutil.copyfile(file, out / "exterior.stl")
    final = trimesh.load(out / "exterior.stl", force="mesh")
    final_qc = core.surface_qc(final)
    distances = dict(source_to_remeshed=kernel["sampled_distance"](original, final, value["sample_count"]),
                     remeshed_to_source=kernel["sampled_distance"](final, original, value["sample_count"]))
    error = float(abs(final.volume/original.volume-1))
    core.write_new(out / "final_geometry_qc.local.json", dict(
        sampled_distances=distances, relative_volume_error=error,
        maximum_sampled_distance_mm=value["maximum_sampled_distance_mm"],
        maximum_relative_volume_error=value["maximum_relative_volume_error"],
        source_surface_sha256=core.sha(source / "exterior.stl"),
        exported_surface_sha256=core.sha(out / "exterior.stl"),
        sampled_distance_is_not_certified_Hausdorff_bound=True))
    if max(d["max_mm"] for d in distances.values()) > value["maximum_sampled_distance_mm"] or error > value["maximum_relative_volume_error"]:
        raise ValueError("Final exported remesh exceeds the declared distance/volume limits")
    if skin_bounds is not None:
        check_yz_bounds(final, skin_bounds, value["maximum_yz_extension_mm"])
    projected = update_contacts(final, contacts, cfg, value)
    projected.update(source_contacts_path=str(source / "contacts.local.json"),
        source_contacts_sha256=core.sha(source / "contacts.local.json"),
        source_projection_surface_path=str(source / "exterior.stl"),
        source_projection_surface_sha256=core.sha(source / "exterior.stl"),
        projection_surface_path=str(out / "exterior.stl"), projection_surface_sha256=core.sha(out / "exterior.stl"))
    shutil.copyfile(source / "contacts.local.json", out / "source_contacts.local.json")
    core.write_new(out / "contacts.local.json", projected)
    final_qc.update(source_surface_sha256=core.sha(source / "exterior.stl"),
        exported_surface_sha256=core.sha(out / "exterior.stl"), operations=operations,
        sampled_distances=distances, relative_volume_error=error,
        synthetic_cylinders=cylinders, skin_bounds_mm=skin_bounds, yz_check_applied=skin_bounds is not None,
        physical_contact_source_to_current_displacement_mm={r["id"]: r["source_to_current_displacement_mm"] for r in projected["physical_contacts"]},
        sampled_distance_is_not_certified_Hausdorff_bound=True, automatic_repair=False, vertex_smoothing=False,
        canonical_CEM_patch_QC_performed=False)
    core.write_new(out / "surface_qc.local.json", final_qc)
