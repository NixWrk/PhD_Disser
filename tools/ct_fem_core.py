"""CT/geometry preparation only. No CEM assembly, FEM solve or model promotion.

Immutable historical geometry kernels are loaded as definitions, without their
subject-specific module entry points or mutable SEG/RADIUS globals.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.io import loadmat, savemat
from scipy.stats import qmc
from skimage.measure import euler_number, marching_cubes
from skimage.morphology import disk
import trimesh

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "MATLAB_TRKG4_real_subjects/tools"
MATERIALS = ["soft_background", "lung", "bone", "heart", "fat"]
PRIORITY = ["lung", "bone", "heart", "fat", "background"]
QUALITY_LIMITS = {"minimum_mean_ratio": .01, "maximum_edge_condition": 10000.}
POINTS = (1024, 4096)
SEED = 20261008
CONTINUATION_STAGES = ("remesh", "mesh", "materials", "refine")
MESH_GENERATION_KEYS = ("size_mm", "steiner_limit", "maximum_relative_volume_error",
                        "maximum_sampled_surface_error_mm", "tetgen_options", "quality_recovery")
KERNEL_FILES = ["georg_ct_model.py", "georg_mesh_quality.py",
                "georg_quality_selected_refinement.py", "georg_lateral_quadrature.py",
                "add_arm_extensions.py", "scorpion_georg_geometry_revision.py", "georg_quality_mesh_prepare.py"]
FLAGS = dict(FEM_was_run=False, global_mesh_convergence_proven=False,
             independent_anatomical_validation=False, quantitative_result_accepted=False,
             automatic_publication=False)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def definitions(filename, names, namespace):
    source = LEGACY / filename
    tree = ast.parse(source.read_text(encoding="utf-8-sig"), filename=str(source))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError(f"Historical kernel definitions changed: {filename}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)


def kernels():
    """Use original numerical functions, with explicit minimal dependencies."""
    import itertools
    from shapely.geometry import Point, Polygon
    from shapely.ops import nearest_points, unary_union
    from typing import Sequence
    namespace = dict(np=np, nib=nib, ndi=ndi, disk=disk, marching_cubes=marching_cubes,
                     trimesh=trimesh, itertools=itertools, math=math, Point=Point, Polygon=Polygon,
                     nearest_points=nearest_points, unary_union=unary_union, Sequence=Sequence,
                     PAIRS=list(itertools.combinations(range(4), 2)),
                     QUALITY_LIMITS=QUALITY_LIMITS.copy(), VOLUME_TOLERANCE=1e-10,
                     BOUNDARY_TOLERANCE_MM=1e-9, write_new=write_new)
    definitions("georg_ct_model.py", ["body_mask", "surface_from_mask", "sampled_distance"], namespace)
    tree = ast.parse((LEGACY / "georg_mesh_quality.py").read_text(encoding="utf-8-sig"))
    definitions("georg_mesh_quality.py", [n.name for n in tree.body if isinstance(n, ast.FunctionDef)], namespace)
    definitions("georg_quality_selected_refinement.py",
                ["select_parents", "predicted_nodes", "bad_green_parents", "refine_screened",
                 "boundary_identity", "verify_nested_geometry"], namespace)
    definitions("georg_lateral_quadrature.py", ["uniform_boxes"], namespace)
    definitions("add_arm_extensions.py", ["ArmExtensionError", "_edge_topology", "_mesh_stats",
                "_require_closed_positive_volume", "_cylinder_mesh", "_boolean_union_manifold"], namespace)
    definitions("scorpion_georg_geometry_revision.py", ["section_polygon"], namespace)
    return namespace


def positive(value, name):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def natural(value, name):
    number = positive(value, name)
    if number != int(number):
        raise ValueError(f"{name} must be an integer")
    return int(number)


def load_config(path):
    path = Path(path).resolve()
    cfg = read(path)
    if cfg.get("schema") != "ct_fem_preparation_v1":
        raise ValueError("Unsupported configuration schema")
    if cfg.get("coordinates") != {"nifti_frame": "RAS", "geometry_frame": "LPS", "units": "mm"}:
        raise ValueError("Explicit NIfTI RAS / geometry LPS / mm declaration required")
    if set(cfg["inputs"]["masks"]) != {"lung", "bone", "heart", "fat"}:
        raise ValueError("Exactly four named masks are required")
    if not cfg["inputs"].get("mask_provenance"):
        raise ValueError("Declare mask provenance; automatic and manual masks are distinct inputs")
    if not cfg["inputs"].get("ct_provenance"):
        raise ValueError("Declare CT provenance without inferring an acquisition phase")
    if cfg["inputs"].get("ct_intensity_units") != "HU":
        raise ValueError("CT intensities must explicitly be declared as HU")
    cfg["inputs"]["ct"] = str((path.parent / cfg["inputs"]["ct"]).resolve())
    cfg["inputs"]["masks"] = {k: str((path.parent / v).resolve()) for k, v in cfg["inputs"]["masks"].items()}
    body = cfg["body"]
    if body.get("strategy") not in ("HU_largest_component_per_slice", "author_STL_LPS", "supplied_STL_LPS"):
        raise ValueError("Explicit supported body-input strategy required")
    if body["strategy"] == "HU_largest_component_per_slice":
        if body["closing_iterations"] != 1 or body["closing_connectivity"] != 1 or body["surface_level"] != .49:
            raise ValueError("This version requires one 6-neighbour closing and surface level .49")
        ratio = float(body["maximum_changed_relative"])
        if not 0 <= ratio <= .001:
            raise ValueError("Body regularization bound must be in [0, .001]")
        if not math.isfinite(float(body["hu_threshold"])):
            raise ValueError("HU threshold must be finite")
    else:
        body["source_stl"] = str((path.parent / body["source_stl"]).resolve())
        from ct_fem_manual_surface import settings as manual_settings
        manual_settings(cfg)
    coverage = body["maximum_outside_body_voxels"]
    if set(coverage) != {"lung", "bone", "heart", "fat"} or any(v != int(v) or v < 0 for v in coverage.values()):
        raise ValueError("Declare nonnegative outside-body voxel bounds for every tissue")
    if cfg["materials"] != {"order": MATERIALS, "priority": PRIORITY,
                             "outside_ct": "soft_background", "points": list(POINTS), "seed": SEED}:
        raise ValueError("Material priority, outside-CT assumption and nested rule must be explicit")
    if cfg["mesh"]["quality_limits"] != QUALITY_LIMITS:
        raise ValueError("Shape limits cannot be relaxed")
    for name in ["max_nodes", "max_tetrahedra", "steiner_limit"]:
        natural(cfg["mesh"][name], name)
    positive(cfg["mesh"]["maximum_seconds"], "mesh.maximum_seconds")
    from ct_fem_tetgen import validate_options
    validate_options(cfg["mesh"])
    from ct_fem_mesh_recovery import validate_settings
    validate_settings(cfg["mesh"]["quality_recovery"])
    natural(cfg["resources"]["max_ct_voxels"], "max_ct_voxels")
    positive(cfg["electrical_context"]["frequency_hz"], "frequency_hz")
    cfg["_config_path"] = str(path)
    return cfg


def load_inputs(cfg):
    ct = nib.load(cfg["inputs"]["ct"])
    if len(ct.shape) != 3 or np.prod(ct.shape) > cfg["resources"]["max_ct_voxels"]:
        raise ValueError("CT dimension or voxel resource bound failed")
    if ct.header.get_xyzt_units()[0] != "mm":
        raise ValueError("NIfTI spatial units must explicitly be mm; no implicit scaling")
    if not np.isfinite(ct.affine).all() or abs(np.linalg.det(ct.affine[:3, :3])) == 0:
        raise ValueError("Invalid CT affine")
    masks = {}
    for name, path in cfg["inputs"]["masks"].items():
        image = nib.load(path)
        if image.shape != ct.shape or not np.array_equal(image.affine, ct.affine):
            raise ValueError(f"Mask grid/affine mismatch: {name}; silent resampling is forbidden")
        if image.header.get_xyzt_units()[0] != "mm":
            raise ValueError(f"Mask units must be mm: {name}")
        data = np.asanyarray(image.dataobj)
        if not np.isfinite(data).all() or not np.isin(data, [0, 1]).all():
            raise ValueError(f"Mask must be finite binary 0/1: {name}")
        masks[name] = data > 0
    hu = np.asanyarray(ct.dataobj)
    if not np.isfinite(hu).all():
        raise ValueError("CT contains nonfinite intensities")
    labels = np.zeros(ct.shape, np.uint8)
    for name, index in [("fat", 4), ("heart", 3), ("bone", 2), ("lung", 1)]:
        labels[masks[name]] = index
    overlap = sum(m.astype(np.uint8) for m in masks.values()) > 1
    return ct, hu, masks, labels, dict(overlap_voxels=int(overlap.sum()),
        mask_voxels={k: int(v.sum()) for k, v in masks.items()},
        empty_masks=[k for k, v in masks.items() if not v.any()], silent_resampling=False)


def lps_to_voxel(points, affine):
    return nib.affines.apply_affine(np.linalg.inv(affine) @ np.diag([-1., -1., 1., 1.]), points)


def material_identity(cfg):
    paths = [cfg["inputs"]["ct"], *cfg["inputs"]["masks"].values()]
    return dict(inputs={str(Path(p).resolve()): sha(p) for p in paths},
                rule=cfg["materials"], mapper_sha256=sha(__file__),
                certifier_sha256=sha(LEGACY / "georg_lateral_quadrature.py"))


def mesh_qc(nodes, tets, boundary=None, max_nodes=1000000, max_tetrahedra=10000000):
    nodes, tets = np.asarray(nodes), np.asarray(tets)
    if (nodes.ndim != 2 or nodes.shape[1] != 3 or not len(nodes) or
            not np.isfinite(nodes).all() or tets.ndim != 2 or tets.shape[1] != 4 or
            not len(tets) or not np.issubdtype(tets.dtype, np.integer) or
            tets.min() < 0 or tets.max() >= len(nodes)):
        raise ValueError("Invalid finite nodes / zero-based integer tetrahedra")
    if len(np.unique(np.sort(tets, axis=1), axis=0)) != len(tets):
        raise ValueError("Duplicate tetrahedra")
    kernel = kernels()
    if not np.all(kernel["volumes"](nodes, tets) > 0):
        raise ValueError("Mesh has nonpositive tetrahedra")
    actual = kernel["boundary_faces"](tets)
    if boundary is not None and (np.asarray(boundary).shape != actual.shape or
            not np.issubdtype(np.asarray(boundary).dtype, np.integer) or
            not np.array_equal(np.unique(np.sort(actual, axis=1), axis=0),
                               np.unique(np.sort(boundary, axis=1), axis=0))):
        raise ValueError("Declared boundary differs from tetrahedral boundary")
    if boundary is not None:
        declared = np.asarray(boundary)
        actual_keys = np.sort(actual, axis=1); declared_keys = np.sort(declared, axis=1)
        actual_order = np.lexsort(tuple(actual_keys[:, i] for i in (2, 1, 0)))
        declared_order = np.lexsort(tuple(declared_keys[:, i] for i in (2, 1, 0)))
        for start in range(0, len(actual), 100000):
            a = nodes[actual[actual_order[start:start+100000]]]
            b = nodes[declared[declared_order[start:start+100000]]]
            an = np.cross(a[:, 1]-a[:, 0], a[:, 2]-a[:, 0])
            bn = np.cross(b[:, 1]-b[:, 0], b[:, 2]-b[:, 0])
            if not np.all(np.einsum("ij,ij->i", an, bn) > 0):
                raise ValueError("Declared boundary orientation differs from outward tetrahedral faces")
        # Boundary IDs belong to the saved parent, not a newly sorted face list.
        actual = declared.copy()
    edges = np.sort(np.concatenate([actual[:, [0, 1]], actual[:, [1, 2]], actual[:, [2, 0]]]), axis=1)
    if not np.all(np.unique(edges, axis=0, return_counts=True)[1] == 2):
        raise ValueError("Mesh boundary is not a closed two-manifold")
    report = kernel["quality"](nodes, tets, QUALITY_LIMITS.copy())
    report["within_resource_limits"] = len(nodes) <= max_nodes and len(tets) <= max_tetrahedra
    report["passed"] = bool(report["passed"] and report["within_resource_limits"])
    report["boundary_order_source"] = "supplied_mesh_boundary" if boundary is not None else "computed_tetrahedral_boundary"
    report["outward_boundary_orientation_checked"] = True
    report.update(FLAGS)
    if not report["passed"]:
        raise ValueError("Mesh fails fixed shape/resource limits")
    return actual, report


def read_mesh(directory, cfg):
    m = loadmat(Path(directory) / "mesh.mat")
    nodes = m["nodes_mm"]
    raw = m["elems"]
    if (not np.issubdtype(raw.dtype, np.integer) or raw.min() < 1 or raw.max() > len(nodes) or
            not np.issubdtype(m["boundary"].dtype, np.integer)):
        raise ValueError("MAT mesh requires one-based integer indices")
    tets = raw.astype(np.int32) - 1
    boundary, report = mesh_qc(nodes, tets, m["boundary"].astype(np.int32)-1,
                              cfg["mesh"]["max_nodes"], cfg["mesh"]["max_tetrahedra"])
    return nodes, tets, boundary, report


def save_mesh(directory, nodes, tets, boundary, report, source_file=None):
    if source_file is None:
        savemat(Path(directory) / "mesh.mat", dict(nodes_mm=nodes, elems=tets+1, boundary=boundary+1))
    else:
        shutil.copyfile(source_file, Path(directory) / "mesh.mat")
    write_new(Path(directory) / "mesh_quality.local.json", dict(report, matrix_index_base=1,
              coordinate_frame="LPS", units="mm", mesh_sha256=sha(Path(directory) / "mesh.mat")))


def integrate_materials(nodes, tets, labels, affine, *, parent=None, chunk=256):
    """Nested historical Sobol rule; reusable parents require caller-pinned inputs.

    parent=(old_nodes, old_tets, element_parent, fractions_by_q, outside_by_q).
    Untouched elements alone are reused; all split elements are sampled anew.
    """
    kernel = kernels()
    vox = lps_to_voxel(nodes, affine)
    low = np.empty((len(tets), 3), np.int32); high = low.copy()
    for start in range(0, len(tets), 100000):
        p = vox[tets[start:start+100000]]
        low[start:start+len(p)] = np.floor(p.min(1)+.5).astype(np.int32)-1
        high[start:start+len(p)] = np.floor(p.max(1)+.5).astype(np.int32)+1
    certified = kernel["uniform_boxes"](labels, low, high)
    inside = np.all((low >= 0) & (high < np.array(labels.shape)), axis=1)
    fully_outside = np.any((high < 0) | (low >= np.array(labels.shape)), axis=1)
    reused = np.zeros(len(tets), bool)
    fractions = {q: np.eye(5)[np.maximum(certified, 0)] for q in POINTS}
    outside = {q: fully_outside.astype(float) for q in POINTS}
    if parent is not None:
        pn, pt, ep, pf, po = parent
        ep = np.asarray(ep)
        if (ep.shape != (len(tets),) or not np.issubdtype(ep.dtype, np.integer) or
                ep.min() < 0 or ep.max() >= len(pt)):
            raise ValueError("Invalid material parent map")
        reused = np.bincount(ep, minlength=len(pt))[ep] == 1
        np.testing.assert_array_equal(nodes[:len(pn)], pn)
        np.testing.assert_array_equal(tets[reused], pt[ep[reused]])
        for q in POINTS:
            if pf[q].shape != (len(pt), 5) or po[q].shape != (len(pt),):
                raise ValueError("Parent fractions/outside metadata shape mismatch")
            if np.any(pf[q] < 0) or not np.array_equal(pf[q].sum(1), np.ones(len(pt))):
                raise ValueError("Invalid parent fractions")
            if not np.isfinite(po[q]).all() or np.any((po[q] < 0) | (po[q] > 1)):
                raise ValueError("Invalid parent outside-CT metadata")
            fractions[q][reused] = pf[q][ep[reused]]
            outside[q][reused] = po[q][ep[reused]]
    targets = np.flatnonzero(~reused & ((certified < 0) | (~inside & ~fully_outside)))
    cuts = np.sort(qmc.Sobol(3, scramble=True, seed=SEED).random_base2(12), axis=1)
    bary = np.diff(np.c_[np.zeros(len(cuts)), cuts, np.ones(len(cuts))], axis=1)
    for start in range(0, len(targets), chunk):
        ids = targets[start:start+chunk]
        p = np.einsum("qj,bjk->bqk", bary, vox[tets[ids]], optimize=True)
        index = np.floor(p+.5).astype(np.int32)
        valid = np.all((index >= 0) & (index < np.array(labels.shape)), axis=2)
        sampled = np.zeros(valid.shape, np.uint8); ijk = index[valid]
        sampled[valid] = labels[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        for q in POINTS:
            fractions[q][ids] = np.column_stack([(sampled[:, :q] == j).mean(1) for j in range(5)])
            outside[q][ids] = (~valid[:, :q]).mean(1)
    volume = kernel["volumes"](nodes, tets)
    report = dict(order=MATERIALS, priority=PRIORITY, points=list(POINTS), seed=SEED,
        quadrature_is_nested=True, certified_cells=int(np.count_nonzero(certified >= 0)),
        unchanged_parent_elements_reused=int(reused.sum()),
        outside_CT_is_soft_background_assumption=True,
        potentially_outside_CT_cells=int(np.count_nonzero(~inside)),
        inventory={str(q): dict(volume_ml=(volume @ fractions[q]/1000).tolist(),
                               sampled_outside_CT_volume_ml=float(volume @ outside[q]/1000)) for q in POINTS})
    for q in POINTS:
        if np.any(fractions[q] < 0) or not np.array_equal(fractions[q].sum(1), np.ones(len(tets))):
            raise ValueError("Invalid integrated fractions")
    return fractions, outside, report


def full_section_join(surface, arm, kernel=None):
    """Parameterised historical embedded_join; one justified algorithm copy.

    Same search/objective/polygon operations as the immutable original. Radius,
    crop plane, centre and search domain are supplied rather than global.
    Only the explicitly declared LPS X-axis cylinder scenario is supported.
    """
    kernel = kernel or kernels()
    if arm["axis"] != "x" or arm["direction"] not in (-1, 1):
        raise ValueError("Full-section joins currently support LPS X, direction -1 or +1")
    radius = positive(arm["radius_mm"], "radius_mm")
    clearance = positive(arm["clearance_mm"], "clearance_mm")
    original = np.asarray(arm["center_yz_mm"], float)
    begin, end, step = map(float, arm["inset_search_mm"])
    if original.shape != (2,) or not np.isfinite(original).all() or not 0 < begin <= end or step <= 0:
        raise ValueError("Invalid explicit cylinder centre / inset search")
    if not np.isfinite([begin, end, step, arm["cut_x_mm"]]).all():
        raise ValueError("Nonfinite cylinder geometry")
    candidates = []
    count = int(np.floor((end-begin)/step+1e-12))+1
    for inset in begin+step*np.arange(count):
        x = arm["cut_x_mm"]-arm["direction"]*inset
        polygon = kernel["section_polygon"](surface, x)
        if polygon is None:
            continue
        possible = polygon.buffer(-(radius+clearance))
        if possible.is_empty:
            continue
        chosen = kernel["nearest_points"](possible, kernel["Point"](original))[0]
        yz = np.array(chosen.coords[0])
        circle = kernel["Point"](yz).buffer(radius, quad_segs=64)
        fraction = polygon.intersection(circle).area/circle.area
        if fraction < 1-1e-10:
            continue
        candidates.append(dict(x_mm=float(x), centre_yz_mm=yz.tolist(), inset_mm=float(inset),
            centre_shift_yz_mm=(yz-original).tolist(),
            score_mm2=float(inset**2+np.sum((yz-original)**2)),
            polygon_area_mm2=float(circle.area),
            intersection_area_mm2=float(polygon.intersection(circle).area),
            cross_section_covered_fraction=float(fraction)))
    if not candidates:
        raise ValueError("No full circular section fits the declared cylinder search")
    chosen = min(candidates, key=lambda c: c["score_mm2"])
    chosen.update(candidate_count=len(candidates), radius_mm=radius, geometric_clearance_mm=clearance,
                  circle_test_is_polygonal=True, anatomically_validated=False)
    return chosen


def surface_qc(surface):
    if (not surface.is_volume or not surface.is_watertight or len(surface.split()) != 1 or
            surface.volume <= 0 or not np.isfinite(surface.vertices).all()):
        raise ValueError("Surface must be one watertight positive volume")
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(surface.vertices, surface.faces.astype(np.int32)))
    ms.compute_selection_by_self_intersections_per_face()
    intersections = ms.current_mesh().selected_face_number()
    intersection_qc = None
    if intersections:
        from ct_fem_surface_intersections import adjudicate
        selected = np.flatnonzero(ms.current_mesh().face_selection_array())
        intersection_qc = adjudicate(surface, selected)
        if not intersection_qc["passed"]:
            raise ValueError("Surface contains unresolved self intersections; no automatic repair/decimation")
    return dict(passed=True, watertight=True, positive_volume=True, connected_components=1,
                self_intersecting_faces=0, detector_selected_faces=intersections,
                intersection_adjudication=intersection_qc, volume_ml=float(surface.volume/1000),
                euler_number=int(surface.euler_number), vertices=len(surface.vertices), faces=len(surface.faces),
                **FLAGS)


def contact_metadata(surface, spec):
    raw = spec["physical_centres_lps_mm"]
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Explicit named physical contact centres are required")
    records = []; ids = set(raw)
    for name, value in raw.items():
        point = np.asarray(value["raw_lps_mm"], float)
        if point.shape != (3,) or not np.isfinite(point).all() or not value.get("provenance"):
            raise ValueError("Contact centre and provenance must be explicit")
        area = positive(value["nominal_area_mm2"], "nominal_area_mm2")
        projected, distance, face = surface.nearest.on_surface(point[None])
        if distance[0] > positive(spec["maximum_projection_mm"], "maximum_projection_mm"):
            raise ValueError(f"Contact projection exceeds explicit limit: {name}")
        records.append(dict(id=name, raw_lps_mm=point.tolist(), projected_lps_mm=projected[0].tolist(),
            displacement_mm=float(distance[0]), surface_face=int(face[0]), nominal_area_mm2=area,
            nominal_disk_radius_mm=float(np.sqrt(area/np.pi)), provenance=value["provenance"],
            projection_is_new_manual_placement=False, displacement_is_placement_error=False))
    if not spec["scenarios"] or len({s["id"] for s in spec["scenarios"]}) != len(spec["scenarios"]):
        raise ValueError("Explicit distinct contact scenarios are required")
    for scenario in spec["scenarios"]:
        pairs = contact_groups(scenario)
        groups = dict(pairs)
        if any(not isinstance(name, str) or not name for name in groups):
            raise ValueError("Group IDs must be nonempty strings")
        flat = [p for group in groups.values() for p in group]
        if not flat or len(flat) != len(set(flat)) or not set(flat) <= ids or any(not g for g in groups.values()):
            raise ValueError("Invalid/duplicated physical contact membership")
        if "present_contact_ids" in scenario:
            present = scenario["present_contact_ids"]
            if (not present or len(present) != len(set(present)) or not set(present) <= ids
                    or not set(flat) <= set(present)):
                raise ValueError("Groups must use only explicitly present physical contacts")
        else:
            present = list(raw)  # Legacy metadata check; not sufficient for CEM.
        co_present = [r for r in records if r["id"] in present]
        if len(np.unique([r["raw_lps_mm"] for r in co_present], axis=0)) != len(co_present):
            raise ValueError("Co-present physical contact IDs have duplicate raw centres")
        if len(np.unique([r["projected_lps_mm"] for r in co_present], axis=0)) != len(co_present):
            raise ValueError("Co-present physical contacts collapse to duplicate projected centres")
        for key in ("drive_groups", "measure_groups"):
            if len(scenario[key]) != 2 or len(set(scenario[key])) != 2 or not set(scenario[key]) <= set(groups):
                raise ValueError("Explicit ordered drive and readout group pairs are required")
        if scenario["inactive_contact_model"] != "individual_floating_CEM_assumption":
            raise ValueError("Explicit inactive-contact assumption required")
    return dict(physical_contacts=records, scenarios=spec["scenarios"], units="mm", frame="LPS",
        physical_contact_count=len(records), contact_faces_constructed=False,
        physical_contact_count_is_catalog_size=True, physical_presence_is_scenario_specific=True,
        projection_surface_role="computational_exterior", surface_face_index_base=0,
        surface_face_is_not_FEM_boundary_id=True,
        finite_area_metadata_only=True, contact_overlap_on_mesh_checked=False,
        CEM_adapter_ready=False, actual_contact_area_is_measured=False, **FLAGS)


def copy_contacts(source, target):
    file = Path(source) / "contacts.local.json"
    if file.exists():
        shutil.copyfile(file, Path(target) / file.name)


def contact_groups(scenario):
    """Keep string IDs and declared order, including non-MATLAB identifiers."""
    groups = scenario["groups"]
    if isinstance(groups, dict):
        return list(groups.items())
    if isinstance(groups, list):
        pairs = [(g["id"], g["contact_ids"]) for g in groups]
        if len({p[0] for p in pairs}) == len(pairs):
            return pairs
    raise ValueError("Groups require distinct string IDs and ordered contact lists")


def check_continuation_contract(stage, cfg, source, previous, sources):
    """Reject an inherited artifact described by new, unexecuted input settings."""
    parent_cfg = previous["configuration"]
    completed = previous.get("completed_geometry_settings", {})
    for block in ("body", "synthetic_cylinders", "surface_qc"):
        effective = completed.get(block, parent_cfg[block])
        if parent_cfg[block] != effective or cfg[block] != effective:
            raise ValueError(f"Executed geometry block '{block}' changed; use a fresh surface preparation")
    if "remeshing" in completed and stage != "remesh" and cfg["remeshing"] != completed["remeshing"]:
        raise ValueError("Executed remeshing settings changed without a remesh stage")
    if "mesh_generation" in completed and stage != "mesh":
        requested = {key: cfg["mesh"][key] for key in MESH_GENERATION_KEYS}
        if requested != completed["mesh_generation"]:
            raise ValueError("Executed mesh-generation settings changed without rebuilding the mesh")
    spec = parent_cfg["contacts"]
    if cfg["contacts"] != spec or list(cfg["contacts"]["physical_centres_lps_mm"]) != list(spec["physical_centres_lps_mm"]):
        raise ValueError("Contact configuration changed; use a fresh preparation")
    for current, parent in zip(cfg["contacts"]["scenarios"], spec["scenarios"]):
        if contact_groups(current) != contact_groups(parent):
            raise ValueError("Contact group ordering changed; use a fresh preparation")
    contact_file = source / "contacts.local.json"
    if not contact_file.is_file() or "contacts.local.json" not in previous["products"]:
        raise ValueError("Completed parent lacks a bound physical-contact artifact")
    actual = read(contact_file)
    if actual.get("units") != "mm" or actual.get("frame") != "LPS":
        raise ValueError("Parent actual contact units/frame disagree with preparation")
    if actual["scenarios"] != spec["scenarios"]:
        raise ValueError("Parent actual contacts disagree with its contact configuration")
    if any(contact_groups(a) != contact_groups(s) for a, s in zip(actual["scenarios"], spec["scenarios"])):
        raise ValueError("Parent actual contact group ordering disagrees with its configuration")
    rows = actual["physical_contacts"]
    if [r["id"] for r in rows] != list(spec["physical_centres_lps_mm"]):
        raise ValueError("Parent actual contact inventory/order disagrees with its configuration")
    for row in rows:
        declared = spec["physical_centres_lps_mm"][row["id"]]
        for field in ("raw_lps_mm", "nominal_area_mm2", "provenance"):
            if row[field] != declared[field]:
                raise ValueError("Parent actual contacts disagree with its contact configuration")
    masks = {}
    for name in ("lung", "bone", "heart", "fat"):
        expected = previous["sources"][parent_cfg["inputs"]["masks"][name]]
        if sources[cfg["inputs"]["masks"][name]] != expected:
            raise ValueError(f"Mask identity changed for {name}; use a fresh surface preparation")
        masks[name] = expected
    return dict(parent_stage=previous["stage"], parent_manifest_sha256=sources[str(source / "preparation.local.json")],
                actual_contacts_sha256=sources[str(contact_file)],
                contact_configuration_preserved=True, mask_sha256_by_role=masks,
                body_and_synthetic_geometry_preserved=True,
                coverage_bounds_preserved_without_relaxation=True)


def artifact_sources(cfg, input_dir=None, previous_sources=None):
    paths = [Path(cfg["_config_path"]), Path(cfg["inputs"]["ct"]),
             *map(Path, cfg["inputs"]["masks"].values()), Path(__file__),
             Path(__file__).with_name("prepare_ct_fem.py"), *(LEGACY / n for n in KERNEL_FILES)]
    paths.append(Path(__file__).with_name("ct_fem_remesh.py"))
    paths.append(Path(__file__).with_name("ct_fem_tetgen.py"))
    paths.append(Path(__file__).with_name("ct_fem_surface_intersections.py"))
    paths.append(Path(__file__).with_name("ct_fem_mesh_recovery.py"))
    if cfg['body']['strategy'] in ('author_STL_LPS', 'supplied_STL_LPS'):
        paths.extend([Path(cfg['body']['source_stl']), Path(__file__).with_name('ct_fem_manual_surface.py')])
    if input_dir is not None:
        paths.extend(p for p in Path(input_dir).rglob("*") if p.is_file())
    for path, digest in cfg.get('segmentation_sources', {}).items():
        if sha(path) != digest:
            raise ValueError('TotalSegmentator source changed: '+path)
        paths.append(Path(path))
    paths.extend(map(Path, previous_sources or {}))
    return {str(p): sha(p) for p in dict.fromkeys(p.resolve() for p in paths)}


def verify_sources(sources):
    for p, digest in sources.items():
        if sha(p) != digest:
            raise ValueError(f"Preparation source changed: {p}")


def verify_artifact(directory):
    directory = Path(directory).resolve()
    manifest = read(directory / "preparation.local.json")
    if (manifest.get("schema") != "ct_fem_preparation_artifact_v1" or
            manifest.get("status") != "prepared_not_validated"):
        raise ValueError("Input preparation is incomplete")
    if manifest["stage"] in ("refine_inherited", "refine_remapped"):
        root = Path(manifest["refinement_root_directory"]).resolve()
        if root != directory.parent or not (root / "preparation.local.json").is_file():
            raise ValueError("Refinement branch has no completed parent stage")
        parent_stage = verify_artifact(root)
        if parent_stage["stage"] != "refine":
            raise ValueError("Refinement branch belongs to an invalid parent stage")
    for name, digest in manifest["products"].items():
        if sha(Path(directory) / name) != digest:
            raise ValueError(f"Input artifact changed: {name}")
    verify_sources(manifest["sources"])
    return manifest


def close_body_once(original):
    return ndi.binary_closing(np.pad(original, 2), structure=ndi.generate_binary_structure(3, 1))[2:-2, 2:-2, 2:-2]


def surface_distance_settings(cfg):
    settings = cfg["surface_qc"]
    if "maximum_sampled_correction_mm" in settings:
        raise ValueError("Ambiguous legacy surface distance setting; declare separate topology_correction and export_fidelity budgets")
    endpoints = {"topology_correction": ("HU_body_native_level_0.49", "closed_body_native_level_0.49"),
                 "export_fidelity": ("closed_body_native_level_0.49", "closed_body_exported_STL")}
    result = {}
    for name, (reference, candidate) in endpoints.items():
        value = settings.get(name, {})
        if value.get("reference") != reference or value.get("candidate") != candidate:
            raise ValueError(f"Explicit {name} comparison endpoints are required")
        reason = value.get("bound_basis")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"Explicit {name} bound basis required; no inherited/automatic tolerance")
        result[name] = dict(value, maximum_sampled_distance_mm=positive(value["maximum_sampled_distance_mm"], name))
    return result


def surface_distance_qc(topology, exported, settings):
    result = {}
    for name, distances in (("topology_correction", topology), ("export_fidelity", exported)):
        bound = settings[name]["maximum_sampled_distance_mm"]
        result[name] = dict(settings[name], distances=distances,
                            passed=max(d["max_mm"] for d in distances.values()) <= bound)
    return dict(schema="ct_fem_body_correction_qc_v1", **result,
                passed=all(r["passed"] for r in result.values()), automatic_threshold_relaxation=False,
                sampled_distance_is_not_certified_Hausdorff_bound=True,
                export_pass_does_not_accept_topology_changes=True)


def build_surface(cfg, out, ct, hu, masks):
    if cfg['body']['strategy'] in ('author_STL_LPS', 'supplied_STL_LPS'):
        from ct_fem_manual_surface import build as manual_surface
        skin, qc = manual_surface(cfg, out, ct, masks)
        return finish_surface(cfg, out, skin, kernels(), qc)
    kernel = kernels(); settings = cfg["body"]; distance_settings = surface_distance_settings(cfg)
    original, added = kernel["body_mask"](hu, threshold=settings["hu_threshold"])
    mask = close_body_once(original)
    change = mask != original; ratio = float(change.sum()/original.sum())
    topology = [int(euler_number(mask, connectivity=c)) for c in (1, 3)]
    losses = {k: dict(outside_before=int(np.count_nonzero(v & ~original)),
                      outside_after=int(np.count_nonzero(v & ~mask)),
                      newly_lost=int(np.count_nonzero(v & original & ~mask))) for k, v in masks.items()}
    if ratio > settings["maximum_changed_relative"] or set(topology) != {1} or any(v["newly_lost"] for v in losses.values()):
        raise ValueError("Bounded topology regularization or tissue-loss check failed")
    if any(v["outside_after"] > settings["maximum_outside_body_voxels"][k] for k, v in losses.items()):
        raise ValueError("Tissue outside the body exceeds the explicit scenario bounds")
    for name, data in [("body_hu_mask.nii.gz", original), ("body_regularized_mask.nii.gz", mask)]:
        image = nib.Nifti1Image(data.astype(np.uint8), ct.affine); image.header.set_xyzt_units("mm")
        nib.save(image, out / name)
    native = kernel["surface_from_mask"](original, ct.affine, level=.49)
    native.export(out / "body_before_regularization.stl")
    regularized_native = kernel["surface_from_mask"](mask, ct.affine, level=.49)
    if regularized_native.euler_number != 2:
        raise ValueError("Regularized body retains a handle")
    regularized_native.export(out / "skin_regularized.stl")
    skin = trimesh.load(out / "skin_regularized.stl", force="mesh")
    surface_qc(skin)
    samples = natural(cfg["surface_qc"]["sample_count"], "sample_count")
    topology_distances = dict(reference_to_candidate=kernel["sampled_distance"](native, regularized_native, samples),
                             candidate_to_reference=kernel["sampled_distance"](regularized_native, native, samples))
    export_distances = dict(reference_to_candidate=kernel["sampled_distance"](regularized_native, skin, samples),
                           candidate_to_reference=kernel["sampled_distance"](skin, regularized_native, samples))
    distance_qc = surface_distance_qc(topology_distances, export_distances, distance_settings)
    distance_qc.update(changed_voxels=int(change.sum()), changed_relative=ratio,
        added_voxels=int(np.count_nonzero(mask & ~original)), removed_voxels=int(np.count_nonzero(original & ~mask)),
        tissue_coverage=losses, original_voxel_euler=[int(euler_number(original, connectivity=c)) for c in (1, 3)],
        corrected_voxel_euler=topology, original_surface_euler=int(native.euler_number),
        corrected_surface_euler=int(regularized_native.euler_number), **FLAGS)
    # Preserve evidence even when the explicit geometry scenario is rejected.
    write_new(out / "body_correction_qc.local.json", distance_qc)
    if not distance_qc["topology_correction"]["passed"]:
        raise ValueError("Sampled topology-correction distance exceeds its explicit scenario bound; see body_correction_qc.local.json")
    if not distance_qc["export_fidelity"]["passed"]:
        raise ValueError("Regularized-native to exported-STL fidelity exceeds its separate bound; see body_correction_qc.local.json")
    qc = dict(changed_voxels=int(change.sum()), changed_relative=ratio,
        added_voxels=int(np.count_nonzero(mask & ~original)), removed_voxels=int(np.count_nonzero(original & ~mask)),
        HU_hole_fill_added_voxels=added, tissue_coverage=losses, voxel_euler=topology,
        topology_correction_qc=distance_qc["topology_correction"], export_fidelity_qc=distance_qc["export_fidelity"],
        body_mask_assumption="Largest 3D component; largest opened component per axial slice; filled cavities")
    return finish_surface(cfg, out, skin, kernel, qc)


def finish_surface(cfg, out, skin, kernel, body_qc):
    """One common full-section union and contact projection for both body inputs."""
    joins, cylinders = [], []
    for arm in cfg["synthetic_cylinders"]:
        join = full_section_join(skin, arm, kernel)
        far = float(arm["cut_x_mm"]) + arm["direction"] * positive(arm["outward_length_mm"], "outward_length_mm")
        section = dict(side="left" if arm["direction"] < 0 else "right", axis_index=0,
                       perpendicular=[1, 2], cut=join["x_mm"], center_perpendicular=join["centre_yz_mm"])
        overlap = positive(arm["inner_overlap_mm"], "inner_overlap_mm")
        segments = natural(arm["segments"], "segments")
        if segments < 64:
            raise ValueError("Cylinder needs at least 64 polygon segments")
        cylinders.append(kernel["_cylinder_mesh"](section, arm["radius_mm"], abs(far-join["x_mm"]), overlap, segments)[0])
        joins.append(dict(join, far_x_mm=far, definition=arm))
    exterior = kernel["_boolean_union_manifold"](skin, cylinders) if cylinders else skin.copy()
    exterior.export(out / "exterior.stl")
    exterior = trimesh.load(out / "exterior.stl", force="mesh")
    tolerance = positive(cfg["surface_qc"]["maximum_yz_extension_mm"], "maximum_yz_extension_mm")
    if cylinders and (np.any(exterior.bounds[0, 1:] < skin.bounds[0, 1:]-tolerance) or
                      np.any(exterior.bounds[1, 1:] > skin.bounds[1, 1:]+tolerance)):
        raise ValueError("Synthetic cylinders protrude beyond the checked Y/Z extents")
    qc = surface_qc(exterior)
    qc.update(body_qc)
    qc.update(
        sampled_distance_is_not_certified_Hausdorff_bound=True,
        synthetic_cylinders=joins, skin_bounds_mm=skin.bounds.tolist(), pre_union_decimation=False,
        surface_roles=({"supplied_skin": "skin_regularized.stl", "computational_exterior": "exterior.stl"}
            if cfg['body']['strategy'] in ('author_STL_LPS', 'supplied_STL_LPS') else
            {"uncorrected_HU_contour": "body_before_regularization.stl", "corrected_CT_skin": "skin_regularized.stl", "computational_exterior": "exterior.stl"}))
    write_new(out / "surface_qc.local.json", qc)
    contacts = contact_metadata(exterior, cfg["contacts"])
    contacts.update(projection_surface_path=str(out / "exterior.stl"),
                    projection_surface_sha256=sha(out / "exterior.stl"))
    write_new(out / "contacts.local.json", contacts)


def build_mesh(cfg, source, out):
    from ct_fem_mesh_recovery import generate_screened
    surface = trimesh.load(source / "exterior.stl", force="mesh")
    surface_qc(surface)
    settings = cfg["mesh"]
    positive(settings["size_mm"], "size_mm")
    nodes, tets, version, surface, projected_contacts, recovery_rounds = generate_screened(cfg, source, out, surface)
    kernel = kernels(); tets = kernel["orient"](nodes, tets)
    try:
        boundary, qc = mesh_qc(nodes, tets, max_nodes=settings["max_nodes"], max_tetrahedra=settings["max_tetrahedra"])
    except ValueError:
        write_new(out / "tetgen_rejected_shape_qc.local.json", kernel["quality"](nodes, tets))
        raise
    solved = trimesh.Trimesh(nodes, boundary, process=True)
    actual_surface_qc = surface_qc(solved)
    source_surface_qc = read(source / "surface_qc.local.json")
    if source_surface_qc.get("synthetic_cylinders"):
        from ct_fem_remesh import check_yz_bounds
        tolerance = cfg["remeshing"]["maximum_yz_extension_mm"] if read(source / "preparation.local.json")["stage"] == "remesh" else cfg["surface_qc"]["maximum_yz_extension_mm"]
        check_yz_bounds(solved, source_surface_qc["skin_bounds_mm"], tolerance)
    n = natural(cfg["surface_qc"]["sample_count"], "sample_count")
    distance = dict(input_to_solved=kernel["sampled_distance"](surface, solved, n),
                    solved_to_input=kernel["sampled_distance"](solved, surface, n))
    error = abs(qc["volume_ml"]*1000/surface.volume-1)
    contacts = read(source / "contacts.local.json")
    reference_file = Path(contacts.get("original_projection_surface_path", contacts["projection_surface_path"]))
    reference_sha = contacts.get("original_projection_surface_sha256", contacts["projection_surface_sha256"])
    if sha(reference_file) != reference_sha:
        raise ValueError("Original exterior identity changed before cumulative boundary QC")
    reference = trimesh.load(reference_file, force="mesh")
    reference_distance = dict(original_to_solved=kernel["sampled_distance"](reference, solved, n),
                              solved_to_original=kernel["sampled_distance"](solved, reference, n))
    reference_error = abs(qc["volume_ml"]*1000/reference.volume-1)
    geometry_passed = (max(error, reference_error) <= settings["maximum_relative_volume_error"] and
        max(d["max_mm"] for d in [*distance.values(), *reference_distance.values()]) <= settings["maximum_sampled_surface_error_mm"])
    write_new(out / "tetgen_boundary_qc.local.json", dict(passed=geometry_passed,
        boundary_distances=distance, relative_volume_error=error,
        original_exterior_path=str(reference_file), original_exterior_sha256=reference_sha,
        original_boundary_distances=reference_distance, relative_volume_error_to_original=reference_error,
        sampled_distance_is_not_certified_Hausdorff_bound=True))
    if not geometry_passed:
        raise ValueError("Actual TetGen boundary exceeds the declared geometry scenario")
    qc.update(tetgen_version=version, boundary_distances=distance,
              relative_volume_error=error, actual_boundary_checked=True,
              actual_surface_qc=actual_surface_qc,
              surface_recovery_rounds=recovery_rounds,
              original_boundary_distances=reference_distance, relative_volume_error_to_original=reference_error,
              original_exterior_sha256=reference_sha, cumulative_geometry_checked=True)
    save_mesh(out, nodes, tets, boundary, qc)
    solved.export(out / "fem_boundary.stl")
    if projected_contacts is None:
        copy_contacts(source, out)
    else:
        shutil.copyfile(source / "contacts.local.json", out / "source_contacts.local.json")
        write_new(out / "contacts.local.json", projected_contacts)


def prepare_mesh_pipeline(config_path, out):
    """One fresh geometry pipeline; failed stages never become later parents."""
    config_path = Path(config_path).resolve(); out = Path(out).resolve()
    if read(config_path).get('schema') == 'ct_fem_ct_pipeline_v1':
        from ct_fem_ct_pipeline import prepare as prepare_ct_pipeline
        return prepare_ct_pipeline(config_path,out)
    cfg = load_config(config_path); digest = sha(config_path)
    if out.exists():
        raise ValueError("Mesh pipeline requires a fresh output directory")
    out.mkdir(parents=True)
    snapshots = out / "source_snapshots"; snapshots.mkdir()
    code = {}
    for path, checksum in artifact_sources(cfg).items():
        if Path(path).suffix == ".py":
            name = checksum+"_"+Path(path).name
            shutil.copyfile(path, snapshots / name); code[name] = checksum
    write_new(out / "pipeline_started.local.json", dict(configuration_path=str(config_path),
        configuration_sha256=digest, source_snapshots=code, **FLAGS))
    completed = []; parent = None
    try:
        for stage in ("surface", "remesh", "mesh"):
            if sha(config_path) != digest:
                raise ValueError("Pipeline configuration changed between stages")
            print(f"CT preparation: {stage}", flush=True)
            target = out / stage
            result = prepare(stage, config_path, target, parent)
            completed.append(dict(stage=stage, directory=str(target),
                manifest_sha256=sha(target / "preparation.local.json"), status=result["status"]))
            parent = target
        verify_artifact(parent)
        manifest = dict(stage="pipeline", status="prepared_not_validated", completed=completed,
            configuration_sha256=digest, mesh_directory=str(parent), geometry_and_shape_QC_passed=True,
            CEM_contacts_built=False, **FLAGS)
        write_new(out / "pipeline.local.json", manifest)
        return manifest
    except BaseException as error:
        write_new(out / "pipeline_failed.local.json", dict(status="failed", completed=completed,
            error_type=type(error).__name__, error=str(error), **FLAGS))
        raise


def save_fractions(out, fractions, outside, report, identity):
    for q in POINTS:
        savemat(out / f"fractions_q{q}.mat", dict(fractions=fractions[q], outside_ct_fraction=outside[q]))
    write_new(out / "material_inventory.local.json", dict(report, identity=identity, **FLAGS))


def build_refinement(cfg, source, out, ct, labels):
    kernel = kernels(); nodes, tets, boundary, _ = read_mesh(source, cfg)
    material = read(source / "material_inventory.local.json")
    if material["identity"] != material_identity(cfg):
        raise ValueError("Material source/rule identity changed; parent reuse is forbidden")
    if material["fraction_origin"] != "CT_mask_integration":
        raise ValueError("Parent reuse requires CT-integrated fractions, not inherited coefficients")
    pf, po = {}, {}
    for q in POINTS:
        m = loadmat(source / f"fractions_q{q}.mat")
        pf[q] = m["fractions"]; po[q] = m["outside_ct_fraction"].ravel()
    settings = cfg["refinement"]
    guides = np.asarray(settings["guides_lps_mm"], float)
    if guides.ndim != 2 or guides.shape[1] != 3 or not len(guides) or not np.isfinite(guides).all():
        raise ValueError("Explicit nonempty finite LPS refinement guides required")
    rounds = settings["promotion_rounds"]; promoted = settings["max_promoted"]
    if rounds != int(rounds) or promoted != int(promoted) or rounds < 0 or promoted < 0:
        raise ValueError("Promotion bounds must be nonnegative integers")
    selected = kernel["select_parents"](nodes, tets, guides, settings["radius_mm"])
    result, qc, _, attempts = kernel["refine_screened"](nodes, tets, boundary, selected, out,
        cfg["mesh"]["max_nodes"], rounds, promoted)
    fn, ft, fb, ep, bp, edges, cp, weights = result
    mesh_qc(fn, ft, fb, cfg["mesh"]["max_nodes"], cfg["mesh"]["max_tetrahedra"])
    nested = kernel["verify_nested_geometry"](nodes, tets, boundary, result,
                                            dict(nodes_mm=nodes, boundary=boundary+1), bp+1)
    # An arbitrary positive diagnostic coefficient, not fitted tissue physics.
    energy = kernel["inherited_field_energy_check"](nodes, tets, fn, ft, edges, ep,
                1+np.arange(len(tets), dtype=float)/max(1, len(tets)), cp, weights)
    if not energy["passed"]:
        raise ValueError("P1 inherited coefficient/energy identity failed")
    savemat(out / "parent_maps.mat", dict(element_parent=ep+1, boundary_parent=bp+1,
        midpoint_edges=edges+1, center_parent_ids=cp+1, center_barycentric_weights=weights))
    remapped, ro, report = integrate_materials(fn, ft, labels, ct.affine,
                                              parent=(nodes, tets, ep, pf, po))
    inherited = {q: pf[q][ep] for q in POINTS}; io = {q: po[q][ep] for q in POINTS}
    identity = material_identity(cfg)
    for name, fractions, outside in [("inherited", inherited, io), ("remapped", remapped, ro)]:
        target = out / name; target.mkdir()
        save_mesh(target, fn, ft, fb, dict(qc, **nested, **FLAGS),
                  source_file=out / "inherited/mesh.mat" if name == "remapped" else None)
        inventory = dict(report, fraction_origin="exact_parent_inheritance" if name == "inherited" else "CT_mask_integration",
            inventory={str(q): dict(volume_ml=(kernel["volumes"](fn, ft) @ fractions[q]/1000).tolist(),
                       sampled_outside_CT_volume_ml=float(kernel["volumes"](fn, ft) @ outside[q]/1000)) for q in POINTS})
        save_fractions(target, fractions, outside, inventory, identity)
        copy_contacts(source, target)
    write_new(out / "refinement_qc.local.json", dict(**nested, energy_check=energy,
        energy_coefficient_is_arbitrary_diagnostic=True, parent_mesh_sha256=sha(source / "mesh.mat"),
        boundary_map_is_to_immediate_parent=True, CEM_contact_faces_inherited=False,
        controls=["parent -> inherited: changed FEM space at inherited coefficients",
                  "inherited -> remapped: changed fractions on identical child mesh"],
        attempts=len(attempts), **FLAGS))


def prepare(stage, config_path, out, input_dir=None):
    if stage == "cem_contacts_input":
        from ct_fem_contacts import prepare as prepare_contacts
        return prepare_contacts(config_path, out, input_dir)
    config_path = Path(config_path).resolve(); config_digest = sha(config_path)
    cfg = load_config(config_path); out = Path(out).resolve()
    source = Path(input_dir).resolve() if input_dir is not None else None
    if out.exists():
        raise ValueError("Use a fresh output directory; no overwrites or automatic resume")
    if stage in CONTINUATION_STAGES and source is None:
        raise ValueError("This stage requires --input pointing to a completed preparation")
    if source is not None and (out.is_relative_to(source) or source.is_relative_to(out)):
        raise ValueError("Output must be separate from the input artifact")
    rawfiles = [Path(cfg["inputs"]["ct"]), *map(Path, cfg["inputs"]["masks"].values()), Path(config_path).resolve()]
    if cfg['body']['strategy'] in ('author_STL_LPS', 'supplied_STL_LPS'):
        rawfiles.append(Path(cfg['body']['source_stl']))
    if any(p.is_relative_to(out) for p in rawfiles):
        raise ValueError("Output cannot contain any source input")
    if source is not None:
        previous_digest = sha(source / "preparation.local.json")
        previous = verify_artifact(source)
        if previous["subject_alias"] != cfg["subject_alias"]:
            raise ValueError("Input artifact belongs to another subject alias")
        allowed = {"remesh": {"surface", "remesh"}, "mesh": {"surface", "remesh"}, "materials": {"mesh"},
                   "refine": {"materials", "refine_remapped"}}
        if stage in allowed and previous["stage"] not in allowed[stage]:
            raise ValueError("Input artifact is not a supported parent for this stage")
    # Pin bytes before loading arrays, including transitive parent dependencies.
    sources = artifact_sources(cfg, source, previous["sources"] if source is not None else None)
    if sources[str(config_path)] != config_digest:
        raise ValueError("Configuration changed while it was read")
    if source is not None:
        if sources[str(source / "preparation.local.json")] != previous_digest:
            raise ValueError("Parent manifest changed while it was read")
        if any(sources[path] != digest for path, digest in previous["sources"].items()):
            raise ValueError("Parent dependency changed before array loading")
        if any(sources[str((source / name).resolve())] != digest for name, digest in previous["products"].items()):
            raise ValueError("Parent product changed before array loading")
        prior_ct = previous["configuration"]["inputs"]["ct"]
        if previous["sources"][prior_ct] != sources[cfg["inputs"]["ct"]]:
            raise ValueError("Mesh/preparation and configured CT identity differ")
    contract = None; inherited_settings = {}
    if source is not None:
        inherited_settings = previous.get("completed_geometry_settings", {})
        if stage in CONTINUATION_STAGES:
            contract = check_continuation_contract(stage, cfg, source, previous, sources)
    if stage == "surface":
        if cfg['body']['strategy'] == 'HU_largest_component_per_slice':
            surface_distance_settings(cfg)
    ct, hu, masks, labels, input_qc = load_inputs(cfg)
    if stage == "remesh":
        from ct_fem_remesh import settings
        settings(cfg)
    out.mkdir(parents=True, exist_ok=False)
    try:
        if stage == "validate":
            write_new(out / "input_qc.local.json", dict(input_qc, shape=list(ct.shape),
                affine=ct.affine.tolist(), spacing_mm=[float(v) for v in ct.header.get_zooms()], **FLAGS))
        elif stage == "surface":
            build_surface(cfg, out, ct, hu, masks)
        elif stage == "remesh":
            from ct_fem_remesh import build
            build(cfg, source, out)
        elif stage == "mesh":
            build_mesh(cfg, source, out)
        elif stage == "materials":
            nodes, tets, boundary, qc = read_mesh(source, cfg)
            save_mesh(out, nodes, tets, boundary, qc, source_file=source / "mesh.mat")
            f, outside, report = integrate_materials(nodes, tets, labels, ct.affine)
            report["fraction_origin"] = "CT_mask_integration"
            save_fractions(out, f, outside, report, material_identity(cfg))
            copy_contacts(source, out)
        elif stage == "refine":
            build_refinement(cfg, source, out, ct, labels)
        else:
            raise ValueError("Unsupported preparation stage")
        verify_sources(sources)
        if stage == "refine":
            branch_sources = dict(sources)
            for name in ["parent_maps.mat", "refinement_qc.local.json"]:
                branch_sources[str(out / name)] = sha(out / name)
            for name in ["inherited", "remapped"]:
                publish(out / name, "refine_"+name, cfg, input_qc, branch_sources, inherited_settings, contract)
        manifest = publish(out, stage, cfg, input_qc, sources, inherited_settings, contract)
        return manifest
    except BaseException as error:
        write_new(out / "preparation_failed.local.json", dict(status="failed", error_type=type(error).__name__,
                  error=str(error), complete_manifest_written=False, **FLAGS))
        raise


def publish(out, stage, cfg, input_qc, sources, inherited_settings=None, continuation_contract=None):
    runtime = {}
    for name in ["numpy", "scipy", "nibabel", "trimesh", "scikit-image", "shapely", "pymeshlab", "manifold3d", "tetgen"]:
        try:
            runtime[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            runtime[name] = None
    products = {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file()}
    verify_sources(sources)
    completed = copy.deepcopy(inherited_settings or {})
    if stage == "surface":
        completed = {key: copy.deepcopy(cfg[key]) for key in ("body", "synthetic_cylinders", "surface_qc")}
    elif stage == "remesh":
        completed["remeshing"] = copy.deepcopy(cfg["remeshing"])
    elif stage == "mesh":
        completed["mesh_generation"] = {key: cfg["mesh"][key] for key in MESH_GENERATION_KEYS}
    elif stage in ("refine", "refine_inherited", "refine_remapped"):
        completed["refinement"] = copy.deepcopy(cfg["refinement"])
    manifest = dict(schema="ct_fem_preparation_artifact_v1", stage=stage,
        status="prepared_not_validated", subject_alias=cfg["subject_alias"],
        ct_provenance=cfg["inputs"]["ct_provenance"],
        mask_provenance=cfg["inputs"]["mask_provenance"], electrical_context=cfg["electrical_context"],
        configuration=cfg, input_qc=input_qc, runtime=runtime, sources=sources, products=products,
        configuration_contains_planned_settings=True, completed_geometry_settings=completed,
        continuation_contract=continuation_contract,
        product_files=[dict(path=str((out / name).resolve()), sha256=digest) for name, digest in products.items()],
        algorithm_loading="Unchanged historical definitions; parameterised full-section join copy",
        CEM_adapter_ready=False, **FLAGS)
    if stage in ("refine_inherited", "refine_remapped"):
        manifest["refinement_root_directory"] = str(out.parent.resolve())
    write_new(out / "preparation.local.json", manifest)
    return manifest
