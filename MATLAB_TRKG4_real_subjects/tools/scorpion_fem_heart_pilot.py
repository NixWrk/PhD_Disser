"""Own-CT five-material heart scenarios; geometry/postprocessing only, no solver.

All paths come from local JSON. ``prepare`` needs the canonical CT material
preparation (q1024/q4096). ``predict`` consumes checked MATLAB forward results.
The parent process alone schedules MATLAB assembly and sparse solves.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from scipy.io import loadmat, savemat
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import ct_fem_core as core
import prepare_static_heart_replacement_pilot as geometry

# These historical geometry helpers have no subject globals or FEM side effects.
geometry.np = np
ORDER = ["soft_background", "lung", "bone", "heart", "fat"]
ACTIVE = [0, 1, 3, 4]
LEVELS = (1024, 4096)
SEED = core.SEED


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def nested_barycentric():
    # Identical rule, seed and prefix order to ct_fem_core.integrate_materials.
    cuts = np.sort(qmc.Sobol(3, scramble=True, seed=SEED).random_base2(12), axis=1)
    return np.diff(np.c_[np.zeros(len(cuts)), cuts, np.ones(len(cuts))], axis=1)


def mask_sphere(mask, affine):
    """Exact volume/centroid of a union of full NIfTI voxels, in LPS millimetres."""
    require(mask.ndim == 3 and mask.dtype == np.bool_ and mask.any(), "Empty/nonbinary whole-heart mask")
    # Slice accumulation avoids an Nx3 allocation for large medical masks.
    count = 0
    moment = np.zeros(3)
    for z in range(mask.shape[2]):
        x, y = np.nonzero(mask[:, :, z])
        count += len(x)
        moment += [x.sum(dtype=np.float64), y.sum(dtype=np.float64), z * len(x)]
    affine = np.asarray(affine, float)
    require(affine.shape == (4, 4) and np.isfinite(affine).all(), "Invalid mask affine")
    volume_mm3 = count * abs(float(np.linalg.det(affine[:3, :3])))
    require(volume_mm3 > 0, "Singular mask affine")
    centre = (affine[:3, :3] @ (moment / count) + affine[:3, 3]) * [-1, -1, 1]
    return dict(volume_ml=volume_mm3 / 1000, centre_lps_mm=centre.tolist(),
                radius_mm=float(np.cbrt(3 * volume_mm3 / (4 * np.pi))), voxels=count,
                definition="full whole-heart mask voxel union; not a chamber or stroke volume")


def frozen_background(fractions):
    f = np.asarray(fractions, float)
    require(f.ndim == 2 and f.shape[1] == 5 and np.isfinite(f).all() and
            np.all(f >= 0) and np.array_equal(f.sum(1), np.ones(len(f))), "Invalid five-material fractions")
    result = f.copy()
    result[:, 0] += result[:, 3]
    result[:, 3] = 0
    return result


def sampled_labels(points_lps_mm, labels, affine):
    voxel = core.lps_to_voxel(points_lps_mm, affine)
    return labels_at_voxels(voxel, labels)


def labels_at_voxels(voxel, labels):
    ijk = np.floor(voxel + .5).astype(np.int64)
    valid = np.all((ijk >= 0) & (ijk < labels.shape), axis=-1)
    answer = np.zeros(valid.shape, np.uint8)
    loc = ijk[valid]
    answer[valid] = labels[loc[:, 0], loc[:, 1], loc[:, 2]]
    return answer, ~valid


def sample_overlay(vertices_mm, labels, affine, centre_mm, radius_mm, bary=None):
    """Joint CT/background/sphere sampling, never a product of marginal fractions."""
    bary = nested_barycentric() if bary is None else np.asarray(bary, float)
    require(bary.shape == (4096, 4), "Expected nested q4096 rule")
    points = np.einsum("qj,bjk->bqk", bary, vertices_mm, optimize=True)
    # Match the canonical operation order before nearest-voxel rounding;
    # affine(point) and interpolating affine(vertices) can differ by an ulp.
    voxel_vertices = core.lps_to_voxel(vertices_mm, affine)
    voxel_points = np.einsum("qj,bjk->bqk", bary, voxel_vertices, optimize=True)
    original, outside = labels_at_voxels(voxel_points, labels)
    background = np.where(original == 3, 0, original)
    inside = np.sum((points - centre_mm) ** 2, axis=2) <= radius_mm ** 2
    overlaid = np.where(inside, 3, background)
    result, baseline, overlap = {}, {}, {}
    for q in LEVELS:
        result[q] = np.column_stack([(overlaid[:, :q] == k).mean(1) for k in range(5)])
        baseline[q] = np.column_stack([(original[:, :q] == k).mean(1) for k in range(5)])
        overlap[q] = np.column_stack([(inside[:, :q] & (background[:, :q] == k)).mean(1) for k in range(5)])
    return result, baseline, overlap, {q: outside[:, :q].mean(1) for q in LEVELS}


def possible_sphere_tets(vertices, centre, radius):
    low, high = vertices.min(1), vertices.max(1)
    nearest = np.maximum(np.maximum(low - centre, centre - high), 0)
    margin = 128 * np.finfo(float).eps * max(1., radius ** 2, float(np.max(abs(vertices))) ** 2)
    return np.sum(nearest ** 2, axis=1) <= radius ** 2 + margin


def prepare(config_path, output, chunk=32):
    require(isinstance(chunk, int) and chunk > 0, "Positive chunk required")
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = read(config_path)
    require(config.get("schema") == "scorpion_heart_pilot_config_v1", "Wrong pilot config schema")
    def local(key):
        return (config_path.parent / config[key]).resolve()
    subject = config["subject_alias"]
    require(subject in ("georg", "nik", "yaroslav"), "Unknown subject")
    cfg = core.load_config(local("geometry_config"))
    require(cfg["subject_alias"] == subject and cfg["electrical_context"]["frequency_hz"] == 100000,
            "Subject/frequency mismatch")
    materials = local("materials_dir")
    manifest = core.verify_artifact(materials)
    require(manifest["subject_alias"] == subject and manifest["stage"] == "materials", "Wrong material preparation")
    inventory = read(materials / "material_inventory.local.json")
    require(inventory["order"] == ORDER and inventory["identity"]["rule"] == cfg["materials"], "Wrong CT material identity/rule")
    for p in [cfg["inputs"]["ct"], *cfg["inputs"]["masks"].values()]:
        require(inventory["identity"]["inputs"].get(str(Path(p).resolve())) == sha(p), "CT/mask differs from own canonical materials")
    require(not output.exists(), "Use a fresh heart output directory")
    ct, hu, masks, labels, ct_qc = core.load_inputs(cfg)
    del hu
    sphere = mask_sphere(masks["heart"], ct.affine)
    del masks
    meshfile = materials / "mesh.mat"
    mesh = loadmat(meshfile)
    nodes, elements = mesh["nodes_mm"], mesh["elems"]
    require(np.isfinite(nodes).all() and np.issubdtype(elements.dtype, np.integer) and
            elements.ndim == 2 and elements.shape[1] == 4 and elements.min() >= 1 and elements.max() <= len(nodes),
            "Invalid original element IDs")
    tets = elements.astype(np.int64) - 1
    n = len(tets)
    original = {q: loadmat(materials / f"fractions_q{q}.mat")["fractions"] for q in LEVELS}
    for f in original.values():
        require(f.shape == (n, 5), "Fraction row count differs from original elements")
        frozen_background(f)
    volumes = np.empty(n)
    candidates = []
    centre = np.array(sphere["centre_lps_mm"])
    rmax = sphere["radius_mm"] * np.cbrt(1.05)
    contains_centre = False
    for start in range(0, n, 65536):
        v = nodes[tets[start:start+65536]]
        det = np.einsum("ij,ij->i", v[:, 1]-v[:, 0], np.cross(v[:, 2]-v[:, 0], v[:, 3]-v[:, 0]))
        require(np.all(det > 0), "Nonpositive tetrahedron determinant")
        volumes[start:start+len(v)] = det / 6000
        candidates.append(start + np.flatnonzero(possible_sphere_tets(v, centre, rmax)))
        if not contains_centre:
            contains_centre = geometry.point_in_tets(centre, v)
    require(contains_centre, "Sphere centre outside original tetrahedral domain")
    faces = mesh["boundary"].astype(np.int64) - 1
    minimum = np.inf
    for start in range(0, len(faces), 65536):
        minimum = min(minimum, float(geometry.triangle_origin_distance_squared(nodes[faces[start:start+65536]]-centre).min()))
    clearance = float(np.sqrt(minimum) - rmax)
    require(clearance > 1e-8, "Largest sphere crosses/touches original body boundary")
    selected = np.concatenate(candidates)
    sources = [config_path, local("geometry_config"), meshfile, materials / "preparation.local.json",
               materials / "material_inventory.local.json",
               local("reference_materials"), Path(__file__), ROOT / "tools/ct_fem_core.py",
               Path(geometry.__file__)]
    sources += [Path(cfg["inputs"]["ct"]), *map(Path, cfg["inputs"]["masks"].values())]
    sources += [materials / f"fractions_q{q}.mat" for q in LEVELS]
    identity = [dict(path=str(p), sha256=sha(p)) for p in sources]
    require(1 <= config.get("threads", 1) <= 6 and int(config.get("threads", 1)) == config.get("threads", 1), "Threads must be 1..6")
    output.mkdir(parents=True)
    bary = nested_barycentric()
    variants, rows = [], []
    element_ids = np.arange(1, n+1, dtype=np.int32)[:, None]
    try:
        for name, factor in [("wholeheart_ct", None), ("sphere_v095", .95), ("sphere_v100", 1.), ("sphere_v105", 1.05)]:
            fractions = {q: original[q].copy() if factor is None else frozen_background(original[q]) for q in LEVELS}
            overlap_ml = {q: np.zeros(5) for q in LEVELS}
            if factor is not None:
                radius = sphere["radius_mm"] * np.cbrt(factor)
                for start in range(0, len(selected), chunk):
                    ids = selected[start:start+chunk]
                    sampled, reproduced, overlap, _ = sample_overlay(nodes[tets[ids]], labels, ct.affine, centre, radius, bary)
                    for q in LEVELS:
                        require(np.array_equal(reproduced[q], original[q][ids]), "Canonical own CT fractions not reproduced exactly")
                        fractions[q][ids] = sampled[q]
                        overlap_ml[q] += volumes[ids] @ overlap[q]
            for q in LEVELS:
                f = fractions[q]
                require(np.array_equal(f.sum(1), np.ones(n)) and np.all(f >= 0), "Partition failed")
                volume = volumes @ f
                nameq = f"{name}_q{q}"
                target = output / (nameq + ".mat")
                savemat(target, dict(fractions=f, element_ids=element_ids), do_compression=False)
                variants.append(dict(id=nameq, geometry=name, q=q, file=str(target), sha256=sha(target)))
                rows.append(dict(id=nameq, volume_ml_by_material=volume.tolist(),
                    target_heart_volume_ml=None if factor is None else sphere["volume_ml"]*factor,
                    overlaid_background_volume_ml=overlap_ml[q].tolist(),
                    sphere_volume_error_ml=None if factor is None else float(volume[3]-sphere["volume_ml"]*factor)))
            print(f"{subject}: {name} CT fractions prepared (q1024/q4096)", flush=True)
        for item in identity:
            require(sha(item["path"]) == item["sha256"], "Input changed during preparation")
        plan = dict(schema="scorpion_heart_pilot_plan_v1", subject_alias=subject, frequency_hz=100000,
            material_order=ORDER, active_material_indices_one_based=[1, 2, 4, 5], source_files=identity,
            mesh_mat=str(meshfile), mesh_sha256=sha(meshfile),
            cem_mat=str(local("cem_mat")), baseline_forward=str(local("baseline_forward")),
            baseline_prepared=str(local("baseline_prepared")), reference_materials=str(local("reference_materials")),
            variants=variants, case_ids=config.get("case_ids", []), threads=int(config.get("threads", 1)),
            geometry=dict(sphere=sphere, rows=rows, ct_qc=ct_qc, boundary_clearance_mm=clearance,
                original_element_count=n, scanned_elements=n, selected_elements=len(selected),
                centroid_filter_used=False, nested_points=list(LEVELS), seed=SEED,
                unchanged_mesh=True, wholeheart_mask_inhalation=True,
                rule="replace original material heart by soft; overlay sphere with priority on frozen background"),
            FEM_was_run=False, quantitative_result_accepted=False, clinical_validation=False,
            shape_or_position_optimized=False, uncertainty_model="not estimated; scenario amplitudes only",
            reader_tables_default_open=False)
        require(1 <= plan["threads"] <= 6, "Threads must be 1..6")
        write(output / "plan.local.json", plan)
        return plan
    except Exception as exc:
        write(output / "failed.local.json", dict(status="geometry_preparation_failed", reason=str(exc), FEM_was_run=False))
        raise


def pulse_prediction(jacobian_logrho, delta_logrho):
    j, amplitude = np.asarray(jacobian_logrho, float), np.asarray(delta_logrho, float)
    require(j.ndim == 3 and j.shape[2] == 4 and np.isfinite(j).all(), "Expected four log-rho derivatives")
    require(amplitude.shape == (4,) and np.isfinite(amplitude).all(), "Explicit four scenario delta_logrho required")
    contributions = j * amplitude[None, None, :]
    return contributions.sum(2), contributions


def predict(forward, scenarios):
    require(forward.get("FEM_was_run") is True and forward.get("frequency_hz") == 100000, "No own 100kHz baseline")
    require(forward.get("subject_alias") == scenarios.get("subject_alias"), "Cross-subject prediction forbidden")
    checks = forward.get("derivative_checks", {})
    require(checks.get("passed") is True, "Baseline derivatives not checked")
    active = forward.get("active_material_indices", checks.get("active_material_indices"))
    require(active == [1, 2, 4, 5], "Derivative material order must be explicit")
    require(forward.get("system_checks") and all(
        c["system_relative_residual"] < 1e-7 and c["reciprocity_ohm"] < 1e-7
        for c in forward["system_checks"]), "Baseline residual/reciprocity failed")
    require(scenarios.get("amplitude_status") == "scenario_not_measured", "Declare amplitudes as scenario_not_measured")
    rows, skipped = [], []
    answers = forward["answers"]
    keys = [(r["case_id"], r["state"], r["rule"]) for r in answers]
    require(len(keys) == len(set(keys)), "Duplicate baseline keys")
    lookup = dict(zip(keys, answers))
    for inhale in answers:
        if inhale["state"] != "inhale":
            continue
        key = (inhale["case_id"], "exhale", inhale["rule"])
        if key not in lookup:
            skipped.append(dict(case_id=inhale["case_id"], rule=inhale["rule"], reason="No paired exhale prediction"))
            continue
        exhale = lookup[key]
        ri, re = np.asarray(inhale["rho_ohm_m"], float), np.asarray(exhale["rho_ohm_m"], float)
        require(ri.shape == re.shape == (5,) and np.all(ri > 0) and np.all(re > 0), "Invalid rho vectors")
        require(ri[2] == re[2], "Bone must remain fixed")
        delta = np.log(re[ACTIVE] / ri[ACTIVE])
        linear, by_material = pulse_prediction(inhale["jacobian_logrho"], delta)
        exact = np.asarray(exhale["transfer_matrix_ohm"]) - np.asarray(inhale["transfer_matrix_ohm"])
        pulses = []
        for scenario in scenarios["pulses"]:
            total, components = pulse_prediction(inhale["jacobian_logrho"], scenario["delta_logrho"])
            pulses.append(dict(id=scenario["id"], delta_logrho=scenario["delta_logrho"],
                delta_transfer_ohm=total.tolist(), components_ohm=components.tolist(),
                amplitude_status="scenario_not_measured", linearization_error="not checked by this postprocessor"))
        rows.append(dict(case_id=inhale["case_id"], rule=inhale["rule"],
            inhale_transfer_ohm=inhale["transfer_matrix_ohm"], exhale_transfer_ohm=exhale["transfer_matrix_ohm"],
            exhale_minus_inhale_ohm=exact.tolist(), linear_delta_ohm=linear.tolist(),
            linear_components_ohm=by_material.tolist(),
            linear_lung_only_ohm=by_material[:, :, 1].tolist(),
            linear_background_only_ohm=by_material[:, :, [0, 3]].sum(2).tolist(),
            linear_wholeheart_property_only_ohm=by_material[:, :, 2].tolist(),
            nonlinear_minus_linear_ohm=(exact-linear).tolist(), pulses=pulses))
    return dict(schema="scorpion_heart_scenarios_v1", subject_alias=forward["subject_alias"], rows=rows,
        skipped=skipped, material_order_active=[ORDER[k] for k in ACTIVE],
        geometry="fixed deep-inhale CT; exhale is conditional material scenario",
        operator="own_drive primary; cross-transfers retained separately; relative source phase unknown",
        baseline_integration_passed=forward.get("integration_passed"),
        quantitative_result_accepted=False, clinical_validation=False, stroke_volume_estimated=False,
        reader_tables_default_open=False)


def summarize(directory):
    """Keep missing/failed attempts visible; three spheres do not establish an inverse."""
    directory = Path(directory)
    plan = read(directory / "plan.local.json")
    solved, missing = {}, []
    for entry in plan["variants"]:
        target = directory / ("solve_" + entry["id"])
        path = target / "report.local.json"
        if not path.exists():
            failure = target / "failed.local.json"
            missing.append(dict(variant=entry["id"], reason=read(failure) if failure.exists() else "not run"))
            continue
        result = read(path)
        require(result["identity"]["plan_sha256"] == sha(directory / "plan.local.json"), "Result belongs to another plan")
        require(result["subject_alias"] == plan["subject_alias"], "Cross-subject result")
        for row in result["answers"]:
            key = (entry["geometry"], entry["q"], row["case_id"])
            require(key not in solved, "Duplicate pilot result")
            solved[key] = row
    comparisons, slopes, quadrature = [], [], []
    all_cases = sorted({k[2] for k in solved})
    volume0 = plan["geometry"]["sphere"]["volume_ml"]
    for case in all_cases:
        for q in LEVELS:
            for name in ("wholeheart_ct", "sphere_v100"):
                row = solved.get((name, q, case))
                if row:
                    difference = np.asarray(row["delta_from_wholeheart_inhale_ohm"])
                    comparisons.append(dict(case_id=case, q=q, geometry=name,
                        delta_from_original_wholeheart_ohm=difference.tolist(),
                        baseline_reproduction_passed=bool(np.max(abs(difference)) < 1e-7) if name == "wholeheart_ct" else None))
            plus, minus = solved.get(("sphere_v105", q, case)), solved.get(("sphere_v095", q, case))
            if plus and minus:
                difference = np.asarray(plus["transfer_matrix_ohm"]) - np.asarray(minus["transfer_matrix_ohm"])
                volumes = {r["id"]: r["volume_ml_by_material"][3] for r in plan["geometry"]["rows"]}
                actual = volumes[f"sphere_v105_q{q}"] - volumes[f"sphere_v095_q{q}"]
                slopes.append(dict(case_id=case, q=q, target_delta_volume_ml=.1*volume0,
                    integrated_delta_volume_ml=actual, minus_to_plus_delta_ohm=difference.tolist(),
                    secant_ohm_per_target_ml=(difference/(.1*volume0)).tolist(),
                    secant_ohm_per_integrated_ml=None if actual <= 0 else (difference/actual).tolist(),
                    status="fixed-centre whole-heart secant; not stroke volume; step convergence not tested"))
        for name in ("wholeheart_ct", "sphere_v095", "sphere_v100", "sphere_v105"):
            fine, coarse = solved.get((name, 4096, case)), solved.get((name, 1024, case))
            if fine and coarse:
                difference = np.asarray(fine["transfer_matrix_ohm"]) - np.asarray(coarse["transfer_matrix_ohm"])
                quadrature.append(dict(case_id=case, geometry=name, fine_minus_coarse_ohm=difference.tolist(),
                    maximum_absolute_difference_ohm=float(np.max(abs(difference))),
                    acceptance="no accuracy budget specified; global mesh convergence not established"))
    return dict(schema="scorpion_heart_pilot_summary_v1", subject_alias=plan["subject_alias"],
        comparisons=comparisons, volume_secants=slopes, quadrature=quadrature, incomplete_or_failed=missing,
        status="incomplete" if missing else "exploratory_forward_complete",
        shape_or_position_optimized=False, inverse_identifiability="not established by three volume states",
        clinical_validation=False, quantitative_result_accepted=False, stroke_volume_estimated=False,
        reader_tables_default_open=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--chunk", type=int, default=32)
    p = sub.add_parser("summarize")
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("predict")
    p.add_argument("--forward", type=Path, required=True)
    p.add_argument("--scenarios", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.config, args.output, args.chunk)
    elif args.command == "predict":
        result = predict(read(args.forward), read(args.scenarios))
        result["sources"] = [dict(path=str(p.resolve()), sha256=sha(p)) for p in (args.forward, args.scenarios, Path(__file__))]
        write(args.output, result)
    else:
        write(args.output, summarize(args.directory))


if __name__ == "__main__":
    main()
