"""Stage-one original-heart volume/translation probes on a sealed CEM mesh.

Reuses the fractional EIDORS exporter and checked solver, without modifying
the historical sphere/ellipsoid pilot. No CT registration or physiological
motion law is estimated. Paths and raw arrays belong to the local output.
"""
from pathlib import Path
import argparse
import os
import sys
import time

import numpy as np
import heart_replacement_electrical_pilot as core
import heart_fractional_electrical_pilot as fractional
from heart_fractional_materials import safe_box_overlap, uniform_tet_barycentric, integrate_fractions
from heart_individual_fractional_materials import TetrahedralHeart, integrate_individual_fractions

ROOT = Path(__file__).resolve().parents[1]
read, write, sha, require = core.read, core.write, core.sha, core.require


def state_design(parameter, steps):
    require(parameter in ("volume", "translation"), "Unknown parameter")
    require(steps and all(np.isfinite(s) and s > 0 for s in steps), "Positive steps required")
    require(len(set(steps)) == len(steps), "Duplicate steps")
    rows = [dict(id="base", parameter="baseline", signed_step=0.0,
                 volume_fraction=0.0, translation_m=[0.0, 0.0, 0.0])]
    for axis in ([None] if parameter == "volume" else range(3)):
        for step in steps:
            for sign in (-1, 1):
                translation = [0.0, 0.0, 0.0]
                if axis is not None:
                    translation[axis] = sign * step / 1000
                value = sign * step
                rows.append(dict(
                    id=("volume" if axis is None else "xyz"[axis]) + "_" +
                       ("m" if sign < 0 else "p") + format(step, ".8g").replace(".", "p"),
                    parameter="volume_fraction" if axis is None else "xyz"[axis] + "_mm",
                    signed_step=value, volume_fraction=value if axis is None else 0.0,
                    translation_m=translation))
    return rows


def select_mesh(mesh, bounds):
    """Vertex-box selection scans every tet, retaining original material IDs."""
    import h5py
    ids, vertices, volumes, materials = [], [], [], []
    with h5py.File(mesh) as f:
        nodes = f["fmdl_m/nodes"][:].T
        elements = f["fmdl_m/elems"]
        labels = f["tissue_id"][:].ravel().astype(np.uint8)
        require(np.isin(labels, [1, 2, 3, 4]).all(), "Unknown material")
        total = 0.0
        for start in range(0, len(labels), 65536):
            e = elements[:, start:start + 65536].T.astype(np.int64) - 1
            v = nodes[e]
            determinants = np.einsum("ij,ij->i", v[:, 1] - v[:, 0],
                                    np.cross(v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]))
            require(np.all(determinants > 0), "Nonpositive tetrahedral determinant")
            ml = determinants / 6 * 1e6
            keep = safe_box_overlap(v, *bounds)
            ids.append(start + np.flatnonzero(keep))
            vertices.append(v[keep])
            volumes.append(ml[keep])
            materials.append(labels[start:start + len(e)][keep])
            total += float(ml.sum())
    return (np.concatenate(ids), np.concatenate(vertices), np.concatenate(volumes),
            np.concatenate(materials), labels, total)


def prepare(args):
    base_dir = args.baseline.resolve()
    out = args.output.resolve()
    require(not out.exists() or not any(out.iterdir()), "Use a new output directory")
    base = core.load_plan(base_dir)
    runner = core.configure(Path(base["runtime"]["deps"]), base["threads"])
    geometry_mesh = Path(base["source_prepared"])
    original_mesh_sha256 = base["source_prepared_sha256"]
    mesh = geometry_mesh
    require(sha(mesh) == original_mesh_sha256, "Original mesh changed")
    contact = next(c.copy() for c in base["contacts"] if c["id"] == "tepc_2")
    extra_inputs = []
    if args.refined_export:
        export_dir = args.refined_export.resolve()
        record = read(export_dir / "heart_refinement_export.json")
        require(record["status"] == "passed_heart_refinement_export" and
                record["source_prepared_sha256"] == original_mesh_sha256,
                "Refinement does not belong to the original model")
        require(record["original_nodes_preserved"] and record["outer_boundary_preserved"] and
                record["contact_faces_preserved"] and record["contact_embedding_relative_error"] < 1e-12,
                "Refinement changed contact geometry")
        mesh = export_dir / record["prepared_file"]
        require(sha(mesh) == record["prepared_sha256"], "Refined mesh changed")
        contact.update(path=str(export_dir / record["contact_file"]), sha256=record["contact_sha256"])
        base.update(source_prepared=str(mesh), source_prepared_sha256=record["prepared_sha256"],
                    node_count=record["nodes"], element_count=record["elements"],
                    ground_zero_based=record["ground_zero_based"])
        extra_inputs = [export_dir / "heart_refinement_export.json", geometry_mesh]
    require(base["contact"]["diameter_mm"] == 5 and base["frequency_hz"] == 50000,
            "This stage requires original 5 mm / 50 kHz CEM")
    out.mkdir(parents=True, exist_ok=True)
    inputs = {}
    for path, digest in [(base_dir / "plan.json", None),
                         (mesh, base["source_prepared_sha256"]),
                         (contact["path"], contact["sha256"])]:
        fractional.bind(inputs, path, digest)
    for path in extra_inputs:
        fractional.bind(inputs, path)
    if args.candidate_parameters:
        fractional.bind(inputs, args.candidate_parameters.resolve())
    with runner.h5py.File(geometry_mesh) as f:
        nodes = f["fmdl_m/nodes"][:].T
        units = "".join(chr(int(c)) for c in f["coordinate_units"][:].ravel())
        require(units == "m", "Prepared coordinates must be metres")
        labels = f["tissue_id"][:].ravel()
        heart_ids = np.flatnonzero(labels == 3)
        connectivity = f["fmdl_m/elems"][:, heart_ids].T.astype(np.int64) - 1
        heart = TetrahedralHeart(nodes[connectivity])
    designs = state_design(args.parameter, args.steps)
    candidate = None
    if args.geometry != "individual":
        require(args.candidate_parameters is not None, "Analytic geometry needs candidate parameters")
        candidate = read(args.candidate_parameters)["candidates"][args.geometry]
        from heart_fractional_materials import candidate_bounds
        centre = np.asarray(candidate["centre_m"])
        lo, hi = candidate_bounds(candidate)
        require(abs(candidate["analytic_volume_ml"] - heart.volume_ml) < 1e-6,
                "Candidate volume differs from original heart")
    else:
        centre = heart.centre_m
        lo, hi = np.asarray(heart.bounds_m)
    bounds = []
    for d in designs:
        require(d["volume_fraction"] > -1, "Nonpositive volume")
        scale = (1 + d["volume_fraction"]) ** (1 / 3)
        shift = np.asarray(d["translation_m"])
        bounds.append((centre + scale * (lo - centre) + shift,
                       centre + scale * (hi - centre) + shift))
    union = (np.min([b[0] for b in bounds], axis=0),
             np.max([b[1] for b in bounds], axis=0))
    ids, vertices, ml, selected_labels, labels, total = select_mesh(mesh, union)
    require(len(labels) == base["element_count"], "Query mesh dimension mismatch")
    if args.geometry == "individual":
        require(np.isin(np.flatnonzero(labels == 3), ids).all(), "Original heart omitted")
    background = np.where(labels == 3, 1, labels).astype(np.uint8)
    runner.sio.savemat(out / "background.mat", {"background": background[:, None]},
                       do_compression=True)
    levels = tuple(sorted(set(args.levels)))
    require(args.electrical_level in levels, "Electrical quadrature absent")
    bary = uniform_tet_barycentric(max(levels))
    states, geometry_rows = [], []
    write(out / "status.json", dict(status="preparing", selected_tetrahedra=len(ids),
          input_heart_tetrahedra=len(heart_ids), pid=os.getpid()))
    for d in designs:
        tick = time.time()
        if candidate is None:
            fractions, diagnostics = integrate_individual_fractions(
                vertices, heart, volume_fraction=d["volume_fraction"],
                translation_m=d["translation_m"], levels=levels,
                barycentric=bary, batch_size=args.batch_size)
        else:
            transformed = dict(candidate,
                centre_m=(centre + np.asarray(d["translation_m"])).tolist(),
                semiaxes_m=(np.asarray(candidate["semiaxes_m"]) *
                            np.cbrt(1 + d["volume_fraction"])).tolist())
            fractions, diagnostics = integrate_fractions(vertices, transformed, levels=levels,
                barycentric=bary, batch_size=args.batch_size)
        analytic = heart.volume_ml * (1 + d["volume_fraction"])
        if d["parameter"] == "baseline" and candidate is None:
            require(np.array_equal(fractions, np.broadcast_to(selected_labels == 3, fractions.shape)),
                    "Identity membership differs from original conforming mesh")
        for j, n in enumerate(levels):
            volume = float(ml @ fractions[j])
            geometry_rows.append(dict(state=d["id"], parameter=d["parameter"],
                signed_step=d["signed_step"], points_per_tet=n,
                analytic_volume_ml=analytic, material_volume_ml=volume,
                volume_error_percent=100 * (volume / analytic - 1)))
        np.savez_compressed(out / (d["id"] + "_geometry.npz"),
                            levels=levels, fractions=fractions, element_indices=ids)
        j = levels.index(args.electrical_level)
        fraction = np.zeros(len(labels), dtype=np.float32)
        fraction[ids] = fractions[j]
        sid = args.geometry + "_q" + str(args.electrical_level) + "_" + d["id"]
        file = out / (sid + "_fraction.mat")
        runner.sio.savemat(file, {"heart_fraction": fraction[:, None]}, do_compression=True)
        states.append(dict(d, id=sid, geometry=args.geometry,
            points_per_tet=args.electrical_level, fraction_mat=file.name,
            fraction_mat_sha256=sha(file), analytic_volume_ml=analytic,
            material_volume_ml=float(ml @ fractions[j]),
            geometry_npz_sha256=sha(out / (d["id"] + "_geometry.npz"))))
        write(out / "geometry_progress.json", dict(completed=len(states), required=len(designs),
              last_state=sid, state_seconds=time.time() - tick))
        print(sid, "geometry seconds", round(time.time() - tick, 2), flush=True)
    write(out / "geometry_qc.json", dict(
        original_volume_ml=heart.volume_ml, original_centre_m=heart.centre_m.tolist(),
        original_heart_tetrahedra=len(heart_ids), selected_tetrahedra=len(ids),
        all_tetrahedra_scanned=len(labels), total_domain_volume_ml=total,
        original_identity_exact=True if candidate is None else None, union_is_not_convex_hull=True,
        geometry=args.geometry, original_geometric_source_sha256=original_mesh_sha256,
        quadrature_levels=list(levels), rows=geometry_rows))
    code = [Path(__file__), ROOT / "tools/heart_individual_fractional_materials.py",
            ROOT / "tools/heart_fractional_materials.py",
            ROOT / "tools/heart_fractional_electrical_pilot.py",
            ROOT / "tools/heart_replacement_electrical_pilot.py",
            ROOT / "tools/run_electrode_sensitivity.py",
            ROOT / "src/run_heart_fractional_electrical_pilot.m",
            ROOT / "src/trkg4_cem_contact_matrix.m", ROOT / "src/trkg4_file_sha256.m"]
    p = {k:base[k] for k in (
        "source_prepared", "source_prepared_sha256", "runtime", "threads",
        "memory_limit_gib", "node_count", "element_count", "ground_zero_based",
        "rho_ohm_m", "conductivity_by_tissue_id", "criteria", "frequency_hz")}
    p.update(schema="heart_fractional_electrical_pilot_v1", stage="individual_geometry_stage1",
        states=states, inputs=list(inputs.values()), contact=contact,
        electrode_model=base["contact"], parameter=args.parameter, geometry=args.geometry,
        original_geometric_source_sha256=original_mesh_sha256,
        geometry_qc_sha256=sha(out / "geometry_qc.json"),
        nominal_volume_ml=heart.volume_ml,
        nominal_centre_m=heart.centre_m.tolist(),
        background_mat_sha256=sha(out / "background.mat"),
        implementation=[dict(path=str(f.resolve()), sha256=sha(f)) for f in code],
        scientific_status="original_individual_geometry_probe_not_physically_validated",
        field_approximation="unchanged P1 mesh; spatial convergence not yet assessed",
        deformation="uniform scaling about original tetra-union volume centroid; translation separate")
    write(out / "plan.json", p)
    (out / "plan.sha256").write_text(sha(out / "plan.json"), encoding="ascii")
    write(out / "status.json", dict(status="ready_for_weighted_assembly",
                                   states=len(states), FEM_solved=False))


def summarize(out, p):
    rows = [r for s in p["states"] if (r := fractional.checked_result(out, p, s)) is not None]
    baseline = next((r for r in rows if r["parameter"] == "baseline"), None)
    derivatives = []
    for plus in rows:
        if plus["signed_step"] <= 0:
            continue
        minus = next((r for r in rows if r["parameter"] == plus["parameter"] and
                      r["signed_step"] == -plus["signed_step"]), None)
        if minus is None or baseline is None:
            continue
        dv = plus["parameter"] == "volume_fraction"
        step = plus["signed_step"]
        denominator = 2 * step * (baseline["analytic_volume_ml"] if dv else 1)
        derivatives.append(dict(parameter=plus["parameter"], step=step,
            units="ohm/ml" if dv else "ohm/mm",
            derivative=(plus["Z_ohm"] - minus["Z_ohm"]) / denominator,
            material_volume_derivative=(plus["material_volume_ml"] - minus["material_volume_ml"]) / denominator,
            expected_material_volume_derivative=1.0 if dv else 0.0))
    step_checks = []
    for parameter in sorted({d["parameter"] for d in derivatives}):
        selected = sorted([d for d in derivatives if d["parameter"] == parameter],
                          key=lambda d: d["step"])
        for fine, coarse in zip(selected, selected[1:]):
            difference = coarse["derivative"] - fine["derivative"]
            step_checks.append(dict(parameter=parameter, fine_step=fine["step"],
                coarse_step=coarse["step"], derivative_difference=difference,
                relative_difference=None if abs(fine["derivative"]) < 1e-14
                    else abs(difference / fine["derivative"])))
    result = dict(status="complete" if len(rows) == len(p["states"]) else "partial",
        completed=len(rows), required=len(p["states"]), rows=rows, derivatives=derivatives,
        step_checks=step_checks,
        spatial_convergence=False, physical_validation=False,
        source_prepared_sha256=p["source_prepared_sha256"], plan_sha256=sha(out / "plan.json"))
    write(out / "summary.json", result)
    return result


def run(args):
    out = args.output.resolve()
    p = fractional.load_plan(out)
    core.configure(Path(p["runtime"]["deps"]), p["threads"])
    lock = out / "run.lock"
    with lock.open("x", encoding="ascii") as f:
        f.write(str(os.getpid()))
    try:
        if any(not (out / s["id"] / "export.json").exists() for s in p["states"]):
            quote = lambda x: "'" + str(x).replace("\\", "/").replace("'", "''") + "'"
            command = "run(" + quote(p["runtime"]["eidors_startup"]) + ");addpath(" + quote(ROOT / "src") + ");"
            command += "run_heart_fractional_electrical_pilot(" + quote(out / "plan.json") + ");"
            core.monitored([p["runtime"]["matlab"], "-batch", command], out,
                           "weighted_export", p["memory_limit_gib"])
        for state in p["states"]:
            fractional.checked_export(out, p, state)
            if fractional.checked_result(out, p, state) is not None:
                continue
            core.monitored([p["runtime"]["python"], "-B", "-X", "utf8",
                str(Path(__file__).resolve()), "worker", "--output", str(out),
                "--state", state["id"]], out, "solve_" + state["id"], p["memory_limit_gib"])
            summarize(out, p)
        result = summarize(out, p)
        require(result["status"] == "complete", "Incomplete stage")
        write(out / "status.json", dict(status="complete", states=len(p["states"]),
                                       FEM_solved=True, spatial_convergence=False))
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--refined-export", type=Path)
    p.add_argument("--geometry", choices=["individual", "sphere", "ellipsoid"], default="individual")
    p.add_argument("--candidate-parameters", type=Path)
    p.add_argument("--parameter", choices=["volume", "translation"], default="volume")
    p.add_argument("--steps", type=float, nargs="+", default=None)
    p.add_argument("--levels", type=int, nargs="+", default=[512, 4096])
    p.add_argument("--electrical-level", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=32)
    for mode in ("run", "summarize", "worker"):
        p = sub.add_parser(mode)
        p.add_argument("--output", type=Path, required=True)
        if mode == "worker":
            p.add_argument("--state", required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.steps is None:
            args.steps = [.005, .01] if args.parameter == "volume" else [1.0, 2.0]
        prepare(args)
    elif args.mode == "run":
        run(args)
    elif args.mode == "worker":
        fractional.worker(args)
    else:
        summarize(args.output.resolve(), fractional.load_plan(args.output.resolve()))


if __name__ == "__main__":
    main()
