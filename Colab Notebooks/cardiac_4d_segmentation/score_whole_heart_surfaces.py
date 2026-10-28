"""Compare all fitted forms using the same two area-sampled target surfaces.

Distances are one-sided (target surface to candidate), in physical mm.
Analytic ellipsoids use the existing exact surface-distance routine;
individual forms use VTK distance to the transformed reference triangulation.
This adds a surface score for I1--I3 without changing fits or source masks.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pyvista as pv
from skimage.measure import marching_cubes

from heart_additional_ellipsoid_methods import ellipsoid_surface_distance
from run_heart_method_comparison import get_mask
from run_reduced_heart_geometry import digest, write_json


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mesh_from_mask(mask, affine):
    vertices, faces, _, _ = marching_cubes(np.pad(mask.astype(np.float32), 1),
        level=.5, allow_degenerate=False)
    vertices = (vertices-1) @ affine[:3, :3].T + affine[:3, 3]
    return vertices, faces


def metrics(distance, weight):
    d = np.abs(np.asarray(distance))
    w = np.asarray(weight) / np.sum(weight)
    assert np.isfinite(d).all() and d.shape == w.shape
    order = np.argsort(d)
    index = min(np.searchsorted(np.cumsum(w[order]), .95), len(d)-1)
    return dict(rms_mm=float(np.sqrt(w @ (d*d))), mean_mm=float(w @ d),
        p95_mm=float(d[order[index]]), sampled_max_mm=float(d.max()))


def run(args):
    inputs = args.inputs
    base = [json.loads(x) for x in (inputs/"geometry_comparison.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = {(x["subject"], x["phase_id"]): x for x in base}
    root = Path(load(inputs/"config.local.json")["derived_root"])
    clouds = {}
    for count, name in ((8192, "library_clouds.json"), (32768, "library_clouds_32768.json")):
        clouds[count] = {c["id"]:c for c in load(inputs/name)["cases"]}
    results = {}
    sources = {str(inputs/name):digest(inputs/name) for name in
        ("geometry_comparison.jsonl", "library_clouds.json", "library_clouds_32768.json", "config.local.json")}
    for directory in args.results:
        for file in directory.rglob("*.json"):
            data = load(file)
            if not isinstance(data, dict) or not all(k in data for k in ("phase_id", "method_id", "metrics")):
                continue
            key = (data["subject"], data["phase_id"], data["method_id"])
            if key in results:
                raise ValueError("Duplicate fit " + str(key))
            row = rows[key[:2]]
            assert data["target_mask_sha256"] == row["provenance"]["sha256"]
            results[key] = data
            sources[str(file)] = digest(file)
    expected = {(s,p,m) for s,p in rows for m in args.methods.split(",")}
    if set(results) != expected:
        raise ValueError(f"Incomplete surface input: {len(expected-set(results))} missing, {len(set(results)-expected)} extra")
    args.output.mkdir(parents=True, exist_ok=False)
    # Independent geometric check for the VTK unsigned triangle-distance path.
    triangle = pv.PolyData(np.array([[0.,0,0],[2,0,0],[0,2,0]]), np.array([3,0,1,2]))
    test = pv.PolyData(np.array([[.2,.2,3.],[.2,.2,-2.]])).compute_implicit_distance(triangle)
    assert np.allclose(np.abs(test["implicit_distance"]), [3.,2.], atol=1e-12)
    reference_meshes, output = {}, []
    for key, result in sorted(results.items()):
        subject, phase, method = key
        row = rows[subject,phase]
        cid = result["case_id"]
        item = dict(subject=subject, phase_id=phase, method_id=method,
            cycle_index=row["cycle_index"], is_reference_phase=row["is_reference_phase"],
            target_mask_sha256=result["target_mask_sha256"],
            direction="target_surface_to_candidate", unit="mm")
        if result.get("metrics") is None:
            item.update(status="fit_unavailable", fit_status=result["status"])
            output.append(item)
            continue
        params = result.get("parameters")
        surface = None
        if method.startswith("I"):
            refkey = subject,row["reference_phase_id"]
            if refkey not in reference_meshes:
                reference_meshes[refkey] = mesh_from_mask(*get_mask(root, rows[refkey]))
            v, f = reference_meshes[refkey]
            transform = np.asarray(params["transform_world"])
            transformed = v @ transform[:3,:3].T + transform[:3,3]
            surface = pv.PolyData(transformed, np.column_stack((np.full(len(f),3),f)))
        scores = {}
        for count in clouds:
            cloud = clouds[count][cid]
            assert cloud["source_metadata"]["mask_sha256"] == row["provenance"]["sha256"]
            points, weight = np.asarray(cloud["points_mm"]), np.asarray(cloud["weights"])
            if method == "T0":
                distance = np.zeros(len(points))
            elif surface is not None:
                distance = pv.PolyData(points).compute_implicit_distance(surface)["implicit_distance"]
            else:
                distance = ellipsoid_surface_distance(points, params["center_mm"],
                    params["semi_axes_mm"], params["axes_matrix"])
            scores[str(count)] = metrics(distance, weight)
        item.update(status="computed", samples=scores,
            rms_32768_mm=scores["32768"]["rms_mm"],
            p95_32768_mm=scores["32768"]["p95_mm"],
            rms_sampling_difference_mm=scores["32768"]["rms_mm"]-scores["8192"]["rms_mm"])
        if method.startswith("I") and row["is_reference_phase"]:
            assert item["rms_32768_mm"] < 1e-6
        output.append(item)
        print(phase,method,round(item["rms_32768_mm"],5),flush=True)
    assert all(digest(name) == value for name,value in sources.items())
    write_json(args.output/"surface_scores.local.json",dict(
        status="same_target_quadrature_completed_not_anatomical_validation",
        implementation_sha256=digest(__file__), source_hashes=sources, rows=output,
        limits=["one-sided distances; not symmetric Hausdorff distance",
            "automatic mask triangulation includes padded FOV closure",
            "two area-sampling levels are a numerical sensitivity check, not a proof of convergence",
            "T0 and individual reference self-distances are construction checks"]))
    flat = [{k:v for k,v in x.items() if k != "samples"} for x in output]
    fields = list(dict.fromkeys(k for x in flat for k in x))
    with (args.output/"surface_scores.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(flat)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs",required=True,type=Path)
    p.add_argument("--results",required=True,type=Path,nargs="+")
    p.add_argument("--methods",required=True)
    p.add_argument("--output",required=True,type=Path)
    run(p.parse_args())
