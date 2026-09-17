"""Prepare immutable file inputs and audit MATLAB M1--M9 library results.

No MATLAB process is started by this tool. Coordinates are physical millimetres.
Upstream algorithms are not reimplemented here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

ROOT = Path(__file__).resolve().parents[1]
METHODS = {f"M{i}": name for i, name in enumerate(
    ["BOOK", "TAUB", "HES", "SOD", "FC", "2-NORM",
     "ellipsoid_fit:xyz", "ellipsoid_fit:free", "sphereFit"], 1)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes(order="C")).hexdigest()


def _json_numpy(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(f'Unsupported JSON type: {type(value).__name__}')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False, default=_json_numpy), encoding="utf-8")


def verify_vendor():
    base = ROOT / "vendor_heart_fits"
    records = json.loads((base / "provenance.json").read_text(encoding="utf-8"))
    for rec in records:
        assert sha(base / rec["package"] / "upstream.zip") == rec["archive_sha256"]
        for f in rec["files"]:
            assert sha(base / f["path"]) == f["sha256"], f["path"]
    return records


def sample_points(points, weights, policy, count):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 10 or not np.isfinite(p).all():
        raise ValueError("points_mm must be finite Nx3 with N>=10 (all nine methods)")
    if np.linalg.matrix_rank(p - p.mean(axis=0)) != 3:
        raise ValueError("Coplanar/collinear input cannot define a 3D fit")
    w = None if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    if w is not None and (len(w) != len(p) or not np.isfinite(w).all()
                          or (w < 0).any() or not (w.sum() > 0)):
        raise ValueError("weights must be finite nonnegative with positive total")
    if policy is None:
        if w is not None:
            raise ValueError("Weights supplied: explicitly choose a weighting policy")
        policy = "equal_vertices"
    if policy == "equal_vertices":
        idx = np.arange(len(p))
    elif policy == "deterministic_weight_resample":
        if w is None or count is None or count < 10:
            raise ValueError("Resampling requires weights and sample_count>=10")
        # Fixed input-order cumulative midpoint quadrature; no random generator.
        # This discretely approximates supplied weights, not continuous surfaces.
        cdf = np.cumsum(w / w.sum())
        cdf[-1] = 1.0
        idx = np.searchsorted(cdf, (np.arange(count) + 0.5) / count, side="right")
    else:
        raise ValueError(f"Unknown weighting policy: {policy}")
    used = p[idx]
    if np.linalg.matrix_rank(used-used.mean(axis=0)) != 3:
        raise ValueError("Resampled points are spatially degenerate")
    return used, {"policy": policy, "input_count": len(p), "fit_count": len(used),
                  "weights_used_by_upstream": False,
                  "weights_ignored_explicitly": w is not None and policy == "equal_vertices",
                  "source_points_sha256_le_f64": array_sha(p),
                  "weights_sha256_le_f64": None if w is None else array_sha(w),
                  "fit_points_sha256_le_f64": array_sha(used),
                  "resample_indices_zero_based": idx.tolist() if policy != "equal_vertices" else [],
                  "convergence_of_resampling": "not_checked"}


def synthetic_cases():
    n = 600
    z = 1 - 2 * (np.arange(n) + .5) / n
    phi = np.arange(n) * np.pi * (3 - np.sqrt(5))
    unit = np.column_stack([np.sqrt(1-z*z)*np.cos(phi),
                            np.sqrt(1-z*z)*np.sin(phi), z])
    # Fixed proper rotation; shape tensors avoid eigenvector sign ambiguities.
    a, b = .47, -.31
    rz = np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    ry = np.array([[np.cos(b),0,np.sin(b)],[0,1,0],[-np.sin(b),0,np.cos(b)]])
    rot = rz @ ry
    result = []
    for name, axes, centre in [("known_sphere", [45,45,45], [23,-17,61]),
                               ("known_rotated_ellipsoid", [60,50,40], [23,-17,61])]:
        axes, centre = np.array(axes), np.array(centre)
        pts = (unit * axes) @ rot.T + centre
        result.append({"id": name, "points_mm": pts.tolist(),
                       "truth": {"centre_mm": centre.tolist(), "axes_mm": axes.tolist(),
                                 "orientation_columns": rot.tolist(),
                                 "shape_tensor_mm2": (rot @ np.diag(axes**2) @ rot.T).tolist(),
                                 "exact_recovery_methods": list(METHODS) if name == "known_sphere"
                                   else ["M1","M2","M3","M4","M5","M6","M8"]}})
    return result


def prepare(args):
    records = verify_vendor()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError("Use a fresh output directory; previous evidence is immutable")
    if args.synthetic:
        cases, original = synthetic_cases(), {"kind":"synthetic_known_surfaces"}
    else:
        src = Path(args.input)
        if src.suffix.lower() == ".mat":
            raw = loadmat(src, simplify_cells=True)
        else:
            raw = json.loads(src.read_text(encoding="utf-8"))
        if raw.get("units") != "mm":
            raise ValueError("Input must explicitly declare units='mm'")
        cases = raw.get("cases", [raw])
        if isinstance(cases, dict):
            cases = [cases]
        original = {"kind":"file", "filename":src.name, "sha256":sha(src)}
    ready, ids = [], set()
    for i, case in enumerate(cases):
        cid = str(case.get("id", f"case_{i+1}"))
        if cid in ids:
            raise ValueError("Duplicate case id")
        ids.add(cid)
        p, sampling = sample_points(case["points_mm"], case.get("weights"),
                                    args.weight_policy, args.sample_count)
        ready.append({"id":cid, "points_mm":p.tolist(), "sampling":sampling,
                      "truth":case.get("truth", {}),
                      "source_metadata":case.get("source_metadata", {})})
    if not np.isfinite(args.regularization) or args.regularization < 0:
        raise ValueError("Explicit finite nonnegative regularization required")
    if not np.isfinite(args.eta) or args.eta < 1:
        raise ValueError("Explicit finite eta>=1 required")
    settings = {"regularization":args.regularization, "eta":args.eta,
                "normalize":True, "forceOrigin":False, "forceAxial":False,
                "automatic_regularization":False, "external_preconditioning":"none"}
    out.mkdir(parents=True)
    doc = {"schema":"heart_library_fits_input_v1", "units":"mm", "source":original,
           "methods":list(METHODS), "settings":settings, "cases":ready,
           "vendor_manifest_sha256":sha(ROOT / "vendor_heart_fits/provenance.json")}
    if args.shared_mat:
        arrays = {}
        for i, case in enumerate(ready):
            key = f"points_{i+1:05d}"
            arrays[key] = np.asarray(case.pop("points_mm"),dtype=float)
            case["points_mat_variable"] = key
        savemat(out / "shared_points.mat", arrays, do_compression=True)
        doc["shared_points_mat"] = "shared_points.mat"
        doc["shared_points_mat_sha256"] = sha(out / "shared_points.mat")
    write_json(out / "input.json", doc)
    write_json(out / "source_snapshot.json", records)
    # MATLAB command is printed only, never executed here.
    esc = lambda p: str(Path(p).resolve()).replace("'", "''").replace("\\", "/")
    command = f"addpath('{esc(ROOT/'src')}'); run_heart_library_fits('{esc(out/'input.json')}', '{esc(out/'results.json')}');"
    (out / "RUN_AFTER_COORDINATOR_APPROVAL.txt").write_text(
        'MATLAB NOT STARTED. Coordinate with the active MATLAB/FEM jobs first.\n'
        + 'matlab -batch "' + command + '"\n', encoding="utf-8")
    print(json.dumps({"prepared":str(out), "cases":len(ready), "fits":9*len(ready),
                      "matlab_status":"not_started"}))


def audit(args):
    inp = json.loads(Path(args.input).read_text(encoding="utf-8"))
    res = json.loads(Path(args.results).read_text(encoding="utf-8"))
    if res["input_sha256"] != sha(args.input):
        raise ValueError("Results do not belong to these exact input bytes")
    expected = {(c["id"], m) for c in inp["cases"] for m in inp["methods"]}
    actual = [(r["case_id"], r["method_id"]) for r in res["fits"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Missing, extra or duplicate fit rows")
    checks = []
    for row in res["fits"]:
        case = next(c for c in inp["cases"] if c["id"] == row["case_id"])
        truth = case["truth"]
        if row["method_id"] not in truth.get("exact_recovery_methods", []):
            continue
        passed = False
        ce = se = None
        if row["valid"]:
            ce = float(np.linalg.norm(np.array(row["centre_mm"])-truth["centre_mm"]))
            se = float(np.linalg.norm(np.array(row["shape_tensor_mm2"])-truth["shape_tensor_mm2"])
                       /np.linalg.norm(truth["shape_tensor_mm2"]))
            passed = ce < 1e-5 and se < 1e-6
        checks.append({"case_id":row["case_id"],"method_id":row["method_id"],
                       "centre_error_mm":ce,"shape_tensor_relative_error":se,"passed":passed})
    report = {"schema":"heart_library_fits_audit_v1", "results_sha256":sha(args.results),
              "exact_recovery_checks":checks, "all_exact_recovery_passed":
                bool(checks) and all(x["passed"] for x in checks),
              "physiological_validation":False}
    dest = Path(args.output)
    if dest.exists(): raise FileExistsError(dest)
    write_json(dest, report)
    print(json.dumps(report))


def self_test():
    # No upstream fit is executed: these are preparation and result-audit tests.
    cases = synthetic_cases()
    for c in cases:
        p = np.array(c["points_mm"]); t = c["truth"]
        q = np.linalg.inv(t["shape_tensor_mm2"])
        assert np.max(np.abs(np.einsum('ni,ij,nj->n',p-t['centre_mm'],q,p-t['centre_mm'])-1)) < 1e-12
    p = np.array(cases[0]["points_mm"])
    x, meta = sample_points(p, None, None, None)
    assert np.array_equal(x,p) and meta['policy']=='equal_vertices'
    w = np.arange(1,len(p)+1,dtype=float)
    a, ma = sample_points(p,w,'deterministic_weight_resample',2400)
    b, mb = sample_points(p,w*7,'deterministic_weight_resample',2400)
    assert np.array_equal(a,b) and ma['fit_count']==2400
    for pp,ww,policy,n in [(p,w,None,None),(p,w[:-1],'equal_vertices',None),
                            (p*np.array([1,1,0]),None,None,None),
                            (p,None,'deterministic_weight_resample',2400)]:
        try: sample_points(pp,ww,policy,n)
        except ValueError: pass
        else: raise AssertionError('Invalid input accepted')
    records = verify_vendor()
    return {'status':'passed','scope':'Python preparation, synthetic truth, source integrity',
            'vendor_packages':len(records),'matlab_library_execution':'not_started'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command',required=True)
    prep = sub.add_parser('prepare')
    src = prep.add_mutually_exclusive_group(required=True)
    src.add_argument('--synthetic',action='store_true'); src.add_argument('--input')
    prep.add_argument('--output',required=True)
    prep.add_argument('--weight-policy',choices=['equal_vertices','deterministic_weight_resample'])
    prep.add_argument('--sample-count',type=int)
    prep.add_argument('--shared-mat',action='store_true',help='Store each cloud once in shared_points.mat')
    prep.add_argument('--regularization',type=float,default=0.0)
    prep.add_argument('--eta',type=float,default=1.0)
    ap = sub.add_parser('audit'); ap.add_argument('--input',required=True)
    ap.add_argument('--results',required=True); ap.add_argument('--output',required=True)
    sub.add_parser('self-test')
    args = p.parse_args()
    if args.command=='prepare': prepare(args)
    elif args.command=='audit': audit(args)
    else: print(json.dumps(self_test()))


if __name__ == '__main__':
    main()
