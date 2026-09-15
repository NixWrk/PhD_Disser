"""Read-only audit of 33.06 inputs and numerical calculations.

Original notebooks, source modules, annotations and external artifacts are never
written. Only this audit directory receives summary JSON. The config is supplied
explicitly; no machine-specific data path is embedded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from exploratory_analysis import (  # noqa: E402
    _static_prediction, dynamic_two_layer_inverse, fit_static_at_h,
    interval_median, normalized_modes, profile_static_h,
)
from two_layer_model import evaluate, geometry_from_size  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save(name, result):
    (HERE / name).write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                      allow_nan=False) + "\n", encoding="utf-8")
    print(name, json.dumps(result, ensure_ascii=False), flush=True)


def baseline(cfg, config_path, art):
    derived = Path(cfg["derived_root"]) / "exp02"
    provenance = []
    by_subject = {}
    for source in art["input_provenance"]:
        rid = source["record_id"]
        bp = derived / "annotations" / "breathing" / f"{rid}.json"
        ep = derived / "annotations" / "ecg" / f"{rid}.json"
        breathing, ecg = read(bp), read(ep)
        path = Path(cfg["data_root"]) / breathing["input"]["relative_path"]
        frame = pd.read_csv(path)
        modes, _ = normalized_modes(breathing)
        t = frame["TIME_s"].to_numpy(float)
        base = frame["BASE_2_Ω"].to_numpy(float)
        subject = art["subjects"][source["subject_id"]]
        i = subject["sizes_mm"].index(source["size_mm"])
        rows = []
        for mode, state in [("задержка_вдох", "inhale"), ("задержка_выдох", "exhale")]:
            level = interval_median(t, base, modes[mode])
            alternatives = {}
            for margin in (0.0, 1.0, 2.0):
                try:
                    alternatives[str(margin)] = interval_median(t, base, modes[mode], margin_s=margin) - level
                except ValueError:
                    alternatives[str(margin)] = None
            left, right = modes[mode]
            mask = (t >= left + 0.5) & (t <= right - 0.5)
            slope = np.polyfit(t[mask] - t[mask][0], base[mask], 1)[0]
            rows.append({"state": state, "median_ohm": level,
                         "saved_delta_ohm": level - subject[f"observed_z_{state}_ohm"][i],
                         "duration_s": right - left, "slope_ohm_s": float(slope),
                         "margin_delta_ohm": alternatives})
        provenance.append({"record_id": rid, "subject_id": source["subject_id"],
                           "size_mm": source["size_mm"],
                           "csv_hash_matches": sha(path) == source["source_csv_sha256"],
                           "breathing_hash_matches": sha(bp) == source["breathing_sidecar_sha256"],
                           "ecg_hash_matches": sha(ep) == source["ecg_sidecar_sha256"],
                           "time_increasing": bool(np.all(np.diff(t) > 0)), "levels": rows})
    for sid, sub in art["subjects"].items():
        prof = sub["static_h_profile"]["profile"]
        best = sub["static_h_profile"]["best"]
        obs = np.column_stack([sub["observed_z_inhale_ohm"], sub["observed_z_exhale_ohm"]])
        pred = np.column_stack([best["predicted_inhale_ohm"], best["predicted_exhale_ohm"]])
        res = pred - obs
        common = res.mean(axis=1)
        energy_fraction = 2 * np.sum(common**2) / np.sum(res**2)
        profile_rms = np.array([row["residual_rms_ohm"] for row in prof])
        dynamic = {}
        for mode, data in sub["dynamic"].items():
            wave = np.array([e["mean_ohm"] for e in data["ensembles"]])
            values = []
            for h_mm in [1, 5, 15, 20, 40]:
                row = min(prof, key=lambda r: abs(1000 * r["h_m"] - h_mm))
                inv = dynamic_two_layer_inverse(np.array(sub["sizes_mm"]) / 1000, wave, row, data["state"])
                values.append({"h_mm": h_mm, "static_rms_ohm": row["residual_rms_ohm"],
                               "delta_rho1_ptp": float(np.ptp(inv["delta_rho1_ohm_m"])),
                               "delta_rho2_ptp": float(np.ptp(inv["delta_rho2_ohm_m"])),
                               "fractional_rho1_abs_max": float(np.max(abs(inv["fractional_rho1"]))),
                               "fractional_rho2_abs_max": float(np.max(abs(inv["fractional_rho2"]))),
                               "dynamic_rms_ohm": inv["residual_rms_ohm"],
                               "condition": inv["operator_condition"]})
            dynamic[mode] = values
        by_subject[sid] = {"paired_residual_correlation": float(np.corrcoef(res.T)[0, 1]),
                          "common_record_residual_fraction_of_rss": float(energy_fraction),
                          "residuals_by_size_ohm": [{"size_mm": s, "inhale": float(r[0]), "exhale": float(r[1])}
                                                   for s, r in zip(sub["sizes_mm"], res)],
                          "profile_local_minima_h_mm": [float(1000 * prof[i]["h_m"])
                                                       for i in range(1, len(prof)-1)
                                                       if profile_rms[i] < min(profile_rms[i-1], profile_rms[i+1])],
                          "failed_optimizer_rows": sum(not r["optimizer_success"] for r in prof),
                          "dynamic_along_profile": dynamic}
    save("baseline.json", {"configuration_hash_matches": sha(config_path) == art["configuration_sha256"],
                           "source_code_hashes": {p: sha(ROOT/p) for p in ["two_layer_model.py", "exploratory_analysis.py", "ttrkg_analysis.py"]},
                           "provenance": provenance, "subjects": by_subject})


def fits(art):
    result = {"reproduction": {}, "synthetic": []}
    for sid, sub in art["subjects"].items():
        sizes = np.array(sub["sizes_mm"]) / 1000
        row = fit_static_at_h(sizes, sub["observed_z_inhale_ohm"], sub["observed_z_exhale_ohm"], .001)
        result["reproduction"][sid] = row
        print(sid, "RMS", row["residual_rms_ohm"], flush=True)
    sizes = np.arange(50, 141, 10) / 1000
    for h, r1, rin, rex in [(.020, 5., 22., 15.), (.015, 15., 4., 7.)]:
        zin = np.array([evaluate(r1, rin, h, *geometry_from_size(s)).z for s in sizes])
        zex = np.array([evaluate(r1, rex, h, *geometry_from_size(s)).z for s in sizes])
        try:
            prof = profile_static_h(sizes, zin, zex, [.001, .005, .01, .015, .02, .03, .04, .08])
            result["synthetic"].append({"truth": [h, r1, rin, rex], "best": prof["best"]})
        except Exception as exc:
            result["synthetic"].append({"truth": [h, r1, rin, rex], "error": repr(exc)})
    save("fits.json", result)


def leave_one_out(art):
    """Independent diagnostic using analytic derivatives, same model and bounds.

    A coarse h grid is explicit. It is a sensitivity check, not a new h estimate.
    Every grid point receives multiple starts; outputs keep fit status.
    """
    all_results = {}
    hs = np.array([1, 2, 3, 5, 8, 10, 12, 15, 18, 20, 25, 30, 40, 60, 80]) / 1000
    for sid, sub in art["subjects"].items():
        sizes = np.array(sub["sizes_mm"]) / 1000
        obs = np.column_stack([sub["observed_z_inhale_ohm"], sub["observed_z_exhale_ohm"]])
        profiles = sub["static_h_profile"]["profile"]
        rows = []
        for omit in range(-1, len(sizes)):
            keep = np.ones(len(sizes), dtype=bool)
            if omit >= 0:
                keep[omit] = False
            y = obs[keep].ravel()
            candidates_h = []
            for h in hs:
                free = min(profiles, key=lambda r: abs(r["h_m"] - h))
                seeds = [[free["rho1_ohm_m"], free["rho2_inhale_ohm_m"], free["rho2_exhale_ohm_m"]],
                         [1, 20, 15], [10, 5, 4], [50, 100, 60]]
                def fun(x):
                    return _static_prediction(x, sizes[keep], h)[0] - y
                def jac(x):
                    return _static_prediction(x, sizes[keep], h)[1]
                candidates = []
                for seed in seeds:
                    try:
                        fit = least_squares(fun, np.log(seed), jac=jac,
                                            bounds=(np.log(.01), np.log(1000)),
                                            ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=600)
                        if fit.success:
                            candidates.append(fit)
                    except RuntimeError:
                        pass
                if not candidates:
                    raise RuntimeError(f"No converged candidate for {sid}, omit={omit}, h={h}")
                fit = min(candidates, key=lambda f: np.dot(f.fun, f.fun))
                candidates_h.append({"h_mm": float(h*1000), "rms_ohm": float(np.sqrt(np.mean(fit.fun**2))),
                                     "rho": np.exp(fit.x).tolist()})
            best = min(candidates_h, key=lambda r: r["rms_ohm"])
            row = {"omitted_size_mm": None if omit < 0 else int(round(sizes[omit]*1000)),
                   "best_on_coarse_grid": best, "profile": candidates_h}
            rows.append(row)
            print(sid, "omit", row["omitted_size_mm"], "best", best, flush=True)
        all_results[sid] = rows
    save("leave_one_out.json", all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=["baseline", "fits", "loo"], required=True)
    args = parser.parse_args()
    cfg = read(args.config)
    artifact = read(Path(cfg["derived_root"]) / "exp02/exploratory/33.06_side_arrays_exploratory.json")
    if args.phase == "baseline":
        baseline(cfg, args.config, artifact)
    elif args.phase == "fits":
        fits(artifact)
    else:
        leave_one_out(artifact)
