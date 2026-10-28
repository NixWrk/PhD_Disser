"""Independent, local linear EIT pulse audit; no FEM, AI, or clinical inference.

Run with --data-root pointing to pulmovista_check_20261028/local and --output
pointing to its pulse_identifiability subdirectory. Only the output directory
is written. Source MAT/JSON files and external pipeline code remain unchanged.

Reusable API: general_design(A, observations, weights). Rows are measurements,
columns are physical parameters, observations have shape (measurements, time).
weights are positive ROW MULTIPLIERS in ||diag(weights)(y-Ax)||_2, not variances.
Coefficients retain the physical units implicit in A. No noise law is assumed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, detrend, sosfiltfilt


FREE = np.array([0, 1, 3])
NAMES = ["soft_background", "lung", "bone", "heart", "fat"]


def rms(a, axis=None):
    return np.sqrt(np.mean(np.asarray(a) ** 2, axis=axis))


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return jsonable(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def write_json(path, data):
    Path(path).write_text(json.dumps(jsonable(data), ensure_ascii=False,
                                    indent=2, allow_nan=False) + "\n", encoding="utf-8")


def general_design(A, observations, weights):
    """Column-equilibrated WLS and FWL diagnostics for any physical design.

    A: m x p in measurement-units / parameter-units; y: m x n (or m).
    The SVD is computed after unit-L2 column normalization. Rank-deficient
    designs return the minimum-norm solution IN NORMALIZED coordinates and
    rank_identified=False; they must not be reported as identified parameters.
    FWL residualizes both the target column and observations on all other
    columns. For a full-rank design it equals the simultaneous coefficient.
    This is linear algebra, NOT a nonlinear solver or a noise confidence model.
    """
    A = np.asarray(A, dtype=float)
    y = np.asarray(observations, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    w = np.asarray(weights, dtype=float).reshape(-1)
    if A.ndim != 2 or y.ndim != 2 or A.shape[0] != y.shape[0] or w.size != A.shape[0]:
        raise ValueError("Expected A[m,p], observations[m,n], weights[m]")
    if not (np.isfinite(A).all() and np.isfinite(y).all() and np.isfinite(w).all()) or np.any(w <= 0):
        raise ValueError("Finite inputs and positive row multipliers required")
    Aw, yw = w[:, None] * A, w[:, None] * y
    scales = np.linalg.norm(Aw, axis=0)
    if np.any(scales == 0):
        raise ValueError("Zero sensitivity column; parameter cannot be identified")
    C = Aw / scales
    U, singular, Vh = np.linalg.svd(C, full_matrices=False)
    tol = max(C.shape) * np.finfo(float).eps * singular[0]
    rank = int(np.count_nonzero(singular > tol))
    Cpinv = np.linalg.pinv(C, rcond=tol / singular[0])
    operator = (Cpinv / scales[:, None]) * w[None, :]
    coefficients = operator @ y
    prediction = A @ coefficients
    residual = y - prediction
    fractions, partial, naive = [], [], []
    for k in range(A.shape[1]):
        others = np.delete(C, k, axis=1)
        hp = C[:, k] - others @ (np.linalg.pinv(others) @ C[:, k])
        yp = yw - others @ (np.linalg.pinv(others) @ yw)
        fraction = float(np.linalg.norm(hp))
        fractions.append(fraction)
        partial.append((hp @ yp) / (hp @ hp) / scales[k]
                       if fraction > tol else np.full(y.shape[1], np.nan))
        naive.append((C[:, k] @ yw) / scales[k])
    fractions = np.asarray(fractions)
    return dict(coefficients=coefficients, prediction=prediction, residual=residual,
                operator=operator, weighted_residual=w[:, None] * residual,
                weighted_design=Aw, normalized_design=C, column_norms=scales,
                normalized_singular_values=singular,
                raw_weighted_singular_values=np.linalg.svd(Aw, compute_uv=False),
                normalized_condition=float(singular[0] / singular[-1]),
                normalized_weakest_right_vector=Vh[-1], column_cosines=C.T @ C,
                centered_column_correlations=np.corrcoef(C.T), rank=rank,
                rank_identified=rank == A.shape[1], partial_unique_fractions=fractions,
                variance_inflation_geometry_only=1 / fractions ** 2,
                partial_coefficients=np.asarray(partial), naive_coefficients=np.asarray(naive),
                relative_residual_pct=100 * np.linalg.norm(w[:, None] * residual) /
                                      np.linalg.norm(yw) if np.linalg.norm(yw) else None)


def design_summary(r):
    keys = ("column_norms", "normalized_singular_values", "raw_weighted_singular_values",
            "normalized_condition", "normalized_weakest_right_vector", "column_cosines",
            "centered_column_correlations", "rank", "rank_identified",
            "partial_unique_fractions", "variance_inflation_geometry_only")
    return {k: r[k] for k in keys}


def load_volume_design(manifest_path):
    """Read saved STL perturbations; never generate a mesh or run FEM.

    JSON contract is also written as geometry_manifest.example.json by main.
    Two columns of S_eit remain soft/lung. First column is dV_eit/dVheart_ml;
    the output parameter order is [delta_volume_ml, delta_sigma_soft,
    delta_sigma_lung]. Actual saved volumes determine the denominator.
    Both central and one-sided derivatives and 2% versus 5% disagreement are
    returned. A derivative is only an effective one-parameter shape law.
    """
    manifest_path = Path(manifest_path)
    m = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if m["material_order"] != NAMES:
        raise ValueError("Material order mismatch")
    required_units = {"V_eit": "V", "S_eit": "V/(S/m)", "Ktt": "ohm/(S/m)",
                      "Ztt": "ohm", "volume_ml": "ml", "sigma_S_m": "S/m"}
    if m["units"] != required_units:
        raise ValueError("Explicit SI-voltage and ml-volume units required")
    variants, provenance = {}, {}
    conditions, source_identity = None, None
    config = plan = None
    if "electrical_config" in m:
        config_path = (manifest_path.parent / m["electrical_config"]).resolve()
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        plan_path = Path(config["material_plan"])
        plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
        # Hashes bind conditions to saved source fmdl; no geometry is recomputed.
        expected_identity = dict(config_sha256=sha(config_path), plan_sha256=sha(plan_path),
            eit_model_sha256=sha(config["eit_model_file"]),
            ttrkg_model_sha256=sha(config["ttrkg_model_file"]))
        conditions = {k: config[k] for k in ("frequency_Hz", "current_A", "montage_basis", "breath_interval_status")}
        conditions["source_models_unchanged_hash_checked"] = True
        conditions["source_STL_sha256_from_verified_plan"] = plan["stl_sha256"]
        conditions["source_mesh_sha256_from_verified_plan"] = plan["mesh_sha256"]
        provenance["electrical_config"] = {"path": str(config_path), "sha256": sha(config_path)}
        provenance["material_plan"] = {"path": str(plan_path), "sha256": sha(plan_path)}
    for name, rel in m["variants"].items():
        path = (manifest_path.parent / rel).resolve()
        d = loadmat(path, simplify_cells=True)
        v, S = np.asarray(d["V_eit"]).ravel(), np.asarray(d["S_eit"])
        k = np.asarray(d["Ktt"]).ravel()
        sig = np.asarray(d["sigma"] if "sigma" in d else d["sigma_S_m"]).ravel()
        row, identity = d.get("row"), d.get("identity")
        if row is not None:
            if config is None or not isinstance(row, dict) or not isinstance(identity, dict):
                raise ValueError("Real row/identity schema requires electrical_config in manifest")
            volumes = np.asarray(row["volume_ml"]).ravel()
            if volumes.shape != (5,):
                raise ValueError("row.volume_ml must contain five tissue volumes")
            volume = float(volumes[3])
            target = float(row["target_volume_ml"])
            if source_identity is None:
                source_identity = identity
            if identity != source_identity:
                raise ValueError("Different solver/config/source EIT/TT identities across variants")
            for field, value in expected_identity.items():
                if identity.get(field) != value:
                    raise ValueError(f"Saved identity does not match actual input: {field}")
            if not np.allclose(np.asarray(row["rho_ohm_m"]).ravel(), 1/sig, rtol=1e-12, atol=0):
                raise ValueError("row.rho_ohm_m and sigma disagree")
            jobs = [j for j in config["jobs"] if j["id"] == row["id"]]
            materials = [p for p in plan["variants"] if p["id"] == row["variant"]]
            if len(jobs) != 1 or len(materials) != 1:
                raise ValueError("Saved variant absent/ambiguous in source config/material plan")
            job, material = jobs[0], materials[0]
            if not job["eit"] or not row["eit_computed"] or job["variant"] != row["variant"] or job["scenario"] != row["scenario"]:
                raise ValueError("EIT/scenario/variant mismatch")
            if not np.allclose(job["rho_ohm_m"], 1/sig, rtol=1e-12, atol=0):
                raise ValueError("Source config conductivity mismatch")
            if row["quadrature"] != material["quadrature"] or row["material_sha256"] != material["material_sha256"]:
                raise ValueError("Material provenance mismatch")
            if not np.allclose(volumes, material["volume_ml"], rtol=1e-12, atol=1e-10) or not np.isclose(target, material["target_volume_ml"], rtol=1e-12):
                raise ValueError("Material volume mismatch")
            if not np.allclose(k, row["Ktt_ohm_per_S_m"], rtol=1e-12, atol=1e-12) or not np.isclose(d["Ztt"], row["Ztt_ohm"], rtol=1e-12):
                raise ValueError("Saved TTRKG arrays and row disagree")
            extra = dict(target_volume_ml=target, tissue_volume_ml=volumes,
                         quadrature=int(row["quadrature"]), row=row, identity=identity)
        else:
            volume = float(np.asarray(d["volume_ml"]).item())
            extra = {}
        if v.shape != (208,) or S.shape != (208, 5) or k.shape != (5,) or sig.shape != (5,):
            raise ValueError(f"Invalid dimensions: {name}")
        values = np.r_[v, S.ravel(), k, sig, np.asarray(d["Ztt"]).ravel(), volume]
        if not np.isfinite(values).all() or np.any(sig <= 0) or volume <= 0 or np.any(v == 0):
            raise ValueError(f"Nonfinite or nonpositive conductivity: {name}")
        variants[name] = dict(V=v, S=S, K=k, sigma=sig,
                              Z=float(np.asarray(d["Ztt"]).item()), volume=volume, **extra)
        provenance[name] = {"path": str(path), "sha256": sha(path)}
    base = variants[m["reference"]]
    if not np.isclose(base["sigma"][3], m["fixed_heart_sigma_S_m"], rtol=1e-10):
        raise ValueError("Heart conductivity conflicts with manifest")
    for d in variants.values():
        if not np.allclose(d["sigma"], base["sigma"], rtol=1e-12, atol=0):
            raise ValueError("Geometry derivative requires IDENTICAL conductivities")
        if d.get("quadrature") != base.get("quadrature"):
            raise ValueError("Do not mix quadrature orders within one design")
    pairs = {}
    for label, pair in m["central_pairs"].items():
        minus, plus = variants[pair[0]], variants[pair[1]]
        if not minus["volume"] < base["volume"] < plus["volume"]:
            raise ValueError("Actual volumes do not bracket reference")
        dv = plus["volume"] - minus["volume"]
        jv = (plus["V"] - minus["V"]) / dv
        jtt = (plus["Z"] - minus["Z"]) / dv
        left = (base["V"] - minus["V"]) / (base["volume"] - minus["volume"])
        right = (plus["V"] - base["V"]) / (plus["volume"] - base["volume"])
        jleft = (base["Z"] - minus["Z"]) / (base["volume"] - minus["volume"])
        jright = (plus["Z"] - base["Z"]) / (plus["volume"] - base["volume"])
        pairs[label] = dict(A=np.column_stack((jv, base["S"][:, 0], base["S"][:, 1])),
                            Ktt=np.array([jtt, base["K"][0], base["K"][1]]),
                            actual_volume_span_ml=dv, JV_eit_V_per_ml=jv,
                            left_JV_eit_V_per_ml=left, right_JV_eit_V_per_ml=right,
                            left_Jtt_ohm_per_ml=jleft, right_Jtt_ohm_per_ml=jright,
                            delta_volume_range_ml=[minus["volume"]-base["volume"], plus["volume"]-base["volume"]],
                            second_derivative_V_per_ml2=2*(right-left)/dv,
                            weighted_asymmetry_pct=100*np.linalg.norm((left-right)/np.abs(base["V"]))/np.linalg.norm(jv/np.abs(base["V"])),
                            asymmetry_pct=100 * np.linalg.norm(left-right) / np.linalg.norm(jv))
    labels = list(pairs)
    comparisons = []
    for i, label in enumerate(labels):
        for other in labels[i+1:]:
            a, b = pairs[label], pairs[other]
            w = 1/np.abs(base["V"])
            comparisons.append(dict(first=label, second=other,
                JV_relative_difference_pct=100*np.linalg.norm(w*(b["JV_eit_V_per_ml"]-a["JV_eit_V_per_ml"]))/np.linalg.norm(w*a["JV_eit_V_per_ml"]),
                Jtt_difference_ohm_per_ml=float(b["Ktt"][0]-a["Ktt"][0])))
    return dict(reference=base, pairs=pairs, step_comparisons=comparisons, manifest=m,
                manifest_sha256=sha(manifest_path), source_hashes=provenance,
                variants=variants, conditions=conditions, source_identity=source_identity,
                parameter_units=["ml", "S/m", "S/m"],
                parameter_order=["heart_volume", "soft_background", "lung"])


def reciprocal_pairs(S, Vpred):
    """Pair duplicate physical sensitivity rows; check unique reciprocal partners.

    This does not infer electrode locations. Keeping paired rows together in
    channel CV prevents the identical ideal forward equation leaking across it.
    """
    relative = np.linalg.norm(S[:, None, :] - S[None, :, :], axis=2)
    relative /= np.maximum(np.linalg.norm(S, axis=1)[:, None], np.finfo(float).tiny)
    np.fill_diagonal(relative, np.inf)
    partner = np.argmin(relative, axis=1)
    if np.any(relative[np.arange(len(S)), partner] > 1e-8) or not np.array_equal(partner[partner], np.arange(len(S))):
        raise ValueError("Unique reciprocal row pairing failed; provide explicit pairs")
    pairs = [(i, int(j)) for i, j in enumerate(partner) if i < j]
    if len(pairs) * 2 != S.shape[0]:
        raise ValueError("Incomplete reciprocal pairing")
    if not all(np.isclose(Vpred[i], Vpred[j], rtol=1e-8, atol=1e-12) for i, j in pairs):
        raise ValueError("Prediction does not satisfy inferred reciprocal pairing")
    return np.asarray(pairs)


def fitted_metrics(result, sigma, K, pairs, band):
    coef = result["coefficients"]
    contributions = 1000 * K[:, None] * coef
    naive, partial = result["naive_coefficients"][2], result["partial_coefficients"][2]
    # Full FWL: residualizing the column as well as the observation is essential.
    heart = coef[2]
    e = result["weighted_residual"]
    yw = result["prediction"] + result["residual"]
    antisym = np.zeros_like(band)
    for i, j in pairs:
        antisym[i] = (band[i] - band[j]) / 2
        antisym[j] = -antisym[i]
    sv = np.linalg.svd(e, compute_uv=False)
    off = e @ e.T / e.shape[1]
    diagonal = np.diag(off)
    residual_corr = off / np.sqrt(np.outer(diagonal, diagonal))
    lag = np.sum(e[:, :-1] * e[:, 1:]) / np.sqrt(np.sum(e[:, :-1]**2) * np.sum(e[:, 1:]**2))
    return dict(relative_residual_pct=result["relative_residual_pct"],
                delta_sigma_rms_S_m=rms(coef, axis=1),
                relative_delta_sigma_rms=rms(coef, axis=1) / sigma,
                relative_delta_sigma_max=np.max(np.abs(coef), axis=1) / sigma,
                coefficient_correlations=np.corrcoef(coef),
                projected_TTRKG_rms_mohm=rms(contributions, axis=1),
                projected_TTRKG_total_rms_mohm=rms(np.sum(contributions, axis=0)),
                projected_soft_lung_rms_mohm=rms(contributions[:2].sum(axis=0)),
                heart_naive_rms_S_m=rms(naive), heart_partial_rms_S_m=rms(partial),
                heart_naive_to_partial_correlation=np.corrcoef(naive, partial)[0, 1],
                heart_naive_minus_partial_rms_over_partial_pct=100*rms(naive-partial)/rms(partial),
                fwl_vs_joint_max_abs_S_m=np.max(np.abs(partial-heart)),
                wrong_residual_only_heart_attenuation=result["partial_unique_fractions"][2]**2,
                residual_channel_rms_quantiles=np.quantile(rms(e, axis=1), [0, .1, .5, .9, 1]),
                residual_first_singular_energy_fraction=sv[0]**2 / np.sum(sv**2),
                residual_lag1_pooled=lag,
                residual_abs_crosschannel_corr_quantiles=np.quantile(np.abs(residual_corr[np.triu_indices(len(e), 1)]), [.1, .5, .9]),
                reciprocal_antisymmetric_voltage_energy_pct=100*np.sum(antisym**2)/np.sum(band**2),
                unweighted_voltage_residual_pct=100*np.linalg.norm(result["residual"])/np.linalg.norm(yw))


def discrepancy_weights(A, y, w, lower=.1, upper=.9):
    """Empirical per-channel MODEL DISCREPANCY; not pure electronic noise.

    Frozen after estimation on a training TIME BLOCK. Quantile clipping avoids
    infinitely trusting a channel with small fitted residual. No common noise
    level or Gaussian distribution is introduced.
    """
    e = general_design(A, y, w)["weighted_residual"]
    s = rms(e, axis=1)
    limits = np.quantile(s, [lower, upper])
    if limits[0] <= 0:
        raise ValueError("Zero empirical scale: cannot infer discrepancy weights")
    return w / np.clip(s, *limits), s, limits


def moving_bootstrap(y, A, w, K, rng, repetitions):
    """Resample observed multichannel blocks; uncertainty in conditional RMS.

    The same frame indices are used for all channels, preserving instantaneous
    covariance. These are descriptive 5–95% sampling intervals conditional on
    fixed J, filter and plateau. They are NOT physiological accuracy intervals.
    """
    x = general_design(A, y, w)["coefficients"]
    out = {}
    for length in (5, 10, 20):
        n = y.shape[1]
        if n < 2*length:
            continue
        values, coefficient_values = [], []
        for _ in range(repetitions):
            starts = rng.integers(0, n-length+1, size=int(np.ceil(n/length)))
            idx = (starts[:, None] + np.arange(length)[None, :]).ravel()[:n]
            values.append(rms(1000*K[:, None]*x[:, idx], axis=1))
            coefficient_values.append(rms(x[:, idx], axis=1))
        out[str(length)] = dict(block_seconds=length/20, repetitions=repetitions,
                                projected_rms_mohm_q05_q50_q95=np.quantile(values, [.05, .5, .95], axis=0),
                                coefficient_rms_q05_q50_q95=np.quantile(coefficient_values, [.05, .5, .95], axis=0))
    return out


def cross_validate(blocks, A, w, K, pairs, nuisance_indices=(0, 1)):
    """Temporal calibration/test splits AND held-out reciprocal channel pairs.

    No waveform is forecast across sessions: test coefficients are estimated
    from retained channels at each test frame; omitted test channels provide
    the check. Temporal training determines only discrepancy weights. The
    physical J predates this audit, so this is conditional reconstruction CV.
    """
    rows, traces = [], {}
    n = A.shape[0]
    for train_key, train in blocks.items():
        wt, scales, limits = discrepancy_weights(A, train["band"], w)
        for test_key, test in blocks.items():
            if train_key == test_key:
                continue
            y = test["band"]
            for method, weights, columns in (("relative_3", w, [0, 1, 2]),
                                             ("discrepancy_3", wt, [0, 1, 2]),
                                             ("relative_soft_lung", w, list(nuisance_indices))):
                design = A[:, columns]
                held_prediction = np.empty_like(y)
                deletions = []
                losses = []
                for fold in range(8):
                    held = pairs[np.arange(len(pairs)) % 8 == fold].ravel()
                    keep = np.setdiff1d(np.arange(n), held)
                    r = general_design(design[keep], y[keep], weights[keep])
                    predicted = design[held] @ r["coefficients"]
                    held_prediction[held] = predicted
                    losses.append(100*np.linalg.norm(w[held, None]*(predicted-y[held])) / np.linalg.norm(w[held, None]*y[held]))
                    deletions.append(r["coefficients"])
                full = general_design(design, y, weights)
                c = full["coefficients"]
                rmstt = rms(1000*K[columns, None]*c, axis=1)
                delete_rms = np.array([rms(1000*K[columns, None]*d, axis=1) for d in deletions])
                disagreement = np.array([100*rms(d-c, axis=1)/rms(c, axis=1) for d in deletions])
                rows.append(dict(train=train_key, test=test_key, method=method,
                                 parameter_indices=columns,
                                 heldout_residual_pct=100*np.linalg.norm(w[:, None]*(y-held_prediction))/np.linalg.norm(w[:, None]*y),
                                 channel_fold_residual_pct=losses,
                                 all_channel_residual_pct_base_metric=100*np.linalg.norm(w[:, None]*full["residual"])/np.linalg.norm(w[:, None]*y),
                                 projected_rms_mohm=rmstt,
                                 coefficient_rms=rms(c, axis=1),
                                 deletion_rms_mohm_min=delete_rms.min(axis=0),
                                 deletion_rms_mohm_max=delete_rms.max(axis=0),
                                 deletion_waveform_change_pct_median=np.median(disagreement, axis=0),
                                 deletion_waveform_change_pct_max=disagreement.max(axis=0),
                                 trained_discrepancy_scale_clip=limits if method=="discrepancy_3" else None))
                traces[f"{train_key}__{test_key}__{method}"] = dict(coefficients=c, heldout_prediction=held_prediction,
                                                                heldout_error=y-held_prediction)
    return rows, traces


def write_csv(path, rows):
    fields = list(rows[0])
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(jsonable(v), ensure_ascii=False) if isinstance(v, (dict, list, np.ndarray)) else v
                             for k, v in row.items()})


def geometry_pass(data_root, output, manifest_path, repetitions=400, seed=20261028, compare_to=None):
    """New saved geometry J + unchanged, hashed real pulse windows; no FEM.

    Temporal/channel CV is conditional on the fixed geometry law and measured
    baseline. No ECG timing, SV or EF is inferred from volume extrema.
    """
    root, out = Path(data_root).resolve(), Path(output).resolve()
    geometry = load_volume_design(manifest_path)
    if geometry["conditions"] is None:
        raise ValueError("Geometry pass requires verified row/identity physical conditions")
    prior_path = root/"pulse_identifiability/audit.local.json"
    arrays_path = root/"pulse_identifiability/audit_arrays.npz"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    prior_completion = json.loads((prior_path.parent/"completion.local.json").read_text(encoding="utf-8"))
    for p in (prior_path, arrays_path):
        if prior_completion["artifacts"][p.name] != sha(p):
            raise ValueError("Prior pulse-window artifact changed")
    reader_hashes = {}
    for record in (2, 4):
        path = root/f"nix_july_{record:02d}_reader.mat"
        expected = next(v for k, v in prior["source_hashes"].items() if Path(k).name == path.name)
        if sha(path) != expected:
            raise ValueError("Real EIT source has changed")
        reader_hashes[str(path)] = expected
    if geometry["conditions"]["frequency_Hz"] != prior["frequency_Hz"]:
        raise ValueError("Frequency changed relative to measured EIT audit")
    source_patient = json.loads((root/"patient_v3/patient_check.local.json").read_text(encoding="utf-8"))
    if geometry["conditions"]["current_A"] != source_patient["current_A"]:
        raise ValueError("Current differs from measured EIT audit")
    with np.load(arrays_path) as old:
        plateaus = {key: {"band": old[key+"_band_V"].copy(), "time": old[key+"_time_s"].copy()}
                    for key in prior["full_plateaus"]}
        blocks = {key: {"band": old[key+"_band_V"].copy(), "time": old[key+"_time_s"].copy()}
                  for key in prior["blocks"]}
    base = geometry["reference"]
    w = 1/np.abs(base["V"])
    paired = reciprocal_pairs(base["S"], base["V"])
    out.mkdir(parents=True, exist_ok=True)
    report = dict(status="computed_conditional_STL_volume_soft_lung_linear_pass",
        parameter_order=geometry["parameter_order"], parameter_units=geometry["parameter_units"],
        geometry=geometry, clinical_validation=False, SV_EF_computed=False, FEM_run=False,
        AI_training=False, same_session_EIT_TTRKG=False,
        waveform_origin="0.7–3 Hz diagnostic band; no ECG-derived cardiac-phase labels",
        old_audit_source=str(prior_path), old_audit_sha256=sha(prior_path),
        observations_sha256=sha(arrays_path), reader_hashes=reader_hashes,
        base_weights="1/abs(new saved baseline EIT voltage); not a noise variance estimate",
        filter=prior["filter"], band_Hz=prior["band_Hz"], blocks=prior["blocks"],
        derivative_sets={}, seed=seed, repetitions=repetitions,
        old_delta_sigma_heart_replaced=True, heart_sigma_S_m=float(base["sigma"][3]),
        implementation_sha256=sha(__file__))
    output_arrays, cv_rows, summary_rows = {}, [], []
    rng = np.random.default_rng(seed)
    for label, pair in geometry["pairs"].items():
        A, K = pair["A"], pair["Ktt"]
        result = dict(design=design_summary(general_design(A, base["V"], w)),
                      volume_column_V_per_ml=pair["JV_eit_V_per_ml"],
                      transfer_coefficients=[float(k) for k in K],
                      transfer_units=["ohm/ml", "ohm/(S/m)", "ohm/(S/m)"],
                      delta_volume_tested_range_ml=pair["delta_volume_range_ml"], plateaus={})
        output_arrays[label+"_A"] = A
        output_arrays[label+"_Ktt"] = K
        for key, item in plateaus.items():
            y = item["band"]
            f = general_design(A, y, w)
            x = f["coefficients"]
            tt = 1000*K[:, None]*x
            predicted_heart = A[:, :1] @ x[:1]
            nuisance = A[:, 1:] @ x[1:]
            lower, upper = pair["delta_volume_range_ml"]
            quadratic = .5*pair["second_derivative_V_per_ml2"][:, None]*x[0]**2
            fatdesign = np.column_stack((A, base["S"][:, 4]))
            ff = general_design(fatdesign, y, w)
            fatK = np.r_[K, base["K"][4]]
            fatt = 1000*fatK[:, None]*ff["coefficients"]
            metrics = dict(interval_s=[float(item["time"][0]), float(item["time"][-1])],
                frames=y.shape[1], delta_volume_rms_ml=float(rms(x[0])),
                delta_volume_min_ml=float(x[0].min()), delta_volume_max_ml=float(x[0].max()),
                delta_volume_full_window_ptp_ml=float(np.ptp(x[0])),
                extrema_are_not_SV=True, coefficient_rms=rms(x, axis=1),
                delta_sigma_rms_S_m=rms(x[1:], axis=1),
                coefficient_correlations=np.corrcoef(x),
                weighted_residual_pct=f["relative_residual_pct"],
                unweighted_voltage_residual_pct=100*np.linalg.norm(f["residual"])/np.linalg.norm(y),
                projected_rms_mohm=rms(tt, axis=1),
                nuisance_projected_rms_mohm=float(rms(tt[1:].sum(axis=0))),
                total_projected_rms_mohm=float(rms(tt.sum(axis=0))),
                heart_naive_to_partial_correlation=np.corrcoef(f["naive_coefficients"][0], x[0])[0, 1],
                heart_naive_to_partial_rms_difference_pct=100*rms(f["naive_coefficients"][0]-x[0])/rms(x[0]),
                FWL_joint_max_difference_ml=float(np.max(np.abs(x[0]-f["partial_coefficients"][0]))),
                beyond_tested_volume_fraction=float(np.mean((x[0]<lower)|(x[0]>upper))),
                local_quadratic_to_linear_heart_norm_pct=100*np.linalg.norm(w[:, None]*quadratic)/np.linalg.norm(w[:, None]*predicted_heart),
                local_quadratic_diagnostic_only=True,
                free_fat_delta_volume_rms_ml=float(rms(ff["coefficients"][0])),
                free_fat_volume_difference_rms_ml=float(rms(ff["coefficients"][0]-x[0])),
                free_fat_volume_difference_pct=100*rms(ff["coefficients"][0]-x[0])/rms(x[0]),
                free_fat_nuisance_projection_change_rms_mohm=float(rms(fatt[1:].sum(axis=0)-tt[1:].sum(axis=0))),
                free_fat_weighted_residual_pct=ff["relative_residual_pct"],
                bootstrap=moving_bootstrap(y, A, w, K, rng, repetitions))
            result["plateaus"][key] = metrics
            for suffix, values in dict(time_s=item["time"], coefficients=x, projected_mohm=tt,
                prediction_V=f["prediction"], residual_V=f["residual"], corrected_after_soft_lung_V=y-nuisance,
                naive_heart_ml=f["naive_coefficients"][0], free_fat_coefficients=ff["coefficients"]).items():
                output_arrays[f"{label}__{key}__{suffix}"] = values
            summary_rows.append(dict(derivative=label, record=key, delta_volume_rms_ml=metrics["delta_volume_rms_ml"],
                delta_volume_min_ml=metrics["delta_volume_min_ml"], delta_volume_max_ml=metrics["delta_volume_max_ml"],
                projected_volume_rms_mohm=metrics["projected_rms_mohm"][0],
                projected_soft_rms_mohm=metrics["projected_rms_mohm"][1],
                projected_lung_rms_mohm=metrics["projected_rms_mohm"][2],
                weighted_residual_pct=metrics["weighted_residual_pct"],
                beyond_tested_volume_fraction=metrics["beyond_tested_volume_fraction"]))
        cv, traces = cross_validate(blocks, A, w, K, paired, nuisance_indices=(1, 2))
        result["cross_validation"] = cv
        for row in cv:
            cv_rows.append(dict(derivative=label, **row))
        for key, data in traces.items():
            output_arrays[f"{label}__CV__{key}__coefficients"] = data["coefficients"]
        report["derivative_sets"][label] = result
    report["step_coefficient_sensitivity"] = []
    labels = list(geometry["pairs"])
    for i, small in enumerate(labels):
        for large in labels[i+1:]:
            for key in plateaus:
                first = output_arrays[f"{small}__{key}__coefficients"]
                second = output_arrays[f"{large}__{key}__coefficients"]
                report["step_coefficient_sensitivity"].append(dict(first=small, second=large, record=key,
                    volume_waveform_difference_rms_ml=rms(second[0]-first[0]),
                    volume_waveform_difference_pct=100*rms(second[0]-first[0])/rms(first[0]),
                    volume_correlation=np.corrcoef(second[0], first[0])[0, 1]))
    if compare_to:
        report["q_comparison"] = compare_geometry_pass(compare_to, report, output_arrays)
    for source in geometry["source_hashes"].values():
        if sha(source["path"]) != source["sha256"]:
            raise ValueError("Geometry source changed during analysis")
    if max(p["FWL_joint_max_difference_ml"] for ds in report["derivative_sets"].values() for p in ds["plateaus"].values()) > 1e-8:
        raise AssertionError("FWL volume coefficient disagrees with joint solution")
    write_json(out/"geometry_audit.local.json", report)
    write_csv(out/"geometry_summary.csv", summary_rows)
    write_csv(out/"geometry_cross_validation.csv", cv_rows)
    np.savez_compressed(out/"geometry_arrays.npz", **output_arrays)
    geometry_plot(out, report, output_arrays)
    geometry_report(out, report)
    artifacts = ["geometry_audit.local.json", "geometry_summary.csv", "geometry_cross_validation.csv",
                 "geometry_arrays.npz", "geometry_diagnostics.png", "REPORT.md"]
    write_json(out/"completion.local.json", dict(status="completed", implementation_sha256=sha(__file__),
        artifacts={name: sha(out/name) for name in artifacts}))
    print(json.dumps(jsonable(dict(status=report["status"], summary=summary_rows,
                                    step_sensitivity=report["step_coefficient_sensitivity"])), ensure_ascii=False))


def compare_geometry_pass(previous_path, current, current_arrays):
    """Compare quadrature passes only when physical and observation inputs match."""
    path = Path(previous_path)
    old = json.loads(path.read_text(encoding="utf-8"))
    completion = json.loads((path.parent/"completion.local.json").read_text(encoding="utf-8"))
    for source in (path, path.parent/"geometry_arrays.npz"):
        if sha(source) != completion["artifacts"][source.name]:
            raise ValueError("Previous geometry pass artifacts changed")
    g1, g2 = old["geometry"], current["geometry"]
    for key in ("eit_model_sha256", "ttrkg_model_sha256", "sensitivity_sha256", "implementation_sha256"):
        if g1["source_identity"][key] != g2["source_identity"][key]:
            raise ValueError(f"q comparison has changed source implementation/model: {key}")
    if g1["conditions"] != g2["conditions"] or old["observations_sha256"] != current["observations_sha256"]:
        raise ValueError("q comparison requires identical physical conditions and observed windows")
    if not np.allclose(g1["reference"]["sigma"], g2["reference"]["sigma"], rtol=1e-12, atol=0):
        raise ValueError("q comparison conductivities differ")
    for name, variant in g2["variants"].items():
        if not np.isclose(g1["variants"][name]["target_volume_ml"], variant["target_volume_ml"], rtol=1e-12):
            raise ValueError("q comparison changes target volumes")
    rows = []
    with np.load(path.parent/"geometry_arrays.npz") as arrays:
        for label, data in current["derivative_sets"].items():
            v1, v2 = np.asarray(g1["pairs"][label]["JV_eit_V_per_ml"]), np.asarray(g2["pairs"][label]["JV_eit_V_per_ml"])
            w = 1/np.abs(g2["reference"]["V"])
            for record in data["plateaus"]:
                key = f"{label}__{record}__coefficients"
                oldcoef, newcoef = arrays[key], current_arrays[key]
                rows.append(dict(derivative=label, record=record,
                    JV_change_pct=100*np.linalg.norm(w*(v2-v1))/np.linalg.norm(w*v1),
                    volume_waveform_difference_rms_ml=rms(newcoef[0]-oldcoef[0]),
                    volume_waveform_difference_pct=100*rms(newcoef[0]-oldcoef[0])/rms(oldcoef[0]),
                    projected_volume_RMS_change_mohm=data["plateaus"][record]["projected_rms_mohm"][0]-old["derivative_sets"][label]["plateaus"][record]["projected_rms_mohm"][0]))
    return dict(previous_path=str(path), previous_sha256=sha(path),
                quadratures=[g1["reference"]["quadrature"], g2["reference"]["quadrature"]], rows=rows,
                is_mesh_convergence_test=False)


def geometry_plot(out, report, arrays):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for row, record in enumerate(("record_02", "record_04")):
        for label in report["derivative_sets"]:
            prefix = f"{label}__{record}__"
            t = arrays[prefix+"time_s"]
            axes[row, 0].plot(t, arrays[prefix+"coefficients"][0], label=label)
            axes[row, 1].plot(t, arrays[prefix+"projected_mohm"][0], label=label+": volume")
        axes[row, 0].set(title=record+": conditional volume coefficient", xlabel="Time, s", ylabel="Delta V, ml")
        axes[row, 1].set(title="TTRKG projection from volume change", xlabel="Time, s", ylabel="mOhm")
        for ax in axes[row]:
            ax.grid(alpha=.2)
            ax.legend()
    fig.savefig(out/"geometry_diagnostics.png", dpi=170)
    plt.close(fig)


def geometry_report(out, r):
    g = r["geometry"]
    q = g["reference"]["quadrature"]
    pieces = [f"# Совместная оценка ΔV сердца, soft и lung: STL, q{q}\n",
        "Выполнен линейный расчёт на сохранённых FEM-напряжениях и реальных ЭИТ-окнах Ника. "
        "Все пять вариантов имеют одинаковые σ и identity; хэши исходных ЭИТ/ТТРКГ-моделей, конфигурации "
        "и плана материалов проверены. MATLAB и FEM на этом этапе не запускались.\n",
        "**Граница интерпретации:** заданное ρ миокарда нельзя автоматически считать эффективным ρ "
        "всего сердечного компартмента. Этот проход с фиксированным ρ является диагностическим; "
        "эффективное базовое ρ отдельно для каждой формы здесь не идентифицировалось. Знак электрического "
        "отклика при одном фиксированном ρ не является основанием отклонить сферу или другую аппроксимацию. "
        "После идентификации базового эффективного ρ для каждой формы динамические чувствительности "
        "и пульсовые оценки должны быть пересчитаны.\n",
        f"Опорный объём сердца по материалам: **{g['reference']['volume']:.6f} мл**; "
        f"целевой STL-объём {g['reference']['target_volume_ml']:.6f} мл. "
        f"ρ сердца = **{1/g['reference']['sigma'][3]:.4g} Ом·м**, фиксированный диагностический сценарий. "
        "Лёгкие — единый компартмент. Предполагаемое состояние: кандидат задержки на глубоком вдохе; "
        "метки дыхательных команд не зарегистрированы.\n",
        "## Оператор и единицы\n",
        "$\\delta u=J_V\\delta V+J_s\\delta\\sigma_s+J_l\\delta\\sigma_l+e$. "
        "Вектор неизвестных: [мл, См/м, См/м]. Геометрический столбец рассчитан по разности "
        "напряжений и фактических объёмов heart из row.volume_ml[3]. Отдельная δσ сердца исключена. "
        "Для ТТРКГ аналогично использована разность Ztt в Ом. Взвешивание — 1/|V_eit опоры|. "
        "Полоса 0,7–3 Гц и оконные экстремумы не устанавливают сердечное происхождение, УО или ФВ.\n"]
    for label, data in r["derivative_sets"].items():
        pair = g["pairs"][label]
        ds = data["design"]
        pieces += [f"## Производная {label}\n",
            f"Фактический проверенный диапазон ΔV: **{pair['delta_volume_range_ml'][0]:.4f}…"
            f"{pair['delta_volume_range_ml'][1]:.4f} мл**. K_V = **{pair['Ktt'][0]:.8g} Ом/мл**. "
            f"Расхождение односторонних ЭИТ-производных в относительной метрике: "
            f"**{pair['weighted_asymmetry_pct']:.3f}%**.\n",
            f"Число обусловленности после нормирования столбцов: **{ds['normalized_condition']:.4f}**; "
            f"сингулярные числа {', '.join(f'{v:.5g}' for v in ds['normalized_singular_values'])}. "
            f"Доля геометрического столбца после удаления soft+lung: **{ds['partial_unique_fractions'][0]:.4f}**. "
            f"Косинусы volume/soft, volume/lung, soft/lung: "
            f"{ds['column_cosines'][0,1]:.4f}, {ds['column_cosines'][0,2]:.4f}, {ds['column_cosines'][1,2]:.4f}.\n"]
        for key, m in data["plateaus"].items():
            pieces.append(f"**{key}: RMS ΔV = {m['delta_volume_rms_ml']:.4f} мл**, диапазон "
                f"{m['delta_volume_min_ml']:.4f}…{m['delta_volume_max_ml']:.4f} мл. "
                f"RMS проекций ТТРКГ [volume, soft, lung] = "
                f"**{', '.join(f'{v:.4f}' for v in m['projected_rms_mohm'])} мОм**. "
                f"Невязка ЭИТ {m['weighted_residual_pct']:.3f}% в относительной метрике и "
                f"{m['unweighted_voltage_residual_pct']:.3f}% в вольтовой. "
                f"За диапазон проверенных объёмов вышли {100*m['beyond_tested_volume_fraction']:.2f}% отсчётов.\n")
            pieces.append(f"При освобождении жира RMS ΔV = {m['free_fat_delta_volume_rms_ml']:.4f} мл; "
                f"изменение временной кривой ΔV — {m['free_fat_volume_difference_pct']:.2f}% исходного RMS, "
                f"изменение суммарной тканевой поправки — {m['free_fat_nuisance_projection_change_rms_mohm']:.4f} мОм RMS. "
                f"Локальный квадратичный индикатор по геометрической производной — "
                f"{m['local_quadratic_to_linear_heart_norm_pct']:.2f}% от нормы линейной сердечной проекции. "
                "Этот индикатор не заменяет нелинейного пересчёта.\n")
            ci = m["bootstrap"]["20"]["coefficient_rms_q05_q50_q95"]
            pieces.append(f"Ресэмплирование наблюдённых блоков по 1 с даёт 5–95% диапазон "
                f"условного RMS ΔV {ci[0,0]:.4f}–{ci[2,0]:.4f} мл. "
                "Систематическая ошибка физического оператора сюда не входит.\n")
        pieces.append("<details><summary>Проверка на раздельных временных блоках и исключённых каналах</summary>\n\n"
            "| Вариант | Мин. невязка, % | Медиана, % | Макс., % |\n|---|---:|---:|---:|\n")
        for method in ("relative_3", "discrepancy_3", "relative_soft_lung"):
            rows = [x for x in data["cross_validation"] if x["method"] == method]
            values = np.array([x["heldout_residual_pct"] for x in rows])
            pieces.append(f"| {method} | {values.min():.3f} | {np.median(values):.3f} | {values.max():.3f} |\n")
        pieces.append("\n</details>\n")
        rows = [x for x in data["cross_validation"] if x["method"] == "discrepancy_3"]
        values = np.array([x["coefficient_rms"][0] for x in rows])
        pieces.append(f"С весами, оценёнными на другом временном блоке, RMS ΔV на проверочных "
            f"блоках составляет {values.min():.4f}–{values.max():.4f} мл. Это перенос весов "
            "реконструкции, не прогноз следующего сердечного цикла. Короткие независимым образом "
            "отфильтрованные блоки содержат только 2–2,5 с; взаимные каналы исключаются совместно.\n")
    pieces.append("## Чувствительность к шагу и q\n")
    for c in g["step_comparisons"]:
        pieces.append(f"{c['first']} → {c['second']}: изменение J_V в относительной метрике "
            f"**{c['JV_relative_difference_pct']:.3f}%**; изменение K_V "
            f"{c['Jtt_difference_ohm_per_ml']:.8g} Ом/мл.\n")
    for c in r["step_coefficient_sensitivity"]:
        pieces.append(f"{c['record']}: изменение кривой ΔV при смене шага "
            f"**{c['volume_waveform_difference_rms_ml']:.4f} мл RMS ({c['volume_waveform_difference_pct']:.2f}%)**.\n")
    if "q_comparison" in r:
        pieces.append(f"Сравнены q{r['q_comparison']['quadratures'][0]} и q{q} при одинаковых физических "
                      "условиях и данных. Это проверка квадратуры распределения материалов, не сходимости FEM-сетки.\n")
        for row in r["q_comparison"]["rows"]:
            pieces.append(f"{row['derivative']}, {row['record']}: изменение J_V {row['JV_change_pct']:.3f}%; "
                f"изменение ΔV {row['volume_waveform_difference_rms_ml']:.5f} мл RMS "
                f"({row['volume_waveform_difference_pct']:.3f}%).\n")
    else:
        pieces.append("Другой порядок q в этом результате не проверен; сходимость по q не установлена.\n")
    pieces += ["## Граница вывода\n",
        "Получена условная кривая изменения объёма по выбранному закону масштабирования STL при фиксированной "
        "проводимости сердца. Её связь с действительной динамикой камер и УО/ФВ ещё не проверена. "
        "Остаток, зависимость от шага, весов и фиксированного жира должны учитываться вместе. "
        "Результаты ЭИТ и измеренная ТТРКГ разных сессий не синхронизировались и не вычитались.\n",
        "- [Численный отчёт и происхождение](geometry_audit.local.json).\n"
        "- [Краткая таблица](geometry_summary.csv).\n"
        "- [Проверка по блокам и каналам](geometry_cross_validation.csv).\n"
        "- [Кривые, поправки, прогнозы и остатки](geometry_arrays.npz).\n"
        "- [График ΔV и его проекции](geometry_diagnostics.png).\n"]
    (out/"REPORT.md").write_text("\n".join(pieces).replace("|\n\n|", "|\n|"), encoding="utf-8")


def effective_fit_pass(data_root, output, source_dir):
    """Consume three completed per-shape fits and reuse exact saved real windows.

    Each shape has its OWN baseline-fitted soft/lung/heart conductivity. That
    fitted state is held fixed for the local geometric derivative and pulse
    pass. High effective rho and either sign of KV_tt are retained, not rejected.
    No parameter fitting, FEM, bootstrap or new temporal filtering occurs here.
    """
    root, out, source = map(lambda p: Path(p).resolve(), (data_root, output, source_dir))
    names = ("STL", "G1", "L1")
    paths = {name: source/(name+".mat") for name in names}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Await all three completed MAT files: " + ", ".join(missing))
    config_path = source.with_suffix(".local.json")
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if config["free_indices"] != [1, 2, 4]:
        raise ValueError("Expected baseline free parameters soft/lung/heart")
    bound = {"config_sha256": config_path, "plan_sha256": Path(config["material_plan"]),
             "reference_sha256": Path(config["reference_prediction_file"]),
             "eit_model_sha256": Path(config["eit_model_file"]),
             "ttrkg_model_sha256": Path(config["ttrkg_model_file"])}
    expected = {key: sha(path) for key, path in bound.items()}
    provenance = {str(path): expected[key] for key, path in bound.items()}
    prior_dir = root/"pulse_identifiability"
    prior_path, pulse_path = prior_dir/"audit.local.json", prior_dir/"audit_arrays.npz"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    completion = json.loads((prior_dir/"completion.local.json").read_text(encoding="utf-8"))
    for path in (prior_path, pulse_path):
        if sha(path) != completion["artifacts"][path.name]:
            raise ValueError("Prior real-window artifact changed")
        provenance[str(path)] = sha(path)
    if prior["frequency_Hz"] != config["frequency_Hz"]:
        raise ValueError("Frequency differs from measured pulse-window audit")
    patient_path = root/"patient_v3/patient_check.local.json"
    patient = json.loads(patient_path.read_text(encoding="utf-8"))
    if patient["current_A"] != config["current_A"]:
        raise ValueError("Current differs from measured EIT reference")
    provenance[str(patient_path)] = sha(patient_path)
    for number in (2, 4):
        path = root/f"nix_july_{number:02d}_reader.mat"
        original = next(v for k, v in prior["source_hashes"].items() if Path(k).name == path.name)
        if sha(path) != original:
            raise ValueError("Real EIT reader data changed")
        provenance[str(path)] = original
    Vref = loadmat(config["reference_prediction_file"])["Vref"].ravel()
    window_names = list(prior["full_plateaus"]) + list(prior["blocks"])
    with np.load(pulse_path) as saved:
        windows = {key: (saved[key+"_band_V"].copy(), saved[key+"_time_s"].copy()) for key in window_names}
    report = dict(status="conditional_per_shape_effective_rho_pulse_pass", parameter_order=["volume_ml", "soft_S_m", "lung_S_m"],
        units=["ml", "S/m", "S/m"], baseline_fit_free=["soft", "lung", "heart"], baseline_fit_fixed=["bone", "fat"],
        pulse_heart_sigma_fixed_at_each_shape_fit=True, effective_rho_independently_measured=False,
        AI_training=False, FEM_run=False, SV_EF_computed=False, shapes={}, window_metadata={},
        source_hashes=provenance, implementation_sha256=sha(__file__), band_Hz=prior["band_Hz"],
        same_session_EIT_TTRKG=False, conditions={k: config[k] for k in ("frequency_Hz", "current_A", "montage_basis", "breath_interval_status")},
        weighting="per-shape 1/abs(V_eit); baseline-fit data shared; not independent validation")
    arrays, rows, checks, shared_identity, fixed_sigma = {}, [], [], None, None
    for name, path in paths.items():
        before = sha(path)
        d = loadmat(path, simplify_cells=True)
        row, identity = d["row"], d["identity"]
        if before != sha(path):
            raise ValueError("MAT changed during load")
        provenance[str(path)] = before
        if row["id"] != name or any(identity.get(k) != v for k, v in expected.items()):
            raise ValueError("Shape identity/source condition mismatch")
        if shared_identity is None:
            shared_identity = identity
        if identity != shared_identity:
            raise ValueError("Shapes have different physical/reference identities")
        S, V = np.asarray(d["S_eit"]), np.asarray(d["V_eit"]).ravel()
        sigma, kt, jv = (np.asarray(d[k]).ravel() for k in ("sigma", "Ktt", "JV_eit"))
        kv, z = float(d["KV_tt"]), float(d["Ztt"])
        if S.shape != (208, 5) or V.shape != (208,) or jv.shape != (208,) or kt.shape != (5,) or sigma.shape != (5,):
            raise ValueError("Invalid small MAT dimensions")
        if not np.isfinite(np.r_[S.ravel(), V, jv, kt, sigma, kv, z]).all() or np.any(sigma <= 0) or np.any(V == 0):
            raise ValueError("Invalid physical arrays")
        if not np.allclose(row["rho_ohm_m"], 1/sigma, rtol=1e-12, atol=0):
            raise ValueError("Stored rho and sigma disagree")
        if fixed_sigma is None:
            fixed_sigma = sigma[[2, 4]]
        if not np.array_equal(fixed_sigma, sigma[[2, 4]]):
            raise ValueError("Fixed bone/fat differ across shapes")
        minus, plus = d["probes"]
        volume = float(np.asarray(row["volume_ml"]).ravel()[3])
        vm, vp = (float(np.asarray(p["volume_ml"]).ravel()[3]) for p in (minus, plus))
        if not vm < volume < vp:
            raise ValueError("Probe volumes do not bracket baseline")
        check_jv = (np.asarray(plus["V_eit"])-np.asarray(minus["V_eit"])).ravel()/(vp-vm)
        check_kv = (float(plus["Ztt"])-float(minus["Ztt"]))/float(vp-vm)
        if not np.allclose(jv, check_jv, rtol=1e-11, atol=1e-14) or not np.isclose(kv, check_kv, rtol=1e-11, atol=1e-14):
            raise ValueError("Saved geometric sensitivity differs from actual probe volumes/vectors")
        A, K, w = np.column_stack((jv, S[:, 0], S[:, 1])), np.r_[kv, kt[:2]], 1/np.abs(V)
        A4, K4 = np.column_stack((A, S[:, 3])), np.r_[K, kt[3]]
        baseline_residual = 100*np.linalg.norm(Vref-V)/np.linalg.norm(Vref)
        if not np.isclose(baseline_residual, row["eit_relative_residual_pct"], rtol=1e-10):
            raise ValueError("Baseline residual does not match shared real Vref")
        state = dict(rho_ohm_m=1/sigma, sigma_S_m=sigma, converged=bool(row["converged"]),
            stop_reason=row["stop_reason"], baseline_voltage_residual_pct=baseline_residual,
            baseline_relative_objective=row["relative_objective"], volume_ml=volume,
            target_volume_ml=row["target_volume_ml"], tested_delta_volume_ml=[vm-volume, vp-volume],
            K_V_ohm_per_ml=kv, Ztt_ohm=z, design=design_summary(general_design(A, Vref, w)),
            unresolved_A4_design=design_summary(general_design(A4, Vref, w)),
            unresolved_A4_parameter_order=["volume_ml", "soft_S_m", "lung_S_m", "heart_S_m"],
            unresolved_A4_status="dynamic composition alternative; not accepted method or identified blood/myocardium mixture",
            identity=identity, source_row=row, pulse_windows={})
        arrays[name+"__A"], arrays[name+"__K"] = A, K
        arrays[name+"__A4"], arrays[name+"__K4"] = A4, K4
        for key, (y, t) in windows.items():
            f = general_design(A, y, w)
            f4 = general_design(A4, y, w)
            x = f["coefficients"]
            x4 = f4["coefficients"]
            tt = 1000*K[:, None]*x
            tt4 = 1000*K4[:, None]*x4
            direct = np.linalg.lstsq(w[:, None]*A, w[:, None]*y, rcond=None)[0]
            checks.append(float(np.max(abs(x-direct))))
            m = dict(shape=name, window=key, baseline_fit_converged=state["converged"],
                weighted_residual_pct=f["relative_residual_pct"],
                unweighted_voltage_residual_pct=100*np.linalg.norm(f["residual"])/np.linalg.norm(y),
                delta_volume_rms_ml=float(rms(x[0])), delta_volume_min_ml=float(x[0].min()),
                delta_volume_max_ml=float(x[0].max()), delta_sigma_soft_rms_S_m=float(rms(x[1])),
                delta_sigma_lung_rms_S_m=float(rms(x[2])), TT_volume_rms_mohm=float(rms(tt[0])),
                TT_soft_rms_mohm=float(rms(tt[1])), TT_lung_rms_mohm=float(rms(tt[2])),
                TT_total_rms_mohm=float(rms(tt.sum(axis=0))), normalized_condition=f["normalized_condition"],
                beyond_probe_range_fraction=float(np.mean((x[0]<vm-volume)|(x[0]>vp-volume))),
                unresolved_A4_volume_rms_ml=float(rms(x4[0])),
                unresolved_A4_volume_rms_change_pct=float(100*(rms(x4[0])/rms(x[0])-1)),
                unresolved_A4_volume_waveform_difference_rms_ml=float(rms(x4[0]-x[0])),
                unresolved_A4_weighted_residual_pct=f4["relative_residual_pct"],
                unresolved_A4_weighted_residual_reduction_pp=f["relative_residual_pct"]-f4["relative_residual_pct"],
                unresolved_A4_unweighted_voltage_residual_pct=100*np.linalg.norm(f4["residual"])/np.linalg.norm(y),
                unresolved_A4_heart_sigma_rms_S_m=float(rms(x4[3])),
                unresolved_A4_max_relative_heart_sigma_change=float(np.max(abs(x4[3]))/sigma[3]),
                unresolved_A4_nonpositive_heart_sigma_fraction=float(np.mean(sigma[3]+x4[3]<=0)),
                unresolved_A4_beyond_probe_range_fraction=float(np.mean((x4[0]<vm-volume)|(x4[0]>vp-volume))),
                unresolved_A4_TT_components_rms_mohm=rms(tt4, axis=1))
            state["pulse_windows"][key] = m
            rows.append(m)
            report["window_metadata"][key] = dict(interval_s=[float(t[0]), float(t[-1])], frames=len(t),
                source="unchanged prior audit array", kind="full_plateau" if key in prior["full_plateaus"] else "independently_filtered_block")
            for suffix, value in dict(time_s=t, coefficients=x, TT_components_mohm=tt,
                residual_V=f["residual"], prediction_V=f["prediction"],
                unresolved_A4_coefficients=x4, unresolved_A4_TT_components_mohm=tt4,
                unresolved_A4_residual_V=f4["residual"], unresolved_A4_prediction_V=f4["prediction"]).items():
                arrays[f"{name}__{key}__{suffix}"] = value
        report["shapes"][name] = state
    correlations = []
    for key in windows:
        for i, first in enumerate(names):
            for second in names[i+1:]:
                a, b = (arrays[f"{n}__{key}__coefficients"][0] for n in (first, second))
                ta, tb = (arrays[f"{n}__{key}__TT_components_mohm"][0] for n in (first, second))
                correlations.append(dict(window=key, first=first, second=second,
                    volume_curve_pearson=float(np.corrcoef(a, b)[0, 1]),
                    TT_volume_curve_pearson=float(np.corrcoef(ta, tb)[0, 1]),
                    volume_curve_difference_rms_ml=float(rms(a-b))))
    if max(checks) > 1e-8:
        raise AssertionError("Unit-scaled solution disagrees with independent least squares")
    report["correlations"], report["direct_lstsq_max_difference"] = correlations, max(checks)
    if any(sha(path) != digest for path, digest in provenance.items()):
        raise ValueError("Input changed during pass")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/"effective_pulse.local.json", report)
    write_csv(out/"pulse_summary.csv", rows)
    write_csv(out/"curve_correlations.csv", correlations)
    np.savez_compressed(out/"pulse_arrays.npz", **arrays)
    text = ["# Пульсовой расчёт после индивидуальной для каждой формы оценки эффективного ρ\n",
        "Использованы STL, G1 и L1 и неизменённые реальные окна прежнего аудита: плато 02 (36–44 с), "
        "04 (30–37 с), а также четыре отдельно отфильтрованных блока. Базовые soft/lung/heart "
        "оценены ранее по одному Vref02; bone/fat фиксированы. Здесь использованы готовые результаты.\n",
        "**Условный расчёт при собственном fitted ρ каждой формы.** Это параметры физической обратной задачи, "
        "не независимо измеренные ρ и не обучение ИИ. Для пульсового прохода σ сердца фиксирована на своей "
        "базовой оценке; неизвестные — ΔV сердца, Δσ soft и Δσ единого lung. Геометрические производные "
        "перепроверены по фактическим объёмам probes. Высокое эффективное ρ и знак производной не служат "
        "основанием отклонить форму. Сходимость базового решателя сохраняется отдельным флагом.\n",
        "<details><summary>Базовые оценки и численная обусловленность</summary>\n\n"
        "| Форма | ρ soft / lung / heart, Ом·м | Сходимость | Базовая невязка, % | κ норм. J | K_V, мОм/мл |\n"
        "|---|---|---|---:|---:|---:|\n"]
    for name, s in report["shapes"].items():
        text.append(f"| {name} | {' / '.join(f'{s['rho_ohm_m'][j]:.4f}' for j in (0,1,3))} | {s['converged']} | "
                    f"{s['baseline_voltage_residual_pct']:.3f} | {s['design']['normalized_condition']:.3f} | {1000*s['K_V_ohm_per_ml']:.5f} |\n")
    text.append("\n</details>\n\n<details><summary>Полные пульсовые плато</summary>\n\n"
        "| Форма / запись | RMS ΔV, мл | Диапазон ΔV, мл | Невязка относит. / вольтовая, % | RMS ТТРКГ volume / soft / lung, мОм |\n"
        "|---|---:|---|---|---|\n")
    for row in rows:
        if row["window"] in prior["full_plateaus"]:
            text.append(f"| {row['shape']} / {row['window']} | {row['delta_volume_rms_ml']:.3f} | "
                f"{row['delta_volume_min_ml']:.3f}…{row['delta_volume_max_ml']:.3f} | "
                f"{row['weighted_residual_pct']:.2f} / {row['unweighted_voltage_residual_pct']:.2f} | "
                f"{row['TT_volume_rms_mohm']:.3f} / {row['TT_soft_rms_mohm']:.3f} / {row['TT_lung_rms_mohm']:.3f} |\n")
    text.append("\n</details>\n\nКорреляции кривых ΔV без изменения знака и без временного совмещения:\n")
    for c in correlations:
        if c["window"] in prior["full_plateaus"]:
            text.append(f"- {c['window']}, {c['first']}–{c['second']}: r = {c['volume_curve_pearson']:.5f}; "
                        f"RMS различия {c['volume_curve_difference_rms_ml']:.3f} мл.\n")
    text.append("\n**Неразрешённая альтернатива A4:** [J_V, J_soft, J_lung, J_heart] допускает "
        "одновременно ΔV и Δσ сердца. Базовая оценка эффективного ρ не определяет динамическое "
        "соотношение крови и миокарда. A4 служит диагностикой этой неоднозначности и не принимается "
        "как методика восстановления объёма. Полный ранг не устраняет большой зависимости оценки "
        "от выбора неизвестных; уменьшение остатка само по себе не выбирает физиологическое объяснение.\n\n"
        "<details><summary>A4: нормирование столбцов, SVD и уникальные доли</summary>\n\n"
        "| Форма | Сингулярные числа | κ A4 | Уникальная доля J_V | Уникальная доля J_heart |\n"
        "|---|---|---:|---:|---:|\n")
    for name, state in report["shapes"].items():
        d4 = state["unresolved_A4_design"]
        text.append(f"| {name} | {', '.join(f'{v:.5g}' for v in d4['normalized_singular_values'])} | "
            f"{d4['normalized_condition']:.3f} | {d4['partial_unique_fractions'][0]:.5f} | "
            f"{d4['partial_unique_fractions'][3]:.5f} |\n")
    text.append("\n</details>\n\n<details><summary>A4: изменение ΔV и невязки относительно A3</summary>\n\n"
        "| Форма / запись | RMS ΔV A4, мл | Изменение RMS, % | RMS разности кривых, мл | Снижение относит. невязки, п.п. |\n"
        "|---|---:|---:|---:|---:|\n")
    for row in rows:
        if row["window"] in prior["full_plateaus"]:
            text.append(f"| {row['shape']} / {row['window']} | {row['unresolved_A4_volume_rms_ml']:.3f} | "
                f"{row['unresolved_A4_volume_rms_change_pct']:.2f} | "
                f"{row['unresolved_A4_volume_waveform_difference_rms_ml']:.3f} | "
                f"{row['unresolved_A4_weighted_residual_reduction_pp']:.3f} |\n")
    text.append("\n</details>\n")
    text.append("\nПолоса 0,7–3 Гц не доказывает сердечное происхождение. ΔV целого эквивалентного сердца "
        "и оконный размах не являются УО. ФВ не рассчитана. Проекции ТТРКГ вычислены через сохранённые "
        "операторы; реальные разные сессии не синхронизировались. Высокая корреляция форм не подтверждает "
        "абсолютную точность объёма. Веса 1/|V_eit| свои для каждой формы, поэтому для сопоставления сохранена "
        "также общая вольтовая метрика.\n\n"
        "- [Результаты, идентичность условий и хэши](effective_pulse.local.json).\n"
        "- [Числа всех шести окон](pulse_summary.csv).\n"
        "- [Корреляции объёма и его ТТРКГ-проекции](curve_correlations.csv).\n"
        "- [Кривые, прогнозы и остатки](pulse_arrays.npz).\n")
    (out/"README.md").write_text("\n".join(text).replace("|\n\n|", "|\n|"), encoding="utf-8")
    files = ("effective_pulse.local.json", "pulse_summary.csv", "curve_correlations.csv", "pulse_arrays.npz", "README.md")
    write_json(out/"completion.local.json", dict(status="completed", code_sha256=sha(__file__),
        artifacts={name: sha(out/name) for name in files}))
    print(json.dumps(jsonable(dict(status=report["status"], full_plateaus=[r for r in rows if r["window"] in prior["full_plateaus"]],
        correlations=[c for c in correlations if c["window"] in prior["full_plateaus"]])), ensure_ascii=False))


def main(data_root, output, repetitions=400, seed=20261028, geometry_manifest=None):
    root, out = Path(data_root).resolve(), Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    files = [root/"patient_v3/reference_prediction.mat", root/"patient_v3/transfer_check.local.json",
             root/"patient_v3/patient_check.local.json", root/"patient_inspiration_candidate_04/patient_check.local.json"]
    small = loadmat(files[0])
    S, sigma = small["S"], small["sigma"].ravel()
    Vpred, Vref = small["Vpred"].ravel(), small["Vref"].ravel()
    if S.shape != (208, 5) or not np.array_equal(np.flatnonzero(~small["fixed"].ravel().astype(bool)), FREE):
        raise ValueError("Unexpected saved design/material mask")
    transfer, first, second = [json.loads(p.read_text(encoding="utf-8-sig")) for p in files[1:]]
    if transfer["names"] != NAMES or first["names"] != NAMES or second["names"] != NAMES:
        raise ValueError("Unexpected material ordering")
    if not np.allclose(sigma, first["conditional_fit"]["sigma_S_m"], rtol=1e-12, atol=0):
        raise ValueError("Exported Jacobian and saved state disagree")
    Kall = np.asarray(transfer["K_sigma_ohm_per_S_m"])
    K, A, w = Kall[FREE], S[:, FREE], 1/np.abs(Vpred)
    pairs = reciprocal_pairs(S, Vpred)
    sos = butter(3, [.7, 3], btype="bandpass", fs=20, output="sos")
    rng = np.random.default_rng(seed)
    report = dict(status="conditional_linear_identifiability_audit", linear_J_only=True,
                  cardiac_origin_proven=False, tissue_pulse_independently_validated=False,
                  cross_session_synchrony=False, FEM_run=False, AI_training=False,
                  frame_rate_Hz=20, band_Hz=[.7, 3], frequency_Hz=first["frequency_Hz"],
                  material_order=NAMES, free_order=[NAMES[i] for i in FREE],
                  sigma_reference_S_m=sigma, Ktt_ohm_per_S_m=Kall,
                  TTRKG_projection="Saved v3 operator, same frequency, hypothetical arm montage; no subtraction from measured TTRKG",
                  plateau_labels="Candidates inferred from signal shape; no verified respiratory command timestamps",
                  second_reference=dict(rho_ohm_m=second["conditional_fit"]["rho_ohm_m"],
                      nonlinear_fit_residual_pct=second["conditional_fit"]["residual_pct"],
                      available_report_log_condition=second["conditional_fit"]["free_log_condition"],
                      compact_J_available=(root/"patient_inspiration_candidate_04/reference_prediction.mat").exists(),
                      calculation_J="v3 for both recordings; no reuse of fourth-state sigma as a new derivative"),
                  filter=dict(type="third-order Butterworth SOS, zero-phase", independent_block_detrend="linear",
                              block_gap_s=.6, block_edge_trim_s=.6),
                  uncertainty="observed block/channel/weight sensitivity and multichannel moving-block bootstrap; no common noise level",
                  source_hashes={str(p): sha(p) for p in files})
    report["design_conductivity"] = design_summary(general_design(A, Vref, w))
    report["design_relative_conductivity"] = design_summary(general_design(A*sigma[FREE], Vref, w))
    report["design_all_five_relative"] = design_summary(general_design(S*sigma, Vref, w))
    report["reciprocal_pairs"] = pairs
    report["full_plateaus"], blocks, arrays = {}, {}, {}
    report["filter_edge_sensitivity"] = []
    report["free_fat_sensitivity"] = {}
    for record, lo, hi in ((2, 36., 44.), (4, 30., 37.)):
        path = root/f"nix_july_{record:02d}_reader.mat"
        reader = loadmat(path)
        v = reader["vv"]
        t = np.arange(v.shape[1])/20
        if not np.isclose(reader["fs_hz"].item(), 20) or not np.allclose(reader["times_s"].ravel()-reader["times_s"].ravel()[0], t, atol=1e-7):
            raise ValueError("Reader time axis mismatch")
        report["source_hashes"][str(path)] = sha(path)
        band = sosfiltfilt(sos, v, axis=1)
        select = (t>=lo) & (t<=hi)
        y = band[:, select]
        r = general_design(A, y, w)
        label = f"record_{record:02d}"
        metrics = fitted_metrics(r, sigma[FREE], K, pairs, y)
        metrics.update(interval_s=[lo, hi], frames=int(select.sum()),
                       bootstrap=moving_bootstrap(y, A, w, K, rng, repetitions))
        report["full_plateaus"][label] = metrics
        arrays[label+"_time_s"] = t[select]
        arrays[label+"_coeff_S_m"] = r["coefficients"]
        arrays[label+"_naive_heart_S_m"] = r["naive_coefficients"][2]
        arrays[label+"_projected_mohm"] = 1000*K[:, None]*r["coefficients"]
        arrays[label+"_band_V"] = y
        savedpath = root/f"patient_v3/nix_july_{record:02d}_conditional_band.mat"
        saved = loadmat(savedpath)
        metrics["saved_projection_delta_sigma_max_difference"] = np.max(np.abs(saved["delta_sigma"][FREE][:, select]-r["coefficients"]))
        metrics["saved_band_max_difference"] = np.max(np.abs(saved["band"][:, select]-y))
        report["source_hashes"][str(savedpath)] = sha(savedpath)
        fatfit = general_design(S[:, [0, 1, 3, 4]], y, w)
        allfit = general_design(S, y, w)
        fat_tt = 1000*Kall[[0, 1, 3, 4], None]*fatfit["coefficients"]
        original_tt = 1000*K[:, None]*r["coefficients"]
        report["free_fat_sensitivity"][label] = dict(
            free_fat_relative_residual_pct=fatfit["relative_residual_pct"],
            free_fat_projected_rms_mohm=rms(1000*Kall[[0, 1, 3, 4], None]*fatfit["coefficients"], axis=1),
            free_fat_coefficient_change_pct=100*rms(fatfit["coefficients"][:3]-r["coefficients"], axis=1)/rms(r["coefficients"], axis=1),
            free_fat_soft_plus_fat_rms_mohm=rms(fat_tt[0]+fat_tt[3]),
            free_fat_soft_plus_fat_difference_from_fixed_fat_mohm=rms(fat_tt[0]+fat_tt[3]-original_tt[0]),
            free_fat_nuisance_sum_difference_mohm=rms(fat_tt[[0, 1, 3]].sum(axis=0)-original_tt[:2].sum(axis=0)),
            free_fat_soft_fat_correlation=np.corrcoef(fat_tt[0], fat_tt[3])[0, 1],
            all_five_relative_residual_pct=allfit["relative_residual_pct"],
            all_five_projected_rms_mohm=rms(1000*Kall[:, None]*allfit["coefficients"], axis=1),
            note="Omitted-material sensitivity only; estimated bone/fat variation not assigned physiological origin")
        mid = (lo+hi)/2
        for suffix, (a, b) in zip(("a", "b"), ((lo, mid-.3), (mid+.3, hi))):
            idx = np.flatnonzero((t>=a-1e-9) & (t<=b+1e-9))
            blockband = sosfiltfilt(sos, detrend(v[:, idx], axis=1, type="linear"), axis=1)
            for trim in (.25, .6, .9):
                inner = (t[idx]>=a+trim-1e-8) & (t[idx]<=b-trim+1e-8)
                independent = general_design(A, blockband[:, inner], w)
                full = general_design(A, band[:, idx[inner]], w)
                key = label+"_"+suffix
                report["filter_edge_sensitivity"].append(dict(block=key, trim_s=trim,
                    projected_rms_mohm=rms(1000*K[:, None]*independent["coefficients"], axis=1),
                    relative_waveform_difference_from_fullrecord_pct=100*rms(independent["coefficients"]-full["coefficients"], axis=1)/rms(full["coefficients"], axis=1)))
                if trim == .6:
                    blocks[key] = dict(band=blockband[:, inner], times=t[idx[inner]], raw_interval_s=[a, b],
                                       retained_interval_s=[float(t[idx[inner][0]]), float(t[idx[inner][-1]])])
    report["blocks"] = {}
    for name, b in blocks.items():
        fit = general_design(A, b["band"], w)
        report["blocks"][name] = dict(raw_interval_s=b["raw_interval_s"], retained_interval_s=b["retained_interval_s"],
                                     frames=b["band"].shape[1], metrics=fitted_metrics(fit, sigma[FREE], K, pairs, b["band"]),
                                     bootstrap=moving_bootstrap(b["band"], A, w, K, rng, repetitions))
        arrays[name+"_band_V"] = b["band"]
        arrays[name+"_time_s"] = b["times"]
    cv, traces = cross_validate(blocks, A, w, K, pairs)
    report["cross_validation"] = cv
    report["weight_clip_sensitivity"] = []
    for key, train in blocks.items():
        # Test only the other record to keep this sweep distinct from in-sample fit.
        for other, test in blocks.items():
            if key[:9] == other[:9]:
                continue
            for lo, hi in ((.05, .95), (.1, .9), (.2, .8)):
                wt, _, limits = discrepancy_weights(A, train["band"], w, lo, hi)
                f = general_design(A, test["band"], wt)
                report["weight_clip_sensitivity"].append(dict(train=key, test=other, quantiles=[lo, hi],
                    empirical_limits=limits, projected_rms_mohm=rms(1000*K[:, None]*f["coefficients"], axis=1)))
    for key, data in traces.items():
        for suffix, value in data.items():
            arrays[key+"__"+suffix] = value
    # Linear algebra checks with REAL observations, including unit invariance.
    test_y = next(iter(blocks.values()))["band"]
    base = general_design(A, test_y, w)
    scales = np.array([1e3, .02, 7.])
    scaled = general_design(A*scales, test_y, w)
    invariance = np.max(np.abs(scaled["coefficients"]*scales[:, None]-base["coefficients"]))
    fwl_error = np.max(np.abs(base["partial_coefficients"]-base["coefficients"]))
    orthogonality = np.max(np.abs(base["normalized_design"].T @ base["weighted_residual"]))
    if max(invariance, fwl_error, orthogonality) > 1e-10:
        raise AssertionError("Linear algebra consistency check failed")
    if any(m["saved_projection_delta_sigma_max_difference"] > 1e-10 or m["saved_band_max_difference"] > 1e-12
           for m in report["full_plateaus"].values()):
        raise AssertionError("Independent calculation disagrees with saved band projection")
    report["numerical_checks"] = dict(unit_rescaling_max_abs=invariance, FWL_max_abs=fwl_error,
                                      weighted_residual_orthogonality=orthogonality)
    if geometry_manifest:
        geometry = load_volume_design(geometry_manifest)
        gw = 1/np.abs(geometry["reference"]["V"])
        report["geometry_operator"] = geometry
        report["geometry_analysis"] = {}
        for name, pair in geometry["pairs"].items():
            gc, gtraces = cross_validate(blocks, pair["A"], gw, pair["Ktt"], pairs, nuisance_indices=(1, 2))
            report["geometry_analysis"][name] = dict(design=design_summary(general_design(pair["A"], test_y, gw)), cross_validation=gc)
            for trace_name, trace in gtraces.items():
                for field, data in trace.items():
                    arrays[f"geometry_{name}__{trace_name}__{field}"] = data
    contract = dict(material_order=NAMES, units={"V_eit": "V", "S_eit": "V/(S/m)", "Ktt": "ohm/(S/m)",
        "Ztt": "ohm", "volume_ml": "ml", "sigma_S_m": "S/m"}, reference="stl_reference",
        fixed_heart_sigma_S_m=.25, fixed_heart_rho_ohm_m=4., heart_rho_status="diagnostic assumption, not measured",
        variants={"stl_reference": "stl_reference.mat", "minus_2": "minus_2.mat", "plus_2": "plus_2.mat",
                  "minus_5": "minus_5.mat", "plus_5": "plus_5.mat"},
        central_pairs={"small": ["minus_2", "plus_2"], "large": ["minus_5", "plus_5"]},
        invariant_requirements=["same voltage channel order, current, frequency, belt and TTRKG contacts",
            "same five conductivities for every geometry", "volume_ml is actual heart-region volume",
            "one lung compartment", "heart sigma fixed; do not also fit delta_sigma_heart"],
        recommended_metadata=["geometry_law", "nominal_volume_or_linear_scale", "frequency_Hz", "current_A",
                              "mesh_hash", "electrode_hash", "material_map_hash", "solver_version"])
    write_json(out/"geometry_manifest.example.json", contract)
    next_sigma = sigma.copy()
    next_sigma[3] = .25
    write_json(out/"next_fem_recommendations.local.json", dict(
        status="proposed diagnostic settings, not measured tissue truth",
        material_order=NAMES, reference_sigma_S_m=next_sigma,
        reference_rho_ohm_m=1/next_sigma,
        reference_basis="v3 conditional soft/lung/bone/fat; author diagnostic heart rho=4 ohm m",
        new_baseline_must_be_evaluated=True,
        changing_heart_sigma_requires_new_J_and_K=True,
        all_materials_fixed_across_geometry_variants=True,
        geometry_volume_scales=[.95, .98, 1., 1.02, 1.05],
        scaling_basis="volume scales; isotropic linear factors are cube roots; use actual mesh volumes for differences",
        soft_lung_derivative_check_steps_S_m=[.00015, .0003],
        step_basis="numerical checks near observed conditional band RMS ~0.0003 S/m; not uncertainty bounds",
        primary_free_parameters=["delta_volume_ml", "delta_sigma_soft_S_m", "delta_sigma_lung_S_m"],
        nuisance_sensitivity="retain fat Jacobian; optional release of fat is diagnostic",
        no_simultaneous_free_heart_conductivity=True))
    report["implementation_sha256"] = sha(__file__)
    report["random_seed"], report["bootstrap_repetitions"] = seed, repetitions
    write_json(out/"audit.local.json", report)
    write_csv(out/"cross_validation.csv", cv)
    write_csv(out/"filter_edge_sensitivity.csv", report["filter_edge_sensitivity"])
    np.savez_compressed(out/"audit_arrays.npz", **arrays)
    make_plots(out, report, arrays)
    write_report(out, report)
    write_json(out/"completion.local.json", dict(status="completed", code_sha256=sha(__file__),
        artifacts={p.name: sha(p) for p in out.iterdir() if p.is_file()
                   and p.name != "completion.local.json" and p.suffix != ".log"}))
    print(json.dumps(jsonable(dict(output=str(out), normalized_condition=report["design_conductivity"]["normalized_condition"],
                                 full_plateaus=report["full_plateaus"], numerical_checks=report["numerical_checks"])), ensure_ascii=False))


def make_plots(out, report, arrays):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    colors = ["#1b7c83", "#c58322", "#a63555"]
    for row, record in enumerate(("record_02", "record_04")):
        t = arrays[record+"_time_s"]
        tt = arrays[record+"_projected_mohm"]
        for j, (label, color) in enumerate(zip(("Soft", "Lung", "Heart"), colors)):
            axes[row, 0].plot(t, tt[j], label=label, color=color)
        axes[row, 0].set(title=record+": conditional TTRKG projection", ylabel="mOhm", xlabel="Record time, s")
        axes[row, 0].legend()
        axes[row, 1].plot(t, 1000*arrays[record+"_coeff_S_m"][2], label="Joint / FWL", color=colors[2])
        axes[row, 1].plot(t, 1000*arrays[record+"_naive_heart_S_m"], label="Heart only", color="gray")
        axes[row, 1].set(title="Heart coefficient depends on nuisance removal", ylabel="mS/m", xlabel="Record time, s")
        axes[row, 1].legend()
    cos = report["design_conductivity"]["column_cosines"]
    axes[0, 2].imshow(cos, vmin=0, vmax=1, cmap="Blues")
    for i in range(3):
        for j in range(3):
            axes[0, 2].text(j, i, f"{cos[i,j]:.3f}", ha="center", va="center", color="white" if cos[i,j]>.7 else "black")
    axes[0, 2].set(xticks=range(3), yticks=range(3), xticklabels=["Soft", "Lung", "Heart"], yticklabels=["Soft", "Lung", "Heart"], title="Unit-column cosine: fixed v3 J")
    cv = report["cross_validation"]
    for i, method in enumerate(("relative_3", "discrepancy_3", "relative_soft_lung")):
        vals = [r["heldout_residual_pct"] for r in cv if r["method"] == method]
        axes[1, 2].scatter(np.full(len(vals), i), vals, alpha=.5)
    axes[1, 2].set(xticks=range(3), xticklabels=["3: relative", "3: trained weights", "Soft+lung"], ylabel="Held-out residual, %", title="Disjoint time blocks + held-out paired channels")
    for ax in axes.ravel():
        ax.grid(alpha=.15)
    fig.savefig(out/"pulse_diagnostics.png", dpi=170)
    plt.close(fig)


def write_report(out, r):
    d = r["design_relative_conductivity"]
    cv = r["cross_validation"]
    def fmt(x):
        return ", ".join(f"{v:.4g}" for v in x)
    parts = ["# Различимость пульсовых вкладов PulmoVista Ника\n",
        "## Граница выполненной проверки\n",
        "Исследованы реальные напряжения записей 02 и 04, кандидатные высокие плато 36–44 и 30–37 с. "
        "Сохранённый якобиан v3 относится к опоре 02. Он применён к обеим записям как фиксированный локальный "
        "линейный оператор. Для опоры 04 доступны итоговые параметры и скалярная диагностика нелинейного расчёта; "
        "её компактный якобиан не использован. MATLAB/FEM не запускался. Геометрия и сторонние ТТРКГ-записи не проверялись.\n",
        "Статус: условный математический результат. Полоса 0,7–3 Гц не устанавливает сердечное происхождение. "
        "Две записи одного человека не являются двумя независимыми испытуемыми. Июльские сессии ЭИТ и ТТРКГ "
        "не считаются синхронными; вычислены только проекции через сохранённый оператор ТТРКГ.\n",
        "## Математическая проверка\n",
        "Использована линейная задача $y=J\\,\\delta\\sigma+e$, $W=\\operatorname{diag}(1/|V_{pred}|)$. "
        "Оцениваются soft, единый lung и heart; bone и fat фиксированы. Для относительных изменений "
        "$A=WJ\\operatorname{diag}(\\sigma_0)$; дополнительно каждый столбец приведён к единичной евклидовой норме. "
        "Нормирование столбцов меняет координаты, но не прогноз.\n",
        f"Нормы столбцов относительного J: **{fmt(d['column_norms'])}**; сингулярные числа: "
        f"**{fmt(d['raw_weighted_singular_values'])}**. После нормирования столбцов: "
        f"**{fmt(d['normalized_singular_values'])}**, число обусловленности **{d['normalized_condition']:.4f}**. "
        f"Косинусы soft/lung, soft/heart, lung/heart: **{fmt(d['column_cosines'][[0,0,1],[1,2,2]])}**. "
        f"Доли столбцов, сохраняющиеся после исключения остальных: **{fmt(d['partial_unique_fractions'])}**. "
        "Полный ранг доказывает различимость только внутри данного фиксированного линейного оператора.\n",
        "Для выделения сердца применена частичная регрессия: $N=[WJ_s,WJ_l]$, "
        "$M_N=I-NN^+$, $h_\\perp=M_NWJ_h$, "
        "$\\widehat{\\delta\\sigma_h}=h_\\perp^TM_NWy/(h_\\perp^Th_\\perp)$. "
        "Это тот же коэффициент, что в совместной регрессии трёх тканей. Если удалить nuisance из y, "
        "но оставить исходный столбец сердца в знаменателе, коэффициент будет искусственно уменьшен: "
        f"множитель **{d['partial_unique_fractions'][2]**2:.4f}**.\n",
        "## Числа на полных плато\n",
        "Полные записи фильтровались тем же Butterworth SOS, что в исходном скрипте. Эти результаты "
        "предназначены для воспроизведения прежних чисел; строгая проверка временного переноса ниже использует "
        "отдельную фильтрацию каждого блока. Порядок троек: soft, lung, heart.\n"]
    for key, p in r["full_plateaus"].items():
        parts.append(f"**{key}, {p['interval_s'][0]:g}–{p['interval_s'][1]:g} с:** невязка "
            f"{p['relative_residual_pct']:.2f}% от нормы взвешенных наблюдений; RMS δσ "
            f"{fmt(p['delta_sigma_rms_S_m'])} См/м; прогноз RMS вкладов ТТРКГ "
            f"**{fmt(p['projected_TTRKG_rms_mohm'])} мОм**. RMS суммы soft+lung "
            f"{p['projected_soft_lung_rms_mohm']:.3f} мОм; RMS суммы всех трёх "
            f"{p['projected_TTRKG_total_rms_mohm']:.3f} мОм. RMS вкладов нельзя просто складывать. "
            f"Корреляция сердечного коэффициента до/после исключения soft+lung "
            f"{p['heart_naive_to_partial_correlation']:.4f}; RMS их различия составляет "
            f"{p['heart_naive_minus_partial_rms_over_partial_pct']:.1f}% RMS совместной оценки.\n")
        parts.append(f"Остаток структурирован: первая сингулярная компонента содержит "
            f"{100*p['residual_first_singular_energy_fraction']:.1f}% его энергии; лаговая корреляция "
            f"{p['residual_lag1_pooled']:.3f}. Медиана |межканальной корреляции| "
            f"{p['residual_abs_crosschannel_corr_quantiles'][1]:.3f}. Антисимметричная часть взаимных "
            f"измерений составляет {p['reciprocal_antisymmetric_voltage_energy_pct']:.4f}% энергии напряжений полосы. "
            "Это наблюдаемые характеристики ошибок и несоответствия модели, а не измеренная дисперсия чистого шума. "
            f"При оценке ошибки в исходных вольтах без нормирования по базовому напряжению невязка "
            f"составляет {p['unweighted_voltage_residual_pct']:.2f}%. Значение выше 100% означает, что "
            "в этой метрике прогноз хуже нулевого пульсового прогноза: минимум относительной ошибки "
            "не гарантирует малого абсолютного остатка. Невязка здесь относится к пульсовой полосе, "
            "а не к прежним 30–32% ошибки базовых напряжений.\n")
    parts += ["## Проверка по раздельным временным блокам\n",
        "Каждое плато разбито на две половины с промежутком 0,6 с. Каждый блок отдельно линейно детрендирован "
        "и отфильтрован; с краёв исключено по 0,6 с. Переноса отсчётов через фильтр между блоками нет. "
        "Остаётся лишь около 2–2,5 с на блок: проверка ограничена короткими фрагментами.\n",
        "В обучающем по времени блоке оценивается только фактический RMS остатка каждого канала. "
        "Для альтернативного WLS эти масштабы ограничены своими 10-м и 90-м процентилями и затем заморожены. "
        "Это калибровка весов физической регрессии, не обучение ИИ. Остаток включает модельную ошибку; "
        "полученные веса не объявляются оптимальными шумовыми весами.\n",
        "На каждом проверочном блоке восемь раз исключаются группы взаимных каналов. Коэффициенты "
        "оцениваются по оставшимся каналам и предсказывают исключённые. Все 104 пары одинаковых идеальных "
        "уравнений остаются внутри одного исключения. Проверяется перенос реконструкции на неизвестные "
        "каналы в другом временном блоке, а не прогноз будущей формы пульса.\n",
        "<details><summary>Ошибки прогноза исключённых каналов</summary>\n\n"
        "| Вариант | Минимум, % | Медиана, % | Максимум, % |\n|---|---:|---:|---:|\n"]
    for method in ("relative_3", "discrepancy_3", "relative_soft_lung"):
        vals = np.array([v["heldout_residual_pct"] for v in cv if v["method"] == method])
        parts.append(f"| {method} | {vals.min():.2f} | {np.median(vals):.2f} | {vals.max():.2f} |\n")
    parts.append("\n</details>\n")
    for method in ("relative_3", "discrepancy_3"):
        rows = [v for v in cv if v["method"] == method]
        vals = np.array([v["projected_rms_mohm"] for v in rows])
        changes = np.array([v["deletion_waveform_change_pct_max"] for v in rows])
        parts.append(f"**{method}:** диапазоны RMS проекции по блокам и выбору обучающего блока: "
            f"soft {vals[:,0].min():.3f}–{vals[:,0].max():.3f}, lung "
            f"{vals[:,1].min():.3f}–{vals[:,1].max():.3f}, heart {vals[:,2].min():.3f}–{vals[:,2].max():.3f} мОм. "
            f"Максимальное изменение формы коэффициента при удалении группы каналов, в % от RMS оценки "
            f"по всем каналам: **{fmt(changes.max(axis=0))}**.\n")
    pairs_cv = {}
    for row in cv:
        pairs_cv.setdefault((row["train"], row["test"]), {})[row["method"]] = row
    delta_loss, shift_soft, added_heart = [], [], []
    for methods in pairs_cv.values():
        old, new, reduced = (methods[k] for k in ("relative_3", "discrepancy_3", "relative_soft_lung"))
        delta_loss.append(old["heldout_residual_pct"]-new["heldout_residual_pct"])
        shift_soft.append(100*(new["projected_rms_mohm"][0]/old["projected_rms_mohm"][0]-1))
        added_heart.append(reduced["heldout_residual_pct"]-old["heldout_residual_pct"])
    parts.append(f"Добавление сердечного столбца улучшает ошибку исключённых каналов на "
        f"{min(added_heart):.2f}–{max(added_heart):.2f} процентного пункта. "
        f"Замена относительных весов на эмпирические улучшает её только на "
        f"{min(delta_loss):.2f}–{max(delta_loss):.2f} пункта, а RMS мягкотканной проекции "
        f"меняется на {min(shift_soft):.1f}…{max(shift_soft):.1f}%. "
        "Это свидетельствует о чувствительности коррекции к критерию согласования; автоматически "
        "принимать новые веса как более точную тканевую оценку нельзя.\n")
    for key, fat in r["free_fat_sensitivity"].items():
        original = r["full_plateaus"][key]["projected_TTRKG_rms_mohm"][0]
        parts.append(f"При освобождении единственной дополнительной δσ жира в {key} мягкотканный "
            f"RMS меняется с {original:.2f} до {fat['free_fat_projected_rms_mohm'][0]:.2f} мОм; "
            f"RMS изменения формы мягкотканного коэффициента — "
            f"{fat['free_fat_coefficient_change_pct'][0]:.1f}% исходного RMS. "
            f"Невязка при этом остаётся {fat['free_fat_relative_residual_pct']:.2f}%. "
            f"RMS суммы soft+fat равен {fat['free_fat_soft_plus_fat_rms_mohm']:.2f} мОм; "
            f"RMS изменения всей поправки soft+lung+fat относительно фиксированного жира — "
            f"{fat['free_fat_nuisance_sum_difference_mohm']:.2f} мОм. "
            "Суммарная поправка устойчивее отдельных тканевых коэффициентов, но её изменение "
            "по этому допущению превышает рассчитанный сердечный вклад.\n")
    for key, p in r["full_plateaus"].items():
        ci = p["bootstrap"]["20"]["projected_rms_mohm_q05_q50_q95"]
        parts.append(f"Для {key} пример диапазонов 5–95% условного RMS при блоке 1 с: "
            f"soft {ci[0,0]:.2f}–{ci[2,0]:.2f}, lung {ci[0,1]:.2f}–{ci[2,1]:.2f}, "
            f"heart {ci[0,2]:.3f}–{ci[2,2]:.3f} мОм. "
            "Это диапазоны ресэмплирования наблюдённой записи, не полная погрешность тканевой коррекции.\n")
    parts += ["## Неопределённость и применимость коррекции\n",
        "В audit.local.json сохранены интервалы 5–95% для RMS, полученные повторным выбором фактически "
        "наблюдавшихся многоканальных блоков длиной 0,25, 0,5 и 1 с. Каналы ресэмплируются совместно. "
        "Эти интервалы описывают вариацию условной амплитуды при данном J; они не включают систематическую "
        "ошибку геометрии, контактов, частоты, принятой проводимости сердца и переноса между сессиями. "
        "Также сохранены чувствительность к отсечению краёв 0,25/0,6/0,9 с, клиппированию весов "
        "5–95/10–90/20–80% и освобождению проводимости жира. Оценка всех пяти тканей — только "
        "диагностика пропущенных параметров, не доказательство пульсации кости.\n",
        "Увеличенная корреляция соседних отсчётов частично обусловлена самой полосовой фильтрацией. "
        "Поэтому ни отсчёты, ни четыре коротких блока не используются как независимые клинические наблюдения. "
        "Ширина bootstrap-интервала не покрывает возможное смещение тканевого параметра.\n",
        "Полный ранг позволяет вычислить отдельные soft/lung/heart коэффициенты, но большая структурированная "
        "невязка и чувствительность к выбору каналов/весов ограничивают их независимую физиологическую "
        "интерпретацию. Для мягкотканной коррекции реализованы совместная регрессия и частичная регрессия; "
        "передавать дальше следует весь подписанный вклад вместе с чувствительностью, а не объявлять его "
        "точным измерением мягкотканного кровенаполнения. Вычитание из реальной ТТРКГ здесь не выполнено.\n",
        "## Вход следующего физического расчёта\n",
        "Следующий оператор заменяет δσ сердца на δV сердца при фиксированном σ сердца = 0,25 См/м "
        "(ρ = 4 Ом·м — диагностический сценарий). На каждый вариант STL сохранить малый MAT: "
        "S_eit (208×5, В/(См/м)), V_eit (208×1, В), Ktt (1×5, Ом/(См/м)), Ztt (1×1, Ом), "
        "volume_ml (1×1, мл), sigma_S_m (5×1). Порядок тканей: soft_background, lung, bone, heart, fat. "
        "Пути связать через geometry_manifest.example.json. Контакты, ток, частота и все σ должны "
        "совпадать между вариантами; лёгкие остаются одним компартментом.\n",
        "Для пар ±2% и ±5% используются фактические объёмы: "
        "$J_V=(V_+-V_-)/(\\mathcal V_+-\\mathcal V_-)$ и "
        "$K_V=(Z_+-Z_-)/(\\mathcal V_+-\\mathcal V_-)$; отдельно проверяются односторонние производные. "
        "Дизайн $[J_V,J_s,J_l]$ оценивает [δV, δσ_s, δσ_l] с единицами [мл, См/м, См/м]. "
        "Не вводить одновременно независимую δσ сердца: это другая постановка с дополнительным "
        "неизвестным. Сравнить малый и большой шаг по изменению производной, уникальной части J_V и "
        "ошибке исключённых каналов. Разброс по шагам включать в чувствительность итоговой δV.\n",
        "Для следующего FEM сохранить также столбец жира, даже если он фиксирован в основной инверсии: "
        "уже имеющийся расчёт показывает существенную чувствительность soft к этому ограничению. "
        "Основной совместный вариант — δV+δσ_soft+δσ_lung; освобождение δσ_fat оставить отдельной "
        "диагностикой. Предусмотреть экспорт вольтовых и относительных остатков, чтобы изменение весов "
        "не скрывало большой ошибки напряжений. Прежние 21,89 и 40,10 Ом·м не использовать как "
        "валидированные свойства сердца; 4 Ом·м также не объявлять измерением.\n",
        "В next_fem_recommendations.local.json записан конкретный начальный диагностический сценарий "
        "σ = [0,3503892; 0,05414499; 0,05222558; 0,25; 0,04335909] См/м в указанном порядке тканей. "
        "После замены проводимости сердца обязательно заново оценить базовые напряжения и J/K: "
        "согласие старого расчёта на новую опору не переносится. Во всех геометрических вариантах "
        "σ одинаковы. Для проверки производных soft/lung предложены ±0,00015 и ±0,0003 См/м: "
        "это вычислительные шаги порядка условного RMS записанной полосы, не границы физиологической нормы "
        "или неопределённости. Масштабы 0,95/0,98/1/1,02/1,05 относятся к объёму; для изотропного "
        "масштабирования координат нужны их кубические корни.\n",
        "Функция general_design(A, observations, weights) принимает наблюдения 208×N; weights — положительные "
        "множители строк в норме остатка, а не дисперсии. load_volume_design(manifest_path) собирает "
        "геометрическую колонку без FEM. CLI --geometry-manifest подключает её к той же проверке по блокам. "
        "Изменение эквивалентного целого сердца не равно УО ЛЖ; клинические УО/ФВ здесь не вычисляются.\n",
        "## Артефакты\n",
        "- [Машинный отчёт и хэши источников](audit.local.json).\n"
        "- [Временная и канальная проверка](cross_validation.csv).\n"
        "- [Чувствительность к фильтрации](filter_edge_sensitivity.csv).\n"
        "- [Коэффициенты, напряжения полосы и прогнозы](audit_arrays.npz).\n"
        "- [Контракт геометрического входа](geometry_manifest.example.json).\n"
        "- [Начальные параметры следующего FEM](next_fem_recommendations.local.json).\n"
        "- [Графики](pulse_diagnostics.png).\n"]
    text = "\n".join(parts).replace("|\n\n|", "|\n|")
    (out/"REPORT.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--geometry-manifest", type=Path)
    parser.add_argument("--geometry-only", action="store_true", help="Run saved geometry pass using unchanged prior pulse windows")
    parser.add_argument("--compare-geometry", type=Path, help="Prior geometry_audit.local.json for q-dependence check")
    parser.add_argument("--effective-fit-dir", type=Path, help="Three completed per-shape STL.mat, G1.mat, L1.mat outputs")
    args = parser.parse_args()
    if args.effective_fit_dir:
        if args.geometry_only or args.geometry_manifest or args.compare_geometry:
            parser.error("--effective-fit-dir is a separate bounded pass")
        effective_fit_pass(args.data_root, args.output, args.effective_fit_dir)
    elif args.geometry_only:
        if args.geometry_manifest is None:
            parser.error("--geometry-only requires --geometry-manifest")
        geometry_pass(args.data_root, args.output, args.geometry_manifest, args.bootstrap_repetitions, args.seed, args.compare_geometry)
    else:
        main(args.data_root, args.output, args.bootstrap_repetitions, args.seed, args.geometry_manifest)
