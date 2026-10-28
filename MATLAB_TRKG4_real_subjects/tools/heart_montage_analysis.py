"""Pure algebra for saved Nik heart/montage probes; NumPy only, no FEM or I/O.

Transport contract (one shape/material/background/contact context per export)::

    {"schema": "heart_montage_analysis_v1", "subject": "Nik",
     "size_coordinate": "V", "heart_order": ["V","x","y","z","rho_heart"],
     "tissue_order": ["rho_soft","rho_lung"],
     "amplitudes": [A_V,A_x,A_y,A_z,A_H,A_S,A_L],
     "amplitude_basis": "explicit source/assumption for these positive scales",
     "montage_ids": ["TT_A","TT_B"], "stack_kind": "alternative_montages",
     "context": {"shape": "STL", "scenario": "material scenario",
       "frequency_Hz": f, "coordinate_frame": "LPS_mm",
       "background_id": "...", "contact_model_id": "...",
       "breath_interval_status": "expert reconstruction/..."},
     "records": [{"id": "base", "heart": [V,x,y,z,rho_H],
       "tissue": [rho_S,rho_L], "volume_ml": V,
       "impedance_ohm": [Z_A,Z_B], "roi_ok": null, "source_id": "hash/ref"}],
     "difference_pairs": [{"parameter": "x", "minus": "xm_h", "plus": "xp_h"}]}

Numbers above are placeholders, not example measurements. For requiv replace V
in size_coordinate/heart_order by requiv (mm), calculated from integrated ml.
All other positions are mm, rho is ohm*m, impedance is real signed ohm. V is
integrated volume; nominal_volume_ml may be retained as extra record metadata.
Each record contains ALL five heart and both tissue parameters. One lung only.
IDs must retain provenance; this module checks declarations, not source hashes,
mesh/contact compatibility, spatial convergence or physical ROI containment.

Typical use: validate_export -> finite_difference for each declared pair ->
step_convergence (raw and explicitly volume-corrected separately) -> assemble
J columns in heart_order + tissue_order -> montage_diagnostics. Use the other
functions on joint/compensation/holdout records chosen by the caller. There is
no implicit selection of a derivative step, law, tissue correction or montage.

Amplitudes scale coordinates; row_weights multiply rows of the residual. Neither
is a noise model. Machine SVD thresholds are numerical, not detection limits.
Minimum-norm fits are representatives, not identified parameters. Restriction
and compensation algebra is separate from finite_probe_check. Old 4D controls
do not validate a new one-CT law; whole-heart volume is not ventricular SV/EF.
Run ``python -B heart_montage_analysis.py --self-test`` for synthetic math only.
"""
from __future__ import annotations

import json
import sys

import numpy as np


HEART = ("V", "x", "y", "z", "rho_heart")
TISSUE = ("rho_soft", "rho_lung")
STACK_KINDS = ("alternative_montages", "simultaneous_recording")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value, ndim, name):
    raw = np.asarray(value)
    _require(not np.iscomplexobj(raw), f"{name}: real signed values required")
    a = np.asarray(value, dtype=float)
    _require(a.ndim == ndim and np.all(np.isfinite(a)), f"{name}: finite {ndim}D array required")
    return a


def _vector(value, length, name):
    a = _array(value, 1, name)
    _require(a.size == length, f"{name}: expected {length} entries")
    return a


def _ratio(a, b):
    return float(a / b) if b != 0 else None


def _svd_factors(a, atol=0.0):
    """Shared rank rule and orthonormal bases, including the full nullspace."""
    u, s, vh = np.linalg.svd(a, full_matrices=True)
    tol = max(float(atol), max(a.shape, default=0) * np.finfo(float).eps * (s[0] if s.size else 0))
    rank = int(np.count_nonzero(s > tol))
    return u, s, vh, rank, float(tol)


def _svd(a, atol=0.0):
    """Full V retains null directions when rows < columns."""
    u, s, vh, rank, tol = _svd_factors(a, atol)
    inv = (vh[:rank].T / s[:rank]) @ u[:, :rank].T
    return inv, s, rank, tol, vh[rank:].T


def _orthogonal_residual(a, values, atol=0.0):
    """Project with U_r U_r.T, avoiding the unstable product A @ pinv(A)."""
    u, _, _, rank, tol = _svd_factors(a, atol)
    basis = u[:, :rank]
    return values - basis @ (basis.T @ values), rank, tol


def _design(jacobian, amplitudes, row_weights):
    j = _array(jacobian, 2, "jacobian")
    _require(min(j.shape) > 0, "nonempty jacobian required")
    scales = _vector(amplitudes, j.shape[1], "amplitudes")
    weights = np.ones(j.shape[0]) if row_weights is None else _vector(row_weights, j.shape[0], "row_weights")
    _require(np.all(scales > 0) and np.all(weights > 0), "scales and row multipliers must be positive")
    return j, scales, weights, weights[:, None] * j * scales


def size_to_volume(size, size_coordinate):
    """Exact coordinate change only; does not turn the FEM heart into a sphere."""
    _require(size_coordinate in ("V", "requiv"), "size_coordinate must be V or requiv")
    _require(size is not None and np.isscalar(size), "baseline size is required")
    _require(np.isfinite(size) and size > 0, "positive size required")
    return float(size if size_coordinate == "V" else 4 * np.pi * size**3 / 3000)


def volume_gradient(size, size_coordinate):
    """dV/dsize, ml/ml or ml/mm, evaluated at the supplied baseline."""
    volume = size_to_volume(size, size_coordinate)
    return 1.0 if size_coordinate == "V" else 3 * volume / size


def _record(record, size_coordinate):
    heart = _vector(record["heart"], 5, "heart")
    tissue = _vector(record["tissue"], 2, "tissue")
    z = _array(record["impedance_ohm"], 1, "impedance_ohm")
    _require(z.size > 0 and heart[4] > 0 and np.all(tissue > 0), "positive rho and nonempty montage data required")
    volume = float(record["volume_ml"])
    _require(np.isfinite(volume) and volume > 0, "positive integrated volume required")
    _require(np.isclose(size_to_volume(heart[0], size_coordinate), volume, rtol=1e-10, atol=1e-10),
             "size coordinate must agree with integrated volume (not nominal volume)")
    _require(record["roi_ok"] is None or type(record["roi_ok"]) is bool, "roi_ok must be true, false or null")
    _require(isinstance(record["id"], str) and bool(record["id"]), "record id required")
    _require(isinstance(record["source_id"], str) and bool(record["source_id"]), "saved source id required")
    return np.r_[heart, tissue], z, volume


def _roi(records):
    flags = [r["roi_ok"] for r in records]
    return "failed" if False in flags else "unknown" if None in flags else "declared_passed_not_rechecked"


def validate_export(payload):
    """Validate the small transport contract without reading/changing any files."""
    _require(payload["schema"] == "heart_montage_analysis_v1", "unsupported schema")
    _require(payload["subject"] in ("Nik", "Nix", "Ник"), "this study is for Nik")
    size = payload["size_coordinate"]
    _require(size in ("V", "requiv"), "unsupported size coordinate")
    order = (size,) + HEART[1:] + TISSUE
    _require(tuple(payload["heart_order"]) + tuple(payload["tissue_order"]) == order,
             "five free heart parameters, followed by soft and ONE lung required")
    amplitudes = _vector(payload["amplitudes"], 7, "amplitudes")
    _require(np.all(amplitudes > 0) and bool(payload["amplitude_basis"]), "explicit positive amplitude scales and basis required")
    ids = payload["montage_ids"]
    _require(len(ids) > 0 and all(isinstance(i, str) and i for i in ids) and len(set(ids)) == len(ids), "unique montage ids required")
    _stack_status(payload["stack_kind"], payload.get("simultaneity_evidence"))
    context = payload["context"]
    for key in ("shape", "scenario", "coordinate_frame", "background_id", "contact_model_id", "breath_interval_status"):
        _require(isinstance(context[key], str) and bool(context[key]), f"context.{key} required")
    frequency = float(context["frequency_Hz"])
    _require(np.isfinite(frequency) and frequency > 0, "positive frequency required")
    records = payload["records"]
    _require(len(records) > 0, "records are empty")
    by_id = {}
    for record in records:
        _, z, _ = _record(record, size)
        _require(z.size == len(ids), "every record must use the same montage order")
        _require(record["id"] not in by_id, "duplicate record id")
        by_id[record["id"]] = record
    for pair in payload["difference_pairs"]:
        _require(pair["parameter"] in order, "unknown difference parameter")
        _require(pair["minus"] in by_id and pair["plus"] in by_id and pair["minus"] != pair["plus"], "missing/distinct pair records required")
    return dict(parameter_order=list(order), record_count=len(records), montage_count=len(ids),
                roi_status=_roi(records), source_hashes_checked=False,
                status="structure_checked_not_physical_validation")


def finite_difference(base, minus, plus, parameter, *, size_coordinate="V", kv=None):
    """Unequal-step three-point derivative at base, plus secant/one-sided slopes.

    Other parameters must be exactly fixed, except integrated size drift on a
    non-size probe. That drift is retained raw; kv (ohm/ml per montage) optionally
    removes it to FIRST order. Without kv the corrected derivative is unavailable.
    """
    order = (size_coordinate,) + HEART[1:] + TISSUE
    _require(parameter in order, "unknown parameter")
    k = order.index(parameter)
    p0, z0, v0 = _record(base, size_coordinate)
    pm, zm, vm = _record(minus, size_coordinate)
    pp, zp, vp = _record(plus, size_coordinate)
    _require(z0.shape == zm.shape == zp.shape, "montage count mismatch")
    fixed = [i for i in range(7) if i not in (0, k)]
    _require(np.array_equal(pm[fixed], p0[fixed]) and np.array_equal(pp[fixed], p0[fixed]), "mixed probe: an unrequested non-size parameter changed")
    hm, hp = p0[k] - pm[k], pp[k] - p0[k]
    _require(hm > 0 and hp > 0, "minus < base < plus required")

    def slopes(ym, yp):
        left, right = -ym / hm, yp / hp
        return dict(central=(hp * left + hm * right) / (hm + hp),
                    secant=(yp - ym) / (hm + hp), minus=left, plus=right,
                    one_sided_difference=right - left)

    raw = slopes(zm - z0, zp - z0)
    corrected = None
    if kv is not None:
        _require(k != 0, "do not subtract the intended size response")
        kv = _vector(kv, z0.size, "kv")
        corrected = slopes(zm - z0 - kv * (vm - v0), zp - z0 - kv * (vp - v0))
    return dict(parameter=parameter, size_coordinate=size_coordinate, base_id=base["id"],
                record_ids=[minus["id"], plus["id"]], steps=[float(hm), float(hp)],
                integrated_delta_volume_ml=[vm - v0, vp - v0], raw=raw, volume_corrected=corrected,
                volume_drift_correction_applied=corrected is not None,
                volume_drift_KV_ohm_per_ml=kv,
                nominal_delta_volume_ml=[None if "nominal_volume_ml" not in r or "nominal_volume_ml" not in base
                                         else r["nominal_volume_ml"] - base["nominal_volume_ml"] for r in (minus, plus)],
                roi_status=_roi([base, minus, plus]),
                scope="finite_step_and_discretization_combined_not_spatial_convergence")


def step_convergence(estimates, *, component="raw", absolute_tolerance=None, relative_tolerance=None):
    """Compare decreasing steps, without inferring a FEM limit or a noise floor.

    Optional tolerances are in derivative units and relative Euclidean norm.
    An explicit absolute tolerance can assess a zero limiting derivative; a
    relative-only criterion cannot. Two levels give one comparison, not proof.
    """
    _require(component in ("raw", "volume_corrected") and len(estimates) >= 2, "need at least two step levels")
    reference = estimates[0]
    for e in estimates:
        _require(all(e[k] == reference[k] for k in ("parameter", "size_coordinate", "base_id")), "mixed difference families")
        _require(e[component] is not None, "requested volume correction was not supplied")
    levels = sorted(estimates, key=lambda e: max(e["steps"]), reverse=True)
    for tol in (absolute_tolerance, relative_tolerance):
        _require(tol is None or (np.isfinite(tol) and tol >= 0), "invalid explicit tolerance")
    comparisons = []
    for coarse, fine in zip(levels, levels[1:]):
        _require(np.all(np.asarray(fine["steps"]) < coarse["steps"]), "both one-sided steps must decrease strictly")
        a, b = np.asarray(coarse[component]["central"]), np.asarray(fine[component]["central"])
        _require(a.shape == b.shape, "montage count mismatch")
        delta = b - a
        rel = _ratio(np.linalg.norm(delta), np.linalg.norm(b))
        decisions = []
        if absolute_tolerance is not None:
            decisions.append(bool(np.max(np.abs(delta)) <= absolute_tolerance))
        if relative_tolerance is not None:
            decisions.append(None if rel is None else rel <= relative_tolerance)
        assessed = bool(decisions) and all(d is not None for d in decisions)
        comparisons.append(dict(coarse_steps=coarse["steps"], fine_steps=fine["steps"],
            signed_change=delta, absolute_change=np.abs(delta), relative_norm_change=rel,
            within_supplied_tolerances=all(decisions) if assessed else None))
    return dict(component=component, comparisons=comparisons,
                absolute_tolerance=absolute_tolerance, relative_tolerance=relative_tolerance,
                roi_status="failed" if any(e["roi_status"] == "failed" for e in levels) else
                           "unknown" if any(e["roi_status"] == "unknown" for e in levels) else "declared_passed_not_rechecked",
                status="successive_differences_only_no_automatic_convergence_claim")


def linear_diagnostics(jacobian, amplitudes, *, row_weights=None):
    """SVD of W J diag(amplitudes), and a separate column-normalized geometry.

    Unique directions and identification use the scaled matrix's common rank
    threshold. Identification requires BOTH a rank increase on adding the
    column and a negligible target component in the numerical nullspace.
    """
    j, scales, weights, a = _design(jacobian, amplitudes, row_weights)
    inv, s, rank, tol, null = _svd(a)
    norms = np.linalg.norm(a, axis=0)
    normalized = np.divide(a, norms, out=np.zeros_like(a), where=norms > 0)
    _, ns, nrank, ntol, _ = _svd(normalized)
    unique = []
    unique_vectors = []
    rank_increase = []
    for k in range(j.shape[1]):
        others = np.delete(a, k, axis=1)
        residual, other_rank, _ = _orthogonal_residual(others, a[:, k], tol)
        unique_vectors.append(residual)
        unique.append(_ratio(np.linalg.norm(residual), norms[k]))
        rank_increase.append(rank - other_rank)
    null_tolerance = 100 * max(a.shape) * np.finfo(float).eps
    identified = (np.asarray(rank_increase) == 1) & (np.linalg.norm(null, axis=1) <= null_tolerance)
    return dict(rank=rank, nullity=j.shape[1] - rank, rank_tolerance=tol,
        singular_values=np.pad(s, (0, j.shape[1] - s.size)),
        normalized_rank=nrank, normalized_rank_tolerance=ntol,
        normalized_singular_values=np.pad(ns, (0, j.shape[1] - ns.size)),
        condition=float(s[0] / s[-1]) if rank == j.shape[1] else None,
        condition_status="finite" if rank == j.shape[1] else "unbounded_unidentified_directions",
        unique_fraction=unique, unique_weighted_response=np.column_stack(unique_vectors),
        amplitude_responses_ohm=j * scales, amplitudes=scales, row_weights=weights,
        null_basis_scaled_coordinates=null, null_basis_physical=scales[:, None] * null,
        individually_identified=identified, column_rank_increase=rank_increase,
        coordinate_resolution=np.eye(j.shape[1]) - null @ null.T,
        representative_inverse_row_norms=np.linalg.norm(scales[:, None] * inv, axis=1),
        measured_noise=None, practical_identifiability="not_assessed",
        scope="local_linear_algebra_not_parameter_or_clinical_validation")


def volume_identified_given_nuisance(heart_jacobian, amplitudes, *, row_weights=None,
                                    size_coordinate="V", base_size=None):
    """Is volume identified with x,y,z,rho_H ALL free? Accept m x 5, e.g. 3 x 5.

    A = W J diag(amplitudes); f = (dV/dsize)*amplitudes[0]*e0. Volume is
    identified iff f belongs to row(A), equivalently N.T @ f = 0 for null(A).
    eta_V projects A[:,0] off A[:,1:5]. These are LOCAL algebraic tests only.
    Numerically, require rank(A) = rank(A[:,1:5]) + 1 at the SAME threshold
    as well as a negligible row-space residual; a small residual alone does
    not identify V when a weak but retained nuisance can fully reproduce it.
    An identified volume functional need not require rank(A)=5. Its estimator
    is returned only when identified; otherwise volume is unbounded along a
    null direction in the unconstrained linear problem. Amplitudes are scales,
    not hard bounds. Soft/lung correction uncertainty is handled separately
    by volume_omission, never silently fixed to zero in a total error claim.
    """
    j, scales, weights, a = _design(heart_jacobian, amplitudes, row_weights)
    _require(j.shape[1] == 5, "exactly V(or requiv),x,y,z,rho_heart required")
    _require(size_coordinate in ("V", "requiv"), "invalid size coordinate")
    dvolume = 1.0 if size_coordinate == "V" else volume_gradient(base_size, size_coordinate)
    inverse, singular, rank, tol, null = _svd(a)
    unique, nuisance_rank, _ = _orthogonal_residual(a[:, 1:], a[:, 0], tol)
    target = np.array([dvolume * scales[0], 0., 0., 0., 0.])
    # Orthogonal row-space projection from V, avoiding inverse amplification.
    rowspace_residual = null @ (null.T @ target)
    relative_residual = float(np.linalg.norm(rowspace_residual) / np.linalg.norm(target))
    numeric_tol = 100 * max(a.shape) * np.finfo(float).eps
    rank_increase = rank - nuisance_rank
    identified = bool(rank_increase == 1 and relative_residual <= numeric_tol)
    estimator = target @ inverse if identified else None
    return dict(volume_identified=identified, rank=rank, heart_unknowns=5, nuisance_rank=nuisance_rank,
        volume_rank_increase=rank_increase,
        eta_V=_ratio(np.linalg.norm(unique), np.linalg.norm(a[:, 0])),
        unique_weighted_volume_amplitude_response=unique,
        target_functional_scaled=target, rowspace_residual_scaled=rowspace_residual,
        target_rowspace_relative_residual=relative_residual, target_rowspace_numerical_tolerance=numeric_tol,
        null_volume_response_ml=target @ null, null_basis_physical=scales[:, None] * null,
        volume_estimator_weights_per_ohm=None if estimator is None else estimator * weights,
        volume_amplification_per_unit_weighted_data_norm=None if estimator is None else float(np.linalg.norm(estimator)),
        singular_values=np.pad(singular, (0, 5 - singular.size)), rank_tolerance=tol,
        status="identified_local_linear_functional" if identified else "unidentified_volume_with_free_XYZ_rhoH",
        tissue_scope="after_supplied_soft_lung_correction_uncertainty_not_evaluated_here",
        practical_identifiability="not_assessed_no_noise_or_error_floor")


def derivative_check(analytic_derivative, difference_estimate, *, component="raw"):
    """Compare a supplied analytic J_rho column with saved finite rho probes.

    The caller supplies the analytic column from its solver export and selects
    the +/-1% or +/-5% pair. No new material solves or inferred rho calibration.
    """
    _require(component in ("raw", "volume_corrected") and difference_estimate[component] is not None,
             "requested difference component unavailable")
    d = difference_estimate[component]
    analytic = _vector(analytic_derivative, len(d["central"]), "analytic_derivative")
    error = np.asarray(d["central"]) - analytic
    return dict(parameter=difference_estimate["parameter"], steps=difference_estimate["steps"],
        component=component, analytic=analytic, finite_central=d["central"], signed_error=error,
        absolute_error=np.abs(error), relative_norm_error=_ratio(np.linalg.norm(error), np.linalg.norm(analytic)),
        one_sided_difference=d["one_sided_difference"], roi_status=difference_estimate["roi_status"],
        scope="analytic_vs_saved_finite_probe_not_physical_validation")


def _stack_status(kind, evidence):
    _require(kind in STACK_KINDS, "explicit stack_kind required")
    if kind == "simultaneous_recording":
        _require(isinstance(evidence, str) and bool(evidence.strip()), "simultaneity needs a recording/protocol evidence reference")
    return "hypothetical_design_not_simultaneous_data" if kind == "alternative_montages" else "caller_declared_simultaneity_not_verified"


def montage_diagnostics(heart_jacobian, tissue_jacobian, amplitudes, *, montage_ids,
                        stack_kind, row_weights=None, simultaneity_evidence=None,
                        size_coordinate="V", base_size=None):
    """Five free heart columns in both cases; soft/lung are separate nuisances.

    Corrected case is CONDITIONAL on exact independently supplied corrections.
    Free case profiles soft/lung out in the chosen metric. Projected singular
    values use the ORIGINAL joint matrix tolerance, avoiding false rank from
    roundoff left by an exactly absorbed heart direction.
    """
    h, t = _array(heart_jacobian, 2, "heart_jacobian"), _array(tissue_jacobian, 2, "tissue_jacobian")
    _require(h.shape[1] == 5 and t.shape == (h.shape[0], 2), "require heart m x 5 and soft/lung m x 2")
    j, scales, weights, a = _design(np.column_stack([h, t]), amplitudes, row_weights)
    _require(len(montage_ids) == h.shape[0] and len(set(montage_ids)) == len(montage_ids), "montage ids mismatch/duplicates")
    status = _stack_status(stack_kind, simultaneity_evidence)

    def analyze_rows(indices):
        jj, ww, aa = j[indices], weights[indices], a[indices]
        full = linear_diagnostics(jj, scales, row_weights=ww)
        effective, tissue_rank, _ = _orthogonal_residual(aa[:, 5:], aa[:, :5], full["rank_tolerance"])
        _, singular, rank, tol, _ = _svd(effective, full["rank_tolerance"])
        return dict(heart_if_exact_tissue_corrections=linear_diagnostics(jj[:, :5], scales[:5], row_weights=ww),
                    volume_identified_given_nuisance=volume_identified_given_nuisance(jj[:, :5], scales[:5], row_weights=ww,
                        size_coordinate=size_coordinate, base_size=base_size),
                    heart_plus_free_tissue=full, tissue_rank=tissue_rank,
                    heart_after_free_tissue_projection=dict(rank=rank, rank_tolerance=tol,
                        singular_values=np.pad(singular, (0, 5 - singular.size)),
                        weighted_amplitude_matrix=effective,
                        unique_fraction_against_all_other_heart_and_tissue=full["unique_fraction"][:5]),
                    correction_assumption="conditional_exact_soft_lung_corrections_not_asserted_available")

    return dict(stack_kind=stack_kind, stack_status=status, montage_ids=list(montage_ids),
                per_montage={name: analyze_rows([i]) for i, name in enumerate(montage_ids)},
                stack=analyze_rows(list(range(h.shape[0]))), measured_noise=None)


def joint_decomposition(base, singles, joint, jacobian, *, size_coordinate="V"):
    """Exact bookkeeping of saved responses, not a new forward calculation.

    joint-linear = (joint-sum(single)) + sum(single-linear) - J*step_mismatch.
    step_mismatch = joint_delta_p - sum(single_delta_p). In particular numerical
    volume mismatch is NOT silently interpreted as a nonlinear interaction.
    """
    _require(len(singles) > 0, "single probes required")
    p0, z0, _ = _record(base, size_coordinate)
    pj, zj, _ = _record(joint, size_coordinate)
    j = _array(jacobian, 2, "jacobian")
    _require(j.shape == (z0.size, 7), "jacobian order is five heart then soft/lung")
    parsed = [_record(r, size_coordinate) for r in singles]
    _require(all(z.shape == z0.shape for _, z, _ in parsed) and zj.shape == z0.shape, "montage count mismatch")
    dp = np.stack([p - p0 for p, _, _ in parsed])
    dy = np.stack([z - z0 for _, z, _ in parsed])
    mismatch = pj - p0 - dp.sum(axis=0)
    interaction = zj - z0 - dy.sum(axis=0)
    single_remainders = dy - dp @ j.T
    adjusted = interaction - j @ mismatch
    return dict(joint_delta_parameters=pj - p0, single_delta_parameters=dp,
        parameter_step_mismatch=mismatch, mismatch_linear_response_ohm=j @ mismatch,
        joint_delta_ohm=zj - z0, sum_single_delta_ohm=dy.sum(axis=0),
        joint_minus_sum_ohm=interaction, single_minus_linear_ohm=single_remainders,
        interaction_after_linear_step_alignment_ohm=adjusted,
        joint_minus_linear_ohm=zj - z0 - j @ (pj - p0),
        decomposition_closure_ohm=(zj - z0 - j @ (pj - p0)) - adjusted - single_remainders.sum(axis=0),
        roi_status=_roi([base, *singles, joint]),
        scope="finite_interaction_plus_discretization_step_alignment_is_only_linear")


def linear_compensation(jacobian, imposed_delta, compensator_indices, amplitudes, *, row_weights=None):
    """Cancel J*imposed_delta using explicit free compensators, with no clipping.

    All other entries of imposed_delta explicitly state the imposed changes.
    Returned physical null directions generate ALL equal-residual linear fits.
    The minimum scaled-norm representative is not a bounded-feasibility solver:
    its exceeding an amplitude does not prove all alternatives exceed it.
    A finite compensated FEM probe must be assessed with finite_probe_check.
    """
    j, scales, weights, a = _design(jacobian, amplitudes, row_weights)
    imposed = _vector(imposed_delta, j.shape[1], "imposed_delta")
    indices = list(compensator_indices)
    _require(indices and len(set(indices)) == len(indices) and
             all(type(k) is int and 0 <= k < j.shape[1] for k in indices), "distinct valid compensator indices required")
    _require(np.all(imposed[indices] == 0), "imposed compensator entries must explicitly be zero")
    b = a[:, indices]
    inv, singular, rank, tol, null = _svd(b)
    compensation = np.zeros(j.shape[1])
    compensation[indices] = scales[indices] * (inv @ (-weights * (j @ imposed)))
    total = imposed + compensation
    physical_null = np.zeros((j.shape[1], null.shape[1]))
    physical_null[indices] = scales[indices, None] * null
    residual = j @ total
    numerical_tol = 100 * max(a.shape) * np.finfo(float).eps * max(
        np.linalg.norm(weights * (j @ imposed)), np.linalg.norm(weights * (j @ compensation)))
    return dict(imposed_delta=imposed, compensation_delta=compensation, total_delta=total,
        residual_ohm=residual, weighted_residual_norm=float(np.linalg.norm(weights * residual)),
        exactly_compensable_at_numerical_tolerance=bool(np.linalg.norm(weights * residual) <= numerical_tol),
        residual_numerical_tolerance=float(numerical_tol), compensator_rank=rank,
        singular_values=singular, rank_tolerance=tol, compensation_null_basis_physical=physical_null,
        compensation_unique=rank == len(indices), amplitude_ratios=np.abs(total) / scales,
        representative_exceeds_amplitudes=np.abs(total) > scales,
        all_solutions="total_delta + compensation_null_basis_physical @ arbitrary_coefficients",
        status="linear_compensability_only_finite_control_required")


def _optional_vector(values, length, name, *, nonnegative=False):
    _require(len(values) == length, f"{name}: expected {length} explicit entries (null for unknown)")
    out = []
    for value in values:
        if value is None:
            out.append(None)
        else:
            number = float(value)
            _require(np.isfinite(number) and (not nonnegative or number >= 0), f"invalid {name}")
            out.append(number)
    return out


def volume_omission(heart_jacobian, tissue_jacobian, omitted_heart_delta, tissue_correction_error,
                    *, base_size, size_coordinate="V", residual_ohm=None,
                    absolute_bounds=None, residual_bound_ohm=None):
    """Per-montage conditional volume error: omitted response / K_V.

    omitted_heart_delta has x,y,z,rho_H entries; tissue error is TRUE minus
    supplied correction for soft/lung. Null means unknown, never zero. Bounds
    have the same six-column order; they are explicit box assumptions, NOT SDs.
    residual_ohm includes unaccounted measurement/model/nonlinear error. Missing
    residual means no complete signed error or total bound can be claimed.
    """
    h, t = _array(heart_jacobian, 2, "heart_jacobian"), _array(tissue_jacobian, 2, "tissue_jacobian")
    _require(h.shape[1] == 5 and t.shape == (h.shape[0], 2), "need five heart and two tissue columns")
    m = h.shape[0]
    delta = (_optional_vector(omitted_heart_delta, 4, "omitted_heart_delta") +
             _optional_vector(tissue_correction_error, 2, "tissue_correction_error"))
    bounds = [None] * 6 if absolute_bounds is None else _optional_vector(absolute_bounds, 6, "absolute_bounds", nonnegative=True)
    residual = [None] * m if residual_ohm is None else _optional_vector(residual_ohm, m, "residual_ohm")
    rb = [None] * m if residual_bound_ohm is None else _optional_vector(residual_bound_ohm, m, "residual_bound_ohm", nonnegative=True)
    kv = h[:, 0] / volume_gradient(base_size, size_coordinate)
    derivatives = np.column_stack([h[:, 1:], t])
    rows = []
    for i, k in enumerate(kv):
        terms = [None if d is None else float(c * d) for c, d in zip(derivatives[i], delta)]
        complete = all(v is not None for v in terms) and residual[i] is not None
        bound_complete = all(v is not None for v in bounds) and rb[i] is not None
        known = sum(v for v in terms if v is not None)
        total = known + residual[i] if complete else None
        bound = float(np.abs(derivatives[i]) @ bounds + rb[i]) if bound_complete else None
        rows.append(dict(montage_index=i, K_V_ohm_per_ml=float(k), terms_ohm=terms,
            terms_equivalent_volume_ml=[None if v is None else _ratio(v, k) for v in terms],
            known_partial_sum_ohm=known, residual_ohm=residual[i], complete_error=complete,
            volume_error_ml=None if total is None else _ratio(total, k),
            conditional_box_bound_ml=None if bound is None else _ratio(bound, abs(k)),
            linear_box_bound_complete=bound_complete, absolute_bounds=bounds,
            ml_per_ohm_amplification=_ratio(1, abs(k)),
            status="undefined_zero_KV" if k == 0 else "conditional_linear_only_inspect_small_KV"))
    return dict(parameter_order=list(HEART[1:] + TISSUE), rows=rows,
                uncertainty_model="explicit_signed_errors_or_box_bounds_no_independence_or_noise_assumed")


def finite_probe_check(base, trial, jacobian, *, size_coordinate="V", tissue_correction_ohm=None):
    """Saved finite response vs linear prediction; no synthetic J*dp substitute.

    Optional explicitly supplied tissue correction enables volume-only error.
    Zero correction must be passed explicitly if it is the intended control.
    Remaining motion AND heart-rho are then deliberately omitted by that volume
    interpretation; a small electrical residual is not unique recovery.
    """
    p0, z0, v0 = _record(base, size_coordinate)
    p, z, v = _record(trial, size_coordinate)
    j = _array(jacobian, 2, "jacobian")
    _require(j.shape == (z0.size, 7) and z.shape == z0.shape, "jacobian/montage mismatch")
    actual, linear = z - z0, j @ (p - p0)
    apparent = error = None
    if tissue_correction_ohm is not None:
        correction = _vector(tissue_correction_ohm, z0.size, "tissue_correction_ohm")
        kv = j[:, 0] / volume_gradient(p0[0], size_coordinate)
        apparent = [_ratio(a - c, k) for a, c, k in zip(actual, correction, kv)]
        error = [None if a is None else a - (v - v0) for a in apparent]
    return dict(record_ids=[base["id"], trial["id"]], delta_parameters=p - p0,
        finite_delta_ohm=actual, linear_delta_ohm=linear, finite_minus_linear_ohm=actual - linear,
        true_integrated_delta_volume_ml=v - v0, apparent_volume_only_delta_ml=apparent,
        volume_only_error_ml=error, roi_status=_roi([base, trial]),
        status="saved_finite_control_not_independent_physical_validation")


def omission_table(base, cases, jacobian, omitted_indices, amplitudes, *, size_coordinate="V"):
    """Rows for supplied unit/finite_single/joint/compensated probe cases.

    cases = [{kind, record, tissue_correction_ohm}]. Each correction is explicit;
    no tissue term disappears by default. omitted_indices is a nonempty subset
    of x/y/z/rho_H (indices 1..4). Other heart contributions are explicitly
    removed using J for this CONDITIONAL isolation of the chosen omission.
    'unit' denotes a caller-specified finite amplitude, never an implicit 1 mm.
    Finite error includes all nonlinear/model remainder, not just the selected
    parameter's interaction. ROI status and amplitude excursions remain visible.
    """
    p0, z0, _ = _record(base, size_coordinate)
    j, scales, _, _ = _design(jacobian, amplitudes, None)
    _require(j.shape == (z0.size, 7), "need all seven derivative columns")
    omitted = list(omitted_indices)
    _require(omitted and len(set(omitted)) == len(omitted) and all(k in (1, 2, 3, 4) for k in omitted), "explicit omitted x/y/z/rho_H subset required")
    kept = [k for k in (1, 2, 3, 4) if k not in omitted]
    kv = j[:, 0] / volume_gradient(p0[0], size_coordinate)
    rows = []
    for case in cases:
        _require(case["kind"] in ("unit", "finite_single", "joint", "compensated"), "unknown probe kind")
        check = finite_probe_check(base, case["record"], j, size_coordinate=size_coordinate)
        correction = _vector(case["tissue_correction_ohm"], z0.size, "tissue_correction_ohm")
        dp = check["delta_parameters"]
        omitted_response = j[:, omitted] @ dp[omitted]
        tissue_error = j[:, 5:] @ dp[5:] - correction
        remaining = check["finite_delta_ohm"] - correction - j[:, kept] @ dp[kept]
        # For requiv, J_r*dr != K_V*dV at finite dr; retain this known coordinate
        # conversion remainder instead of attributing it to physical nonlinearity.
        size_coordinate_remainder = j[:, 0] * dp[0] - kv * check["true_integrated_delta_volume_ml"]
        for i, k in enumerate(kv):
            finite_omitted = remaining[i] - k * check["true_integrated_delta_volume_ml"]
            rows.append(dict(kind=case["kind"], record_id=case["record"]["id"], montage_index=i,
                omitted_indices=omitted, delta_parameters=dp, amplitude_ratios=np.abs(dp) / scales,
                K_V_ohm_per_ml=float(k), omitted_linear_ohm=float(omitted_response[i]),
                tissue_correction_error_linear_ohm=float(tissue_error[i]),
                finite_minus_all_linear_ohm=float(check["finite_minus_linear_ohm"][i]),
                size_coordinate_remainder_ohm=float(size_coordinate_remainder[i]),
                finite_unexplained_after_retained_terms_ohm=float(finite_omitted),
                linear_omission_volume_error_ml=_ratio(omitted_response[i], k),
                finite_volume_error_ml=_ratio(finite_omitted, k),
                roi_status=check["roi_status"],
                status="conditional_omission_test_not_justification_to_fix_unknowns"))
    return rows


def discretization_comparison(coarse_base, coarse_trial, fine_base, fine_trial, *, size_coordinate="V"):
    """Compare saved q64/q256 (or other declared levels), without setting QC limits.

    Inputs are two baseline/probe pairs. Non-size parameters must match; actual
    integrated volumes may differ and are reported. No automatic rejection of
    a physical hypothesis from a failed numerical/integration control.
    """
    parsed = [_record(r, size_coordinate) for r in (coarse_base, coarse_trial, fine_base, fine_trial)]
    (cb, zb, vb), (cp, zp, vp), (fb, zfb, vfb), (fp, zfp, vfp) = parsed
    _require(zb.shape == zp.shape == zfb.shape == zfp.shape, "montage count mismatch")
    _require(np.array_equal(cb[1:], fb[1:]) and np.array_equal(cp[1:], fp[1:]), "non-size states differ between discretizations")
    return dict(baseline_difference_ohm=zfb - zb,
        coarse_delta_ohm=zp - zb, fine_delta_ohm=zfp - zfb,
        delta_difference_ohm=(zfp - zfb) - (zp - zb),
        baseline_volume_difference_ml=vfb - vb, trial_volume_difference_ml=vfp - vp,
        coarse_integrated_delta_volume_ml=vp - vb, fine_integrated_delta_volume_ml=vfp - vfb,
        roi_status=_roi([coarse_base, coarse_trial, fine_base, fine_trial]),
        scope="integration_comparison_not_mesh_convergence_or_physical_rejection")


def law_evidence_check(evidence):
    """Conservative declaration gate, NOT an automatic validation certificate.

    Required evidence fields: law_id, no_ai, construction_ids, holdout_ids,
    current_law_holdout_checked, previous_4d_only, operational_phase_inputs,
    roi_conditions, amplitude_basis. Holdout IDs should identify whole source
    series/experiments (not interleaved phase IDs from the construction series).
    """
    reasons = []
    if not evidence.get("law_id"):
        reasons.append("law_not_identified")
    if evidence.get("no_ai") is not True:
        reasons.append("no_AI_not_confirmed")
    build, holdout = evidence.get("construction_ids", []), evidence.get("holdout_ids", [])
    if not build or not holdout or set(build) & set(holdout):
        reasons.append("missing_or_overlapping_construction_and_holdout_sources")
    if evidence.get("current_law_holdout_checked") is not True:
        reasons.append("current_law_holdout_not_checked")
    if evidence.get("previous_4d_only") is not False:
        reasons.append("previous_4D_is_not_validation_of_current_law")
    if evidence.get("operational_phase_inputs") != []:
        reasons.append("target_phase_geometry_is_used_or_not_excluded")
    if not evidence.get("roi_conditions") or not evidence.get("amplitude_basis"):
        reasons.append("ROI_or_amplitude_conditions_missing")
    return dict(eligible_declarations=not reasons, reasons=reasons,
                independent_validation_established=False,
                status="declarations_only_review_current_nonlinear_holdout_and_transfer_separately")


def restriction_check(jacobian, mapping, true_delta, amplitudes, *, evidence, row_weights=None,
                      size_coordinate="V", base_size=None):
    """Explicit local law delta_p = mapping @ reduced_coordinates.

    Rows cover ALL original unknowns. Zero rows explicitly fix those changes;
    dependent rows tie parameters. Mapping columns specify physical changes per
    dimensionless reduced coordinate. Compare the closest allowed parameter
    vector and the electrical-fit representative; neither is a validated law.
    This test uses LINEAR synthetic data, never labels it a nonlinear holdout.
    """
    j, scales, weights, _ = _design(jacobian, amplitudes, row_weights)
    _require(j.shape[1] in (5, 7), "restriction must cover all five heart parameters, optionally soft/lung")
    _require(size_coordinate in ("V", "requiv"), "invalid size coordinate")
    dvolume = 1.0 if size_coordinate == "V" else volume_gradient(base_size, size_coordinate)
    b = _array(mapping, 2, "mapping")
    _require(b.shape[0] == j.shape[1] and b.shape[1] > 0, "mapping must cover every unknown explicitly")
    truth = _vector(true_delta, j.shape[1], "true_delta")
    omitted_scaled, map_rank, _ = _orthogonal_residual(b / scales[:, None], truth / scales)
    _require(map_rank == b.shape[1], "redundant/zero law coordinates")
    omitted = scales * omitted_scaled
    allowed = truth - omitted
    reduced = j @ b
    inverse, _, rank, _, null = _svd(weights[:, None] * reduced)
    representative = b @ (inverse @ (weights * (j @ truth)))
    electrical_residual, _, _ = _orthogonal_residual(weights[:, None] * reduced, weights * (j @ truth))
    kv = j[:, 0] / dvolume
    return dict(mapping=b, fixed_zero_change_rows=np.flatnonzero(np.all(b == 0, axis=1)),
        closest_allowed_delta=allowed, omitted_delta=omitted, omitted_response_ohm=j @ omitted,
        omitted_equivalent_volume_ml=[_ratio(v, k) for v, k in zip(j @ omitted, kv)],
        electrical_fit_representative=representative, parameter_error=representative - truth,
        representative_linear_volume_error_ml=float(dvolume * (representative[0] - truth[0])),
        electrical_residual_ohm=j @ (truth - representative),
        orthogonal_fit_residual_ohm=electrical_residual / weights, reduced_rank=rank,
        reduced_unknowns=b.shape[1], physical_null_basis=b @ null,
        reduced_diagnostics=linear_diagnostics(reduced, np.ones(b.shape[1]), row_weights=weights),
        law_evidence=law_evidence_check(evidence),
        status="conditional_linear_restriction_test_not_nonlinear_holdout_or_validation")


def jsonable(value):
    """Strict JSON conversion. Nonfinite values are errors, not hidden as null."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float):
        _require(np.isfinite(value), "nonfinite result cannot be exported")
    return value


def self_test():
    """Deterministic mathematical self-tests; no project inputs, FEM, or writes."""
    checks = []

    def close(actual, expected):
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)

    def rejects(function):
        try:
            function()
        except (ValueError, KeyError):
            return
        raise AssertionError("invalid input accepted")

    p0 = np.array([500., 0., 0., 0., 4., 3., 18.])
    j = np.array([[2., 3., -4., 5., 6., 7., -8.],
                  [-1., 2., 3., -4., 5., -6., 7.]])
    z0 = np.array([100., 110.])
    scales = np.array([5., 1., 1., 1., .04, .03, .18])

    def record(name, dp=None, extra=None, roi=True, size="V"):
        dp = np.zeros(7) if dp is None else np.asarray(dp, float)
        p = p0 + dp
        volume = p[0]
        if size == "requiv":
            p[0] = (3000 * volume / (4 * np.pi)) ** (1 / 3)
        return dict(id=name, heart=p[:5].tolist(), tissue=p[5:].tolist(), volume_ml=float(volume),
                    impedance_ohm=(z0 + j @ dp + (0 if extra is None else np.asarray(extra))).tolist(),
                    roi_ok=roi, source_id="synthetic:" + name)

    base = record("base")
    # Every heart direction nonzero: one scalar cannot identify any one of them.
    one = linear_diagnostics(j[:1, :5], scales[:5])
    assert one["rank"] == 1 and one["nullity"] == 4 and one["condition"] is None
    close(one["unique_fraction"], np.zeros(5))
    close(j[:1, :5] @ one["null_basis_physical"], np.zeros((1, 4)))
    assert not np.any(one["individually_identified"])
    zero = linear_diagnostics(np.zeros((1, 5)), np.ones(5))
    assert zero["rank"] == 0 and zero["unique_fraction"] == [None] * 5
    eye = linear_diagnostics(np.eye(5), np.ones(5))
    assert eye["rank"] == 5 and np.all(eye["individually_identified"])
    close(eye["unique_fraction"], np.ones(5))
    for m in (3, 4):
        d = linear_diagnostics(np.eye(5)[:m], np.ones(5))
        assert d["rank"] == m and d["nullity"] == 5 - m
    checks.append("scalar_rank_nullspace_zero_orthogonal_three_four_montages")

    # Rank 3 can identify V while other parameters remain mutually ambiguous.
    identifiable = np.array([[1., 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 1, 1]])
    functional = volume_identified_given_nuisance(identifiable, [5, 1, 2, 3, .4], row_weights=[2, .5, 3])
    assert functional["rank"] == 3 and functional["volume_identified"]
    close(functional["eta_V"], 1.)
    close(functional["volume_estimator_weights_per_ohm"] @ identifiable, [1, 0, 0, 0, 0])
    ambiguous = identifiable.copy(); ambiguous[0, 3] = 1
    lost = volume_identified_given_nuisance(ambiguous, np.ones(5))
    assert lost["rank"] == 3 and not lost["volume_identified"]
    close(lost["eta_V"], 0.)
    assert lost["volume_estimator_weights_per_ohm"] is None
    assert np.linalg.norm(lost["null_volume_response_ml"]) > .1
    unseen = volume_identified_given_nuisance(np.zeros((3, 5)), np.ones(5))
    assert unseen["eta_V"] is None and not unseen["volume_identified"]
    rr = (3000 * 500 / (4 * np.pi)) ** (1 / 3)
    ir = identifiable.copy(); ir[:, 0] *= volume_gradient(rr, "requiv")
    radial = volume_identified_given_nuisance(ir, np.ones(5), size_coordinate="requiv", base_size=rr)
    close(radial["volume_estimator_weights_per_ohm"], [1, 0, 0])
    rejects(lambda: volume_identified_given_nuisance(ir, np.ones(5), size_coordinate="requiv"))
    checks.append("volume_target_rowspace_identified_at_rank3_or_ambiguous_at_same_rank")

    # Soft nuisance exactly absorbs V; projected roundoff must not invent rank.
    h = np.eye(5)
    t = np.column_stack([h[:, 0], h[:, 1]])
    stack = montage_diagnostics(h, t, np.ones(7), montage_ids=list("ABCDE"), stack_kind="alternative_montages")
    assert stack["stack_status"] == "hypothetical_design_not_simultaneous_data"
    assert stack["stack"]["heart_if_exact_tissue_corrections"]["rank"] == 5
    assert stack["stack"]["heart_after_free_tissue_projection"]["rank"] == 3
    close(stack["stack"]["heart_plus_free_tissue"]["unique_fraction"][:2], [0, 0])
    scalar = montage_diagnostics(j[:1, :5], j[:1, 5:], scales, montage_ids=["TT"], stack_kind="alternative_montages")
    assert scalar["stack"]["heart_after_free_tissue_projection"]["rank"] == 0
    rejects(lambda: montage_diagnostics(h, t, np.ones(7), montage_ids=list("ABCDE"), stack_kind="simultaneous_recording"))
    # Unit changes cannot alter dimensionless A or its singular values.
    conversion = np.array([1000., 1., 1., 1., 1.])
    units = linear_diagnostics(j[:, :5] / conversion, scales[:5] * conversion)
    close(units["singular_values"], linear_diagnostics(j[:, :5], scales[:5])["singular_values"])
    checks.append("tissue_separation_projection_tolerance_stack_provenance_unit_invariance")

    # Independent nonlinear polynomial: central error in cubic term is O(h^2).
    estimates = []
    for step in (1., .5, .25):
        dx = np.array([0., step, 0., 0., 0., 0., 0.])
        plus = record(f"p{step}", dx, [2 * step**2 + step**3, 0])
        minus = record(f"m{step}", -dx, [2 * step**2 - step**3, 0])
        estimate = finite_difference(base, minus, plus, "x")
        close(estimate["raw"]["central"], j[:, 1] + [step**2, 0])
        estimates.append(estimate)
    conv = step_convergence(estimates)
    changes = [r["absolute_change"][0] for r in conv["comparisons"]]
    close(changes[0] / changes[1], 4.)
    assert conv["comparisons"][0]["within_supplied_tolerances"] is None
    # Unequal steps: the 3-point formula, unlike the secant, differentiates x^2 exactly.
    minus = record("unequal_m", [0, -.5, 0, 0, 0, 0, 0], [.25, 0])
    plus = record("unequal_p", [0, 1, 0, 0, 0, 0, 0], [1, 0])
    close(finite_difference(base, minus, plus, "x")["raw"]["central"], j[:, 1])
    # Raw translations contain numerical integrated-volume drift; corrected do not.
    minus = record("drift_m", [-.2, -1, 0, 0, 0, 0, 0])
    plus = record("drift_p", [.3, 1, 0, 0, 0, 0, 0])
    drift = finite_difference(base, minus, plus, "x", kv=j[:, 0])
    close(drift["raw"]["central"], j[:, 1] + .25 * j[:, 0])
    close(drift["volume_corrected"]["central"], j[:, 1])
    rejects(lambda: finite_difference(base, record("bad", [0, -1, .1, 0, 0, 0, 0]), plus, "x"))
    rejects(lambda: step_convergence([estimates[0], estimates[0]]))
    checks.append("smaller_steps_cubic_order_asymmetric_steps_volume_drift_mixed_probe_rejection")

    # Analytic conductivity-law derivative checked against independent finite rho.
    rho_checks = []
    analytic_rho = np.array([-2., 3.]) / p0[4]**2
    for fraction in (.01, .05):
        probes = []
        for sign in (-1, 1):
            dr = sign * fraction * p0[4]
            r = record(f"rho_{sign}_{fraction}", [0, 0, 0, 0, dr, 0, 0])
            r["impedance_ohm"] = (z0 + np.array([2., -3.]) * (1 / (p0[4] + dr) - 1 / p0[4])).tolist()
            probes.append(r)
        difference = finite_difference(base, *probes, "rho_heart")
        checked = derivative_check(analytic_rho, difference)
        close(checked["finite_central"], analytic_rho / (1 - fraction**2))
        assert difference["volume_drift_correction_applied"] is False
        rho_checks.append(checked)
    assert rho_checks[0]["relative_norm_error"] < rho_checks[1]["relative_norm_error"]
    assert drift["volume_drift_correction_applied"] is True
    checks.append("analytic_rho_column_vs_one_and_five_percent_saved_finite_controls")

    # xy cross term survives joint-sum; x^2 single nonlinearity stays separate.
    sx = record("sx", [0, 1, 0, 0, 0, 0, 0], [2, 0])
    sy = record("sy", [0, 0, 1, 0, 0, 0, 0])
    joint = record("joint", [.1, 1, 1, 0, 0, 0, 0], [5, 0])
    decomposition = joint_decomposition(base, [sx, sy], joint, j)
    close(decomposition["interaction_after_linear_step_alignment_ohm"], [3, 0])
    close(decomposition["single_minus_linear_ohm"].sum(axis=0), [2, 0])
    close(decomposition["decomposition_closure_ohm"], [0, 0])
    checks.append("joint_sum_single_nonlinearity_and_parameter_mismatch_identity")

    # V-rho and x-V compensation have independent analytic signed answers.
    imposed = np.array([0., 1., 0., 0., 0., 0., 0.])
    comp = linear_compensation(j[:1], imposed, [0], scales)
    close(comp["compensation_delta"][0], -1.5)
    assert comp["exactly_compensable_at_numerical_tolerance"]
    volume_step = np.array([1., 0, 0, 0, 0, 0, 0])
    vrho = linear_compensation(j[:1], volume_step, [4], scales)
    close(vrho["compensation_delta"][4], -1 / 3)
    multiple = linear_compensation(j[:1], imposed, [0, 4], scales)
    assert not multiple["compensation_unique"]
    close(j[:1] @ multiple["compensation_null_basis_physical"], [[0.]])
    impossible = linear_compensation(np.eye(5), [1, 0, 0, 0, 0], [1], np.ones(5))
    assert not impossible["exactly_compensable_at_numerical_tolerance"]
    # Saved nonlinear control disproves exact finite cancellation of that candidate.
    finite = record("compensated", comp["total_delta"], [.4, 0])
    trial = finite_probe_check(base, finite, j, tissue_correction_ohm=[0, 0])
    close(trial["linear_delta_ohm"][0], 0.)
    close(trial["finite_delta_ohm"][0], .4)
    close(trial["volume_only_error_ml"][0], 1.7)
    checks.append("signed_compensation_null_family_infeasibility_and_finite_counterexample")

    # Omitted motion and correction errors are signed; no fabricated residual.
    om = volume_omission(j[:, :5], j[:, 5:], [1, 0, 0, 0], [.1, 0], base_size=500.)
    assert om["rows"][0]["volume_error_ml"] is None
    complete = volume_omission(j[:, :5], j[:, 5:], [1, 0, 0, 0], [.1, 0], base_size=500.,
        residual_ohm=[.2, 0], absolute_bounds=[1, 0, 0, 0, .1, 0], residual_bound_ohm=[.2, 0])
    close(complete["rows"][0]["volume_error_ml"], 1.95)
    close(complete["rows"][0]["conditional_box_bound_ml"], 1.95)
    no_k = j[:1, :5].copy(); no_k[0, 0] = 0
    undefined = volume_omission(no_k, j[:1, 5:], [1, 0, 0, 0], [0, 0], base_size=500., residual_ohm=[0])
    assert undefined["rows"][0]["volume_error_ml"] is None
    no_k[0, 0] = 1e-20
    large = volume_omission(no_k, j[:1, 5:], [1, 0, 0, 0], [0, 0], base_size=500., residual_ohm=[0])
    close(large["rows"][0]["volume_error_ml"], 3e20)
    checks.append("signed_omission_unknown_errors_box_bounds_zero_and_small_KV")

    # Omission table includes finite remainder and retains unknown/failed ROI.
    table = omission_table(base, [dict(kind="joint", record=joint, tissue_correction_ohm=[0, 0])], j, [1, 2], scales)
    close(table[0]["linear_omission_volume_error_ml"], -.5)
    close(table[0]["finite_volume_error_ml"], 2.)
    for roi, expected in ((None, "unknown"), (False, "failed")):
        flag = finite_probe_check(base, record("roi", roi=roi), j)
        assert flag["roi_status"] == expected
    radbase, radtrial = record("rbase", size="requiv"), record("rtrial", [5, 1, 0, 0, 0, 0, 0], size="requiv")
    r0 = radbase["heart"][0]
    jr = j.copy(); jr[:, 0] *= volume_gradient(r0, "requiv")
    rt = omission_table(radbase, [dict(kind="joint", record=radtrial, tissue_correction_ohm=[0, 0])], jr, [1], scales, size_coordinate="requiv")
    close(rt[0]["finite_volume_error_ml"], 1.5)
    close(rt[0]["finite_unexplained_after_retained_terms_ohm"],
          rt[0]["omitted_linear_ohm"] + rt[0]["finite_minus_all_linear_ohm"] + rt[0]["size_coordinate_remainder_ohm"])
    qfine = record("qfine", [0, 1, 0, 0, 0, 0, 0], [.1, -.2])
    disc = discretization_comparison(base, record("qcoarse", [0, 1, 0, 0, 0, 0, 0]), base, qfine)
    close(disc["delta_difference_ohm"], [.1, -.2])
    checks.append("omission_table_ROI_radius_coordinate_remainder_discretization")

    evidence = dict(law_id="test-law", no_ai=True, construction_ids=["series-A"], holdout_ids=["series-B"],
        current_law_holdout_checked=True, previous_4d_only=False, operational_phase_inputs=[],
        roi_conditions="declared test region", amplitude_basis="synthetic test scales")
    assert law_evidence_check(evidence)["eligible_declarations"]
    assert not law_evidence_check(dict(evidence, previous_4d_only=True))["eligible_declarations"]
    assert not law_evidence_check(dict(evidence, holdout_ids=["series-A"]))["eligible_declarations"]
    assert not law_evidence_check(dict(evidence, operational_phase_inputs=["target_phase_center"]))["eligible_declarations"]
    assert not law_evidence_check(dict(evidence, no_ai=False))["eligible_declarations"]
    restrict = restriction_check(j[:1, :5], np.eye(5)[:, :1], imposed[:5], scales[:5], evidence=evidence)
    close(restrict["representative_linear_volume_error_ml"], 1.5)
    close(restrict["omitted_equivalent_volume_ml"], [1.5])
    close(restrict["electrical_residual_ohm"], [0])
    assert not restrict["law_evidence"]["independent_validation_established"]
    tied = np.array([[1.], [2.], [0.], [0.], [0.]])
    tiedcheck = restriction_check(j[:1, :5], tied, [1, 2, 0, 0, 0], scales[:5], evidence=evidence)
    close(tiedcheck["omitted_delta"], np.zeros(5))
    checks.append("explicit_exclusion_tying_bias_and_no_AI_current_holdout_evidence_gate")

    # Regression 1: a small null-volume component alone is NOT identification.
    # The weak nuisance is above the common SVD cutoff and still spans V fully.
    weak_nuisance = np.array([[1., 1e-14, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1]])
    weak_volume = volume_identified_given_nuisance(weak_nuisance, np.ones(5))
    weak_design = linear_diagnostics(weak_nuisance, np.ones(5))
    assert weak_volume["rank"] == weak_volume["nuisance_rank"] == 3
    assert weak_volume["target_rowspace_relative_residual"] < weak_volume["target_rowspace_numerical_tolerance"]
    assert weak_volume["volume_rank_increase"] == weak_design["column_rank_increase"][0] == 0
    assert not weak_volume["volume_identified"] and weak_volume["volume_estimator_weights_per_ohm"] is None
    np.testing.assert_array_equal(weak_design["individually_identified"], [False, False, True, False, False])
    close(weak_volume["eta_V"], 0.)
    assert functional["volume_rank_increase"] == 1 and functional["volume_identified"]
    checks.append("regression_weak_nuisance_requires_rank_increase_for_volume_and_individual_flags")

    # Regression 2: H belongs to span(T), even when T is nearly collinear.
    # T @ pinv(T) formerly left ~1.5e-7 and falsely reported residual rank 1.
    q, _ = np.linalg.qr(np.array([[1., 2., 3.], [3., 1., 4.], [2., 4., 1.]]))
    nearly_tissue = np.column_stack([q[:, 0], q[:, 0] + 1e-9 * q[:, 1]])
    absorbed_heart = nearly_tissue @ np.array([[1., 2, 3, 4, 5], [-2, -1, 3, -3, -5]])
    absorbed = montage_diagnostics(absorbed_heart, nearly_tissue, np.ones(7),
        montage_ids=["A", "B", "C"], stack_kind="alternative_montages")
    projected = absorbed["stack"]["heart_after_free_tissue_projection"]
    assert projected["rank"] == 0 and absorbed["stack"]["tissue_rank"] == 2
    assert np.linalg.norm(projected["weighted_amplitude_matrix"]) < 1e-13
    nearly_heart = np.column_stack([absorbed_heart[:, 0], nearly_tissue, np.zeros((3, 2))])
    projected_volume = volume_identified_given_nuisance(nearly_heart, np.ones(5))
    projected_design = linear_diagnostics(nearly_heart, np.ones(5))
    assert projected_volume["eta_V"] < 1e-13 and not projected_volume["volume_identified"]
    assert projected_design["unique_fraction"][0] < 1e-13
    resolution = projected_design["coordinate_resolution"]
    assert np.linalg.norm(resolution @ resolution - resolution) < 1e-13
    nearly_mapping = np.vstack([nearly_tissue, np.zeros((2, 2))])
    restricted_truth = nearly_mapping @ np.array([1., -2.])
    projected_restriction = restriction_check(np.eye(5), nearly_mapping, restricted_truth,
                                              np.ones(5), evidence=evidence)
    assert np.linalg.norm(projected_restriction["omitted_delta"]) < 1e-13
    assert np.linalg.norm(projected_restriction["orthogonal_fit_residual_ohm"]) < 1e-13
    checks.append("regression_orthonormal_projections_do_not_invent_unique_directions_or_residual_rank")

    payload = dict(schema="heart_montage_analysis_v1", subject="Nik", size_coordinate="V",
        heart_order=list(HEART), tissue_order=list(TISSUE), amplitudes=scales.tolist(), amplitude_basis="synthetic",
        montage_ids=["A", "B"], stack_kind="alternative_montages",
        context=dict(shape="STL", scenario="synthetic", frequency_Hz=1., coordinate_frame="LPS_mm",
            background_id="synthetic", contact_model_id="synthetic", breath_interval_status="synthetic"),
        records=[base, minus, plus], difference_pairs=[dict(parameter="x", minus=minus["id"], plus=plus["id"])])
    before = json.dumps(payload, sort_keys=True)
    assert validate_export(payload)["record_count"] == 3
    assert json.dumps(payload, sort_keys=True) == before
    rejects(lambda: validate_export(dict(payload, tissue_order=["rho_left", "rho_right"])))
    rejects(lambda: linear_diagnostics([[1 + 1j]], [1]))
    rejects(lambda: linear_diagnostics([[1]], [0]))
    rejects(lambda: jsonable(float("nan")))
    for result in (one, zero, eye, functional, lost, unseen, radial, rho_checks, stack, scalar, units, conv, drift, decomposition, comp, vrho,
                   multiple, impossible, trial, om, complete, undefined, large, table, rt, disc, restrict, tiedcheck,
                   weak_volume, weak_design, absorbed, projected_volume, projected_design, projected_restriction):
        json.dumps(jsonable(result), allow_nan=False)
    checks.append("transport_contract_no_input_mutation_invalid_inputs_strict_JSON")
    return dict(status="passed", checks=checks,
                scope="mathematical_and_synthetic_selftest_only_no_FEM_or_physical_validation")


if __name__ == "__main__":
    if sys.argv[1:] != ["--self-test"]:
        raise SystemExit("Usage: python -B heart_montage_analysis.py --self-test; import pure functions for saved arrays/records")
    print(json.dumps(self_test(), ensure_ascii=False, allow_nan=False))
