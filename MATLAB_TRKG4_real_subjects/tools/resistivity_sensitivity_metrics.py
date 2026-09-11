"""Local resistivity scores for arbitrary user-defined electrode assemblies.

Inputs are signed derivatives for homogeneous changes of WHOLE compartments:
e.g. soft tissue, the whole effective heart (blood + myocardium), and lungs.
Compartment order and assembly geometry belong to the caller. Local probes and
elementwise absolute sensitivities are excluded; spatial cancellation must
already be retained in each supplied compartment derivative.

These are mathematical postprocessing scores, not physiological validation.
There is no rho/Z or volume normalization, FEM execution, or assumed tissue range.
All inputs must be nonempty, finite real numeric arrays (converted to float64).
Invalid inputs/shapes raise ValueError; unsafe floating-point arithmetic raises
FloatingPointError instead of silently returning overflow or underflow.
"""

import numpy as np


def _finite_real(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or array.size == 0:
        raise ValueError(f"{name} must contain nonempty real numeric data")
    with np.errstate(all="raise"):
        array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def conductivity_to_resistivity(sigma, dZ_dsigma):
    """Return (rho, dZ_drho) via rho=1/sigma, dZ_drho=-sigma**2*dZ_dsigma.

    sigma is strictly positive conductivity [S/m]; dZ_dsigma has units
    Ohm/(S/m). Outputs have units Ohm*m and Ohm/(Ohm*m), respectively.
    Standard NumPy broadcasting is allowed, including scalars; BOTH outputs
    have the common broadcast shape. The caller must align compartment axes
    and hold geometry and all other parameters fixed in the supplied derivative.
    """
    sigma = _finite_real(sigma, "sigma")
    derivative = _finite_real(dZ_dsigma, "dZ_dsigma")
    if np.any(sigma <= 0):
        raise ValueError("sigma must be strictly positive")
    sigma, derivative = np.broadcast_arrays(sigma, derivative)
    with np.errstate(all="raise"):
        rho = 1.0 / sigma
        # Avoid forming sigma**2 before multiplying a potentially small derivative.
        dZ_drho = -sigma * (sigma * derivative)
    return rho, dZ_drho


def range_scaled_metrics(dZ_drho, delta_rho, *, target_rtol=1e-12):
    """Return per-compartment local linear range-scale scores and ratios.

    dZ_drho [Ohm/(Ohm*m)] and positive delta_rho [Ohm*m] must each have a
    last compartment axis of IDENTICAL length. Only leading axes broadcast,
    e.g. derivatives (states, assemblies, compartments), scales (compartments,)
    or (states, 1, compartments). Each output has the common broadcast shape.

    Dictionary keys:
      signed_response: dZ_drho * delta_rho [Ohm].
      absolute_response: abs(signed_response) [Ohm].
      selectivity: abs_target / sum(abs_all), dimensionless; NaN if all zero.
      nuisance_sum: sum(abs_all) - abs_target [Ohm], evaluated by summing
          other compartments directly to retain small nuisance contributions.
      nuisance_to_target: nuisance_sum / abs_target, dimensionless, or NaN.
      selectivity_undefined, nuisance_to_target_undefined: explicit bool masks.

    Every compartment is treated in turn as target. Opposite signs do not
    cancel in the absolute-score denominator or nuisance sum. target_rtol is
    a finite scalar in [0, 1): a target whose selectivity <= target_rtol has an
    undefined nuisance_to_target ratio. This relative guard preserves common
    scale invariance. Set it to zero explicitly for exact-zero-only masking;
    selectivity itself remains defined for every nonzero total.

    delta_rho is a caller-chosen comparison scale, not a perturbation solved
    here. The score is a derivative at one state times that scale; it is NOT
    the finite full-range response, nor a claimed cardiac/respiratory amplitude.
    Use common scales and compartment ordering when comparing assemblies.
    """
    derivative = _finite_real(dZ_drho, "dZ_drho")
    scale = _finite_real(delta_rho, "delta_rho")
    tolerance = _finite_real(target_rtol, "target_rtol")
    if tolerance.ndim != 0 or not 0 <= tolerance < 1:
        raise ValueError("target_rtol must be a scalar in [0, 1)")
    if derivative.ndim == 0 or scale.ndim == 0:
        raise ValueError("dZ_drho and delta_rho need a last compartment axis")
    if derivative.shape[-1] != scale.shape[-1]:
        raise ValueError("last compartment dimensions must match exactly")
    if np.any(scale <= 0):
        raise ValueError("delta_rho must be strictly positive")
    derivative, scale = np.broadcast_arrays(derivative, scale)
    with np.errstate(all="raise"):
        signed = derivative * scale
        absolute = np.abs(signed)
        total = np.sum(absolute, axis=-1, keepdims=True)
        selectivity_undefined = np.broadcast_to(total == 0, signed.shape)
        selectivity = np.full_like(signed, np.nan)
        np.divide(absolute, total, out=selectivity, where=~selectivity_undefined)
        nuisance = np.empty_like(absolute)
        for target in range(absolute.shape[-1]):
            nuisance[..., target] = (
                np.sum(absolute[..., :target], axis=-1)
                + np.sum(absolute[..., target + 1:], axis=-1)
            )
        ratio_undefined = selectivity_undefined | (selectivity <= tolerance)
        ratio = np.full_like(signed, np.nan)
        np.divide(nuisance, absolute, out=ratio, where=~ratio_undefined)
    return {
        "signed_response": signed,
        "absolute_response": absolute,
        "selectivity": selectivity,
        "nuisance_sum": nuisance,
        "nuisance_to_target": ratio,
        "selectivity_undefined": selectivity_undefined,
        "nuisance_to_target_undefined": ratio_undefined,
    }


def summarize_over_states(values):
    """Return min, max, sign_changed, min_abs, max_abs over axis 0 (states).

    Input shape is (states, ...); a 1-D input summarizes scalar observations.
    Other dimensions are preserved. Extrema retain the input units.
    sign_changed means at least two of {-1, 0, +1} occur: negative/positive
    reversal AND a sampled zero together with a nonzero value count. All-zero
    or constant-sign samples return False. This is not a claim about the path
    between unordered states, or evidence of an unsampled zero crossing.

    min_abs/max_abs are extrema of absolute SAMPLED values, not of the
    continuous interval between min and max. No probabilities, confidence
    intervals, or unsampled-state bounds are inferred. NaNs (including
    undefined ratios) are rejected; callers must handle their masks explicitly.
    """
    values = _finite_real(values, "values")
    if values.ndim == 0:
        raise ValueError("values must have a state axis")
    minimum = np.min(values, axis=0)
    maximum = np.max(values, axis=0)
    return {
        "min": minimum,
        "max": maximum,
        "sign_changed": np.sign(minimum) != np.sign(maximum),
        "min_abs": np.min(np.abs(values), axis=0),
        "max_abs": np.max(np.abs(values), axis=0),
    }
