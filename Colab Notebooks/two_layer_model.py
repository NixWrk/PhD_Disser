"""Canonical scalar two-layer half-space model for notebook series 30–32.

All lengths are in metres, resistivities in ohm-metres, and the returned
transfer impedance is the real signed value in ohms.  The model assumes point
electrodes on a flat two-layer half-space; it does not model phase, contact
impedance, ribs, curvature, or an instrument gain/offset.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isfinite, pi

import numpy as np


@dataclass(frozen=True)
class Evaluation:
    z: float
    d_rho1: float
    d_rho2: float
    d_h: float
    d_a: float
    d_b: float
    n_terms: int


def geometry_from_size(size: float, beta: float = 0.5) -> tuple[float, float]:
    """Return ``(a, b)`` for current-electrode spacing ``size = 2a``."""
    if not isfinite(size) or size <= 0:
        raise ValueError("size must be a positive finite length")
    if not isfinite(beta) or not 0 < beta < 1:
        raise ValueError("beta=b/a must lie strictly between 0 and 1")
    a = size / 2.0
    return a, beta * a


def _validate(rho1: float, rho2: float, h: float, a: float, b: float) -> None:
    values = {"rho1": rho1, "rho2": rho2, "h": h, "a": a, "b": b}
    for name, value in values.items():
        if not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not b < a:
        raise ValueError("the closed formula requires 0 < b < a")


def _evaluate_n(
    rho1: float, rho2: float, h: float, a: float, b: float, n_terms: int
) -> Evaluation:
    contrast = (rho2 - rho1) / (rho1 + rho2)
    d1, d2 = a - b, a + b
    idx = np.arange(1, n_terms + 1, dtype=float)
    g1 = 1.0 / np.sqrt(d1 * d1 + (2.0 * idx * h) ** 2)
    g2 = 1.0 / np.sqrt(d2 * d2 + (2.0 * idx * h) ** 2)
    dg = g1 - g2
    powers = np.power(contrast, idx)
    powers_minus_one = np.power(contrast, idx - 1.0)

    base = rho1 / pi * (1.0 / d1 - 1.0 / d2)
    z = base + 2.0 * rho1 / pi * np.sum(powers * dg)

    t_sum = np.sum(idx * powers_minus_one * dg)
    d_rho2 = 4.0 * rho1**2 / (pi * (rho1 + rho2) ** 2) * t_sum
    d_rho1 = z / rho1 - 4.0 * rho1 * rho2 / (
        pi * (rho1 + rho2) ** 2
    ) * t_sum

    u_sum = np.sum(idx**2 * powers * (g1**3 - g2**3))
    d_h = -8.0 * rho1 * h / pi * u_sum
    d_a = rho1 / pi * (1.0 / d2**2 - 1.0 / d1**2) + 2.0 * rho1 / pi * np.sum(
        powers * (d2 * g2**3 - d1 * g1**3)
    )
    d_b = rho1 / pi * (1.0 / d1**2 + 1.0 / d2**2) + 2.0 * rho1 / pi * np.sum(
        powers * (d1 * g1**3 + d2 * g2**3)
    )
    return Evaluation(
        float(z),
        float(d_rho1),
        float(d_rho2),
        float(d_h),
        float(d_a),
        float(d_b),
        n_terms,
    )


def _close(previous: Evaluation, current: Evaluation, rtol: float, atol: float) -> bool:
    names = ("z", "d_rho1", "d_rho2", "d_h", "d_a", "d_b")
    return all(
        abs(getattr(current, name) - getattr(previous, name))
        <= atol + rtol * max(abs(getattr(current, name)), abs(getattr(previous, name)))
        for name in names
    )


def evaluate(
    rho1: float,
    rho2: float,
    h: float,
    a: float,
    b: float,
    *,
    rtol: float = 1e-11,
    atol: float = 1e-12,
    max_terms: int = 16384,
) -> Evaluation:
    """Evaluate impedance and analytic first derivatives with convergence checks."""
    _validate(rho1, rho2, h, a, b)
    if rtol <= 0 or atol < 0:
        raise ValueError("rtol must be positive and atol non-negative")
    if max_terms < 16:
        raise ValueError("max_terms must be at least 16")

    n_terms = 8
    previous = _evaluate_n(rho1, rho2, h, a, b, n_terms)
    while n_terms < max_terms:
        n_terms = min(2 * n_terms, max_terms)
        current = _evaluate_n(rho1, rho2, h, a, b, n_terms)
        if _close(previous, current, rtol, atol):
            return current
        previous = current
    raise RuntimeError(f"series did not converge by max_terms={max_terms}")


def transfer_impedance(
    rho1: float, rho2: float, h: float, a: float, b: float, **kwargs: float
) -> float:
    return evaluate(rho1, rho2, h, a, b, **kwargs).z


def apparent_resistivity(z: float, a: float, b: float) -> float:
    if not all(isfinite(value) for value in (z, a, b)) or not 0 < b < a:
        raise ValueError("z, a, and b must be finite and satisfy 0 < b < a")
    return z * pi * (a * a - b * b) / (2.0 * b)


def forward_curve(
    sizes: np.ndarray | list[float],
    rho1: float,
    rho2: float,
    h: float,
    beta: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return signed impedances and the Jacobian columns ``rho1, rho2, h``."""
    z_values: list[float] = []
    rows: list[list[float]] = []
    for size in np.asarray(sizes, dtype=float):
        a, b = geometry_from_size(float(size), beta)
        result = evaluate(rho1, rho2, h, a, b)
        z_values.append(result.z)
        rows.append([result.d_rho1, result.d_rho2, result.d_h])
    return np.asarray(z_values), np.asarray(rows)


@lru_cache(maxsize=8)
def _ratio_grid_and_powers(
    ratio_min: float,
    ratio_max: float,
    n_points: int,
    n_terms: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a cached log-ratio grid and powers of its reflection factors."""
    ratios = np.geomspace(ratio_min, ratio_max, n_points)
    contrasts = (ratios - 1.0) / (ratios + 1.0)
    indices = np.arange(1, n_terms + 1, dtype=float)
    powers = np.power(contrasts[np.newaxis, :], indices[:, np.newaxis])
    ratios.setflags(write=False)
    powers.setflags(write=False)
    return ratios, powers


def isoimpedance_curves(
    sizes: np.ndarray | list[float],
    observed_z: np.ndarray | list[float],
    h: float,
    beta: float = 0.5,
    *,
    ratio_min: float = 1e-4,
    ratio_max: float = 1e4,
    n_points: int = 161,
    n_terms: int = 8192,
) -> dict[str, np.ndarray | int]:
    """Parameterize fixed-impedance solution curves in the rho1-rho2 plane.

    For every electrode size and observed positive transfer impedance, the
    function varies the ratio rho2/rho1 on a logarithmic grid. The series
    then gives an impedance kernel K such that Z = rho1 * K; therefore
    rho1 = observed_z / K and rho2 = ratio * rho1. Each returned row
    is the solution set for one observation at the supplied fixed thickness.

    relative_kernel_change compares the full series with its first half.
    It is a numerical truncation diagnostic for plotting, not an uncertainty
    estimate for the physical model.
    """
    sizes_array = np.asarray(sizes, dtype=float)
    z_array = np.asarray(observed_z, dtype=float)
    if sizes_array.ndim != 1 or z_array.ndim != 1:
        raise ValueError("sizes and observed_z must be one-dimensional")
    if sizes_array.size == 0 or sizes_array.shape != z_array.shape:
        raise ValueError("sizes and observed_z must have the same non-zero length")
    if not np.all(np.isfinite(z_array)) or not np.all(z_array > 0):
        raise ValueError("observed_z must contain positive finite impedances")
    if not isfinite(h) or h <= 0:
        raise ValueError("h must be a positive finite length")
    if not isfinite(beta) or not 0 < beta < 1:
        raise ValueError("beta=b/a must lie strictly between 0 and 1")
    if (
        not isfinite(ratio_min)
        or not isfinite(ratio_max)
        or ratio_min <= 0
        or ratio_max <= ratio_min
    ):
        raise ValueError("ratio bounds must be positive, finite, and increasing")
    if n_points < 3 or n_terms < 16 or n_terms % 2:
        raise ValueError("n_points must be at least 3 and n_terms an even integer >= 16")

    ratios, powers = _ratio_grid_and_powers(
        float(ratio_min), float(ratio_max), int(n_points), int(n_terms)
    )
    half_terms = n_terms // 2
    rho1_rows: list[np.ndarray] = []
    rho2_rows: list[np.ndarray] = []
    relative_changes: list[float] = []
    indices = np.arange(1, n_terms + 1, dtype=float)
    for size, z_value in zip(sizes_array, z_array, strict=True):
        a, b = geometry_from_size(float(size), beta)
        d1, d2 = a - b, a + b
        g1 = 1.0 / np.sqrt(d1 * d1 + (2.0 * indices * h) ** 2)
        g2 = 1.0 / np.sqrt(d2 * d2 + (2.0 * indices * h) ** 2)
        delta_geometry = g1 - g2
        base_geometry = 1.0 / d1 - 1.0 / d2
        kernel = (base_geometry + 2.0 * (delta_geometry @ powers)) / pi
        kernel_half = (
            base_geometry
            + 2.0 * (delta_geometry[:half_terms] @ powers[:half_terms])
        ) / pi
        if not np.all(np.isfinite(kernel)) or not np.all(kernel > 0):
            raise RuntimeError("isoimpedance kernel is non-positive or non-finite")
        relative_changes.append(
            float(np.max(np.abs(kernel - kernel_half) / np.maximum(np.abs(kernel), 1e-15)))
        )
        rho1 = float(z_value) / kernel
        rho1_rows.append(rho1)
        rho2_rows.append(ratios * rho1)

    return {
        "rho2_to_rho1_ratio": ratios.copy(),
        "rho1_ohm_m": np.asarray(rho1_rows),
        "rho2_ohm_m": np.asarray(rho2_rows),
        "relative_kernel_change": np.asarray(relative_changes),
        "n_terms": int(n_terms),
    }


def surface_kernel(
    distance: float,
    rho1: float,
    rho2: float,
    h: float,
    *,
    rtol: float = 1e-11,
    max_terms: int = 16384,
) -> float:
    """Surface potential per unit current for one point source."""
    if not all(isfinite(value) and value > 0 for value in (distance, rho1, rho2, h)):
        raise ValueError("distance, resistivities, and h must be positive and finite")
    contrast = (rho2 - rho1) / (rho1 + rho2)

    def partial(n_terms: int) -> float:
        idx = np.arange(1, n_terms + 1, dtype=float)
        series = np.sum(
            np.power(contrast, idx)
            / np.sqrt(distance * distance + (2.0 * idx * h) ** 2)
        )
        return rho1 / (2.0 * pi) * (1.0 / distance + 2.0 * series)

    n_terms = 8
    previous = partial(n_terms)
    while n_terms < max_terms:
        n_terms = min(2 * n_terms, max_terms)
        current = partial(n_terms)
        if abs(current - previous) <= rtol * max(abs(current), abs(previous)):
            return float(current)
        previous = current
    raise RuntimeError(f"surface-kernel series did not converge by {max_terms} terms")


def transfer_impedance_coordinates(
    current_plus: float,
    current_minus: float,
    voltage_plus: float,
    voltage_minus: float,
    rho1: float,
    rho2: float,
    h: float,
) -> float:
    """Signed transfer impedance for four distinct collinear coordinates."""
    points = (current_plus, current_minus, voltage_plus, voltage_minus)
    if not all(isfinite(point) for point in points) or len(set(points)) != 4:
        raise ValueError("all four electrode coordinates must be finite and distinct")

    def kernel(x: float, y: float) -> float:
        return surface_kernel(abs(x - y), rho1, rho2, h)

    return (
        kernel(voltage_plus, current_plus)
        - kernel(voltage_plus, current_minus)
        - kernel(voltage_minus, current_plus)
        + kernel(voltage_minus, current_minus)
    )


__all__ = [
    "Evaluation",
    "apparent_resistivity",
    "evaluate",
    "forward_curve",
    "geometry_from_size",
    "isoimpedance_curves",
    "surface_kernel",
    "transfer_impedance",
    "transfer_impedance_coordinates",
]
