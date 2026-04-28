"""
Python sketch adapted from the Mathcad 14 worksheet:
31_28_26mod_Rasceti_2fix (3).xmcd

The script keeps the worksheet's calculation structure and uses only the
Python standard library. Values are in the same base units as the Mathcad file.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt


@dataclass(frozen=True)
class ModelParams:
    """Default values taken from the Mathcad worksheet."""

    h1: float = 1e-3
    rho1: float = 774e-2
    r1: float = 3e-3
    rho2: float = 100e-2
    a0: float = 30e-3
    x0: float = 0.0
    bl0: float = 15e-3
    y0: float = 0.0
    br0: float = 15e-3
    n_terms: int = 100


def legendre_p(n: int, x: float) -> float:
    """Legendre polynomial P_n(x), replacing Mathcad Leg(n, x)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 1.0
    if n == 1:
        return x

    p_nm2 = 1.0
    p_nm1 = x
    for k in range(2, n + 1):
        p_n = ((2 * k - 1) * x * p_nm1 - (k - 1) * p_nm2) / k
        p_nm2, p_nm1 = p_nm1, p_n
    return p_nm1


def r_0a(rad: float, h: float, a: float, x: float, y: float) -> float:
    return sqrt((rad + h) ** 2 + (a - y) ** 2 + x**2)


def r_m(rad: float, h: float, b: float, x: float, y: float) -> float:
    return sqrt((rad + h) ** 2 + (b - y) ** 2 + x**2)


def r_0b(rad: float, h: float, a: float, x: float, y: float) -> float:
    return sqrt((rad + h) ** 2 + (a + y) ** 2 + x**2)


def r_n(rad: float, h: float, b: float, x: float, y: float) -> float:
    return sqrt((rad + h) ** 2 + (b + y) ** 2 + x**2)


def cos_theta_am(rad: float, h: float, a: float, b: float, x: float, y: float) -> float:
    r0a = r_0a(rad, h, a, x, y)
    rm = r_m(rad, h, b, x, y)
    return (r0a**2 + rm**2 - (a - b) ** 2) / (2 * r0a * rm)


def cos_theta_an(rad: float, h: float, a: float, b: float, x: float, y: float) -> float:
    r0a = r_0a(rad, h, a, x, y)
    rn = r_n(rad, h, b, x, y)
    return (r0a**2 + rn**2 - (a + b) ** 2) / (2 * r0a * rn)


def cos_theta_bm(rad: float, h: float, a: float, b: float, x: float, y: float) -> float:
    r0b = r_0b(rad, h, a, x, y)
    rm = r_m(rad, h, b, x, y)
    return (r0b**2 + rm**2 - (a + b) ** 2) / (2 * r0b * rm)


def cos_theta_bn(rad: float, h: float, a: float, b: float, x: float, y: float) -> float:
    r0b = r_0b(rad, h, a, x, y)
    rn = r_n(rad, h, b, x, y)
    return (r0b**2 + rn**2 - (a - b) ** 2) / (2 * r0b * rn)


def _contrast_factor(n: int, rho1: float, rho2: float) -> float:
    numerator = n * rho2 - n * rho1
    denominator = (n + 1) * rho2 + n * rho1
    return numerator / denominator


def dz_am(
    rho1: float,
    rho2: float,
    a: float,
    b: float,
    rad: float,
    h: float,
    x: float,
    y: float,
    n_terms: int = 100,
) -> float:
    r0a = r_0a(rad, h, a, x, y)
    rm = r_m(rad, h, b, x, y)
    c = cos_theta_am(rad, h, a, b, x, y)
    total = 0.0
    for n in range(n_terms + 1):
        total += (
            (rad / rm) ** (n + 1)
            * (rad / r0a) ** n
            * _contrast_factor(n, rho1, rho2)
            * legendre_p(n, c)
        )
    return rho1 / (pi * r0a) * total


def dz_an(
    rho1: float,
    rho2: float,
    a: float,
    b: float,
    rad: float,
    h: float,
    x: float,
    y: float,
    n_terms: int = 100,
) -> float:
    r0a = r_0a(rad, h, a, x, y)
    rn = r_n(rad, h, b, x, y)
    c = cos_theta_an(rad, h, a, b, x, y)
    total = 0.0
    for n in range(n_terms + 1):
        total += (
            (rad / rn) ** (n + 1)
            * (rad / r0a) ** n
            * _contrast_factor(n, rho1, rho2)
            * legendre_p(n, c)
        )
    return rho1 / (pi * r0a) * total


def dz_bm(
    rho1: float,
    rho2: float,
    a: float,
    b: float,
    rad: float,
    h: float,
    x: float,
    y: float,
    n_terms: int = 100,
) -> float:
    r0b = r_0b(rad, h, a, x, y)
    rm = r_m(rad, h, b, x, y)
    c = cos_theta_bm(rad, h, a, b, x, y)
    total = 0.0
    for n in range(n_terms + 1):
        total += (
            (rad / rm) ** (n + 1)
            * (rad / r0b) ** n
            * _contrast_factor(n, rho1, rho2)
            * legendre_p(n, c)
        )
    return rho1 / (pi * r0b) * total


def dz_bn(
    rho1: float,
    rho2: float,
    a: float,
    b: float,
    rad: float,
    h: float,
    x: float,
    y: float,
    n_terms: int = 100,
) -> float:
    r0b = r_0b(rad, h, a, x, y)
    rn = r_n(rad, h, b, x, y)
    c = cos_theta_bn(rad, h, a, b, x, y)
    total = 0.0
    for n in range(n_terms + 1):
        total += (
            (rad / rn) ** (n + 1)
            * (rad / r0b) ** n
            * _contrast_factor(n, rho1, rho2)
            * legendre_p(n, c)
        )
    return rho1 / (pi * r0b) * total


def dz_total(
    rho1: float,
    rho2: float,
    a: float,
    bl: float,
    br: float,
    rad: float,
    h: float,
    x: float,
    y: float,
    n_terms: int = 100,
) -> float:
    return (
        dz_am(rho1, rho2, a, bl, rad, h, x, y, n_terms)
        - dz_an(rho1, rho2, a, br, rad, h, x, y, n_terms)
        + dz_bn(rho1, rho2, a, br, rad, h, x, y, n_terms)
        - dz_bm(rho1, rho2, a, bl, rad, h, x, y, n_terms)
    )


def z_base(
    rho1: float,
    rho2: float,
    a: float,
    bl: float,
    br: float,
    rad: float,
    h: float,
    x: float,
    y: float,
    n_terms: int = 100,
) -> float:
    base = (rho1 * 2 * bl) / (pi * (a**2 - bl**2))
    return base + dz_total(rho1, rho2, a, bl, br, rad, h, x, y, n_terms)


def z_bases(
    rho1: float,
    rho2: float,
    a: float,
    b: float,
    rad: float,
    h: float,
    x: float,
    y: float,
    n_terms: int = 100,
) -> float:
    return z_base(rho1, rho2, a, b, b, rad, h, x, y, n_terms)


def ro1_base_1(h: float, rad: float, zbase: float, a: float, b: float) -> float | None:
    return _find_rho_by_zbase(h, rad, zbase, a, b, y=0.0, dro=0.2, i_min=-10, i_max=10)


def ro1_base_ac(h: float, rad: float, zbase: float, a: float, b: float, y: float = 0.0) -> float | None:
    if y == 0.0:
        return _find_rho_by_zbase(h, rad, zbase, a, b, y=0.0, dro=0.1, i_min=-20, i_max=20)
    return _find_rho_by_zbase(h, rad, zbase, a, b, y=y, dro=0.1, i_min=-40, i_max=60)


def _find_rho_by_zbase(
    h: float,
    rad: float,
    zbase: float,
    a: float,
    b: float,
    y: float,
    dro: float,
    i_min: int,
    i_max: int,
    inclusion_rho: float = 1.35,
) -> float | None:
    for i in range(i_min, i_max + 1):
        rho_left = 5 + dro * i
        rho_right = 5 + dro * (i + 1)
        zb1 = z_base(rho_left, inclusion_rho, a, b, b, rad, h, 0.0, y)
        zb2 = z_base(rho_right, inclusion_rho, a, b, b, rad, h, 0.0, y)
        if zb2 > zbase and zb1 < zbase:
            return rho_left
    return None


def dro_fl(dz: float) -> float:
    """Mathcad droFL(dz), preserved as written in the worksheet."""
    return dz * pi * (0.05**2 - 0.025**3) / (2 * 0.025)


def d_radius(
    target_dz: float,
    d_ro: float,
    rho1: float,
    h: float,
    rad: float,
    a: float,
    b: float,
    y: float,
    inclusion_rho: float = 1.35,
) -> float | None:
    dr = 0.0001
    reference = z_base(rho1, inclusion_rho, a, b, b, rad, h, 0.0, y)
    for i in range(0, 151):
        zb1 = reference - z_base(rho1 - d_ro, inclusion_rho, a, b, b, rad - dr * i, h, 0.0, y)
        zb2 = reference - z_base(rho1 - d_ro, inclusion_rho, a, b, b, rad - dr * (i + 1), h, 0.0, y)
        if zb1 > target_dz and zb2 < target_dz:
            return i * dr
    return None


def d_radius_x_change(
    target_dz: float,
    d_ro: float,
    dy: float,
    h: float,
    rad: float,
    zbase: float,
    a: float,
    b: float,
    y: float,
) -> float | None:
    rho = ro1_base_ac(h, rad, zbase, a, b, y + dy)
    if rho is None:
        return None
    return d_radius(-target_dz, d_ro, rho, h, rad, a, b, y + dy)


def run_default_case(params: ModelParams = ModelParams()) -> dict[str, float]:
    return {
        "dZ": dz_total(
            params.rho1,
            params.rho2,
            params.a0,
            params.bl0,
            params.br0,
            params.r1,
            params.h1,
            params.x0,
            params.y0,
            params.n_terms,
        ),
        "Zbase": z_base(
            params.rho1,
            params.rho2,
            params.a0,
            params.bl0,
            params.br0,
            params.r1,
            params.h1,
            params.x0,
            params.y0,
            params.n_terms,
        ),
        "Zbases": z_bases(
            params.rho1,
            params.rho2,
            params.a0,
            params.bl0,
            params.r1,
            params.h1,
            params.x0,
            params.y0,
            params.n_terms,
        ),
    }


if __name__ == "__main__":
    for name, value in run_default_case().items():
        print(f"{name} = {value:.12g}")
