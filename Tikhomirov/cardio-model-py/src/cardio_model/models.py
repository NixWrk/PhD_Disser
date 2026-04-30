"""Аналитические модели импеданса: OneLayer, TwoLayer, Sphere.

Порт `Kernel/core/Models.m` (Mathematica 11.3).

Физический смысл: реоимпедансные модели тела (торакального сегмента),
используемые для нахождения параметров сердца через подгонку.

  • OneLayerModel — гомогенный проводящий цилиндр.
  • TwoLayerModel — два слоя (ткани + жир), бесконечная серия n=1..100.
  • SphereModel   — цилиндр + шарообразное сердце; ряд Лежандра n=0..100.

Публичное API (коммит #11):
    one_layer_model(rho1, a, b)
    two_layer_model(rho1, rho2, h, a, b)

Публичное API (коммит #13):
    sphere_model(rho1, rho2, a, b, R, h, x, y)
    sphere_influence(rho1, rho2, a, b, R, h, x, y)   ← для тестирования
"""

from __future__ import annotations

import numpy as np
from scipy.special import eval_legendre

# ---------------------------------------------------------------------------
# OneLayerModel
# ---------------------------------------------------------------------------


def one_layer_model(rho1: float, a: float, b: float) -> float:
    """Однослойная модель импеданса.

    Порт `OneLayerModel[ρ1, a, b]` из Models.m::

        Z = ρ1 · 2b / (π · (a² − b²))

    Параметры
    ---------
    rho1 : float
        Удельное сопротивление тканевого слоя (Ом·м).
    a : float
        Большая полуось эллипса торакального сегмента (м).
    b : float
        Малая полуось (расстояние между электродами / 2) (м).

    Возвращает
    ----------
    float — измеренный импеданс (Ом).
    """
    return (rho1 * 2.0 * b) / (np.pi * (a**2 - b**2))


# ---------------------------------------------------------------------------
# TwoLayerModel
# ---------------------------------------------------------------------------


def _two_layer_under_sum(rho1: float, rho2: float, h: float,
                          a: float, b: float, i: int) -> float:
    """Член ряда для TwoLayerModel (TwoLUnderSum в Mathematica).

    i-й член::

        ((ρ2-ρ1)/(ρ2+ρ1))^i · (1/√((a-b)²+(2ih)²) − 1/√((a+b)²+(2ih)²))
    """
    ratio = ((rho2 - rho1) / (rho2 + rho1)) ** i
    two_ih = 2.0 * i * h
    return ratio * (
        1.0 / np.sqrt((a - b) ** 2 + two_ih ** 2)
        - 1.0 / np.sqrt((a + b) ** 2 + two_ih ** 2)
    )


def two_layer_model(rho1: float, rho2: float, h: float,
                    a: float, b: float, *, n_max: int = 100) -> float:
    """Двухслойная модель импеданса (тело + жировой слой).

    Порт `TwoLayerModel[ro1, ro2, h, a, b]` из Models.m::

        Z = ρ1/π · (1/(a−b) − 1/(a+b))
          + 2ρ1/π · Σ_{i=1..n_max} TwoLUnderSum(ρ1, ρ2, h, a, b, i)

    Параметры
    ---------
    rho1 : float   Удельное сопротивление внешнего слоя (Ом·м).
    rho2 : float   Удельное сопротивление жира (Ом·м).
    h : float      Толщина жирового слоя (м).
    a : float      Большая полуось торакального сечения (м).
    b : float      Малая полуось / полу-расстояние электродов (м).
    n_max : int    Количество членов ряда (по умолчанию 100, как в Mathematica).

    Возвращает
    ----------
    float — импеданс (Ом).
    """
    base = rho1 / np.pi * (1.0 / (a - b) - 1.0 / (a + b))
    series = sum(_two_layer_under_sum(rho1, rho2, h, a, b, i) for i in range(1, n_max + 1))
    return base + 2.0 * rho1 / np.pi * series


# ---------------------------------------------------------------------------
# SphereModel — вспомогательные функции
# ---------------------------------------------------------------------------


def _zep(rho1: float, rho2: float, R: float, rp: float, re: float,
         cos_ep: float, n_max: int = 100) -> float:
    """Суммарное влияние сферы в точке (rp, re, cos_ep).

    Порт `Zep[ρ1, ρ2, R, rp, re, cosEP]` из Models.m::

        Zep = ρ1/(π·re) · Σ_{n=0..n_max}
              [(R/rp)^{n+1} · (R/re)^n · n(ρ2-ρ1)/((n+1)ρ2+n·ρ1) · Pn(cosEP)]

    Замечание: член n=0 всегда равен нулю (числитель n·(ρ2-ρ1)=0 при n=0),
    поэтому суммирование фактически начинается с n=1.
    """
    n_arr = np.arange(0, n_max + 1, dtype=float)                # [0, 1, ..., 100]
    geom = (R / rp) ** (n_arr + 1) * (R / re) ** n_arr          # (R/rp)^(n+1) · (R/re)^n
    denom = (n_arr + 1) * rho2 + n_arr * rho1                   # (n+1)·ρ2 + n·ρ1
    coeff = n_arr * (rho2 - rho1) / denom                       # n·(ρ2-ρ1)/denom  [n=0 → 0.0]
    legendre_vals = eval_legendre(n_arr.astype(int), cos_ep)    # Pn(cosEP) vectorised
    return (rho1 / (np.pi * re)) * float(np.dot(geom * coeff, legendre_vals))


def _sub_z(rho1: float, rho2: float, a: float, b: float,
           R: float, h: float, x: float, y: float,
           rm: float, ra: float, rn: float, rb: float) -> float:
    """SubZ — четыре вызова Zep с разными косинусами.

    Порт `SubZ[ρ1, ρ2, a, b, R, h, x, y, rm, ra, rn, rb]` из Models.m.
    """
    cos_am = (ra**2 + rm**2 - (a - b)**2) / (2.0 * ra * rm)
    cos_bm = (rb**2 + rm**2 - (a + b)**2) / (2.0 * rb * rm)
    cos_an = (ra**2 + rn**2 - (a + b)**2) / (2.0 * ra * rn)
    cos_bn = (rb**2 + rn**2 - (a - b)**2) / (2.0 * rb * rn)

    return (
        _zep(rho1, rho2, R, rm, ra, cos_am)
        - _zep(rho1, rho2, R, rn, ra, cos_an)
        + _zep(rho1, rho2, R, rn, rb, cos_bn)
        - _zep(rho1, rho2, R, rm, rb, cos_bm)
    )


def sphere_influence(rho1: float, rho2: float,
                     a: float, b: float, R: float, h: float,
                     x: float, y: float) -> float:
    """Поправка от шарообразного сердца к однослойному импедансу.

    Порт `SphereInfluence[ρ1, ρ2, a, b, R, h, x, y]` из Models.m.

    Параметры
    ---------
    rho1 : float   Удельное сопротивление тела (Ом·м).
    rho2 : float   Удельное сопротивление крови/сердца (Ом·м).
    a : float      Большая полуось (м).
    b : float      Малая полуось (м).
    R : float      Радиус сферы-сердца (м).
    h : float      Глубина залегания центра сферы от поверхности (м).
    x : float      Смещение центра сферы по оси x (м).
    y : float      Смещение центра сферы по оси y (м).

    Возвращает
    ----------
    float — поправочный импеданс (Ом).
    """
    Rh = R + h
    rm = np.sqrt(Rh**2 + (b - y)**2 + x**2)
    rn = np.sqrt(Rh**2 + (b + y)**2 + x**2)
    ra = np.sqrt(Rh**2 + (a - y)**2 + x**2)
    rb = np.sqrt(Rh**2 + (a + y)**2 + x**2)
    return _sub_z(rho1, rho2, a, b, R, h, x, y, rm, ra, rn, rb)


def sphere_model(rho1: float, rho2: float,
                 a: float, b: float, R: float, h: float,
                 x: float, y: float) -> float:
    """Полная модель импеданса с шарообразным сердцем.

    Порт `SphereModel[ρ1, ρ2, a, b, R, h, x, y]` из Models.m::

        Z = OneLayerModel(ρ1, a, b) + SphereInfluence(...)

    Параметры
    ---------
    rho1 : float   Удельное сопротивление тела (Ом·м).
    rho2 : float   Удельное сопротивление крови (Ом·м).
    a : float      Большая полуось (м).
    b : float      Малая полуось (м).
    R : float      Радиус сферы-сердца (м).
    h : float      Глубина центра (м).
    x : float      Смещение по x (м).
    y : float      Смещение по y (м).

    Возвращает
    ----------
    float — полный импеданс (Ом).
    """
    return one_layer_model(rho1, a, b) + sphere_influence(rho1, rho2, a, b, R, h, x, y)
