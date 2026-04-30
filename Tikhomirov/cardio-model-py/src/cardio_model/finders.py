"""Численный поиск параметров моделей (root-finding, табличный поиск).

Порт ``Kernel/core/ModelFinders.m``.

Все функции-«finders» в Mathematica используют табличный поиск
(``Partition[Table[...], 2, 1]`` + ``Select``), а не ``FindRoot``.
Это намеренно: модель немонотонна при малых R, и численные решатели
нестабильны. В Python реализован тот же подход: строим таблицу значений
функции на равномерной сетке и ищем первый интервал пересечения.

Исключение: ``find_ro_one`` — ``OneLayerModel`` линейна по ρ₁, поэтому
используется аналитическая формула (это совпадает с тем, что ``FindRoot``
даёт мгновенно).

Публичное API:
    find_ro_one(Zb, a, b)
    find_h_two(Zb, ro1, ro2, a, b)
    find_r_sphere(Zb, ro1, ro2, a, b, h, x, y)
    find_h_sphere(Zb, ro1, ro2, a, b, R, x, y)
    find_ro_sphere(Zb, ro2, a, b, R, h, x, y)
    find_dr_sphere(Zb, dZ, ro1, ro2, a, b, R, h, x, y)
    find_dr_sphere_dxdy(Zb, dZ, dx, dy, ro1, ro2, a, b, R, h, x, y)
    find_dr_sphere_fl(Zb, dZ, dro1, ro1, ro2, a, b, R, h, x, y)
    find_dr_sphere_fl_dxdy(Zb, dZ, dx, dy, dro1, ro1, ro2, a, b, R, h, x, y)
"""

from __future__ import annotations

import numpy as np

from cardio_model.models import (
    one_layer_model,
    sphere_model,
    two_layer_model,
)

__all__ = [
    "find_ro_one",
    "find_h_two",
    "find_r_sphere",
    "find_h_sphere",
    "find_ro_sphere",
    "find_dr_sphere",
    "find_dr_sphere_dxdy",
    "find_dr_sphere_fl",
    "find_dr_sphere_fl_dxdy",
]


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def _table_range(start: float, stop: float, step: float) -> np.ndarray:
    """Аналог ``{i, start, stop, step}`` из Mathematica.

    Использует ``np.linspace`` чтобы избежать накопления ошибок float при
    маленьком шаге (``np.arange`` может дать на 1 элемент больше/меньше).
    """
    n = round((stop - start) / step) + 1
    return np.linspace(start, stop, n)


def _tabular_crossing(
    func,
    start: float,
    stop: float,
    step: float,
    target: float,
    bias: float = 0.0,
    increasing: bool = True,
) -> float:
    """Первое пересечение ``func(i) + bias == target`` на равномерной сетке.

    Аналог Mathematica-шаблона::

        Partition[Table[{i, func(i) + bias}, {i, start, stop, step}], 2, 1]
        → Select первой пары с нужным знаком перехода
        → return i-значение левого конца пары

    Parameters
    ----------
    func      : callable(float) -> float
    start/stop/step : границы и шаг сетки
    target    : искомое значение
    bias      : добавляется к func(i) (как "−0.0001" в оригинале)
    increasing: True — ищем пересечение снизу вверх (<=, >=);
                False — сверху вниз (>=, <=)
    """
    i_vals = _table_range(start, stop, step)
    f_vals = np.fromiter(
        (func(iv) + bias for iv in i_vals), dtype=float, count=len(i_vals)
    )
    if increasing:
        mask = (f_vals[:-1] <= target) & (f_vals[1:] >= target)
    else:
        mask = (f_vals[:-1] >= target) & (f_vals[1:] <= target)
    idx = np.argmax(mask)
    if not mask[idx]:
        direction = "↑" if increasing else "↓"
        raise ValueError(
            f"No crossing {direction} of target={target} found in "
            f"[{start}, {stop}] (step={step}, bias={bias})"
        )
    return float(i_vals[idx])


# ---------------------------------------------------------------------------
# OneLayerModel finders
# ---------------------------------------------------------------------------


def find_ro_one(Zb: float, a: float, b: float) -> float:
    """Найти ρ₁ для ``OneLayerModel`` (аналитически).

    Порт ``FindRoOne[Zb, a, b]`` из ModelFinders.m.

    ``OneLayerModel = ρ₁·2b / (π·(a²−b²))``
    →  ρ₁ = Zb·π·(a²−b²) / (2b).
    """
    return float(Zb * np.pi * (a ** 2 - b ** 2) / (2.0 * b))


# ---------------------------------------------------------------------------
# TwoLayerModel finders
# ---------------------------------------------------------------------------


def find_h_two(Zb: float, ro1: float, ro2: float, a: float, b: float) -> float:
    """Найти h для ``TwoLayerModel`` (табличный поиск).

    Порт ``FindHTwo[Zb, ro1, ro2, a, b]``.
    Сетка: h ∈ [0, 0.05], шаг 0.0005; пересечение снизу вверх.
    """
    return _tabular_crossing(
        lambda i: two_layer_model(ro1, ro2, i, a, b),
        start=0.0, stop=0.05, step=0.0005,
        target=Zb, increasing=True,
    )


# ---------------------------------------------------------------------------
# SphereModel finders
# ---------------------------------------------------------------------------


def find_r_sphere(
    Zb: float, ro1: float, ro2: float,
    a: float, b: float, h: float, x: float, y: float,
) -> float:
    """Найти R для ``SphereModel`` (табличный поиск).

    Порт ``FindRSphere[Zb, ro1, ro2, a, b, h, x, y]``.
    Сетка: R ∈ [0, 0.1], шаг 0.0005; пересечение **сверху вниз**.
    Bias +0.0001 — как в оригинале: ``SphereModel[...]+0.0001``.
    """
    return _tabular_crossing(
        lambda i: sphere_model(ro1, ro2, a, b, i, h, x, y),
        start=0.0, stop=0.1, step=0.0005,
        target=Zb, bias=+0.0001, increasing=False,
    )


def find_h_sphere(
    Zb: float, ro1: float, ro2: float,
    a: float, b: float, R: float, x: float, y: float,
) -> float:
    """Найти h для ``SphereModel`` (табличный поиск).

    Порт ``FindHSphere[Zb, ro1, ro2, a, b, R, x, y]``.
    Сетка: h ∈ [0, 0.05], шаг 0.0005; пересечение снизу вверх.
    """
    return _tabular_crossing(
        lambda i: sphere_model(ro1, ro2, a, b, R, i, x, y),
        start=0.0, stop=0.05, step=0.0005,
        target=Zb, increasing=True,
    )


def find_ro_sphere(
    Zb: float, ro2: float,
    a: float, b: float, R: float, h: float, x: float, y: float,
) -> float:
    """Найти ρ₁ для ``SphereModel`` (табличный поиск).

    Порт ``FindRoSphere[Zb, ro2, a, b, R, h, x, y]``.
    Сетка: ρ₁ ∈ [0.1, 50], шаг 0.05; пересечение снизу вверх.
    """
    return _tabular_crossing(
        lambda i: sphere_model(i, ro2, a, b, R, h, x, y),
        start=0.1, stop=50.0, step=0.05,
        target=Zb, increasing=True,
    )


def find_dr_sphere(
    Zb: float, dZ: float,
    ro1: float, ro2: float,
    a: float, b: float, R: float, h: float, x: float, y: float,
) -> float:
    """Найти dR — изменение радиуса сферы по двум импедансам.

    Порт ``FindDRSphere[Zb, dZ, ro1, ro2, a, b, R, h, x, y]``.

    Если ro1==0, автоматически ищет ρ₁ через ``find_ro_sphere``.
    Сетка: dR ∈ [0, 0.02], шаг 0.0001; bias −0.0001.

    Ищет первое i, при котором::

        -SphereModel(R) + SphereModel(R−i) − 0.0001  переходит через  dZ
    """
    real_ro1 = ro1 if ro1 != 0.0 else find_ro_sphere(Zb, ro2, a, b, R, h, x, y)
    base = sphere_model(real_ro1, ro2, a, b, R, h, x, y)
    return _tabular_crossing(
        lambda i: -base + sphere_model(real_ro1, ro2, a, b, R - i, h, x, y),
        start=0.0, stop=0.02, step=0.0001,
        target=dZ, bias=-0.0001, increasing=True,
    )


def find_dr_sphere_dxdy(
    Zb: float, dZ: float,
    dx: float, dy: float,
    ro1: float, ro2: float,
    a: float, b: float, R: float, h: float, x: float, y: float,
) -> float:
    """Найти dR с учётом смещения центра сферы по x и y.

    Порт ``FindDRSphereDxDy[Zb, dZ, dx, dy, ro1, ro2, a, b, R, h, x, y]``.
    Сетка: i ∈ [−0.02, 0.02], шаг 0.0001 (расширен для dx/dy).
    """
    real_ro1 = ro1 if ro1 != 0.0 else find_ro_sphere(Zb, ro2, a, b, R, h, x, y)
    base = sphere_model(real_ro1, ro2, a, b, R, h, x, y)
    return _tabular_crossing(
        lambda i: -base + sphere_model(real_ro1, ro2, a, b, R - i, h, x + dx, y + dy),
        start=-0.02, stop=0.02, step=0.0001,
        target=dZ, bias=-0.0001, increasing=True,
    )


def find_dr_sphere_fl(
    Zb: float, dZ: float,
    dro1: float,
    ro1: float, ro2: float,
    a: float, b: float, R: float, h: float, x: float, y: float,
) -> float:
    """Найти dR с учётом первого (жирового) слоя.

    Порт ``FindDRSphereFl[Zb, dZ, dro1, ro1, ro2, a, b, R, h, x, y]``.
    Отличие от ``find_dr_sphere``: второй вызов модели использует ``ρ₁−dρ₁``.
    Сетка: i ∈ [0, 0.02], шаг 0.0001; bias −0.0001.
    """
    real_ro1 = ro1 if ro1 != 0.0 else find_ro_sphere(Zb, ro2, a, b, R, h, x, y)
    base = sphere_model(real_ro1, ro2, a, b, R, h, x, y)
    return _tabular_crossing(
        lambda i: -base + sphere_model(real_ro1 - dro1, ro2, a, b, R - i, h, x, y),
        start=0.0, stop=0.02, step=0.0001,
        target=dZ, bias=-0.0001, increasing=True,
    )


def find_dr_sphere_fl_dxdy(
    Zb: float, dZ: float,
    dx: float, dy: float,
    dro1: float,
    ro1: float, ro2: float,
    a: float, b: float, R: float, h: float, x: float, y: float,
) -> float:
    """Найти dR с учётом первого слоя и смещения центра сферы.

    Порт ``FindDRSphereFlDxDy[Zb, dZ, dx, dy, dro1, ro1, ro2, a, b, R, h, x, y]``.
    Сетка: i ∈ [−0.02, 0.02], шаг 0.0001.
    """
    real_ro1 = ro1 if ro1 != 0.0 else find_ro_sphere(Zb, ro2, a, b, R, h, x, y)
    base = sphere_model(real_ro1, ro2, a, b, R, h, x, y)
    return _tabular_crossing(
        lambda i: -base + sphere_model(real_ro1 - dro1, ro2, a, b, R - i, h, x + dx, y + dy),
        start=-0.02, stop=0.02, step=0.0001,
        target=dZ, bias=-0.0001, increasing=True,
    )
