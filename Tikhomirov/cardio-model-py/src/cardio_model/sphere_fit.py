"""Подгонка эквивалентной сферы по 2D-контуру сердца (Nelder-Mead).

Порт ``Kernel/part1/SphereMovingFunction.m``.

Алгоритм: минимизировать сумму квадратов отклонений расстояний от точек
контура до предполагаемого центра сферы — относительно заданного радиуса.

Публичное API:
    req_for_heart(vmri) -> float              # радиус экв. сферы из объёма МРТ
    mass_center_contour(contour) -> np.ndarray  # центр масс полигона
    circle_radius_by_contour_sqr(contour) -> float  # радиус из площади
    eq_sph_nm_center_radius(contour) -> tuple  # Nelder-Mead: центр + радиус
    eq_sph_nm_radius(contour, center) -> tuple  # Nelder-Mead: только радиус
    eq_sph_nm_center(contour, radius) -> tuple  # Nelder-Mead: только центр
    equal_sphere(contour, method, radius_if_need) -> tuple
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, Bounds
from shapely.geometry import Polygon

__all__ = [
    "req_for_heart",
    "mass_center_contour",
    "circle_radius_by_contour_sqr",
    "eq_sph_nm_center_radius",
    "eq_sph_nm_radius",
    "eq_sph_nm_center",
    "equal_sphere",
    "sum_delta",
]


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def _delta_n(point1: np.ndarray, point2: np.ndarray, req: float) -> float:
    """Отклонение расстояния |point1−point2| от req."""
    return abs(req - float(np.linalg.norm(point1 - point2)))


def sum_delta(points: np.ndarray, center: np.ndarray, eq: float) -> float:
    """Сумма квадратов отклонений расстояний от center до каждой точки.

    Порт ``SumDelta[points, center, eq]``::

        Sum_i (|points[i] - center| - eq)^2
    """
    dists = np.linalg.norm(points - center, axis=1)
    return float(np.sum((dists - eq) ** 2))


# ---------------------------------------------------------------------------
# Радиус сферы из объёма МРТ
# ---------------------------------------------------------------------------


def req_for_heart(vmri: float) -> float:
    """Радиус эквивалентной сферы, рассчитанный из объёма МРТ.

    Порт ``ReqForHeart[Vmri]``::

        r = CubeRoot[2·3·Vmri / (0.6·4·π)] · 10

    Parameters
    ----------
    vmri : float — объём из МРТ (мл).

    Returns
    -------
    float — радиус в мм.
    """
    return float(np.cbrt(2.0 * 3.0 * vmri / (0.6 * 4.0 * np.pi)) * 10.0)


# ---------------------------------------------------------------------------
# Центр масс и радиус из площади
# ---------------------------------------------------------------------------


def mass_center_contour(contour: np.ndarray) -> np.ndarray:
    """Центр масс 2D-полигона.

    Порт ``MassCenterContour[contour]`` через ``RegionCentroid[Polygon[contour]]``.

    Parameters
    ----------
    contour : (N, 2) float — точки контура сердца (не обязательно замкнутые).

    Returns
    -------
    (2,) float — координаты центра масс.
    """
    poly = Polygon(contour)
    c = poly.centroid
    return np.array([c.x, c.y])


def circle_radius_by_contour_sqr(contour: np.ndarray) -> float:
    """Радиус окружности, эквивалентной контуру по площади.

    Порт ``CircleRadiusByContourSqr[contour]``::

        r = sqrt(Area(Polygon(contour)) / π)
    """
    poly = Polygon(contour)
    return float(np.sqrt(poly.area / np.pi))


# ---------------------------------------------------------------------------
# Nelder-Mead оптимизация
# ---------------------------------------------------------------------------


def eq_sph_nm_center_radius(
    contour: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Найти центр и радиус эквивалентной сферы (Nelder-Mead).

    Порт ``EqSphNMCenterRadius[contour]``::

        NMinimize[{SumDelta[contour, {x,y}, r], -40<x<40, -60<y<40, 20<r<60},
                  {x, y, r}, Method→"NelderMead"]

    Returns
    -------
    (center: (2,) float, radius: float)
    """
    pts = np.asarray(contour, dtype=float)

    def objective(xyr: np.ndarray) -> float:
        return sum_delta(pts, xyr[:2], xyr[2])

    # Начальное приближение: центр масс + средний радиус
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    r0 = np.mean(np.linalg.norm(pts - [cx, cy], axis=1))
    x0 = np.array([cx, cy, r0])

    bounds = Bounds([-40.0, -60.0, 20.0], [40.0, 40.0, 60.0])
    res = minimize(
        objective, x0,
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 50_000},
    )
    return np.array([res.x[0], res.x[1]]), float(res.x[2])


def eq_sph_nm_radius(
    contour: np.ndarray,
    center_point: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Найти радиус при известном центре (Nelder-Mead).

    Порт ``EqSphNMRadius[contour, centerPoint]``::

        NMinimize[{SumDelta[contour, centerPoint, r], 0<r<60}, r, Method→"NelderMead"]

    Returns
    -------
    (center: (2,) float, radius: float)
    """
    pts = np.asarray(contour, dtype=float)
    center = np.asarray(center_point, dtype=float)

    dists = np.linalg.norm(pts - center, axis=1)
    r0 = float(dists.mean())

    res = minimize(
        lambda r: sum_delta(pts, center, r[0]),
        [r0],
        method="Nelder-Mead",
        bounds=Bounds([0.0], [60.0]),
        options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 10_000},
    )
    return center.copy(), float(res.x[0])


def eq_sph_nm_center(
    contour: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, float]:
    """Найти центр при известном радиусе (Nelder-Mead).

    Порт ``EqSphNMCenter[contour, radius]``::

        NMinimize[{SumDelta[contour, {x,y}, radius], -40<x<40, -40<y<40},
                  {x,y}, Method→"NelderMead"]

    Returns
    -------
    (center: (2,) float, radius: float)
    """
    pts = np.asarray(contour, dtype=float)

    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    bounds = Bounds([-40.0, -40.0], [40.0, 40.0])

    res = minimize(
        lambda xy: sum_delta(pts, xy, radius),
        [cx, cy],
        method="Nelder-Mead",
        bounds=bounds,
        options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 10_000},
    )
    return np.array([res.x[0], res.x[1]]), float(radius)


# ---------------------------------------------------------------------------
# Диспетчер EqualSphere
# ---------------------------------------------------------------------------

_EQUAL_SPHERE_METHODS = {
    "Use_Contour",
    "Use_Contour_Radius",
    "Use_Contour_Center",
    "Use_Contour_MassCenter",
    "Use_Contour_Verb",
    "Use_Contour_Radius_Verb",
    "Use_Contour_Center_Verb",
    "Use_Contour_MassCenter_Verb",
}


def equal_sphere(
    contour: np.ndarray,
    method: str = "Use_Contour",
    radius_if_need: float | None = None,
) -> tuple[np.ndarray, float]:
    """Диспетчер методов подгонки эквивалентной сферы.

    Порт ``EqualSphere[contour, method, radiusIfNeed]``.

    Поддерживаемые методы
    ---------------------
    ``"Use_Contour"``          — NM: центр + радиус одновременно.
    ``"Use_Contour_Radius"``   — NM: только радиус; центр = MassCenter.
    ``"Use_Contour_Center"``   — NM: только центр; радиус = radius_if_need.
    ``"Use_Contour_MassCenter"`` — центр = MassCenter; радиус из площади.

    Returns
    -------
    (center: (2,) float, radius: float)
    """
    contour = np.asarray(contour, dtype=float)
    base = method.replace("_Verb", "")

    if base == "Use_Contour":
        return eq_sph_nm_center_radius(contour)
    if base == "Use_Contour_Radius":
        center = mass_center_contour(contour)
        return eq_sph_nm_radius(contour, center)
    if base == "Use_Contour_Center":
        if radius_if_need is None:
            raise ValueError("radius_if_need must be provided for Use_Contour_Center")
        return eq_sph_nm_center(contour, float(radius_if_need))
    if base == "Use_Contour_MassCenter":
        center = mass_center_contour(contour)
        r = circle_radius_by_contour_sqr(contour)
        return center, r

    raise ValueError(
        f"Unknown EqualSphere method: {method!r}. "
        f"Use one of: {sorted(_EQUAL_SPHERE_METHODS)}"
    )
