"""Тесты подгонки эквивалентной сферы (SphereMovingFunction).

Фаза 4, коммит #28.
Покрывает: req_for_heart, sum_delta, mass_center_contour,
           circle_radius_by_contour_sqr, equal_sphere (все методы),
           eq_sph_nm_center_radius, eq_sph_nm_radius, eq_sph_nm_center.
Эталонные значения NM из ``tests/reference_values.json`` (допуск ±0.5).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cardio_model.sphere_fit import (
    circle_radius_by_contour_sqr,
    eq_sph_nm_center,
    eq_sph_nm_center_radius,
    eq_sph_nm_radius,
    equal_sphere,
    mass_center_contour,
    req_for_heart,
    sum_delta,
)
from cardio_model.data.patients import get_contour

_REF_PATH = Path(__file__).parent / "reference_values.json"
with _REF_PATH.open(encoding="utf-8") as _fh:
    _REF = json.load(_fh)

# Эталонные NM-результаты (стохастичен, допуск ±0.5)
_NM_REF = _REF["computed"]["eq_sphere_nm"]


# ===========================================================================
# ReqForHeart
# ===========================================================================


class TestReqForHeart:
    def test_deterministic(self) -> None:
        """Формула детерминирована — два вызова дают одинаковый результат."""
        assert req_for_heart(100.0) == req_for_heart(100.0)

    def test_formula(self) -> None:
        """cbrt(2·3·Vmri / (0.6·4·π))·10 — точная проверка."""
        vmri = 100.0
        expected = float(np.cbrt(2.0 * 3.0 * vmri / (0.6 * 4.0 * np.pi)) * 10.0)
        assert abs(req_for_heart(vmri) - expected) < 1e-12

    def test_positive(self) -> None:
        assert req_for_heart(50.0) > 0

    def test_scales_with_volume(self) -> None:
        """Удвоение объёма → радиус умножается на cbrt(2)."""
        r1 = req_for_heart(100.0)
        r2 = req_for_heart(200.0)
        assert abs(r2 / r1 - np.cbrt(2.0)) < 1e-10


# ===========================================================================
# SumDelta
# ===========================================================================


class TestSumDelta:
    def test_zero_when_all_points_on_sphere(self) -> None:
        """Если все точки ровно на расстоянии eq от center → sum_delta = 0."""
        theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        r = 10.0
        points = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        center = np.array([0.0, 0.0])
        assert sum_delta(points, center, r) < 1e-20

    def test_nonzero_when_off(self) -> None:
        """Точки не на сфере → сумма > 0."""
        pts = np.array([[5.0, 0.0], [0.0, 5.0], [-3.0, 4.0]])
        center = np.array([0.0, 0.0])
        # req = 3 (не совпадает с расстояниями 5, 5, 5)
        assert sum_delta(pts, center, 3.0) > 0

    def test_exact_value(self) -> None:
        """Точное значение: одна точка на расстоянии 5, req=3 → (5-3)²=4."""
        pts = np.array([[3.0, 4.0]])  # dist = 5
        center = np.array([0.0, 0.0])
        assert abs(sum_delta(pts, center, 3.0) - 4.0) < 1e-12

    def test_symmetric_pairs(self) -> None:
        """Симметричные точки → sum_delta ≥ 0."""
        pts = np.array([[10.0, 0.0], [-10.0, 0.0], [0.0, 10.0], [0.0, -10.0]])
        center = np.array([0.0, 0.0])
        result = sum_delta(pts, center, 8.0)
        # (10-8)^2 * 4 = 16
        assert abs(result - 16.0) < 1e-12


# ===========================================================================
# MassCenterContour
# ===========================================================================


class TestMassCenterContour:
    def test_square_center(self) -> None:
        """Квадрат → центроид в геометрическом центре."""
        square = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
        c = mass_center_contour(square)
        np.testing.assert_allclose(c, [2.0, 2.0], atol=1e-10)

    def test_equilateral_triangle(self) -> None:
        """Равносторонний треугольник → центроид на пересечении медиан."""
        tri = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 3.0 * np.sqrt(3)]])
        c = mass_center_contour(tri)
        np.testing.assert_allclose(c, [3.0, np.sqrt(3)], atol=1e-10)

    def test_output_shape(self) -> None:
        contour = get_contour("Ivan")
        c = mass_center_contour(contour)
        assert c.shape == (2,)

    def test_inside_bounding_box(self) -> None:
        """Центроид находится внутри bounding box контура."""
        contour = get_contour("Artem")
        c = mass_center_contour(contour)
        assert contour[:, 0].min() <= c[0] <= contour[:, 0].max()
        assert contour[:, 1].min() <= c[1] <= contour[:, 1].max()


# ===========================================================================
# CircleRadiusByContourSqr
# ===========================================================================


class TestCircleRadiusByContourSqr:
    def test_square(self) -> None:
        """Квадрат 4×4: площадь=16 → r = sqrt(16/π)."""
        square = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
        r = circle_radius_by_contour_sqr(square)
        expected = np.sqrt(16.0 / np.pi)
        assert abs(r - expected) < 1e-10

    def test_circle_approximation(self) -> None:
        """Для почти-круга с радиусом R результат ≈ R."""
        theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        R = 15.0
        pts = R * np.column_stack([np.cos(theta), np.sin(theta)])
        r = circle_radius_by_contour_sqr(pts)
        assert abs(r - R) < 0.1

    def test_positive(self) -> None:
        assert circle_radius_by_contour_sqr(get_contour("Ivan")) > 0


# ===========================================================================
# EqSphNMCenterRadius (стохастичен: допуск ±0.5)
# ===========================================================================


class TestEqSphNMCenterRadius:
    @pytest.mark.parametrize("name", ["Ivan", "Alex", "Artem"])
    def test_against_wolfram(self, name: str) -> None:
        """eq_sph_nm_center_radius ≈ Wolfram EqSphNMCenterRadius (допуск ±0.5)."""
        contour = get_contour(name)
        center, radius = eq_sph_nm_center_radius(contour)
        ref = _NM_REF[name]
        assert abs(center[0] - ref["center"][0]) <= 0.5, (
            f"{name}: center[0]={center[0]:.3f} vs {ref['center'][0]:.3f}"
        )
        assert abs(center[1] - ref["center"][1]) <= 0.5, (
            f"{name}: center[1]={center[1]:.3f} vs {ref['center'][1]:.3f}"
        )
        assert abs(radius - ref["radius"]) <= 0.5, (
            f"{name}: radius={radius:.3f} vs {ref['radius']:.3f}"
        )

    def test_output_types(self) -> None:
        center, radius = eq_sph_nm_center_radius(get_contour("Ivan"))
        assert center.shape == (2,)
        assert isinstance(radius, float)

    def test_radius_in_bounds(self) -> None:
        """Радиус должен быть в диапазоне [20, 60]."""
        _, r = eq_sph_nm_center_radius(get_contour("Artem"))
        assert 20.0 <= r <= 60.0


# ===========================================================================
# EqSphNMRadius
# ===========================================================================


class TestEqSphNMRadius:
    def test_returns_given_center(self) -> None:
        """center_point передаётся «как есть» — возвращается без изменений."""
        contour = get_contour("Ivan")
        center_in = mass_center_contour(contour)
        center_out, _ = eq_sph_nm_radius(contour, center_in)
        np.testing.assert_allclose(center_out, center_in, atol=1e-10)

    def test_radius_positive(self) -> None:
        contour = get_contour("Ivan")
        center = mass_center_contour(contour)
        _, r = eq_sph_nm_radius(contour, center)
        assert r > 0

    def test_radius_reduces_sum_delta(self) -> None:
        """Найденный радиус даёт меньший sum_delta, чем нулевой."""
        contour = get_contour("Alex")
        pts = np.asarray(contour, dtype=float)
        center = mass_center_contour(contour)
        _, r_opt = eq_sph_nm_radius(contour, center)
        assert sum_delta(pts, center, r_opt) < sum_delta(pts, center, 1.0)


# ===========================================================================
# EqSphNMCenter
# ===========================================================================


class TestEqSphNMCenter:
    def test_returns_given_radius(self) -> None:
        """Радиус radius_if_need возвращается без изменений."""
        contour = get_contour("Ivan")
        ref = _NM_REF["Ivan"]
        _, r_out = eq_sph_nm_center(contour, ref["radius"])
        assert abs(r_out - ref["radius"]) < 1e-10

    def test_center_reduces_sum_delta(self) -> None:
        """Найденный центр даёт меньший sum_delta, чем начало координат."""
        contour = get_contour("Artem")
        pts = np.asarray(contour, dtype=float)
        radius = _NM_REF["Artem"]["radius"]
        center_opt, _ = eq_sph_nm_center(contour, radius)
        assert sum_delta(pts, center_opt, radius) < sum_delta(pts, np.zeros(2), radius)

    def test_output_shape(self) -> None:
        contour = get_contour("Alex")
        center, radius = eq_sph_nm_center(contour, 50.0)
        assert center.shape == (2,)
        assert isinstance(radius, float)


# ===========================================================================
# EqualSphere — диспетчер
# ===========================================================================


class TestEqualSphere:
    def test_use_contour_returns_pair(self) -> None:
        contour = get_contour("Ivan")
        center, r = equal_sphere(contour, "Use_Contour")
        assert center.shape == (2,)
        assert isinstance(r, float)

    def test_use_contour_radius(self) -> None:
        """Use_Contour_Radius: NM только по радиусу, центр = MassCenter."""
        contour = get_contour("Ivan")
        center_mass = mass_center_contour(contour)
        center, r = equal_sphere(contour, "Use_Contour_Radius")
        np.testing.assert_allclose(center, center_mass, atol=1e-10)
        assert r > 0

    def test_use_contour_center(self) -> None:
        """Use_Contour_Center: NM только по центру, радиус = radius_if_need."""
        contour = get_contour("Ivan")
        r_given = 45.0
        center, r = equal_sphere(contour, "Use_Contour_Center", radius_if_need=r_given)
        assert abs(r - r_given) < 1e-10
        assert center.shape == (2,)

    def test_use_contour_center_requires_radius(self) -> None:
        """Use_Contour_Center без radius_if_need → ValueError."""
        contour = get_contour("Ivan")
        with pytest.raises(ValueError, match="radius_if_need"):
            equal_sphere(contour, "Use_Contour_Center")

    def test_use_contour_mass_center_deterministic(self) -> None:
        """Use_Contour_MassCenter: детерминирован — два вызова одинаковы."""
        contour = get_contour("Artem")
        c1, r1 = equal_sphere(contour, "Use_Contour_MassCenter")
        c2, r2 = equal_sphere(contour, "Use_Contour_MassCenter")
        np.testing.assert_array_equal(c1, c2)
        assert r1 == r2

    def test_use_contour_mass_center_values(self) -> None:
        """Use_Contour_MassCenter: centre = MassCenter, r = sqrt(Area/π)."""
        contour = get_contour("Ivan")
        center, r = equal_sphere(contour, "Use_Contour_MassCenter")
        np.testing.assert_allclose(center, mass_center_contour(contour), atol=1e-10)
        assert abs(r - circle_radius_by_contour_sqr(contour)) < 1e-10

    def test_verb_suffix_ignored(self) -> None:
        """_Verb суффикс принимается без ошибки."""
        contour = get_contour("Ivan")
        center, r = equal_sphere(contour, "Use_Contour_MassCenter_Verb")
        # должен работать как Use_Contour_MassCenter
        np.testing.assert_allclose(center, mass_center_contour(contour), atol=1e-10)

    def test_unknown_method_raises(self) -> None:
        contour = get_contour("Ivan")
        with pytest.raises(ValueError, match="Unknown"):
            equal_sphere(contour, "Use_Something_Weird")
