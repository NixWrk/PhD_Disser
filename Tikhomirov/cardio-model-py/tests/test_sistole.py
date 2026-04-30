"""Тесты построения систолического контура (SistoleFunction).

Фаза 4, коммит #29.
Покрывает: sistole_none, valve_point, valve_contour_move,
           sistole_valve_move, sistole_contour (диспетчер).
Нет Wolfram-эталона для выходных координат — проверяем инварианты:
  * первая/последняя точки зафиксированы/сдвинуты согласно методу
  * форма выходного массива
  * геометрические свойства (угол, расстояние)
"""

from __future__ import annotations

import numpy as np
import pytest

from cardio_model.sistole import (
    sistole_none,
    sistole_contour,
    sistole_valve_move,
    valve_contour_move,
    valve_point,
)
from cardio_model.data.patients import get_contour
from cardio_model.data.moves import edge_move_5pt
from cardio_model.data.mri import VALVE_MOVE


# ===========================================================================
# sistole_none
# ===========================================================================


class TestSistoleNone:
    def test_first_last_unchanged(self) -> None:
        """SistoleNone: первая и последняя точки остаются без изменений."""
        contour = get_contour("Ivan")
        move = edge_move_5pt("Ivan")
        result = sistole_none(contour, move)
        np.testing.assert_array_equal(result[0],  contour[0])
        np.testing.assert_array_equal(result[-1], contour[-1])

    def test_zero_move_unchanged(self) -> None:
        """Нулевое смещение → систолический контур совпадает с диастолическим."""
        contour = get_contour("Alex")
        move = np.zeros(5)
        result = sistole_none(contour, move)
        np.testing.assert_allclose(result, contour, atol=1e-10)

    def test_output_shape(self) -> None:
        """Форма не меняется: (7, 2) → (7, 2)."""
        for name in ["Ivan", "Alex", "Artem"]:
            contour = get_contour(name)
            result = sistole_none(contour, edge_move_5pt(name))
            assert result.shape == contour.shape, f"{name}: {result.shape}"

    def test_positive_move_increases_radius(self) -> None:
        """Положительное смещение внутренней точки увеличивает полярный радиус."""
        contour = get_contour("Ivan")
        move = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        result = sistole_none(contour, move)
        # Полярный радиус для внутренних точек
        inner_r_before = np.hypot(contour[1:-1, 0], contour[1:-1, 1])
        inner_r_after  = np.hypot(result[1:-1, 0],  result[1:-1, 1])
        assert np.all(inner_r_after > inner_r_before - 1e-10)

    @pytest.mark.parametrize("name", ["Ivan", "Alex", "Artem"])
    def test_edge_move_5pt_integrates(self, name: str) -> None:
        """edge_move_5pt (отрицательные значения) уменьшает размер контура."""
        contour = get_contour(name)
        move = edge_move_5pt(name)  # отрицательные → сжатие
        result = sistole_none(contour, move)
        # Средний полярный радиус внутренних точек уменьшается
        r_before = np.hypot(contour[1:-1, 0], contour[1:-1, 1]).mean()
        r_after  = np.hypot(result[1:-1, 0],  result[1:-1, 1]).mean()
        assert r_after < r_before


# ===========================================================================
# valve_point
# ===========================================================================


class TestValvePoint:
    def test_shift_distance(self) -> None:
        """Новые точки сдвинуты от исходных на valve_move."""
        p_first = np.array([0.0, 0.0])
        p_last  = np.array([10.0, 0.0])
        vm = 5.0
        nf, nl = valve_point(p_first, p_last, vm)
        assert abs(np.linalg.norm(nf - p_first) - vm) < 1e-10
        assert abs(np.linalg.norm(nl - p_last)  - vm) < 1e-10

    def test_parallel_shift(self) -> None:
        """Отрезок [pFirst, pLast] остаётся параллельным после сдвига."""
        p_first = np.array([0.0, 0.0])
        p_last  = np.array([4.0, 3.0])
        nf, nl = valve_point(p_first, p_last, 2.0)
        # direction before and after
        d_before = p_last  - p_first
        d_after  = nl - nf
        # vectors should be parallel (collinear):
        cross = d_before[0] * d_after[1] - d_before[1] * d_after[0]
        assert abs(cross) < 1e-10

    def test_perpendicular_to_valve_line(self) -> None:
        """Смещение перпендикулярно вектору (pFirst − pLast)."""
        p_first = np.array([0.0, 0.0])
        p_last  = np.array([6.0, 0.0])
        nf, _ = valve_point(p_first, p_last, 5.0)
        shift = nf - p_first
        valve_dir = p_last - p_first
        # dot product должен быть ≈ 0 (перпендикулярно)
        assert abs(np.dot(shift, valve_dir)) < 1e-10

    def test_coincident_raises(self) -> None:
        """Совпадающие точки → ValueError."""
        with pytest.raises(ValueError):
            valve_point(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 5.0)

    def test_symmetric_input(self) -> None:
        """Симметричный вход → оба конца сдвинуты одинаково."""
        p_first = np.array([-5.0, 0.0])
        p_last  = np.array([5.0,  0.0])
        nf, nl = valve_point(p_first, p_last, 3.0)
        # Сдвиг должен быть одинаков (параллельный перенос)
        np.testing.assert_allclose(nf - p_first, nl - p_last, atol=1e-10)


# ===========================================================================
# valve_contour_move
# ===========================================================================


class TestValveConturMove:
    def test_inner_points_unchanged(self) -> None:
        """Внутренние точки контура не изменяются."""
        contour = get_contour("Ivan")
        result = valve_contour_move(contour, 5.0)
        np.testing.assert_array_equal(result[1:-1], contour[1:-1])

    def test_endpoints_shifted(self) -> None:
        """Первая и последняя точки сдвигаются."""
        contour = get_contour("Ivan")
        result = valve_contour_move(contour, 5.0)
        # Точки изменились
        assert not np.allclose(result[0],  contour[0])
        assert not np.allclose(result[-1], contour[-1])

    def test_output_shape_preserved(self) -> None:
        for name in ["Ivan", "Alex", "Artem"]:
            c = get_contour(name)
            r = valve_contour_move(c, VALVE_MOVE[name])
            assert r.shape == c.shape

    def test_zero_valve_move_unchanged(self) -> None:
        """При нулевом смещении клапана контур не меняется."""
        contour = get_contour("Alex")
        result = valve_contour_move(contour, 0.0)
        np.testing.assert_allclose(result, contour, atol=1e-10)


# ===========================================================================
# sistole_valve_move
# ===========================================================================


class TestSistoleValveMove:
    def test_shape(self) -> None:
        for name in ["Ivan", "Alex", "Artem"]:
            contour = get_contour(name)
            move = edge_move_5pt(name)
            result = sistole_valve_move(contour, move, VALVE_MOVE[name])
            assert result.shape == contour.shape

    def test_zero_move_equals_valve_only(self) -> None:
        """Нулевое радиальное смещение → результат = valve_contour_move(diastole)."""
        contour = get_contour("Ivan")
        move = np.zeros(5)
        res_svm = sistole_valve_move(contour, move, 5.0)
        res_vcm = valve_contour_move(contour, 5.0)
        np.testing.assert_allclose(res_svm, res_vcm, atol=1e-10)

    def test_inner_radii_change(self) -> None:
        """Внутренние точки смещаются за счёт move."""
        contour = get_contour("Artem")
        move = edge_move_5pt("Artem")
        result = sistole_valve_move(contour, move, VALVE_MOVE["Artem"])
        # Внутренние точки систолы ≠ диастолы (move ≠ 0)
        assert not np.allclose(result[1:-1], contour[1:-1])


# ===========================================================================
# sistole_contour — диспетчер
# ===========================================================================


class TestSistoleContour:
    def test_none_method_equals_sistole_none(self) -> None:
        """Method='None' даёт то же, что sistole_none."""
        contour = get_contour("Ivan")
        move = edge_move_5pt("Ivan")
        r1 = sistole_contour(contour, move, 5.0, "None")
        r2 = sistole_none(contour, move)
        np.testing.assert_array_equal(r1, r2)

    def test_valve_move_method_equals_sistole_valve_move(self) -> None:
        """Method='ValveMove' даёт то же, что sistole_valve_move."""
        contour = get_contour("Ivan")
        move = edge_move_5pt("Ivan")
        vm = VALVE_MOVE["Ivan"]
        r1 = sistole_contour(contour, move, vm, "ValveMove")
        r2 = sistole_valve_move(contour, move, vm)
        np.testing.assert_array_equal(r1, r2)

    def test_unknown_method_raises(self) -> None:
        contour = get_contour("Ivan")
        with pytest.raises(ValueError, match="Unknown"):
            sistole_contour(contour, np.zeros(5), 5.0, "WeirdMethod")

    @pytest.mark.parametrize("method", ["None", "ValveMove"])
    @pytest.mark.parametrize("name", ["Ivan", "Alex", "Artem"])
    def test_all_methods_output_shape(self, name: str, method: str) -> None:
        contour = get_contour(name)
        move = edge_move_5pt(name)
        result = sistole_contour(contour, move, VALVE_MOVE[name], method)
        assert result.shape == contour.shape
