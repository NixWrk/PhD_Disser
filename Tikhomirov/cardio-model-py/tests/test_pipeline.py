"""Тесты верхнеуровневого пайплайна (EqualSphereMoveModellingLib).

Фаза 5, коммит #37.
Покрывает: dxdy_find, h_contour_dxdy_for_each_channel,
           equal_sphere_move_modelling, radial_evaluation, total_modelling.

Нет Wolfram-эталона для конечного SV — проверяем:
  * форму и типы выходных данных
  * геометрические инварианты dxdy_find
  * детерминированность (Use_Contour_MassCenter)
  * структуру TotalModelling (2×2×3)
  * знак/порядок величин для SV (должен быть мал, т.к. импедансный dR мал)
"""

from __future__ import annotations

import numpy as np
import pytest

from cardio_model.pipeline import (
    dxdy_find,
    equal_sphere_move_modelling,
    h_contour_dxdy_for_each_channel,
    radial_evaluation,
    total_modelling,
)
from cardio_model.data.patients import get_contour


# ===========================================================================
# dxdy_find
# ===========================================================================


class TestDxDyFind:
    def test_zero_center_returns_zero(self) -> None:
        """Нулевое смещение центра → {0, 0}."""
        pt = np.array([30.0, -10.0])
        result = dxdy_find(pt, np.array([0.0, 0.0]))
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_output_shape(self) -> None:
        result = dxdy_find(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        assert result.shape == (2,)

    def test_zero_point1_returns_zero(self) -> None:
        """Нулевой point1 → {0, 0} (неопределённый угол)."""
        result = dxdy_find(np.array([0.0, 0.0]), np.array([3.0, 4.0]))
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_collinear_angle_zero(self) -> None:
        """Коллинеарные векторы → angle=0 → dy=0, |dx|=|point2|."""
        pt1 = np.array([1.0, 0.0])
        pt2 = np.array([3.0, 0.0])
        result = dxdy_find(pt1, pt2)
        l = np.linalg.norm(pt2)
        # angle = 0 → {-l*cos(0), l*sin(0)} = {-l, 0}
        assert abs(result[0] - (-l)) < 1e-10
        assert abs(result[1]) < 1e-10

    def test_perpendicular_angle_90(self) -> None:
        """Перпендикулярные векторы → angle=π/2 → dx=0, |dy|=|point2|."""
        pt1 = np.array([1.0, 0.0])
        pt2 = np.array([0.0, 2.0])
        result = dxdy_find(pt1, pt2)
        l = np.linalg.norm(pt2)
        # angle = π/2 → {-l*cos(π/2), l*sin(π/2)} = {0, l}
        assert abs(result[0]) < 1e-10
        assert abs(result[1] - l) < 1e-10

    def test_magnitude_is_norm_of_point2(self) -> None:
        """Норма результата = норма point2 (т.к. это проекция на ортонормированный базис)."""
        pt1 = np.array([3.0, 4.0])
        pt2 = np.array([1.0, 2.0])
        result = dxdy_find(pt1, pt2)
        # |result| = sqrt(dx² + dy²)
        # dx = -l*cos(a), dy = l*sin(a), l = |pt2|
        # |result|² = l²cos²(a) + l²sin²(a) = l²
        assert abs(np.linalg.norm(result) - np.linalg.norm(pt2)) < 1e-10


# ===========================================================================
# h_contour_dxdy_for_each_channel
# ===========================================================================


class TestHContourDxDyForEachChannel:
    def test_shape(self) -> None:
        """Форма результата: (5, 2) для каждого пациента."""
        for name in ["Ivan", "Alex", "Artem"]:
            result = h_contour_dxdy_for_each_channel(name, np.array([1.0, 0.5]))
            assert result.shape == (5, 2), f"{name}: {result.shape}"

    def test_zero_center_move_gives_zeros(self) -> None:
        """Нулевое смещение центра → нулевые dxdy для всех каналов."""
        result = h_contour_dxdy_for_each_channel("Ivan", np.zeros(2))
        np.testing.assert_array_equal(result, np.zeros((5, 2)))

    def test_uses_inner_points(self) -> None:
        """Должно использовать 5 внутренних точек контура."""
        name = "Artem"
        contour = get_contour(name)
        inner = contour[1:-1]
        center_move = np.array([2.0, 1.0])
        result = h_contour_dxdy_for_each_channel(name, center_move)
        # Ручная проверка первой строки
        expected_row0 = dxdy_find(inner[0], center_move)
        np.testing.assert_allclose(result[0], expected_row0, atol=1e-12)


# ===========================================================================
# equal_sphere_move_modelling
# ===========================================================================


class TestEqualSphereMoveModelling:
    def test_output_shape(self) -> None:
        """Выход: (2,) float — смещение центра в мм."""
        result = equal_sphere_move_modelling("Ivan", "None", "Use_Contour_MassCenter")
        assert result.shape == (2,)

    def test_deterministic_mass_center(self) -> None:
        """Use_Contour_MassCenter детерминирован → два вызова дают одинаковый результат."""
        r1 = equal_sphere_move_modelling("Ivan", "None", "Use_Contour_MassCenter")
        r2 = equal_sphere_move_modelling("Ivan", "None", "Use_Contour_MassCenter")
        np.testing.assert_array_equal(r1, r2)

    @pytest.mark.parametrize("method", ["None", "ValveMove"])
    def test_valve_vs_none_different(self, method: str) -> None:
        """Use_Contour_MassCenter + ValveMove даёт другой центр, чем без движения клапана."""
        r_none  = equal_sphere_move_modelling("Ivan", "None",      "Use_Contour_MassCenter")
        r_valve = equal_sphere_move_modelling("Ivan", "ValveMove", "Use_Contour_MassCenter")
        # Хотя бы одна компонента отличается
        assert not np.allclose(r_none, r_valve)

    def test_use_atrium_shape(self) -> None:
        """use_atrium=True тоже возвращает (2,) float."""
        result = equal_sphere_move_modelling(
            "Ivan", "None", "Use_Contour_MassCenter", use_atrium=True
        )
        assert result.shape == (2,)


# ===========================================================================
# radial_evaluation
# ===========================================================================


class TestRadialEvaluation:
    @pytest.mark.parametrize("name", ["Ivan", "Alex", "Artem"])
    @pytest.mark.parametrize("sistole_method", ["None", "ValveMove"])
    def test_returns_float(self, name: str, sistole_method: str) -> None:
        """radial_evaluation возвращает float."""
        sv = radial_evaluation(
            name, sistole_method, "Use_Contour_MassCenter", "None"
        )
        assert isinstance(sv, float)

    @pytest.mark.parametrize("rad_method", ["None", "FirstLayer"])
    def test_sv_reasonable_range(self, rad_method: str) -> None:
        """SV должен быть в разумном диапазоне (не сотни литров)."""
        sv = radial_evaluation("Ivan", "None", "Use_Contour_MassCenter", rad_method)
        # dR порядка 0.001–0.01 м → смещение контура ~1–10 мм → разница объёмов небольшая
        assert abs(sv) < 500.0, f"{rad_method}: SV={sv}"

    def test_unknown_rad_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            radial_evaluation("Ivan", "None", "Use_Contour_MassCenter", "Weird")

    def test_first_layer_differs_from_none(self) -> None:
        """Method='FirstLayer' даёт другой SV, чем 'None' (fl-поправка ≠ 0)."""
        sv_none = radial_evaluation("Ivan", "None", "Use_Contour_MassCenter", "None")
        sv_fl   = radial_evaluation("Ivan", "None", "Use_Contour_MassCenter", "FirstLayer")
        assert sv_none != sv_fl


# ===========================================================================
# total_modelling
# ===========================================================================


class TestTotalModelling:
    def test_structure(self) -> None:
        """TotalModelling возвращает 2×2×3 структуру (12 комбинаций)."""
        result = total_modelling("Ivan")
        assert len(result) == 2, "2 sistole_methods"
        for row_sm in result:
            assert len(row_sm) == 2, "2 eq_sphere_methods"
            for row_esm in row_sm:
                assert len(row_esm) == 3, "3 rad_eval_methods"

    def test_all_floats(self) -> None:
        """Все 12 элементов — float."""
        result = total_modelling("Ivan")
        for row_sm in result:
            for row_esm in row_sm:
                for sv in row_esm:
                    assert isinstance(sv, float), f"not float: {sv!r}"

    def test_all_reasonable(self) -> None:
        """Все SV в разумных пределах."""
        result = total_modelling("Ivan")
        for row_sm in result:
            for row_esm in row_sm:
                for sv in row_esm:
                    assert abs(sv) < 500.0, f"unreasonable SV: {sv}"
