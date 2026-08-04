"""Строгие регрессии для расхождений, найденных при аудите Wolfram-порта.

Эти тесты намеренно проверяют не общие свойства результата, а конкретную
семантику исходного Wolfram-кода:

* ``Round[x, 1]`` — округление до целого, а не до одного знака;
* ``RadEvalMethod4`` — округление промежуточного dr, но не результата;
* ограничения ``NMinimize`` для центра и радиуса;
* вещественный шаг в ``Table[{i, 1, 101, step}]``;
* полные массивы ``TotalModelling`` из сохранённого Wolfram notebook.
"""

from __future__ import annotations

import numpy as np
import pytest

import cardio_model.geometry as geometry_module
import cardio_model.radial as radial_module
from cardio_model.data.patients import get_contour
from cardio_model.geometry import list_of_parallel_sections
from cardio_model.pipeline import total_modelling
from cardio_model.radial import rad_eval_method4
from cardio_model.sphere_fit import eq_sph_nm_center_radius
from cardio_model.volume import volume_by_contour, volume_by_section


def test_volume_round_uses_wolfram_quantum_not_decimal_places() -> None:
    """Round[10.49, 1] в Wolfram даёт 10, тогда как round(10.49, 1) даёт 10.5."""
    raw_volume_ml = 10.49
    height_mm = 1.0
    diameter_mm = np.sqrt(raw_volume_ml * 4000.0 / np.pi)
    section = np.array([[0.0, 0.0], [diameter_mm, 0.0]])

    result = volume_by_section([section, section.copy()], height_mm)

    assert result == 10.0


@pytest.mark.parametrize(
    "name, expected",
    [("Ivan", 485.0), ("Alex", 604.0), ("Artem", 622.0)],
)
def test_patient_volume_is_exact_wolfram_integer(name: str, expected: float) -> None:
    """Пациентские эталоны должны совпадать точно после Round[..., 1]."""
    assert volume_by_contour(get_contour(name), h_step=3.0) == expected


def test_method4_rounds_dr_before_transform_but_not_final_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Изолированно фиксирует порядок округления в Wolfram RadEvalMethod4."""
    raw_dr = 0.007181031777
    rounded_dr = 0.0072
    radius = 0.05
    dxdy_mm = np.tile([1.234, 2.345], (5, 1))

    monkeypatch.setattr(radial_module, "find_ro_one", lambda *_args: 0.01)
    monkeypatch.setattr(
        radial_module,
        "find_dr_sphere_fl_dxdy",
        lambda *_args: raw_dr,
    )

    obj = {
        "flDZ": 0.0,
        "zBase": [1.0] * 5,
        "dZRad": [0.1] * 5,
    }
    param = {
        "flSize": [0.05, 0.025],
        "a": [0.05] * 5,
        "b": [0.025] * 5,
        "R": [radius] * 5,
        "h": [0.02] * 5,
        "x": [0.0] * 5,
        "y": [0.0] * 5,
    }

    result = rad_eval_method4(obj, param, dxdy_mm)

    dx = dxdy_mm[0, 1] / 1000.0
    dy = dxdy_mm[0, 0] / 1000.0
    expected = radius - np.sqrt((radius - rounded_dr) ** 2 - dx**2) - dy
    np.testing.assert_allclose(result, np.full(5, expected), rtol=0.0, atol=1e-15)
    assert expected != np.round(expected, 4), "пример обязан различать два порядка округления"


@pytest.mark.parametrize(
    "center, radius",
    [
        ((100.0, 0.0), 10.0),
        ((0.0, 100.0), 80.0),
        ((0.0, 0.0), 10.0),
    ],
)
def test_center_radius_fit_honours_wolfram_bounds(
    center: tuple[float, float],
    radius: float,
) -> None:
    """Контуры с безусловным оптимумом вне области остаются внутри bounds."""
    theta = np.linspace(0.0, 2.0 * np.pi, 80, endpoint=False)
    contour = np.column_stack(
        [
            center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
        ]
    )

    fitted_center, fitted_radius = eq_sph_nm_center_radius(contour)

    assert -40.0 <= fitted_center[0] <= 40.0
    assert -60.0 <= fitted_center[1] <= 40.0
    assert 20.0 <= fitted_radius <= 60.0


def test_parallel_sections_follow_wolfram_table_for_fractional_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Шаг 2.5 должен дать 1, 3.5, ..., 101 без преобразования в int."""
    visited: list[float] = []

    def fake_get_l_point(_points: np.ndarray, shift: float) -> np.ndarray:
        visited.append(shift)
        return np.zeros((2, 2))

    monkeypatch.setattr(geometry_module, "get_l_point", fake_get_l_point)
    sections = list_of_parallel_sections(np.zeros((2, 2)), step=2.5)

    expected = 1.0 + 2.5 * np.arange(41)
    np.testing.assert_array_equal(visited, expected)
    assert len(sections) == len(expected)


def test_fractional_volume_step_is_not_truncated_to_integer() -> None:
    """Старая реализация int(2.5) давала для Ивана ошибочные 606 мл."""
    assert volume_by_contour(get_contour("Ivan"), h_step=2.5) == 485.0


@pytest.mark.parametrize(
    "name, expected",
    [
        (
            "Ivan",
            [
                [[44.5, 78.0, 14.0], [44.5, 78.0, 21.0]],
                [[38.5, 82.0, 23.0], [38.5, 82.0, 44.5]],
            ],
        ),
        (
            "Alex",
            [
                [[19.5, 79.5, 33.5], [19.5, 79.5, 17.5]],
                [[18.5, 61.0, 55.5], [18.5, 61.0, 53.5]],
            ],
        ),
        (
            "Artem",
            [
                [[14.5, 54.0, -44.5], [14.5, 54.0, -2.5]],
                [[-3.0, 88.5, -14.0], [-3.0, 88.5, 55.0]],
            ],
        ),
    ],
)
def test_total_modelling_matches_saved_wolfram_notebook_exactly(
    name: str,
    expected: list[list[list[float]]],
) -> None:
    """Все 36 конечных SV совпадают с Output SphereMoveModelling.nb."""
    assert total_modelling(name) == expected
