"""Тесты геометрии контура и расчёта объёма.

Фаза 3, коммиты #17-#18.
Покрывает: kubic_interpol, list_of_parallel_sections,
           volume_by_contour, sv_by_contour_and_5move.
Эталон: ``tests/reference_values.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cardio_model.geometry import (
    get_coord_after_move,
    kubic_interpol,
    list_of_parallel_sections,
)
from cardio_model.volume import (
    cut_konus_circle_volume,
    sv_by_contour_and_5move,
    volume_by_contour,
    volume_by_section,
)
from cardio_model.data.patients import get_contour
from cardio_model.data.moves import edge_move_5pt

_REF_PATH = Path(__file__).parent / "reference_values.json"
with _REF_PATH.open(encoding="utf-8") as _fh:
    _REF = json.load(_fh)


# ===========================================================================
# KubicInterpol
# ===========================================================================


class TestKubicInterpol:
    @pytest.mark.parametrize("name", ["Ivan", "Alex", "Artem"])
    def test_output_length(self, name: str) -> None:
        """7 узлов, n=30 → (7-1)*(30+1) = 186 точек."""
        contour = get_contour(name)
        result = kubic_interpol(contour)
        expected_len = _REF["computed"]["kubic_interpol_length"][name]
        assert result.shape == (expected_len, 2), (
            f"{name}: shape {result.shape} != ({expected_len}, 2)"
        )

    @pytest.mark.parametrize("name", ["Ivan", "Alex", "Artem"])
    def test_endpoints_preserved(self, name: str) -> None:
        """Первая и последняя точки интерполяции совпадают с узлами."""
        contour = get_contour(name)
        interp = kubic_interpol(contour)
        np.testing.assert_allclose(interp[0],  contour[0],  atol=1e-10)
        np.testing.assert_allclose(interp[-1], contour[-1], atol=1e-10)

    def test_x_range_covered(self) -> None:
        """Интерполяция перекрывает диапазон x от первой до последней точки."""
        pts = np.array([[0, 0], [1, 2], [3, 1], [5, 3],
                        [7, 0], [9, 2], [10, 1]], dtype=float)
        interp = kubic_interpol(pts)
        assert interp[:, 0].min() <= pts[:, 0].min() + 0.5
        assert interp[:, 0].max() >= pts[:, 0].max() - 0.5

    def test_no_nan_in_output(self) -> None:
        """Интерполяция не производит NaN."""
        pts = np.array([[-33, -8], [-32, -24], [11, -51], [58, -37],
                        [63, -11], [36, 27], [27, 29]], dtype=float)
        interp = kubic_interpol(pts)
        assert not np.any(np.isnan(interp))


# ===========================================================================
# GetCoordAfterMove
# ===========================================================================


class TestGetCoordAfterMove:
    def test_zero_move_unchanged(self) -> None:
        """Нулевое смещение не меняет контур."""
        contour = get_contour("Ivan")
        moved = get_coord_after_move(contour, np.zeros(7))
        np.testing.assert_allclose(moved, contour, atol=1e-10)

    def test_positive_move_increases_radius(self) -> None:
        """Положительное смещение увеличивает полярный радиус."""
        pts = np.array([[3.0, 4.0]])  # r = 5
        moved = get_coord_after_move(pts, np.array([1.0]))  # r → 6
        r_new = np.hypot(moved[0, 0], moved[0, 1])
        assert abs(r_new - 6.0) < 1e-10

    def test_angle_preserved(self) -> None:
        """Угол не изменяется при радиальном смещении."""
        pts = np.array([[3.0, 4.0]])
        moved = get_coord_after_move(pts, np.array([2.0]))
        theta_orig = np.arctan2(4.0, 3.0)
        theta_new  = np.arctan2(moved[0, 1], moved[0, 0])
        assert abs(theta_orig - theta_new) < 1e-10


# ===========================================================================
# CutKonusCircleVolume
# ===========================================================================


class TestCutKonusCircleVolume:
    def test_cylinder(self) -> None:
        """При d1==d2 формула даёт объём цилиндра: π·r²·h."""
        d = 10.0  # диаметр
        h = 5.0
        r = d / 2.0
        expected = np.pi * r ** 2 * h
        # CutKonus с d1==d2: π·h/12·(d²+d·d+d²) = π·h/12·3d² = π·h·d²/4 = π·h·r²
        result = cut_konus_circle_volume(h, d, d)
        assert abs(result - expected) < 1e-10

    def test_zero_top(self) -> None:
        """При d_low=0 — пирамида: V = π·r²·h/3."""
        d = 6.0
        h = 4.0
        expected = np.pi * (d / 2) ** 2 * h / 3.0
        result = cut_konus_circle_volume(h, d, 0.0)
        assert abs(result - expected) < 1e-10


# ===========================================================================
# VolumeByContour (эталон из Wolfram)
# ===========================================================================


class TestVolumeByContour:
    @pytest.mark.parametrize("name,expected", [
        ("Ivan",  485.0),
        ("Alex",  604.0),
        ("Artem", 622.0),
    ])
    def test_against_wolfram(self, name: str, expected: float) -> None:
        """volume_by_contour совпадает с Wolfram VolumeByContour[..., 3]."""
        contour = get_contour(name)
        vol = volume_by_contour(contour, h_step=3.0)
        # Wolfram использует Round[..., 1] — допуск ±0.5 мл
        assert abs(vol - expected) <= 0.5, (
            f"{name}: got {vol}, expected {expected}"
        )

    def test_positive(self) -> None:
        assert volume_by_contour(get_contour("Ivan")) > 0


# ===========================================================================
# SVbyContourAnd5Move
# ===========================================================================


class TestSVbyContourAnd5Move:
    @pytest.mark.parametrize("name,expected", [
        ("Ivan",  -216.0),
        ("Alex",  -135.5),
        ("Artem", -105.0),
    ])
    def test_against_wolfram(self, name: str, expected: float) -> None:
        """sv_by_contour_and_5move совпадает с Wolfram SVbyContourAnd5Move."""
        contour = get_contour(name)
        move5   = edge_move_5pt(name)
        sv = sv_by_contour_and_5move(contour, move5)
        # Допуск ±0.5 мл (как и для volume_by_contour)
        assert abs(sv - expected) <= 0.5, (
            f"{name}: got {sv}, expected {expected}"
        )

    @pytest.mark.parametrize("name", ["Ivan", "Alex", "Artem"])
    def test_negative_with_inverted_move(self, name: str) -> None:
        """edge_move_5pt возвращает инвертированные значения → SV < 0."""
        contour = get_contour(name)
        move5   = edge_move_5pt(name)
        assert sv_by_contour_and_5move(contour, move5) < 0
