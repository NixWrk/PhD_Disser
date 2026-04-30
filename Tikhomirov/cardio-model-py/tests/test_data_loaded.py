"""Фаза 1, #09 — верификация загрузки данных пациентов.

Проверяет:
  - все пациенты имеют 5-канальные параметры a/b/R/h/x/y
  - dZRad сходится с эталонными значениями из reference_values.json
  - контуры имеют форму (7, 2)
  - МРТ-контуры имеют форму (11, 2) и замкнуты (первая == последняя)
  - IMP-массивы имеют ожидаемые формы
  - .txt файлы движений загружаются и имеют 5 строк
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cardio_model.data.patients import get_radial, get_param, get_contour, RADIAL, PARAMS, CONTOURS
from cardio_model.data.mri import (
    get_contour_with_atrial, get_atrial_points,
    VALVE_MOVE, R_BY_MRI_BY_VOLUME, CONTOURS_WITH_ATRIAL,
)
from cardio_model.data.imp import ARTEM_RADIAL_ARRAY, ARTEM_RADIAL_ARRAY_FIX, ARTEM_SPHERE_MOVING, get_artem_radial_at
from cardio_model.data.moves import load_moves, edge_move_5pt, edge_move_7pt

_REF = json.loads((Path(__file__).parent / "reference_values.json").read_text(encoding="utf-8"))
_PATIENTS = ["Ivan", "Alex", "Artem"]


# ---------------------------------------------------------------------------
# patients.py
# ---------------------------------------------------------------------------

class TestPatients:
    @pytest.mark.parametrize("name", _PATIENTS)
    def test_radial_has_5_channels(self, name: str) -> None:
        rad = get_radial(name)
        assert rad["dZRad"].shape == (5,)
        assert rad["zBase"].shape == (5,)

    @pytest.mark.parametrize("name", _PATIENTS)
    def test_param_has_5_channels(self, name: str) -> None:
        p = get_param(name)
        for key in ("a", "b", "R", "h", "x", "y"):
            assert p[key].shape == (5,), f"param[{key!r}] у {name} не 5-канальный"

    @pytest.mark.parametrize("name", _PATIENTS)
    def test_dZRad_matches_reference(self, name: str) -> None:
        ref = _REF["patients"][name]["radial_dZRad"]
        got = get_radial(name)["dZRad"]
        np.testing.assert_allclose(got, ref, atol=1e-6, err_msg=f"dZRad {name}")

    @pytest.mark.parametrize("name", _PATIENTS)
    def test_param_R_matches_reference(self, name: str) -> None:
        ref = _REF["patients"][name]["param_R"]
        got = get_param(name)["R"]
        np.testing.assert_allclose(got, ref, atol=1e-6, err_msg=f"param R {name}")

    @pytest.mark.parametrize("name", _PATIENTS)
    def test_contour_shape(self, name: str) -> None:
        c = get_contour(name)
        assert c.shape == (7, 2), f"Контур {name}: ожидали (7,2), получили {c.shape}"

    @pytest.mark.parametrize("name", _PATIENTS)
    def test_contour_first_point(self, name: str) -> None:
        ref = _REF["patients"][name]["contour_first"]
        got = get_contour(name)[0].tolist()
        assert got == ref, f"Первая точка контура {name}: {got} != {ref}"

    def test_unknown_patient_raises(self) -> None:
        with pytest.raises(ValueError):
            get_radial("Unknown")
        with pytest.raises(ValueError):
            get_param("Unknown")
        with pytest.raises(ValueError):
            get_contour("Unknown")


# ---------------------------------------------------------------------------
# mri.py
# ---------------------------------------------------------------------------

class TestMRI:
    @pytest.mark.parametrize("name", _PATIENTS)
    def test_contour_with_atrial_shape(self, name: str) -> None:
        c = get_contour_with_atrial(name)
        assert c.shape == (11, 2)

    @pytest.mark.parametrize("name", _PATIENTS)
    def test_contour_with_atrial_closed(self, name: str) -> None:
        """Первая и последняя точки совпадают (замкнутый контур)."""
        c = get_contour_with_atrial(name)
        np.testing.assert_array_equal(c[0], c[-1])

    @pytest.mark.parametrize("name", _PATIENTS)
    def test_atrial_points_shape(self, name: str) -> None:
        pts = get_atrial_points(name)
        assert pts.shape == (3, 2)

    @pytest.mark.parametrize("name", _PATIENTS)
    def test_valve_move(self, name: str) -> None:
        assert VALVE_MOVE[name] == _REF["mri"][name]["valve_move"]

    @pytest.mark.parametrize("name", _PATIENTS)
    def test_r_by_volume(self, name: str) -> None:
        assert R_BY_MRI_BY_VOLUME[name] == _REF["mri"][name]["r_by_volume"]


# ---------------------------------------------------------------------------
# imp.py
# ---------------------------------------------------------------------------

class TestIMP:
    def test_artem_radial_array_shape(self) -> None:
        assert ARTEM_RADIAL_ARRAY.shape == (25, 6)

    def test_artem_radial_array_fix_shape(self) -> None:
        assert ARTEM_RADIAL_ARRAY_FIX.shape == (25, 6)

    def test_artem_sphere_moving_shape(self) -> None:
        assert ARTEM_SPHERE_MOVING.shape == (25, 4)

    def test_artem_sphere_moving_row0(self) -> None:
        ref = _REF["imp"]["artem_sphere_moving_row0"]
        np.testing.assert_allclose(ARTEM_SPHERE_MOVING[0], ref, atol=1e-9)

    def test_get_artem_radial_at_returns_dict(self) -> None:
        r = get_artem_radial_at(8)
        assert r["dZRad"].shape == (5,)
        assert r["zBase"].shape == (5,)

    def test_get_artem_radial_at_out_of_range(self) -> None:
        with pytest.raises(IndexError):
            get_artem_radial_at(0)
        with pytest.raises(IndexError):
            get_artem_radial_at(26)


# ---------------------------------------------------------------------------
# moves/
# ---------------------------------------------------------------------------

class TestMoves:
    @pytest.mark.parametrize("name,expected_cols", [
        ("Alex",  10),
        ("Ivan",  10),
        ("Artem",  9),
    ])
    def test_load_moves_shape(self, name: str, expected_cols: int) -> None:
        m = load_moves(name)
        assert m.shape == (5, expected_cols), f"moves {name}: {m.shape}"

    @pytest.mark.parametrize("name", _PATIENTS)
    def test_edge_move_5pt_shape(self, name: str) -> None:
        v = edge_move_5pt(name)
        assert v.shape == (5,)

    @pytest.mark.parametrize("name", _PATIENTS)
    def test_edge_move_7pt_shape(self, name: str) -> None:
        v = edge_move_7pt(name)
        assert v.shape == (7,)

    @pytest.mark.parametrize("name", _PATIENTS)
    def test_edge_move_7pt_boundary(self, name: str) -> None:
        """Граничные точки 7pt == первой и последней точке 5pt."""
        v5 = edge_move_5pt(name)
        v7 = edge_move_7pt(name)
        assert v7[0] == v5[0]
        assert v7[-1] == v5[-1]

    def test_unknown_patient_raises(self) -> None:
        with pytest.raises(ValueError):
            load_moves("Unknown")
