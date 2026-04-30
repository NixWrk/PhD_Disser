"""Тесты численного поиска параметров (ModelFinders).

Фаза 2, коммит #15.
Покрывает: find_ro_one, find_ro_sphere, find_dr_sphere, find_dr_sphere_fl,
           самосогласованность (FindRoSphere → SphereModel(rho1) == Zb).
Эталонные значения берутся из ``tests/reference_values.json``
(секция ``computed.find_dr_sphere``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cardio_model.finders import (
    find_dr_sphere,
    find_dr_sphere_fl,
    find_ro_one,
    find_ro_sphere,
    find_h_two,
)
from cardio_model.models import one_layer_model, sphere_model, two_layer_model

# ---------------------------------------------------------------------------
# Эталонные данные
# ---------------------------------------------------------------------------

_REF_PATH = Path(__file__).parent / "reference_values.json"
with _REF_PATH.open(encoding="utf-8") as _fh:
    _REF = json.load(_fh)

# ---------------------------------------------------------------------------
# Типичные пациентские параметры (Ivan, channel 0)
# ---------------------------------------------------------------------------

_RO2 = 1.35
_IVAN = _REF["patients"]["Ivan"]
_ZB   = _IVAN["radial_zBase"][0]        # 35
_DZ   = _IVAN["radial_dZRad"][0]        # 0.057
_A    = _IVAN["param_a"][0]              # 0.04
_B    = _IVAN["param_b"][0]              # 0.02
_R    = _IVAN["param_R"][0]              # 0.051
_H    = _IVAN["param_h"][0]              # 0.03
_X    = _IVAN["param_x"][0]              # 0.0
_Y    = _IVAN["param_y"][0]              # 0.047


# ===========================================================================
# find_ro_one
# ===========================================================================


class TestFindRoOne:
    def test_analytic(self) -> None:
        """find_ro_one — аналитическая инверсия OneLayerModel."""
        Zb = 35.0
        ro1 = find_ro_one(Zb, _A, _B)
        assert abs(one_layer_model(ro1, _A, _B) - Zb) < 1e-10

    def test_round_trip(self) -> None:
        """find_ro_one(one_layer_model(ro1)) == ro1."""
        ro1 = 2.5
        Zb = one_layer_model(ro1, _A, _B)
        assert abs(find_ro_one(Zb, _A, _B) - ro1) < 1e-10

    def test_positive(self) -> None:
        assert find_ro_one(35.0, _A, _B) > 0

    def test_fl_layer_params(self) -> None:
        """Проверка для параметров первого слоя (flSize = [0.06, 0.03])."""
        fl_a, fl_b = 0.06, 0.03
        ro1 = find_ro_one(0.045, fl_a, fl_b)
        assert abs(one_layer_model(ro1, fl_a, fl_b) - 0.045) < 1e-10


# ===========================================================================
# find_ro_sphere (самосогласованность)
# ===========================================================================


class TestFindRoSphere:
    def test_self_consistent(self) -> None:
        """find_ro_sphere → SphereModel(rho1) ≈ Zb (в пределах шага сетки)."""
        Zb = float(_ZB)
        ro1 = find_ro_sphere(Zb, _RO2, _A, _B, _R, _H, _X, _Y)
        # Шаг сетки 0.05 → погрешность SphereModel может быть в пределах значения
        # производной dSphereModel/drho1 * 0.05
        val = sphere_model(ro1, _RO2, _A, _B, _R, _H, _X, _Y)
        # SphereModel(ro1) <= Zb <= SphereModel(ro1 + 0.05)
        val_next = sphere_model(ro1 + 0.05, _RO2, _A, _B, _R, _H, _X, _Y)
        assert val <= Zb <= val_next or abs(val - Zb) < 1.0, (
            f"ro1={ro1:.3f} дал SphereModel={val:.2f}, ожидали ≈ Zb={Zb}"
        )

    def test_positive_ro1(self) -> None:
        ro1 = find_ro_sphere(float(_ZB), _RO2, _A, _B, _R, _H, _X, _Y)
        assert ro1 > 0


# ===========================================================================
# find_dr_sphere (против эталона Wolfram)
# ===========================================================================


class TestFindDrSphere:
    @pytest.mark.parametrize("name", ["Ivan", "Alex", "Artem"])
    def test_method1_reference(self, name: str) -> None:
        """find_dr_sphere совпадает с Wolfram RadEvalMethod1 (ro1=0)."""
        patient = _REF["patients"][name]
        expected = _REF["computed"]["find_dr_sphere"][name]
        for i in range(5):
            dr = np.round(
                find_dr_sphere(
                    float(patient["radial_zBase"][i]),
                    float(patient["radial_dZRad"][i]),
                    0.0, _RO2,
                    float(patient["param_a"][i]),
                    float(patient["param_b"][i]),
                    float(patient["param_R"][i]),
                    float(patient["param_h"][i]),
                    float(patient["param_x"][i]),
                    float(patient["param_y"][i]),
                ),
                4,  # Round[x, 0.0001]
            )
            assert abs(dr - expected[i]) < 1e-8, (
                f"{name} ch{i}: got {dr}, expected {expected[i]}"
            )

    def test_dr_positive(self) -> None:
        """dR должен быть положительным для типичных входных данных."""
        dr = find_dr_sphere(_ZB, _DZ, 0.0, _RO2, _A, _B, _R, _H, _X, _Y)
        assert dr > 0

    def test_dr_small(self) -> None:
        """dR должен быть намного меньше R (разумное значение)."""
        dr = find_dr_sphere(_ZB, _DZ, 0.0, _RO2, _A, _B, _R, _H, _X, _Y)
        assert dr < _R


# ===========================================================================
# find_dr_sphere_fl (Method2)
# ===========================================================================


class TestFindDrSphereFl:
    @pytest.mark.parametrize("name", ["Ivan", "Alex", "Artem"])
    def test_method2_reference(self, name: str) -> None:
        """find_dr_sphere_fl совпадает с Wolfram RadEvalMethod2."""
        patient = _REF["patients"][name]
        expected = _REF["computed"]["rad_eval_method2"][name]
        fl_dz = float(patient["radial_flDZ"])
        fl_a  = float(patient["param_flSize"][0])
        fl_b  = float(patient["param_flSize"][1])
        dro1 = find_ro_one(fl_dz, fl_a, fl_b)
        for i in range(5):
            dr = np.round(
                find_dr_sphere_fl(
                    float(patient["radial_zBase"][i]),
                    float(patient["radial_dZRad"][i]),
                    dro1,
                    0.0, _RO2,
                    float(patient["param_a"][i]),
                    float(patient["param_b"][i]),
                    float(patient["param_R"][i]),
                    float(patient["param_h"][i]),
                    float(patient["param_x"][i]),
                    float(patient["param_y"][i]),
                ),
                4,
            )
            assert abs(dr - expected[i]) < 1e-8, (
                f"{name} ch{i}: got {dr}, expected {expected[i]}"
            )


# ===========================================================================
# find_h_two (самосогласованность)
# ===========================================================================


class TestFindHTwo:
    def test_self_consistent(self) -> None:
        """find_h_two → TwoLayerModel(h) ≈ Zb в пределах шага сетки.

        FindHTwo ищет ВОЗРАСТАЮЩЕЕ пересечение: TwoLayerModel растёт с h
        когда ro2 < ro1 (кровь/ткань проводит лучше внешнего цилиндра).
        С ro1=2.0, ro2=1.35, a=0.05, b=0.025 функция монотонно растёт на [0, 0.05].
        """
        ro1, ro2 = 2.0, 1.35   # ro2 < ro1 → function increases with h
        a, b = 0.05, 0.025
        Zb = two_layer_model(ro1, ro2, 0.02, a, b)
        h = find_h_two(Zb, ro1, ro2, a, b)
        val = two_layer_model(ro1, ro2, h, a, b)
        # val <= Zb <= val(h+step) — в пределах шага 0.0005
        val_next = two_layer_model(ro1, ro2, h + 0.0005, a, b)
        assert val <= Zb <= val_next or abs(val - Zb) < 1e-4
