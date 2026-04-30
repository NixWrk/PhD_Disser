"""Тесты радиального расчёта перемещений (RadialEvaluation).

Фаза 2, коммит #16.
Покрывает: rad_eval_method1, rad_eval_method2, rad_eval диспетчер.
Эталон: ``tests/reference_values.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cardio_model.radial import rad_eval, rad_eval_method1, rad_eval_method2
from cardio_model.data.patients import get_param, get_radial

_REF_PATH = Path(__file__).parent / "reference_values.json"
with _REF_PATH.open(encoding="utf-8") as _fh:
    _REF = json.load(_fh)


# ===========================================================================
# RadEvalMethod1
# ===========================================================================


class TestRadEvalMethod1:
    @pytest.mark.parametrize("name", ["Ivan", "Alex", "Artem"])
    def test_against_wolfram(self, name: str) -> None:
        """rad_eval_method1 совпадает с Wolfram RadEvalMethod1."""
        obj   = get_radial(name)
        param = get_param(name)
        got      = rad_eval_method1(obj, param)
        expected = np.array(_REF["computed"]["rad_eval_method1"][name])
        np.testing.assert_array_equal(got, expected)

    @pytest.mark.parametrize("name", ["Ivan", "Alex", "Artem"])
    def test_shape(self, name: str) -> None:
        obj, param = get_radial(name), get_param(name)
        assert rad_eval_method1(obj, param).shape == (5,)

    @pytest.mark.parametrize("name", ["Ivan", "Alex", "Artem"])
    def test_positive(self, name: str) -> None:
        obj, param = get_radial(name), get_param(name)
        assert np.all(rad_eval_method1(obj, param) > 0)

    @pytest.mark.parametrize("name", ["Ivan", "Alex", "Artem"])
    def test_rounded_to_4_decimal(self, name: str) -> None:
        """Результат округлён до 4 знаков (Round[x, 0.0001])."""
        obj, param = get_radial(name), get_param(name)
        result = rad_eval_method1(obj, param)
        np.testing.assert_array_equal(result, np.round(result, 4))


# ===========================================================================
# RadEvalMethod2
# ===========================================================================


class TestRadEvalMethod2:
    @pytest.mark.parametrize("name", ["Ivan", "Alex", "Artem"])
    def test_against_wolfram(self, name: str) -> None:
        """rad_eval_method2 совпадает с Wolfram RadEvalMethod2."""
        obj   = get_radial(name)
        param = get_param(name)
        got      = rad_eval_method2(obj, param)
        expected = np.array(_REF["computed"]["rad_eval_method2"][name])
        np.testing.assert_array_equal(got, expected)

    @pytest.mark.parametrize("name", ["Ivan", "Alex", "Artem"])
    def test_method2_larger_than_method1(self, name: str) -> None:
        """Method2 (с первым слоем) должен давать бо́льшие dR, чем Method1."""
        obj, param = get_radial(name), get_param(name)
        r1 = rad_eval_method1(obj, param)
        r2 = rad_eval_method2(obj, param)
        # Жировой слой увеличивает эффективную dZ → большее dR
        assert np.all(r2 >= r1), f"{name}: Method2 не >= Method1: {r2} vs {r1}"


# ===========================================================================
# RadEval диспетчер
# ===========================================================================


class TestRadEval:
    def test_method1_dispatch(self) -> None:
        obj, param = get_radial("Ivan"), get_param("Ivan")
        np.testing.assert_array_equal(
            rad_eval(obj, param, "Method_1"),
            rad_eval_method1(obj, param),
        )

    def test_method2_dispatch(self) -> None:
        obj, param = get_radial("Artem"), get_param("Artem")
        np.testing.assert_array_equal(
            rad_eval(obj, param, "Method_2"),
            rad_eval_method2(obj, param),
        )

    def test_default_is_method1(self) -> None:
        obj, param = get_radial("Alex"), get_param("Alex")
        np.testing.assert_array_equal(
            rad_eval(obj, param),
            rad_eval_method1(obj, param),
        )

    def test_unknown_method_raises(self) -> None:
        obj, param = get_radial("Ivan"), get_param("Ivan")
        with pytest.raises(ValueError, match="Unknown"):
            rad_eval(obj, param, "Method_99")
