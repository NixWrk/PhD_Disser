"""Тесты физических моделей импеданса.

Фаза 2, коммиты #12 и #14.
Покрывает: one_layer_model, two_layer_model, sphere_model, sphere_influence.
"""

from __future__ import annotations

import numpy as np
import pytest

from cardio_model.models import (
    one_layer_model,
    two_layer_model,
    sphere_model,
    sphere_influence,
)

# ---------------------------------------------------------------------------
# Параметры, типичные для пациентских каналов
# ---------------------------------------------------------------------------
RHO1 = 1.35      # Ом·м — удельное сопротивление тканей
RHO2 = 1.35      # Ом·м — одинаковое → поправка должна обнуляться
A  = 0.04        # м
B  = 0.02        # м
R  = 0.042       # м — радиус сферы (канал 3 Ivan)
H  = 0.022       # м — глубина
X  = 0.0         # м
Y  = 0.035       # м


class TestOneLayerModel:
    def test_basic_formula(self) -> None:
        """Z = ρ1·2b / (π·(a²-b²))."""
        z = one_layer_model(RHO1, A, B)
        expected = RHO1 * 2.0 * B / (np.pi * (A**2 - B**2))
        assert abs(z - expected) < 1e-12

    def test_positive(self) -> None:
        """Импеданс всегда положителен при rho1>0, a>b>0."""
        assert one_layer_model(1.0, 0.05, 0.02) > 0

    def test_scales_linearly_with_rho(self) -> None:
        """Z пропорционален ρ1."""
        z1 = one_layer_model(1.0, A, B)
        z2 = one_layer_model(2.0, A, B)
        assert abs(z2 / z1 - 2.0) < 1e-12

    def test_scales_with_b(self) -> None:
        """Z возрастает с b (при фиксированных a, rho1)."""
        z_small = one_layer_model(RHO1, A, 0.01)
        z_large = one_layer_model(RHO1, A, 0.03)
        assert z_large > z_small

    def test_realistic_range(self) -> None:
        """Для a=0.04м, b=0.02м, ρ1=1.35 результат ~14 Ом (компонент модели).

        Примечание: zBase~35..108 включает дополнительные компоненты (контакты,
        кожа); one_layer_model возвращает только объёмный вклад тела.
        """
        z = one_layer_model(RHO1, A, B)
        # Z = 1.35·2·0.02 / (π·(0.04²−0.02²)) ≈ 14.32 Ом
        assert 5 < z < 30, f"Нереалистичный импеданс: {z:.2f} Ом"


class TestTwoLayerModel:
    def test_equal_rho_equals_one_layer(self) -> None:
        """При rho1==rho2 TwoLayer = OneLayer + поправка ≈ OneLayer.

        Когда ρ1==ρ2, все члены ряда имеют коэффициент (ρ2-ρ1)=0 → сумма = 0,
        и получается OneLayer + 0.
        """
        z_one = one_layer_model(RHO1, A, B)
        z_two = two_layer_model(RHO1, RHO1, H, A, B)
        np.testing.assert_allclose(z_two, z_one, rtol=1e-6)

    def test_fat_layer_increases_impedance(self) -> None:
        """Жировой слой (rho2 > rho1) увеличивает общий импеданс."""
        z_no_fat = two_layer_model(RHO1, RHO1, 0.001, A, B)
        z_with_fat = two_layer_model(RHO1, 5.0, 0.01, A, B)
        assert z_with_fat > z_no_fat

    def test_series_convergence(self) -> None:
        """Ряд сходится: разница между n=80 и n=100 < 1e-8."""
        z80 = two_layer_model(RHO1, 2.0, H, A, B, n_max=80)
        z100 = two_layer_model(RHO1, 2.0, H, A, B, n_max=100)
        assert abs(z100 - z80) < 1e-8, f"Ряд не сошёлся: Δ={abs(z100-z80):.2e}"

    def test_positive(self) -> None:
        assert two_layer_model(RHO1, 2.0, H, A, B) > 0


class TestSphereModel:
    def test_equal_rho_equals_one_layer(self) -> None:
        """При rho1==rho2 сфера не вносит поправку → SphereModel == OneLayer."""
        z_one = one_layer_model(RHO1, A, B)
        z_sph = sphere_model(RHO1, RHO1, A, B, R, H, X, Y)
        # Поправка от сферы при равных ρ обращается в ноль
        np.testing.assert_allclose(z_sph, z_one, rtol=1e-6)

    def test_influence_is_zero_for_equal_rho(self) -> None:
        infl = sphere_influence(RHO1, RHO1, A, B, R, H, X, Y)
        np.testing.assert_allclose(infl, 0.0, atol=1e-8)

    def test_influence_sign(self) -> None:
        """rho2 < rho1 (кровь проводит лучше) → отрицательная поправка."""
        infl = sphere_influence(RHO1, 0.7, A, B, R, H, X, Y)
        assert infl < 0, f"Ожидали отрицательную поправку, получили {infl:.4f}"

    def test_sphere_model_positive(self) -> None:
        """Суммарный импеданс положителен при реалистичных параметрах."""
        z = sphere_model(RHO1, 0.7, A, B, R, H, X, Y)
        assert z > 0

    def test_legendre_series_convergence(self) -> None:
        """Ряд Лежандра сходится: разница n=80 vs n=100 < 1e-8."""
        from cardio_model.models import _zep
        rp, re = 0.07, 0.06
        cos_ep = 0.5
        z80  = _zep(RHO1, 0.7, R, rp, re, cos_ep, n_max=80)
        z100 = _zep(RHO1, 0.7, R, rp, re, cos_ep, n_max=100)
        assert abs(z100 - z80) < 1e-8, f"Zep не сошёлся: Δ={abs(z100-z80):.2e}"

    @pytest.mark.parametrize("name,ch", [
        ("Ivan",  0),
        ("Ivan",  2),
        ("Artem", 0),
        ("Alex",  4),
    ])
    def test_realistic_patient_channel(self, name: str, ch: int) -> None:
        """SphereModel для каналов пациентов возвращает разумный импеданс (Ом)."""
        from cardio_model.data.patients import get_param, get_radial
        p = get_param(name)
        r = get_radial(name)
        z = sphere_model(
            RHO1, RHO2,
            p["a"][ch], p["b"][ch],
            p["R"][ch], p["h"][ch],
            p["x"][ch], p["y"][ch],
        )
        zb = r["zBase"][ch]
        # Импеданс должен быть в 0.1x..10x от базового
        assert 0.1 * zb < z < 10 * zb, (
            f"{name} ch{ch}: z={z:.2f}, zBase={zb:.2f} — нереалистично"
        )
