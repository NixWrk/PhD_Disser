"""Расчёт перемещений границы сердца по радиальному импедансу.

Порт ``Kernel/core/RadialEvaluation.m``.

Публичное API:
    rad_eval_method1(obj, param) -> np.ndarray[5]   # FindDRSphere × 5 каналов
    rad_eval_method2(obj, param) -> np.ndarray[5]   # + первый слой
    rad_eval_method4(obj, param, dxdy) -> np.ndarray[5]  # + смещение центра
    rad_eval(obj, param, method="Method_1") -> np.ndarray[5]

Формат ``obj`` (dict/Association):
    "zBase"  : list[float]  — базовые импедансы по каналам (Ом)
    "dZRad"  : list[float]  — измеренные dZ (Ом)
    "flDZ"   : float        — dZ первого слоя
    "flBase" : float        — (не используется напрямую в Method1/2)

Формат ``param`` (dict/Association):
    "a", "b", "R", "h", "x", "y" : list[float], длина 5
    "hFat"    : list[float]  — толщина жира (1 значение)
    "flSize"  : list[float]  — [a_fl, b_fl]

Все методы применяют ``np.round(result, 4)`` (порт ``Round[x, 0.0001]``).
"""

from __future__ import annotations

import numpy as np

from cardio_model.finders import (
    find_dr_sphere,
    find_dr_sphere_fl,
    find_dr_sphere_fl_dxdy,
    find_ro_one,
)

__all__ = [
    "rad_eval_method1",
    "rad_eval_method2",
    "rad_eval_method4",
    "rad_eval",
]

_RO2: float = 1.35   # ro2 = удельное сопротивление крови (Ом·м); зафиксировано во всех методах


# ---------------------------------------------------------------------------
# RadEvalMethod1
# ---------------------------------------------------------------------------


def rad_eval_method1(obj: dict, param: dict) -> np.ndarray:
    """Порт ``RadEvalMethod1[obj, param]``.

    Для каждого из 5 каналов вызывает ``FindDRSphere`` с ro1=0
    (авто-поиск ρ₁ через ``FindRoSphere``).

    Возвращает np.ndarray[5] с точностью 4 знака (Round[x, 0.0001]).
    """
    result = np.array([
        find_dr_sphere(
            float(obj["zBase"][i]), float(obj["dZRad"][i]),
            0.0, _RO2,
            float(param["a"][i]), float(param["b"][i]),
            float(param["R"][i]), float(param["h"][i]),
            float(param["x"][i]), float(param["y"][i]),
        )
        for i in range(5)
    ])
    return np.round(result, 4)


# ---------------------------------------------------------------------------
# RadEvalMethod2
# ---------------------------------------------------------------------------


def rad_eval_method2(obj: dict, param: dict) -> np.ndarray:
    """Порт ``RadEvalMethod2[obj, param]``.

    Добавляет поправку первого слоя (жировая ткань): вычисляет dρ₁ из
    ``find_ro_one(flDZ, flSize[0], flSize[1])`` и передаёт в
    ``FindDRSphereFl``.

    Возвращает np.ndarray[5] с точностью 4 знака.
    """
    dro1 = find_ro_one(
        float(obj["flDZ"]),
        float(param["flSize"][0]),
        float(param["flSize"][1]),
    )
    result = np.array([
        find_dr_sphere_fl(
            float(obj["zBase"][i]), float(obj["dZRad"][i]),
            dro1,
            0.0, _RO2,
            float(param["a"][i]), float(param["b"][i]),
            float(param["R"][i]), float(param["h"][i]),
            float(param["x"][i]), float(param["y"][i]),
        )
        for i in range(5)
    ])
    return np.round(result, 4)


# ---------------------------------------------------------------------------
# RadEvalMethod4
# ---------------------------------------------------------------------------


def rad_eval_method4(obj: dict, param: dict, dxdy: np.ndarray) -> np.ndarray:
    """Порт ``RadEvalMethod4[obj, param, dxdy]``.

    Как Method2, плюс смещение центра сферы (dxdy) для каждого канала.

    ``dxdy`` — массив 5×2 в мм (строки — каналы, столбцы — [dy_local, dx_local]).
    Важно: в оригинале x и y перестановлены — ``dxdy[[i,2]]/1000`` идёт в dx,
    ``dxdy[[i,1]]/1000`` в dy (Mathematica 1-based: col2 → dx, col1 → dy).

    После нахождения dr: result[i] = R[i] - sqrt((R[i]-dr[i])^2 - (dx[i])^2) - dy[i].

    Возвращает np.ndarray[5] с точностью 4 знака.
    """
    dro1 = find_ro_one(
        float(obj["flDZ"]),
        float(param["flSize"][0]),
        float(param["flSize"][1]),
    )
    dr = np.array([
        find_dr_sphere_fl_dxdy(
            float(obj["zBase"][i]), float(obj["dZRad"][i]),
            # сначала col2 (dy_local), потом col1 (dx_local) — как в оригинале
            float(dxdy[i, 1]) / 1000.0, float(dxdy[i, 0]) / 1000.0,
            dro1,
            0.0, _RO2,
            float(param["a"][i]), float(param["b"][i]),
            float(param["R"][i]), float(param["h"][i]),
            float(param["x"][i]), float(param["y"][i]),
        )
        for i in range(5)
    ])
    R_arr = np.array([float(param["R"][i]) for i in range(5)])
    dx_arr = np.array([float(dxdy[i, 1]) / 1000.0 for i in range(5)])
    dy_arr = np.array([float(dxdy[i, 0]) / 1000.0 for i in range(5)])
    result = R_arr - np.sqrt((R_arr - dr) ** 2 - dx_arr ** 2) - dy_arr
    return np.round(result, 4)


# ---------------------------------------------------------------------------
# RadEval — диспетчер
# ---------------------------------------------------------------------------


def rad_eval(obj: dict, param: dict, method: str = "Method_1") -> np.ndarray:
    """Диспетчер методов.  Порт ``RadEval[obj, param, EvalMethod->"Method_1"]``.

    Поддерживаемые значения ``method``:
      - ``"Method_1"`` → ``rad_eval_method1``
      - ``"Method_2"`` → ``rad_eval_method2``
    """
    if method == "Method_1":
        return rad_eval_method1(obj, param)
    if method == "Method_2":
        return rad_eval_method2(obj, param)
    raise ValueError(f"Unknown rad_eval method: {method!r}. Use 'Method_1' or 'Method_2'.")
