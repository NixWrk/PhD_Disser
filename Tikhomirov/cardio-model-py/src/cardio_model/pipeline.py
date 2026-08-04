"""Верхнеуровневый пайплайн: EqualSphereMoveModelling, RadialEvaluation, TotalModelling.

Порт ``Kernel/part1/EqualSphereMoveModellingLib.m``.

Схема вычислений:
  1. EqualSphereMoveModelling — находит смещение центра эквивалентной сферы
     (диастола → систола) по МРТ-контуру.
  2. DxDyFind / HCoutourDxDyForEachChannel — проецирует смещение центра
     на локальную систему каждого радиального канала.
  3. RadialEvaluation — оценивает ударный объём через радиальную импедансометрию.
  4. TotalModelling — перебирает все комбинации методов.

Публичное API:
    dxdy_find(point1, point2) -> np.ndarray[2]
    h_contour_dxdy_for_each_channel(name, center_move) -> np.ndarray[5, 2]
    equal_sphere_move_modelling(name, sistole_method, eq_sphere_method,
                                *, use_atrium=False) -> np.ndarray[2]
    radial_evaluation(name, sistole_method, eq_sphere_method,
                      rad_eval_method) -> float
    total_modelling(name) -> list[list[list[float]]]
"""

from __future__ import annotations

import numpy as np

from cardio_model.data.moves import edge_move_5pt
from cardio_model.data.mri import (
    R_BY_MRI_BY_VOLUME,
    VALVE_MOVE,
    get_contour_with_atrial,
)
from cardio_model.data.patients import get_contour, get_param, get_radial
from cardio_model.radial import rad_eval_method1, rad_eval_method2, rad_eval_method4
from cardio_model.sistole import sistole_contour
from cardio_model.sphere_fit import equal_sphere
from cardio_model.volume import volume_by_contour

__all__ = [
    "dxdy_find",
    "h_contour_dxdy_for_each_channel",
    "equal_sphere_move_modelling",
    "radial_evaluation",
    "total_modelling",
]


# ---------------------------------------------------------------------------
# DxDyFind — проекция смещения центра на локальную систему канала
# ---------------------------------------------------------------------------


def dxdy_find(
    point1: np.ndarray,
    point2: np.ndarray,
) -> np.ndarray:
    """Проекция вектора point2 в локальную систему, заданную point1.

    Порт ``DxDyFind[point1_, point2_]`` из RadialEvaluation.m::

        angle = |VectorAngle[point1, point2]|
        l     = EuclideanDistance[{0,0}, point2]
        res   = {-l·cos(angle), l·sin(angle)}
        If point2 == {0,0}: res = {0, 0}

    Семантика: l — модуль смещения (|point2|); angle — угол между
    направлением «канал→сердце» (point1) и вектором смещения (point2).
    Компонента вдоль point1 берётся с минусом (удаление), поперечная — с плюсом.

    Parameters
    ----------
    point1 : (2,) float — направляющий вектор канала (контурная точка).
    point2 : (2,) float — смещение центра сферы (мм).

    Returns
    -------
    (2,) float — [dx_local, dy_local].
    """
    p1 = np.asarray(point1, dtype=float)
    p2 = np.asarray(point2, dtype=float)

    if np.allclose(p2, 0.0):
        return np.zeros(2)

    n1 = np.linalg.norm(p1)
    n2 = np.linalg.norm(p2)

    if n1 < 1e-15:
        # Нулевой point1 → неопределённый угол → возвращаем нули
        return np.zeros(2)

    cos_angle = np.clip(np.dot(p1, p2) / (n1 * n2), -1.0, 1.0)
    angle = float(abs(np.arccos(cos_angle)))
    length = n2

    return np.array([-length * np.cos(angle), length * np.sin(angle)])


# ---------------------------------------------------------------------------
# HCoutourDxDyForEachChannel
# ---------------------------------------------------------------------------


def h_contour_dxdy_for_each_channel(
    name: str,
    center_move: np.ndarray,
) -> np.ndarray:
    """Локальные смещения центра для каждого из 5 импедансных каналов.

    Порт ``HCoutourDxDyForEachChannel[name, dxdy]``::

        pointOfContourForChannel = Drop[Drop[HeartContours[name], 1], -1]
        Table[DxDyFind[point, dxdy], {point, ...}]

    Внутренние точки контура (без первой и последней) соответствуют
    позициям 5 радиальных каналов.

    Parameters
    ----------
    name        : str             — "Ivan" / "Alex" / "Artem".
    center_move : (2,) float      — смещение центра эквивалентной сферы (мм).

    Returns
    -------
    (5, 2) float — локальные dx/dy для каждого канала.
    """
    contour = get_contour(name)
    inner_points = contour[1:-1]   # 5 внутренних точек

    return np.array([
        dxdy_find(pt, center_move)
        for pt in inner_points
    ])


# ---------------------------------------------------------------------------
# EqualSphereMoveModelling
# ---------------------------------------------------------------------------


def equal_sphere_move_modelling(
    name: str,
    sistole_method: str,
    eq_sphere_method: str,
    *,
    use_atrium: bool = False,
) -> np.ndarray:
    """Смещение центра эквивалентной сферы при переходе диастола → систола.

    Порт ``EqualSphereMoveModelling[name, sistoleMethod, eqSphereMethod, isUseAtrium]``.

    Алгоритм:
    1. Загрузить контур (с предсердием или без).
    2. Загрузить вектор движения краёв.
    3. Построить систолический контур через ``sistole_contour``.
    4. Подогнать эквивалентную сферу к обоим контурам.
    5. Вернуть разность центров: circle_sistole.center − circle_diastole.center.

    Parameters
    ----------
    name             : str  — "Ivan" / "Alex" / "Artem".
    sistole_method   : str  — "None" или "ValveMove".
    eq_sphere_method : str  — "Use_Contour", "Use_Contour_Radius", ...
    use_atrium       : bool — если True, использовать контур с предсердием (11 точек).

    Returns
    -------
    (2,) float — вектор смещения центра (мм).
    """
    valve_move = VALVE_MOVE[name]
    r_by_volume = R_BY_MRI_BY_VOLUME[name]

    if use_atrium:
        contour = get_contour_with_atrial(name)         # (11, 2)
        move5 = edge_move_5pt(name)
        # Mathematica: HeartEdgeMove5point~Join~{0, 0, 0, 0}
        # inner = drop first+last from 11-point → 9 points
        move = np.concatenate([move5, np.zeros(4)])     # (9,)
    else:
        contour = get_contour(name)                     # (7, 2)
        move = edge_move_5pt(name)                      # (5,)

    sistole_cont = sistole_contour(contour, move, valve_move, sistole_method)

    circle_diastole = equal_sphere(contour,      eq_sphere_method, r_by_volume)
    circle_sistole  = equal_sphere(sistole_cont, eq_sphere_method, r_by_volume)

    center_move = circle_sistole[0] - circle_diastole[0]
    return center_move


# ---------------------------------------------------------------------------
# RadialEvaluation
# ---------------------------------------------------------------------------

_RAD_EVAL_METHODS = {"None", "FirstLayer", "FirstLayer_CenterMove"}


def radial_evaluation(
    name: str,
    sistole_method: str,
    eq_sphere_method: str,
    rad_eval_method: str,
) -> float:
    """Ударный объём через радиальную импедансометрию.

    Порт ``RadialEvaluation[name, sistoleMethod, eqSphereMethod, radEvalMethod]``.

    Алгоритм:
    1. Оценить смещение центра сферы (EqualSphereMoveModelling).
    2. Получить dxdy для каждого канала (HCoutourDxDyForEachChannel).
    3. Вычислить радиальное смещение по нужному методу.
    4. Построить систолический контур (смещение × 1000 с инверсией знака).
    5. Рассчитать объёмы диастолического/систолического контуров.
    6. Вернуть SV = (V_d − V_s) / 2.

    Parameters
    ----------
    name             : str.
    sistole_method   : "None" или "ValveMove".
    eq_sphere_method : "Use_Contour" / "Use_Contour_Radius" / ...
    rad_eval_method  : "None" / "FirstLayer" / "FirstLayer_CenterMove".

    Returns
    -------
    float — ударный объём (мл).
    """
    # Шаг 1–2: смещение центра сферы и локальные dxdy
    center_move = equal_sphere_move_modelling(name, sistole_method, eq_sphere_method)
    dxdy = h_contour_dxdy_for_each_channel(name, center_move)

    # Шаг 3: радиальное смещение.
    # Method1/2 возвращают Round[..., 0.0001]; Method4 округляет только
    # промежуточное dr, а геометрически скорректированный результат — нет.
    model_param = get_param(name)
    exp_data    = get_radial(name)

    if rad_eval_method == "None":
        radial_move = rad_eval_method1(exp_data, model_param)
    elif rad_eval_method == "FirstLayer":
        radial_move = rad_eval_method2(exp_data, model_param)
    elif rad_eval_method == "FirstLayer_CenterMove":
        radial_move = rad_eval_method4(exp_data, model_param, dxdy)
    else:
        raise ValueError(
            f"Unknown RadialEvaluation method: {rad_eval_method!r}. "
            f"Use one of: {sorted(_RAD_EVAL_METHODS)}"
        )

    # Шаг 4: систолический контур (radialMove в метрах → *1000 мм; инверсия знака)
    contour     = get_contour(name)
    valve_move  = VALVE_MOVE[name]
    sistole_cont = sistole_contour(
        contour,
        -radial_move * 1000.0,   # перевод м→мм + инверсия знака (как в Wolfram)
        valve_move,
        sistole_method,
    )

    # Шаг 5–6: объёмы и SV
    v_diastole = volume_by_contour(contour,      h_step=3.0)
    v_sistole  = volume_by_contour(sistole_cont, h_step=3.0)

    return float((v_diastole - v_sistole) / 2.0)


# ---------------------------------------------------------------------------
# TotalModelling
# ---------------------------------------------------------------------------

_TOTAL_SISTOLE_METHODS   = ["None", "ValveMove"]
_TOTAL_EQ_SPHERE_METHODS = ["Use_Contour", "Use_Contour_Radius"]
_TOTAL_RAD_EVAL_METHODS  = ["None", "FirstLayer", "FirstLayer_CenterMove"]


def total_modelling(name: str) -> list[list[list[float]]]:
    """Перебор всех комбинаций методов (TotalModelling).

    Порт ``TotalModelling[name, verbose]`` (verbose=False).

    Перебирает:
        sistole_methods   × eq_sphere_methods × rad_eval_methods
        = 2 × 2 × 3 = 12 комбинаций

    Returns
    -------
    list[list[list[float]]] — вложенный список SV:
        result[i][j][k] = sv при:
            i ∈ _TOTAL_SISTOLE_METHODS,
            j ∈ _TOTAL_EQ_SPHERE_METHODS,
            k ∈ _TOTAL_RAD_EVAL_METHODS.
    """
    result = []
    for sm in _TOTAL_SISTOLE_METHODS:
        row_sm = []
        for esm in _TOTAL_EQ_SPHERE_METHODS:
            row_esm = []
            for rem in _TOTAL_RAD_EVAL_METHODS:
                sv = radial_evaluation(name, sm, esm, rem)
                row_esm.append(sv)
            row_sm.append(row_esm)
        result.append(row_sm)
    return result
