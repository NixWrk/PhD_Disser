"""Построение систолического контура из диастолического + смещения + клапанное кольцо.

Порт ``Kernel/part1/SistoleFunction.m``.

Алгоритм:
    1. ``sistole_none`` — сдвигает внутренние точки (без первой и последней)
       на ``move`` в полярных координатах; первая и последняя — фиксированы.
    2. ``valve_point`` — находит новые позиции первой/последней точки при
       смещении клапанного кольца перпендикулярно себе на ``valve_move`` мм.
    3. ``valve_contour_move`` — обёртка: меняет только первую/последнюю точку.
    4. ``sistole_valve_move`` — ``sistole_none`` + ``valve_contour_move``.
    5. ``sistole_contour`` — диспетчер (``"None"`` / ``"ValveMove"``).

Публичное API:
    sistole_none(diastole, move) -> np.ndarray
    valve_point(p_first, p_last, valve_move) -> tuple[np.ndarray, np.ndarray]
    valve_contour_move(contour, valve_move) -> np.ndarray
    sistole_valve_move(diastole, move, valve_move) -> np.ndarray
    sistole_contour(diastole, move, valve_move, method) -> np.ndarray
"""

from __future__ import annotations

import numpy as np

from cardio_model.geometry import get_coord_after_move

__all__ = [
    "sistole_none",
    "valve_point",
    "valve_contour_move",
    "sistole_valve_move",
    "sistole_contour",
]

_SISTOLE_METHODS = {"None", "ValveMove"}


# ---------------------------------------------------------------------------
# SistoleNone
# ---------------------------------------------------------------------------


def sistole_none(
    diastole: np.ndarray,
    move: np.ndarray,
) -> np.ndarray:
    """Систолический контур: внутренние точки сдвигаются, края зафиксированы.

    Порт ``SistoleNone[diastole, move]``::

        conturWithoutValve = Drop[Drop[diastole, 1], -1]
        {pFirst} ∪ GetCoordAfterMove[conturWithoutValve, move] ∪ {pLast}

    Parameters
    ----------
    diastole : (N, 2) float — диастолический контур.
    move     : (N-2,) float — радиальное смещение для внутренних точек (мм).

    Returns
    -------
    (N, 2) float — систолический контур.
    """
    diastole = np.asarray(diastole, dtype=float)
    move = np.asarray(move, dtype=float)

    p_first = diastole[0]
    p_last  = diastole[-1]
    inner   = diastole[1:-1]

    moved_inner = get_coord_after_move(inner, move)

    return np.vstack([p_first, moved_inner, p_last])


# ---------------------------------------------------------------------------
# ValvePoint
# ---------------------------------------------------------------------------


def valve_point(
    p_first: np.ndarray,
    p_last: np.ndarray,
    valve_move: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Сместить первую/последнюю точку клапанного кольца.

    Порт ``ValvePoint[pFirst, pLast, valveMove]``::

        vector    = pFirst − pLast
        Cross[vector] в 2D → {−vy, vx}   (поворот на 90° CCW)
        moveVector = Normalize[Cross[vector]] * valveMove
        {pFirst + moveVector, pLast + moveVector}

    Клапанное кольцо параллельно самому себе смещается перпендикулярно
    вектору ``pFirst − pLast`` на расстояние ``valve_move``.

    Parameters
    ----------
    p_first   : (2,) float — первая точка контура (левый конец клапанного кольца).
    p_last    : (2,) float — последняя точка контура.
    valve_move : float — смещение клапана (мм).

    Returns
    -------
    (new_first, new_last) : два (2,) float — новые позиции крайних точек.
    """
    p_first = np.asarray(p_first, dtype=float)
    p_last  = np.asarray(p_last,  dtype=float)

    vector = p_first - p_last
    # 2D «крест» в Mathematica: Cross[{vx, vy}] = {-vy, vx}
    cross = np.array([-vector[1], vector[0]])
    norm = np.linalg.norm(cross)
    if norm < 1e-15:
        raise ValueError(
            "valve_point: p_first и p_last совпадают — нормальная плоскость не определена."
        )
    move_vector = (cross / norm) * float(valve_move)

    return p_first + move_vector, p_last + move_vector


# ---------------------------------------------------------------------------
# ValveConturMove
# ---------------------------------------------------------------------------


def valve_contour_move(
    contour: np.ndarray,
    valve_move: float,
) -> np.ndarray:
    """Обновить позиции крайних точек контура при движении клапана.

    Порт ``ValveConturMove[contour, valveMove]``::

        {newFirst} ∪ Drop[Drop[contour, 1], -1] ∪ {newLast}

    Внутренние точки остаются без изменений; первая и последняя двигаются
    перпендикулярно линии клапанного кольца.

    Parameters
    ----------
    contour    : (N, 2) float — систолический контур (после ``sistole_none``).
    valve_move : float        — смещение клапана (мм).

    Returns
    -------
    (N, 2) float.
    """
    contour = np.asarray(contour, dtype=float)
    p_first = contour[0]
    p_last  = contour[-1]
    new_first, new_last = valve_point(p_first, p_last, valve_move)
    return np.vstack([new_first, contour[1:-1], new_last])


# ---------------------------------------------------------------------------
# SistoleValveMove
# ---------------------------------------------------------------------------


def sistole_valve_move(
    diastole: np.ndarray,
    move: np.ndarray,
    valve_move: float,
) -> np.ndarray:
    """Систолический контур с движением клапанного кольца.

    Порт ``SistoleValveMove[diastole, move, valveMove]``::

        contour = SistoleNone[diastole, move]
        ValveConturMove[contour, valveMove]

    Parameters
    ----------
    diastole   : (N, 2) float.
    move       : (N-2,) float.
    valve_move : float.

    Returns
    -------
    (N, 2) float.
    """
    contour = sistole_none(diastole, move)
    return valve_contour_move(contour, valve_move)


# ---------------------------------------------------------------------------
# SistoleContour — диспетчер
# ---------------------------------------------------------------------------


def sistole_contour(
    diastole: np.ndarray,
    move: np.ndarray,
    valve_move: float,
    method: str,
) -> np.ndarray:
    """Диспетчер методов построения систолического контура.

    Порт ``SistoleContour[diastole, move, valveMove, method]``.

    Поддерживаемые методы
    ---------------------
    ``"None"``      — ``sistole_none``: только радиальный сдвиг, клапан неподвижен.
    ``"ValveMove"`` — ``sistole_valve_move``: радиальный сдвиг + движение клапана.

    Parameters
    ----------
    diastole   : (N, 2) float.
    move       : (N-2,) float.
    valve_move : float.
    method     : str.

    Returns
    -------
    (N, 2) float — систолический контур.
    """
    if method == "None":
        return sistole_none(diastole, move)
    if method == "ValveMove":
        return sistole_valve_move(diastole, move, valve_move)
    raise ValueError(
        f"Unknown SistoleContour method: {method!r}. "
        f"Use one of: {sorted(_SISTOLE_METHODS)}"
    )
