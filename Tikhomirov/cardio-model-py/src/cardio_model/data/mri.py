"""МРТ-данные: контуры с предсердием, движение клапана, R по объёму.

Порт `Kernel/history/MRIdata.m`:
    HeartContoursWithAtrial, AtrialPoints, ValveMove, RbyMRIbyVolume.

Обращение:
    from cardio_model.data.mri import (
        CONTOURS_WITH_ATRIAL, ATRIAL_POINTS,
        VALVE_MOVE, R_BY_MRI_BY_VOLUME,
        get_contour_with_atrial, get_atrial_points,
    )
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Семиточечные контуры сердца + предсердие (координаты в мм)
# Источник: MRIdata.m :: artem7contourWithAtrial / ivan7contourWithAtrial / alex7contourWithAtrial
# Первая и последняя точки совпадают (замкнутый контур).
# ---------------------------------------------------------------------------

CONTOURS_WITH_ATRIAL: dict[str, np.ndarray] = {
    "Ivan": np.array(
        [[-33, -8], [-32, -24], [11, -51], [58, -37], [63, -11],
         [36, 27], [27, 29], [11, 32], [-11, 24], [-30, 7], [-33, -8]],
        dtype=float,
    ),
    "Alex": np.array(
        [[-39, 12], [-40, -10], [-16, -63], [42, -63], [65, -40],
         [42, 10], [35, 13], [20, 17], [-5, 20], [-22, 17], [-39, 12]],
        dtype=float,
    ),
    "Artem": np.array(
        [[-39, -22], [-37, -27], [12, -47], [67, -42], [76, -11],
         [41, 29], [31, 32], [-2, 35], [-31, 32], [-38, 0], [-39, -22]],
        dtype=float,
    ),
}

# ---------------------------------------------------------------------------
# Точки предсердия (координаты в мм)
# Источник: MRIdata.m :: AtrialPoints
# ---------------------------------------------------------------------------

ATRIAL_POINTS: dict[str, np.ndarray] = {
    "Ivan":  np.array([[11, 32], [-11, 24], [-30, 7]], dtype=float),
    "Alex":  np.array([[20, 17], [-5, 20], [-22, 17]], dtype=float),
    "Artem": np.array([[-2, 35], [-31, 32], [-38, 0]], dtype=float),
}

# ---------------------------------------------------------------------------
# Движение клапана (мм) и эквивалентный радиус по объёму МРТ (мм)
# Источник: MRIdata.m :: ValveMove / RbyMRIbyVolume
# ---------------------------------------------------------------------------

VALVE_MOVE: dict[str, float] = {"Alex": 5.0, "Ivan": 5.0, "Artem": 5.0}
R_BY_MRI_BY_VOLUME: dict[str, float] = {"Alex": 50.0, "Ivan": 50.0, "Artem": 50.0}

_NAMES = frozenset({"Ivan", "Alex", "Artem"})


def get_contour_with_atrial(name: str) -> np.ndarray:
    """Вернуть контур сердца с предсердием (11 точек, замкнутый).

    Параметры
    ---------
    name : str
        Один из: "Ivan", "Alex", "Artem".

    Возвращает
    ----------
    np.ndarray формы (11, 2) — координаты [[x, y], ...] в мм.
    """
    if name not in _NAMES:
        raise ValueError(f"Неизвестный пациент {name!r}. Доступны: {sorted(_NAMES)}")
    return CONTOURS_WITH_ATRIAL[name]


def get_atrial_points(name: str) -> np.ndarray:
    """Вернуть 3 точки предсердия (мм).

    Параметры
    ---------
    name : str
        Один из: "Ivan", "Alex", "Artem".

    Возвращает
    ----------
    np.ndarray формы (3, 2) — координаты [[x, y], ...] в мм.
    """
    if name not in _NAMES:
        raise ValueError(f"Неизвестный пациент {name!r}. Доступны: {sorted(_NAMES)}")
    return ATRIAL_POINTS[name]
