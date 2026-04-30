"""Пациентские данные: Ivan, Alex, Artem.

Порт `GetIvanRadial/Param`, `GetAlexRadial/Param`, `GetArtemRadial/Param`
из `Kernel/core/RadialEvaluation.m`, и семиточечных контуров из
`Kernel/core/VolumeCalc.m`.

Структуры данных:
    radial  — импедансные данные (dZRad, zBase, flDZ, flBase)
    params  — геометрические параметры модели (a, b, R, h, x, y, hFat, flSize)
    contour — 7-точечный контур сердца в мм (x, y)

Обращение:
    from cardio_model.data.patients import RADIAL, PARAMS, CONTOURS, get_radial, get_param
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Радиальные (импедансные) данные
# Источник: RadialEvaluation.m :: GetIvanRadial / GetArtemRadial / GetAlexRadial
# ---------------------------------------------------------------------------

RADIAL: dict[str, dict] = {
    "Ivan": {
        # N[{57, 110, 130, 67, 115}/1000]
        "dZRad": np.array([57, 110, 130, 67, 115]) / 1000.0,
        "zBase": np.array([35, 33, 33, 30, 42], dtype=float),
        "flDZ": 0.045,
        "flBase": 35.0,
    },
    "Artem": {
        # N[{40, 55, 110, 83, 76}/1000]
        "dZRad": np.array([40, 55, 110, 83, 76]) / 1000.0,
        "zBase": np.array([70, 67, 58, 67, 76], dtype=float),
        "flDZ": 0.1,
        "flBase": 79.0,
    },
    "Alex": {
        # N[{72, 63, 54, 42, 69}/1000]
        "dZRad": np.array([72, 63, 54, 42, 69]) / 1000.0,
        "zBase": np.array([95, 83, 73, 70, 108], dtype=float),
        "flDZ": 0.125,
        "flBase": 64.0,
    },
}

# ---------------------------------------------------------------------------
# Геометрические параметры модели (5 каналов)
# Источник: RadialEvaluation.m :: GetIvanParam / GetArtemParam / GetAlexParam
# Все размеры в метрах.
# ---------------------------------------------------------------------------

PARAMS: dict[str, dict] = {
    "Ivan": {
        "a":      np.array([0.04,  0.04,   0.05,   0.05,   0.04]),
        "b":      np.array([0.02,  0.02,   0.025,  0.025,  0.02]),
        "R":      np.array([0.051, 0.040,  0.042,  0.042,  0.041]),
        "h":      np.array([0.03,  0.019,  0.022,  0.021,  0.03]),
        "x":      np.array([0.0,   0.0,    0.0,    0.0,    0.0]),
        "y":      np.array([0.047, 0.026,  0.035,  0.048,  0.025]),
        "hFat":   np.array([0.0]),
        "flSize": np.array([0.06,  0.03]),
    },
    "Artem": {
        "a":      np.array([0.04,  0.04,   0.05,   0.05,   0.04]),
        "b":      np.array([0.02,  0.02,   0.025,  0.025,  0.02]),
        "R":      np.array([0.037, 0.032,  0.047,  0.039,  0.037]),
        "h":      np.array([0.031, 0.025,  0.024,  0.025,  0.03]),
        "x":      np.array([0.0,   0.0,    0.0,    0.0,    0.0]),
        "y":      np.array([0.027, 0.04,   0.015,  0.039,  0.027]),
        "hFat":   np.array([0.007]),
        "flSize": np.array([0.06,  0.03]),
    },
    "Alex": {
        "a":      np.array([0.04,  0.04,   0.05,   0.05,   0.04]),
        "b":      np.array([0.02,  0.02,   0.025,  0.025,  0.02]),
        "R":      np.array([0.055, 0.050,  0.042,  0.05,   0.055]),
        "h":      np.array([0.037, 0.037,  0.037,  0.037,  0.037]),
        "x":      np.array([0.0,   0.0,    0.0,    0.0,    0.0]),
        "y":      np.array([0.015, 0.045,  0.04,   0.05,   0.055]),
        "hFat":   np.array([0.013]),
        "flSize": np.array([0.06,  0.03]),
    },
}

# ---------------------------------------------------------------------------
# Семиточечные контуры сердца (координаты в мм)
# Источник: VolumeCalc.m :: ivan7pointContour / alex7pointContour / artem7pointContour
# ---------------------------------------------------------------------------

CONTOURS: dict[str, np.ndarray] = {
    "Ivan":  np.array([[-33, -8], [-32, -24], [11, -51], [58, -37], [63, -11], [36, 27], [27, 29]], dtype=float),
    "Alex":  np.array([[-39, 12], [-40, -10], [-16, -63], [42, -63], [65, -40], [42, 10], [35, 13]], dtype=float),
    "Artem": np.array([[-39, -22], [-37, -27], [12, -47], [67, -42], [76, -11], [41, 29], [31, 32]], dtype=float),
}

_NAMES = frozenset({"Ivan", "Alex", "Artem"})


def get_radial(name: str) -> dict:
    """Вернуть импедансные данные пациента.

    Параметры
    ---------
    name : str
        Один из: "Ivan", "Alex", "Artem".

    Возвращает
    ----------
    dict с ключами "dZRad", "zBase", "flDZ", "flBase".
    """
    if name not in _NAMES:
        raise ValueError(f"Неизвестный пациент {name!r}. Доступны: {sorted(_NAMES)}")
    return RADIAL[name]


def get_param(name: str) -> dict:
    """Вернуть геометрические параметры модели для пациента.

    Параметры
    ---------
    name : str
        Один из: "Ivan", "Alex", "Artem".

    Возвращает
    ----------
    dict с ключами "a", "b", "R", "h", "x", "y", "hFat", "flSize".
    """
    if name not in _NAMES:
        raise ValueError(f"Неизвестный пациент {name!r}. Доступны: {sorted(_NAMES)}")
    return PARAMS[name]


def get_contour(name: str) -> np.ndarray:
    """Вернуть 7-точечный контур сердца пациента (мм).

    Параметры
    ---------
    name : str
        Один из: "Ivan", "Alex", "Artem".

    Возвращает
    ----------
    np.ndarray формы (7, 2) — координаты [[x, y], ...] в мм.
    """
    if name not in _NAMES:
        raise ValueError(f"Неизвестный пациент {name!r}. Доступны: {sorted(_NAMES)}")
    return CONTOURS[name]
