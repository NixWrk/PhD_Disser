"""Загрузчик данных движения границ сердца из .txt файлов.

Порт `HeartEdgeMoving`, `HeartEdgeMove5point`, `HeartEdgeMove7point`
из `Kernel/core/VolumeCalc.m`.

Файлы alex.txt / ivan.txt / artem.txt скопированы 1-в-1 из
`WolframMath-master/Kernel/core/`. Формат: пробело/табуляция-разделённые
числа, 5 строк (каналы) × N столбцов (точки времени).

Обращение:
    from cardio_model.data.moves import load_moves, edge_move_5pt, edge_move_7pt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_DATA_DIR = Path(__file__).parent

_FILES: dict[str, str] = {
    "Alex":  "alex.txt",
    "Ivan":  "ivan.txt",
    "Artem": "artem.txt",
}
_NAMES = frozenset(_FILES)


def load_moves(name: str) -> np.ndarray:
    """Загрузить матрицу движения из файла (аналог `HeartEdgeMoving[person]`).

    Параметры
    ---------
    name : str
        Один из: "Ivan", "Alex", "Artem".

    Возвращает
    ----------
    np.ndarray формы (5, N) — движение по 5 каналам в N моментов времени (мм).
    """
    if name not in _NAMES:
        raise ValueError(f"Неизвестный пациент {name!r}. Доступны: {sorted(_NAMES)}")
    path = _DATA_DIR / _FILES[name]
    # Import["...", "Table"] в Mathematica читает whitespace-separated числа
    return np.loadtxt(path, dtype=float)


def edge_move_5pt(name: str) -> np.ndarray:
    """Вернуть вектор перемещений на «конец» движения по 5 каналам.

    Аналог `-Map[Last, HeartEdgeMoving[person]]` в Mathematica —
    берёт последнюю (максимальную) точку каждого канала и инвертирует знак.

    Возвращает
    ----------
    np.ndarray формы (5,) в мм.
    """
    moves = load_moves(name)
    return -moves[:, -1]


def edge_move_7pt(name: str) -> np.ndarray:
    """Вернуть вектор перемещений для 7-точечного контура.

    Аналог `HeartEdgeMove7point[person]` — добавляет первую и последнюю
    точку к 5-точечному вектору (граничное условие: «закреплённые концы»).

    Возвращает
    ----------
    np.ndarray формы (7,) в мм.
    """
    pts = edge_move_5pt(name)
    return np.concatenate([[pts[0]], pts, [pts[-1]]])
