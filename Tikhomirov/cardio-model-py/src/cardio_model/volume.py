"""Расчёт объёма по контуру: усечённый конус (Симпсон), SV.

Порт ``Kernel/core/VolumeCalc.m`` (вычислительная часть).

Публичное API:
    cut_konus_circle_volume(h, d_high, d_low) -> float
    volume_by_section(list_of_section, h) -> float   # в мл
    volume_by_contour(points, h_step=3) -> float     # в мл
    sv_by_contour_and_5move(diastole, move5) -> float
"""

from __future__ import annotations

import numpy as np

from cardio_model.geometry import (
    get_coord_after_move,
    kubic_interpol,
    list_of_parallel_sections,
)

__all__ = [
    "cut_konus_circle_volume",
    "volume_by_section",
    "volume_by_contour",
    "sv_by_contour_and_5move",
]


# ---------------------------------------------------------------------------
# Объём усечённого конуса
# ---------------------------------------------------------------------------


def cut_konus_circle_volume(h: float, d_high: float, d_low: float) -> float:
    """Объём усечённого конуса (кругового сечения).

    Порт ``CutKonusCircleVolume[h, dHigh, dLow]``::

        V = 1/12 · π · h · (dHigh² + dHigh·dLow + dLow²)

    Единицы: если h и d в мм, то V в мм³.
    """
    return np.pi * h / 12.0 * (d_high ** 2 + d_high * d_low + d_low ** 2)


# ---------------------------------------------------------------------------
# Объём по набору сечений
# ---------------------------------------------------------------------------


def volume_by_section(
    list_of_section: list[np.ndarray],
    h: float,
) -> float:
    """Объём по набору параллельных хорд (метод усечённых конусов).

    Порт ``VolumeBySection[listOfSection, h, SimpsonMethod->"Circle"]``.

    Parameters
    ----------
    list_of_section : list[(2,2)] — список хорд (каждая — пара точек).
    h               : float       — расстояние между сечениями (мм).

    Returns
    -------
    float — объём в мл (Round[Total/1000, 1]).
    """
    # Длины хорд
    lengths = np.array([
        np.linalg.norm(section[0] - section[1])
        for section in list_of_section
    ])
    # Попарные соседние сечения → усечённые конусы
    volumes = [
        cut_konus_circle_volume(h, lengths[i], lengths[i + 1])
        for i in range(len(lengths) - 1)
    ]
    return round(float(np.sum(volumes)) / 1000.0, 1)


# ---------------------------------------------------------------------------
# Объём по контуру
# ---------------------------------------------------------------------------


def volume_by_contour(points: np.ndarray, h_step: float = 3.0) -> float:
    """Объём сердца по 7-точечному контуру.

    Порт ``VolumeByContour[points, hStep]``::

        VolumeByContour = VolumeBySection(
            ListOfParallelSectionForContour(KubicInterpol(points), hStep),
            hStep
        )

    Parameters
    ----------
    points : (7, 2) float — 7-точечный контур сердца.
    h_step : float        — шаг между сечениями в мм (default 3).

    Returns
    -------
    float — объём в мл.
    """
    interp = kubic_interpol(points)
    sections = list_of_parallel_sections(interp, step=int(h_step))
    return volume_by_section(sections, h_step)


# ---------------------------------------------------------------------------
# Ударный объём через 5-точечное перемещение
# ---------------------------------------------------------------------------


def sv_by_contour_and_5move(
    diastole: np.ndarray,
    move5: np.ndarray,
) -> float:
    """Ударный объём по диастолическому контуру и 5 смещениям.

    Порт ``SVbyContourAnd5Move[diastole, move5]``::

        move7 = -Join[{First[move5]}, move5, {Last[move5]}]
        sistole = GetCoordAfterMove[diastole, move7]
        SV = N[(VolumeByContour[diastole, 3] - VolumeByContour[sistole, 3]) / 2]

    Примечание: знак результата может быть отрицательным, если move5 представляет
    инвертированные смещения (как при вызове через ``HeartEdgeMove5point``).

    Parameters
    ----------
    diastole : (7, 2) float — диастолический контур.
    move5    : (5,) float   — смещения по 5 радиальным каналам.

    Returns
    -------
    float — ударный объём (мл), знак зависит от знака move5.
    """
    diastole = np.asarray(diastole, dtype=float)
    move5 = np.asarray(move5, dtype=float)

    # move7: дополнить первой и последней точкой, инвертировать
    move7 = -np.concatenate([[move5[0]], move5, [move5[-1]]])
    sistole = get_coord_after_move(diastole, move7)

    vol_d = volume_by_contour(diastole, h_step=3.0)
    vol_s = volume_by_contour(sistole, h_step=3.0)
    return float((vol_d - vol_s) / 2.0)
