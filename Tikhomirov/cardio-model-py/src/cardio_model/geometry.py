"""Геометрия контура: полярные сдвиги, кубическая интерполяция, параллельные сечения.

Порт ``Kernel/core/VolumeCalc.m`` (геометрическая часть).

Публичное API:
    get_coord_after_move(dec_coord, move) -> np.ndarray
    kubic_interpol(points, n=30, fringe=None) -> np.ndarray  # 186 точек для 7 узлов
    get_l_point(points, shift) -> np.ndarray | None          # хорда на высоте shift
    list_of_parallel_sections(after_cubic, step) -> list[np.ndarray]
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "get_coord_after_move",
    "kubic_interpol",
    "get_l_point",
    "list_of_parallel_sections",
]

# ---------------------------------------------------------------------------
# Полярный сдвиг контура
# ---------------------------------------------------------------------------


def get_coord_after_move(
    dec_coord: np.ndarray,
    move: np.ndarray,
) -> np.ndarray:
    """Переместить контур: добавить ``move`` к полярным радиусам.

    Порт ``GetCoordAfterMove[decCoord, move]`` из VolumeCalc.m::

        Cartesian → Polar → r += move → Cartesian

    Parameters
    ----------
    dec_coord : (N, 2) float — контур в декартовых координатах.
    move      : (N,) float  — радиальное смещение для каждой точки.

    Returns
    -------
    (N, 2) float — сдвинутый контур.
    """
    dec_coord = np.asarray(dec_coord, dtype=float)
    move = np.asarray(move, dtype=float)

    r = np.hypot(dec_coord[:, 0], dec_coord[:, 1])
    theta = np.arctan2(dec_coord[:, 1], dec_coord[:, 0])
    r_new = r + move
    return np.column_stack([r_new * np.cos(theta), r_new * np.sin(theta)])


# ---------------------------------------------------------------------------
# Кубическая эрмитова интерполяция (KubicInterpol)
# ---------------------------------------------------------------------------

# Граничные условия по умолчанию (fringe): производные на концах
_DEFAULT_FRINGE = np.array([[-0.1, -0.5], [-1.0, 1.0]], dtype=float)


def _build_matrix_m(hords: np.ndarray) -> np.ndarray:
    """Матрица СЛАУ для кубической интерполяции (MatrixM в Wolfram).

    Строки 0 и n-1: единичные (граничные условия применяются после решения).
    Остальные: трёхдиагональная система с весами arc-length.

    Parameters
    ----------
    hords : (n,) float — длины хорд; ``hords[0] = 0``.

    Returns
    -------
    (n, n) float — матрица системы.
    """
    n = len(hords)
    M = np.zeros((n, n))
    M[0, 0] = 1.0
    M[n - 1, n - 1] = 1.0
    for row in range(1, n - 1):
        M[row, row]     = 2.0 * (hords[row] + hords[row + 1])
        M[row, row - 1] = hords[row + 1]   # sub-diagonal
        M[row, row + 1] = hords[row]        # super-diagonal
    return M


def _build_matrix_r(hords: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Правая часть СЛАУ (MatrixR в Wolfram).

    Строки 0 и n-1: ``{1, 1}`` — заглушка; будут заменены fringe после решения.
    Остальные: формула трёхточечного конечного разностного приближения.

    Parameters
    ----------
    hords  : (n,) float
    points : (n, 2) float

    Returns
    -------
    (n, 2) float
    """
    n = len(hords)
    R = np.zeros((n, 2))
    R[0]     = [1.0, 1.0]
    R[n - 1] = [1.0, 1.0]
    for row in range(1, n - 1):
        hi  = hords[row]
        hi1 = hords[row + 1]
        # Wolfram: list[i]^2*(points[i+1]-points[i]) + list[i+1]^2*(points[i]-points[i-1])
        # 0-based row: hords[row]^2*(P[row+1]-P[row]) + hords[row+1]^2*(P[row]-P[row-1])
        R[row] = (3.0 / (hi * hi1)) * (
            hi  ** 2 * (points[row + 1] - points[row])
            + hi1 ** 2 * (points[row]     - points[row - 1])
        )
    return R


def _get_p(
    t: float,
    pts: np.ndarray,
    dp: np.ndarray,
    hords: np.ndarray,
    k: int,
) -> float:
    """Кубическая эрмитова интерполяция в точке t ∈ [0, 1] интервала k.

    Порт ``getP[t, points, dp, tk, k]`` из VolumeCalc.m (1-based k → 0-based k).

    Basis functions (Hermite):
        H00 = 2t³ − 3t² + 1   (left value)
        H10 = t(t−1)² · hk    (left tangent, scaled by chord length)
        H01 = −2t³ + 3t²      (right value)
        H11 = t²(t−1) · hk    (right tangent)
    """
    h = hords[k + 1]   # Mathematica tk[[k+1]] → Python hords[k] (0-based k)
    f1 = 2.0 * t ** 3 - 3.0 * t ** 2 + 1.0
    f2 = -2.0 * t ** 3 + 3.0 * t ** 2
    f3 = t * (t - 1.0) ** 2 * h
    f4 = t ** 2 * (t - 1.0) * h
    return f1 * pts[k] + f2 * pts[k + 1] + f3 * dp[k] + f4 * dp[k + 1]


def kubic_interpol(
    points: np.ndarray,
    n: int = 30,
    fringe: np.ndarray | None = None,
) -> np.ndarray:
    """Кубическая эрмитова интерполяция контура по узловым точкам.

    Порт ``KubicInterpol[points, n, fringe]`` из VolumeCalc.m.
    Дефолтный вызов: ``KubicInterpol[points]`` ↔ ``kubic_interpol(points)``.

    Алгоритм:
    1. Длины хорд ``hords`` (arc-length param).
    2. Трёхдиагональная СЛАУ → производные ``dp`` в узлах.
    3. Переопределить dp[0] = fringe[0], dp[-1] = fringe[1].
    4. Эрмитова интерполяция на равномерной сетке t ∈ [0..1] в каждом интервале.

    Parameters
    ----------
    points : (m, 2) float — узловые точки (например, 7 точек контура сердца).
    n      : int          — количество субинтервалов на каждом интервале (default 30).
    fringe : (2, 2) float — производные на первом и последнем узле;
                            default ``[[-0.1, -0.5], [-1, 1]]``.

    Returns
    -------
    ((m-1)*(n+1), 2) float — интерполированный контур.
    Для m=7, n=30: 6 × 31 = 186 точек (граничные точки дублируются — как в Wolfram).
    """
    points = np.asarray(points, dtype=float)
    if fringe is None:
        fringe = _DEFAULT_FRINGE.copy()
    else:
        fringe = np.asarray(fringe, dtype=float)

    m = len(points)
    # Длины хорд: hords[0]=0, hords[k]=dist(k-1, k) для k=1..m-1
    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
    hords = np.concatenate([[0.0], dists])

    # СЛАУ для производных
    M = _build_matrix_m(hords)
    R = _build_matrix_r(hords, points)
    dp = np.linalg.solve(M, R)

    # Переопределить граничные производные (как dp[[1]]=fringe[[1]] в Wolfram)
    dp[0]    = fringe[0]
    dp[m - 1] = fringe[1]

    pts_x = points[:, 0]
    pts_y = points[:, 1]
    dp_x  = dp[:, 0]
    dp_y  = dp[:, 1]

    # t-значения: {i, 0, 1, 1/n} → linspace(0, 1, n+1) (n+1 точек)
    t_vals = np.linspace(0.0, 1.0, n + 1)

    # Собрать (m-1) × (n+1) таблицы координат
    out_x = np.array([
        [_get_p(t, pts_x, dp_x, hords, k) for t in t_vals]
        for k in range(m - 1)
    ])
    out_y = np.array([
        [_get_p(t, pts_y, dp_y, hords, k) for t in t_vals]
        for k in range(m - 1)
    ])

    # Flatten[outX, 1] + Transpose → (N, 2)
    return np.column_stack([out_x.ravel(), out_y.ravel()])


# ---------------------------------------------------------------------------
# Параллельные сечения контура
# ---------------------------------------------------------------------------


def _l2p(p1: np.ndarray, p2: np.ndarray, x: float, shift: float) -> float:
    """Прямая через p1 и p2, сдвинутая на ``-shift``, вычисленная в точке x.

    Порт ``L2P[{p1, p2}, x, shift]``::

        y = (p2y - p1y)/(p2x - p1x) * (x - p1x) + p1y - shift
    """
    dx = p2[0] - p1[0]
    if abs(dx) < 1e-15:
        return p1[1] - shift    # вертикальная опорная линия — вернуть как есть
    return (p2[1] - p1[1]) / dx * (x - p1[0]) + p1[1] - shift


def get_l_point(
    points: np.ndarray,
    shift: float,
) -> np.ndarray | None:
    """Найти хорду контура на высоте shift от базовой линии.

    Порт ``GetLPoint[list, shift]`` из VolumeCalc.m.

    Базовая линия — прямая через первую и последнюю точки контура.
    Сдвигает её вниз на ``shift`` и находит два пересечения с контуром.
    Возвращает первую и последнюю из найденных граничных точек.

    Parameters
    ----------
    points : (N, 2) float — интерполированный контур.
    shift  : float        — сдвиг базовой линии.

    Returns
    -------
    (2, 2) float — пара точек хорды, или None если пересечений < 2.
    """
    p_first = points[0]
    p_last  = points[-1]

    crossing_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for j in range(len(points) - 1):
        p1, p2 = points[j], points[j + 1]
        f1 = p1[1] - _l2p(p_first, p_last, p1[0], shift)
        f2 = p2[1] - _l2p(p_first, p_last, p2[0], shift)
        if (f1 > 0 and f2 < 0) or (f1 < 0 and f2 > 0):
            crossing_pairs.append((p1, p2))

    if len(crossing_pairs) < 2:
        return None

    # Mathematica: Flatten[point1, 1][[{1, 4}]] — 0-based: flat[0] и flat[3]
    # flat = [pair0_p1, pair0_p2, pair1_p1, pair1_p2, ...]
    flat = [p for pair in crossing_pairs for p in pair]
    if len(flat) < 4:
        return None
    return np.array([flat[0], flat[3]])


def list_of_parallel_sections(
    after_cubic: np.ndarray,
    step: int = 3,
) -> list[np.ndarray]:
    """Список хорд контура на сетке сдвигов.

    Порт ``ListOfParallelSectionForContour[afterCubick, step]``::

        Select[Table[GetLPoint[afterCubick, i], {i, 1, 101, step}], Length[#]>1 &]

    Parameters
    ----------
    after_cubic : (N, 2) float — кубически интерполированный контур.
    step        : int           — шаг сдвига (default 3).

    Returns
    -------
    list[np.ndarray (2, 2)] — только непустые хорды.
    """
    sections = []
    for shift in range(1, 102, step):   # {i, 1, 101, step}
        pt = get_l_point(after_cubic, float(shift))
        if pt is not None:
            sections.append(pt)
    return sections
