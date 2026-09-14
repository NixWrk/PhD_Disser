"""Build the method notebook about replacing a heart by an equivalent sphere.

The scientific prose lives in ``heart_sphere_methods_ru.md``.  This builder only
creates six clearly synthetic mathematical illustrations, replaces the figure
tokens in that prose with notebook attachments, and exports a reader HTML copy.
It does not run FEM or any other numerical solver.
"""

from __future__ import annotations

import argparse
import base64
import html as html_lib
import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEXT = REPO_ROOT / "MATLAB_TRKG4_real_subjects" / "tools" / "heart_sphere_methods_ru.md"
DEFAULT_NOTEBOOK = REPO_ROOT / "Colab Notebooks" / "40.15_Эквивалентная_сфера_сердца_методики_и_проверка.ipynb"
DEFAULT_FIGURE_DIR = REPO_ROOT / "MATLAB_TRKG4_real_subjects" / "output" / "exploratory" / "heart_sphere_methods_figures"

DISCLAIMER = "Схема / синтетический пример; не данные КТ или FEM"
CELL_RE = re.compile(r"(?m)^<!-- CELL -->[ \t]*(?:\r?\n|$)")

PALETTE = {
    "ink": "#243447",
    "muted": "#5c6b73",
    "grid": "#d8e1e8",
    "blue": "#277da1",
    "teal": "#43aa8b",
    "orange": "#f08a5d",
    "gold": "#e9c46a",
    "purple": "#7b61a8",
    "red": "#d1495b",
    "train": "#277da1",
    "frame_validation": "#f4a261",
    "assembly_validation": "#7b61a8",
}

FIGURE_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("@@FIG_GEOMETRY@@", "figure_01_geometry.png", "Синтетический эллипсоид и сфера равного объёма"),
    ("@@FIG_CRITERIA@@", "figure_02_criteria.png", "Сравнение критериев эквивалентного радиуса"),
    ("@@FIG_DYNAMIC@@", "figure_03_dynamic.png", "Строго синтетические Δc_x(τ), V(τ), r(τ)"),
    ("@@FIG_DECOMPOSITION@@", "figure_04_decomposition.png", "Контрольная последовательность геометрий и электрических откликов"),
    ("@@FIG_EQUAL_BASELINE@@", "figure_05_equal_baseline.png", "Одинаковый базовый уровень и разные производные"),
    ("@@FIG_VALIDATION@@", "figure_06_validation.png", "Схема разбиения обучения и проверки"),
)

FIGURE_NUMBERS = {
    "figure_01_geometry.png": 1,
    "figure_02_criteria.png": 2,
    "figure_03_dynamic.png": 4,
    "figure_04_decomposition.png": 5,
    "figure_05_equal_baseline.png": 3,
    "figure_06_validation.png": 6,
}

_PLOT_CACHE = None


def _plotting():
    """Import plotting dependencies lazily and select a headless backend."""

    global _PLOT_CACHE
    if _PLOT_CACHE is None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 10,
                "axes.titlesize": 13,
                "axes.labelsize": 10,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.edgecolor": PALETTE["muted"],
                "axes.labelcolor": PALETTE["ink"],
                "xtick.color": PALETTE["ink"],
                "ytick.color": PALETTE["ink"],
                "figure.dpi": 120,
                "savefig.dpi": 180,
            }
        )
        _PLOT_CACHE = (plt, np, ListedColormap, FancyBboxPatch, Patch, Rectangle)
    return _PLOT_CACHE


def _finish_figure(fig, path: Path) -> None:
    """Add the common provenance line and save a whitespace-safe PNG."""

    plt, *_ = _plotting()
    fig.text(
        0.5,
        0.012,
        DISCLAIMER,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=PALETTE["muted"],
        bbox={"facecolor": "white", "edgecolor": PALETTE["grid"], "boxstyle": "round,pad=0.25"},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.30, facecolor="white")
    plt.close(fig)


def _configure_grid(ax) -> None:
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def _make_geometry(path: Path) -> None:
    plt, np, _, _, Patch, _ = _plotting()
    a, b, c = 42.0, 26.0, 18.0
    r_volume = (a * b * c) ** (1.0 / 3.0)
    u = np.linspace(0.0, 2.0 * np.pi, 80)
    v = np.linspace(0.0, np.pi, 42)
    uu, vv = np.meshgrid(u, v)

    ellipsoid = (
        a * np.cos(uu) * np.sin(vv),
        b * np.sin(uu) * np.sin(vv),
        c * np.cos(vv),
    )
    sphere = (
        r_volume * np.cos(uu) * np.sin(vv),
        r_volume * np.sin(uu) * np.sin(vv),
        r_volume * np.cos(vv),
    )

    fig = plt.figure(figsize=(11.0, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        *ellipsoid,
        color=PALETTE["blue"],
        alpha=0.48,
        linewidth=0,
        antialiased=True,
        shade=True,
    )
    ax.plot_surface(
        *sphere,
        color=PALETTE["orange"],
        alpha=0.34,
        linewidth=0,
        antialiased=True,
        shade=True,
    )
    ax.scatter([0.0], [0.0], [0.0], s=46, color=PALETTE["red"], depthshade=False, zorder=5)
    ax.text2D(
        0.03,
        0.83,
        f"a = 42, b = 26, c = 18 мм\nЦентр: (0, 0, 0) мм\n$r_V=(abc)^{{1/3}}={r_volume:.2f}$ мм",
        transform=ax.transAxes,
        fontsize=10,
        color=PALETTE["ink"],
        bbox={"facecolor": "white", "edgecolor": PALETTE["grid"], "alpha": 0.92, "boxstyle": "round,pad=0.35"},
    )
    ax.set_title("Рисунок 1. Синтетическая геометрия: эллипсоид и сфера равного объёма", pad=18, color=PALETTE["ink"])
    ax.set_xlabel("x, мм\n(условный пример)", labelpad=8)
    ax.set_ylabel("y, мм\n(условный пример)", labelpad=8)
    ax.set_zlabel("z, мм\n(условный пример)", labelpad=8)
    extent = 1.18 * max(a, b, c)
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_zlim(-extent, extent)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except AttributeError:
        pass
    ax.view_init(elev=23, azim=-58)
    ax.legend(
        handles=[
            Patch(facecolor=PALETTE["blue"], edgecolor="none", alpha=0.62, label="Эллипсоид (a, b, c)"),
            Patch(facecolor=PALETTE["orange"], edgecolor="none", alpha=0.62, label="Сфера радиуса $r_V$"),
            Patch(facecolor=PALETTE["red"], edgecolor="none", alpha=0.95, label="Общий центр"),
        ],
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        fontsize=9,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.14)
    _finish_figure(fig, path)


def _make_criteria(path: Path) -> None:
    plt, np, _, _, _, _ = _plotting()
    a, b, c = 42.0, 26.0, 18.0
    r_volume = (a * b * c) ** (1.0 / 3.0)
    m2 = (a**2 + b**2 + c**2) / 5.0
    r_two = np.sqrt(5.0 * m2 / 3.0)
    volume_ratio_r2 = (r_two / r_volume) ** 3
    m2_ratio_rv = (r_volume / r_two) ** 2

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.9))
    fig.suptitle("Рисунок 2. Сравнение сохраняемых характеристик на одном синтетическом эллипсоиде", color=PALETTE["ink"], y=0.98)

    ax = axes[0]
    labels = ["$r_V$\nсохраняемый объём", "$r_2$\nсохраняемый момент $m_2$"]
    values = [r_volume, r_two]
    bars = ax.bar(labels, values, color=[PALETTE["blue"], PALETTE["purple"]], width=0.56)
    ax.set_ylabel("Эквивалентный радиус, мм")
    ax.set_title("Радиусы вычислены аналитически", color=PALETTE["ink"])
    ax.set_ylim(0.0, max(values) * 1.28)
    _configure_grid(ax)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.65, f"{value:.2f} мм", ha="center", va="bottom", color=PALETTE["ink"], fontweight="bold")
    ax.text(
        0.03,
        0.96,
        "$r_V=(abc)^{1/3}$\n"
        "$m_2=\\langle\\|\\mathbf{x}-\\mathbf{c}\\|^2\\rangle=(a^2+b^2+c^2)/5$\n"
        "$r_2=\\sqrt{5m_2/3}$",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": PALETTE["grid"], "boxstyle": "round,pad=0.35"},
    )

    ax = axes[1]
    x = np.arange(2)
    width = 0.34
    volume_characteristic = [1.0, volume_ratio_r2]
    m2_characteristic = [m2_ratio_rv, 1.0]
    bars_v = ax.bar(x - width / 2.0, volume_characteristic, width, color=PALETTE["teal"], label="$V/V_{элл}$")
    bars_m = ax.bar(x + width / 2.0, m2_characteristic, width, color=PALETTE["gold"], label="$m_2/m_{2,элл}$")
    ax.set_xticks(x, ["Сфера $r_V$", "Сфера $r_2$"])
    ax.set_ylabel("Отношение к эллипсоиду, 1")
    ax.set_title("Какая сохраняемая характеристика совпадает точно", color=PALETTE["ink"])
    ax.set_ylim(0.0, max(volume_ratio_r2, 1.0, m2_ratio_rv) * 1.25)
    ax.axhline(1.0, color=PALETTE["ink"], linewidth=1.0, linestyle="--", alpha=0.7)
    _configure_grid(ax)
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    for bars in (bars_v, bars_m):
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=9, color=PALETTE["ink"])
    ax.text(
        0.03,
        0.05,
        "Для эллипсоида $m_2=\\langle\\|\\mathbf{x}-\\mathbf{c}\\|^2\\rangle$;\n"
        "для шара $m_2=3r^2/5$.",
        transform=ax.transAxes,
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": PALETTE["grid"], "boxstyle": "round,pad=0.35"},
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.17, wspace=0.30)
    _finish_figure(fig, path)


def _make_dynamic(path: Path) -> None:
    plt, np, _, _, _, _ = _plotting()
    tau = np.linspace(0.0, 1.0, 360)
    delta_c_x = 4.0 * np.sin(2.0 * np.pi * tau)
    volume = 72.0 + 10.0 * np.sin(2.0 * np.pi * tau - 0.35)
    radius = (3.0 * 1000.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)

    fig, axes = plt.subplots(3, 1, figsize=(10.8, 8.2), sharex=True)
    fig.suptitle(
        "Рисунок 4. Строго синтетический пример: "
        "$\\Delta c_x(\\tau)$, $V(\\tau)$ и $r(\\tau)$",
        color=PALETTE["ink"],
        y=0.985,
    )
    series = [
        (delta_c_x, "$\\Delta c_x(\\tau)$, мм", PALETTE["blue"], "Одна компонента смещения задана синусоидой"),
        (volume, "$V(\\tau)$, мл", PALETTE["orange"], "Объём задан аналитически"),
        (radius, "$r(\\tau)$, мм", PALETTE["purple"], "Радиус выведен из $V(\\tau)$: 1 мл = 1000 мм³"),
    ]
    for ax, (values, ylabel, color, note) in zip(axes, series):
        ax.plot(tau, values, color=color, linewidth=2.4)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.0, 1.0)
        _configure_grid(ax)
        ax.text(
            0.015,
            0.82,
            note,
            transform=ax.transAxes,
            fontsize=9,
            color=PALETTE["muted"],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80},
        )
    axes[-1].set_xlabel("$\\tau$, безразмерная условная фаза")
    axes[0].axhline(0.0, color=PALETTE["muted"], linewidth=0.8, alpha=0.7)
    axes[1].set_ylim(volume.min() - 4.0, volume.max() + 4.0)
    axes[2].set_ylim(radius.min() - 0.5, radius.max() + 0.5)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.14, hspace=0.38)
    _finish_figure(fig, path)


def _make_decomposition(path: Path) -> None:
    plt, _, _, FancyBboxPatch, _, _ = _plotting()
    fig, ax = plt.subplots(figsize=(13.0, 8.0))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    fig.suptitle("Рисунок 5. Контрольные геометрии и сравнение электрических откликов",
                 color=PALETTE["ink"], fontsize=14, y=0.96)

    def panel(x, y, title, formula, note, color):
        ax.add_patch(FancyBboxPatch((x, y), .28, .205,
                    boxstyle="round,pad=0.012", facecolor="white",
                    edgecolor=color, linewidth=1.6))
        ax.text(x+.14, y+.168, title, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color=color)
        ax.text(x+.14, y+.104, formula, ha="center", va="center",
                fontsize=13, color=PALETTE["ink"])
        ax.text(x+.14, y+.037, note, ha="center", va="center",
                fontsize=10, color=PALETTE["muted"])

    panel(.035, .66, "1. Опорная форма", "$H_0$",
          "Исходные объём и центр", PALETTE["ink"])
    panel(.36, .66, "2. Только перенос", "$T H_0$",
          "Объём и форма сохраняются", PALETTE["blue"])
    panel(.685, .66, "3. Только масштабирование", "$D H_0$",
          "Центр сохраняется; объём меняется", PALETTE["teal"])
    panel(.035, .37, "4. Оба преобразования", "$T D H_0$",
          "Сначала D, затем T", PALETTE["orange"])
    panel(.36, .37, "5. Полная форма кадра", "$H_t$",
          "Все различия сегментированных форм", PALETTE["red"])
    panel(.685, .37, "6. Разность откликов",
          "$R_{\\mathrm{shape}}=F(H_t)-F(TDH_0)$",
          "Не воспроизведённый моделью отклик", PALETTE["purple"])

    ax.text(.5, .617,
            "$T$: перенос на $\\mathbf{c}_t-\\mathbf{c}_0$;  "
            "$D$: масштабирование от $\\mathbf{c}_0$ с коэффициентом $(V_t/V_0)^{1/3}$",
            ha="center", va="center", fontsize=11, color=PALETTE["ink"])
    ax.add_patch(FancyBboxPatch((.035, .075), .93, .205,
                 boxstyle="round,pad=0.012", facecolor="#f5f8fa",
                 edgecolor=PALETTE["grid"]))
    ax.text(.5, .238, "Взаимодействие переноса и масштабирования",
            ha="center", va="center", fontsize=11, fontweight="bold", color=PALETTE["ink"])
    ax.text(.5, .182,
            "$I_{TD}=F(TDH_0)-F(TH_0)-F(DH_0)+F(H_0)$",
            ha="center", va="center", fontsize=13, color=PALETTE["ink"])
    ax.text(.5, .108,
            "F — импеданс при фиксированных остальных условиях.\n"
            "Разности откликов не являются отдельно измеренными физиологическими источниками.",
            ha="center", va="center", fontsize=10, color=PALETTE["muted"])
    fig.subplots_adjust(left=.015, right=.985, top=.94, bottom=.07)
    _finish_figure(fig, path)

def _make_equal_baseline(path: Path) -> None:
    plt, np, _, _, _, _ = _plotting()
    p = np.linspace(-1.0, 1.0, 300)
    y_a = 1.0 + 0.35 * p + 0.12 * p**2
    y_b = 1.0 + 1.10 * p + 0.12 * p**2
    tangent_p = np.linspace(-0.34, 0.34, 80)
    tangent_a = 1.0 + 0.35 * tangent_p
    tangent_b = 1.0 + 1.10 * tangent_p

    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    ax.plot(p, y_a, color=PALETTE["blue"], linewidth=2.6, label="$y_A(p)=1+0,35p+0,12p^2$")
    ax.plot(p, y_b, color=PALETTE["orange"], linewidth=2.6, label="$y_B(p)=1+1,10p+0,12p^2$")
    ax.plot(tangent_p, tangent_a, color=PALETTE["blue"], linewidth=1.4, linestyle="--", alpha=0.75, label="$dy_A/dp|_0=0,35$")
    ax.plot(tangent_p, tangent_b, color=PALETTE["orange"], linewidth=1.4, linestyle="--", alpha=0.75, label="$dy_B/dp|_0=1,10$")
    ax.scatter([0.0, 0.0], [1.0, 1.0], s=55, color=[PALETTE["blue"], PALETTE["orange"]], edgecolor="white", linewidth=1.0, zorder=4)
    ax.axvline(0.0, color=PALETTE["muted"], linewidth=0.9, alpha=0.7)
    ax.axhline(1.0, color=PALETTE["muted"], linewidth=0.9, alpha=0.7)
    ax.set_xlabel("Параметр p, безразмерный")
    ax.set_ylabel("Отклик y(p), безразмерный")
    ax.set_title("Рисунок 3. Одинаковый y(0), разный локальный наклон", color=PALETTE["ink"])
    ax.text(
        0.03,
        0.95,
        "$y_A(0)=y_B(0)=1$, но $dy/dp|_{p=0}$ различается",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        color=PALETTE["ink"],
        bbox={"facecolor": "white", "edgecolor": PALETTE["grid"], "boxstyle": "round,pad=0.35"},
    )
    _configure_grid(ax)
    ax.legend(loc="lower right", fontsize=9, frameon=True, ncol=2)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.17)
    _finish_figure(fig, path)


def _make_validation(path: Path) -> None:
    plt, _, _, _, _, Rectangle = _plotting()
    fig = plt.figure(figsize=(13.0, 7.5))
    ax = fig.add_axes((.075, .22, .45, .62))
    info = fig.add_axes((.59, .16, .39, .68))
    fig.suptitle("Рисунок 6. Подгонка и проверка на разных сборках и временных блоках",
                 color=PALETTE["ink"], y=.96, fontsize=14)
    for row in range(4):
        for col in range(6):
            kind, label = ("assembly_validation", "В") if row == 3 else (
                ("frame_validation", "К") if col in (1, 4) else ("train", "О"))
            ax.add_patch(Rectangle((col-.46, row-.40), .92, .80,
                         facecolor=PALETTE[kind], edgecolor="white", linewidth=2))
            ax.text(col, row, label, ha="center", va="center", color="white",
                    fontsize=13, fontweight="bold")
    ax.set(xlim=(-.52, 5.52), ylim=(3.52, -.52))
    ax.set_xticks(range(6), [f"B{i}" for i in range(1, 7)])
    ax.set_yticks(range(4), [f"Сборка {s}" for s in "ABCD"])
    ax.set_xlabel("Временные блоки; внутри блока несколько кадров", labelpad=12)
    ax.set_ylabel("Электродная сборка")
    ax.set_title("Пример разделения электрических откликов", fontsize=12, pad=15)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    info.axis("off")
    info.text(0, .98, "Правило фиксируется до подгонки", va="top",
              fontsize=12, fontweight="bold", color=PALETTE["ink"])
    for y, kind, label in [
        (.85, "train", "О — подгонка: A–C × B1, B3, B4, B6"),
        (.75, "frame_validation", "К — проверка: A–C × B2, B5"),
        (.65, "assembly_validation", "В — новая сборка: D × все блоки")]:
        info.add_patch(Rectangle((0, y-.022), .045, .044,
                       facecolor=PALETTE[kind], transform=info.transAxes))
        info.text(.065, y, label, va="center", fontsize=10, color=PALETTE["ink"])
    info.text(0, .52, "Геометрическая сфера", va="top", color=PALETTE["blue"],
              fontsize=11, fontweight="bold")
    info.text(0, .455, "Параметры определяются по заявленной маске.\n"
              "Электрические отклики не используются для подгонки.",
              va="top", fontsize=10, color=PALETTE["ink"])
    info.text(0, .31, "Электрическая сфера", va="top", color=PALETTE["purple"],
              fontsize=11, fontweight="bold")
    info.text(0, .245, "Проверочные отклики не используются при построении.\n"
              "Они служат для оценки переноса на другие условия.",
              va="top", fontsize=10, color=PALETTE["ink"])
    info.text(0, .09, "Блоки и сборки выбираются заранее.\n"
              "Соседство кадров не обеспечивает независимость.\n"
              "Сравнение внутри модели не заменяет физическую валидацию.",
              va="top", fontsize=9.5, color=PALETTE["muted"])
    _finish_figure(fig, path)

def _make_figures(figure_dir: Path) -> Dict[str, Path]:
    builders: Mapping[str, Callable[[Path], None]] = {
        "figure_01_geometry.png": _make_geometry,
        "figure_02_criteria.png": _make_criteria,
        "figure_03_dynamic.png": _make_dynamic,
        "figure_04_decomposition.png": _make_decomposition,
        "figure_05_equal_baseline.png": _make_equal_baseline,
        "figure_06_validation.png": _make_validation,
    }
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for _, filename, _ in FIGURE_SPECS:
        path = figure_dir / filename
        builders[filename](path)
        paths[filename] = path
    return paths


def _split_markdown_cells(source: str) -> List[str]:
    parts = [part.strip() for part in CELL_RE.split(source) if part.strip()]
    if not parts:
        raise ValueError("Markdown text is empty after splitting on <!-- CELL -->")
    return parts


def _portable_relative(path: Path) -> str | None:
    """Return a repository-relative POSIX path without exposing machine paths."""

    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return None


def _cli_cell_source(text_path: Path, notebook_path: Path, html_path: Path, figure_dir: Path) -> str:
    relative_paths = [_portable_relative(item) for item in (text_path, notebook_path, html_path, figure_dir)]
    if any(item is None for item in relative_paths):
        return (
            "# Воспроизводимость: пути вне репозитория намеренно не встраиваются.\n"
            "# Запустите CLI из корня репозитория с параметрами --text, --notebook,\n"
            "# --html и --figure-dir. Этот код FEM не запускает.\n"
            "print(\"Code-cell содержит только инструкцию сборки отчёта; FEM не запускается.\")"
        )
    text_rel, notebook_rel, html_rel, figures_rel = relative_paths
    command_repr = repr(
        [
            "--text",
            text_rel,
            "--notebook",
            notebook_rel,
            "--html",
            html_rel,
            "--figure-dir",
            figures_rel,
            "--include-cli-cell",
        ]
    )
    return (
        "from pathlib import Path\n"
        "import subprocess\n"
        "import sys\n\n"
        "# Воспроизводим только сборку markdown, PNG-иллюстраций и HTML; FEM не запускается.\n"
        "repo = Path.cwd()\n"
        "while not (repo / 'MATLAB_TRKG4_real_subjects').is_dir() and repo != repo.parent:\n"
        "    repo = repo.parent\n"
        "builder = repo / 'MATLAB_TRKG4_real_subjects' / 'tools' / 'build_heart_sphere_methods_report.py'\n"
        f"arguments = {command_repr}\n"
        "if builder.exists():\n"
        "    subprocess.run([sys.executable, '-B', str(builder), *arguments], cwd=repo, check=True)\n"
        "    print('Отчёт пересобран через CLI; FEM не запускался.')\n"
        "else:\n"
        "    print('Локальный репозиторий не найден; CLI-cell оставлена без запуска.')"
    )


def _new_notebook(text_path: Path, notebook_path: Path, html_path: Path, figure_dir: Path, *, include_cli_cell: bool):
    import nbformat

    source = text_path.read_text(encoding="utf-8")
    # Dollar delimiters survive the notebook Markdown renderer and nbconvert.
    source = source.replace(r"\(", "$").replace(r"\)", "$")
    source = source.replace(r"\[", "$$").replace(r"\]", "$$")
    cells = _split_markdown_cells(source)
    token_counts = {token: source.count(token) for token, _, _ in FIGURE_SPECS}
    missing = [token for token, count in token_counts.items() if count == 0]
    if missing:
        raise ValueError("Markdown is missing required figure token(s): " + ", ".join(missing))

    figure_paths = _make_figures(figure_dir)
    by_token = {
        token: (
            filename,
            f"Рисунок {FIGURE_NUMBERS[filename]}. {alt}",
            figure_paths[filename],
        )
        for token, filename, alt in FIGURE_SPECS
    }
    notebook = nbformat.v4.new_notebook(
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "report_kind": "heart_sphere_methods",
            "scientific_status": "method_development_not_validated",
            "figure_provenance": DISCLAIMER,
            "fem_execution": False,
            "source_format": "UTF-8 Markdown; cells separated by <!-- CELL -->",
        }
    )

    for raw_cell in cells:
        cell_source = raw_cell
        attachments = {}
        for token, (filename, alt, figure_path) in by_token.items():
            if token not in cell_source:
                continue
            cell_source = cell_source.replace(token, f"![{alt}](attachment:{filename})")
            attachments[filename] = {
                "image/png": base64.b64encode(figure_path.read_bytes()).decode("ascii")
            }
        cell = nbformat.v4.new_markdown_cell(cell_source)
        if attachments:
            cell["attachments"] = attachments
        notebook.cells.append(cell)

    if include_cli_cell:
        notebook.cells.append(
            nbformat.v4.new_code_cell(_cli_cell_source(text_path, notebook_path, html_path, figure_dir))
        )
    return notebook, figure_paths


def _validate_html(body: str, notebook) -> None:
    """Ensure the reader export has no code inputs or prompt DOM nodes."""

    tag_patterns = (
        r'<[^>]*class=["\'][^"\']*\bjp-InputArea\b[^"\']*["\']',
        r'<[^>]*class=["\'][^"\']*\bjp-InputPrompt\b[^"\']*["\']',
        r'<[^>]*class=["\'][^"\']*\bjp-OutputPrompt\b[^"\']*["\']',
        r'<[^>]*class=["\'][^"\']*\binput_area\b[^"\']*["\']',
        r'<[^>]*class=["\'][^"\']*\binput_prompt\b[^"\']*["\']',
    )
    for pattern in tag_patterns:
        if re.search(pattern, body, flags=re.IGNORECASE):
            raise RuntimeError("HTML export still contains a notebook input or prompt element")

    decoded = html_lib.unescape(body)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        source = cell.source.strip()
        if source and source in decoded:
            raise RuntimeError("HTML export still contains a code-cell source")

    attachment_count = body.count("data:image/png;base64,")
    if attachment_count < len(FIGURE_SPECS):
        raise RuntimeError(f"HTML export contains {attachment_count} PNG attachments; expected at least {len(FIGURE_SPECS)}")


def _export_html(notebook, html_path: Path) -> None:
    from bs4 import BeautifulSoup
    from nbconvert import HTMLExporter

    exporter = HTMLExporter()
    exporter.exclude_input = True
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    body, _ = exporter.from_notebook_node(notebook)
    soup = BeautifulSoup(body, "html.parser")
    for node in soup.select(".jp-InputArea"):
        if node.select_one(".jp-RenderedMarkdown"):
            node.unwrap()
        else:
            node.decompose()
    for selector in (
        ".jp-InputPrompt",
        ".jp-OutputPrompt",
        ".jp-Collapser",
        ".input_area",
        ".input_prompt",
    ):
        for node in soup.select(selector):
            node.decompose()
    for node in soup.select(".jp-Cell-inputWrapper"):
        node.unwrap()
    body = str(soup)
    _validate_html(body, notebook)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(body, encoding="utf-8", newline="\n")


def build_report(text_path: Path, notebook_path: Path, html_path: Path, figure_dir: Path, *, include_cli_cell: bool = False) -> Dict[str, object]:
    """Build the notebook and HTML report without executing notebook cells."""

    notebook, figure_paths = _new_notebook(
        text_path,
        notebook_path,
        html_path,
        figure_dir,
        include_cli_cell=include_cli_cell,
    )
    import nbformat

    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(nbformat.writes(notebook), encoding="utf-8", newline="\n")
    _export_html(notebook, html_path)
    return {
        "notebook": notebook_path,
        "html": html_path,
        "figure_dir": figure_dir,
        "figures": [path for path in figure_paths.values()],
        "markdown_cells": sum(cell.cell_type == "markdown" for cell in notebook.cells),
        "code_cells": sum(cell.cell_type == "code" for cell in notebook.cells),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", type=Path, default=DEFAULT_TEXT, help="UTF-8 Markdown source")
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK, help="Output .ipynb path")
    parser.add_argument("--html", type=Path, default=None, help="Output reader .html path; defaults to notebook stem")
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR, help="Directory for six generated PNG files")
    parser.add_argument(
        "--include-cli-cell",
        action="store_true",
        help="Add an optional reproducibility code-cell that invokes this builder only",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    html_path = args.html if args.html is not None else args.notebook.with_suffix(".html")
    result = build_report(
        args.text,
        args.notebook,
        html_path,
        args.figure_dir,
        include_cli_cell=args.include_cli_cell,
    )
    printable = {
        key: [str(item) for item in value] if isinstance(value, list) else str(value)
        for key, value in result.items()
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
