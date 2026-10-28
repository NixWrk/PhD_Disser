"""Построение поясняющей схемы методики без выполнения научных расчётов."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parent


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    fig.patch.set_facecolor("white")
    ax.set(xlim=(0, 12.8), ylim=(0, 7.2))
    ax.axis("off")

    def box(x, y, w, h, title, body, *, reference=False):
        ax.add_patch(Rectangle((x, y), w, h, linewidth=1.25,
                              edgecolor="#59636d" if reference else "#233d53",
                              facecolor="#f4f5f6" if reference else "#f0f5f8",
                              linestyle="--" if reference else "-"))
        ax.text(x+w/2, y+h-0.23, title, ha="center", va="top", fontsize=11,
                fontweight="bold", color="#152c3d")
        ax.text(x+w/2, y+0.20, body, ha="center", va="bottom", fontsize=10.4,
                color="#152c3d", linespacing=1.40)

    def arrow(start, end, *, reference=False):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                    mutation_scale=14, linewidth=1.35,
                                    linestyle="--" if reference else "-",
                                    color="#59636d" if reference else "#233d53",
                                    shrinkA=3, shrinkB=3))

    box(0.20, 5.65, 3.45, 1.20, "Одна КТ-фаза",
        "Индивидуальная геометрия\nи тканевые области")
    box(4.65, 5.65, 3.45, 1.20, "Поясная ЭИТ",
        "Исходные электрические\nизмерения и схема электродов")
    box(9.10, 5.65, 3.45, 1.20, "ТТРКГ и ЭКГ",
        "Импедансные сигналы\nи временные признаки")

    box(0.20, 3.35, 3.45, 1.45, "Прямая модель и ткани",
        "Геометрия и контакты\nЭффективные свойства тканей\nЧувствительности измерений")
    box(4.65, 3.35, 3.45, 1.45, "Разделение вкладов",
        "Сердце, мягкие ткани и лёгкое\nСогласование ЭИТ и ТТРКГ\nПроверка различимости")
    box(9.10, 3.35, 3.45, 1.45, "Параметры целого сердца",
        "Объём, положение и движение\nЭффективная проводимость\nОценка неоднозначности")

    box(9.10, 0.70, 3.45, 1.35, "Динамика камер",
        "Преобразование параметров\nи определение механических фаз")
    box(4.65, 0.70, 3.45, 1.35, "УО и ФВ",
        "Диастолический и систолический\nобъёмы камеры\nОшибка и область применимости")
    box(0.20, 0.70, 3.45, 1.35, "4D-КТ при разработке",
        "Проверочный референс\nГеометрия, движение и камеры",
        reference=True)

    arrow((1.925, 5.65), (1.925, 4.80))
    arrow((5.00, 5.65), (3.55, 4.80))
    arrow((10.825, 5.65), (8.05, 4.80))
    arrow((3.65, 4.075), (4.65, 4.075))
    arrow((8.10, 4.075), (9.10, 4.075))
    arrow((10.825, 3.35), (10.825, 2.05))
    arrow((9.10, 1.375), (8.10, 1.375))
    arrow((3.65, 1.375), (4.65, 1.375), reference=True)
    arrow((1.925, 2.05), (1.925, 3.35), reference=True)
    ax.text(1.925, 2.70, "проверка", ha="center", va="center", fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
    ax.text(6.4, 0.12, "Пунктир — проверка при разработке. Остальные стрелки — исследуемые преобразования.",
            ha="center", va="bottom", fontsize=10, color="#3e4750")
    fig.subplots_adjust(left=0.015, right=0.985, top=0.98, bottom=0.04)
    out = ROOT/"figures"/"method"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out/f"01_method_sequence.{ext}", dpi=220, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
