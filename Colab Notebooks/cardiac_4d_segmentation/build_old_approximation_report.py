"""Build the reader-facing 21.12 notebook from completed CT approximation sweeps."""
from __future__ import annotations

import argparse
import base64
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import nbformat as nbf
import numpy as np


METHODS = ["T0","I1","I2","I3","G1","G2","G3a","G3b","G3c","G4a","G4b","G5",
           "L1","L2","L3a","L3b","L3c","L3d","L4a","L4b",
           "M1","M2","M3","M4","M5","M6","M7","M8","M9"]
RFBR = {"M1": (14, 5, 5, 5), "M2": (9, 7, 5, 5), "M3": (5, 16, 4, 5),
        "M4": (5, 16, 4, 5), "M5": (30, 9, 12, 9), "M6": (38, 6, 6, 5)}
NAMES = {"M1":"BOOK", "M2":"TAUB", "M3":"HES", "M4":"SOD", "M5":"FC", "M6":"2-NORM",
         "M7":"эллипсоид xyz", "M8":"эллипсоид свободный", "M9":"алгебраическая сфера"}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def figure(summary, cases, output):
    lookup = {(r["case"], r["method"]): r for r in summary}
    metrics = [
        ("median_dice", "Медиана Dice", "viridis", 0, 1, False),
        ("median_absolute_volume_error_percent", "Медиана |ошибки объёма|, %", "magma_r", None, None, True),
        ("median_center_shift_mm", "Медиана смещения центра, мм", "magma_r", None, None, True),
        ("rms_frontal_displacement_error_mm", "СКО фронтального хода центра, мм", "magma_r", None, None, True),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 13), layout="constrained")
    for ax, (key, title, cmap, lo, hi, lower_better) in zip(axes, metrics):
        data = np.array([[float(lookup[(case, method)][key]) for case in cases] for method in METHODS])
        display = np.log1p(data) if key != "median_dice" else data
        image = ax.imshow(display, aspect="auto", cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title, fontsize=10, pad=10)
        ax.set_xticks(range(len(cases)), [f"Серия {i+1}" for i in range(len(cases))], rotation=35, ha="right")
        ax.set_yticks(range(len(METHODS)), METHODS, fontsize=8)
        ax.set_xticks(np.arange(-.5, len(cases), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(METHODS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        for boundary in (3.5, 11.5, 19.5):
            ax.axhline(boundary, color="#161616", lw=1.3)
        for i, method in enumerate(METHODS):
            for j, case in enumerate(cases):
                value = data[i, j]
                label = f"{value:.2f}" if key == "median_dice" else f"{value:.1f}"
                rgba = image.cmap(image.norm(display[i, j]))
                luminance = .2126*rgba[0] + .7152*rgba[1] + .0722*rgba[2]
                ax.text(j, i, label, ha="center", va="center", fontsize=6.2,
                        color="black" if luminance > .6 else "white")
        # log1p colour transformation applies only to visualization, not printed values.
        cb = fig.colorbar(image, ax=ax, fraction=.035, pad=.02)
        if key != "median_dice":
            ticks = np.linspace(float(np.nanmin(display)), float(np.nanmax(display)), 5)
            cb.set_ticks(ticks)
            cb.set_ticklabels([f"{np.expm1(v):.1f}" for v in ticks])
        cb.ax.tick_params(labelsize=7)
    fig.suptitle("Геометрические показатели на 20 фазах каждой серии", fontsize=14)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def trajectory_figure(rows, cases, output):
    # Representative paths are chosen in advance by family, not by observed score.
    selection = ["T0", "G4b", "L4b", "M1", "M5", "M9"]
    by = {(r["case"], r["method"]): [] for r in rows}
    for r in rows:
        by[(r["case"], r["method"])].append(r)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), layout="constrained")
    palette = {"T0":"#111111", "G4b":"#c62828", "L4b":"#1565c0",
               "M1":"#ef6c00", "M5":"#6a1b9a", "M9":"#2e7d32"}
    for i, (ax, case) in enumerate(zip(axes, cases)):
        for j, method in enumerate(selection):
            values = sorted(by[(case, method)], key=lambda x: float(x["phase_percent"]))
            x = [float(x["dx_ras_mm"]) for x in values]
            y = [float(x["dz_ras_mm"]) for x in values]
            ax.plot(x, y, marker="o" if method == "T0" else None,
                    lw=2.7 if method == "T0" else 1.7,
                    ms=3, alpha=1 if method == "T0" else .9,
                    color=palette[method], label=method)
        ax.axhline(0, color="#888", lw=.5); ax.axvline(0, color="#888", lw=.5)
        ax.set_title(f"Серия {i+1}")
        ax.set_xlabel("Смещение вправо, мм (RAS +x)")
        ax.set_ylabel("Смещение к голове, мм (RAS +z)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=6, fontsize=9)
    fig.suptitle("Фронтальные годографы центров относительно первой фазы")
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def image_cell(path, relative):
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    cell = nbf.v4.new_code_cell(f'from IPython.display import Image, display\ndisplay(Image(filename="{relative}"))')
    cell.execution_count = 1
    cell.outputs = [nbf.v4.new_output("display_data", data={"image/png": data}, metadata={})]
    return cell


def build(args):
    root = Path(args.summary)
    result = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if result["failures"]:
        raise ValueError(f"Cannot report full sweep with {len(result['failures'])} failed fits")
    cases = result["cases"]
    summary = read_csv(root / "method_summary.csv")
    paths = read_csv(root / "trajectories.csv")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    matrix_path, paths_path = figures / "all_methods_matrix.png", figures / "selected_frontal_hodographs.png"
    figure(summary, cases, matrix_path)
    trajectory_figure(paths, cases, paths_path)
    rel_matrix = matrix_path.relative_to(output.parent).as_posix()
    rel_paths = paths_path.relative_to(output.parent).as_posix()
    lookup = {(r["case"], r["method"]): r for r in summary}
    leader_lines = ["| Серия | Индивидуальная форма | Сфера | Эллипсоид | Библиотечная форма |",
                    "|:--|:--|:--|:--|:--|"]
    for i, case in enumerate(cases, 1):
        winners = []
        for group in (("I1","I2","I3"),
                      ("G1","G2","G3a","G3b","G3c","G4a","G4b","G5"),
                      ("L1","L2","L3a","L3b","L3c","L3d","L4a","L4b"),
                      ("M1","M2","M3","M4","M5","M6","M7","M8","M9")):
            winner = max(group, key=lambda method: float(lookup[(case, method)]["median_dice"]))
            winners.append(f"{winner} ({float(lookup[(case, winner)]['median_dice']):.3f})")
        leader_lines.append(f"| Серия {i} | " + " | ".join(winners) + " |")
    def triplet(method, key, digits=3):
        return " / ".join(f"{float(lookup[(case, method)][key]):.{digits}f}" for case in cases)
    analysis = (
        f"Во всех трёх сериях наибольший Dice среди преобразований индивидуальной формы имеет I1 "
        f"({triplet('I1', 'median_dice')}); среди сфер — G4b ({triplet('G4b', 'median_dice')}), "
        f"среди эллипсоидов — L4b ({triplet('L4b', 'median_dice')}), "
        f"среди библиотечных методов — M5/FC ({triplet('M5', 'median_dice')}). "
        f"Однако медиана абсолютной ошибки объёма у G4b составляет "
        f"{triplet('G4b', 'median_absolute_volume_error_percent', 1)}%, "
        f"у L4b — {triplet('L4b', 'median_absolute_volume_error_percent', 1)}%, "
        f"у M5 — {triplet('M5', 'median_absolute_volume_error_percent', 1)}%. "
        "Равновеликие G1 и L1 имеют нулевую ошибку объёма по определению, "
        f"но их Dice ниже: {triplet('G1', 'median_dice')} и "
        f"{triplet('L1', 'median_dice')} соответственно. Следовательно, лидерство "
        "по перекрытию зависит от того, допускается ли ошибка объёма; выбор "
        "следует делать по заранее заданному сочетанию геометрических и электрических критериев. "
        "Нулевое расхождение траекторий I1 и G1 с центром маски задано формулой их построения "
        "и не является независимой проверкой движения."
    )
    target_excursions = []
    for case in cases:
        target_excursions.append(max(np.hypot(float(r["dx_ras_mm"]), float(r["dz_ras_mm"]))
            for r in paths if r["case"] == case and r["method"] == "T0"))
    trajectory_analysis = (
        "Для автоматической маски максимальное фронтальное удаление центра от первой фазы "
        f"равно {' / '.join(f'{x:.1f}' for x in target_excursions)} мм по трём сериям. "
        "Среднеквадратичная ошибка фронтального перемещения относительно этой маски у G4b "
        f"составляет {triplet('G4b', 'rms_frontal_displacement_error_mm', 1)} мм, "
        f"у L4b — {triplet('L4b', 'rms_frontal_displacement_error_mm', 1)} мм, "
        f"у BOOK/M1 — {triplet('M1', 'rms_frontal_displacement_error_mm', 1)} мм. "
        "Следовательно, даже метод с лучшим перекрытием внутри семейства может воспроизводить "
        "пофазный ход центра хуже, чем предполагает его Dice. Эти отклонения получены относительно "
        "автоматической маски крови и не являются ошибкой по отношению к опубликованному электрическому годографу."
    )
    lines = ["| Метод | Семейство | Серия 1 | Серия 2 | Серия 3 |",
             "|:--|:--|--:|--:|--:|"]
    for method in METHODS:
        family = ("анатомическая маска" if method == "T0" else
                  "индивидуальная форма" if method.startswith("I") else
                  "сфера" if method.startswith("G") else
                  "эллипсоид" if method.startswith("L") else
                  NAMES.get(method, "библиотечная форма"))
        values = [float(lookup[(case, method)]["median_dice"]) for case in cases]
        lines.append(f"| {method} | {family} | " + " | ".join(f"{v:.3f}" for v in values) + " |")
    rfbr_lines = ["| Метод в отчёте | До заполнения: вдох / выдох, % | После заполнения: вдох / выдох, % | Наши серии 1 / 2 / 3, % |",
                  "|:--|:--:|:--:|:--:|"]
    for method, values in RFBR.items():
        local = [float(lookup[(case, method)]["median_absolute_volume_error_percent"]) for case in cases]
        rfbr_lines.append(f"| {NAMES[method]} (здесь {method}) | {values[0]} / {values[1]} | {values[2]} / {values[3]} | "
                          + " / ".join(f"{x:.1f}" for x in local) + " |")
    cells = []
    cells.append(nbf.v4.new_markdown_cell("""# 21.12. Полное сравнение аппроксимаций камерной крови на исторической 4D-КТ и результаты РФФИ

**Статус:** вычислительное сравнение геометрических моделей. Цель — определить, какие формы и критерии аппроксимации сохраняют объём, положение и пофазное движение автоматической области крови. Электрическая эквивалентность и тождество исходных исследований РФФИ этим расчётом не устанавливаются.

На трёх исторических сериях 4D-КТ рассмотрено по 20 фаз. Целевая область каждой фазы — объединение четырёх масок камер крови TotalSegmentator; крупные сосуды не включены. Та же область используется для всех методов внутри каждой фазы. Исходные маски не заменены вручную, а перегородки специально не заполнены. Таким образом, полученные величины характеризуют согласие приближений с данной *автоматической* сегментацией, а не доказанную точность относительно истинной анатомии."""))
    cells.append(nbf.v4.new_markdown_cell("""## Постановка и происхождение данных

Для каждой серии используются 20 исходных NIfTI-фаз и по четыре маски камер TotalSegmentator на фазу. Их контрольные суммы, физическая аффинная матрица, конфигурация методов и параметры оптимизации сохранены в локальных протоколах расчёта. Координаты выражены в миллиметрах в системе RAS+: ось x направлена вправо относительно тела, y — кпереди, z — к голове. Фронтальную траекторию образуют компоненты (Δx, Δz) относительно фазы 0%; одинаковая фаза служит началом для всех кандидатов в данной серии.

**Реестр:** автоматическая маска T0 — самосравнение; I1–I3 — преобразования индивидуальной формы первой фазы; G1–G5 (с подвариантами) — восемь сфер; L1–L4 (с подвариантами) — восемь эллипсоидов; M1–M9 — девять отдельных библиотечных подгонок MATLAB. Итого 29 представлений и 1740 пар «фаза–метод». Все 1740 пар получили численную оценку без зарегистрированных отказов; геометрическая корректность найденной формы не является свидетельством глобального минимума или физиологической пригодности. Библиотечные подгонки выполнены по одинаковым детерминированным выборкам поверхности из 8192 точек. Совпадение названия G/L и M не означает совпадения формул. Параметры кандидатов в этом опыте определяются с использованием маски соответствующей фазы: непосредственно по её геометрии или по её объёму и центру. Поэтому результат характеризует способность формы аппроксимировать известную КТ-область, а не способность восстановить неизвестное движение из электрических измерений.

Исторические контурные методы И1–И3 из РФФИ/Тихомирова **не входят в число 29 исполненных методов**. Для их строгого запуска ещё требуется установить плоскость исходного контура, подготовку области и правило задания третьей координаты. Латинские I1–I3, представленные здесь, имеют другое определение. Названия Sphere, Ellipse и MinVolEllipse в РФФИ также не отождествляются автоматически с M7–M9: точная историческая реализация и настройки этих трёх вариантов пока не подтверждены, а минимальный по объёму охватывающий эллипсоид отдельным алгоритмом в этом прогоне не вычислялся. Протоколы в `results/old_approximation_sweep_20260925/` фиксируют исходные хэши и версии алгоритмов."""))
    cells.append(nbf.v4.new_markdown_cell("""## Определения вариантов

Коды указывают способ определения параметров, а не тип электродной сборки. «Фиксированный объём» означает равенство непрерывного объёма модели объёму целевой маски; «фиксированный центр» — совпадение с её центром объёма.

| Код | Геометрическое правило |
|:--|:--|
| T0 | Автоматическая маска той же фазы; внутренний ориентир сравнения. |
| I1 / I2 / I3 | Исходная индивидуальная форма первой фазы: изотропное масштабирование / жёсткий перенос-поворот с масштабированием / аффинное преобразование по моментам. |
| G1 / G2 | Сфера с центром маски и радиусом по равному объёму / по второму моменту. |
| G3a / G3b / G3c | Подгонка сферической поверхности: фиксированный центр / свободный центр / фиксированный объём. |
| G4a / G4b | Максимизация дискретного перекрытия сферы: фиксированный / свободный объём. |
| G5 | Алгебраическая подгонка сферы по точкам поверхности. |
| L1 / L2 | Эллипсоид по вторым моментам: полуоси масштабированы до объёма маски / без этого масштабирования. |
| L3a / L3b / L3c / L3d | Подгонка поверхности эллипсоида: фиксированный центр со свободным объёмом / свободные центр и объём / фиксированные центр и объём / свободный центр при фиксированном объёме. |
| L4a / L4b | Максимизация дискретного перекрытия эллипсоида со свободным центром: фиксированный / свободный объём. |
| M1–M6 | Библиотечная подгонка квадрики: BOOK, TAUB, HES, SOD, FC, 2-NORM соответственно. |
| M7 / M8 / M9 | Библиотечный эллипсоид с осями координат / со свободной ориентацией / алгебраическая сфера. |

Алгоритмические ограничения локальной оптимизации, параметры выборки и точные функции MATLAB сохранены в расчётных протоколах и исходном коде. Методы, оптимизирующие дискретное перекрытие, проверены на выбранной сетке, но сертификата глобального оптимума не имеют."""))
    cells.append(nbf.v4.new_markdown_cell("""## Показатели сравнения

**Коэффициент Dice** равен удвоенному объёму пересечения приближения с маской, делённому на сумму их объёмов. Значение 1 соответствует полному совпадению на сетке исходной КТ. В расчёте эллипсоиды и сферы дискретизированы по центрам вокселей; это численная оценка перекрытия, а не точный интеграл двух непрерывных тел.

**Ошибка объёма**, % = 100 × (объём непрерывного приближения / объём целевой маски − 1). На рисунке показана медиана её абсолютного значения за 20 фаз. Малое значение этого показателя не гарантирует правильного положения или поверхности; у равновеликой сферы G1 оно равно нулю по определению.

**Смещение центра**, мм — евклидово расстояние между центром приближения и центром целевой маски в одной фазе. **Ошибка фронтального хода**, мм — среднеквадратичная за 20 фаз длина разности двух перемещений относительно их собственных центров в фазе 0%. Этот показатель сравнивает форму траектории, устраняя постоянное смещение между моделями. Нулевой ход G1/G2 при одинаковом центре не означает совпадения радиусов.

Все медианы считаются отдельно для каждой серии. Фазы одного человека коррелированы, поэтому 20 фаз не трактуются как 20 независимых наблюдений и здесь не приводятся фиктивные доверительные интервалы."""))
    cells.append(nbf.v4.new_markdown_cell("""## Рисунок 1. Полная матрица геометрического соответствия

Каждая строка — один зарегистрированный метод, каждый столбец в каждой панели — одна и та же серия КТ. В ячейке напечатано численное значение; цвет помогает увидеть различия, но не заменяет числа. Для объёма и смещения цветовая шкала построена по log(1+x), чтобы крупные ошибки не скрывали различия малых; подписи и шкалы приведены в исходных единицах. Чёрные горизонтальные линии отделяют анатомические, сферические, эллипсоидальные и библиотечные представления. В рамках оценки перекрытия более высокий Dice предпочтителен; для трёх остальных показателей предпочтительны малые значения. Это **разные критерии**, и ни один из них отдельно не устанавливает качество электрического сигнала."""))
    cells.append(image_cell(matrix_path, rel_matrix))
    cells.append(nbf.v4.new_markdown_cell("""Проверка всех методов на одной целевой области позволяет увидеть цену выбранного ограничения. Методы с жёстким сохранением объёма могут иметь нулевую ошибку объёма при заметной потере формы. Метод, хорошо приближающий поверхность, может смещать центр. Сравнивать следует весь набор показателей по одной фазе и по одной серии; усреднение по разным людям не заменяет такого сопоставления. Маска T0 имеет единичный Dice и нулевые ошибки **по определению** и не служит внешним анатомическим эталоном. Далее для каждого семейства указан метод с наибольшей медианой Dice по 20 фазам. Это геометрическое описание, а не выбор для электрического восстановления.

""" + "\n".join(leader_lines) + "\n\n" + analysis))
    cells.append(nbf.v4.new_markdown_cell("## Таблица 1. Dice по всем методам и сериям\n\nМедианы по 20 фазам. Названия кодовых методов расшифрованы в реестре выше; у M1–M9 указан конкретный библиотечный алгоритм.\n\n" + "\n".join(lines)))
    cells.append(nbf.v4.new_markdown_cell("""## Рисунок 2. Примеры траекторий центра

Для читаемости показаны заранее заданные представители семейств: T0, G4b, L4b, M1/BOOK, M5/FC и M9. Все 29 методов, включая не показанные отдельными линиями, представлены в последней панели рисунка 1 и в числовой таблице `trajectories.csv`. Все кривые приведены к собственному центру в фазе 0%, а не искусственно совмещены по оптимальной регистрации. Идентичные или почти идентичные траектории могут накладываться друг на друга. Направления осей соответствуют КТ RAS+, а не схеме координат РФФИ."""))
    cells.append(image_cell(paths_path, rel_paths))
    cells.append(nbf.v4.new_markdown_cell(trajectory_analysis))
    rfbr_image = output.parent / "results/nix_cardiac_cycle_fat_20260916/rfbr_xy_hodographs.png"
    if rfbr_image.exists():
        cells.append(nbf.v4.new_markdown_cell("""## Рисунок 3. Опубликованные электрические годографы РФФИ

Фрагменты рисунков 27, 29 и 31 источника показаны отдельно от годографов КТ. В отчёте они получены обратным расчётом по пяти каналам и сглажены. Оси рисунка соответствуют авторской системе РФФИ; её фронтальная плоскость не является плоскостью (x,y) КТ RAS+. Совпадение участников, определения сферы и фаз пока не установлено, поэтому наложение кривых и численная ошибка между ними здесь не вычисляются. Источник: «Отчёт РФФИ 2021 v2.1.pdf», PDF с. 50–53; полный хэш приведён ниже."""))
        cells.append(image_cell(rfbr_image, rfbr_image.relative_to(output.parent).as_posix()))
    cells.append(nbf.v4.new_markdown_cell("""## Сопоставление с опубликованным расчётом РФФИ

В [отчёте РФФИ 2021, раздел 8.5, таблицы 23–24](zotero://open-pdf/library/items/3NAQJNXQ?page=97) девять названных методов сравнивались на моделях крови **четырёх** добровольцев при вдохе и выдохе. После заполнения перегородок двумерной операцией с окружностью радиусом 20 мм значения существенно изменились. Ниже приведены шесть методов, для которых названия совпадают с M1–M6. В двух столбцах РФФИ приведены опубликованные средние ошибки объёма для этих *названий* методов; в последнем — наши медианы абсолютной ошибки за 20 фаз. Историческая версия библиотек и её точные аргументы не установлены; здесь M1–M6 — воспроизводимые современные реализации с собственными хэшами. Столбцы разделены по происхождению и не являются парным тестом воспроизведения.

""" + "\n".join(rfbr_lines)))
    cells.append(nbf.v4.new_markdown_cell("""В [разделе 11 и таблице 34 того же отчёта](zotero://open-pdf/library/items/3NAQJNXQ?page=144) показан **другой пофазный опыт**: 20 кадров на цикл у четырёх добровольцев, исходные 3D-области из Inobitec, предварительное заполнение перегородок и аппроксимация BOOK. Средняя ошибка объёма сферы по добровольцам указана как **18, 14, 14, 18%**, эллипсоида — **3,8, 7,0, 3,7, 3,4%**. Эти значения нельзя вычитать из наших G1, L1 или M1 по строкам: G1/L1 принудительно сохраняют объём, наша область не заполнена по историческому правилу, а соответствие серии и добровольца не документировано. Сравнение показывает, какие величины и методические факторы требуется воспроизвести, но не измеряет погрешность РФФИ.

Годографы [раздела 5.3, рисунки 26–31](zotero://open-pdf/library/items/3NAQJNXQ?page=49) представляют результат **обратной электрической задачи** по пяти импедансным каналам и 25 интервалам цикла, начиная с МРТ-контура. Рисунок 2 настоящего отчёта получен непосредственно из геометрии 4D-КТ. Исторические и наши линии нельзя количественно накладывать до подтверждения пары доброволец–исследование, согласования плоскости и знаков осей, фаз и определения объекта. На обоих рисунках видны перемещения порядка миллиметров и различия формы кривых между сериями, но такое визуальное сходство масштаба остаётся гипотезой, а не проверкой точности восстановления."""))
    cells.append(nbf.v4.new_markdown_cell("""## Вывод и границы применимости

Этот прогон даёт полную геометрическую сравнительную основу для выбора кандидатов следующего **электрического** расчёта. Он отвечает на вопрос, насколько каждая форма сохраняет автоматическую область крови, её объём и движение центра на трёх конкретных 4D-КТ. Численный лидер по Dice или объёму пока не является предпочтительной моделью для TEPC/ТТРКГ: в электрическом сигнале разные участки сердца имеют разный вес, а изменение объёма может смешиваться со смещением центра.

Далее необходимо: (1) проверить чувствительность результата к заполнению перегородок и плотности выборки поверхности; (2) на одних и тех же фазах рассчитать прямой импеданс исходной анатомической формы и каждого допустимого семейства при одинаковой материальной карте и электродах; (3) сопоставить ошибки изменения импеданса и возможность разделить влияние объёма и координат; (4) только при подтверждённом происхождении исторических серий выполнить парное сравнение годографов РФФИ после независимого согласования осей и фаз. До этих проверок речь идёт о разработке и отборе гипотез, а не о валидированной замене сердца сферой или эллипсоидом.

Воспроизведение: `run_old_approximation_sweep.py`, `export_old_surface_clouds.py`, MATLAB `run_heart_library_fits`, `score_old_library_fits.py`, `summarize_old_approximation_sweep.py`, затем этот генератор. Локальная конфигурация путей и исходные КТ исключены из Git. Машиночитаемые сводки: `method_summary.csv`, `trajectories.csv`, `summary.json`; протоколы сохраняют хэши входов и кода. Первоисточник РФФИ: «Отчёт РФФИ 2021 v2.1.pdf», SHA-256 `973704070f9943ddbb83c3f196ca16b9b83fc4ceb4f2784be5cf089fe7a4d2f8`."""))
    notebook = nbf.v4.new_notebook(cells=cells, metadata={"language_info": {"name": "python"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    nbf.validate(notebook)
    nbf.write(notebook, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    build(parser.parse_args())
