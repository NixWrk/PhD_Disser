"""Create a reader-facing notebook for the Tikhomirov 2D-fit CT adaptation."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import nbformat as nbf
import numpy as np
import pandas as pd

from run_tikhomirov_ct_contours import (HERE, METHODS, PLANES, SCOPES,
    equiarc, load_phase, planar_centroid, read_json, silhouette, voxel_centroid)

RESULTS = HERE / "results/tikhomirov_audit_20260928"
FIT = RESULTS / "fit_v2"
FIGS = RESULTS / "figures"
FIGS.mkdir(exist_ok=True)
CASES = ("historical_01_sequence_02", "historical_02_sequence_11", "historical_03_sequence_11")
SHORT = {case: f"Архивная серия {i}" for i, case in enumerate(CASES, 1)}
COLORS = {"CT": "#252525", "LS128": "#2456a6", "Area128": "#df7f20",
          "LS7_uniform": "#8a409d", "FixedCenter128": "#4f9e84"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def draw_contours(rows: pd.DataFrame, config: dict) -> Path:
    case = config["cases"][0]
    sequence, masks = Path(case["sequence_dir"]), Path(case["masks_dir"])
    manifest = read_json(sequence / "conversion_manifest.json")
    status_path = masks / "status/run_status.json"
    if not status_path.exists():
        status_path = masks / "run_status.json"
    status = read_json(status_path)
    phase = sorted(manifest["phases"], key=lambda p: p["phase_percent"])[0]
    masks_by_scope, affine, _ = load_phase(case, phase, status)
    mask = masks_by_scope["blood_four_chambers"]
    ct = voxel_centroid(mask, affine)
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.3), layout="constrained")
    for ax, plane, title in zip(axs, PLANES, ("Сагиттальная проекция КТ", "Фронтальная проекция КТ")):
        contour, pix, hull = silhouette(mask, affine, plane)
        cut = rows[(rows.case == case["case_id"]) & (rows.scope == "blood_four_chambers") &
                   (rows.plane == plane) & (rows.phase == "phase_00")]
        ax.plot(contour[:, 0], contour[:, 1], color="#333333", lw=1.7, label="Внешняя огибающая")
        ax.scatter(*equiarc(contour, 7).T, color=COLORS["LS7_uniform"], s=24, zorder=3,
                   label="7 равномерных точек")
        for method, style in (("LS128", "--"), ("Area128", ":")):
            row = cut[cut.method == method].iloc[0]
            center = (row.center_1_mm, row.center_2_mm)
            ax.add_patch(Circle(center, row.radius_mm, fill=False, color=COLORS[method],
                                ls=style, lw=1.9, label=method))
            ax.scatter(*center, color=COLORS[method], s=35, zorder=4)
        ax.scatter(*planar_centroid(ct, plane), marker="D", color="black", s=42,
                   zorder=5, label="Объёмный центр КТ")
        ax.set(title=title, xlabel="К голове, мм" if plane == "sagittal" else "Вправо, мм",
               ylabel="Кпереди, мм" if plane == "sagittal" else "К голове, мм")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=.15)
        ax.text(.02, .02, f"Площадь огибающей / проекции = {hull/pix:.2f}",
                transform=ax.transAxes, fontsize=8, va="bottom")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="upper center", ncol=5,
               bbox_to_anchor=(.5, 1.045))
    path = FIGS / "ct_contour_initial.png"
    fig.savefig(path, dpi=175, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_trajectories(rows: pd.DataFrame) -> Path:
    fig, axs = plt.subplots(3, 2, figsize=(11.3, 13.0), layout="constrained")
    for i, case in enumerate(CASES):
        for j, plane in enumerate(("frontal", "sagittal")):
            ax = axs[i, j]
            data = rows[(rows.case == case) & (rows.scope == "blood_four_chambers") &
                        (rows.plane == plane)]
            for method, label, xcol, ycol in (("LS128", "Контур: МНК, 128 точек",
                                                "center_1_mm", "center_2_mm"),
                                               ("Area128", "Контур: равная площадь",
                                                "center_1_mm", "center_2_mm"),
                                               ("LS128", "Центр объёма КТ",
                                                "ct_centroid_1_mm", "ct_centroid_2_mm")):
                d = data[data.method == method].sort_values("phase_percent")
                xy = d[[xcol, ycol]].to_numpy()
                xy = xy - xy[0]
                color = COLORS["CT"] if label == "Центр объёма КТ" else COLORS[method]
                lw = 2.1 if label == "Центр объёма КТ" else 1.5
                ax.plot(xy[:, 0], xy[:, 1], "-o", ms=3.4, lw=lw, color=color, label=label)
                ax.scatter(xy[0, 0], xy[0, 1], marker="s", color=color, s=65, zorder=5)
            ax.axhline(0, color="#bbbbbb", lw=.7)
            ax.axvline(0, color="#bbbbbb", lw=.7)
            ax.set(title=f"{SHORT[case]} · {'фронтальная' if plane == 'frontal' else 'сагиттальная'}",
                   xlabel="Δ вправо, мм" if plane == "frontal" else "Δ к голове, мм",
                   ylabel="Δ к голове, мм" if plane == "frontal" else "Δ кпереди, мм")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=.18)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="best")
    path = FIGS / "ct_hodograph_comparison.png"
    fig.savefig(path, dpi=175)
    plt.close(fig)
    return path


def draw_matrix(summary: pd.DataFrame) -> Path:
    methods = ("LS128", "Area128", "FixedCenter128", "LS7_uniform")
    fig, axs = plt.subplots(1, 2, figsize=(13.0, 4.8))
    fig.subplots_adjust(left=.13, right=.89, bottom=.18, top=.84, wspace=.18)
    for ax, plane, title in zip(axs, ("frontal", "sagittal"),
                                ("Фронтальная проекция", "Сагиттальная проекция")):
        matrix = np.array([[summary[(summary.case == case) &
                                     (summary.scope == "blood_four_chambers") &
                                     (summary.plane == plane) &
                                     (summary.method == method)].iloc[0]
                            .rms_displacement_difference_mm for method in methods]
                           for case in CASES])
        im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=3.2, aspect="auto")
        ax.set(xticks=np.arange(len(methods)), yticks=np.arange(3),
               xticklabels=("МНК\n128", "Площадь\n128", "Центр\n128", "МНК\n7"),
               yticklabels=[SHORT[c] for c in CASES] if plane == "frontal" else ["", "", ""],
               title=title)
        for i in range(3):
            for j in range(len(methods)):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=9)
    cax = fig.add_axes((.915, .19, .02, .62))
    fig.colorbar(im, cax=cax, label="Различие смещений, мм")
    path = FIGS / "ct_displacement_difference_matrix.png"
    fig.savefig(path, dpi=175, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_radius(rows: pd.DataFrame) -> Path:
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4), layout="constrained")
    for ax, plane, title in zip(axs, ("frontal", "sagittal"),
                                ("Фронтальная проекция", "Сагиттальная проекция")):
        for i, case in enumerate(CASES):
            d = rows[(rows.case == case) & (rows.scope == "blood_four_chambers") &
                     (rows.plane == plane) & (rows.method == "LS128")].sort_values("phase_percent")
            ax.plot(d.phase_percent, d.radius_mm - d.radius_mm.iloc[0], "-o", ms=3.3,
                    label=SHORT[case], color=plt.get_cmap("tab10")(i))
        ax.set(title=title, xlabel="Фаза КТ-реконструкции, %", ylabel="Δ радиуса 2D-окружности, мм")
        ax.grid(alpha=.18)
    axs[0].legend(fontsize=8)
    path = FIGS / "ct_circle_radius_changes.png"
    fig.savefig(path, dpi=175)
    plt.close(fig)
    return path


def draw_scope_effect(summary: pd.DataFrame) -> Path:
    fig, axs = plt.subplots(1, 2, figsize=(11.8, 4.8), layout="constrained")
    labels = ("Кровь 4 камер · КТ", "Кровь 4 камер · окружность",
              "Желудочки · КТ", "Желудочки · окружность")
    colors = ("#343434", "#2456a6", "#929292", "#df7f20")
    x = np.arange(3)
    width = .19
    for ax, plane, title in zip(axs, ("frontal", "sagittal"),
                                ("Фронтальная проекция", "Сагиттальная проекция")):
        values = []
        for case in CASES:
            four = summary[(summary.case == case) & (summary.scope == "blood_four_chambers") &
                           (summary.plane == plane) & (summary.method == "LS128")].iloc[0]
            vent = summary[(summary.case == case) & (summary.scope == "ventricles") &
                           (summary.plane == plane) & (summary.method == "LS128")].iloc[0]
            values.append((four.ct_max_excursion_mm, four.fit_max_excursion_mm,
                           vent.ct_max_excursion_mm, vent.fit_max_excursion_mm))
        values = np.asarray(values)
        for j, (label, color) in enumerate(zip(labels, colors)):
            bars = ax.bar(x + (j - 1.5) * width, values[:, j], width,
                          color=color, label=label)
            ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7)
        ax.set(title=title, xticks=x, xticklabels=[SHORT[c] for c in CASES],
               ylabel="Максимальное смещение от первой фазы, мм")
        ax.grid(axis="y", alpha=.17)
        ax.set_axisbelow(True)
        ax.set_ylim(0, float(values.max()) * 1.25)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=4,
               fontsize=8, bbox_to_anchor=(.5, 1.045))
    path = FIGS / "ct_scope_effect.png"
    fig.savefig(path, dpi=175, bbox_inches="tight")
    plt.close(fig)
    return path


def image_cell(path: Path):
    source = "from IPython.display import Image, display\n" + \
             f"display(Image(filename='results/tikhomirov_audit_20260928/figures/{path.name}'))"
    cell = nbf.v4.new_code_cell(source)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    cell.outputs = [nbf.v4.new_output("display_data", data={"image/png": encoded}, metadata={})]
    cell.execution_count = 1
    return cell


def report() -> Path:
    rows = pd.read_csv(FIT / "phase_fits.csv", encoding="utf-8-sig")
    summary = pd.read_csv(FIT / "summary.csv", encoding="utf-8-sig")
    assert len(rows) == 960 and len(summary) == 48
    config = read_json(RESULTS / "input.local.json")
    figures = [draw_contours(rows, config), draw_trajectories(rows),
               draw_matrix(summary), draw_scope_effect(summary), draw_radius(rows)]
    blood = summary[(summary.scope == "blood_four_chambers") & (summary.method == "LS128")]
    frontal = blood[blood.plane == "frontal"].set_index("case").loc[list(CASES)]
    sagittal = blood[blood.plane == "sagittal"].set_index("case").loc[list(CASES)]
    seven = summary[(summary.scope == "blood_four_chambers") &
                    (summary.method == "LS7_uniform")].set_index(["case", "plane"])
    dense = blood.set_index(["case", "plane"])
    largest_sample_change = float(np.max(np.abs(seven.rms_displacement_difference_mm -
                                             dense.rms_displacement_difference_mm)))
    table = ["| Серия | КТ, макс. смещение во фронтальной плоскости, мм | МНК-128, макс. смещение, мм | Различие траекторий во фронтальной плоскости, мм | Различие траекторий в сагиттальной плоскости, мм |",
             "|:--|--:|--:|--:|--:|"]
    for case in CASES:
        f, s = frontal.loc[case], sagittal.loc[case]
        table.append(f"| {SHORT[case]} | {f.ct_max_excursion_mm:.2f} | {f.fit_max_excursion_mm:.2f} | {f.rms_displacement_difference_mm:.2f} | {s.rms_displacement_difference_mm:.2f} |")
    nb = nbf.v4.new_notebook()
    nb.metadata.update({"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                        "language_info": {"name": "python"}})
    cells = [nbf.v4.new_markdown_cell("""# 21.13. Контурные сферы Тихомирова и годографы исторической 4D-КТ

**Вычислительный эксперимент; разработка методики.** Проверено, как математические способы подгонки двумерной окружности, сохранённые в коде Тихомирова, ведут себя на пофазных контурах крови сердца из архивной 4D-КТ. Расчёт позволяет сопоставить *геометрические* траектории центров разных определений на одних и тех же изображениях. Он не восстанавливает центр из электроимпедансных сигналов и не устанавливает совпадение с опубликованными годографами РФФИ.

**Главное наблюдение.** Центр окружности по контуру и объёмный центр маски дают различающиеся траектории, особенно во фронтальной проекции. Следовательно, выбор геометрического определения центра сам по себе влияет на годограф; совпадение или расхождение с электрическим годографом нельзя приписывать только качеству обратной задачи."""),
             nbf.v4.new_markdown_cell(r"""## Историческая методика и граница её восстановления

В диссертации Тихомирова единая сфера представляла кровь желудочков вместе с кровью предсердий. Предсистолический контур строили по сагиттальному МРТ-изображению и проектировали относительно электродов. Из десяти характерных точек семь описывали желудочки: две точки клапанного кольца и пять пересечений контура с осями электродных систем; ещё три относились к предсердиям. Сравнивали (1) совместную подгонку центра и радиуса методом наименьших квадратов радиальных отклонений и (2) центр площади контура с радиусом окружности равной площади. В исходном эксперименте автор предпочёл аппроксимацию общей крови желудочков и предсердий методом наименьших квадратов по критерию перемещения границы; этот вывод относится к тем МРТ и электродным осям. Для проверки движения использовали пять импедансных каналов, параметры мягких тканей и минимум суммы квадратов разности модельных и измеренных изменений импеданса. Весь кардиоцикл делили на 25 интервалов. [Диссертация, § 4.2–4.5](zotero://open-pdf/library/items/NUZKBZGS?page=75); [отчёт РФФИ, § 5.2–5.3](zotero://open-pdf/library/items/3NAQJNXQ?page=46).

**Прямая и обратная задачи не тождественны геометрической подгонке контура.** Основная аналитическая электрическая модель использовала однородное полупространство с эффективным сопротивлением $\rho_1$ и сферу крови с сопротивлением $\rho_2$. Для каждой фазы электрическая обратная задача подбирала изменения радиуса и трёх координат центра по пяти измеренным изменениям импеданса; по 25 оценкам центра строился годограф. Отдельный канал кровенаполнения мягких тканей использовался для поправки на изменение внешней среды. В модели отсутствуют отдельные объёмы костей и лёгких, а также раздельные области мышц и миокарда. Это *однослойность геометрии внешней среды*, несмотря на наличие двух электрических материалов — среды и сферы. [Диссертация, § 2.1 и 4.4–4.5](zotero://open-pdf/library/items/NUZKBZGS?page=42); [код прямых моделей](../../Tikhomirov/WolframMath-master/Kernel/core/Models.m).

**Обоснование и граница допущения.** Автор сообщает об измерениях *на выдохе* и мотивирует объединение окружающих тканей тем, что сопротивление лёгких на выдохе приближается к сопротивлениям миокарда и мышц. В его таблице для 100 кГц приведены 3,68 Ом·м для лёгких на выдохе, 4,6 Ом·м для миокарда и среднее 2,7 Ом·м для мышц; для последующих расчётов мягких тканей принято эффективное значение 4,2 Ом·м. **Равенство проводимостей не утверждалось.** Автор настоящего проекта уточнил, что измерения проводились *на глубоком выдохе*; в проверенном фрагменте диссертации глубина выдоха не указана, поэтому это уточнение требует проверки по протоколу. Отдельно исследована двуслойная постановка с жиром и сферой в ANSYS, но сохранённые аналитические функции `TwoLayerModel` и `SphereModel` не составляют единого решателя полной двуслойной обратной задачи со сферой. Следовательно, опубликованный годограф нельзя считать прямой траекторией анатомического центра или доказательством пренебрежимо малого влияния неоднородных тканей. [Диссертация, § 2.1 и 2.3](zotero://open-pdf/library/items/NUZKBZGS?page=42); [правила сопоставления](COORDINATE_CONVENTIONS_HODOGRAPHS.md).

Исходный Wolfram-код `SphereMovingFunction.m` действительно содержит критерий

$$E(\mathbf c,r)=\sum_{i=1}^{n}(\|\mathbf p_i-\mathbf c\|-r)^2,$$

а также варианты с фиксированным центром и радиусом по площади. В Python-переносе `cardio-model-py` сохранены эти операции. На текущем окружении его проверка дала **235 пройденных тестов**; это проверка переноса вычислительного ядра и исторических численных примеров, а не проверка его работы на наших КТ или электродных сборках. Просмотренный код не содержит завершённого совместного пофазного восстановления $(\Delta r,\Delta x,\Delta y,\Delta z)$ из пяти каналов по критерию диссертации. В `IMPdata.m` имеются 25 значений пятиканального сигнала и таблично сохранённая последовательность параметров для одного исторического примера, но отсутствует подтверждённая связь этих чисел с тремя сериями КТ, исследуемыми ниже."""),
             nbf.v4.new_markdown_cell("""## Что именно перенесено на 4D-КТ

**Измеренные данные:** исходные пофазные КТ трёх архивных серий, по 20 реконструкций цикла. **Автоматически рассчитанные данные:** маски крови четырёх камер TotalSegmentator v2, их объединение и отдельно объединение желудочков. Перегородки специальной морфологической процедурой не заполнялись. Перед использованием каждой фазы проверены хэши КТ и всех четырёх камерных масок, совпадение аффинного преобразования и полнота фаз. Это контроль файлов и координат, а не независимая валидация точности каждой границы.

Для каждой фазы маску проектировали на сагиттальную плоскость $(S,A)$ или фронтальную $(R,S)$ в системе NIfTI RAS+. Чтобы получить единую внешнюю огибающую отделённых камер, брали выпуклую оболочку проекции. Её границу равномерно пересэмплировали по длине дуги на 128 точек; дополнительный вариант использует 7 равномерных точек. Затем вызывали функции существующего Python-переноса `EqualSphere`: совместная МНК-подгонка, центр и радиус по площади, а также подгонка радиуса при фиксированном центре площади. Начало локальных координат было одним и тем же для всех фаз серии — объёмный центр маски первой фазы. Параметрические границы исторического переносного кода сохранены.

**Ключевое допущение:** такая внешняя оболочка и равномерные точки заменяют ручные МРТ-ориентиры лишь для проверки математического критерия. Они не воспроизводят исходный выбор пяти электродных осей, конкретный срез и исходную процедуру получения десяти точек. Выпуклая оболочка может добавлять площадь и не является исторической процедурой заполнения перегородок. Радиус окружности по двумерной проекции не обязан сохранять объём трёхмерной крови."""),
             nbf.v4.new_markdown_cell("""## Рисунок 1. Почему различаются центры

Показана первая фаза первой архивной серии. Тёмная линия — внешняя огибающая проекции четырёх камер; синий пунктир — окружность наименьших квадратов; оранжевый пунктир — окружность равной площади; чёрный ромб — объёмный центр исходной маски. Фиолетовые точки показывают вариант с семью *равномерными*, а не историческими электродными ориентирами. Сагиттальная и фронтальная проекции дают разные двумерные объекты, поэтому центры и радиусы нельзя переносить между ними."""),
             image_cell(figures[0]),
             nbf.v4.new_markdown_cell(r"""## Показатель сопоставления траекторий

Для каждого способа центр проекции $\mathbf c_k(t)$ сравнивался с проекцией объёмного центра маски $\mathbf m(t)$ после вычитания собственного начального положения. Показана среднеквадратичная **разница смещений**:

$$D=\sqrt{\frac{1}{N}\sum_t\left\|[\mathbf c_k(t)-\mathbf c_k(0)]-[\mathbf m(t)-\mathbf m(0)]\right\|^2}.$$

Отдельно рассчитано смещение между абсолютными центрами. Эти числа характеризуют **различие двух определений центра на одной автоматической маске**, а не ошибку относительно независимого истинного положения сердца. Нулевая разница не доказывала бы электрической эквивалентности сферической формы."""),
             nbf.v4.new_markdown_cell("""## Рисунок 2. Полученные по КТ годографы

Линии показывают смещение относительно первой фазы каждой серии. Квадраты отмечают начало реконструированного цикла; последующие точки идут через 5% диапазона фаз. Чёрный годограф относится к объёмному центру крови, синий — к окружности наименьших квадратов, оранжевый — к окружности равной площади. Все кривые в каждой панели получены **из одной и той же маски**; различие обусловлено определением центра и проекцией. Связывать их физиологические отметки с R-зубцом отчёта РФФИ без фазовой регистрации нельзя."""),
             image_cell(figures[1]),
             nbf.v4.new_markdown_cell("""## Численный результат

""" + "\n".join(table) + "\n\nДля совместной МНК-подгонки на 128 точках различие смещений во фронтальной плоскости составило **" +
             f"{frontal.rms_displacement_difference_mm.min():.2f}–{frontal.rms_displacement_difference_mm.max():.2f} мм**; в сагиттальной — **{sagittal.rms_displacement_difference_mm.min():.2f}–{sagittal.rms_displacement_difference_mm.max():.2f} мм**. " +
             "Во всех трёх сериях максимальный размах фронтального смещения центра МНК-окружности меньше размаха объёмного центра маски. Это не доказательство занижения электрически восстановленного движения: сравниваются разные геометрические определения. " +
             f"Замена 128 точек семью равномерными меняла показатель $D$ не более чем на {largest_sample_change:.2f} мм в изученных сериях; исходные семь точек Тихомирова были привязаны к электродным осям, поэтому этот тест не оценивает именно их влияние. Все 960 пофазных вариантов вычислены; в совместной МНК-подгонке верхняя граница радиуса исторического кода не достигалась."),
             nbf.v4.new_markdown_cell("""## Рисунок 3. Различие смещений для всех контурных вариантов

Цвет и подписанное значение показывают $D$ в миллиметрах; шкала общая для двух панелей. «Площадь-128» и «Центр-128» имеют одинаковую траекторию центра **по определению**: оба фиксируют центр площади контура, но дают различный радиус. Поэтому одна только матрица годографов не может выбрать между этими вариантами. Для такого выбора необходим анализ изменения радиуса, формы и затем электрического отклика."""),
             image_cell(figures[2]),
             nbf.v4.new_markdown_cell("""## Рисунок 4. Что меняется при исключении предсердий

Для каждой серии показан максимальный размах от первой фазы отдельно для объёмного центра КТ и центра окружности МНК-128. Чёрные и синие столбцы относятся к объединённой крови четырёх камер, серые и оранжевые — только к крови желудочков. В четырёхкамерной области окружность во всех трёх сериях имеет меньший фронтальный размах, чем объёмный центр; в желудочковой области — больший. Это демонстрирует сильную зависимость закона движения от **определения замещаемой области**. Указанные размахи принадлежат разным объектам; меньшая разница между столбцами сама по себе не доказывает физиологического или электрического преимущества метода."""),
             image_cell(figures[3]),
             nbf.v4.new_markdown_cell("""## Рисунок 5. Изменение радиуса окружности

На графиках отложен радиус МНК-окружности относительно первой фазы. Различие между фронтальной и сагиттальной кривой одного человека ожидаемо: это радиусы **окружностей двумерных проекций**, а не оценки одного и того же трёхмерного радиуса сферы. Даже если две кривые центров совпадают, разные законы изменения радиуса могут дать разные электрические сигналы. Применять формулу объёма шара к этим радиусам без отдельного правила перехода от 2D к 3D неправомерно."""),
             image_cell(figures[4]),
             nbf.v4.new_markdown_cell("""## Сопоставление с РФФИ и ограничения вывода

В РФФИ опубликованы годографы, восстановленные **из пяти электрических каналов**, а здесь — центры, рассчитанные **из КТ-геометрии**. Это разные определяемые объекты. На рисунке 25 отчёта ось $x$ показана горизонтально, $y$ — вертикально; в тексте диссертации (§ 4.4) $x$ названа вертикальной осью тела, $y$ — фронтальной, $z$ — сагиттальной. Согласованное преобразование между этими историческими координатными записями и RAS+ пока не документировано. Пары «доброволец РФФИ — архивная серия 4D-КТ», независимой синхронизации 25 электрических интервалов с 20 КТ-фазами и одинакового определения исходной области также нет. Поэтому **числовую ошибку относительно опубликованных годографов вычислять нельзя**; их рисунки остаются гипотезой о масштабе и характере движения. [Рис. 27, 29, 31 РФФИ](zotero://open-pdf/library/items/3NAQJNXQ?page=50); [свод правил сопоставления осей](COORDINATE_CONVENTIONS_HODOGRAPHS.md).

В этом опыте нет независимой ручной проверки каждого пофазного контура и его центра, нет количественной модели погрешности сегментации, измеренного шума каналов и проверки электрического поля. Выпуклая оболочка добавляла к площади проекции около нескольких процентов, заполняя вогнутости. Изменение плоскости и правила точек может изменить вывод; выбор «лучшего» метода по одной матрице $D$ был бы необоснованным.

**Следующая доступная проверка:** на одной и той же КТ, материальной карте и выбранных сборках вычислить в EIDORS синтетический отклик анатомической области и каждой сферы. Далее решать обратную задачу на этих синтетических сигналах, оценивать погрешности восстановления смещения центра и изменения объёма, а также их взаимную неидентифицируемость. Координаты и сигналы исторического пятиканального опыта у нас отсутствуют; получение их от автора не предполагается. Жёстко сохранённые в историческом коде числа могут служить только внутренним примером проверки формул с установленным происхождением, но не референсом для архивных КТ. Прямое численное расхождение с РФФИ до документального сопоставления людей, координат и фаз останется неопределённым. Проверка на синтетическом EIDORS устанавливает свойства модели в заданной постановке, а не экспериментальную точность реального прибора."""),
             nbf.v4.new_markdown_cell("""## Воспроизведение и происхождение

Расчётное ядро: [`run_tikhomirov_ct_contours.py`](run_tikhomirov_ct_contours.py), построение рисунков и отчёта: [`build_tikhomirov_ct_report.py`](build_tikhomirov_ct_report.py). Исходный порт: [`sphere_fit.py`](../../Tikhomirov/cardio-model-py/src/cardio_model/sphere_fit.py); первичный Wolfram-файл: [`SphereMovingFunction.m`](../../Tikhomirov/WolframMath-master/Kernel/part1/SphereMovingFunction.m). Идентичность первоисточников зафиксирована в [реестре диссертации и отчётов](../../MATLAB_TRKG4_real_subjects/docs/HEART_SPHERE_SOURCE_CORPUS_20260914.json). Локальный протокол с хэшами кода и конфигурации — `results/tikhomirov_audit_20260928/fit_v2/protocol.json`; пофазные параметры — `phase_fits.csv`, сводка — `summary.csv`. Конфигурация указывает на внешние локальные маски и не публикуется. Пересчёт: `python run_tikhomirov_ct_contours.py --config <локальная конфигурация> --output <пустая папка>`, затем `python build_tikhomirov_ct_report.py`. Опубликованные значения в этом ноутбуке — сохранённый результат указанного прогона; повторное открытие HTML его не пересчитывает.""")]
    nb.cells = cells
    target = HERE / "21.13_Контурные_сферы_Тихомирова_и_годографы_исторической_4D_КТ.ipynb"
    nbf.write(nb, target)
    print(json.dumps({"notebook": str(target), "figures": [str(p) for p in figures],
                      "cells": len(cells)}, ensure_ascii=False))
    return target


if __name__ == "__main__":
    report()
