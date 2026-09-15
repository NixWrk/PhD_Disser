"""Scientific catalogue and classification figure for the 19 C01 forward operators."""

import pandas as pd
from IPython.display import HTML, display
import plotly.graph_objects as go
import plotly.io as pio


MODEL_CATALOG = [
    dict(id="M01", key="reference", group="КТ/FEM-референс",
         name="КТ/FEM: исходное лёгкое",
         view="Криволинейное КТ-тело и исходная замкнутая маска лёгких.",
         simplify="Лёгочная геометрия не сокращается; все внелёгочные ткани объединены в один фон ρ₁.",
         reason="Задать синтетические данные с известными ρ₁,ρ₂ и оценить, как сокращения геометрии влияют на их восстановление.",
         method="Метод конечных элементов (FEM) на общей тетраэдральной сетке; в основном тесте точечные электроды (PEM).",
         contacts="Фактические поверхностные узлы КТ/FEM.",
         limit="Это референс выбранной двухтканной дискретной модели, а не независимая физическая истина."),
    dict(id="M02", key="plane", group="FEM-сокращения в КТ-теле",
         name="Плоская граница внутри КТ-тела",
         view="Наружная поверхность тела остаётся КТ-криволинейной; область глубже плоскости d=h(0) назначается лёгким.",
         simplify="Криволинейная проксимальная и дистальная границы лёгкого заменяются одной плоскостью.",
         reason="В сравнении с M01 проверить замену лёгочной границы плоскостью при сохранении конечного КТ-тела и контактов.",
         method="Объёмный FEM на той же сетке; меняется только тканевая маска.",
         contacts="Те же узлы КТ/FEM.",
         limit="Искусственная лёгочная область продолжается до внешней границы тела; модель диагностическая."),
    dict(id="M03", key="ellipsoid", group="FEM-сокращения в КТ-теле",
         name="Эллипсоид внутри КТ-тела",
         view="Проксимальная часть лёгочной маски заменена эллипсоидом; дистальная часть исходной КТ-маски сохранена.",
         simplify="Центр и ориентация определяются по выбранной части КТ-маски; объём задаёт масштаб полуосей, а отношения собственных чисел матрицы вторых моментов — их пропорции.",
         reason="В сравнении с M01 проверить, достаточно ли центра, ориентации и трёх полуосей для описания проксимальной поверхности.",
         method="Объёмный FEM на общей сетке.",
         contacts="Те же узлы КТ/FEM.",
         limit="Точное одновременное сохранение объёма и всех вторых моментов не обеспечивается. Сопряжение с дистальной КТ-маской создаёт искусственную составную геометрию."),
    dict(id="M04", key="ellipsoid_depth", group="FEM-сокращения в КТ-теле",
         name="Эллипсоид с привязкой по глубине",
         view="Тот же эллипсоид сдвинут вдоль d на 9,315 мм до совпадения первого центрального входа с h(0).",
         simplify="Сохраняются параметры эллипсоида; добавляется одна привязка к центральной глубине.",
         reason="В сравнении с M03 проверить эффект привязки эллипсоида к центральной глубине КТ при тех же полуосях и ориентации.",
         method="Объёмный FEM на общей сетке.",
         contacts="Те же узлы КТ/FEM.",
         limit="Совпадение одной глубины не гарантирует правильную площадь, кривизну или дистальную границу."),
    dict(id="M05", key="extended_lung", group="FEM-сокращения в КТ-теле",
         name="Продолженное лёгкое внутри КТ-тела",
         view="После первого пересечения исходного лёгкого каждый глобальный луч вдоль d остаётся лёгочным до границы тела.",
         simplify="Удаляются все последующие выходы из лёгкого, включая возврат к фону под диафрагмой.",
         reason="В сравнении с M01 оценить влияние удаления дистальных границ и повторных пересечений; это диагностическая проверка допущения.",
         method="Объёмный FEM на общей сетке.",
         contacts="Те же узлы КТ/FEM.",
         limit="Продолжение не является анатомическим лёгким и не выделяет отдельно диафрагму."),
    dict(id="M06", key="uniform_transverse", group="FEM-сокращения в КТ-теле",
         name="Общая поперечная кривизна в КТ-теле",
         view="Полный центральный контур t=0 переносится по одной круговой функции глубинного сдвига gκ(t).",
         simplify="В локальном окне зависимость формы от поперечной координаты t задаётся одним параметром кривизны κ; вне окна сохраняется КТ.",
         reason="Проверить относительно M01, достаточно ли центрального контура и одной поперечной кривизны для восстановления ρ₁,ρ₂.",
         method="Объёмный FEM; локальное поле плавно сопрягается с исходной КТ вне окна.",
         contacts="Те же узлы КТ/FEM.",
         limit="Одна функция сдвига не передаёт изменение формы и асимметрию поперечных сечений."),
    dict(id="M07", key="variable_transverse", group="FEM-сокращения в КТ-теле",
         name="Переменная поперечная форма в КТ-теле",
         view="Знаковое поле границы изменяется кубически по t; коэффициенты являются картами по s и d.",
         simplify="Полная КТ-поверхность заменяется центральным полем и тремя пространственными картами коэффициентов.",
         reason="В сравнении с M06 проверить, оправдано ли усложнение поперечной формы уменьшением ошибки восстановления относительно M01.",
         method="Объёмный FEM; локальная аппроксимация сопрягается с исходной КТ.",
         contacts="Те же узлы КТ/FEM.",
         limit="Это гибкая, но ещё не малопараметрическая модель; три коэффициента являются полями, а не тремя числами."),
    dict(id="M08", key="m3h", group="Численные модели полупространства",
         name="M3-H: постоянное сечение в полупространстве",
         view="Плоская кожа d=0; центральный контур лёгкого бесконечно продолжается без изменения вдоль t.",
         simplify="Удаляется конечная наружная форма тела, а трёхмерное включение становится трансляционно-инвариантным.",
         reason="Проверить пригодность постоянного сечения в полупространстве для быстрого расчёта и последующего восстановления ρ₁,ρ₂.",
         method="Численная 2.5D-схема: преобразование Фурье по t и метод граничных элементов (BEM) по контуру; замкнутой формулы нет.",
         contacts="Узлы КТ ортогонально проецируются на d=0 с сохранением s,t.",
         limit="Бесконечная инвариантность вдоль t и плоская кожа являются сильными изменениями задачи."),
    dict(id="M09", key="m4h", group="Численные модели полупространства",
         name="M4-H: переменная форма в полупространстве",
         view="Плоская кожа d=0 и замкнутое трёхмерное криволинейное лёгочное включение.",
         simplify="Наружное КТ-тело заменяется полупространством; сохраняется трёхмерная форма включения.",
         reason="Проверить расчёт без объёмной сетки с сохранением переменной трёхмерной формы; сопоставить точность и скорость с M01 и M08.",
         method="Трёхмерный BEM на треугольной поверхности включения; численное решение интегральной системы.",
         contacts="Проекции узлов КТ на d=0.",
         limit="Расхождение с КТ/FEM включает одновременно плоскую кожу, перенос контактов и сокращение поверхности."),
    dict(id="M10", key="planar_nominal", group="Плоские двухслойные формулы",
         name="Плоская аналитическая: номинальные точки",
         view="Два бесконечных по горизонтали плоских слоя; граница на h=67,00 мм; четыре точки лежат на одной прямой.",
         simplify="Вся анатомия заменяется одним числом h и номинальными расстояниями сборки.",
         reason="Задать исходное двуслойное приближение, которое требуется улучшить; точность восстановления оценивается относительно M01.",
         method="Ряд зеркальных источников, 256 членов; FEM и BEM не вызываются.",
         contacts="Номинальные координаты −L/2, −L/4, L/4, L/2.",
         limit="Нет кривизны кожи, конечного тела, дистальной границы лёгкого и фактического смещения контактов."),
    dict(id="M11", key="planar_projected", group="Плоские двухслойные формулы",
         name="Плоская аналитическая: проекции контактов",
         view="Та же двухслойная плоскость h=67,00 мм, но используются фактические взаимные расстояния проекций узлов.",
         simplify="Анатомия остаётся плоской; номинальные координаты заменяются проекциями поверхностных узлов КТ.",
         reason="В сравнении с M10 оценить влияние перехода от номинальных расстояний к расстояниям между проекциями контактов.",
         method="Тот же ряд зеркальных источников, 512 членов.",
         contacts="Проекции КТ/FEM-узлов на d=0.",
         limit="Улучшение координат контактов не возвращает криволинейную наружную и внутреннюю границы."),
    dict(id="M12", key="planar_nearest", group="Плоские двухслойные формулы",
         name="Плоская аналитическая: кратчайшее расстояние",
         view="Плоская двухслойная среда с h=51,02 мм вместо направленной центральной глубины 67,00 мм.",
         simplify="КТ сводится к кратчайшему евклидову расстоянию от центра до лёгочной поверхности.",
         reason="В сравнении с M11 проверить влияние другого определения h при тех же проекциях контактов.",
         method="Ряд зеркальных источников, 512 членов.",
         contacts="Проекции КТ/FEM-узлов на d=0.",
         limit="Кратчайший отрезок не совпадает с направлением d и не описывает профиль под всей сборкой."),
    dict(id="M13", key="averaged", group="Усреднение плоских ядер",
         name="Среднее ядро: номинальные точки",
         view="Для каждой пары электродов берётся семейство локальных плоских двухслойных ядер с h=h(s), затем они усредняются по s.",
         simplify="Профиль первого входа сохраняется, но общее поле в среде с переменной границей не решается.",
         reason="В сравнении с M10 проверить, уменьшает ли учёт профиля h(s) ошибку восстановления при номинальных контактах.",
         method="Ряд из 256 членов плюс численное интегрирование по профилю методом трапеций.",
         contacts="Номинальные точки на оси s.",
         limit="Усреднение является эвристикой: среднее локальных решений не решает единую задачу с криволинейной границей."),
    dict(id="M14", key="averaged_projected", group="Усреднение плоских ядер",
         name="Среднее ядро: проекции контактов",
         view="То же усреднение G(r,h(s),q), но r и пределы участка берутся из проецированных контактов.",
         simplify="Сохраняются профиль первого входа и фактические плоские расстояния; трёхмерное перераспределение поля исключено.",
         reason="Сопоставление с M11 проверяет учёт профиля h(s), а с M13 — переход к проекциям контактов.",
         method="Ряд из 512 членов и трапецеидальное усреднение по s.",
         contacts="Проекции КТ/FEM-узлов.",
         limit="Разные локальные ядра не образуют единую физическую границу."),
    dict(id="M15", key="mean_h", group="Усреднение плоских ядер",
         name="Средняя глубина по паре",
         view="Каждая из четырёх пар получает одно среднее h на своём интервале; при любом пропуске лёгкого на интервале используется h=∞.",
         simplify="Распределение h(s) заменяется четырьмя эффективными числами.",
         reason="В сравнении с M14 проверить замену усреднения ядер расчётом при средней глубине; эти операции не эквивалентны.",
         method="Плоский ряд из 512 членов после вычисления среднего h; при h=∞ остаётся однородное ядро.",
         contacts="Проекции КТ/FEM-узлов.",
         limit="Правило h=∞ при отсутствии пересечения хотя бы на части интервала — эвристическое исключение лёгочного вклада данной пары. Оно не задаёт доказанной верхней или нижней границы ошибки."),
    dict(id="M16", key="central_finite", group="Слоистые лучевые формулы",
         name="Три слоя: первый центральный интервал",
         view="Фон от кожи до первого входа, конечный интервал лёгкого, затем фоновое полупространство.",
         simplify="Вся сборка получает две центральные плоские границы d₁,d₂.",
         reason="В сравнении с M11 проверить влияние возврата к фону после первого центрального лёгочного интервала; тканей по-прежнему две.",
         method="Спектральная рекурсия слоёв и квадратура Гаусса–Лежандра с функцией Бесселя J₀.",
         contacts="Проекции КТ/FEM-узлов.",
         limit="Для всех пар приняты одни плоские границы, поэтому боковая кривизна потеряна. В C01 центральный интервал один, и M16 совпадает с M18."),
    dict(id="M17", key="mean_finite", group="Слоистые лучевые формулы",
         name="Три слоя: средний первый интервал",
         view="Для каждой пары усредняются первый вход и первая толщина там, где лёгкое присутствует; результат смешивается с однородной долей.",
         simplify="Набор лучей заменяется долей покрытия и двумя условными средними.",
         reason="В сравнении с M16 проверить замену одного центрального интервала средними по участку каждой пары с учётом отсутствия лёгкого на части участка.",
         method="Слоистая квадратура для среднего интервала плюс линейная смесь с однородным ядром.",
         contacts="Проекции КТ/FEM-узлов.",
         limit="Линейная смесь по доле покрытия является эвристикой: она не учитывает общее поле и перераспределение тока между участками."),
    dict(id="M18", key="central_stack", group="Слоистые лучевые формулы",
         name="Все интервалы центрального луча",
         view="Все входы и выходы центрального луча задают чередующиеся плоские слои ρ₁/ρ₂. В C01 интервал один: M18 совпадает с M16.",
         simplify="Последовательность пересечений из КТ-профиля сохраняется и по принятому допущению распространяется на всю плоскость.",
         reason="В сравнении с M16 проверить влияние дополнительных центральных интервалов, если они есть. Для C01 эта проверка сводится к совпадению реализаций.",
         method="Многослойная спектральная рекурсия и квадратура Гаусса–Лежандра.",
         contacts="Проекции КТ/FEM-узлов.",
         limit="Принято допущение о плоских слоях по одному лучу; его анатомическая адекватность не доказана. Совпадение с M16 при одном интервале — проверка согласованности, а не независимая валидация."),
    dict(id="M19", key="mean_stack", group="Слоистые лучевые формулы",
         name="Среднее ядро всех интервалов",
         view="Для каждого s строится собственная последовательность плоских слоёв по всем пересечениям; полученные ядра усредняются по паре.",
         simplify="Сохраняются разные последовательности слоёв вдоль s, но исключается их совместное пространственное поле и изменение профиля по t.",
         reason="В сравнении с M18 проверить усреднение по продольному профилю, а с M14 — учёт всех пересечений вместо одного первого входа.",
         method="Многослойная квадратура для каждого луча и последующее трапецеидальное усреднение.",
         contacts="Проекции КТ/FEM-узлов.",
         limit="Усреднение слоистых ядер является эвристикой и не решает единую криволинейную трёхмерную задачу."),
]


GROUP_COLORS = {
    "КТ/FEM-референс": "#37474f",
    "FEM-сокращения в КТ-теле": "#1565c0",
    "Численные модели полупространства": "#00897b",
    "Плоские двухслойные формулы": "#ef6c00",
    "Усреднение плоских ядер": "#8e24aa",
    "Слоистые лучевые формулы": "#6d4c41",
}


def model_catalog_table():
    frame = pd.DataFrame(MODEL_CATALOG)
    frame["id"] = frame["id"] + "<br><code>" + frame["key"] + "</code>"
    view = frame[["id", "group", "name", "view", "simplify", "reason", "method", "contacts", "limit"]].rename(
        columns={"id": "№ / код реализации", "group": "Идея", "name": "Реализация",
                 "view": "Как выглядит и что сохраняет", "simplify": "Что упрощается",
                 "reason": "Зачем введена", "method": "Как рассчитывается",
                 "contacts": "Контакты", "limit": "Главное ограничение"})
    style = """
    <style>
    table.c01-catalog {border-collapse:collapse; width:100%; font-size:12px; line-height:1.25;}
    table.c01-catalog th {background:#e7edf5; position:sticky; top:0; z-index:1;}
    table.c01-catalog th, table.c01-catalog td {border:1px solid #c8d2df; padding:6px; vertical-align:top; text-align:left;}
    table.c01-catalog tr:nth-child(even) {background:#f8fafc;}
    table.c01-catalog td:nth-child(1) {white-space:nowrap; font-weight:700;}
    </style>
    """
    display(HTML(style + view.to_html(index=False, escape=False, classes="c01-catalog")))


def model_catalog_figure():
    groups = list(GROUP_COLORS)
    ids, labels, parents = ["root"], ["19 реализаций"], [""]
    values, colors = [len(MODEL_CATALOG)], ["#cfd8dc"]
    custom = [["", "", "", "", ""]]
    for group in groups:
        group_id = "group:" + group
        rows = [row for row in MODEL_CATALOG if row["group"] == group]
        ids.append(group_id); labels.append(group); parents.append("root")
        values.append(len(rows)); colors.append(GROUP_COLORS[group]); custom.append(["", "", "", "", ""])
        for row in rows:
            ids.append(row["key"]); labels.append(row["id"] + " · " + row["name"]); parents.append(group_id)
            values.append(1); colors.append(GROUP_COLORS[group])
            custom.append([row["view"], row["simplify"], row["reason"], row["method"], row["limit"]])
    figure = go.Figure(go.Treemap(
        ids=ids, labels=labels, parents=parents, values=values, branchvalues="total",
        marker=dict(colors=colors, line=dict(color="white", width=1.2)), customdata=custom,
        textinfo="label+value",
        hovertemplate=("<b>%{label}</b><br>%{customdata[0]}<br><br>"
                       "<b>Упрощение:</b> %{customdata[1]}<br><b>Зачем:</b> %{customdata[2]}<br>"
                       "<b>Расчёт:</b> %{customdata[3]}<br><b>Ограничение:</b> %{customdata[4]}<br>"
                       "Площадь блока показывает число реализаций; точность и скорость здесь не отображаются.<extra></extra>"),
        pathbar=dict(visible=True)))
    figure.update_layout(
        title=("Классификация 19 реализаций прямого оператора C01<br>"
               "<sup>7 FEM · 2 BEM · 10 формульных, квадратурных и эвристических реализаций</sup>"), height=720,
        margin=dict(l=10, r=10, t=70, b=10), uniformtext=dict(minsize=10, mode="hide"),
        meta=dict(operator_count=19, fem_count=7, halfspace_numerical_count=2,
                  formula_or_quadrature_count=10, keys=[row["key"] for row in MODEL_CATALOG]))
    display(HTML(pio.to_html(
        figure, include_plotlyjs=True, full_html=False,
        config={"responsive": True, "displaylogo": False})))



def _scheme_data(key):
    import numpy as np
    x = np.linspace(-70.0, 70.0, 141)
    curved_skin = 4.0 * (x / 70.0) ** 2
    flat_skin = np.zeros_like(x)
    skin = curved_skin if key in {"reference", "plane", "ellipsoid", "ellipsoid_depth",
                                  "extended_lung", "uniform_transverse", "variable_transverse"} else flat_skin
    electrodes_x = np.array([-70.0, -35.0, 35.0, 70.0])
    if key not in {"planar_nominal", "averaged"}:
        electrodes_x = np.array([-68.0, -33.0, 36.0, 69.0])
    regions = []
    guides = []

    irregular_top = 61 + 8 * np.sin((x + 20) / 24) + 4 * np.sin(x / 9)
    irregular_bottom = 116 + 9 * np.sin((x - 5) / 31)
    if key == "reference":
        regions = [(x, irregular_top, irregular_bottom)]
    elif key == "plane":
        regions = [(x, np.full_like(x, 67.0), np.full_like(x, 150.0))]
    elif key in {"ellipsoid", "ellipsoid_depth"}:
        theta = np.linspace(0, 2 * np.pi, 241)
        centre_y = 101.0 if key == "ellipsoid" else 110.0
        radius_y = 43.0
        ex = 55.0 * np.cos(theta)
        ey = centre_y + radius_y * np.sin(theta)
        regions = [(ex, ey, ey)]
        guides = [(np.array([-70, 70]), np.array([67, 67]), "h(0); эллипсоид условный")]
    elif key == "extended_lung":
        regions = [(x, irregular_top, np.full_like(x, 150.0))]
    elif key == "uniform_transverse":
        top = 64 + 15 * (x / 70) ** 2
        regions = [(x, top, top + 53)]
    elif key == "variable_transverse":
        top = 63 + 10 * (x / 70) ** 2 + 7 * (x / 70) ** 3
        bottom = 116 + 6 * np.sin(x / 28) - 5 * (x / 70)
        regions = [(x, top, bottom)]
    elif key == "m3h":
        regions = [(x, irregular_top, irregular_bottom)]
        guides = [(np.array([-58, 58]), np.array([139, 139]), "контур без изменения вдоль t")]
    elif key == "m4h":
        top = 60 + 7 * np.sin(x / 24) + 5 * (x / 70) ** 2
        bottom = 115 + 8 * np.sin(x / 30)
        regions = [(x, top, bottom)]
        guides = [(np.array([-58, 58]), np.array([139, 139]), "замкнутое 3D-включение")]
    elif key in {"planar_nominal", "planar_projected"}:
        regions = [(x, np.full_like(x, 67.0), np.full_like(x, 150.0))]
    elif key == "planar_nearest":
        regions = [(x, np.full_like(x, 51.02), np.full_like(x, 150.0))]
    elif key in {"averaged", "averaged_projected"}:
        for lo, hi, depth in [(-70, -38, 72), (-34, -3, 58), (2, 33, 76), (38, 70, 64)]:
            xx = np.array([lo, hi])
            regions.append((xx, np.full(2, depth), np.full(2, 148.0)))
        guides = [(x, 67 + 12 * np.sin(x / 28), "локальные ядра G(r,h(s),q),<br>не соседние блоки ткани")]
    elif key == "mean_h":
        regions = [(x, np.full_like(x, 70.0), np.full_like(x, 150.0))]
        guides = [(np.array([20, 70]), np.array([48, 48]), "нет пересечения на части участка:<br>для пары h=∞")]
    elif key == "central_finite":
        regions = [(x, np.full_like(x, 58.0), np.full_like(x, 108.0))]
        guides = [(x, np.full_like(x, 58.0), "d₁"), (x, np.full_like(x, 108.0), "d₂")]
    elif key == "mean_finite":
        xx = x[x <= 28]
        regions = [(xx, np.full_like(xx, 62.0), np.full_like(xx, 105.0))]
        guides = [(np.array([28, 28]), np.array([35, 137]), "вес смеси ядер,<br>не боковая граница лёгкого")]
    elif key == "central_stack":
        regions = [(x, np.full_like(x, 43.0), np.full_like(x, 72.0)),
                   (x, np.full_like(x, 96.0), np.full_like(x, 122.0))]
        guides = [(x, np.full_like(x, y), "граница слоя (общая схема)") for y in [43.0, 72.0, 96.0, 122.0]]
    elif key == "mean_stack":
        for lo, hi, levels in [(-70, -25, [(48, 76), (101, 124)]),
                               (-20, 20, [(58, 109)]),
                               (25, 70, [(69, 93), (112, 132)])]:
            xx = np.array([lo, hi])
            for top, bottom in levels:
                regions.append((xx, np.full(2, top), np.full(2, bottom)))
        guides = [(np.array([-23, -23]), np.array([35, 137]), "разные последовательности слоёв"),
                  (np.array([23, 23]), np.array([35, 137]), "усредняются ядра,<br>не области ткани")]
    else:
        raise KeyError(key)

    region_x, region_y = [], []
    for rx, top, bottom in regions:
        if len(rx) > 2 and (top is bottom or (np.asarray(top) == np.asarray(bottom)).all()):
            region_x.extend(rx.tolist() + [None])
            region_y.extend(np.asarray(top).tolist() + [None])
        else:
            region_x.extend(np.asarray(rx).tolist() + np.asarray(rx)[::-1].tolist() + [None])
            region_y.extend(np.asarray(top).tolist() + np.asarray(bottom)[::-1].tolist() + [None])
    guide_x, guide_y, guide_text = [], [], []
    for gx, gy, text in guides:
        guide_x.extend(np.asarray(gx).tolist() + [None])
        guide_y.extend(np.asarray(gy).tolist() + [None])
        guide_text.extend([""] * max(0, len(gx) - 1) + [text, None])
    electrode_y = np.interp(electrodes_x, x, skin) - 4.0
    return [
        go.Scatter(x=x, y=skin, mode="lines", name="наружная поверхность / плоскость кожи",
                   line=dict(color="#5d4037", width=4), hoverinfo="skip"),
        go.Scatter(x=region_x, y=region_y, mode="lines", fill="toself", name="ρ₂: включение или слои отдельных ядер",
                   line=dict(color="#00a8cc", width=2), fillcolor="rgba(0,168,204,0.24)", hoverinfo="skip"),
        go.Scatter(x=guide_x, y=guide_y, mode="lines+text", name="правило сведения",
                   line=dict(color="#7b1fa2", width=2, dash="dash"), text=guide_text,
                   textposition="top center", hoverinfo="skip"),
        go.Scatter(x=electrodes_x, y=electrode_y, mode="markers+text", name="точечные электроды",
                   marker=dict(color=["#d32f2f", "#f9a825", "#f9a825", "#d32f2f"], size=10),
                   text=["I+", "V+", "V−", "I−"], textposition="top center", hoverinfo="skip"),
    ]


def model_scheme_figure():
    figure = go.Figure()
    trace_groups = []
    for row_index, row in enumerate(MODEL_CATALOG):
        start = len(figure.data)
        for trace in _scheme_data(row["key"]):
            trace.visible = row_index == 0
            trace.showlegend = row_index == 0
            figure.add_trace(trace)
        trace_groups.append(range(start, len(figure.data)))

    buttons = []
    for row_index, row in enumerate(MODEL_CATALOG):
        visible = [False] * len(figure.data)
        for trace_index in trace_groups[row_index]:
            visible[trace_index] = True
        buttons.append(dict(
            label=row["id"] + " · " + row["name"],
            method="update",
            args=[
                dict(visible=visible, showlegend=[v and (i - min(trace_groups[row_index]) >= 0)
                                                   for i, v in enumerate(visible)]),
                {"title.text": ("<b>" + row["id"] + ". " + row["name"] + "</b><br>"
                                "<sup>" + row["view"] + "<br>Условная схема принципа, не срез КТ C01; размеры не используются в расчёте. Фон — ρ₁.</sup>")}
            ]))

    first = MODEL_CATALOG[0]
    figure.update_layout(
        title=("<b>" + first["id"] + ". " + first["name"] + "</b><br>"
               "<sup>" + first["view"] + "<br>Условная схема принципа, не срез КТ C01; размеры не используются в расчёте. Фон — ρ₁.</sup>"),
        width=1100, height=650, margin=dict(l=80, r=40, t=155, b=75),
        updatemenus=[dict(type="dropdown", direction="down", x=0, y=1.18,
                          xanchor="left", yanchor="top", active=0, buttons=buttons)],
        xaxis=dict(title="условная координата сечения (схема принципа)", range=[-78, 78]),
        yaxis=dict(title="глубина внутрь среды, условные единицы", range=[155, -14], scaleanchor="x", scaleratio=0.9),
        plot_bgcolor="white", legend=dict(orientation="h", y=-0.16),
        meta=dict(operator_count=19, schematic=True, keys=[row["key"] for row in MODEL_CATALOG]))
    figure.update_xaxes(showgrid=True, gridcolor="#e6ebf2", zeroline=False)
    figure.update_yaxes(showgrid=True, gridcolor="#e6ebf2", zeroline=False)
    display(HTML(pio.to_html(
        figure, include_plotlyjs=False, full_html=False,
        config={"responsive": True, "displaylogo": False})))

