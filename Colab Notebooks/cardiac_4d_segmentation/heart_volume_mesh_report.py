"""Reader notebook for the matched-quadrature, two-mesh heart-volume experiment."""
from pathlib import Path
import json
import hashlib
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "MATLAB_TRKG4_real_subjects/output/exploratory/heart_mesh_volume_block_20260916/comparison/summary.json"
FIGURES = ROOT / "Colab Notebooks/cardiac_4d_segmentation/results/heart_mesh_volume_block_20260916"
NAMES = {"individual":"Индивидуальная форма", "sphere":"Сфера", "ellipsoid":"Эллипсоид"}
LEVELS = {"coarse":"Исходная сетка", "fine":"Уточнённая сетка"}
METRICS = {"Sminus":"Уменьшение объёма", "derivative":"Центральная оценка", "Splus":"Увеличение объёма"}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fmt(x, digits=3): return "не определено" if x is None else f"{x:.{digits}f}".replace(".",",")
def load(path=DEFAULT):
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    if data["status"]!="complete_matched_comparison" or data["points_per_tet"]!=16384:
        raise ValueError("A complete matched-quadrature comparison is required")
    return data


def table(headers, rows):
    display(Markdown("| "+" | ".join(headers)+" |\n|"+"|".join(["---"]*len(headers))+"|\n"+
        "\n".join("| "+" | ".join(map(str,row))+" |" for row in rows)))


def physical_table(data):
    settings=data["physical_settings"]
    table(["Исходная геометрическая величина","Значение"],[["Объём целого эффективного сердца, мл",fmt(settings["nominal_volume_ml"],6)]])
    labels={"soft":"Мягкие ткани","soft_tissues":"Мягкие ткани","heart":"Эффективное целое сердце","lung":"Лёгкие","lungs":"Лёгкие","bone":"Кости","bones":"Кости"}
    rho=settings["rho_ohm_m"]
    table(["Компартмент","ρ, Ом·м"], [[labels.get(k,k),fmt(v,6)] for k,v in rho.items()])
    contact=settings["electrode_model"]
    table(["Контактный параметр","Значение"],[["Диаметр, мм",fmt(contact["diameter_mm"],1)],
        ["Номинальная площадь, мм²",fmt(contact["area_mm2"],6)],
        ["Удельный контактный импеданс, Ом·м²",fmt(contact["z_contact_ohm_m2"],10)],
        ["Доля номинальной площади",fmt(contact["area_fraction"],2)]])
    table(["Сетка","Фактические площади четырёх контактов, мм²"], [[LEVELS[level], "; ".join(fmt(a,4) for a in areas)] for level,areas in data["actual_contact_areas_mm2"].items()])
    table(["Сетка","Узлы","Тетраэдры"],[[LEVELS[level],str(int(info["node_count"])),str(int(info["element_count"]))] for level,info in data["meshes"].items()])


def baseline_table(data):
    table(["Форма","Z исходной сетки, Ом","Z уточнённой сетки, Ом","Различие, Ом","Различие, %"],
        [[NAMES[g],fmt(r["Z_coarse_ohm"],6),fmt(r["Z_fine_ohm"],6),fmt(r["Z_fine_minus_coarse_ohm"],6),fmt(100*r["relative_to_coarse"])] for g,r in data["baseline"].items()])


def save(fig, name):
    FIGURES.mkdir(parents=True,exist_ok=True)
    fig.savefig(FIGURES/name,dpi=160,bbox_inches="tight")
    display(Image(filename=str(FIGURES/name)));plt.close(fig)


def response_figure(data):
    fig,axes=plt.subplots(1,3,figsize=(15,4.3),sharey=True,constrained_layout=True)
    for ax,(geometry,label) in zip(axes,NAMES.items()):
        for level,style,color in [("coarse","--","#6e6e6e"),("fine","-","#1565c0")]:
            points={0.:0.}
            for r in data["derivatives"][geometry][level]:
                points[-r["dV_one_sided"]]=1000*(r["Z_minus_ohm"]-r["Z_baseline_ohm"])
                points[r["dV_one_sided"]]=1000*(r["Z_plus_ohm"]-r["Z_baseline_ohm"])
            xx=sorted(points);ax.plot(xx,[points[x] for x in xx],style,marker="o",color=color,label=LEVELS[level])
        ax.axhline(0,color="black",lw=.5);ax.axvline(0,color="black",lw=.5)
        ax.set(title=label,xlabel="Заданное изменение объёма, мл",ylabel="ΔZ относительно своей базы, мОм")
        ax.grid(alpha=.2);ax.legend(fontsize=9)
    save(fig,"01_volume_response_meshes.png")


def sensitivity_figure(data):
    values=[]
    for level in LEVELS:
        values.append(np.array([[next(r for r in data["derivatives"][g][level] if r["step"]==.005)[key]*1000 for key in METRICS] for g in NAMES]))
    limit=max(float(np.max(np.abs(v))) for v in values)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
    for ax,(level,label),v in zip(axes,LEVELS.items(),values):
        im=ax.imshow(v,cmap="RdBu_r",vmin=-limit,vmax=limit,aspect="auto")
        ax.set_xticks(range(3),list(METRICS.values()),rotation=12);ax.set_yticks(range(3),list(NAMES.values()));ax.set_title(label)
        for i in range(3):
            for j in range(3):ax.text(j,i,fmt(v[i,j],2),ha="center",va="center",color="white" if abs(v[i,j])>.6*limit else "black")
    fig.colorbar(im,ax=axes,label="Разностная оценка, мОм/мл",shrink=.8)
    save(fig,"02_directional_sensitivity_meshes.png")


def shape_error_figure(data):
    values=[]
    for level in LEVELS:
        selected={r["metric"]:r for r in data["shape_errors"] if r["mesh"]==level and r["step"]==.005}
        values.append(np.array([[100*selected[key][g+"_absolute_relative_error"] for key in METRICS] for g in ("sphere","ellipsoid")]))
    limit=max(float(np.max(v)) for v in values)
    fig,axes=plt.subplots(1,2,figsize=(13,3.9),constrained_layout=True)
    for ax,(level,label),v in zip(axes,LEVELS.items(),values):
        im=ax.imshow(v,cmap="YlOrRd",vmin=0,vmax=limit,aspect="auto")
        ax.set_xticks(range(3),list(METRICS.values()),rotation=12);ax.set_yticks(range(2),["Сфера","Эллипсоид"]);ax.set_title(label)
        for i in range(2):
            for j in range(3):ax.text(j,i,fmt(v[i,j],1)+"%",ha="center",va="center",color="white" if v[i,j]>.6*limit else "black")
    fig.colorbar(im,ax=axes,label="Отклонение от индивидуальной формы, %",shrink=.8)
    save(fig,"03_shape_error_meshes.png")


def sensitivity_table(data):
    rows=[]
    for g in NAMES:
        for level in LEVELS:
            for r in data["derivatives"][g][level]:
                rows.append([NAMES[g],LEVELS[level],fmt(r["step"]*100,1),*[fmt(r[k]*1000,4) for k in METRICS],fmt(abs(r["one_sided_normalized_difference"])*100,2)])
    table(["Форма","Сетка","Полуширина, %","S−, мОм/мл","Sс, мОм/мл","S+, мОм/мл","Асимметрия, %"],rows)


def quadrature_table(data):
    history=data.get("electrical_quadrature_history")
    if history is None:
        raise ValueError("This report requires the electrical quadrature comparison")
    table(["Форма","Полуширина, %","Оценка","4096, мОм/мл","16 384, мОм/мл","Различие, %"],
        [[NAMES[r["geometry"]],fmt(r["step"]*100,1),METRICS[r["metric"]],fmt(r["value_q4096"]*1000,4),fmt(r["value_q16384"]*1000,4),fmt(None if r["absolute_relative_change_to_q16384"] is None else r["absolute_relative_change_to_q16384"]*100,3)] for r in history["rows"]])


def volume_table(data):
    rows=[]
    for g in NAMES:
        for level in LEVELS:
            records=data["geometry_qc"][g][level]
            for step in (.005,.01):
                errors=[]
                for q in (4096,8192,16384):
                    plus=next(r for r in records if r["points_per_tet"]==q and r["signed_step"]==step)
                    minus=next(r for r in records if r["points_per_tet"]==q and r["signed_step"]==-step)
                    exact=plus["analytic_volume_ml"]-minus["analytic_volume_ml"]
                    errors.append(fmt(100*((plus["material_volume_ml"]-minus["material_volume_ml"])/exact-1),4))
                baseline=next(r for r in records if r["points_per_tet"]==16384 and r["signed_step"]==0)
                one=step*baseline["analytic_volume_ml"]
                eminus=100*((baseline["material_volume_ml"]-minus["material_volume_ml"])/one-1)
                eplus=100*((plus["material_volume_ml"]-baseline["material_volume_ml"])/one-1)
                rows.append([NAMES[g],LEVELS[level],fmt(step*100,1),*errors,fmt(eminus,4),fmt(eplus,4)])
    table(["Форма","Сетка","Полуширина, %","Ошибка при 4096, %","При 8192, %","При 16 384, %","Уменьшение при 16 384, %","Увеличение при 16 384, %"],rows)


def mesh_change_table(data):
    rows=[]
    for g in NAMES:
        coarse={r["step"]:r for r in data["derivatives"][g]["coarse"]}
        for fine in data["derivatives"][g]["fine"]:
            h=fine["step"];rows.append([NAMES[g],fmt(h*100,1),*[fmt(100*(fine[k]-coarse[h][k])/abs(coarse[h][k]),2) for k in METRICS]])
    table(["Форма","Полуширина, %","Изменение S−, %","Изменение Sс, %","Изменение S+, %"],rows)


def build(summary=DEFAULT):
    import nbformat
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    summary=Path(summary).resolve();data=load(summary)
    dest=ROOT/"Colab Notebooks/40.18_Сеточная_устойчивость_объёмного_отклика_сердца.ipynb"
    d=next(x for x in data["derivatives"]["individual"]["fine"] if x["step"]==.005)
    unchanged=sum(r["preference_unchanged"] for r in data["preference_stability"])
    errors={r["metric"]:r for r in data["shape_errors"] if r["mesh"]=="fine" and r["step"]==.005}
    intro=f"""# Устойчивость объёмного отклика сердца при уточнении FEM-сетки

**Статус: вычислительный эксперимент; первый этап численной проверки продолжается.**

## Что проверено и какой результат получен

Выполнены 25 новых электрических решений: 15 на уточнённой сетке и 10 контрольных
на исходной. С учётом пяти ранее полученных решений индивидуальной формы
в основном сравнении двух сеток сопоставлены 30 состояний. Во всех случаях использовано по 16 384 точки интегрирования
в элементах, для которых требуется численное вычисление доли материала.

Для индивидуальной формы на уточнённой сетке при изменении объёма на ±0,5%
получены односторонние оценки **{fmt(d['Sminus']*1000,2)} и {fmt(d['Splus']*1000,2)} мОм/мл**.
Центральная оценка составляет **{fmt(d['derivative']*1000,2)} мОм/мл**.
Эллипсоид оказался ближе к индивидуальной форме, чем сфера, на обеих сетках.
Предпочтение сохранилось в **{unchanged} из 6** проверенных сочетаний направления оценки и шага.
Это описательная проверка на двух сетках, без вывода о статистической значимости.

Предыдущая постановка, способы построения фигур и результаты первого блока
приведены в [отчёте 40.17](40.17_Сравнение_форм_сердца_и_электрического_отклика.html).
Настоящий отчёт отвечает на более узкий вопрос: насколько изменяются электрический
отклик и сравнительные выводы при уточнении внутренней сетки.

## 1. Постановка вычислительного эксперимента

Использована одна статическая индивидуальная модель Nix, диагональная сборка
TEPC-2 с четырьмя конечными контактами диаметром 5 мм, полная электродная модель
(CEM) и частота 50 кГц. Свойства тканей заданы расчётными значениями; они
не являются индивидуальными измерениями. Удельный контактный импеданс перенесён
из прежнего сравнения проекта и также не измерен для данного опыта. Сердечная область представляет целое
эффективное сердце — смесь крови и миокарда. Модель проводимости действительная,
скалярная; ёмкостная составляющая не учитывается.

Положение и форма контактов, наружная поверхность тела, свойства тканей и геометрия
каждой сердечной модели между сетками сохранены. Уточнены только внутренние элементы.
Поэтому сравнение характеризует внутреннюю дискретизацию поля при неизменной
дискретизации поверхности и контактов.
"""
    methods=r"""Фон принят неподвижным: исходная область сердца заполнена мягкими тканями,
лёгкие и кости сохранены как отдельные материалы. Текущая сердечная область
накладывается на этот фон; при её уходе восстанавливается прежний фоновый материал.
Пересечение с лёгкими или костями не отклоняет состояние. Это согласованное допущение
о материальной карте, без механической деформации окружающих органов.

Индивидуальная форма задана объединением исходных сердечных тетраэдров.
Сфера сохраняет исходный объём; эллипсоид построен по вторым моментам и приведён
к тому же объёму. Для каждой формы заданы пять состояний: исходное и изменения
объёма −1%, −0,5%, +0,5%, +1%. Масштабирование равномерное, центр неподвижен;
у эллипсоида сохраняются ориентация и отношения полуосей. Это численные пробы,
а не принятые физиологические амплитуды сердечного цикла.

На обеих сетках отдельно собраны материальные матрицы средствами EIDORS.
Старые матрицы на новые состояния не перенесены. Учтено частичное заполнение
тетраэдра сердечным материалом; поле аппроксимируется линейными функциями.
Проверены невязка решения, баланс тока, взаимность, контрольные суммы входов
и неизменность расчётной геометрии контактов. Эти проверки не заменяют физическую
валидацию модели.

## 2. Базовый импеданс и изменение сигнала

Базовый импеданс сравнивается при одной и той же форме на двух сетках.
Для анализа реакции на объём из каждого состояния вычитается собственный
базовый импеданс той же формы и той же сетки:

$$\Delta Z_G(\varepsilon)=Z_G(V_0(1+\varepsilon))-Z_G(V_0).$$

Здесь $G$ обозначает форму, $V_0$ — её исходный объём, $\varepsilon$ — заданное
относительное изменение объёма. Поэтому постоянный сдвиг базового импеданса
не переносится автоматически в ошибку изменения сигнала.
"""
    derivative=r"""**Рисунок 1. Изменение импеданса при одинаковом изменении объёма.** В каждом
графике сопоставлена одна форма на двух сетках. Вертикальная координата — изменение
относительно собственной базы. Наклоны и расстояние между кривыми показывают,
сохраняется ли реакция на объём после уточнения поля; сами базовые уровни приведены
отдельно в таблице. Шкала вертикальной оси одинакова во всех панелях. Соединяющие линии служат для чтения пяти расчётных точек.

## 3. Односторонние и центральные оценки чувствительности

При положительной полуширине $\delta V$ определены

$$S_-={Z(V_0)-Z(V_0-\delta V)\over\delta V},\qquad
S_+={Z(V_0+\delta V)-Z(V_0)\over\delta V},\qquad S_c={S_-+S_+\over2}.$$

Все оценки имеют единицы Ом/мл. Отрицательное значение означает уменьшение
импеданса при увеличении объёма в соответствующем интервале. В знаменателе
используется заданное аналитическое изменение объёма. Замена его фактическим
дискретным изменением могла бы скрыть ошибку передачи геометрии в материалы.

Асимметрия в таблице определена как
$A=100|S_+-S_-|/[(|S_-|+|S_+|)/2]$. Она сравнивает два направления при одном шаге;
сама по себе не устанавливает ни физиологическую причину различия, ни существование
единственной производной при стремлении шага к нулю.
"""
    errors_text=r"""**Рисунок 2. Чувствительность трёх форм на двух сетках при полуширине 0,5%.**
Цвет и подпись кодируют одну и ту же знаковую величину в мОм/мл; шкала общая.
Сравнение соседних столбцов выявляет асимметрию, сравнение панелей — влияние сетки.
Центральный столбец усредняет крайние и не заменяет их раздельной проверки.

В следующей таблице сеточное изменение каждой оценки вычислено как
$100(S_{\mathrm{уточн}}-S_{\mathrm{исх}})/|S_{\mathrm{исх}}|$.
Знак относится к численному значению оценки, а не к её модулю. Например, для
отрицательной оценки отрицательное сеточное изменение означает увеличение модуля.

## 4. Воспроизведение отклика индивидуальной формы

Для сферы и эллипсоида отдельно вычислено

$$E_G^{(d)}=100{|S_G^{(d)}-S_I^{(d)}|\over|S_I^{(d)}|},\qquad d\in\{-,c,+\},$$

где индекс $I$ обозначает индивидуальную форму. Сравнение проводится внутри одной
сетки и для одинакового направления. Для односторонней оценки при одинаковом
$\delta V$ это также относительная ошибка конечного изменения сигнала.
Индивидуальная форма служит вычислительным ориентиром; её отклик не является
экспериментальным эталоном.

Преимущество эллипсоида перед сферой описывается разностью $E_{\mathrm{сфера}}-E_{\mathrm{эллипсоид}}$.
Её знак проверяется непосредственно на обеих сетках. Ошибки разных форм могут быть
связаны, поэтому устойчивость их предпочтения не выводится из раздельных ошибок
каждой формы и не требует предположения об их независимости.
"""
    interpretation=f"""**Рисунок 3. Отклонение от индивидуальной формы при полуширине 0,5%.**
Чем меньше число и светлее ячейка, тем ближе соответствующая оценка к индивидуальной
форме. Обе панели используют одну шкалу. Сравниваются именно совпадения отклика,
поэтому меньшая ошибка не означает большей абсолютной чувствительности сборки.

На уточнённой сетке отклонения эллипсоида составляют **{fmt(100*errors['Sminus']['ellipsoid_absolute_relative_error'],1)}%**
при уменьшении объёма, **{fmt(100*errors['derivative']['ellipsoid_absolute_relative_error'],1)}%**
для центральной оценки и **{fmt(100*errors['Splus']['ellipsoid_absolute_relative_error'],1)}%** при увеличении.
Для сферы соответственно получены **{fmt(100*errors['Sminus']['sphere_absolute_relative_error'],1)}%**,
**{fmt(100*errors['derivative']['sphere_absolute_relative_error'],1)}%** и **{fmt(100*errors['Splus']['sphere_absolute_relative_error'],1)}%**.
Раздельное представление этих чисел не позволяет принять близкое среднее за
одинаковую точность в обоих направлениях.

## 5. Достаточность интегрирования и ограничения проверки

Одинаковое число точек интегрирования устраняет различие заданной подробности
расчётов, но не доказывает одинаковой достигнутой точности. Её геометрический
индикатор — отношение разности материальных объёмов двух симметричных состояний
к заданной разности. В таблице приведено отклонение этого отношения от единицы,
в процентах; точное значение ошибки равно нулю. Уровни 4096 и 8192 использованы
здесь только для геометрической проверки. Все новые электрические решения
выполнены при 16 384 точках. Два последних столбца отдельно показывают ошибку
изменения материального объёма относительно исходного состояния при уменьшении
и увеличении; их знаменатель — заданное одностороннее изменение объёма.
"""
    quadrature_explanation="""Дополнительно электрические результаты исходной сетки сопоставлены
с прежними решениями при 4096 точках. В таблице приведены обе односторонние
и центральная оценки, а также модуль различия относительно значения при 16 384
точках. Проверено совпадение сетки, сопротивлений, контактов и исходной геометрии.
Для уточнённой сетки повторное электрическое решение с другим числом точек
в этот блок не входит; геометрическая таблица выше не заменяет такую проверку.
"""
    scale=data["mesh_motion_scales"]
    motion=next(r for r in scale["sphere_radial_displacements"] if r["step"]==.005)
    scale_text=f"""Для оценки пространственных масштабов: радиус равновеликой сферы составляет
{fmt(scale['equal_volume_sphere_radius_mm'],2)} мм. Увеличение её объёма на 0,5%
смещает сферическую границу всего на {fmt(motion['radius_increase_mm'],3)} мм.
Медиана длин уникальных рёбер элементов исходной сердечной области составляет
{fmt(scale['edges']['coarse']['median_mm'],2)} мм на исходной сетке и
{fmt(scale['edges']['fine']['median_mm'],2)} мм на уточнённой. Эти длины описывают
выбранную область и не являются оценкой электрической ошибки.

Такое сопоставление показывает, почему уточнение долей материала и уточнение поля
проверяются отдельно: перемещение границы может быть значительно меньше размера
элемента. Линейные функции внутри него имеют постоянный градиент и не описывают
скачок градиента внутри пересечённого элемента. Из одного отношения размеров
нельзя установить величину ошибки или причину наблюдаемой асимметрии.
"""
    coarse=next(x for x in data["derivatives"]["individual"]["coarse"] if x["step"]==.005)
    preference_rows=data["preference_stability"]
    recommended="эллипсоид" if all(r["preferred_fine"]=="ellipsoid" and r["preferred_coarse"]=="ellipsoid" for r in preference_rows) else "форма с меньшей ошибкой в выбранном направлении"
    practical=f"""**Для следующего этапа сравнения предпочтителен {recommended}.** Это рекомендация
для данного способа построения, одной модели Nix и сборки TEPC-2. При полуширине
0,5% на уточнённой сетке центральная оценка эллипсоида отличается от индивидуальной
на {fmt(100*errors['derivative']['ellipsoid_absolute_relative_error'],1)}%, а сферы —
на {fmt(100*errors['derivative']['sphere_absolute_relative_error'],1)}%. Результат
поддерживает дальнейшую проверку эллипсоида, но ещё не разрешает заменять им сердце
при количественном восстановлении объёма: ошибки двух направлений составляют
{fmt(100*errors['Sminus']['ellipsoid_absolute_relative_error'],1)}% и
{fmt(100*errors['Splus']['ellipsoid_absolute_relative_error'],1)}%.

Равенство объёмов само по себе не обеспечивает равенства электрического отклика.
Сфера в этом опыте воспроизводит примерно половину модуля центральной оценки
индивидуальной формы. Эллипсоид сохраняет также вытянутость и ориентацию формы,
поэтому представляет более содержательное геометрическое приближение. Однако
данный опыт не разделяет влияние этих признаков и положения отдельных участков
границы; приписывать улучшение одному из них пока нельзя.

Уточнение сетки изменило центральную оценку индивидуальной формы по модулю на
{fmt(100*abs(d['derivative']-coarse['derivative'])/abs(coarse['derivative']),2)}%,
а односторонние оценки — на
{fmt(100*abs(d['Sminus']-coarse['Sminus'])/abs(coarse['Sminus']),2)}% и
{fmt(100*abs(d['Splus']-coarse['Splus'])/abs(coarse['Splus']),2)}%.
Асимметрия выросла с {fmt(100*abs(coarse['one_sided_normalized_difference']),1)}%
до {fmt(100*abs(d['one_sided_normalized_difference']),1)}%. Следовательно,
центральная оценка скрывает часть сеточной изменчивости. Индивидуальную форму
необходимо сохранять как вычислительный ориентир и проверять оба направления
раздельно. Этот опыт не устанавливает превосходство эллипсоида над всеми методами
построения сферы из реестра: здесь использована только одна равновеликая сфера.
"""
    conclusion="""Малая ошибка изменения материального объёма не является верхней границей
электрической ошибки: разные участки границы имеют разную чувствительность.
Одновременно малая невязка линейной системы характеризует точность решения
дискретной задачи, а не близость этой задачи к непрерывной модели.

## 6. Практический вывод и следующий шаг

"""+practical+"""
В этом блоке непосредственно проверены изменения сигнала и предпочтение форм
на двух сетках при одинаковой подробности интегрирования. Числа в таблицах позволяют
раздельно оценить три эффекта: изменение базового импеданса, изменение реакции
на объём и расхождение упрощённой формы с индивидуальной.

**Двух уровней внутренней сетки недостаточно для заявления о доказанной
пространственной сходимости.** Допустимая ошибка итоговой оценки функции сердца
заранее не задана; наблюдаемые различия не превращаются автоматически в критерий
пригодности. Наружная сетка и дискретизация контактов в этом опыте не уточнялись.
Не выполнены физическая валидация, совмещение статической КТ с 4D-КТ, переносы
сердца, другие сборки, состояния сопротивлений или участники.

Следующий проверочный расчёт должен уточнить область движущейся границы ещё раз
и проследить изменения конечного сигнала и обеих односторонних оценок. Отдельный
контроль с одинаковым фоновым материалом около границы позволит исследовать,
какая часть асимметрии связана с неоднородным фоном. Основная постановка с мягкими
тканями, лёгкими и костями при этом сохраняется. Причины асимметрии пока не разделены.

Только после численной проверки имеет смысл расширять геометрическую задачу
на движение по данным 4D-КТ. Ни изменение объёма целого эффективного сердца,
ни объём эквивалентной сферы в данном опыте не интерпретируются как ударный объём
желудочка или фракция выброса.
"""
    maxima={key:max(q[level]["maxima"][key] for q in data["numerical_qc"].values() for level in LEVELS) for key in next(iter(data["numerical_qc"].values()))["coarse"]["maxima"]}
    numerical=f"""Во всех 30 сравниваемых состояниях проверки решения пройдены.
Максимальная относительная невязка составила {maxima['max_relative_residual']:.2e}
при допуске 1e−7; максимальное абсолютное расхождение взаимных импедансов —
{maxima['reciprocity_absolute_ohm']:.2e} Ом, меньше даже абсолютной части допуска
1e−8 Ом. Максимальная ошибка баланса тока составила
{maxima['max_current_error_A']:.2e} А при допуске 1e−7 А.
Это подтверждает решение заданных дискретных систем с принятой точностью;
ошибка геометрии и пространственной аппроксимации оценивается отдельно.
"""
    nb=nbformat.v4.new_notebook()
    nb.metadata.update(kernelspec={"display_name":"Python 3","language":"python","name":"python3"},language_info={"name":"python"},scientific_status="exploratory_not_validated",summary_sha256=sha(summary),builder_sha256=sha(__file__))
    setup="from pathlib import Path\nimport sys\nroot=Path.cwd()\nif root.name=='Colab Notebooks': root=root.parent\nsys.path.insert(0,str(root/'Colab Notebooks/cardiac_4d_segmentation'))\nimport heart_volume_mesh_report as report\ndata=report.load(root/"+repr(summary.relative_to(ROOT).as_posix())+")"
    parts=[intro,"CODE physical_table",methods,"CODE baseline_table","CODE response_figure",derivative,"CODE sensitivity_table","CODE sensitivity_figure",errors_text.split("## 4.")[0],"CODE mesh_change_table","## 4."+errors_text.split("## 4.",1)[1],"CODE shape_error_figure",interpretation,"CODE volume_table",quadrature_explanation,"CODE quadrature_table",scale_text,numerical,conclusion]
    nb.cells=[nbformat.v4.new_code_cell(setup)]
    for part in parts:
        nb.cells.append(nbformat.v4.new_code_cell("report."+part[5:]+"(data)") if part.startswith("CODE ") else nbformat.v4.new_markdown_cell(part))
    NotebookClient(nb,timeout=120,kernel_name="python3",resources={"metadata":{"path":str(ROOT)}}).execute()
    nbformat.validate(nb);nbformat.write(nb,dest)
    exporter=HTMLExporter();exporter.exclude_input=True;exporter.exclude_input_prompt=True;exporter.exclude_output_prompt=True
    html,_=exporter.from_notebook_node(nb);dest.with_suffix('.html').write_text(html,encoding="utf-8")
    FIGURES.mkdir(parents=True,exist_ok=True)
    receipt={"notebook_sha256":sha(dest),"html_sha256":sha(dest.with_suffix('.html')),"summary_sha256":sha(summary),"builder_sha256":sha(__file__),"FEM_reexecuted":False,"image_outputs":sum('image/png' in o.get('data',{}) for c in nb.cells for o in c.get('outputs',[]))}
    (FIGURES/'build.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding="utf-8")
    print(json.dumps(receipt))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--summary',type=Path,default=DEFAULT)
    build(parser.parse_args().summary)
