"""Prepare frozen scientific figures. No FEM, fitting, signal processing or network.

Raster relabeling is confined to pre-inspected title rectangles; all other
pixels are checked exactly. Five plots use only whitelisted original plotting
functions and their frozen JSON summaries. Run render_visuals.cjs afterwards.
"""
from pathlib import Path
import copy, hashlib, importlib.util, io, json, re, shutil, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text

REV=Path(__file__).resolve().parents[1]
ROOT=REV.parents[3]
OUT=REV/'figures'
OUT.mkdir(exist_ok=True)
(OUT/'html_exports').mkdir(exist_ok=True)
P28=REV.parent/'revision_20261028/figures'
P29=REV.parent/'revision_20261029_contract/figures/html_exports'
SC=ROOT/'Colab Notebooks/scorpion_v2'

def sha(b): return hashlib.sha256(b).hexdigest()
def rel(p): return p.relative_to(ROOT).as_posix()
def record_file(p):
    with Image.open(p) as im: w,h=im.size
    return dict(path=rel(p),width=w,height=h,unit='px',sha256=sha(p.read_bytes()))

def anon(s):
    for a,b in [('Георгия','У2'),('Георгий','У2'),('Георг','У2'),('Ярослава','У3'),('Ярослав','У3'),('Ника','У1'),('Ник','У1')]:
        s=re.sub(r'(?<![\w])'+a+r'(?![\w])',b,s)
    for a,b in [('nik','У1'),('nix','У1'),('georg','У2'),('yaroslav','У3'),('Н','У1'),('Г','У2'),('Я','У3')]:
        s=re.sub(r'(?<![\w])'+a+r'(?![\w])',b,s)
    return s

FONT=Path('C:/Windows/Fonts/arial.ttf')
def patch_titles(src,dst,patches):
    original=Image.open(src).convert('RGB'); im=original.copy(); d=ImageDraw.Draw(im)
    mask=np.zeros((im.height,im.width),dtype=bool)
    for bbox,lines in patches:
        x0,y0,x1,y1=bbox; d.rectangle((x0,y0,x1-1,y1-1),fill='white');mask[y0:y1,x0:x1]=True
        for text,x,y,size in lines:
            font=ImageFont.truetype(str(FONT),size)
            d.text((x,y),text,font=font,fill='black',anchor='mt')
    before=np.asarray(original);after=np.asarray(im)
    if not np.array_equal(before[~mask],after[~mask]):raise RuntimeError('Scientific pixels changed')
    im.save(dst,dpi=(300,300))
    return {'method':'participant labels replaced only in pre-inspected title areas',
            'title_rectangles_xyxy':[b for b,l in patches],
            'outside_title_pixels_exactly_equal':True,
            'outside_title_pixel_sha256':sha(before[~mask].tobytes()),'source_sha256':sha(src.read_bytes())}

def copy_png(src,dst):
    shutil.copyfile(src,dst)
    return {'method':'byte-for-byte reuse','source_path':rel(src),'source_sha256':sha(src.read_bytes())}

spec=importlib.util.spec_from_file_location('frozen_plot_helper',REV.parent/'revision_20261029_contract/tools/export_anonymous_html_figures.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
pulse=json.loads((SC/'_assets/measured_pulse_volume_20261028.summary.json').read_text(encoding='utf-8'))
fem=json.loads((SC/'_assets/fem_smart_20261026.summary.json').read_text(encoding='utf-8'))
pn,pfunc,pmods=helper.plotting_namespace(SC/'build_measured_pulse_report.py',{'show_figure'},{'PROFILE','COLOR','CHAMBERS'},np,plt)
fn,ffunc,fmods=helper.plotting_namespace(SC/'build_scorpion_fem_smart_report.py',{'_plot_setup','case_parts','fig_svd','fig_breath'},{'ROLES','ROLE_NAMES'},np,plt)

def render_frozen(create,dst,data):
    data_sha=sha(helper.encoded(data));fig=create();fig.canvas.draw()
    bbox=fig.get_tightbbox(fig.canvas.get_renderer()).padded(.1).frozen();fig.set_layout_engine('none')
    before=helper.numeric_scene(fig,np);changes=[]
    for t in fig.findobj(match=Text):
        old=t.get_text();new=anon(old)
        if old!=new:t.set_text(new);changes.append({'after':new})
    fig.canvas.draw();after=helper.numeric_scene(fig,np)
    if before!=after or sha(helper.encoded(data))!=data_sha:raise RuntimeError('Numerical scene changed')
    texts=[t.get_text() for t in fig.findobj(match=Text)]
    if any(re.search(r'Ник|Георг|Ярослав|\b(?:nik|nix|georg|yaroslav)\b',s,re.I) for s in texts):raise RuntimeError('Internal name remains')
    fig.savefig(dst,dpi=220,bbox_inches=bbox,metadata={'Software':'Frozen plotting functions; participant codes only'})
    plt.close(fig)
    return dict(method='original plotting function; only participant Text objects relabeled',
                frozen_data_unchanged=True,numerical_scene_before_after_equal=True,
                numerical_scene_sha256=sha(helper.encoded(before)),display_changes=changes,
                matplotlib=matplotlib.__version__)

exports=[]
old_manifest=json.loads((P29/'manifest.json').read_text(encoding='utf-8'))
byfile={x['filename']:x for x in old_manifest['figures']}
for name in ['01_n_ct_tissue_masks.png','02_n_candidate_lateral_montage.png','03_g_ct_contour_regularization.png']:
    src=P29/name;dst=OUT/'html_exports'/name
    if name.startswith('01'):
        trans=patch_titles(src,dst,[((0,0,1864,43),[('КТ участника У1, аксиальный срез z = −154,86 мм',932,3,34)])])
    elif name.startswith('02'):
        trans=patch_titles(src,dst,[((0,0,2625,48),[('У1: индивидуальная геометрическая гипотеза боковой установки',1312,2,35)])])
    else:trans=copy_png(src,dst)
    old=byfile[name]
    exports.append(dict(filename=name,**record_file(dst),source_html=old['source_path'],source_sha256=old['source_html_sha256'],
                        source_locator=old['locator'],caption=anon(old['caption']),restrictions=[anon(x) for x in old['limitations']],transformation=trans))

for name,create,data,entry in [
    ('04_tt_ecg_four_cycles.png',lambda:pn['show_figure'](pulse,1),pulse,'show_figure(summary,1)'),
    ('05_tissue_identifiability_singular_values.png',lambda:fn['fig_svd'](fem),fem,'fig_svd(summary)'),
    ('06_measured_pulse_vs_volume_probe.png',lambda:pn['show_figure'](pulse,3),pulse,'show_figure(summary,3)'),
    ('07_ct_whole_heart_and_chambers.png',lambda:pn['show_figure'](pulse,4),pulse,'show_figure(summary,4)'),
    ('08_respiratory_contrast_observation_model.png',lambda:fn['fig_breath'](fem),fem,'fig_breath(summary)'),
]:
    dst=OUT/'html_exports'/name;trans=render_frozen(create,dst,data);old=byfile[name]
    exports.append(dict(filename=name,**record_file(dst),source_html=old['source_path'],source_sha256=old['source_html_sha256'],
                        source_locator=old['locator'],caption=anon(old['caption']),restrictions=[anon(x) for x in old['limitations']],
                        plotting_entrypoint=entry,transformation=trans))

figures=[]
def add(id,slug,chapter,caption,purpose,restrictions,source,aliases=None):
    e=dict(id=id,aliases=aliases or [slug],suggested_chapter=chapter,caption=caption,purpose=purpose,
           restrictions=restrictions,source=source,path=rel(OUT/f'{id}_{slug}.png'),width=None,height=None,
           width_cm=17,height_cm=None,status='prepared_pending_visual_review',assets=[])
    figures.append(e);return e,OUT/f'{id}_{slug}.png'
def from_export(id,slug,ch,idx,caption,purpose,restr):
    ex=exports[idx-1];e,dst=add(id,slug,ch,caption,purpose,restr,{'html':ex['source_html'],'locator':ex['source_locator'],'sha256':ex['source_sha256']})
    shutil.copyfile(ROOT/ex['path'],dst);e.update(record_file(dst));e['transformation']=ex['transformation'];return e
def cached(id,slug,ch,src,caption,purpose,restr,oldfig,patch=None):
    e,dst=add(id,slug,ch,caption,purpose,restr,{'document':'author_snapshot.docx','figure':oldfig,'cached_png':rel(P28/src),'sha256':sha((P28/src).read_bytes())})
    e['transformation']=patch_titles(P28/src,dst,patch) if patch else copy_png(P28/src,dst);e.update(record_file(dst));return e

e=cached('V01','method_chain',1,'36893a49eaf9.png',
    'Схема определения функций сердца: индивидуальная КТ задаёт геометрию, поясная ЭИТ — оценки электрических параметров тканей, ТТРКГ с ЭКГ — наблюдения для восстановления динамики. Пунктир обозначает проверку по 4D-КТ; камерное преобразование выделено в самостоятельный этап.',
    'Объяснить роли измерений и порядок передачи параметров.', ['Схема исследуемых преобразований; 4D-КТ используется как референс разработки.'],'4.1')
shutil.copyfile(ROOT/'Colab Notebooks/Отчёт_ГПХ_РНЦХ/report/figures/method/01_method_sequence.svg',OUT/'V01_method_chain.svg')
e['svg_path']=rel(OUT/'V01_method_chain.svg')
add('V02','measurement_setup',2,
    'Схема проведения исследований и измерительных конфигураций. Все процедуры выполняли лёжа на спине с вытянутыми за голову, не сцепленными руками. В исследовании участвовали три мужчины 25–30 лет: У1 и У3 — нормостеники, У2 — гиперстеник. Боковая сборка имеет токовые A, B и измерительные M, N контакты; варианты ТТРКГ различаются группировкой контактов на руках. Пояс ЭИТ располагали циркулярно вокруг грудной клетки приблизительно на уровне сосков либо непосредственно ниже этого уровня.',
    'Показать позу и роли контактов до обработки сигналов.',
    ['Векторная схема без анатомического масштаба; положение контактов на индивидуальной КТ не измерено.',
     'ЭИТ и ТТРКГ относятся к самостоятельным регистрациям; общий протокол не устанавливает их синхронность.'],
    {'protocol':'AUTHOR_CONFIRMATIONS.md','document_blocks':['p0730–p0777'],'html':'H40 §1; H43 §2'})
from_export('V03','ct_segmentation',2,1,
    'КТ У1 и автоматические тканевые маски на аксиальном сечении вблизи центроида сердечной области. Показаны исходное сечение, лёгкие, жир и кости; розовый контур соответствует автоматической маске целого сердца. Координаты RAS приведены в миллиметрах.',
    'Показать анатомический источник и выделение тканей до построения модели.',
    ['Маски автоматические, их границы не приняты независимым экспертом.','Изображение не является примером ручной сегментации Inobitec.'])
from_export('V04','geometry_preparation',3,3,
    'Локальные КТ-сечения У2 до и после одного замыкания наружного контура. Синий контур — исходный, оранжевый — изменённый; красные кресты — точки локальной пробы вершин. Верхний ряд соответствует направлению проверки до → после, нижний — после → до; индексы вокселей отсчитываются от нуля.',
    'Объяснить контроль изменения геометрии и сохранённый отказ допуска.',
    ['Прежний допуск 1,25 мм не выполнен; анатомическая приемлемость коррекции не установлена.','Выборочные и локальные максимумы не являются глобальным расстоянием Хаусдорфа.'])
from_export('V05','lateral_ct_placement',2,2,
    'Кандидатная область боковой сборки на КТ У1. Жёлтая точка показывает геометрическую гипотезу центра, зелёные точки — варианты его положения; A–M–N–B обозначают пример сборки 140 мм по кожной дуге. Оси заданы в миллиметрах; системы RAS и LPS подписаны отдельно.',
    'Различить подтверждённый порядок установки и его расчётную локализацию на КТ.',
    ['Положение является геометрическим сценарием, а не измеренными координатами фактической установки.','Диагностический контур не прошёл все проверки подготовки.'])

plot_specs=[
 ('V06','breathing_windows',2,1,'Полная запись У1: общий импеданс, ЭКГ и пульсовая полоса. Цветные области показывают внутренние кандидатные окна задержек на вдохе и выдохе; неоднозначные переходы сохраняются. Общий протокол: спокойное дыхание, задержка на вдохе, усиленное дыхание, задержка на выдохе.',
  'Показать процедуру выбора окон до отбора сердечных циклов.', ['Границы восстановлены ретроспективно; приборные метки команд отсутствуют.','Обзорные кривые прорежены с сохранением минимумов и максимумов; расчёты использовали исходные отсчёты.']),
 ('V08','multisize_levels',2,2,'Боковой импеданс У1 при токовых базах 60, 100 и 140 мм: медианы на кандидатных задержках глубокого вдоха и выдоха, Ом. Основной размерный ряд записан при ТТРКГ 2×2; точка 60 мм при 1×1 относится к отдельному включению.',
  'Показать реальный размерный ряд до подбора тканевых параметров.', ['Монтажи регистрировались последовательно; контакт и физиологическое состояние могли различаться.','Это импеданс отведения, не сопротивление отдельной ткани.']),
 ('V10','h_rho_profile',6,16,'Условный подбор двух сопротивлений по трём размерам боковой сборки У1 при отдельных фиксированных h. Показаны среднеквадратичная невязка, ρ₁ и ρ₂ для вдоха и выдоха; каждое h задаёт самостоятельную обратную задачу.',
  'Показать зависимость результата от h и выход на границы поиска.', ['Сетка h не является анатомическим диапазоном или доверительным интервалом.','Минимумы вдоха и выдоха не измеряют дыхательное изменение толщины.']),
 ('V11','heldout_size',6,18,'Пульсовой прогноз исключённого размера боковой сборки У1 по двум другим при h=120 мм. Сплошные линии — сохранённые медианные циклы, пунктир — линейный прогноз; фаза QRS–QRS безразмерна.',
  'Показать процедуру проверки достаточности двух сборок.', ['Базовые сопротивления уже подбирались по трём статическим уровням; исключена только пульсовая кривая.','Критерий малости изменения на вдохе не выполнен; воспроизводимость между последовательными записями не установлена.']),
]
for id,slug,ch,num,caption,purpose,restr in plot_specs:
    add(id,slug,ch,caption,purpose,restr,{'html':rel(next(SC.glob('33.40*.html'))),'figure':num,'sha256':sha(next(SC.glob('33.40*.html')).read_bytes())})

add('V07','cycle_processing',4,
    'Четыре сохранённых цикла обычной записи У1: ЭКГ, ТТРКГ 2×2 и боковая сборка 140 мм.',
    'Показать временной блок обычной пары ТТРКГ и боковой сборки.',
    ['Механические конечные фазы не назначены; использованы сохранённые массивы без новой обработки.'],
    {'html':rel(next(SC.glob('33.40*.html'))),'figures':[4,5]})

add('V09','layered_model',3,
    'Схема однородного полупространства и двухслойной модели при одинаковом тетраполярном монтаже A–M–N–B. Внешняя токовая база равна L, внутренняя измерительная — L/2; наружный слой задаётся эффективными h и ρ₁, внутреннее полупространство — ρ₂. Контакты имеют конечные площадки.',
    'Постепенно ввести геометрию и параметры аналитической модели.',
    ['Схема без анатомического масштаба; линии электрического поля не изображены.','Усреднение по дискам в аналитической модели отличается от полной контактной модели CEM.'],
    {'document_blocks':['p0730–p0732','p1572–p1583'],'html':rel(next(SC.glob('33.40*.html'))),'cell':'928c4f63'})
add('V12','shape_transform_scheme',3,
    'Схема аппроксимации опорной области и её преобразований. Сфера и моментный эллипсоид сравниваются при общем объёме; масштабирование s изменяет объём в s³ раз, перенос d меняет положение. Круг и эллипс показаны как проекции трёхмерных форм.',
    'Объяснить операции до графиков объёма, перекрытия и электрического отклика.',
    ['Условные контуры не являются индивидуальной STL или измеренной геометрией.','Ориентация и отношения полуосей в показанном законе фиксированы; механическое вытеснение тканей не моделируется.'],
    {'document_blocks':['p0974–p0991','p1158–p1160','p2021–p2031'],'equations':['6.6','6.9','6.17','9.1']})
cached('V13','geometry_controls',5,'27676a97e7a0.png',
    'Геометрическое сравнение форм У1 на первом доступном R–R: объём, изменение объёма от опоры, ошибка перемещения и коэффициент Dice. T0 — исходная маска, I1–I3 — индивидуальные преобразования, G1/G2/G3a — сферические варианты; линии соединяют доступные фазовые оценки.',
    'Различить выполнение заданных ограничений и совпадение анатомических областей.',
    ['Целевые объёмы и центры известны из КТ; совпадение по построению не является восстановлением по сигналу.','Исходные численные таблицы сейчас недоступны; использован проверенный сохранённый PNG.'],'6.3')
cached('V14','eit_baseline_residual',6,'reader/08_01_baseline_fit.png',
    'Исправленная базовая инверсия ЭИТ У1, запись 02, опорный участок 36–44 с. Синий — измеренный вектор в принятой шкале читателя, оранжевый — FEM-прогноз; снизу — знаковый остаток в мВ по 208 измерительным строкам. Вольтовая невязка составляет 32,4365%.',
    'Показать данные, прогноз и необъяснённый остаток до интерпретации тканевых параметров.',
    ['Положение пояса и контактные параметры заданы алгоритмически.','Независимая абсолютная калибровка шкалы не выполнена.'],'8.1',
    [((0,0,1920,36),[('У1 · ЭИТ, запись 02 · опорный участок 36–44 с · невязка 32,44%',960,3,29)])])
cached('V15','tissue_contributions',6,'ee60fc3b2861.png',
    'Условные вклады мягкотканного фона, единого лёгочного компартмента и сердечной области в отведение «рука–рука» У1 на четырёх кандидатных плато. Время — внутри соответствующей записи ЭИТ, вклад — в мОм; полоса 0,7–3 Гц, частота модели 79,505 кГц.',
    'Показать знак и временную форму переноса тканевых изменений в другое отведение.',
    ['Одновременно измеренная ТТРКГ на рисунке отсутствует.','Проекции относятся к проводимости фиксированных областей; физиологические доли пульса не определены.'],'8.2')
cached('V16','geometry_tissue_mixing',6,'reader_motion/09_5_02_target_chain_volume_sensitivity.png',
    'Объёмный коэффициент ТТРКГ до и после тканевой инверсии геометрического сигнала. Серый — исходный коэффициент, зелёный — остаток при S2, оранжевый — при S3; S2 освобождает фон и лёгкое, S3 дополнительно сердце. Истинные изменения тканевых сопротивлений равны нулю.',
    'Показать риск вычитания геометрического сигнала как тканевого.',
    ['Условный вычислительный контроль У1; не измеренная ошибка коррекции у человека.','Д4 и О-STL — разные сценарные материальные опоры; масштабы панелей различаются.'],'8.3 / 9.5')
cached('V17','motion_sensitivity',5,'reader_motion/09_3_tt_motion_sensitivity.png',
    'Чувствительность ТТРКГ У1 к переносу формы L1. Верхний ряд — Д4, нижний — собственная материальная опора Э-L1; показаны отношения координатной и объёмной производных, изменение Kᵥ в семи положениях и кажущийся объём при заданных сдвигах ±5 мм с постоянным объёмом.',
    'Разделить перенос сердца, его электрический вклад и ошибку пересчёта в объём.',
    ['±5 мм — диагностические воздействия, а не измеренный физиологический диапазон.','Большой кажущийся объём при малом Kᵥ означает отказ пересчёта; смена монтажа является другим воздействием.'],'9.3')
cached('V18','finite_inverse_errors',5,'reader_motion/09_4_03_finite_probe_volume_errors.png',
    'Ошибки объёма на восьми конечных STL-пробах У1. Вверху — диагностическая семипараметрическая инверсия ЭИТ, внизу — оценка тканевых изменений при фиксированной геометрии, их вычитание из ТТРКГ и деление на Kᵥ. STL, G1 и L1 — обратные формы; слева Д4, справа О-STL.',
    'Проверить перенос локальных выводов на заданные конечные воздействия.',
    ['Диагностическая геометрическая ЭИТ-инверсия не заменяет целевую схему восстановления по ТТРКГ.','Сохранены крупные знаковые ошибки и разные масштабы панелей; истинные Δρ равны нулю.'],'9.4')
cached('V19','volume_rho_compensation',5,'reader_montage/ch09_06_finite_compensation.png',
    'Конечная компенсация объёмных проб ±1% изменением эффективного сопротивления сердца У1. Контур — чистая объёмная проба, заполнение — линейно выбранная компенсация, крест — результат одного уточнения Ньютона. Параметры подобраны по исходному монтажу и проверены в двух альтернативных.',
    'Показать конечный контрпример однозначному объяснению импеданса одним объёмом.',
    ['Малый остаток специально построен и не является точностью прибора.','Альтернативные монтажи расчётные; физиологический диапазон компенсирующих изменений не установлен.'],'9.6')
from_export('V20','pulse_volume_negative',7,6,
    'Размах четырёх измеренных циклов ТТРКГ каждого участника и размах ответа сферы в проверенном диапазоне 0,95–1,05V₀. Столбцы — измерения, линии — два материальных сценария; в каждом из двенадцати интервалов измеренный размах больше модельного.',
    'Сохранить отрицательную проверку масштаба простой объёмной модели.',
    ['Пульсовые вклады других тканей ещё не вычтены; центр и фон модели фиксированы.','Экстраполяция за рассчитанный диапазон не является восстановленным УО; результат не исключает другие сферические постановки.'])
from_export('V21','whole_heart_chambers',7,7,
    'Изменения объёмов автоматических масок целого сердца и камер У1/У2 от первой доступной фазы каждого отдельного R–R. Временная ось относится к собственному КТ-интервалу; четыре панели имеют разные нулевые опоры. Отрезки соединяют доступные точки и не замыкают циклы.',
    'Обосновать самостоятельный переход от наружной сердечной области к камерным объёмам.',
    ['Автоматические маски не имеют независимой оценки анатомической ошибки.','Выборочные размахи неполных окон не являются подтверждённым УО; КТ и импеданс относятся к разным регистрациям.'])
cached('V22','wrong_extrema_counterexample',7,'reader_chamber/10_01_guarded_chambers.png',
    'Камерные объёмы У1 на отложенных фазах и контрпример близкого размаха при неверных экстремумах. Чёрный — КТ, остальные линии — сохранённые прогнозы по импедансу, индивидуальной фазовой формуле и импедансу с производной. На 43% максимум электрического прогноза ЛЖ совпадает с минимумом КТ, хотя ошибка размаха составляет +0,168 мл.',
    'Различить правильный размах, правильную траекторию и функциональную интерпретацию.',
    ['Восемь проверочных точек собраны из четырёх калибровок при одной известной опоре 14%.','Окно 14–92% неполное; линии не являются траекторией одной формулы, размах не назван УО.'],'10.1',
    [((0,0,1925,103),[('У1: камерные объёмы на отложенных фазах',962,42,36)])])

# Supplemental scientific panels retain the same V identity, with explicit aliases.
extra_jobs=[('V07','four_cycles',exports[3]),('V10','tissue_singular_values',exports[4]),('V08','respiratory_contrast',exports[7])]
for id,alias,ex in extra_jobs:
    e=next(x for x in figures if x['id']==id);e['aliases'].append(alias)
    e['assets'].append(dict(alias=alias,**ex))
e=next(x for x in figures if x['id']=='V04');dst=OUT/'V04_anatomical_model.png'
tr=copy_png(P28/'dfaf500992b1.png',dst)
e['assets'].append(dict(alias='anatomical_model',**record_file(dst),caption='Расчётная геометрия У1: сердце, лёгкие, наружная поверхность, модельные руки и 16 площадок ЭИТ. Положение пояса на КТ алгоритмическое; авторский ориентир установки пояса не является измеренной КТ-плоскостью.',source={'cached_png':rel(P28/'dfaf500992b1.png'),'document_figure':'6.1 / 7.1'},transformation=tr,restrictions=['Положение пояса и руки являются модельными входами.']))
e['aliases'].append('anatomical_model')
for e in figures:
    if e['width'] is not None:e['height_cm']=round(e['width_cm']*e['height']/e['width'],2)

manifest=dict(schema_version=1,figure_mode='with_figures',formatting_mode='section_only',
    canonical_source='author_snapshot.docx',canonical_sha256=sha((REV/'audit/author_snapshot.docx').read_bytes()),
    author_confirmations_sha256=sha((REV/'AUTHOR_CONFIRMATIONS.md').read_bytes()),
    participant_display={'У1':'Н','У2':'Г','У3':'Я'},
    participant_context={'sex':'мужчины','age_years':'25–30','body_type':{'У1':'нормостеник','У2':'гиперстеник','У3':'нормостеник'},'position':'лёжа на спине; руки вытянуты за головой, не сцеплены'},
    figures=sorted(figures,key=lambda x:x['id']),html_exports=exports,
    constraints={'FEM_run':False,'fitting_run':False,'signal_processing_run':False,'raw_data_published':False,'imagegen_used':False,'chapters_modified':False},
    runtime={'python':sys.version,'matplotlib':matplotlib.__version__},
    alias_policy='aliases are semantic; explicit alias_paths select supplemental panels; chapter aliases may be added later')
for e in manifest['figures']:
    e['alias_paths']={a:e['path'] for a in e['aliases']}
    for a in e['assets']:e['alias_paths'][a['alias']]=a['path']
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'primary_figures':len(figures),'html_exports':len(exports),'prepared':sum(x['width'] is not None for x in figures)},ensure_ascii=False))
