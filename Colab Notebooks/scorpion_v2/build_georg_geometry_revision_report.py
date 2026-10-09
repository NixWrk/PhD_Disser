"""Corrected geometry reader; deliberately performs no FEM inversion."""
from pathlib import Path
import argparse

import nibabel as nib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import trimesh
from scipy.ndimage import map_coordinates

from build_georg_fem_reports import ROOT, OUT as OLD, GEOMETRY, new_cells, export, number, table, read, sha, write

OUT=ROOT/'MATLAB_TRKG4_real_subjects/output/exploratory/scorpion_georg_geometry_20261009'
SOURCE=ROOT/'MATLAB_TRKG4_real_subjects/data/georg/stl/body_solid_ct_1mm_r10_volume_fill.stl'


def verify(out=OUT):
    out=Path(out);g=read(out/'geometry.local.json')
    if sha(SOURCE)!=g['source_sha256'] or sha(out/'body_embedded_arms_370mm.stl')!=g['surface_sha256']:
        raise ValueError('Geometry input hash mismatch')
    producer=ROOT/'MATLAB_TRKG4_real_subjects/tools/scorpion_georg_geometry_revision.py'
    if sha(producer)!=g['producer_sha256']:raise ValueError('Geometry producer changed')
    if not g['watertight'] or g['connected_components']!=1 or g['self_intersecting_faces']:
        raise ValueError('Geometry quality check failed')
    return g


def join_table(g):
    return pd.DataFrame([{'Рука':{'left':'Правая (x < 0)','right':'Левая (x > 0)'}[key],
        'Заглубление, мм':j['inset_mm'],'Сдвиг y, мм':j['centre_shift_yz_mm'][0],
        'Сдвиг z, мм':j['centre_shift_yz_mm'][1],
        'Площадь сечения цилиндра, мм²':j['analytic_section_area_mm2'],
        'Площадь многоугольника, мм²':j['polygon_area_mm2'],
        'Доля сечения внутри тела, %':100*j['cross_section_covered_fraction']}
        for key,j in g['joins'].items()])


def scene(out=OUT):
    out=Path(out);surface=trimesh.load(out/'body_embedded_arms_370mm.stl',force='mesh')
    v=surface.vertices;f=surface.faces
    fig=go.Figure(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],
        opacity=.25,color='#bbb',name='Исправленная поверхность',hoverinfo='skip'))
    contacts=read(out/'contacts.local.json')
    for i,size in enumerate(contacts['sizes_mm']):
        xyz=np.array(contacts['lateral']['centres_by_size_mm'][i])
        fig.add_trace(go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='lines+markers',
            name=f'Боковая сборка {size} мм',marker=dict(size=3),visible=True if size==140 else 'legendonly'))
    for montage in ('1x4','2x2'):
        groups=contacts['ttrkg_'+montage]
        for i,points in enumerate(groups):
            xyz=np.array(points);current=i in (0,3)
            fig.add_trace(go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='markers',
                name=f"{montage.replace('x','×')}: {'токовые' if current else 'измерительные'}",
                marker=dict(size=6 if current else 4,color='#c43131' if current else '#247bb5'),
                showlegend=i in (0,1),legendgroup=f'{montage}-{current}',
                visible=True if montage=='1x4' else 'legendonly',
                hovertemplate='x=%{x:.2f}; y=%{y:.2f}; z=%{z:.2f} мм<extra></extra>'))
    fig.update_layout(template='plotly_white',height=680,title='Продольные боковые сборки и верхние/нижние токовые контакты',
        scene=dict(aspectmode='data',xaxis_title='x LPS, мм',yaxis_title='y LPS, мм',zaxis_title='z LPS, мм',
            camera=dict(eye=dict(x=-1.3,y=-1.8,z=.9)),uirevision='corrected-georg-camera'),
        legend=dict(orientation='h',y=-.05),margin=dict(b=110))
    return fig


def prepare_sections(out=OUT):
    out=Path(out)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    # The CT affine is RAS; displayed geometry is LPS.
    ct=nib.load(OLD/'segmentation/static_ct.nii.gz');hu=np.asanyarray(ct.dataobj)
    labels=np.zeros(ct.shape,dtype=np.uint8)
    colours=['#4097dd','#dbc790','#e76b6b','#e0ab28']
    for name,value in [('fat',4),('heart',3),('bone',2),('lung',1)]:
        image=nib.load(OLD/'segmentation'/(name+'.nii.gz'))
        np.testing.assert_allclose(image.affine,ct.affine,rtol=0,atol=1e-5)
        labels[np.asanyarray(image.dataobj)>0]=value
    contacts=read(out/'contacts.local.json');y=contacts['lateral']['y_mm']
    x=np.linspace(-225,225,650);z=np.linspace(-371.5,-19.5,600);xx,zz=np.meshgrid(x,z)
    transform=np.linalg.inv(ct.affine)@np.diag([-1.,-1.,1.,1.])
    xyz=np.c_[xx.ravel(),np.full(xx.size,y),zz.ravel()]
    ijk=nib.affines.apply_affine(transform,xyz)
    image=map_coordinates(hu,ijk.T,order=0,mode='constant',cval=-1024).reshape(xx.shape)
    mask=map_coordinates(labels,ijk.T,order=0,mode='constant',cval=0).reshape(xx.shape)
    from lung_depth_profile import profile
    source=trimesh.load(SOURCE,force='mesh')
    lung=nib.load(OLD/'segmentation/lung.nii.gz')
    depth=profile(source,np.asanyarray(lung.dataobj)>0,lung.affine,y,
        contacts['lateral']['z_mm'])
    depth.update(surface_sha256=sha(SOURCE),lung_mask_sha256=sha(OLD/'segmentation/lung.nii.gz'),
        producer_sha256=sha(Path(__file__).with_name('lung_depth_profile.py')),
        builder_sha256=sha(__file__))
    write(out/'lung_depth_profile.local.json',depth)
    pd.DataFrame({k:depth[k] for k in ['s_mm','z_lps_mm','skin_x_lps_mm','lung_x_lps_mm','depth_mm','flags']}).to_csv(
        out/'lung_depth_profile.local.csv',index=False)
    points=np.array(contacts['lateral']['centres_by_size_mm'][contacts['sizes_mm'].index(140)])
    fig,axes=plt.subplots(1,2,figsize=(12,6),constrained_layout=True)
    for i,ax in enumerate(axes):
        extent=[x[0],x[-1],z[0],z[-1]]
        ax.imshow(image,extent=extent,origin='lower',cmap='gray',vmin=-700,vmax=300)
        ax.imshow(np.ma.masked_where(mask==0,mask),extent=extent,origin='lower',
            cmap=ListedColormap(colours),vmin=1,vmax=4,alpha=.5,interpolation='nearest')
        if i:
            ax.plot(points[:,0],points[:,2],'o-',color='white',markeredgecolor='black',markersize=6)
            for name,p in zip(['A','M','N','B'],points):
                ax.annotate(name,(p[0],p[2]),xytext=(-16,3),textcoords='offset points',color='white',
                    bbox=dict(facecolor='black',alpha=.5,pad=1))
            ax.plot(depth['skin_x_lps_mm'],depth['z_lps_mm'],color='white',lw=1,label='Наружная поверхность')
            ax.plot(depth['lung_x_lps_mm'],depth['z_lps_mm'],color='#00efff',lw=2,label='Первая граница лёгкого')
            for offset in [-70,-35,0,35,70]:
                j=depth['s_mm'].index(offset)
                ax.plot([depth['skin_x_lps_mm'][j],depth['lung_x_lps_mm'][j]],
                    [depth['z_lps_mm'][j]]*2,color='#00efff',lw=1,ls='--')
            ax.set(xlim=(-230,0),ylim=(-300,-120),title='Сборка 140 мм; общий центр s = 0')
        else:ax.set(title=f'Полное сечение y LPS = {number(y)} мм')
        ax.set(xlabel='x LPS, мм',ylabel='z LPS, мм');ax.set_aspect('equal')
    fig.legend(handles=[Patch(color=c,label=n) for c,n in zip(colours,['Лёгкие','Кости','Сердце','Жир'])],
        loc='lower center',ncol=4,bbox_to_anchor=(.5,-.02))
    fig.savefig(out/'ct_longitudinal_sections.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4.5),constrained_layout=True)
    ax.plot(depth['s_mm'],depth['depth_mm'],color='#247bb5',lw=2)
    for offset,label in [(-70,'A'),(-35,'M'),(0,'Центр'),(35,'N'),(70,'B')]:
        j=depth['s_mm'].index(offset);value=depth['depth_mm'][j]
        ax.axvline(offset,color='#777',lw=.8,ls='--')
        ax.plot(offset,value,'o',color='#247bb5')
        ax.annotate(label,(offset,value),xytext=(0,12),textcoords='offset points',ha='center')
    ax.set(xlabel='s — смещение вдоль оси сборки от общего центра, мм',
        ylabel='d — расстояние до маски лёгкого\nвдоль +x LPS, мм',xlim=(-74,74))
    ax.margins(y=.12)
    ax.grid(alpha=.25)
    fig.savefig(out/'lung_depth_along_montage.png',dpi=180);plt.close(fig)
    # Actual sections of the unextended surface at the chosen buried interfaces.
    import sys
    tool_dir=ROOT/'MATLAB_TRKG4_real_subjects/tools';sys.path.insert(0,str(tool_dir))
    from scorpion_georg_geometry_revision import section_polygon, RADIUS
    from shapely.geometry import Point
    source=trimesh.load(SOURCE,force='mesh');g=verify(out)
    fig,axes=plt.subplots(1,2,figsize=(12,6),constrained_layout=True)
    for ax,(key,j) in zip(axes,g['joins'].items()):
        polygon=section_polygon(source,j['x_mm'])
        parts=polygon.geoms if polygon.geom_type=='MultiPolygon' else [polygon]
        for part in parts:
            xy=np.array(part.exterior.coords);ax.fill(xy[:,0],xy[:,1],color='#ddd',edgecolor='#555')
        disk=Point(j['centre_yz_mm']).buffer(RADIUS,quad_segs=64);xy=np.array(disk.exterior.coords)
        ax.fill(xy[:,0],xy[:,1],color='#247bb5',alpha=.5,edgecolor='#15527b')
        ax.set(title=f"{'Правая' if key=='left' else 'Левая'} рука; x = {number(j['x_mm'])} мм",
            xlabel='y LPS, мм',ylabel='z LPS, мм');ax.set_aspect('equal');ax.grid(alpha=.2)
    fig.savefig(out/'full_cylinder_joins.png',dpi=180);plt.close(fig)
    write(out/'section_provenance.local.json',dict(ct_sha256=sha(OLD/'segmentation/static_ct.nii.gz'),
        mask_sha256={n:sha(OLD/'segmentation'/(n+'.nii.gz')) for n in ['heart','lung','bone','fat']},
        plane_y_lps_mm=y,all_contacts_in_display_plane=True,poses_are_examples=True,
        interpolation='nearest-neighbour CT and masks',builder_sha256=sha(__file__)))


def build(out=OUT):
    out=Path(out);g=verify(out);prepare_sections(out)
    cells,md,code=new_cells()
    md('# 20.20. Исправленная индивидуальная геометрия КТ/FEM Георгия\n\n'
       '**Редакция от 9 октября 2026 года. Статус: подготовка геометрии; новый FEM-расчёт не выполнен.** '
       'Автор уточнил ориентацию боковых сборок и токовых контактов и потребовал полного сечения '
       'соединения цилиндров рук с телом. Предыдущая геометрия этим условиям не соответствовала. '
       'Численные результаты 33.22 от 8 октября сохранены как исторический расчёт и не относятся '
       'к настоящей геометрии.\n\n'
       '## 1. Входы и последовательность\n\n'
       'Для подготовки конечноэлементной модели (FEM) использованы прежняя наружная поверхность '
       'полного статического КТ-тома и четыре маски: '
       'сердце, лёгкие, кости и жир. Остальной объём представляет мягкотканный фон. '
       'Маски получены автоматически TotalSegmentator 2.18; независимая точность границ не установлена. '
       'КТ снята на глубоком вдохе с задержкой дыхания и относится к другой сессии, чем Scorpion v2. '
       'Динамическая КТ сердца не подменяет полный том грудной клетки.\n\n'
       'Порядок данного этапа: проверить сечения исходной поверхности → подобрать заглубление цилиндров '
       '→ объединить объёмы и проверить поверхность → исправить контакты → показать сечения КТ вдоль '
       'боковых сборок → подготовить полный поиск положения и новую FEM-инверсию. '
       'Выполнены геометрические этапы; поиск положения и новая инверсия ещё не выполнены.')
    relative=out.relative_to(ROOT).as_posix()
    code("from pathlib import Path\nimport sys\nfrom IPython.display import display,HTML,Image\nhere=Path.cwd()\n"
         "module_dir=next(p for p in [here,here/'scorpion_v2',here/'Colab Notebooks/scorpion_v2'] if (p/'build_georg_geometry_revision_report.py').exists())\n"
         "sys.path.insert(0,str(module_dir.resolve()))\nfrom build_georg_geometry_revision_report import ROOT,verify,join_table,scene,table\n"
         f"OUT=ROOT/{relative!r}\ng=verify(OUT)\n"
         "def show(fig): display(HTML(fig.to_html(full_html=False,include_plotlyjs=True,config={'displaylogo':False})))\n")
    md('## 2. Полное соединение цилиндров рук\n\n'
       'Автор сообщил обхват 37 см для обеих рук на уровне токовых электродов и принял цилиндрическую '
       'модель. Радиус равен 370/(2π) = 58,8873 мм; площадь поперечного сечения — 10 894,16 мм². '
       'Прежние срезы наружной поверхности имели площади около 2721 и 3676 мм². '
       'Присоединение большого цилиндра только через эти срезы не обеспечивало требуемого полного контакта.\n\n'
       'Новый подбор рассматривает плоскости, смещённые внутрь исходной поверхности на 0,5–109,5 мм '
       'с шагом 1 мм. В каждой плоскости центр круга выбирается в области, где всё сечение цилиндра '
       'помещается внутри тела, с геометрическим запасом 0,15 мм. Из допустимых кандидатов выбирается '
       'минимум суммы квадратов заглубления и сдвига центра относительно исходного среза. '
       'Это геометрическое правило, а не восстановление анатомического положения плеча.\n\n'
       'Получены заглубления 22,5 мм справа и 16,5 мм слева; сдвиги центров представлены в таблице. '
       'Полное сечение находится внутри исходного объёма. Круг представлен 256-сторонним многоугольником; '
       'отличие его площади от площади круга составляет около 0,010%. При объединении добавлен внутренний '
       'нахлёст 0,5 мм. Цилиндры не выходят за исходные пределы по y и z; их продолжение по x '
       'является заданной моделью рук.\n\n'
       f'Итоговая поверхность замкнута, имеет одну связную компоненту и положительный объём '
       f'{number(g["volume_ml"],6)} мл. Самопересекающихся граней не обнаружено. '
       'Эти проверки подтверждают целостность поверхности и полноту выбранных сечений; '
       'точность реальной формы рук ими не установлена.')
    code('display(table(join_table(g)))')
    md('## 3. Контакты и продольные сборки\n\n'
       'По уточнению автора в монтаже 1×4 единственный токовый контакт каждой руки перенесён '
       'с передней поверхности наверх, поворотом на 90°. В монтаже 2×2 добавлен диаметрально '
       'противоположный нижний контакт. Четыре измерительных контакта 1×4 расположены через 90°; '
       'передний и задний измерительные контакты 2×2 пока заданы как сценарий. '
       'В полной электродной модели (CEM) контакты одной электрической группы '
       'должны иметь общий потенциал.\n\n'
       'Продольные расстояния контактов рук сохранены: 20 мм для измерительных и 120 мм для токовых '
       'от прежней плоскости обрезки КТ. Начало отсчёта не перенесено на заглублённое соединение. '
       'Эта плоскость является модельным ориентиром, а не измеренной анатомической точкой. '
       'Номинальные центры проверены на конечной поверхности с допуском 0,02 мм.\n\n'
       'В системе координат LPS положительные оси x, y и z направлены соответственно влево, '
       'назад и к голове относительно испытуемого; их начало перенесено из КТ. '
       'Боковые сборки теперь ориентированы по z LPS — вдоль тела. '
       'Для всех размеров сохранён один иллюстративный центр y = 0, z = −210 мм; '
       'его фактическая локализация по эксперименту пока неизвестна. Центры A, M, N, B '
       'задаются на уровнях −L/2, −L/4, L/4, L/2 относительно общего центра, '
       'где L — номинальное расстояние между токовыми центрами в миллиметрах. Они проецируются '
       'на правую поверхность лучами вдоль +x LPS. Отрицательная сторона x соответствует правой стороне человека. '
       'Это геометрический пример, используемый для проверки направления; тройка прежних выбранных точек '
       'не используется вместо полного поиска.\n\n'
       '**Рисунок 1.** Исправленная наружная поверхность и центры контактов. Монтаж 2×2 и другие размеры '
       'можно включить через легенду. Маркеры изображают центры, а не дискретизированные FEM-площадки. '
       'Для новой CEM-постановки принят номинальный диаметр контакта 5 мм и площадь 19,635 мм²; '
       'фактическая проводящая площадь Scorpion ещё не установлена.')
    code('show(scene(OUT))')
    md('## 4. КТ-сечение по оси сборок\n\n'
       '**Рисунок 2.** Продольное КТ-сечение y LPS = 0 мм: полный том и сборка 140 мм. '
       'Все размеры имеют один центр; меньшие сборки занимают внутренние участки той же оси. '
       'Метки A и B обозначают токовые центры, M и N — измерительные. Показаны маски лёгких, костей, '
       'сердца и жира; отсутствие ткани в данном сечении не означает отсутствия её маски в объёме. '
       'Использована интерполяция ближайшего соседа. Геометрическая поверхность и КТ могут различаться '
       'из-за обработки наружного контура; это расхождение ещё требует оценки. '
       'Голубая линия отмечает первое пересечение лучей вдоль +x LPS с маской правого лёгкого; '
       'пунктиром показаны расстояния от наружной поверхности до этой линии. '
       'Направление выбрано для сопоставления с принятой проекцией электродов. '
       'Оно не является локальной нормалью к коже или кратчайшим расстоянием до лёгкого.')
    code("display(Image(filename=str(OUT/'ct_longitudinal_sections.png')))")
    md('**Рисунок 3.** Профиль анатомического расстояния до маски лёгкого вдоль общей оси сборок. '
       'Смещение s отсчитывается вдоль z LPS от иллюстративного центра на уровне −210 мм: '
       'от −70 до +70 мм с шагом 1 мм. Положительное направление s соответствует направлению к голове. '
       'A, M, N и B — центры контактов сборки 140 мм. Расстояние d вычислено от наружной STL-поверхности '
       'до первого ненулевого вокселя маски лёгкого вдоль +x LPS; луч заканчивается на плоскости x LPS = 0, '
       'чтобы не подменить правое лёгкое левым. Шаг вдоль луча составляет 0,5 мм; '
       'ошибка такого поиска границы не превышает одного шага относительно границы дискретной маски. '
       'Погрешности сегментации, обработки контура тела и восстановления положения этим пределом не охвачены. '
       'Если лёгкое на луче отсутствует или поле КТ неполно, расстояние не назначается.\n\n'
       'Профиль построен независимо от импеданса и сопротивлений. Он показывает пространственную '
       'неоднородность анатомии при данном примере положения. Электрически эквивалентная толщина h '
       'двуслойной модели пока не определена через этот профиль: способ такого сведения требуется '
       'проверить по индивидуальной конечноэлементной модели.')
    code("display(Image(filename=str(OUT/'lung_depth_along_montage.png')))")
    md('## 5. Полный перерасчёт положения и сопротивлений\n\n'
       'Следующий этап — поиск общего центра по всей допустимой правой поверхности грудной клетки '
       'с проверкой размещения всех пяти размеров. Направление вдоль тела задаёт номинальную ориентацию; '
       'ошибку установки необходимо исследовать отдельно. Для каждого положения сопротивления '
       'мягкотканного фона, лёгкого и жира следует оптимизировать совместно либо вложенным поиском. '
       'Поиск положения при одной фиксированной тройке сопротивлений не заменяет эту обратную задачу. '
       'Цель — карта невязки и допустимых тканевых параметров по всей области, с оценкой '
       'геометрической и электрической неоднозначности. Это план следующего расчёта, а не выполненный результат.\n\n'
       'Новая поверхность требует новой объёмной сетки, повторного отображения четырёх тканевых масок, '
       'контактных площадок CEM и всех пассивных электродов соответствующего монтажа. '
       'Матрицы и оценки 8 октября повторно использовать нельзя. Необходимо проверить '
       'сходимость по сетке и интегрированию, взаимность и производные, а затем повторить инверсию. '
       'Для двуслойного поиска в 33.21 приняты физиологически мотивированные литературные интервалы '
       'при 100 кГц. Их перенос на мягкотканный фон, лёгкое и жир FEM требует отдельного обоснования '
       'для каждого компартмента; прежние технические пределы 0,1–100 Ом·м не задают физиологический диапазон.\n\n'
       'До этого пересчёта индивидуальные ρ₁, ρ₂ и ρж не выбраны для коррекции ТТРКГ. '
       'Влияние площади контактов на сигнал и чувствительность каждого компартмента следует '
       'сравнивать после проверки геометрии и независимо для токовых и измерительных площадок. '
       'Расчёт УО и ФВ здесь не выполнялся. Хэши поверхности, кода, КТ и масок сохранены '
       'в локальном паспорте; первичные массивы не включены в читательский экспорт.')
    md('## 6. Ручное размещение электродов ТЕПС\n\n'
       'Для Георгия подготовлен [интерактивный инструмент размещения электродов](../../MATLAB_TRKG4_real_subjects/output/exploratory/georg_tepc_manual_20261009/georg_tepc_picker.html) '
       'на основе того же инструмента, который использовался для Ника. '
       'Он показывает исправленную наружную поверхность и четыре независимых слоя: сердце, лёгкие, '
       'кости и жир. Руки имеют цилиндрическую форму по принятому допущению; их продолжение за пределы '
       'поля КТ не является сегментированной анатомией.\n\n'
       'Последовательность работы: выбрать ТЕПС 2–7 и соответствующую запись 012–017; указать канал, '
       'если он известен; выбрать на поверхности четыре центра I+, V+, V−, I−; указать автора '
       'реконструкции, её основание и неопределённость; сохранить JSON и CSV. '
       'I обозначает токовые, V — измерительные электроды. Выбор схемы не задаёт координаты '
       'автоматически. При переходе к следующей записи точки очищаются, а ракурс сохраняется. '
       'Альтернативное положение можно сохранить под отдельным именем.\n\n'
       'В JSON сохраняются связь с записью, номер канала либо его неизвестность, координаты LPS '
       'в миллиметрах, контрольные суммы исходной геометрии и масок и статус ручной реконструкции. '
       'CSV содержит таблицу четырёх центров; для воспроизводимости его следует сохранять вместе '
       'с JSON. Знаки контактов необходимо проверить по подключению. Экспорт использует совместимую '
       'с MATLAB схему четырёх поверхностных центров, а назначение ТЕПС указано в паспорте измерения. '
       'Центры выбираются на облегчённой поверхности и перед FEM должны быть повторно спроецированы '
       'на исходную расчётную поверхность. Площади контактов и контактное сопротивление здесь не задаются. '
       'Координаты Ника не перенесены; фактическая установка Георгия ещё не размечена.')
    export(GEOMETRY,cells,out)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT)
    build(parser.parse_args().out)
