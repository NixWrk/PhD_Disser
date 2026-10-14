"""Read-only notebook for the corrected preparation and repeated joint FEM fit.

No numerical solver or model preparation is invoked by this report.
"""
from pathlib import Path
import argparse
import hashlib
import html
import json
import numpy as np

FOLDER=Path(__file__).resolve().parent
PIPE=FOLDER.parents[1]/'MATLAB_TRKG4_real_subjects'
PREPARATION=PIPE/'output/exploratory/georg_quality_rebuild_20261013'
DEFAULT=PREPARATION/'contact_corrected_v2'
STEM='33.29_Повторный_FEM_подбор_с_контролем_сетки_Георгия_Scorpion_v2'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def number(value,places=3):
    return f'{value:.{places}f}'.replace('.',',')


def scientific_number(value):
    mantissa,separator,exponent=f'{value:.3g}'.partition('e')
    if not separator:return mantissa.replace('.',',')
    superscript=str(int(exponent)).translate(str.maketrans('-0123456789','⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))
    return mantissa.replace('.',',')+'·10'+superscript


def table(title,columns,rows):
    from IPython.display import HTML
    header=''.join('<th>'+html.escape(str(value))+'</th>' for value in columns)
    body=''.join('<tr>'+''.join('<td>'+html.escape(str(value))+'</td>' for value in row)+'</tr>' for row in rows)
    return HTML('<details><summary>'+html.escape(title)+'</summary><div class="scientific-table">'
                '<table><thead><tr>'+header+'</tr></thead><tbody>'+body+'</tbody></table></div></details>')


def verify_historical_source(source,preparation):
    path=Path(source['path'])
    if not path.is_absolute():path=PIPE/path
    if path.exists() and sha(path)==source['sha256']:return
    try:relative=path.resolve().relative_to(PIPE)
    except ValueError:raise ValueError('External numerical evidence changed')
    candidates=[preparation/'source_snapshots/quality_run_sources'/relative,
                preparation/'contact_corrected_v2/source_snapshots'/relative]
    if not any(candidate.exists() and sha(candidate)==source['sha256'] for candidate in candidates):
        raise ValueError('Archived numerical source is unavailable: '+path.name)


def load(work=DEFAULT):
    work=Path(work).resolve()
    selector=work/'report_selection.local.json'
    if selector.exists():
        selected_report=read(selector)
        chosen=(work/selected_report['final_work_directory']).resolve()
        if selected_report['schema']!='georg_quality_completed_report_selection_v1' or not chosen.is_relative_to(work):
            raise ValueError('Invalid completed-report selection')
        for filename,key in [('selected_candidate.local.json','selected_candidate_sha256'),
                             ('numerical_audit_selected_v1.local.json','numerical_audit_sha256'),
                             ('contact_discretization_selected.local.json','contact_comparison_sha256')]:
            if sha(chosen/filename)!=selected_report[key]:raise ValueError('Final selected report evidence changed')
        work=chosen
    state=read(work/'run_state.local.json');cfg=read(work/'run_config.local.json')
    if state['status']!='complete_with_explicit_numerical_results':
        raise ValueError('The complete repeated fit and planned selected-candidate controls are required')
    initial=read(work/cfg['initial_audit_file'])
    final=read(work/'numerical_audit_selected_v1.local.json')
    historical=read(PREPARATION/'numerical_audit_author_v2.local.json')
    for source in historical['sources']:verify_historical_source(source,PREPARATION)
    for audit in (initial,final):
        for source in audit['sources']:
            verify_historical_source(source,PREPARATION)
    selected=read(work/'selected_candidate.local.json')
    if sha(work/'selected_candidate.local.json')!=final['selected_candidate_sha256']:
        raise ValueError('The selected candidate changed after its controls')
    contact_initial=read(work/cfg.get('initial_contact_file','contact_discretization_initial.local.json'))
    for source in contact_initial['sources']:verify_historical_source(source,PREPARATION)
    contact_final=read(work/'contact_discretization_selected.local.json')
    for field in ('parameters','pose'):
        if contact_final[field]!=selected['best'][field]:
            raise ValueError('Contact comparison belongs to another selected candidate: '+field)
    coverage=read(work/'contact_coverage_selected.local.json')
    if coverage['coarse_forward_sha256']!=sha(work/'coarse_forward_selected_v1.local.json'):
        raise ValueError('Contact refinement coverage belongs to another candidate')
    for source in contact_final['sources']:
        verify_historical_source(source,PREPARATION)
    models=[(work/p).resolve() for p in cfg['control_models']]
    plan=read(models[-1]/'plan.local.json')
    masks={}
    for name in ('lung','bone','heart','fat'):
        matched=[r for r in plan['sources'] if Path(r['path']).name==name+'.nii.gz']
        if len(matched)!=1:raise ValueError('Ambiguous material mask: '+name)
        verify_historical_source(matched[0],PREPARATION)
        masks[name]=Path(matched[0]['path'])
    input_directory=(work/cfg['input_directory']).resolve()
    run_directory=input_directory/('joint_own_drive_'+cfg['run_label'])
    if 'search' not in selected:
        if not selected.get('search_stopped_for_time'):
            raise ValueError('Missing search termination evidence')
        completed=[read(p) for p in run_directory.glob('evaluation_*.local.json')]
        selected['search']=dict(evaluations=max(r['evaluation'] for r in completed),
            sweeps=None,final_step=None,resolution_reached=False,budget_reached=False,
            reconstructed_from_completed_checkpoints=True)
    distribution=read(input_directory.parent/'fraction_distribution.local.json')
    for source in distribution['sources']:
        if sha(source['path'])!=source['sha256']:raise ValueError('Tissue distribution evidence changed')
    extra_root=input_directory.parent/'selected_refinement40_v1'
    extra=None
    if (extra_root/'preparation_failed.local.json').exists():
        extra_candidate=read(extra_root/'candidate.local.json')
        extra_candidate=extra_candidate.get('best',extra_candidate)
        for field in ('parameters','pose'):
            if extra_candidate[field]!=selected['best'][field]:
                raise ValueError('Additional refinement tested a different candidate')
        extra=read(extra_root/'preparation_failed.local.json')
        if extra['passed'] is not False or extra['limits_changed'] is not False:
            raise ValueError('Unexpected additional refinement outcome')
    resource_file=input_directory.parent/'selected_refinement_resource_screen40.local.json'
    resource=read(resource_file) if resource_file.exists() else None
    if resource is not None:
        if (resource['candidate_sha256']!=sha(work/'selected_candidate.local.json') or
            resource['parent_mesh_sha256']!=sha(models[-1]/'mesh.mat')):
            raise ValueError('Additional resource screen belongs to another candidate or mesh')
    shared=read(PREPARATION/'shared_selection_result.local.json')
    if (shared.get('selection_state')!='selected_geometry_only' or
        shared.get('quantitative_inversion_allowed') is not False or
        (PIPE/shared['fem_directory']).resolve()!=models[-1]):
        raise ValueError('Common geometry selection does not match the reported model')
    return dict(work=work,preparation=PREPARATION,state=state,initial=initial,historical=historical,
        final=final,selected=selected,best=selected['best'],
        config=cfg,input_directory=input_directory,models=models,
        distribution=distribution,masks=masks,additional_refinement=extra,resource_screen=resource,
        run_directory=run_directory,
        selected_forwards=[read(work/(p+'_selected_v1.local.json')) for p in
                           ('coarse_forward','fine_inherited','fine_remapped')],
        contact_initial=contact_initial,contact_final=contact_final,coverage=coverage,
        coarse=read(models[0]/'mesh_quality.local.json'),
        fine=read(models[-1]/'mesh_quality.local.json'),
        contour=read(PREPARATION/'geometry_v5/contour_qc.local.json'),
        surface=read(PREPARATION/'mesh_v9/surface_qc.local.json'),
        inputs=read(input_directory/'inputs.local.json'))


def input_table(data):
    return table('Таблица 1. Происхождение входов и допущений повторного расчёта',
        ['Вход','Источник','Статус и ограничение'],[
        ['КТ','Статическая серия 301; 512 × 512 × 280; шаг 0,879 × 0,879 × 1,25 мм',
         'Глубокий вдох с задержкой подтверждён автором; тот же том используется для обоих дыхательных состояний'],
        ['Лёгкие, кости, сердце, жир','Сохранённые автоматические маски TotalSegmentator 2.18.0',
         'Совпадение воксельных сеток проверено; независимая анатомическая точность не установлена'],
        ['Наружный контур','Порог КТ −450 HU, морфологическое открытие, компонента тела и заполнение полостей; затем описанная ниже регуляризация',
         'Рассчитанная геометрия; ограниченная проверка относительно исходной КТ'],
        ['Импедансы','24 уровня из записей боковых сборок Scorpion v2 на выбранных задержках',
         'Измеренные сигналы в Ом; уровни — результат обработки кандидатных дыхательных интервалов'],
        ['Роли каналов и общий центр монтажа','Последовательность подключений и общий центр подтверждены автором',
         'Не подбираются роли каналов; численные координаты центра восстанавливаются'],
        ['Ориентировочная точка центра','Указана автором по изображению КТ',
         'Начальное приближение; погрешность положения не измерена'],
        ['Обхват обеих рук','37 см на уровне токовых контактов, сообщение автора',
         'Цилиндрическая форма, положение соединения и мягкотканное заполнение рук — допущения'],
        ['Площадки','Номинальные диски диаметром 5 мм, стандарт проекта',
         'Фактический размер и удельный контактный импеданс отдельно не измерены'],
        ['Материалы','Литературный профиль при подтверждённых автором 100 кГц',
         'Фон, лёгкое и жир подбираются; кости и вся маска сердца фиксированы'],
        ['Сопоставимость последовательных записей','Условие объединённого расчёта',
         'Изменения состояния и контакта не измерены отдельной моделью неопределённости']])


def mesh_table(data):
    rows=[]
    for name,mesh in [('Базовая',data['coarse']),('Локально сгущённая',data['fine'])]:
        rows.append([name,f"{mesh['nodes']:,}".replace(',',' '),f"{mesh['tetrahedra']:,}".replace(',',' '),
            number(mesh['mean_ratio_minimum'],6),number(mesh['edge_condition_maximum'],1),
            'Да' if mesh['passed'] else 'Нет'])
    return table('Таблица 3. Качество формы тетраэдров',
        ['Сетка','Узлы','Тетраэдры','Минимальный показатель формы','Максимальная обусловленность рёбер','Контроль пройден'],rows)


def repair_table(data):
    return table('Таблица 2. Исправления подготовки: причина, действие и проверка',
        ['Проблема','Что изменено','Что проверено; граница вывода'],[
        ['Пустое сечение могло выделить фон как тело',
         'Выбор компоненты тела допускает только положительные метки; пустое сечение остаётся пустым',
         'Регрессионные тесты пройдены. На исходной КТ Георгия маска не изменилась'],
        ['Неоднозначная топология извлечённого контура',
         'Одна итерация шестисвязного 3D-замыкания; изоповерхность 0,49; объединение до упрощения',
         'Добавлены 3318 вокселей; замкнутая связная поверхность без найденных самопересечений'],
        ['Почти касательное присоединение цилиндров рук',
         'Полное покрытие сечения туловищем; зазор от искусственного верхнего среза 2 мм',
         'Проверены пересечение и целостность поверхности; анатомическое положение соединения остаётся допущением'],
        ['Физическая площадка и электрическая группа смешивались',
         'Сохранены 12 постоянных площадок рук; группирование отдельно для каждого монтажа; крест содержит обе оси',
         'Один общий CEM-расчёт схемы; неиспользуемые площадки плавающие. Продолжение монтажа рук во время крестов — допущение'],
        ['Грубая граница выбора контактов исключала допустимые пробные положения',
         'Опора положения отделена от сетки выбора площадок; новый подбор использует более подробную опору',
         'Пять из шести прежних отказов исчезли при неизменных координатах и допусках; один отказ сохранился'],
        ['Сгущение одновременно меняло пространство решения и проводимость',
         'Два варианта одной сгущённой сетки: точное наследование коэффициентов и новый расчёт тканевых долей',
         'Общая граница и контакты сохранены; эффекты сравниваются раздельно при фиксированных параметрах'],
        ['Сохранённый результат мог читаться с изменёнными исходниками или кэшем',
         'Привязка к контрольным суммам исполненного кода, планов, сеток и правил; свежий пересчёт лучшей точки',
         'Проверяется происхождение каждой контрольной пары; прежние расчёты и их исходники сохранены отдельно']])


def contour_image(data):
    import base64
    from IPython.display import HTML
    path=data['preparation']/'geometry_v5/ct_topology_regularization.local.png'
    return HTML('<img alt="Сравнение исходного и исправленного контура на КТ" '
                'style="width:100%;max-width:1200px" src="data:image/png;base64,'
                +base64.b64encode(path.read_bytes()).decode('ascii')+'">')


def inheritance_table(data):
    q=data['fine']
    return table('Таблица 4. Контроль вложенного сгущения и общей геометрии',
        ['Проверка','Сохранённый результат'],[
        ['Все объёмы положительны','Да' if q['all_positive'] else 'Нет'],
        ['Исходные узлы неизменны','Да' if q['original_nodes_unchanged'] else 'Нет'],
        ['Согласованная наружная граница','Да' if q['conforming_boundary_verified'] else 'Нет'],
        ['Объём обеих сеток, мл',number(q['volume_ml'],6)],
        ['Максимальная относительная разность объёма родителя и суммы его потомков',scientific_number(q['parent_volume_maximum_relative_difference'])],
        ['Максимальная относительная разность площади родительской грани и суммы её потомков',scientific_number(q['fine1_boundary_identity']['maximum_relative_area_difference'])],
        ['Максимальное смещение центроида родительской грани, мм',scientific_number(q['fine1_boundary_identity']['maximum_centroid_difference_mm'])],
        ['Нарушения порога формы',q['failing_mean_ratio']],
        ['Нарушения порога обусловленности рёбер',q['failing_edge_condition']],
        ['Глобальная сходимость FEM','Этой проверкой не установлена']])


def shape_quality_figure(data,include_js=False):
    import plotly.graph_objects as go
    fig=go.Figure()
    for label,key,color in [('Базовая','coarse','#326ba8'),('Сгущённая','fine','#df8b26')]:
        q=data[key]
        fig.add_trace(go.Scatter(x=['Минимум','1 %','5 %','50 %','95 %'],
            y=[q['mean_ratio_minimum'],*q['mean_ratio_quantiles']],name=label,
            mode='markers',marker=dict(color=color,size=9)))
    fig.add_trace(go.Scatter(x=['Минимум','1 %','5 %','50 %','95 %'],y=[.01]*5,
        name='Минимально допустимый показатель',mode='lines',line=dict(color='#333333',dash='dash')))
    fig.update_layout(xaxis_title='Характеристика сохранённого распределения',
        yaxis_title='Показатель формы тетраэдра',
        yaxis=dict(type='log',tickvals=[.01,.1,1],ticktext=['0,01','0,1','1']))
    return figure_html(fig,include_js)


def edge_quality_figure(data,include_js=False):
    import plotly.graph_objects as go
    fig=go.Figure()
    for label,key,color in [('Базовая','coarse','#326ba8'),('Сгущённая','fine','#df8b26')]:
        q=data[key]
        fig.add_trace(go.Scatter(x=['50 %','95 %','99 %','Максимум'],
            y=[*q['edge_condition_quantiles'],q['edge_condition_maximum']],name=label,
            mode='markers',marker=dict(color=color,size=9)))
    fig.add_trace(go.Scatter(x=['50 %','95 %','99 %','Максимум'],y=[10000]*4,
        name='Максимально допустимая обусловленность',mode='lines',line=dict(color='#333333',dash='dash')))
    fig.update_layout(xaxis_title='Характеристика сохранённого распределения',
        yaxis_title='Обусловленность матрицы рёбер',
        yaxis=dict(type='log',tickvals=[1,10,100,1000,10000],ticktext=['1','10','100','1000','10 000']))
    return figure_html(fig,include_js)


def residual_table(data):
    import pandas as pd
    o=pd.read_csv(data['input_directory']/'observations.local.csv')
    o=o[o.state!='quiet_breathing'].reset_index(drop=True)
    predicted=np.asarray(data['best']['predicted_Z_ohm']);rows=[]
    for i,row in o.iterrows():
        rows.append([row.record_id.rsplit('-',1)[-1],row.size_mm,int(row.channel),
            'Продольная' if row.axis=='longitudinal' else 'Поперечная',
            'Вдох' if row.state=='inhale' else 'Выдох',number(row.Z_ohm),number(predicted[i]),
            number(predicted[i]-row.Z_ohm),number(100*(predicted[i]-row.Z_ohm)/row.Z_ohm,1)])
    return table('Таблица 11. Все 24 невязки повторного подбора',
        ['Запись','Размер, мм','Канал','Ось','Состояние','Измерено, Ом','Рассчитано, Ом','Модель минус измерение, Ом','Относительная невязка, %'],rows)


def contact_figure(data,include_js=False):
    import plotly.graph_objects as go
    c=data['contact_final'];labels=observation_labels(c);fig=go.Figure()
    fig.add_trace(go.Bar(x=labels,y=c['signed_ratio'],name='Изменение площадок / допуск',marker_color='#378a78'))
    for value,label in [(-1,'Нижняя граница допуска'),(1,'Верхняя граница допуска')]:
        fig.add_trace(go.Scatter(x=labels,y=[value]*len(labels),mode='lines',name=label,
            line=dict(color='#333333',dash='dash')))
    fig.update_layout(xaxis=dict(title='Запись, ось сборки и дыхательное состояние',tickangle=-50),
        yaxis_title='(Z новых площадок − Z наследуемых площадок) / допуск',height=710,
        margin=dict(l=90,r=35,t=80,b=180))
    return figure_html(fig,include_js)


def sensitivity_figure(data,include_js=False):
    import plotly.graph_objects as go
    jac=np.asarray(data['best']['jacobian_logrho'])
    if jac.shape!=(24,6):raise ValueError('Expected the saved 24 by 6 resistivity Jacobian')
    fig=go.Figure(go.Heatmap(z=jac,x=['ρ₁ · вдох','ρ₂ · вдох','ρ₃ · вдох',
        'ρ₁ · выдох','ρ₂ · выдох','ρ₃ · выдох'],y=observation_labels(data['final']),
        colorscale='RdBu_r',zmid=0,colorbar=dict(title='Ом'),
        hovertemplate='%{y}<br>%{x}<br>∂Z/∂lnρ=%{z:.3f} Ом<extra></extra>'))
    fig.update_layout(xaxis_title='Свободное сопротивление и дыхательное состояние',
        yaxis_title='Запись, ось и состояние',height=880,margin=dict(l=235,r=100,t=50,b=100))
    result=figure_html(fig,include_js)
    return result


def sensitivity_table(data):
    singular=np.asarray(data['best']['singular_values']);rows=[]
    for i,value in enumerate(singular):rows.append([i+1,number(value,6)])
    return table('Таблица 14. Сингулярные числа якобиана сопротивлений при фиксированном положении',
        ['Номер','Сингулярное число, Ом'],rows)


def breathing_contrast_figure(data,include_js=False):
    import plotly.graph_objects as go
    a=data['final'];best=data['best'];labels=[];measured=[];predicted=[]
    target=np.asarray(best['target_Z_ohm']);model=np.asarray(best['predicted_Z_ohm'])
    for i,state in enumerate(a['states']):
        if state!='inhale':continue
        matches=[j for j in range(len(a['states'])) if a['states'][j]=='exhale'
            and a['record_ids'][j]==a['record_ids'][i] and a['axes'][j]==a['axes'][i]]
        if len(matches)!=1:raise ValueError('Ambiguous paired respiratory contrast')
        j=matches[0];labels.append(a['record_ids'][i].rsplit('-',1)[-1]+' · '+
            ('продольная' if a['axes'][i]=='longitudinal' else 'поперечная'))
        measured.append(target[j]-target[i]);predicted.append(model[j]-model[i])
    fig=go.Figure()
    for name,value,symbol,color in [('Эксперимент',measured,'circle','#326ba8'),
                                   ('Модель',predicted,'square','#df8b26')]:
        fig.add_trace(go.Scatter(x=labels,y=value,mode='markers',name=name,
            marker=dict(size=10,symbol=symbol,color=color)))
    fig.add_trace(go.Scatter(x=labels,y=[0]*len(labels),mode='lines',name='Нет изменения',
        line=dict(color='#333333',dash='dash')))
    fig.update_layout(xaxis=dict(title='Запись и ось сборки',tickangle=-45),
        yaxis_title='Z выдоха − Z вдоха, Ом',height=650,margin=dict(l=90,r=35,t=80,b=170))
    return figure_html(fig,include_js)


def mesh_section_figure(data,include_js=False,fine=False):
    import sys
    from scipy.io import loadmat
    import plotly.graph_objects as go
    sys.path.insert(0,str(PIPE/'tools'))
    from georg_quality_mesh_section import section_lines
    mesh=loadmat(data['models'][-1 if fine else 0]/'mesh.mat',variable_names=['nodes_mm','elems'])
    pose=data['best']['pose']
    lines=section_lines(mesh['nodes_mm'],mesh['elems'].astype(np.int64)-1,
        np.asarray(pose['centre_lps_mm']),pose['primary_unit_xyz'],-np.asarray(pose['normal_unit_xyz']))
    fig=go.Figure(go.Scattergl(x=lines['x'],y=lines['y'],mode='lines',
        line=dict(width=.6,color='#326ba8'),name='Сечения тетраэдров',hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[0],y=[0],mode='markers',name='Центр сборок',
        marker=dict(color='#bd3535',size=10,symbol='diamond')))
    fig.update_layout(xaxis=dict(title='Вдоль основной оси от центра, мм',range=[-80,80],constrain='domain'),
        yaxis=dict(title='По внутренней нормали от центра, мм',range=[55,-5],scaleanchor='x',scaleratio=1,constrain='domain'),
        height=650,margin=dict(l=95,r=35,t=80,b=85))
    return figure_html(fig,include_js)


def coarse_section_figure(data,include_js=False):
    return mesh_section_figure(data,include_js,False)


def fine_section_figure(data,include_js=False):
    return mesh_section_figure(data,include_js,True)


def invalid_contact_table(data):
    rows=[]
    for path in sorted(data['run_directory'].glob('evaluation_*.local.json')):
        r=read(path)
        if r['status']!='invalid_geometry':continue
        contacts=r.get('patch_qc',{}).get('contacts',[])
        maximum=lambda key,scale=1: (number(scale*max(c[key] for c in contacts),3)
                                     if contacts else 'Не рассчитано')
        rows.append([r['evaluation'],*map(lambda v:number(v,2),r['parameters'][-3:]),
            maximum('area_error',100),maximum('centroid_offset_mm'),maximum('extent_mm')])
    return table('Таблица 15. Отклонённые пробные положения и показатели площадок',
        ['Номер оценки','Первое смещение, мм','Второе смещение, мм','Угол, °',
         'Максимальная ошибка площади, %','Максимальное смещение центроида, мм',
         'Максимальная протяжённость от центра, мм'],rows)


def starting_candidates_table(data):
    plan=read(data['run_directory']/'plan.local.json');rows=[]
    for i,x in enumerate(plan['starts'],1):
        rows.append([i,'; '.join(number(v,3) for v in np.exp(x[:3])),
            '; '.join(number(v,3) for v in np.exp(x[3:6])),
            *[number(v,2) for v in x[6:]]])
    return table('Таблица 16. Фактически использованные начальные кандидаты',
        ['Кандидат','ρ₁; ρ₂; ρ₃ на вдохе, Ом·м','ρ₁; ρ₂; ρ₃ на выдохе, Ом·м',
         'Первое смещение, мм','Второе смещение, мм','Угол, °'],rows)


def coverage_table(data):
    import pandas as pd
    observations=pd.read_csv(data['input_directory']/'observations.local.csv')
    patches={r['record_id']:r for r in data['selected_forwards'][0]['patches']}
    arms=np.asarray(read(data['preparation']/'geometry_v5/fem_contacts.local.json')['arm_physical_centres_lps_mm'])
    rows=[]
    for contact in data['coverage']['contacts']:
        if contact['all_faces_refined']:continue
        xyz=patches[contact['record_id']]['xyz'][contact['physical_contact']-1]
        labels=[]
        record=observations[(observations.record_id==contact['record_id'])&(observations.state=='inhale')]
        for _,observation in record.iterrows():
            montage=next(m for m in data['best']['pose']['electrodes']
                         if m['size_mm']==observation.size_mm and m['axis']==observation.axis)
            for index,centre in enumerate(montage['centres_lps_mm']):
                if np.linalg.norm(np.asarray(centre)-xyz)<1e-7:
                    labels.append(f'{int(observation.size_mm)} мм · '+
                        ('продольная' if observation.axis=='longitudinal' else 'поперечная')+
                        ' · '+['I+','U+','U−','I−'][index])
        if not labels and np.min(np.linalg.norm(arms-xyz,axis=1))>=1e-7:
            raise ValueError('Contact coverage coordinate has no physical montage match')
        label='; '.join(labels) if labels else 'Постоянная площадка на руке'
        rows.append([contact['record_id'].rsplit('-',1)[-1],label,contact['physical_contact'],
            *[number(v,2) for v in xyz],number(contact['area_mm2'],3),
            number(100*contact['refined_boundary_area_fraction'],2)])
    return table('Таблица 17. Контактные площадки с неполным покрытием локальным сгущением',
        ['Запись','Отведение и контакт','Номер физической площадки в схеме','x LPS, мм','y LPS, мм','z LPS, мм',
         'Площадь, мм²','Площадь на уточнённых граничных гранях, %'],rows)


def search_table(data):
    search=data['selected']['search'];n=search['evaluations']
    rows=[read(p) for p in sorted(data['run_directory'].glob('evaluation_*.local.json'))]
    counted=[r for r in rows if r['evaluation']<=n]
    best=data['best'];step=search['final_step']
    show_step=lambda start,end,places: ('Не сохранены при остановке по времени' if step is None
        else '; '.join(number(v,places) for v in step[start:end]))
    return table('Таблица 10. Выполненный объём поиска и критерий остановки',
        ['Характеристика','Результат'],[
        ['Исходные точки',len(data['config']['start_indices'])],
        ['Завершённые оценки поиска, включая отказы',n],
        ['Завершённые FEM-кандидаты поиска',sum(r['status']=='numerical_candidate_not_validated' for r in counted)],
        ['Отказы дискретного представления геометрии и площадок',sum(r['status']=='invalid_geometry' for r in counted)],
        ['Свежие проверки после поиска',sum(r['evaluation']>n for r in rows)],
        ['Начатые обходы направлений, включая неполные',search['sweeps'] if search['sweeps'] is not None else 'Не сохранены'],
        ['Достигнуто целевое разрешение','Да' if search['resolution_reached'] else 'Нет'],
        ['Последние шаги логарифмов ρ₁, ρ₂, ρ₃: вдох',show_step(0,3,4)],
        ['Последние шаги логарифмов ρ₁, ρ₂, ρ₃: выдох',show_step(3,6,4)],
        ['Последние шаги двух смещений центра, мм',show_step(6,8,2)],
        ['Последний шаг угла, °',show_step(8,9,2)],
        ['Смещения выбранного центра в касательной плоскости, мм','; '.join(number(v,2) for v in best['pose']['offsets_uv_mm'])],
        ['Угол выбранного кандидата, °',number(best['pose']['angle_deg'],2)],
        ['Глобальный минимум и единственность','Не доказаны']])


def observation_labels(audit):
    return [record.rsplit('-',1)[-1]+(' · продольная' if axis=='longitudinal' else ' · поперечная')
            +(' · вдох' if state=='inhale' else ' · выдох')
            for record,axis,state in zip(audit['record_ids'],audit['axes'],audit['states'],strict=True)]


def numerical_table(data):
    rows=[]
    for label,audit in [('Первоначальные площадки',data['historical']),
                        ('Исправленные площадки: начало',data['initial']),('Исправленные площадки: найденная точка',data['final'])]:
        for name,key in [('Только сетка','spatial'),('Только тканевые доли','fractions'),('Суммарное изменение','total')]:
            check=audit[key];rows.append([label,name,number(max(check['difference_ohm'])),
                check['failed_values'],'Да' if check['passed'] else 'Нет'])
    return table('Таблица 6. Раздельные численные проверки импеданса',
        ['Условия','Изменяемая часть модели','Максимум |ΔZ|, Ом','Превышения среди 24 строк','Допуск выдержан'],rows)


def rho_table(data):
    bounds=data['inputs']['search_bounds_ohm_m'];best=data['best'];rows=[]
    for i,name in enumerate(['Остаточный мягкотканный фон, ρ₁','Лёгкое, ρ₂','Жир, ρ₃']):
        values=[best['rho_inhale_ohm_m'][i],best['rho_exhale_ohm_m'][i]]
        edge=any(abs(value-bounds[i][j])<1e-7 for value in values for j in (0,1))
        rows.append([name,number(values[0]),number(values[1]),
            number(bounds[i][0])+'–'+number(bounds[i][1]),'Да' if edge else 'Нет'])
    return table('Таблица 9. Предварительные эффективные удельные сопротивления',
        ['Материал','Вдох, Ом·м','Выдох, Ом·м','Область поиска, Ом·м','Кандидат на границе'],rows)


def source_table(data):
    own=np.asarray(data['best']['own_transfer_ohm']);counter=np.asarray(data['best']['counter_transfer_ohm'])
    target=np.asarray(data['best']['target_Z_ohm']);rows=[]
    for name,value in [('Собственный источник: постановка подбора',own),
        ('Равные токи, совпадающая фаза',own+counter),('Равные токи, противоположные фазы',own-counter),
        ('Равные токи, сдвиг фаз 90° и модуль',np.hypot(own,counter))]:
        rows.append([name,number(np.sqrt(np.mean((value-target)**2)))])
    return table('Таблица 13. Диагностические сценарии возбуждения без нового подбора',
        ['Сценарий при тех же сопротивлениях и положении','Среднеквадратическая невязка, Ом'],rows)


def contact_table(data):
    first=data['contact_initial']['summary'];last=data['contact_final']
    return table('Таблица 7. Изменение от дискретизации контактных площадок',
        ['Условия','Максимум |ΔZ|, Ом','Превышения среди 24 уровней'],[
            ['Старые площадки → более подробная опора',number(first['maximum_absolute_delta_ohm']),first['failed_values']],
            ['В найденной точке: следующее уточнение площадок',number(max(last['difference_ohm'])),last['failed_values']]])


def control_rows_table(data):
    a=data['final'];c=data['contact_final'];rows=[]
    if (a['record_ids'],a['axes'],a['states'])!=(c['record_ids'],c['axes'],c['states']):
        raise ValueError('Contact and volume controls have different observation order')
    for i,label in enumerate(observation_labels(a)):
        rows.append([label,*[number(a[k]['signed_difference_ohm'][i],4) for k in ('spatial','fractions','total')],
            number(c['signed_difference_ohm'][i],4),
            *[number(abs(a[k]['signed_difference_ohm'][i])/a[k]['tolerance_ohm'][i],3) for k in ('spatial','fractions','total')],
            number(c['absolute_ratio'][i],3)])
    return table('Таблица 8. Все 24 раздельные численные сравнения в найденной точке',
        ['Запись, ось и состояние','ΔZ сетки, Ом','ΔZ долей, Ом','ΔZ суммы, Ом','ΔZ площадок, Ом',
         '|ΔZ|/допуск: сетка','|ΔZ|/допуск: доли','|ΔZ|/допуск: сумма','|ΔZ|/допуск: площадки'],rows)


def fraction_distribution_table(data):
    rows=[]
    for rule in data['distribution']['rules']:
        for i,name in enumerate(['Мягкотканный фон','Лёгкое','Кость','Целое сердце','Жир']):
            rows.append([rule['points'],name,f"{rule['changed_elements_by_material'][i]:,}".replace(',',' '),
                number(rule['maximum_absolute_fraction_change'][i],6),
                f"{rule['positive_changes_by_material'][i]:,}".replace(',',' '),
                f"{rule['negative_changes_by_material'][i]:,}".replace(',',' ')])
    return table('Таблица 5. Поэлементное изменение тканевых долей на одной сетке',
        ['Пробных точек','Материал','Элементы с изменением','Максимум |Δw|','Доля увеличилась: элементы','Доля уменьшилась: элементы'],rows)


def figure_html(fig,include_js=False):
    from IPython.display import HTML
    fig.update_layout(template='plotly_white',font=dict(size=15),height=fig.layout.height or 560,separators=', ',
        margin=fig.layout.margin.to_plotly_json() or dict(l=75,r=35,t=55,b=105),
        legend=fig.layout.legend.to_plotly_json() or dict(orientation='h',y=1.13),uirevision='preserve-view')
    return HTML(fig.to_html(full_html=False,include_plotlyjs=include_js,
        config=dict(displaylogo=False,responsive=True,scrollZoom=True)))


def numerical_figure(data,key='final',include_js=False):
    import plotly.graph_objects as go
    audit=data[key];fig=go.Figure()
    labels=observation_labels(audit)
    for field,name,color in [('spatial','Изменение сетки','#326ba8'),
        ('fractions','Пересчёт тканевых долей','#df8b26'),('total','Суммарное изменение','#885ba2')]:
        values=np.asarray(audit[field]['difference_ohm'])/np.asarray(audit[field]['tolerance_ohm'])
        fig.add_trace(go.Bar(x=labels,y=values,name=name,marker_color=color,
            customdata=np.c_[audit[field]['difference_ohm'],audit[field]['tolerance_ohm']],
            hovertemplate='%{x}<br>|ΔZ|=%{customdata[0]:.4f} Ом<br>Допуск=%{customdata[1]:.4f} Ом<extra>%{fullData.name}</extra>'))
    fig.add_trace(go.Scatter(x=labels,y=np.ones(len(labels)),mode='lines',name='Граница допуска',
        line=dict(color='#333333',dash='dash')))
    fig.update_layout(barmode='group',xaxis=dict(title='Запись, ось сборки и дыхательное состояние',tickangle=-50),
        yaxis=dict(title='|ΔZ| / допуск',rangemode='tozero'),height=710,
        margin=dict(l=75,r=35,t=80,b=180))
    return figure_html(fig,include_js)


def fit_figure(data,include_js=False):
    import plotly.graph_objects as go
    import pandas as pd
    O=pd.read_csv(data['input_directory']/'observations.local.csv');O=O[O.state!='quiet_breathing'].reset_index(drop=True)
    predicted=np.asarray(data['best']['predicted_Z_ohm']);fig=go.Figure()
    for phase,label,color in [('inhale','Вдох','#b7482b'),('exhale','Выдох','#23778a')]:
        ids=np.flatnonzero(O.state==phase);labels=[O.record_id[i].rsplit('-',1)[-1]+' · '+str(O.size_mm[i])+' мм'+
            (' · продольная' if O.axis[i]=='longitudinal' else ' · поперечная') for i in ids]
        fig.add_trace(go.Scatter(x=labels,y=O.Z_ohm.iloc[ids],name=label+': эксперимент',mode='lines+markers',
            line=dict(color=color,dash='dash'),marker=dict(symbol='circle',size=9)))
        fig.add_trace(go.Scatter(x=labels,y=predicted[ids],name=label+': модель',mode='lines+markers',
            line=dict(color=color),marker=dict(symbol='square',size=9)))
    fig.update_layout(xaxis=dict(title='Запись, размер и ось сборки',tickangle=-45),
        yaxis=dict(title='Передаточный импеданс, Ом'),height=700,margin=dict(l=75,r=35,t=80,b=170))
    return figure_html(fig,include_js)


def history_figure(data,include_js=False):
    import plotly.graph_objects as go
    run=data['run_directory']
    rows=[read(path) for path in sorted(run.glob('evaluation_*.local.json'))]
    valid=[row for row in rows if row['status']=='numerical_candidate_not_validated']
    budget=data['selected']['search']['evaluations']
    searching=[row for row in valid if row['evaluation']<=budget]
    fresh=[row for row in valid if row['evaluation']>budget]
    ids=[row['evaluation'] for row in searching];values=np.sqrt([row['mean_square_ohm2'] for row in searching])
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=ids,y=values,name='Каждый рассчитанный кандидат',mode='markers',marker=dict(color='#787878',size=7)))
    fig.add_trace(go.Scatter(x=ids,y=np.minimum.accumulate(values),name='Лучшее достигнутое значение',mode='lines',line=dict(color='#285f9a')))
    if fresh:
        fig.add_trace(go.Scatter(x=[r['evaluation'] for r in fresh],
            y=np.sqrt([r['mean_square_ohm2'] for r in fresh]),name='Свежая проверка лучшей точки',
            mode='markers',marker=dict(symbol='diamond-open',color='#378a78',size=11)))
    rejected=[row['evaluation'] for row in rows if row['status']=='invalid_geometry']
    if rejected:
        fig.add_trace(go.Scatter(x=rejected,y=[max(values)]*len(rejected),name='Отказ геометрии: условный уровень',
            mode='markers',marker=dict(symbol='x',color='#b7482b',size=9)))
    fig.update_layout(xaxis_title='Номер оценки',yaxis_title='Среднеквадратическая невязка, Ом')
    return figure_html(fig,include_js)


def coordinate_table(data):
    rows=[]
    for montage in data['best']['pose']['electrodes']:
        axis='Продольная' if montage['axis']=='longitudinal' else 'Поперечная'
        for name,xyz in zip(['I+','U+','U−','I−'],montage['centres_lps_mm']):
            rows.append([montage['size_mm'],axis,name,*[number(v,2) for v in xyz]])
    return table('Таблица 12. Расчётные центры всех боковых контактов',
        ['Размер, мм','Ось','Контакт','x LPS, мм','y LPS, мм','z LPS, мм'],rows)


def placement_figure(data,include_js=False):
    import plotly.graph_objects as go
    import pymeshlab
    import trimesh
    import nibabel as nib
    from skimage.measure import marching_cubes
    surface=trimesh.load(data['preparation']/'mesh_v9/body_meshing.stl',force='mesh')
    visual=pymeshlab.MeshSet();visual.add_mesh(pymeshlab.Mesh(surface.vertices,surface.faces.astype(np.int32)))
    visual.meshing_decimation_quadric_edge_collapse(targetfacenum=12000,preservenormal=True,preservetopology=True)
    mesh=visual.current_mesh();v=mesh.vertex_matrix();f=mesh.face_matrix()
    fig=go.Figure(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],
        color='#c6aa93',opacity=.35,name='Наружный контур',showlegend=True,hoverinfo='skip'))
    for name,label,color in [('lung','Маска лёгких','#4ba3bb'),('bone','Маска костей','#9b9b95'),
                             ('heart','Маска целого сердца','#b55061'),('fat','Маска жира','#d9ad3c')]:
        source=nib.load(data['masks'][name])
        vertices,faces,_,_=marching_cubes((np.asanyarray(source.dataobj)>0).astype(np.uint8),
            level=.5,step_size=3)
        xyz=nib.affines.apply_affine(source.affine,vertices)*[-1.,-1.,1.]
        fig.add_trace(go.Mesh3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],
            i=faces[:,0],j=faces[:,1],k=faces[:,2],color=color,opacity=.35,name=label,
            visible='legendonly',showlegend=True,hoverinfo='skip'))
    for montage in data['best']['pose']['electrodes']:
        xyz=np.asarray(montage['centres_lps_mm']);longitudinal=montage['axis']=='longitudinal'
        label=str(montage['size_mm'])+' мм · '+('продольная' if longitudinal else 'поперечная')
        fig.add_trace(go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='markers+lines',name=label,
            marker=dict(size=4,color='#df7927' if longitudinal else '#267d91',symbol=['circle','square','square','circle']),
            line=dict(width=2,color='#df7927' if longitudinal else '#267d91'),
            text=['I+','U+','U−','I−'],hovertemplate='%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f} мм<extra>%{fullData.name}</extra>'))
    contact_description=read(data['preparation']/'geometry_v5/fem_contacts.local.json')
    arm=np.asarray(contact_description['arm_physical_centres_lps_mm'])
    current_ids=set()
    for montage in contact_description['scenarios']:
        for group in (montage['active_patch_ids'][0],montage['active_patch_ids'][3]):current_ids.update(group)
    fig.add_trace(go.Scatter3d(x=arm[:,0],y=arm[:,1],z=arm[:,2],mode='markers',
        name='Постоянные площадки ТТРКГ на руках',marker=dict(size=4,color='#7953a0',
            symbol=['circle' if i in current_ids else 'square' for i in range(len(arm))]),
        text=['Токовая площадка I' if i in current_ids else 'Измерительная площадка U' for i in range(len(arm))]))
    author=read(data['input_directory']/'author_centre_20261012.local.json')['centre_lps_mm']
    for name,xyz,symbol,color in [('Ориентировочная точка автора',author,'x','#a85922'),
        ('Найденный общий центр',data['best']['pose']['centre_lps_mm'],'diamond','#b22b58')]:
        fig.add_trace(go.Scatter3d(x=[xyz[0]],y=[xyz[1]],z=[xyz[2]],mode='markers',name=name,
            marker=dict(symbol=symbol,color=color,size=8)))
    limits=np.c_[surface.bounds[0]-5,surface.bounds[1]+5]
    fig.update_layout(scene=dict(
        xaxis=dict(title='x LPS, мм',range=limits[0],autorange=False,nticks=5),
        yaxis=dict(title='y LPS, мм',range=limits[1],autorange=False,nticks=5),
        zaxis=dict(title='z LPS, мм',range=limits[2],autorange=False,nticks=5),
        aspectmode='data',uirevision='preserve-view',camera=dict(eye=dict(x=-2.3,y=-1.75,z=1.05))),
        height=740,margin=dict(l=85,r=270,t=20,b=35),legend=dict(orientation='v',x=1.01,y=1))
    return figure_html(fig,include_js)


def reader_text(data):
    best=data['best'];rmse=np.sqrt(best['mean_square_ohm2']);centre=best['pose']['centre_lps_mm']
    selected=data['selected'];search=selected.get('search',{})
    status='лимит оценок достигнут' if search.get('budget_reached') else ('остановлен по времени' if selected.get('search_stopped_for_time') else 'см. сохранённый критерий остановки')
    text=r'''# Повторный совместный FEM-подбор по боковым сборкам Георгия

## Постановка и доказательный статус

Цель расчёта — одновременно подобрать положение общего центра, угол сборок и эффективные удельные сопротивления мягкотканного фона, лёгкого и жира по всем боковым отведениям с помощью конечноэлементной модели (FEM). Рассматриваются 24 измеренных уровня импеданса на задержках вдоха и выдоха, включая обе оси крестов 140 и 60 мм. Тестовая запись спокойного дыхания не включена в эту целевую функцию. Номера каналов и роли сборок сохранены по принятой последовательности эксперимента; они не являются параметрами подгонки.

Результат остаётся предварительным. Исправление геометрии и успешные проверки линейного решателя не устанавливают численную сходимость, идентифицируемость тканей или физиологическую точность. Данный расчёт не оценивает ударный объём и фракцию выброса.

<!-- opening_summary -->

<!-- input_table -->

Маски четырёх тканей сохранены без повторной автоматической сегментации: исправляется наружный контур и перенос этих масок в сетку. На пересечениях масок применён явный приоритет: лёгкое, кость, сердце, жир, затем оставшийся мягкотканный фон. Фон включает все области, не отнесённые к четырём маскам; его сопротивление не является отдельным измерением сопротивления мышцы. Вне поля зрения КТ, в том числе в синтетических цилиндрах рук, принят тот же фон. После исправления контура три вокселя маски жира остаются вне маски тела; этот остаток отмечен в контроле и не объявлен анатомически исправленным.

Геометрия и перенос вокселей используют одну систему координат LPS. При преобразовании из NIfTI меняются знаки первых двух координат; перед сборкой электрических операторов миллиметры переводятся в метры. Поиск и результаты имеют единицы Ом·м для сопротивления, Ом для передаточного импеданса и миллиметры LPS для положения.

## Исправление наружного контура и подготовка сетки

<!-- repair_table -->

В функции выделения тела исправлена обработка пустого сечения после морфологического открытия: фон больше не может стать компонентой тела. Повторное выделение тела из исходной КТ Георгия с исправленным кодом дало прежнюю маску без изменений. Следовательно, этот программный дефект не повлиял на маску, использованную в данном расчёте.

Отдельная проблема касалась топологии наружной поверхности. Выполнена одна итерация трёхмерного морфологического замыкания с шестисвязной окрестностью. Добавлены 3318 вокселей, или 3,204 мл; воксели не удалялись. Доля изменённой маски составила 0,00913 %, включая 28 вокселей в заданной боковой области. Поверхность извлечена на уровне 0,49 бинарной маски с сохранением исходного разрешения до объединения с руками. Это ограниченная численная регуляризация автоматически выделенного контура; независимая анатомическая точность не установлена.

<!-- contour_image -->

**Рисунок 1. Исходный и регуляризованный контуры на КТ.** Синий контур соответствует исходной маске тела; оранжевый — маске после ограниченного трёхмерного замыкания. Серый фон показывает исходную КТ. Представлены локальные участки выбранных сечений с изменениями контура; подписаны координата z LPS и число изменённых вокселей во всём соответствующем сечении. Эти фрагменты не ограничены областью боковых электродов. Совпадение контуров вне отмеченных участков и малая доля изменённых вокселей не доказывают анатомическую точность всей поверхности.

Цилиндрические руки имеют обхват 37 см, по уточнению автора для обеих рук на уровне токовых электродов. Это измеренный исходный размер; представление рук цилиндрами является допущением. Каждый цилиндр заглублён до полного покрытия поперечного сечения туловищем. Зазор от искусственной верхней плоскости среза увеличен до 2 мм для устранения почти касательного соединения. Центры соединений рассчитаны, а не восстановлены по анатомическим ориентирам.

В качестве общей расчётной поверхности принята фактическая наружная граница базовой тетраэдральной сетки. Её максимальное расстояние до исправленного исходного контура по 15 000 пробным точкам в каждом направлении составило 0,446 мм, относительное расхождение объёмов — 0,01194 %. Это выборочная геометрическая проверка, без гарантии максимального расстояния между всеми точками поверхностей. Неуспешные предшествующие варианты сохранены; строгая проверка воспроизведения прежней полигональной границы не объявлена пройденной.

Для пространственного сравнения выполнено согласованное вложенное сгущение: выбранный тетраэдр делится на восемь, а соседние элементы получают согласованные переходы. Исходные узлы и наружная многогранная поверхность сохраняются. Первое уточнение охватило 76 764 родительских элемента, около 3,30 % исходной сетки, в областях радиусом 12 мм около направляющих центров. После исправления площадок подготовлена ещё одна вложенная сетка с 829 578 узлами и 4 174 356 тетраэдрами, с радиусом уточнения 6 мм около 28 боковых и 12 постоянных контактов на руках. В качестве начальных параметров использован сохранённый кандидат предыдущего прерванного поиска, без переноса его невязки. Проверка относится к локальному уточнению и не доказывает сходимость всего объёма или всего окна поиска положения.

Показатель формы тетраэдра равен единице для правильного тетраэдра и уменьшается при вырождении. Обусловленность матрицы трёх рёбер характеризует вытянутость и близость элемента к вырождению. Сохранены заранее заданные пороги: показатель формы не меньше 0,01 и обусловленность не больше 10 000. Все объёмы положительны, нарушений этих порогов нет. Для принятия сопротивлений дополнительно требуется проверка изменения самого электрического решения.

Показатель формы вычисляется как

$$q_e=\frac{12(3V_e)^{2/3}}{\sum_{k=1}^{6}\ell_{e,k}^{\,2}},$$

где $V_e$ — положительный объём тетраэдра, $\ell_{e,k}$ — длины его шести рёбер; $q_e$ безразмерен. Для второго показателя составляется матрица из трёх векторов рёбер, выходящих из первой вершины в сохранённой нумерации, и рассчитывается отношение её наибольшего сингулярного числа к наименьшему. Этот показатель зависит от выбора опорной вершины; это не число обусловленности полной FEM-системы.

<!-- mesh_table -->

<!-- inheritance_table -->

<!-- shape_quality_figure -->

**Рисунок 2. Показатель формы тетраэдров.** Синие точки относятся к базовой сетке, оранжевые — к сгущённой. Показаны минимум и сохранённые процентили распределения: например, ниже значения в позиции «1 %» находится 1 % элементов. Пунктирная линия обозначает порог 0,01. Вертикальная шкала логарифмическая. Точки не соединены: по нескольким процентилям полное распределение не восстанавливалось.

<!-- edge_quality_figure -->

**Рисунок 3. Обусловленность матрицы рёбер тетраэдров.** Синие и оранжевые точки показывают сохранённые процентили и максимум для базовой и сгущённой сеток. Пунктирная линия обозначает порог 10 000. Вертикальная шкала логарифмическая; меньшее значение означает лучшую обусловленность. Это характеристика формы элементов, а не обусловленности полной электрической системы.

## Физические электроды и их электрические соединения

На руках в каждой записи сохраняются одни и те же 12 физических площадок. Для монтажей 1×1, 1×4 и 2×2 меняется объединение площадок в электрические группы. В 1×1 контакты I и U находятся сверху, по уточнению автора от 12 октября; токовые контакты 1×4 также расположены сверху. Монтаж 2×2 включает диаметрально противоположные нижние контакты, в том числе измерительные U. Неиспользуемая площадка имеет отдельный плавающий потенциал и нулевой суммарный внешний ток. Это модельное допущение об отключённом металлическом контакте. Наличие всех площадок на руках во время крестов принято как продолжение монтажа и требует отдельного подтверждения.

Каждый крест содержит восемь боковых контактов. Измерения обеих осей рассчитываются на одной полной схеме контактов. Использована полная электродная модель — CEM, учитывающая конечную площадь и контактный импеданс. Номинальный диаметр диска 5 мм принят по стандарту проекта, но для этого эксперимента независимо не измерен. Удельный контактный импеданс равен 0,000159155 Ом·м² и также задан как сценарий.

Номинальный размер L означает расстояние между внешними токовыми контактами I. Расстояние между измерительными контактами U равно L/2; последовательные промежутки I–U, U–U и U–I равны L/4, L/2 и L/4. Для 140 мм это 35, 70 и 35 мм. По протоколу автор подтвердил это правило изготовления, однако координаты после наклейки независимо не измерены.

Фактические площадки составлены из граничных треугольников. Допустимы относительная ошибка площади меньше 12 %, смещение её центра меньше 1 мм и максимальное удаление узла от заданного центра меньше 3,7 мм. В начальной контрольной точке у центра автора все площадки прошли эти условия. Прежняя стартовая позиция со смещением 20 мм не прошла ограничение формы одного поперечного контакта; такой отказ не заменяется ослаблением допуска. На вложенной сетке площадка состоит точно из дочерних граней прежнего контакта, поэтому сравнение сохраняет её площадь и физическое положение.

Обнаружено, что грубая сетка выбора площадок искусственно ограничивала поиск положения: пять из шести отклонённых смещений прошли тот же контроль при выборе площадок на более подробной границе. Координаты, анатомия и допуски при этом не менялись. Поэтому первый поиск остановлен, а его оценки сохранены как диагностика. В исправленной постановке опора построения центра и нормалей отделена от сетки выбора контактов. Центр и нормали вычисляются по базовой сетке исправленной геометрии; сетка выбора контактов содержит 605 573 узла и допускает более точное представление диска. Карты наследования контактных граней и геометрической поверхности хранятся отдельно. Отказ прежней дискретизации не трактуется как невозможность установить электрод физически.

Для записей 007–010 сохранён ранее рассчитанный сценарий ТТРКГ 1×4. Выбор между 1×1 и 1×4 для этого блока пока не подтверждён. Боковые роли известны, однако эта оставшаяся неопределённость должна учитываться при физической интерпретации. Возбуждение рассчитывается по собственной токовой паре каждого отведения; межканальный режим прибора не определяется данным подбором.

## Разделение влияния сетки и тканевых долей

Из КТ сохранены отдельные маски лёгкого, костей, сердца и жира. Остальные участки тела и искусственные руки относятся к мягкотканному фону. Для линейного элемента используется объёмно усреднённая проводимость:

$$\sigma_e=\frac{w_{e,\text{мт}}}{\rho_1}+\frac{w_{e,\text{л}}}{\rho_2}
+\frac{w_{e,\text{ж}}}{\rho_3}+\frac{w_{e,\text{к}}}{\rho_{\text{к}}}
+\frac{w_{e,\text{с}}}{\rho_{\text{с}}},$$

где $w_{e,\text{мт}}$, $w_{e,\text{л}}$, $w_{e,\text{ж}}$, $w_{e,\text{к}}$ и $w_{e,\text{с}}$ — объёмные доли мягкотканного фона, лёгкого, жира, костей и целого сердца в элементе $e$; $\rho_1$, $\rho_2$ и $\rho_3$ — подбираемые сопротивления фона, лёгкого и жира; $\rho_{\text{к}}$ и $\rho_{\text{с}}$ — фиксированные сопротивления костей и маски сердца в Ом·м; $\sigma_e$ — проводимость в См/м. Сумма пяти долей равна единице. Доли вычисляются по фиксированным воксельным маскам с вложенными наборами 1024 и 4096 пробных точек. Такая проверка интегрирования уточняет доли при неизменной сетке и не заменяет пространственного сгущения.

В объёме тела решается уравнение сохранения тока, а на электродах задаются условия полной электродной модели:

$$\nabla\!\cdot(\sigma\nabla u)=0,\qquad
u+z_c\sigma\frac{\partial u}{\partial n}=U_g\ \text{на }E_g,\qquad
\int_{E_g}\sigma\frac{\partial u}{\partial n}\,dS=I_g,$$

где $u$ — потенциал в ткани, $\sigma$ — заданная по элементам проводимость, $n$ — наружная нормаль, $z_c$ — удельный контактный импеданс, $E_g$ — объединение физических площадок одной электрической группы, $U_g$ — их общий потенциал, $I_g$ — заданный суммарный ток группы. При этом соглашении положительный $I_g$ означает ток, вводимый в тело; физическая плотность тока равна $\mathbf j=-\sigma\nabla u$. На остальной наружной поверхности нормальная составляющая тока равна нулю. Сумма токов всех групп равна нулю; произвольный уровень потенциала фиксируется при решении линейной системы. Для соединённых площадок ограничен суммарный ток группы, а не ток каждой площадки по отдельности.

Передаточный импеданс отведения рассчитывается по потенциалам измерительных групп:

$$Z=\frac{U_{U+}-U_{U-}}{I_s},$$

где $U_{U+}$ и $U_{U-}$ — потенциалы положительной и отрицательной измерительных групп, $I_s$ — ток собственной токовой пары этого отведения. Численный ток 1 А используется как нормировка линейной задачи для получения $Z$ в омах и не является заявленной амплитудой прибора. На каждой пробе поиска сопротивления задают проводимости элементов, затем собирается и решается электрическая система, получаются 24 уровня $Z$ и вычисляется их невязка относительно наблюдений.

В данной постановке проводимости вещественные и изотропные. Диэлектрическая составляющая и направленная проводимость тканей отдельно не моделируются. Сопоставление такого передаточного импеданса с записанным BEI является физическим допущением; фактический способ демодуляции сигнала этим расчётом не установлен.

В первом сравнении базовая сетка заменяется вложенной, а проводимость каждого дочернего элемента точно наследуется от родителя. Таким образом выделяется влияние пространства конечных элементов. Для первоначальной пары относительное расхождение энергии продолженного проверочного кусочно-линейного поля составило 2,22·10⁻¹⁶. Для исправленной пары оно равно нулю в сохранённой арифметике при обоих начальных наборах проводимости; допуск равен 10⁻¹⁰. Это проверка корректности переноса заданного поля, без решения новой электрической задачи. Новое FEM-решение может измениться, поскольку пространство допустимых полей расширено.

Во втором сравнении сетка уже не меняется, но тканевые доли дочерних элементов пересчитываются по исходным маскам. Тем самым выделяется влияние пространственного представления материалов. Отдельно проверяется суммарное изменение между базовой сеткой и сгущённой сеткой с пересчитанными долями. Взаимная компенсация двух изменений не считается прохождением раздельных проверок.

Для пространственного и материального сравнений допуск равен $0,02+0,005|Z|$ Ом, а для интегрирования долей — $0,02+0,001|Z|$ Ом; в каждом случае $Z$ относится к более уточнённому правому расчёту. На исходном контроле у авторского центра максимальные изменения составили 1,762 Ом от сетки, 1,093 Ом от тканевых долей и 2,855 Ом суммарно. Допуск превышен соответственно в 6, 12 и 16 из 24 строк. Сопротивления обеих фаз в этом контроле одинаковы, поэтому это 12 различных условий расчёта, повторённых для двух наборов экспериментальных уровней.

Суммарные объёмы тканей почти не изменились при пересчёте долей; это не обеспечивает совпадения импеданса. Существенным остаётся распределение проводимости внутри отдельных элементов. Для первоначальной пары основной материал и область, определяющие выявленное изменение, не установлены. Проверка линейных систем, теоремы взаимности и производных для обоих крестов прошла. Максимальная разность от увеличения числа пробных точек в первом контроле равна 0,0153 Ом и существенно меньше выявленных пространственных изменений.

В начальном контроле исправленной постановки пространственное изменение уменьшилось до 0,209 Ом, изменение от тканевых долей — до 0,517 Ом, суммарное — до 0,628 Ом. Во всех 24 строках соблюдён установленный допуск. Эти значения относятся к другому начальному набору сопротивлений и более подробной сетке; их нельзя считать эффектом одного лишь изменения площадок. После нового подбора проверки повторяются при найденных параметрах.

Отдельное сравнение старых и исправленных площадок выполнено на одной и той же объёмной сетке при одинаковых сопротивлениях и положении. Максимальное изменение составило 1,198 Ом, с превышением допуска в 14 из 24 значений. Поэтому изменение от точности контактов рассматривается третьим самостоятельным компонентом. В найденной точке оно проверяется ещё раз при следующем уточнении площадок, отдельно от изменения пространства и тканевых долей.

<!-- fraction_distribution_table -->

В исправленной паре при 4096 пробных точках поэлементно изменились доли мягкотканного фона и жира в 135 537 тетраэдрах, или примерно 3,25 % элементов сетки. Максимальное изменение одной доли равно 0,8552. Это разность между унаследованной средней долей родительского элемента и заново рассчитанной долей его меньшего потомка. Она не является оценкой ошибки сегментации или изменения анатомии. Доли лёгкого, костей и целого сердца в этой паре совпадают точно. Такая неизменность не подтверждает достаточного разрешения этих тканей: проверка не изменила их материальное представление. Выявленный эффект пересчёта долей связан с более подробным представлением границы «фон–жир». Влияния фона и жира на импеданс этой проверкой не разделяются: уменьшение одной доли сопровождается увеличением другой. Эти результаты относятся к исправленной паре и не устанавливают причину прежнего расхождения всех материалов.

<!-- numerical_table -->

<!-- contact_table -->

<!-- numerical_figure -->

**Рисунок 4. Раздельные численные изменения в найденной точке.** Столбцы показывают модуль изменения импеданса, разделённый на соответствующий допуск: синий — только сетка; оранжевый — пересчёт долей на одной сетке; фиолетовый — суммарное изменение. Чёрная пунктирная линия на уровне 1 обозначает границу допуска. Значения выше неё не прошли проверку. Каждая строка содержит запись, ось сборки и дыхательное состояние. Щелчок по легенде скрывает соответствующий ряд.

<!-- contact_figure -->

**Рисунок 5. Влияние отдельного уточнения контактных площадок.** Зелёные столбцы показывают знаковую разность импеданса при прямом выборе площадок на сгущённой границе и при наследовании прежних площадок, разделённую на допуск. Объёмная сетка, тканевые доли, сопротивления, координаты центров и электрические группы одинаковы. Положительное значение означает увеличение импеданса. Чёрные пунктирные линии +1 и −1 ограничивают допустимый интервал. Контроль охватывает боковые и постоянные площадки рук; изменение не приписывается исключительно боковым электродам.

<!-- control_rows_table -->

Знак каждого изменения в таблице соответствует более подробному варианту минус исходный. Суммарное изменение сетки и долей является суммой этих двух знаковых разностей. Изменение площадок рассчитано отдельной последующей парой. Допуски относятся к каждой паре: они не являются оценками дисперсии экспериментальных уровней или доверительными интервалами сопротивлений.

<!-- selected_numerical_sentence -->

<!-- additional_refinement_sentence -->

## Совместный подбор удельных сопротивлений и положения

Подбираются девять параметров: по три эффективных сопротивления для вдоха и выдоха, два смещения общего центра по касательным направлениям и общий угол. Начальная ориентировочная точка автора имеет координаты (−204,346; −20,215; −207,500) мм в системе LPS: положительные направления x, y и z соответствуют левой стороне, спине и голове пациента. Точка задана по изображению и не является независимо измеренным положением. Поиск ограничен смещениями ±30 мм и углом ±20°; это вычислительная область, а не доверительный интервал установки.

Контакты всех размеров строятся от одного центра. Продольное направление связано с осью z LPS, поперечное перпендикулярно ему в местной касательной плоскости. Смещения I и U задаются в этой плоскости, затем точки проецируются на треугольную поверхность. Номинальный размер поэтому не является доказанным расстоянием вдоль кожи. Общая касательная конструкция поиска перенесена из расчётов Ника; индивидуальная геометрия и измеренные уровни относятся к Георгию.

В расчёте экспериментальный «горизонтальный» размерный ряд сопоставлен продольной касательной к поверхности грудной клетки, а «вертикальное» отведение 005 и соответствующая ось крестов — перпендикулярной касательной. Это соглашение принято с учётом авторского уточнения о направлении сборок вдоль тела. В кресте 006 канал 1 относится к основной продольной оси, канал 2 сохраняет вторую ось из записи 005. Номера каналов и преемственность отведений подтверждены отдельно; анатомическое соответствие экспериментальных названий осей независимо не зарегистрировано. Поиск угла в пределах ±20° не проверяет альтернативную перестановку осей на 90°.

Частота зондирования 100 кГц подтверждена автором и используется в модели. Выполнен ограниченный FEM-поиск с унаследованными литературными интервалами трёх материалов при этой частоте, связанными с [общим справочником электрических свойств тканей](../40.08_Литературные_диапазоны_электрических_свойств_тканей.md). Авторское решение от 9 октября явно относится к двуслойной ветке; распространение ограничения на жир является условием данного FEM-прогона, без отдельного подтверждения. Область вне этих интервалов не исследована. Полученный кандидат не является результатом полного технического поиска 0,1–100 Ом·м, предусмотренного общими правилами обратной FEM-задачи. Индивидуальная применимость литературных интервалов не доказана; во время расчёта их не меняли. Кости фиксированы на уровне 48,098 Ом·м, вся маска сердца — на уровне 4,649 Ом·м как условный материал миокарда. Кровь и миокард внутри этой маски отдельно не моделируются. Между вдохом и выдохом геометрия КТ глубокого вдоха неизменна: изменение сопротивлений может компенсировать неучтённое изменение анатомии.

Всем материалам задана действительная скалярная проводимость. Анизотропия мышц и комплексные электрические свойства тканей не включены. Поэтому различие продольного и поперечного отведений в данной модели возникает за счёт геометрии и распределения тканевых масок. Если после численных проверок сохраняется расхождение осей креста, оно само по себе не позволяет выбрать между ошибкой установки, недостаточностью скалярной модели, изменением физиологического состояния и особенностью записи.

Минимизируется средний квадрат невязки по 24 уровням; всем уровням приписан одинаковый вес. Дыхательные интервалы сохранены из существующей кандидатной разметки; их принятие не изменялось. Последовательные записи считаются сопоставимыми по состоянию испытуемого как допущение данного объединения. Различия времени, контакта и состояния не заменены дополнительными коэффициентами усиления или смещения.

Равные веса отдельных уровней не означают равного веса каждого размера: для 140 мм включены 12 уровней, для 60 мм — 6, для 80, 100 и 120 мм — по 2. Повторные геометрии сохранены как отдельные записи. Целевая функция является арифметическим критерием подбора; независимость и одинаковая дисперсия ошибок измерений не установлены, поэтому её значение не используется для построения статистических доверительных интервалов.

Использован ограниченный прямой поиск по сопротивлениям, положению и углу с четырьмя исходными точками и первым лимитом 24 оценки. Лимит выбран по измеренному времени прямого расчёта, чтобы сохранить запас для проверки найденного кандидата в исходном сроке до 10 часов. Прежние сопротивления использованы только как начальное приближение. Оценки невязки прерванного поиска не перенесены в новый поиск. Недопустимые площадки и выход контактов за сканированную поверхность исключаются. После выбора лучшей точки электрическое решение пересчитывается заново с очищенным кэшем состояний, затем повторяются раздельные численные проверки при фиксированных найденных параметрах.

Вначале сравниваются четыре исходных кандидата. Затем из лучшего выполняется один общий покоординатный поиск: проверяются положительные и отрицательные изменения каждого из девяти параметров, а после обхода принимается лучшая пробная точка. Если улучшения нет, шаги уменьшаются вдвое. Четыре исходных кандидата поэтому не означают четыре независимо доведённых до сходимости поиска. Ограничение числа оценок может остановить алгоритм раньше уменьшения шагов до заданного разрешения.

В счётчик обходов входят и неполные обходы, прерванные лимитом оценок. В сохранённой реализации шаг уменьшается также после неполного обхода без улучшения. Поэтому последние сохранённые шаги сами по себе не доказывают, что все соответствующие направления проверены. Для данного кандидата критерий разрешения поиска не достигнут.

Начальные шаги по натуральным логарифмам сопротивлений фона, лёгкого и жира равны 0,18; 0,25 и 0,30 для каждой фазы, по двум смещениям центра — 10 мм, по углу — 5°. Критерий разрешения требует уменьшения этих шагов ниже 0,005; 0,01 и 0,01 для логарифмов сопротивлений и ниже 0,5 мм и 0,5° для положения и угла. Эти численные пороги относятся к алгоритму поиска и не являются точностью измерения параметров.

Отказ по критериям площади, центроида или протяжённости контакта означает непригодность его дискретного представления на данной сетке. Он не доказывает физическую невозможность такого расположения. Поэтому отказанные положения сохраняются в истории поиска, а их исключение ограничивает вывод о минимуме во всём заданном окне.

<!-- result_sentence -->

<!-- rho_table -->

<!-- search_table -->

<!-- fit_figure -->

**Рисунок 6. Измеренные и рассчитанные уровни импеданса.** Круги и пунктирные линии обозначают экспериментальные уровни; квадраты и сплошные линии — найденный модельный кандидат. Красный цвет относится к вдоху, синий — к выдоху. Линии соединяют отдельные записи для удобства сравнения и не представляют непрерывную зависимость во времени или по размеру. Все четыре ряда имеют независимые переключатели в легенде.

<!-- residual_table -->

<!-- residual_sentence -->

<!-- history_figure -->

**Рисунок 7. История совместного подбора.** Серые точки показывают невязку каждого завершённого FEM-кандидата поиска; синяя сплошная линия — наименьшую невязку, достигнутую к этому номеру оценки. Зелёный незакрашенный ромб после лимита поиска обозначает свежую проверку лучшей точки с очищенным кэшем, которая не является новой попыткой поиска. Красные кресты обозначают геометрические отказы и помещены на условном верхнем уровне: высота креста не является значением невязки. История характеризует выполненный ограниченный поиск; она не доказывает единственность или глобальность минимума.

<!-- coordinate_table -->

<!-- placement_figure -->

**Рисунок 8. Найденное положение боковых сборок на расчётном контуре.** Оранжевые ряды относятся к продольным отведениям, синие — к поперечным; фиолетовые точки — к постоянным площадкам ТТРКГ на руках. Круги обозначают токовые I, квадраты — измерительные U. Линии соединяют центры и не изображают токовые траектории. Коричневый крест — ориентировочная точка автора, красный ромб — найденный общий центр. Через легенду отдельно включаются маски лёгких, костей, целого сердца и жира. Наружная поверхность для просмотра упрощена до 12 000 граней; тканевые поверхности извлечены с шагом три вокселя без последующего схлопывания треугольников. Эти визуальные копии не использовались в FEM. Каждый размер и каждую ткань можно скрыть через легенду. При вращении сцены и переключении видимости ракурс сохраняется.

## Диагностика возбуждения

Полная модель сохраняет также передачу от токовой пары второго канала к измерительным электродам данного отведения. При неизменных найденных сопротивлениях и положении можно сравнить собственное возбуждение и три заданных сценария равных токов двух каналов. Эти значения не являются результатами дополнительных обратных задач. Сценарий со сдвигом 90° дополнительно предполагает регистрацию модуля комплексного напряжения. Выбирать реальную фазу или способ демодуляции только по меньшей невязке нельзя; нужны сведения о приборе или разделяющий эксперимент.

<!-- source_table -->

## Ограничения и дальнейшее использование

<!-- sensitivity_figure -->

**Рисунок 9. Локальная чувствительность к сопротивлениям в найденной точке.** Карта показывает сохранённые производные импеданса по натуральным логарифмам шести свободных сопротивлений при фиксированных координатах и угле. Красный цвет соответствует положительной производной, синий — отрицательной, белый — нулю. Производная имеет единицу Ом: например, её произведение на малое относительное изменение сопротивления приближённо задаёт изменение импеданса. Нули для параметров другой дыхательной фазы обусловлены раздельными электрическими решениями. Эта производная не равна доле ткани в измеренном сигнале и не включает ошибку её сегментации.

<!-- sensitivity_table -->

<!-- sensitivity_sentence -->

<!-- breathing_contrast_figure -->

**Рисунок 10. Парное изменение импеданса между задержками выдоха и вдоха.** Синие круги показывают экспериментальную разность «выдох минус вдох», оранжевые квадраты — ту же разность для модели при найденных параметрах. Чёрная пунктирная линия соответствует отсутствию изменения. Знаки выше и ниже неё означают соответственно увеличение и уменьшение импеданса на выдохе. Пары составлены внутри одной записи и оси сборки; это разность двух уровней задержек, а не амплитуда отдельного дыхательного цикла. Интервалы остаются кандидатной разметкой. Подбор минимизировал ошибку абсолютных уровней; отдельная невязка дыхательного контраста в целевую функцию не добавлялась.

<!-- breathing_interpretation -->

Численные изменения необходимо оценивать при найденных сопротивлениях и всех фактических контактных площадках. Одно локальное сгущение не доказывает покрытие всего окна положения. Если выбранные площадки находятся вне сгущённых участков, требуется дополнительное уточнение вокруг найденного монтажа; при сохранении расхождений — расширение области в глубину и между контактами. Положительные показатели качества формы тетраэдров сами по себе не подтверждают точность импеданса.

Литературная область и малое значение целевой функции не устанавливают сопротивления отдельных тканей. Требуются профили невязки, дополнительные исходные точки и проверка совместной идентифицируемости сопротивлений и положения. Якобиан по шести сопротивлениям, сохранённый для кандидата, не включает координаты и угол и не является проверкой идентифицируемости всех девяти параметров. Полученные значения пока не передаются как принятые поправочные параметры для расчёта ТТРКГ, ударного объёма или фракции выброса.

Большая остаточная невязка при исчерпанном бюджете также не доказывает несовместимость экспериментальных данных с моделью: недостаточное исследование целевой функции остаётся одной из причин. Для различения причин нужны как продолжение оптимизации и профили параметров, так и раздельные сценарии геометрии дыхания, скалярных свойств тканей, подключения второго источника и повторной установки электродов. Малый прямой вклад сердца и костей на данном этапе не установлен; их фиксированные свойства являются условием расчёта.

Исправленная геометрия закреплена как общий вход будущих расчётов Георгия. Вместе с ней сохранены положительные локальные численные проверки и отрицательный результат покрытия площадок сгущением. Общий допуск количественной инверсии остаётся закрытым; предварительные исследовательские расчёты требуют явного указания модели и исходных условий. Исходные авторские координаты ТЕПС сохранены отдельно от проекций на исправленную поверхность; максимальное смещение при таком переносе составляет 4,55 мм и не является измеренной ошибкой установки электродов. Прежний общий указатель сохранён в резервной копии.

Ноутбук читает сохранённые расчёты и проверяет их контрольные суммы. Подготовка моделей и FEM выполняются отдельно через каноническую MATLAB-точку входа. Исторические отчёты сохраняют прежнюю геометрию, параметры и отрицательные проверки. Первичные КТ, маски, сетки и физиологические записи остаются локальными.

## Сечения расчётных сеток в области бокового монтажа

<!-- coarse_section_figure -->

**Рисунок 11. Сечение базовой сетки через найденный центр.** Плоскость задана основной осью сборок и внутренней нормалью к поверхности в центре. Синие отрезки — пересечения граней тетраэдров с этой плоскостью; красный ромб — центр. Горизонтальная координата отсчитывается вдоль основной оси, вертикальная — по внутренней нормали, обе в миллиметрах. Окно охватывает ±80 мм вдоль оси и от −5 до 55 мм по нормали. Вертикальная координата не является расстоянием от каждой точки до кожи. Тонкие многоугольники сечения могут возникать при прохождении плоскости около вершины правильного объёмного элемента и сами по себе не означают плохую форму тетраэдра.

<!-- fine_section_figure -->

**Рисунок 12. Сечение локально сгущённой сетки.** Плоскость, диапазоны осей, масштаб и обозначения совпадают с рисунком 11. Синие отрезки показывают фактические пересечения меньших тетраэдров, без дорисовки равномерной сетки. При увеличении видны области локального уточнения и более крупные элементы в глубине. Изображение поясняет расположение сгущения; количественный вывод о его достаточности основан на раздельных проверках импеданса выше. Каждый график имеет собственные настройки масштаба и отображения.

<!-- invalid_contact_table -->

Для площадок применены неизменные ограничения: относительная ошибка площади меньше 12 %, смещение центроида меньше 1 мм, максимальное расстояние точек площадки от заданного центра меньше 3,7 мм. В таблице показан максимум каждого показателя среди физических контактов соответствующей оценки; максимумы могут относиться к разным площадкам. Одно положение может повториться с разными сопротивлениями. При отказе электрическая невязка не вычислялась; её нельзя приравнивать к нулю или трактовать как измеренное большое сопротивление.

<!-- starting_candidates_table -->

Начальные кандидаты различаются как сопротивлениями, так и положением; их число не равно числу различных начальных положений. В этой серии три кандидата имеют нулевые смещения, один — первое смещение −20 мм; начальный угол у всех нулевой. Дальнейшие пробные положения определяются описанным покоординатным поиском. Ограничение ±30 мм задаёт допустимую область, но не означает равномерного обследования всей этой области.

<!-- coverage_table -->

Таблица содержит только площадки, не полностью покрытые локальным сгущением. Номер площадки задаётся внутри полной физической схемы конкретной записи, до объединения контактов в электрические группы. Для сопоставления разных записей приведены координаты центра; одинаковые центры могут повторяться. Нулевая доля означает, что ни одна граничная грань этой площадки не разделилась на дочерние грани в сравниваемой паре сеток. Это не означает отсутствие электрода или нулевую чувствительность к окружающему объёму.

Если граничные грани контакта не менялись, его прямой выбор на уточнённой сетке может повторить прежнюю площадку. Малое изменение импеданса в таком случае не устраняет пробел пространственной проверки этого контакта.
'''
    sentence=(f'Лучшая достигнутая среднеквадратическая невязка — **{number(rmse)} Ом**; '
        f'центр — **({"; ".join(number(v,2) for v in centre)}) мм LPS**, угол — **{number(best["pose"]["angle_deg"],2)}°**. '
        f'Критерий завершения: {status}. Достижение требуемого пространственного разрешения поиска '
        f'{"подтверждено в сохранённом критерии" if search.get("resolution_reached") else "не подтверждено"}. '
        'Это лучший кандидат выполненного поиска, без доказательства глобального минимума и без независимой физиологической валидации.')
    if np.max(np.abs(best['pose']['offsets_uv_mm']))<1e-8 and abs(best['pose']['angle_deg'])<1e-8:
        sentence+=' Лучший кандидат сохранил исходный спроецированный центр и нулевой угол. Новая локализация при заданном разрешении поиска не установлена; сохранение старта не подтверждает истинное положение.'
    forwards=data['selected_forwards'];q=max(max(r['difference_ohm']) for r in forwards)
    errors=[v for r in forwards for test in r['derivatives'] for v in test['relative_errors']]
    numerical=(f'В найденной точке максимальное изменение от увеличения числа пробных точек '
        f'с 1024 до 4096 равно {number(q,5)} Ом. Допуск интегрирования '
        f'{"соблюдён во всех контрольных вариантах" if data["final"]["quadrature_passed"] else "не соблюдён"}. '
        f'Максимальная относительная невязка линейных систем — '
        f'{scientific_number(max(r["maximum_system_residual"] for r in forwards))}; '
        f'максимальное нарушение взаимности — '
        f'{scientific_number(max(r["maximum_reciprocity_ohm"] for r in forwards))} Ом. '
        f'На базовой сетке выполнена проверка производных для обоих крестов; максимальная относительная ошибка — '
        f'{scientific_number(max(errors))} при допуске 10⁻⁴. На двух уточнённых вариантах эта проверка отдельно не повторялась. Эти проверки относятся к '
        'исполненной дискретной модели и не проверяют точность анатомии.')
    coverage=data['coverage'];contact=data['contact_final']
    numerical+=(f' При отдельном уточнении контактных площадок максимальное изменение '
        f'составило {number(max(contact["difference_ohm"]),4)} Ом; '
        f'превышений допуска — {contact["failed_values"]} из 24. '
        f'Минимальная доля площади фактического контакта, попавшая в сгущение границы, — '
        f'{number(100*coverage["minimum_refined_boundary_area_fraction"],2)} %. '
        f'Полное покрытие всех площадок сгущением '
        f'{"подтверждено" if coverage["all_actual_contact_faces_refined"] else "не подтверждено"}. '
        'Это контроль граничных граней электродов; он не подтверждает сходимость '
        'во всём объёме тела или во всей области поиска положения.')
    missing=sum(not r['all_faces_refined'] for r in coverage['contacts'])
    if missing:
        numerical+=(f' Неполное покрытие обнаружено в {missing} из '
            f'{len(coverage["contacts"])} описаний площадок по отдельным записям '
            '(повторяющиеся физические центры здесь учитываются повторно). '
            'Поэтому даже соблюдение допуска изменения импеданса не подтверждает '
            'достаточного сгущения у всех контактов найденного положения; '
            'конкретные площадки перечислены в таблице 17.')
    extra_text=''
    if data['additional_refinement'] is not None:
        attempt=data['additional_refinement']['attempts'][-1]
        quality=attempt['quality']
        if quality.get('reason')!='node limit exceeded before allocating refinement':
            raise ValueError('Additional refinement requires a specific scientific explanation')
        grouped=lambda value:f'{value:,}'.replace(',',' ')
        extra_text=('**Дополнительное сгущение в найденном положении не выполнено.** '
            f'При том же радиусе 6 мм вокруг фактических центров боковых и постоянных '
            f'контактов выбраны {grouped(attempt["selected_parents"])} родительских элементов. '
            f'Точный подсчёт новых узлов заданной схемы сгущения дал '
            f'{grouped(quality["predicted_nodes"])} узлов при заранее установленном пределе '
            f'{grouped(quality["maximum_fine_nodes"])}. Построение остановлено до выделения '
            'новой сетки, поэтому её качество и электрическое решение не проверены. '
            'Это ограничение вычислительного ресурса; оно не является отказом по форме '
            'тетраэдров и не отменяет результатов уже выполненного сравнения исходной пары. '
            'Дополнительной ступени проверки сходимости в найденной точке нет.')
        if data['resource_screen'] is not None:
            extra_text+=' Дополнительный предварительный подсчёт для меньших областей дал: '+ '; '.join(
                f'радиус {number(r["radius_mm"],1)} мм — {grouped(r["predicted_nodes"])} узлов'
                for r in data['resource_screen']['attempts'])+'. '
            extra_text+=('Эти варианты также превышают тот же предел и не построены. '
                'Меньший радиус сам по себе не обеспечивает покрытия конечной площадки. '
                'Результат не исключает другой стратегии локального уточнения, например '
                'только около неохваченных контактов; такая стратегия здесь не проверена.')
    if data['config']['max_evaluations']>24:
        sentence+=(f' После первого лимита выполнено продолжение той же исправленной задачи '
            f'с лимитом {data["config"]["max_evaluations"]} оценок. Совпадение зависимостей проверено перед '
            'повторным использованием оценок; исходный срок работы сохранён.')
    jac=np.asarray(best['jacobian_logrho']);sv=np.asarray(best['singular_values'])
    cosines=[np.dot(jac[:,i],jac[:,j])/(np.linalg.norm(jac[:,i])*np.linalg.norm(jac[:,j]))
        for i,j in [(0,2),(3,5)]]
    sensitivity=(f'Отношение наибольшего сингулярного числа к наименьшему равно '
        f'{number(sv[0]/sv[-1],1)}. Косинусы углов между столбцами чувствительности '
        f'фона и жира равны {number(cosines[0],4)} для вдоха и {number(cosines[1],4)} '
        'для выдоха. Близость косинуса к единице означает сходство направлений '
        'изменения уровней импеданса при малых изменениях этих сопротивлений. '
        'Это локальная диагностика при фиксированном положении, без оценки '
        'доверительных интервалов и без проверки всех девяти параметров.')
    target=np.asarray(best['target_Z_ohm']);model=np.asarray(best['predicted_Z_ohm'])
    residual=model-target;labels=observation_labels(data['final'])
    worst=np.argsort(np.abs(residual))[::-1][:3]
    residual_text='Наибольшие абсолютные невязки получены для следующих уровней: '+ '; '.join(
        f'{labels[i]} — измерено {number(target[i],2)} Ом, модель {number(model[i],2)} Ом, '
        f'разность {number(residual[i],2)} Ом' for i in worst)+'. '
    residual_text+=('Это расхождения текущего кандидата с наблюдениями. Они не являются '
        'доказательством дефекта соответствующей записи и не служат основанием её исключения.')
    audit=data['final'];pairs=[]
    for i,state in enumerate(audit['states']):
        if state!='inhale':continue
        matches=[j for j,s in enumerate(audit['states']) if s=='exhale'
            and audit['record_ids'][j]==audit['record_ids'][i] and audit['axes'][j]==audit['axes'][i]]
        if len(matches)!=1:raise ValueError('Ambiguous breathing comparison')
        pairs.append((i,matches[0]))
    measured_change=np.asarray([target[j]-target[i] for i,j in pairs])
    model_change=np.asarray([model[j]-model[i] for i,j in pairs])
    breathing=(f'В эксперименте разность «выдох минус вдох» отрицательна в '
        f'{np.count_nonzero(measured_change<0)} из {len(pairs)} пар; модель даёт положительную '
        f'разность в {np.count_nonzero(model_change>0)} парах. Знаки экспериментального и модельного '
        f'изменений не совпадают в {np.count_nonzero(np.sign(measured_change)!=np.sign(model_change))} парах. '
        'Таким образом, найденный кандидат не согласует полностью даже направление дыхательного '
        'изменения. Это наблюдение относится к текущему ограниченному поиску с неподвижной '
        'геометрией вдоха и не доказывает несовместимости всей модели с экспериментом.')
    if 'parent_work_directory' in data['config']:
        parent=read(PIPE/data['config']['parent_work_directory']/'selected_candidate.local.json')['best']
        if parent['target_Z_ohm']!=best['target_Z_ohm']:
            raise ValueError('Continuation changed target levels or their order')
        old_error=np.asarray(parent['predicted_Z_ohm'])-target
        improvement=100*(np.dot(old_error,old_error)-np.dot(residual,residual))/np.dot(old_error,old_error)
        sentence+=(f' По сравнению с первым лимитом невязка уменьшилась с '
            f'{number(np.sqrt(parent["mean_square_ohm2"]))} до {number(rmse)} Ом '
            f'(сумма квадратов — на {number(improvement,2)} %).')
        if parent['parameters'][:6]==best['parameters'][:6]:
            sentence+=' В выбранном кандидате изменилось только положение; шесть удельных сопротивлений остались прежними.'
    summary=(f'На исправленной модели выполнены {search["evaluations"]} оценок совместного '
        f'подбора. Лучшая достигнутая невязка равна {number(rmse)} Ом. '
        f'Локальные раздельные проверки сетки и тканевых долей '
        f'{"прошли установленный допуск" if data["final"]["separated_comparison_passed"] else "выявили превышения установленного допуска"}; '
        f'отдельное уточнение контактов '
        f'{"прошло проверку" if data["contact_final"]["passed"] else "не прошло проверку"}. ')
    if not coverage['all_actual_contact_faces_refined']:
        summary+='Часть площадок не покрыта сгущением; достаточность сетки в найденном положении остаётся непроверенной. '
    summary+=('Ниже приведены исходные условия, исправления, все сравнения и ограничения '
        'интерпретации найденного кандидата.')
    return (text.replace('<!-- result_sentence -->',sentence)
        .replace('<!-- opening_summary -->',summary)
        .replace('<!-- selected_numerical_sentence -->',numerical)
        .replace('<!-- additional_refinement_sentence -->',extra_text)
        .replace('<!-- sensitivity_sentence -->',sensitivity)
        .replace('<!-- residual_sentence -->',residual_text)
        .replace('<!-- breathing_interpretation -->',breathing))


def build(work=DEFAULT):
    import nbformat as nbf
    from nbconvert import HTMLExporter
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    data=load(work);text=reader_text(data);functions=dict(mesh_table=mesh_table,
        input_table=input_table,
        repair_table=repair_table,contour_image=contour_image,inheritance_table=inheritance_table,
        shape_quality_figure=shape_quality_figure,edge_quality_figure=edge_quality_figure,
        residual_table=residual_table,contact_figure=contact_figure,
        control_rows_table=control_rows_table,
        sensitivity_figure=sensitivity_figure,sensitivity_table=sensitivity_table,
        breathing_contrast_figure=breathing_contrast_figure,
        coarse_section_figure=coarse_section_figure,fine_section_figure=fine_section_figure,
        invalid_contact_table=invalid_contact_table,
        starting_candidates_table=starting_candidates_table,
        coverage_table=coverage_table,
        search_table=search_table,
        fraction_distribution_table=fraction_distribution_table,
        numerical_table=numerical_table,numerical_figure=numerical_figure,
        rho_table=rho_table,fit_figure=fit_figure,history_figure=history_figure,coordinate_table=coordinate_table,
        placement_figure=placement_figure,source_table=source_table,contact_table=contact_table)
    setup='from build_georg_quality_repeat_report import load, '+', '.join(functions)+'\nreport_data = load()'
    if Path(work).resolve()!=DEFAULT.resolve():
        setup='from build_georg_quality_repeat_report import load, PIPE, '+', '.join(functions)+'\nreport_data = load(PIPE / '+repr(Path(work).resolve().relative_to(PIPE).as_posix())+')'
    setup_cell=nbf.v4.new_code_cell(setup);setup_cell.execution_count=1
    cells=[];js=True;setup_added=False
    import re
    chunks=re.split(r'(<!-- \w+ -->)',text)
    shell=InteractiveShell.instance()
    with capture_output(display=True) as captured:result=shell.run_cell(setup,store_history=False)
    if result.error_before_exec or result.error_in_exec:raise RuntimeError('Reader initialization failed')
    for piece in chunks:
        if piece.startswith('<!-- '):
            key=piece[5:-4].strip()
            if key not in functions:continue
            if not setup_added:cells.append(setup_cell);setup_added=True
            if key.endswith('figure'):
                code=f'display({key}(report_data,include_js={js!r}))';js=False
            else:code=f'display({key}(report_data))'
            shell.push({key:functions[key]})
            cell=nbf.v4.new_code_cell('from IPython.display import display\n'+code)
            with capture_output(display=True) as captured:result=shell.run_cell(cell.source,store_history=False)
            if result.error_before_exec or result.error_in_exec:raise RuntimeError('Reader cell failed: '+key)
            cell.execution_count=sum(x.cell_type=='code' for x in cells)+1
            cell.outputs=[nbf.v4.new_output('display_data',data=x.data,metadata=x.metadata) for x in captured.outputs]
            cells.append(cell)
        elif piece.strip():cells.append(nbf.v4.new_markdown_cell(piece.strip()))
    notebook=nbf.v4.new_notebook(cells=cells,metadata=dict(
        kernelspec=dict(name='python3',display_name='Python 3',language='python'),
        execution_mode='exact reader cells executed in-process with IPython; no FEM',
        quantitative_result_accepted=False,selected_candidate_sha256=sha(data['work']/'selected_candidate.local.json')))
    nbf.validate(notebook);nbf.write(notebook,FOLDER/(STEM+'.ipynb'))
    exporter=HTMLExporter();exporter.exclude_input=True;exporter.exclude_input_prompt=True;exporter.exclude_output_prompt=True
    reader,_=exporter.from_notebook_node(notebook)
    reader=reader.replace('</head>','<style>.scientific-table{overflow-x:auto}details{margin:1em 0}'
        'summary{cursor:pointer;font-weight:600}th,td{white-space:normal;vertical-align:top}</style></head>')
    (FOLDER/(STEM+'.html')).write_text(reader,encoding='utf-8')
    (FOLDER/(STEM+'.md')).write_text(text,encoding='utf-8')
    return notebook


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,default=DEFAULT)
    args=p.parse_args();build(args.work);print(STEM)
