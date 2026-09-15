"""Reader tables and plots from saved, explicitly bounded geometry experiments."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REGIONS = {
    'whole_heart': 'Целое сердце', 'blood_pool': 'Объединённая область крови',
    'heart_ventricle_left': 'Левый желудочек', 'heart_ventricle_right': 'Правый желудочек',
    'heart_atrium_left': 'Левое предсердие', 'heart_atrium_right': 'Правое предсердие',
}
MODELS = {'sphere': 'Сфера Г1', 'ellipsoid_volume_scaled': 'Эллипсоид Л1',
          'individual_isotropic': 'Индивидуальная форма'}
TEMPORAL = {'constant': 'Постоянное значение', 'linear_neighbors': 'Интерполяция между соседними фазами',
            'one_harmonic': 'Одна гармоника'}


def load_results(directory):
    directory = Path(directory)
    status = json.loads((directory / 'run_status.json').read_text(encoding='utf-8'))
    if status['status'] != 'completed':
        raise ValueError('Geometry calculation is not complete')
    rows = [json.loads(line) for line in (directory / 'geometry_comparison.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(rows) != status['completed_phase_region_pairs']:
        raise ValueError('Incomplete geometry records')
    return rows


def _number(x, decimals=3):
    return f'{float(x):.{decimals}f}'.replace('.', ',')


def _table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                       '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                       *['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows]])


def geometry_table(records):
    rows = []
    for region, label in REGIONS.items():
        selected = [r for r in records if r['region'] == region and not r['is_reference_phase']]
        medians = [np.median([r['comparisons'][m]['overlap']['dice_voxel_center_approx'] for r in selected]) for m in MODELS]
        rows.append([label, len(selected), *[_number(v) for v in medians]])
    return _table(['Область', 'Число фаз', *MODELS.values()], rows)


def overlap_figure(records):
    values = np.array([[np.median([r['comparisons'][model]['overlap']['dice_voxel_center_approx']
        for r in records if r['region'] == region and not r['is_reference_phase']])
        for model in MODELS] for region in REGIONS])
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(values, vmin=0, vmax=1, cmap='viridis', aspect='auto')
    ax.set_xticks(range(3), ['Сфера Г1\nравный объём', 'Эллипсоид Л1\nмоменты и объём', 'Индивидуальная форма\nмасштаб и перенос'])
    ax.set_yticks(range(6), REGIONS.values())
    for i in range(6):
        for j in range(3):
            ax.text(j, i, _number(values[i,j]), ha='center', va='center', color='black' if values[i,j]>.65 else 'white')
    fig.colorbar(im, ax=ax, label='Медиана коэффициента Dice')
    fig.tight_layout()
    return fig


def paired_figure(records):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, model in zip(axes, ['ellipsoid_volume_scaled', 'individual_isotropic']):
        for i, region in enumerate(REGIONS):
            for s, subject in enumerate(['adam', 'nix', 'georg']):
                selected = [r for r in records if r['region']==region and r['subject']==subject and not r['is_reference_phase']]
                delta = [r['comparisons'][model]['overlap']['dice_voxel_center_approx'] -
                         r['comparisons']['sphere']['overlap']['dice_voxel_center_approx'] for r in selected]
                ax.scatter(delta, np.full(len(delta), i+(s-1)*.18), s=17, alpha=.7,
                           color=['#2674ac','#bd4e2c','#498c52'][s], label=subject.capitalize() if i==0 else None)
        ax.axvline(0, color='black', lw=.8)
        ax.set_title(MODELS[model])
        ax.set_xlabel('Разность Dice относительно сферы Г1')
        ax.grid(axis='x', alpha=.25)
    axes[0].set_yticks(range(6), REGIONS.values())
    axes[0].invert_yaxis()
    fig.legend(*axes[1].get_legend_handles_labels(), loc='lower center', ncol=3, frameon=False)
    fig.tight_layout(rect=[0, .07, 1, 1])
    return fig


def qc_table(records):
    rows=[]
    for region, label in REGIONS.items():
        counts=[]
        for subject in ['adam','nix','georg']:
            selected=[r for r in records if r['region']==region and r['subject']==subject]
            counts.append(f"{sum(r['provenance']['touches_source_image_boundary'] for r in selected)} / {len(selected)}")
        rows.append([label,*counts])
    return _table(['Область','Adam','Nix','Georg'], rows)


def temporal_table(result):
    rows=[]
    # Compare methods on exactly the same held-out states, including harmonic eligibility.
    common={(r['subject'],r['cycle_index'],r['region'],r['phase_id']) for r in result['predictions'] if r['method']=='one_harmonic'}
    for region, label in REGIONS.items():
        for method, method_label in TEMPORAL.items():
            selected=[r for r in result['predictions'] if r['region']==region and r['method']==method
                      and (r['subject'],r['cycle_index'],r['region'],r['phase_id']) in common]
            if not selected:
                continue
            ev=[abs(r['volume_error_percent']) for r in selected]
            ec=[r['centroid_error_mm'] for r in selected]
            rows.append([label, method_label, len(selected), _number(np.median(ev),2),
                         _number(max(ev),2),_number(np.median(ec),2)])
    return _table(['Область','Способ','Число фаз','Медиана модуля относительной ошибки объёма, %','Максимум модуля относительной ошибки объёма, %',
                   'Медиана расстояния между центрами, мм'], rows)


def temporal_figure(result):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for ax, (region,label) in zip(axes.flat,REGIONS.items()):
        keys={(r['subject'],r['cycle_index'],r['phase_id']) for r in result['predictions'] if r['region']==region and r['method']=='one_harmonic'}
        values=[[abs(r['volume_error_percent']) for r in result['predictions'] if r['region']==region and r['method']==method
                and (r['subject'],r['cycle_index'],r['phase_id']) in keys] for method in TEMPORAL]
        ax.boxplot(values, whis=1.5, tick_labels=['Постоянное\nзначение','Линейная\nинтерполяция','Гармоника'], showmeans=False)
        ax.set_title(label)
        ax.set_ylabel('Модуль относительной ошибки объёма, %')
        ax.grid(axis='y',alpha=.25)
    fig.tight_layout()
    return fig


def coverage_figure(records):
    fig, ax=plt.subplots(figsize=(10,4.5))
    groups=[]
    for subject in ['adam','nix','georg']:
        for cycle in sorted({r['cycle_index'] for r in records if r['subject']==subject}):
            rows=[r for r in records if r['subject']==subject and r['cycle_index']==cycle and r['region']=='whole_heart']
            rows.sort(key=lambda r:r['phase_percent_within_cycle'])
            groups.append((subject,cycle,rows))
    for i,(subject,cycle,rows) in enumerate(groups):
        x=[r['phase_percent_within_cycle'] for r in rows]
        ax.plot(x,np.full(len(x),i),'o-',color=['#2674ac','#bd4e2c','#498c52'][['adam','nix','georg'].index(subject)])
    ax.set_yticks(range(len(groups)),[f'{s.capitalize()}, цикл {c+1}' for s,c,_ in groups])
    ax.invert_yaxis();ax.set_xlim(0,100)
    ax.set_xlabel('Положение фазы внутри исходного R–R-интервала, %')
    ax.grid(axis='x',alpha=.25);fig.tight_layout()
    return fig


def discretization_figure(records):
    fig,ax=plt.subplots(figsize=(9,4))
    values=[[100*(r['comparisons'][m]['overlap']['candidate_sampled_volume_ml']/r['moments']['volume_ml']-1)
             for r in records if not r['is_reference_phase']] for m in MODELS]
    ax.boxplot(values,whis=1.5,tick_labels=['Сфера','Эллипсоид','Индивидуальная форма'])
    ax.axhline(0,color='black',lw=.8);ax.set_ylabel('Отклонение дискретного объёма от заданного, %', fontsize=8)
    ax.grid(axis='y',alpha=.25);fig.tight_layout()
    return fig


def affine_table(records, affine):
    lookup={(r['subject'],r['cycle_index'],r['phase_id'],r['region']):r for r in records}
    rows=[]
    for region,label in REGIONS.items():
        chosen=[r for r in affine if r['region']==region and not r['is_reference_phase']]
        basic=[lookup[(r['subject'],r['cycle_index'],r['phase_id'],r['region'])]['comparisons']['individual_isotropic']['overlap']['dice_voxel_center_approx'] for r in chosen]
        rotated=[r['comparisons']['individual_rotation']['overlap']['dice_voxel_center_approx'] for r in chosen]
        transformed=[r['comparisons']['individual_affine']['overlap']['dice_voxel_center_approx'] for r in chosen]
        delta=np.asarray(transformed)-basic
        rows.append([label,*[_number(np.median(v)) for v in (basic,rotated,transformed)],
                     _number(np.median(delta)),f'{sum(delta>0)} / {len(delta)}'])
    return _table(['Область','Равномерный масштаб и перенос','Равномерный масштаб, поворот и перенос','Неравномерный масштаб, поворот и перенос',
                   'Медиана парного изменения Dice','Число фаз с улучшением при неравномерном масштабе'],rows)


def affine_figure(records, affine):
    lookup={(r['subject'],r['cycle_index'],r['phase_id'],r['region']):r for r in records}
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,model,title in zip(axes,['individual_rotation','individual_affine'],['Равномерный масштаб и поворот','Неравномерный масштаб и поворот']):
        for i,region in enumerate(REGIONS):
            for j,subject in enumerate(['adam','nix','georg']):
                chosen=[r for r in affine if r['region']==region and r['subject']==subject and not r['is_reference_phase']]
                delta=[]
                for r in chosen:
                    base=lookup[(r['subject'],r['cycle_index'],r['phase_id'],r['region'])]['comparisons']['individual_isotropic']['overlap']['dice_voxel_center_approx']
                    delta.append(r['comparisons'][model]['overlap']['dice_voxel_center_approx']-base)
                ax.scatter(delta,np.full(len(delta),i+(j-1)*.18),s=17,alpha=.7,
                           color=['#2674ac','#bd4e2c','#498c52'][j],label=subject.capitalize() if i==0 else None)
        ax.axvline(0,color='black',lw=.8);ax.set_title(title)
        ax.set_xlabel('Изменение Dice относительно равномерного масштаба и переноса', fontsize=8);ax.grid(axis='x',alpha=.25)
    axes[0].set_yticks(range(6),REGIONS.values());axes[0].invert_yaxis()
    fig.legend(*axes[1].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False)
    fig.tight_layout(rect=[0,.07,1,1])
    return fig


def overlay_schematic():
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    x,y=np.meshgrid(np.linspace(-1.1,1.1,350),np.linspace(-1.1,1.1,350))
    body=(x/.98)**2+(y/1.03)**2<=1
    lung=(((x-.42)/.28)**2+(y/.76)**2<=1)|(((x+.42)/.28)**2+(y/.76)**2<=1)
    bone=(x/.075)**2+((y+.63)/.16)**2<=1
    background=np.zeros(x.shape,dtype=int)
    background[body]=1;background[body&lung]=2;background[body&bone]=3
    first=((x+.12)/.43)**2+((y-.06)/.42)**2<=1
    second=((x-.20)/.36)**2+((y-.05)/.36)**2<=1
    states=[background.copy(),background.copy(),background.copy()]
    states[1][first&body]=4;states[2][second&body]=4
    fig,axes=plt.subplots(1,3,figsize=(11,4.4))
    cmap=ListedColormap(['white','#f0ddc1','#7dc4dd','#7d828b','#b73c62'])
    for ax,state,title in zip(axes,states,['Неподвижный фон без сердца','Сердце: положение 1','Сердце: положение 2']):
        ax.imshow(state,origin='lower',extent=[-1.1,1.1,-1.1,1.1],cmap=cmap,vmin=0,vmax=4,interpolation='nearest')
        ax.set_title(title,fontsize=11);ax.set_axis_off()
    axes[2].contour(x,y,first.astype(float),levels=[.5],colors=['#222222'],linestyles='--',linewidths=1)
    fig.legend(handles=[Patch(color=color,label=label) for color,label in
        [('#f0ddc1','Мягкие ткани'),('#7dc4dd','Лёгкие'),('#7d828b','Кость'),('#b73c62','Сердце')]],
        loc='lower center',ncol=4,frameon=False)
    fig.suptitle('Схема принятого правила; не индивидуальная анатомия',fontsize=11,y=.99)
    fig.tight_layout(rect=[0,.09,1,.95])
    return fig


def construction_table(result):
    names = {'sphere':'Г1 · объём', 'sphere_second_moment':'Г2 · второй момент',
        'sphere_surface_fixed_center':'Г3а · поверхность, центр фиксирован',
        'sphere_surface_free_center':'Г3б · поверхность, центр подбирается',
        'ellipsoid_volume_scaled':'Л1 · моменты и объём', 'ellipsoid_raw_moment':'Л2 · моменты'}
    rows=[]
    for ex in result['examples']:
        for method, label in names.items():
            data=ex['methods'][method];p=data['parameters']
            size = _number(p['radius_mm'],2) if 'radius_mm' in p else ' / '.join(_number(x,2) for x in p['semi_axes_mm'])
            rows.append([ex['subject'].capitalize(),label,size,
                _number(round(data['analytic_volume_error_percent'],2) or 0,2),_number(data['center_shift_mm'],2),
                _number(data['overlap']['dice_voxel_center_approx'],3),
                _number(p['surface_RMS_mm'],2) if 'surface_RMS_mm' in p else '—'])
    return _table(['Участник','Способ','Радиус либо три полуоси, мм',
        'Разность объёмов, %','Смещение центра, мм','Dice','Среднеквадратичная ошибка поверхности, мм'],rows)
