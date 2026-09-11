"""Figures and scientific tables for the rho refinement; no FEM is run here."""
from pathlib import Path
import json
import hashlib
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from IPython.display import display, Markdown, Image
from analyze_resistivity_refinement import BASE, PARAMS, read_plan

NAMES={'soft':'Мягкотканный фон','heart':'Сердечная область','lung':'Лёгкие'}
COLORS={'soft':'#0d6a8a','heart':'#b8374d','lung':'#3c8b62'}
KINDS={'point_disc_5mm':'Диски 5 мм','circumferential_ring':'Манжеты 5 мм','wide_cuff_equivalent_area':'Манжеты 15 мм','outer_planes_inner_rings':'Токовые плоскости и измерительные манжеты'}


def label(m):
    bits=m.split('_',2)
    return f'{bits[0][2:]}/{bits[1][3:]} мм; {KINDS[bits[2]]}'


def load_report():
    plan,profile=read_plan()
    result={'plan':plan,'profile':profile,'base':BASE,'fig':BASE/'report/figures'}
    result['fig'].mkdir(parents=True,exist_ok=True)
    summary=BASE/'analysis/refinement_summary.json'
    result['complete']=summary.exists()
    if result['complete']:
        result['summary']=json.loads(summary.read_text(encoding='utf-8'))
        assert result['summary']['plan_sha256']==hashlib.sha256((BASE/'refinement_plan.json').read_bytes()).hexdigest()
        for name,expected in result['summary']['output_sha256'].items():
            assert hashlib.sha256((BASE/'analysis'/name).read_bytes()).hexdigest()==expected, 'Analysis table integrity mismatch: '+name
        for key,file in [('states','states_rho.csv'),('audit','interpolation_checks.csv'),('pairs','pair_checks.csv'),('cards','scorecard_levels.csv'),('stability','scorecard_stability.csv'),('endpoints','endpoint_excursions.csv')]:
            result[key]=pd.read_csv(BASE/'analysis'/file)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':115,'savefig.dpi':170})
    return result


def save(r,fig,number,caption):
    fig.savefig(r['fig']/f'figure_{number:02d}.png',bbox_inches='tight')
    fig.savefig(r['fig']/f'figure_{number:02d}.svg',bbox_inches='tight')
    display(Image(filename=str(r['fig']/f'figure_{number:02d}.png')));plt.close(fig)
    display(Markdown(f'**Рисунок {number}.** {caption}'))


def status(r):
    if not r['complete']:
        display(Markdown('**Полный расчёт ещё выполняется.** В отчёте приведены зафиксированный протокол и структура перебора. Итоговые оценки достаточности и предпочтения сборок появятся только после завершения всех предусмотренных состояний.'))
        path=BASE/'progress.json'
        if path.exists():
            p=json.loads(path.read_text(encoding='utf-8-sig'))
            display(Markdown('Снимок журнала расчёта: '+p.get('updated_local','время не указано')+'. Значения в этой промежуточной копии не обновляются при простом открытии HTML.'))
            display(pd.DataFrame([{'Завершено новых состояний':p['completed_new_states'],'Требуется новых состояний':p['new_states_required'],'Повторно использовано старых узлов':r['plan']['reused_states']}]).style.hide(axis='index'))
        return
    s=r['summary']
    display(Markdown('**Решение по зафиксированным критериям: '+('перебор прошёл предусмотренные проверки.' if s['adequate_on_tested_states'] else 'требуется дальнейшее уточнение перебора.')+'** Это результат численного эксперимента для выбранной модели и диапазонов.'))
    display(pd.DataFrame([{'Условие':k,'Результат':'Выполнено' if v else 'Не выполнено'} for k,v in s['criteria_decisions_ru'].items()]))


def design(r):
    fig,axes=plt.subplots(1,3,figsize=(13,3.1),layout='constrained')
    for ax,k in zip(axes,PARAMS):
        x=np.asarray(r['plan']['axes_rho'][k]);c=(x[:-1]+x[1:])/2
        ax.scatter(x,np.zeros(5),s=65,c=COLORS[k],label='Пять уровней ρ')
        ax.scatter(x[::2],np.zeros(3),s=160,facecolors='none',edgecolors='#222',label='Вложенные три уровня')
        ax.scatter(c,np.ones(4)*.18,marker='x',s=65,c='#c47013',label='Промежуточные уровни')
        ax.set(title=NAMES[k],xlabel='ρ, Ом·м',ylim=(-.16,.4),yticks=[])
        ax.tick_params(axis='x',labelsize=9)
        ax.grid(axis='x',alpha=.2)
    axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.3),ncol=3,fontsize=9)
    save(r,fig,1,'Уровни сопротивления по трём координатам. Все сочетания пяти основных уровней образуют 125 узлов. Все сочетания четырёх промежуточных уровней образуют 64 центра ячеек для независимого контроля интерполяции.')


def assembly_table(r):
    rows=[]
    for i,m in enumerate(r['plan']['montages'],1):
        bits=m.split('_',2)
        rows.append({'№':i,'Электроды':KINDS[bits[2]],'Ориентир — измерительная манжета/диск, мм':int(bits[0][2:]),'Измерительный — токовый электрод, мм':int(bits[1][3:])})
    display(pd.DataFrame(rows).style.hide(axis='index'))


def assembly_map(r,number):
    kinds=['circumferential_ring','outer_planes_inner_rings','point_disc_5mm','wide_cuff_equivalent_area']
    kind_names=['Манжеты 5 мм','Наружные плоскости\n+ внутренние манжеты','Диски 5 мм','Манжеты 15 мм']
    colors={'circumferential_ring':'#3978ad','outer_planes_inner_rings':'#c78031','point_disc_5mm':'#7e65ad','wide_cuff_equivalent_area':'#369380'}
    pale={'circumferential_ring':'#dceaf5','outer_planes_inner_rings':'#f4e4ce','point_disc_5mm':'#e8e0f1','wide_cuff_equivalent_area':'#dcefe9'}
    combinations=[(20,100),(20,80),(40,100),(40,80)]
    matrix=np.zeros((4,4),dtype=int)
    for i,montage in enumerate(r['plan']['montages'],1):
        first,second,kind=montage.split('_',2)
        position=(int(first[2:]),int(second[3:]))
        matrix[combinations.index(position),kinds.index(kind)]=i
    assert np.array_equal(np.sort(matrix.ravel()),np.arange(1,17))

    fig=plt.figure(figsize=(14,11),layout='constrained')
    grid=fig.add_gridspec(5,1,height_ratios=[1,1,1,1,2.5])
    roles=['I+','V+','V−','I−']
    for row,(kind,name) in enumerate(zip(kinds,kind_names)):
        ax=fig.add_subplot(grid[row]);ax.axis('off');ax.set(xlim=(-5,5),ylim=(-.75,1))
        ax.add_patch(Rectangle((-1.05,-.30),2.1,.60,facecolor='#e4e8ed',edgecolor='#9aa7b3'))
        ax.text(0,0,'Туловище',ha='center',va='center')
        for start in (-4.45,1.05):
            ax.add_patch(Rectangle((start,-.11),3.4,.22,facecolor='#dce2e8',edgecolor='none'))
        for j,x in enumerate((-3.9,-2.0,2.0,3.9)):
            if kind=='point_disc_5mm':
                ax.add_patch(Circle((x,.11),.10,color=colors[kind]))
            elif kind=='outer_planes_inner_rings' and j in (0,3):
                ax.plot([x,x],[-.30,.30],lw=5,color=colors[kind],solid_capstyle='round')
            else:
                width=.34 if kind=='wide_cuff_equivalent_area' else .12
                ax.add_patch(Rectangle((x-width/2,-.24),width,.48,color=colors[kind]))
            ax.text(x,.50,roles[j],ha='center',fontweight='bold')
        ax.text(-4.8,.84,name.replace('\n',' '),fontweight='bold',color=colors[kind])
        if row==0:
            ax.annotate('',xy=(2.0,-.39),xytext=(1.05,-.39),arrowprops={'arrowstyle':'<->','color':'#444'})
            ax.text(1.525,-.64,'d: 20 или 40 мм',ha='center',fontsize=9)
            ax.annotate('',xy=(3.9,-.39),xytext=(2.0,-.39),arrowprops={'arrowstyle':'<->','color':'#444'})
            ax.text(2.95,-.64,'l: 80 или 100 мм',ha='center',fontsize=9)

    ax=fig.add_subplot(grid[4]);ax.axis('off')
    row_names=[f'd = {d} мм; l = {l} мм' for d,l in combinations]
    table=ax.table(cellText=[[f'№ {value}' for value in row] for row in matrix],
                   rowLabels=row_names,colLabels=kind_names,cellLoc='center',rowLoc='center',
                   loc='center',bbox=[.16,.02,.82,.93])
    table.auto_set_font_size(False);table.set_fontsize(11)
    for column,kind in enumerate(kinds):
        table[(0,column)].set_facecolor(colors[kind]);table[(0,column)].set_text_props(color='white',weight='bold')
        for row in range(1,5):
            table[(row,column)].set_facecolor(pale[kind]);table[(row,column)].set_text_props(weight='bold',size=14)
    for row in range(1,5):
        table[(row,-1)].set_facecolor('#eef1f4');table[(row,-1)].set_text_props(weight='bold')
    ax.set_title('Как номер сборки связан с типом контакта и расстояниями',fontsize=13,pad=10)
    save(r,fig,number,'Схема четырёх типов контактов и карта номеров 16 сборок. I — наружные токовые электроды, V — внутренние измерительные; d — расстояние от границы туловища до измерительного электрода, l — расстояние от измерительного до наружного токового электрода. Схема симметрична относительно туловища; анатомический масштаб условный.')


def sensitivity_matrix(r,number):
    if not r['complete']:
        return
    kinds=['circumferential_ring','outer_planes_inner_rings','point_disc_5mm','wide_cuff_equivalent_area']
    kind_names=['Манжеты\n5 мм','Наружные плоскости\n+ внутренние манжеты','Диски\n5 мм','Манжеты\n15 мм']
    combinations=[(20,100),(20,80),(40,100),(40,80)]
    row_names=[f'd={d}; l={l} мм' for d,l in combinations]
    cells={}
    numbers={}
    for assembly_number,montage in enumerate(r['plan']['montages'],1):
        first,second,kind=montage.split('_',2)
        cell=(combinations.index((int(first[2:]),int(second[3:]))),kinds.index(kind))
        cells[montage]=cell
        numbers[cell]=assembly_number
    preferred={(item['compartment'],item['metric']):set(item['augmented_numbers']) for item in r['summary']['preferred_sets']}
    cards=r['cards'][r['cards'].stage=='grid5_plus_checks']
    fig,axes=plt.subplots(2,3,figsize=(17,10.5),layout='constrained')
    metric_rows=[('min_B','Минимальное B',1.0,'Минимальное B, Ом'),('min_Q','Минимальное Q',100.0,'Минимальное Q, %')]
    for plot_row,(field,criterion,factor,colorbar_label) in enumerate(metric_rows):
        for plot_column,compartment in enumerate(PARAMS):
            ax=axes[plot_row,plot_column]
            values=np.full((4,4),np.nan)
            frame=cards[cards.compartment==compartment]
            for _,item in frame.iterrows():
                values[cells[item.montage]]=float(item[field])*factor
            low=float(np.nanmin(values));high=float(np.nanmax(values))
            if np.isclose(low,high):
                image=ax.imshow(values,cmap='viridis',vmin=low-1,vmax=high+1)
            else:
                image=ax.imshow(values,cmap='viridis',vmin=low,vmax=high)
            leaders=preferred[(compartment,criterion)]
            for matrix_row in range(4):
                for matrix_column in range(4):
                    assembly_number=numbers[(matrix_row,matrix_column)]
                    value=values[matrix_row,matrix_column]
                    relative=0.5 if np.isclose(low,high) else (value-low)/(high-low)
                    if field=='min_B':
                        digits=1 if high>=100 else 3 if high>=1 else 6
                    else:
                        digits=2 if high>=10 else 4
                    shown=f'{value:.{digits}f}'.replace('.',',')
                    prefix='★ ' if assembly_number in leaders else ''
                    ax.text(matrix_column,matrix_row,f'{prefix}№ {assembly_number}\n{shown}',
                            ha='center',va='center',fontsize=9.5,fontweight='bold',
                            color='black' if relative>.58 else 'white')
                    if assembly_number in leaders:
                        ax.add_patch(Rectangle((matrix_column-.47,matrix_row-.47),.94,.94,
                                               fill=False,edgecolor='#00e5ff',linewidth=3))
            ax.set_xticks(range(4),kind_names,fontsize=8)
            ax.set_yticks(range(4),row_names,fontsize=9)
            ax.tick_params(length=0)
            if field=='min_B':
                spread=100*(high-low)/high
                subtitle=f'разброс между сборками {spread:.3f} %'.replace('.',',')
            else:
                subtitle=f'диапазон {low:.4f}–{high:.4f} %'.replace('.',',')
            ax.set_title(f'{NAMES[compartment]}\n{colorbar_label}; {subtitle}',fontsize=11)
            bar=fig.colorbar(image,ax=ax,fraction=.046,pad=.025)
            bar.set_label(colorbar_label,fontsize=8)
            bar.ax.tick_params(labelsize=8)
    fig.suptitle('Цветовые матрицы чувствительности: ярче — больше минимальный целевой показатель',fontsize=15,fontweight='bold')
    fig.text(.5,.005,'★ и бирюзовая рамка — предпочтительное множество по заранее заданной полосе численной неразличимости. Цветовая шкала своя для каждой панели; сравнивать оттенки между панелями нельзя.',ha='center',fontsize=10)
    save(r,fig,number,'Минимальные значения локально масштабированной чувствительности B и избирательности Q для 16 сборок. Расположение ячеек повторяет карту рисунка 2. Яркий цвет соответствует большему минимальному значению показателя; звезда и бирюзовая рамка отмечают предпочтительное множество. Каждая панель имеет собственную числовую цветовую шкалу, поэтому оттенки сопоставимы только внутри одной панели.')



def sensitivity_metric_explanation(r,number):
    if not r['complete']:
        return
    cards=r['cards'][r['cards'].stage=='grid5_plus_checks'].copy()
    parsed=cards.montage.str.extract(r'in(?P<d>20|40)_out(?P<l>80|100)_(?P<kind>.+)')
    cards=pd.concat([cards,parsed],axis=1)
    cards[['d','l']]=cards[['d','l']].astype(int)

    effects={}
    for metric in ('min_B','min_Q'):
        values=[]
        for compartment in PARAMS:
            subset=cards[cards.compartment==compartment].set_index(['d','l','kind'])[metric]
            wide=subset.xs('wide_cuff_equivalent_area',level='kind')
            narrow=subset.xs('circumferential_ring',level='kind')
            values.append(float((100*(wide-narrow)/narrow).median()))
        effects[metric]=np.asarray(values)

    example_rho={'soft':float(r['plan']['axes_rho']['soft'][0]),'heart':float(r['plan']['axes_rho']['heart'][-1]),'lung':float(r['plan']['axes_rho']['lung'][-1])}
    selected={}
    for key,montage in [('Манжета 5 мм','in20_out100_circumferential_ring'),
                        ('Манжета 15 мм','in20_out100_wide_cuff_equivalent_area')]:
        frame=r['states'][r['states'].montage==montage]
        mask=np.ones(len(frame),dtype=bool)
        for compartment in PARAMS:
            mask &= np.isclose(frame['rho_'+compartment].to_numpy(),example_rho[compartment])
        row=frame.loc[mask]
        assert len(row)==1
        selected[key]=row.iloc[0]

    fig=plt.figure(figsize=(16,8.5),layout='constrained')
    grid=fig.add_gridspec(2,2,height_ratios=[.72,1.8])
    formula=fig.add_subplot(grid[0,:]);formula.axis('off')
    formula.text(.01,.86,'(а) Определения показателей',fontsize=13,fontweight='bold',va='top')
    formula.text(.03,.54,r'$S_t=\partial Z/\partial\rho_t$',fontsize=19,color='#273746')
    formula.text(.31,.54,r'$B_t=|S_t|\,\Delta\rho_t$',fontsize=19,color='#273746')
    formula.text(.61,.54,r'$Q_t=\dfrac{B_t}{B_{soft}+B_{heart}+B_{lung}}$',fontsize=19,color='#273746')
    formula.text(.03,.18,'локальный наклон, м⁻¹',fontsize=11)
    formula.text(.31,.18,'ненормированное масштабированное влияние, Ом',fontsize=11)
    formula.text(.61,.18,'нормированная избирательность, безразмерная величина',fontsize=11)

    effect_ax=fig.add_subplot(grid[1,0])
    x=np.arange(3);width=.34
    bars_b=effect_ax.bar(x-width/2,effects['min_B'],width,label='Изменение минимального B',color='#3f7cac')
    bars_q=effect_ax.bar(x+width/2,effects['min_Q'],width,label='Изменение минимального Q',color='#d9882f')
    effect_ax.axhline(0,color='#444',lw=.8)
    effect_ax.set_xticks(x,[NAMES[k].replace(' область','').replace(' фон','') for k in PARAMS])
    effect_ax.set_ylabel('Относительное изменение при 15 мм вместо 5 мм, %')
    effect_ax.set_title('(б) Медиана по четырём согласованным парам d, l')
    effect_ax.set_ylim(-6.8,6.4);effect_ax.grid(axis='y',alpha=.2);effect_ax.legend(fontsize=9)
    for bars in (bars_b,bars_q):
        for bar,value in zip(bars,bars.datavalues):
            label=f'{value:+.4f}' if abs(value)<.1 else f'{value:+.2f}'
            y=value+(.20 if value>=0 else -.34)
            effect_ax.text(bar.get_x()+bar.get_width()/2,y,label.replace('.',','),ha='center',
                           va='bottom' if value>=0 else 'top',fontsize=9,fontweight='bold')

    composition=fig.add_subplot(grid[1,1])
    names=list(selected)
    y=np.arange(2)
    left=np.zeros(2)
    for compartment in PARAMS:
        values=np.asarray([100*float(selected[name]['Q_'+compartment]) for name in names])
        composition.barh(y,values,left=left,color=COLORS[compartment],label=NAMES[compartment])
        left += values
    composition.set_yticks(y,names)
    composition.invert_yaxis()
    composition.set_xlim(0,100)
    composition.set_xlabel('Q, %; сумма трёх компартментов равна 100 %')
    composition.set_title('(в) Состав Q в состоянии минимального Q')
    composition.grid(axis='x',alpha=.2)
    composition.legend(loc='lower center',bbox_to_anchor=(.5,-.25),ncol=3,fontsize=9)
    narrow=selected['Манжета 5 мм'];wide=selected['Манжета 15 мм']
    composition.text(.01,-.31,
        ('Сердце: '+f"{100*narrow.Q_heart:.4f} → {100*wide.Q_heart:.4f} %;  "
         'лёгкие: '+f"{100*narrow.Q_lung:.4f} → {100*wide.Q_lung:.4f} %").replace('.',','),
        transform=composition.transAxes,fontsize=10)
    composition.text(.01,-.40,
        ('Сумма B: '+f"{narrow.B_soft+narrow.B_heart+narrow.B_lung:.3f} → "
         f"{wide.B_soft+wide.B_heart+wide.B_lung:.3f} Ом").replace('.',','),
        transform=composition.transAxes,fontsize=10)
    zoom=composition.inset_axes([.55,.12,.42,.48])
    target_x=np.arange(2);target_width=.34
    narrow_q=100*np.asarray([float(narrow.Q_heart),float(narrow.Q_lung)])
    wide_q=100*np.asarray([float(wide.Q_heart),float(wide.Q_lung)])
    zoom.bar(target_x-target_width/2,narrow_q,target_width,color='#8797a5',label='5 мм')
    zoom.bar(target_x+target_width/2,wide_q,target_width,color='#d9882f',label='15 мм')
    zoom.set_xticks(target_x,['Сердце','Лёгкие'],fontsize=8)
    zoom.set_ylabel('Q, %',fontsize=8);zoom.tick_params(axis='y',labelsize=8)
    zoom.set_ylim(0,max(wide_q)*1.25);zoom.grid(axis='y',alpha=.2)
    zoom.set_title('Увеличенный фрагмент',fontsize=9)
    zoom.legend(fontsize=7,ncol=2,loc='upper left')

    fig.suptitle('Почему избирательность может расти без увеличения целевого абсолютного влияния',fontsize=15,fontweight='bold')
    save(r,fig,number,'Связь производной S, локально масштабированного абсолютного влияния B и относительной избирательности Q. Панель б показывает медианное изменение минимальных B и Q при замене манжеты 5 мм на манжету 15 мм в четырёх согласованных сочетаниях d и l. Панель в показывает состав Q в узле, где для согласованной пары с d = 20 мм и l = 100 мм достигаются минимальные сердечное и лёгочное Q; итоговый критерий по каждой сборке определяется по всей параметрической области.')


def error_maps(r,which,number):
    if not r['complete']: return
    frame=r['audit'] if which=='derivative' else r['pairs']
    field='S_error_over_tolerance' if which=='derivative' else 'S_error_over_tolerance'
    data=frame.groupby(['check_index','compartment'])[field].max().unstack('compartment').reindex(columns=PARAMS)
    centres={k:(np.asarray(r['plan']['axes_rho'][k][:-1])+np.asarray(r['plan']['axes_rho'][k][1:]))/2 for k in PARAMS}
    fig,axes=plt.subplots(3,4,figsize=(14,9),layout='constrained')
    maxval=max(1.,float(data.max().max()))
    for j,k in enumerate(PARAMS):
        cube=data[k].to_numpy().reshape(4,4,4)
        for z in range(4):
            ax=axes[j,z]
            im=ax.imshow(cube[:,:,z].T,origin='lower',vmin=0,vmax=maxval,cmap='magma',aspect='auto')
            ax.set_xticks(range(4),[f'{x:.2f}' for x in centres['soft']]);ax.set_yticks(range(4),[f'{x:.2f}' for x in centres['heart']])
            ax.set_xlabel('ρ мягких тканей, Ом·м');ax.set_ylabel('ρ сердца, Ом·м')
            ax.set_title(f'{NAMES[k]}\nρ лёгких = {centres["lung"][z]:.2f} Ом·м',fontsize=10)
            for x in range(4):
                for y in range(4): ax.text(x,y,f'{cube[x,y,z]:.2f}',ha='center',va='center',fontsize=8,color='white' if cube[x,y,z]<maxval*.65 else 'black')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Модуль ошибки / допустимая ошибка',shrink=.75)
    subject='производной, максимальное среди 16 сборок' if which=='derivative' else 'парной разности модулей производных, максимальное среди 120 пар'
    save(r,fig,number,f'Отношение ошибки к допуску для {subject}. Значение больше единицы означает невыполнение критерия. Каждый столбец соответствует одному промежуточному сопротивлению лёгких; внутри панели изменяются сопротивления мягких тканей и сердца.')


def derivative_ranges(r,number=2):
    if not r['complete']: return
    fig,axes=plt.subplots(1,3,figsize=(13,6.5),layout='constrained')
    states=r['states'];order=r['plan']['montages']
    for ax,k in zip(axes,PARAMS):
        for i,m in enumerate(order):
            values=states[states.montage==m]['S_'+k]
            ax.plot([values.min(),values.max()],[i+1,i+1],lw=3,c=COLORS[k]);ax.scatter([values.min(),values.max()],[i+1,i+1],c=COLORS[k],s=17)
        ax.set(title=NAMES[k],xlabel='S = ∂Z/∂ρ, м⁻¹',yticks=np.arange(1,17),ylabel='Номер сборки');ax.grid(alpha=.2)
    save(r,fig,number,'Знаковые диапазоны производных по всем 125 основным и 64 контрольным состояниям каждой сборки. Концы отрезков обозначают найденные выборочные экстремумы; непрерывные экстремумы по всему диапазону не предполагаются известными.')


def minima(r,metric,number):
    if not r['complete']: return
    fig,axes=plt.subplots(1,3,figsize=(13,7),layout='constrained')
    stage_styles=[('grid3','3³','#b2bbc5',-.20),('grid5','5³','#196b9e',0),('grid5_plus_checks','5³ и контроль','#d47727',.20)]
    factor=100 if metric=='min_Q' else 1
    for ax,k in zip(axes,PARAMS):
        for stage,lab,c,offset in stage_styles:
            frame=r['cards'][(r['cards'].stage==stage)&(r['cards'].compartment==k)].set_index('montage').reindex(r['plan']['montages'])
            ax.scatter(frame[metric]*factor,np.arange(1,17)+offset,c=c,s=27,label=lab)
        ax.set(title=NAMES[k],xlabel='Минимальное Q, %' if metric=='min_Q' else 'Минимальное B, Ом',ylabel='Номер сборки',yticks=np.arange(1,17));ax.grid(alpha=.2)
    axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=3)
    name='избирательности Q' if metric=='min_Q' else 'целевого масштабированного влияния B'
    save(r,fig,number,f'Изменение минимальных значений {name} при увеличении числа состояний. Минимум определяется отдельно для каждой сборки и компартмента. Сравниваются вложенные 27 узлов, полный набор 125 узлов и его объединение с 64 независимыми контрольными состояниями.')


def pair_matrix(r,number):
    if not r['complete']: return
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    order=r['plan']['montages'];idx={m:i for i,m in enumerate(order)}
    data=r['pairs'].groupby(['assembly_a','assembly_b','compartment']).S_error_over_tolerance.max()
    vmax=max(1.,float(data.max()))
    for ax,k in zip(axes,PARAMS):
        matrix=np.full((16,16),np.nan)
        for (a,b,comp),v in data.items():
            if comp==k:matrix[idx[a],idx[b]]=matrix[idx[b],idx[a]]=v
        im=ax.imshow(matrix,vmin=0,vmax=vmax,cmap='magma',origin='lower')
        ax.set(title=NAMES[k],xlabel='Номер сборки',ylabel='Номер сборки',xticks=np.arange(0,16,2),yticks=np.arange(0,16,2))
        ax.set_xticklabels(np.arange(1,17,2));ax.set_yticklabels(np.arange(1,17,2))
    fig.colorbar(im,ax=axes,label='Наибольшее отношение ошибки разности к допуску',shrink=.8)
    save(r,fig,number,'Контроль межсборочных различий: для каждой пары показано наибольшее отношение ошибки к допуску среди 64 контрольных состояний. Диагональ не содержит сравнения. Ошибки двух сборок проверяются совместно, поэтому их возможная компенсация сохранена.')


def selection(r):
    if not r['complete']: return
    rows=[]
    for item in r['summary']['preferred_sets']:
        rows.append({'Цель':NAMES[item['compartment']],'Критерий':item['metric'],'3³':', '.join(str(x) for x in item['grid3_numbers']),'5³':', '.join(str(x) for x in item['grid5_numbers']),'5³ и контроль':', '.join(str(x) for x in item['augmented_numbers']),'Множество устойчиво':'Да' if item['stable'] else 'Нет'})
    display(pd.DataFrame(rows).style.hide(axis='index'))
    display(Markdown('Числа обозначают сборки из таблицы геометрии. Множества формируются с учётом заранее заданной численной полосы. Несколько номеров в ячейке означают отсутствие единственного предпочтительного кандидата по данному критерию на соответствующем этапе.'))


def error_table(r):
    if not r['complete']: return
    rows=[]
    for k in PARAMS:
        a=r['audit'][r['audit'].compartment==k];p=r['pairs'][r['pairs'].compartment==k]
        rows.append({'Компартмент':NAMES[k],'Не прошли проверки S':int((~a.S_pass).sum()),'Макс. ошибка S, %':100*a.S_relative_error.max(),'Макс. ошибка Q, п.п.':100*a.Q_absolute_error.max(),'Не прошли парные проверки |S|':int((~p.S_accuracy_pass).sum()),'Не прошли парные проверки Q':int((~p.Q_accuracy_pass).sum())})
    display(pd.DataFrame(rows).style.hide(axis='index').format(precision=4))


def excursions(r,number=8):
    if not r['complete']: return
    fig,axes=plt.subplots(1,3,figsize=(13,6.5),layout='constrained')
    for ax,k in zip(axes,PARAMS):
        for i,m in enumerate(r['plan']['montages'],1):
            e=r['endpoints'][(r['endpoints'].montage==m)&(r['endpoints'].compartment==k)].excursion_ohm
            ax.plot([e.min(),e.max()],[i,i],lw=3,c=COLORS[k])
        ax.set(title=NAMES[k],xlabel='Модуль изменения Z между границами ρ, Ом',ylabel='Номер сборки',yticks=np.arange(1,17));ax.grid(alpha=.2)
    save(r,fig,number,'Конечные изменения импеданса при переходе между границами сопротивления одного компартмента. Для каждого целевого параметра перебраны 25 комбинаций двух остальных сопротивлений. Эти широкие модельные изменения не интерпретируются как амплитуды пульса или дыхания.')
