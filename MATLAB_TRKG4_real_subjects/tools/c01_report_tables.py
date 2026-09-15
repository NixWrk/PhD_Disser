"""Reader tables derived from the unified C01 baseline outputs."""
import json
import numpy as np
import pandas as pd
from IPython.display import display,HTML,Markdown
from c01_baseline_study import ROOT,SOURCE,HALF,OUT
from c01_baseline_analysis import LABELS,CONTACTS,checked
from c01_sensitivity_figure import sensitivity_figure
from c01_model_catalog import model_catalog_figure, model_catalog_table, model_scheme_figure

def table(d):
    d=d.copy()
    for c in d:
        if d[c].dtype==bool:d[c]=d[c].map({True:'да',False:'нет'})
    display(HTML(d.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}'.replace('.',','))))

def synthetic_frame():
    raw=pd.read_csv(OUT/'inverse_recovery.csv');rows=[]
    for name in LABELS:
        t=raw[raw.model==name]
        rows.append([LABELS[name],abs(t.error1_pct).median(),abs(t.error1_pct).max(),abs(t.error2_pct).median(),abs(t.error2_pct).max(),int(t.rho2_at_bound.sum())])
    return pd.DataFrame(rows,columns=['Модель','ρ₁: медиана |ошибки|, %','ρ₁: максимум, %','ρ₂: медиана |ошибки|, %','ρ₂: максимум, %','ρ₂ на границе, из 65'])

def synthetic(subset=None):
    d=synthetic_frame()
    if subset is not None:d=d[d['Модель'].isin([LABELS[k] for k in subset])]
    table(d)

def experimental():
    d=pd.read_csv(OUT/'experimental_fits.csv');hold=pd.read_csv(OUT/'conditional_holdouts.csv');rows=[]
    for row in d.itertuples():
        t=hold[hold.model==row.model]
        rows.append([LABELS[row.model],row.rho1,row.rho2,row.train_rmse,np.sqrt(np.mean(t.error_ohm**2)),bool(row.rho2_at_bound)])
    c=json.loads((OUT/'cem_fit_summary.json').read_text('utf-8'))['best']
    rows.append(['КТ/FEM с конечными электродами',c['rho1'],c['rho2'],c['rmse_ohm'],np.nan,bool(np.isclose(c['rho2'],checked()['bounds_ohm_m'][1]).any())])
    d=pd.DataFrame(rows,columns=['Модель','ρ₁, Ом·м','ρ₂, Ом·м','RMSE, Ом','Прогноз исключённого размера: RMSE, Ом','ρ₂ на границе'])
    table(d.fillna('не проверено'))

def synthesis():
    raw=pd.read_csv(OUT/'inverse_recovery.csv');best=raw[raw.model=='variable_transverse'];common=raw[raw.model=='uniform_transverse']
    candidates=raw[raw.model!='reference'].groupby('model').error2_pct.agg(lambda x:abs(x).median());winner=candidates.idxmin()
    def f(x):return f'{x:.2f}'.replace('.',',')
    display(Markdown(f"**Результат общего теста.** Наименьшая медиана абсолютной ошибки ρ₂ среди приближений получена для модели «{LABELS[winner]}»: **{f(candidates[winner])}%**. Для переменной поперечной формы медиана равна {f(abs(best.error2_pct).median())}%, максимум — {f(abs(best.error2_pct).max())}%; для общей кривизны — {f(abs(common.error2_pct).median())}% и {f(abs(common.error2_pct).max())}% соответственно. Это ранжирование по заданному критерию на 65 синтетических состояниях; оно не устанавливает лучший результат для каждой пары сопротивлений или истинные свойства тканей человека."))

def geometry_table():
    g=checked();q=pd.read_csv(OUT/'geometry_mask_qc.csv')
    vols=pd.concat([pd.read_csv(SOURCE/'geometry_volumes.csv').rename(columns={'changed_from_CT_litres':'changed_litres'}),pd.read_csv(OUT/'geometry_volumes.csv')],ignore_index=True)
    d=q.merge(vols[['model','lung_litres']],on='model',validate='one_to_one');d['model']=d.model.map(LABELS)
    d['skin_lung_area_cm2']=d.skin_lung_area_mm2/100
    table(d[['model','lung_litres','central_discrete_entry_d_mm','skin_lung_area_cm2','contacts_adjacent_to_lung']].rename(columns={
      'model':'Модель','lung_litres':'Лёгочная область, л','central_discrete_entry_d_mm':'Первый вход в маску, d, мм',
      'skin_lung_area_cm2':'Лёгкое на внешней границе, см²','contacts_adjacent_to_lung':'Контактов рядом с лёгочными элементами'}))
    display(Markdown('**Параметры по КТ.** Полуоси эллипсоида: '+'; '.join(f'{x:.2f}'.replace('.',',') for x in g['ellipsoid_radii_mm'])+' мм; перенос для привязки центрального входа: '+f"{g['anchoring_shift_mm']:.2f}".replace('.',',')+' мм. Эти величины получены до электрической подгонки.'))

def qc_table():
    g=checked();a=json.loads((OUT/'analysis_summary.json').read_text('utf-8'))
    q=pd.read_csv(OUT/'analytic_quadrature_qc.csv');raw=pd.read_csv(OUT/'pem_library.csv');ref=pd.read_csv(OUT/'reference_holdouts.csv')
    rows=[['Моделей в едином тесте',a['models']],['Заданных пар сопротивлений',a['states']],['Базовых синтетических инверсий',a['inverse_count']],
          ['Максимальная ошибка интерполяции КТ, Ом',a['max_reference_interpolation_ohm']],
          ['Максимальное различие квадратур при ρ₁ = 1 Ом·м, Ом',q.max_quad_difference.max()]]
    for name in ['reciprocity_abs','relative_residual']:
        if name in raw:rows.append(['Максимум '+('нарушения взаимности, Ом при ρ₁ = 1 Ом·м' if name=='reciprocity_abs' else 'относительной невязки FEM'),raw[name].max()])
    table(pd.DataFrame([[label,str(value) if isinstance(value,int) else f'{value:.6g}'.replace('.',',')] for label,value in rows],columns=['Проверка','Значение']))

def finite_table():
    d=pd.read_csv(SOURCE/'cem_inverse.csv').sort_values('rmse_ohm').groupby('model',sort=False).first().reset_index()
    d=d[['model','rho1','rho2','error1_pct','error2_pct','rmse_ohm']];d['model']=d.model.map(LABELS)
    table(d.rename(columns={'model':'Геометрия','rho1':'ρ₁, Ом·м','rho2':'ρ₂, Ом·м','error1_pct':'Ошибка ρ₁, %','error2_pct':'Ошибка ρ₂, %','rmse_ohm':'RMSE, Ом'}))

def h_table():
    d=pd.read_csv(SOURCE/'geometry_control_recovery.csv');d=d[d.model!='grid_control'].copy()
    d['Смещение истинной границы, мм']=d.model.map({'h_minus2':-2,'h_minus1':-1,'h_plus1':1,'h_plus2':2})
    table(d[['Смещение истинной границы, мм','rho1','rho2','error2_pct']].rename(columns={'rho1':'Оценка ρ₁, Ом·м','rho2':'Оценка ρ₂, Ом·м','error2_pct':'Ошибка ρ₂, %'}))

def timing_table():
    t=pd.read_csv(OUT/'library_timings.csv');r=t[t.model=='reference']
    fem=pd.read_csv(HALF/'fem_benchmark.csv');b=pd.read_csv(HALF/'direct_timings.csv')
    values=[['Прямой КТ/FEM: девять Z и производные',fem.factor_solve_derivatives_seconds.median(),'Матрицы объёмов уже подготовлены'],
       ['Прямой расчёт постоянного сечения: Фурье/BEM',b[b.model=='m3h'].seconds.median(),'Граничная матрица уже подготовлена'],
       ['Прямой расчёт переменной формы: BEM',b[b.model=='m4h'].seconds.median(),'Граничная матрица уже подготовлена'],
       ['Библиотека КТ/FEM: девять Z и производные',r[r.task=='nine_Z_and_J_cached'].seconds.median(),'Фиксированные геометрия и точечные электроды'],
       ['Библиотека КТ/FEM: ограниченная инверсия',r[r.task=='bounded_inverse_cached'].seconds.median(),'Оба сопротивления; библиотека готова']]
    table(pd.DataFrame([[x,f'{y:.6g}'.replace('.',','),z] for x,y,z in values],columns=['Операция','Медиана времени, с','Условия']))


def finite_direct_table():
    old=pd.read_csv(SOURCE/'cem_baseline.csv')
    new=pd.read_csv(OUT/'cem_baseline.csv')
    allrows=pd.concat([old,new],ignore_index=True)
    truth=old[old.model=='reference'][['L_mm','Z']].rename(columns={'Z':'reference_Z'})
    checked_rows=allrows.merge(truth,on='L_mm',validate='many_to_one')
    assert not checked_rows.duplicated(['model','L_mm']).any()
    assert checked_rows.groupby('model').size().eq(9).all()
    assert np.isfinite(checked_rows[['Z','reference_Z']]).all().all()
    rows=[]
    for name,t in checked_rows.groupby('model',sort=False):
        e=t.Z-t.reference_Z
        rows.append([LABELS[name],np.sqrt(np.mean(e**2)),(100*abs(e)/abs(t.reference_Z)).max()])
    table(pd.DataFrame(rows,columns=['Геометрия','RMSE относительно КТ/CEM, Ом','Максимум относительного отклонения, %']))

def real_cem_result():
    c=json.loads((OUT/'cem_fit_summary.json').read_text('utf-8'))['best']
    p=pd.read_csv(OUT/'experimental_fits.csv').set_index('model').loc['reference']
    f=lambda x:f'{x:.3f}'.replace('.',',')
    display(Markdown(f"**КТ-модель и представление электрода.** Для точечных контактов RMSE равна {f(p.train_rmse)} Ом; для конечных площадок — {f(c['rmse_ohm'])} Ом при ρ₁ = {f(c['rho1'])} и ρ₂ = {f(c['rho2'])} Ом·м. Оба старта CEM сошлись к практически одинаковому решению. В обоих представлениях ρ₂ достигает верхней рабочей границы. Конечные электроды заметно улучшают согласование абсолютного импеданса, однако правильность оценки лёгочного сопротивления этим не установлена."))
# Functions are imported explicitly by the executed notebook setup.
