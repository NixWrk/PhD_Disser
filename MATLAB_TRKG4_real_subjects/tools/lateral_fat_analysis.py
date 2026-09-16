"""Check and present fixed-montage full-static five-tissue FEM results."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lateral_fat_prepare import PIPE,BASE,OUT as INITIAL_OUT,PREVIOUS,sha,read,write
OUT=INITIAL_OUT/'refined'


def analyze(out=OUT):
    out=Path(out);p=read(out/'plan.json');e=read(out/'execution.json')
    assert e['complete'] and e['rows']==189
    for x in p['sources']:assert sha(PIPE/x['path'])==x['sha256'],x['path']
    assert sha(out/'plan.json')==e['identity']['plan_sha256']
    assert sha(PIPE/'src/run_lateral_fat_forward.m')==e['identity']['worker_sha256']
    assert sha(out/'responses.csv')==e['responses_sha256']
    assert sha(out/'requests.csv')==p['requests_sha256']
    for n,h in p['fractions_sha256'].items():assert sha(out/n)==h,n
    d=pd.read_csv(out/'responses.csv');q=pd.read_csv(out/'requests.csv').set_index('request_id')
    assert len(d)==len(q)*9 and not d.duplicated(['request_id','L_mm']).any()
    assert set(d.request_id)==set(q.index) and np.isfinite(d.select_dtypes('number')).all().all()
    fields=['rho1','rho2','rho_bone','rho_heart','rho_fat','quadrature_points','frequency_hz']
    for rid,g in d.groupby('request_id'):
        assert sorted(g.L_mm)==p['sizes_mm'];assert all(g.model==q.loc[rid,'model'])
        np.testing.assert_allclose(g[fields],np.tile(q.loc[rid,fields].to_numpy(float),(9,1)),rtol=1e-12)
    assert d.relative_residual.max()<1e-7 and d.reciprocity_abs.max()<1e-7
    controls={}
    for kind in ['equal_material_limit','fat_equal_limit']:
        c=d[d.kind==kind].pivot(index='L_mm',columns='model',values='Z')
        controls[kind]=float((c.max(axis=1)-c.min(axis=1)).max());assert controls[kind]<1e-6
    coarse,fine=sorted(d.quadrature_points.unique())
    a=d[(d.kind=='literature')&(d.quadrature_points==coarse)]
    b=d[(d.kind=='literature')&(d.quadrature_points==fine)]
    check=b.merge(a,on=['frequency_hz','state','model','L_mm'],suffixes=('_fine','_coarse'),validate='one_to_one')
    check['difference_ohm']=check.Z_fine-check.Z_coarse
    check['difference_pct']=100*check.difference_ohm/check.Z_fine
    check['within_numeric_tolerance']=check.difference_ohm.abs()<=.02+.001*check.Z_fine.abs()
    check.to_csv(out/'quadrature_comparison.csv',index=False)
    # Preserve a failed quadrature check explicitly; never label it mesh convergence.
    previous=pd.read_csv(INITIAL_OUT/'responses.csv')
    repeated=a.merge(previous[(previous.kind=='literature')&(previous.quadrature_points==coarse)],on=['frequency_hz','state','model','L_mm'],suffixes=('','_previous'),validate='one_to_one')
    assert len(repeated)==72 and np.max(np.abs(repeated.Z-repeated.Z_previous))<1e-6
    main=b.copy();old=pd.read_csv(PREVIOUS/'comparison.csv');old=old[old.model=='bones_heart']
    main=main.merge(old[['frequency_hz','state','L_mm','Z','Z_experiment']].rename(columns={'Z':'Z_previous_four_tissue'}),
        on=['frequency_hz','state','L_mm'],validate='many_to_one')
    main['residual_ohm']=main.Z-main.Z_experiment;main['residual_pct']=100*main.residual_ohm/main.Z_experiment
    main.to_csv(out/'comparison.csv',index=False)
    wide=main.pivot(index=['frequency_hz','state','L_mm'],columns='model',values='Z').reset_index()
    wide=wide.merge(old[['frequency_hz','state','L_mm','Z','Z_experiment']].rename(columns={'Z':'previous_four_tissue'}),on=['frequency_hz','state','L_mm'],validate='one_to_one')
    wide['fat_delta_ohm']=wide.ts_with_fat-wide.ts_without_fat
    wide['fat_delta_pct']=100*wide.fat_delta_ohm/wide.ts_without_fat
    wide['segmentation_delta_ohm']=wide.ts_without_fat-wide.previous_four_tissue
    wide['combined_delta_ohm']=wide.ts_with_fat-wide.previous_four_tissue
    wide.to_csv(out/'increments.csv',index=False)
    metrics=[]
    for (f,state),g in wide.groupby(['frequency_hz','state']):
        for model in ['previous_four_tissue','ts_without_fat','ts_with_fat']:
            err=g[model]-g.Z_experiment
            metrics.append(dict(frequency_hz=f,state=state,model=model,rmse_ohm=float(np.sqrt(np.mean(err**2))),
                mean_bias_ohm=float(err.mean()),mape_pct=float(np.mean(np.abs(err)/g.Z_experiment)*100)))
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    qc=dict(complete=True,new_CEM_solutions=len(d),main_five_tissue_solutions=36,no_parameter_fitting=True,no_localization=True,
        max_relative_residual=float(d.relative_residual.max()),max_reciprocity_ohm=float(d.reciprocity_abs.max()),
        equal_material_checks_ohm=controls,quadrature_all_within_tolerance=bool(check.within_numeric_tolerance.all()),
        quadrature_max_difference_ohm=float(check.difference_ohm.abs().max()),quadrature_max_difference_pct=float(check.difference_pct.abs().max()),
        quadrature_points=[int(coarse),int(fine)],
        numeric_tolerance='abs difference <= 0.02 ohm + 0.001 abs(Z_fine); numerical scenario, not instrument noise',
        global_spatial_convergence=False,physical_validation=False,
        inputs_sha256={n:sha(out/n) for n in ['plan.json','responses.csv','requests.csv','execution.json','material_inventory.csv','literature_values.csv',f'matrix_qc_q{coarse}.json',f'matrix_qc_q{fine}.json']},
        outputs_sha256={n:sha(out/n) for n in ['comparison.csv','increments.csv','metrics.csv','quadrature_comparison.csv']},analysis_sha256=sha(__file__))
    write(out/'qc.json',qc)
    print(pd.DataFrame(metrics).to_string(index=False));print('QC',qc['quadrature_max_difference_ohm'],qc['quadrature_all_within_tolerance'])
    assert qc['quadrature_all_within_tolerance'],'Quadrature needs refinement before interpretation'
    return main,wide


def figure(out=OUT):
    qc=read(out/'qc.json');assert qc['complete'] and qc['quadrature_all_within_tolerance']
    for n,h in {**qc['inputs_sha256'],**qc['outputs_sha256']}.items():assert sha(out/n)==h,n
    d=pd.read_csv(out/'increments.csv')
    fig=make_subplots(rows=2,cols=2,vertical_spacing=.22,horizontal_spacing=.09,
        subplot_titles=['Вдох: абсолютный импеданс','Условный выдох: абсолютный импеданс',
                        'Вдох: отдельные изменения модели','Условный выдох: отдельные изменения модели'])
    series=[('Z_experiment','Эксперимент','#cf3f3f'),('previous_four_tissue','Прежние маски, без жира','#8293a2'),
            ('ts_without_fat','Новые маски, жир = фон','#2166ac'),('ts_with_fat','Новые маски + жир','#d4820b')]
    for f in [50000,100000]:
        for col,state in enumerate(['inhale','exhale'],1):
            a=d[(d.frequency_hz==f)&(d.state==state)].sort_values('L_mm')
            for field,label,color in series:
                fig.add_trace(go.Scatter(x=a.L_mm,y=a[field],name=label,legendgroup=field,showlegend=col==1,
                    visible=f==50000,mode='lines+markers',line=dict(color=color)),row=1,col=col)
            for field,label,color in [('fat_delta_ohm','Эффект свойств жира','#d4820b'),('segmentation_delta_ohm','Эффект замены масок','#2166ac')]:
                fig.add_trace(go.Scatter(x=a.L_mm,y=a[field],name=label,legendgroup=field,showlegend=col==1,
                    visible=f==50000,mode='lines+markers',line=dict(color=color)),row=2,col=col)
    for col in [1,2]:
        fig.update_yaxes(range=[0,max(115,float(d.ts_with_fat.max())*1.1)],title='Z, Ом' if col==1 else None,row=1,col=col)
        fig.update_yaxes(title='Изменение Z, Ом' if col==1 else None,zeroline=True,zerolinecolor='#555',row=2,col=col)
        fig.update_xaxes(title='Размер сборки L, мм',tickvals=[50,70,90,110,130,140],row=2,col=col)
    fig.update_layout(template='plotly_white',width=1280,height=880,font=dict(family='Arial',size=14),
        margin=dict(l=80,r=35,t=115,b=125),hovermode='x unified',title=dict(text='Добавление жира при прежнем положении сборок · 50 кГц',x=.5),
        legend=dict(orientation='h',y=-.16,x=.5,xanchor='center'),updatemenus=[dict(x=0,y=1.14,buttons=[
            dict(label=f'{f//1000} кГц',method='update',args=[{'visible':[i//12==j for i in range(24)]},
                {'title.text':f'Добавление жира при прежнем положении сборок · {f//1000} кГц'}]) for j,f in enumerate([50000,100000])])])
    return fig


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT);a=ap.parse_args()
    analyze(a.out);fig=figure(a.out);fig.write_html(a.out/'comparison.html',include_plotlyjs=True,config={'displaylogo':False})
