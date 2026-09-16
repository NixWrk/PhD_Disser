"""Compare 50/100 kHz resistive CEM scenarios with identical anatomy and data."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lateral_literature_forward import BASE, OUT as LOW, PIPE, sha, read, write, validate, compare

HIGH = PIPE/'output/exploratory/lateral_literature_100khz_20260916'


def analyze():
    frames=[]; metrics=[]; values=[]
    plans=[]
    for directory, frequency in [(LOW,50000),(HIGH,100000)]:
        plan, _, _=validate(directory)
        assert plan['frequency_hz']==frequency
        plans.append(plan)
        compare(directory)
        for dest, name in [(frames,'comparison.csv'),(metrics,'metrics.csv'),(values,'literature_values.csv')]:
            d=pd.read_csv(directory/name)
            d['frequency_hz']=frequency
            dest.append(d)
    for key in ['geometry','sizes_mm','diameter_mm','contact_ohm_m2','observations','mapping','electrode_count_per_measurement','simultaneous_all_sizes']:
        assert plans[0][key]==plans[1][key], key
    low_sources={s['path']:s['sha256'] for s in plans[0]['sources']}
    high_sources={s['path']:s['sha256'] for s in plans[1]['sources']}
    assert low_sources==high_sources
    data=pd.concat(frames,ignore_index=True)
    data.to_csv(HIGH/'frequency_comparison.csv',index=False)
    pd.concat(metrics,ignore_index=True).to_csv(HIGH/'frequency_metrics.csv',index=False)
    pd.concat(values,ignore_index=True).to_csv(HIGH/'frequency_values.csv',index=False)
    wide=data.pivot(index=['state','model','L_mm'],columns='frequency_hz',values='Z')
    wide['change_ohm']=wide[100000]-wide[50000]
    wide['change_pct']=100*wide.change_ohm/wide[50000]
    wide.reset_index().to_csv(HIGH/'frequency_change.csv',index=False)
    write(HIGH/'frequency_qc.json',dict(complete=True,new_CEM_solves_100khz=36,previous_CEM_solves_50khz=36,
        fitted_parameters=0,same_geometry_contacts_observations=True,
        scope='two real-conductivity scenarios, not full complex-frequency validation',
        min_impedance_change_pct=float(wide.change_pct.min()),max_impedance_change_pct=float(wide.change_pct.max()),
        input_sha256={str((d/n).relative_to(PIPE).as_posix()):sha(d/n)
                     for d in (LOW,HIGH) for n in ('plan.json','points_responses.csv','comparison.csv','qc.json')},
        output_sha256={n:sha(HIGH/n) for n in ('frequency_comparison.csv','frequency_metrics.csv','frequency_values.csv','frequency_change.csv')},
        source_sha256={str(Path(p).relative_to(PIPE).as_posix()):sha(p)
                       for p in (__file__, PIPE/'tools/lateral_literature_forward.py')}))
    return pd.concat(metrics,ignore_index=True)


def checked_data():
    qc=read(HIGH/'frequency_qc.json')
    assert qc['complete']
    for n,h in {**qc['input_sha256'],**qc['source_sha256']}.items():assert sha(PIPE/n)==h,n
    for n,h in qc['output_sha256'].items():assert sha(HIGH/n)==h,n
    return pd.read_csv(HIGH/'frequency_comparison.csv')


def figure():
    data=checked_data()
    fig=make_subplots(rows=1,cols=2,shared_yaxes=True,horizontal_spacing=.08,
        subplot_titles=['Вдох: сопоставление с inflated','Выдох: условное сопоставление с deflated'])
    for col,state in enumerate(['inhale','exhale'],1):
        obs=data[(data.state==state)&(data.model=='reference')&(data.frequency_hz==50000)].sort_values('L_mm')
        fig.add_trace(go.Scatter(x=obs.L_mm,y=obs.Z_experiment,name='Эксперимент',mode='lines+markers',
            line=dict(color='#d64b31',width=2),marker=dict(size=8),legendgroup='experiment',showlegend=col==1),row=1,col=col)
        for frequency,color in [(50000,'#828a93'),(100000,'#2166ac')]:
            for model,label in [('reference','КТ'),('variable_transverse','модель 4')]:
                g=data[(data.state==state)&(data.frequency_hz==frequency)&(data.model==model)].sort_values('L_mm')
                fig.add_trace(go.Scatter(x=g.L_mm,y=g.Z,name=f'{frequency//1000} кГц: {label}',mode='lines+markers',
                    line=dict(color=color,width=2.5,dash='solid' if model=='reference' else 'dot'),
                    marker=dict(size=6),legendgroup=f'{frequency}_{model}',showlegend=col==1,
                    visible=True if model=='reference' else 'legendonly', customdata=g[['rho1','rho2']],
                    hovertemplate='L=%{x} мм<br>Z=%{y:.3f} Ом<br>ρ₁=%{customdata[0]:.4f}; ρ₂=%{customdata[1]:.4f} Ом·м<extra>%{fullData.name}</extra>'),row=1,col=col)
    fig.update_xaxes(title='Размер сборки L, мм',tickvals=[50,70,90,110,130,140])
    fig.update_yaxes(range=[0,115],title='Импеданс Z, Ом',row=1,col=1)
    fig.update_yaxes(range=[0,115],row=1,col=2)
    fig.update_layout(template='plotly_white',width=1250,height=590,
        font=dict(family='Arial',size=14),margin=dict(l=70,r=25,t=85,b=115),
        title=dict(text='Прямой расчёт: литературные свойства при 50 и 100 кГц',x=.5,xanchor='center'),
        legend=dict(orientation='h',x=.5,xanchor='center',y=-.2,font=dict(size=12)),hovermode='x unified')
    return fig


if __name__=='__main__':
    print(analyze().to_string(index=False))
    fig=figure()
    fig.write_html(HIGH/'frequency_comparison.html',include_plotlyjs=True,config={'displaylogo':False})
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True)
        page=browser.new_page(viewport={'width':1280,'height':640},device_scale_factor=1.5)
        page.goto((HIGH/'frequency_comparison.html').as_uri(),wait_until='networkidle')
        page.locator('.js-plotly-plot').screenshot(path=str(HIGH/'frequency_comparison.png'))
        browser.close()
