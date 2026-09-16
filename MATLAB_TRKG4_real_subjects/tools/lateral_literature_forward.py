"""Unfitted literature-point forward CEM check; consumes the frozen 33.09 mesh."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PIPE = Path(__file__).resolve().parents[1]
BASE = PIPE / 'output/exploratory/lateral_array_design_20260916'
OUT = PIPE / 'output/exploratory/lateral_literature_forward_20260916'
LIT = PIPE / 'output/exploratory/geometry_validation_20260910'
MODELS = {'reference': 'Исходная КТ-маска', 'variable_transverse': 'Модель 4'}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def source(path):
    return {'path': Path(path).relative_to(PIPE).as_posix(), 'sha256': sha(path)}


def prepare(out=OUT, frequency_hz=50000):
    assert frequency_hz in (50000, 100000), "This comparison defines 50 and 100 kHz only."
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'plan.json').exists():
        raise FileExistsError('Frozen plan exists; use a fresh output directory.')
    selection = read(LIT/'literature_selection.json')
    raw_name, raw_sha = next(iter(selection['source_files'].items()))
    raw = PIPE / raw_name.replace('\\', '/')
    assert sha(raw) == raw_sha
    anchors = pd.read_csv(LIT/'literature_anchors.csv').set_index('tissue')
    rows = []
    for line, row in enumerate(csv.reader(raw.read_text(encoding='cp1252').splitlines(), delimiter='\t'), 1):
        if len(row) < 48 or row[1] not in ('Muscle', 'Lung (Inflated)', 'Lung (Deflated)'):
            continue
        delta = np.array([float(row[i]) for i in (28,31,35,38)])
        tau = np.array([float(row[i]) for i in (29,32,36,39)]) * [1e-12,1e-9,1e-6,1e-3]
        alpha = np.array([float(row[i]) for i in (30,33,37,40)])
        omega = 2*np.pi*frequency_hz
        eps = np.sum(delta/(1+(1j*omega*tau)**(1-alpha)))
        sigma = float(row[34])-omega*8.8542e-12*eps.imag
        rho = 1/sigma
        if frequency_hz == 50000:
            assert np.isclose(rho, anchors.loc[row[1], 'resistivity_50khz_ohm_m'], rtol=1e-12)
        rows.append(dict(tissue=row[1], source_line=line, frequency_hz=frequency_hz,
                         conductivity_S_m=sigma, rho_ohm_m=rho))
    a = pd.DataFrame(rows).set_index('tissue')
    assert set(a.index) == {'Muscle','Lung (Inflated)','Lung (Deflated)'}
    points = pd.DataFrame([
        dict(point_id='inflated', kind='literature_named_tissues', rho1=a.loc['Muscle','rho_ohm_m'], rho2=a.loc['Lung (Inflated)','rho_ohm_m']),
        dict(point_id='deflated', kind='literature_named_tissues', rho1=a.loc['Muscle','rho_ohm_m'], rho2=a.loc['Lung (Deflated)','rho_ohm_m']),
    ])
    points.to_csv(out/'points.csv', index=False)
    a.reset_index().to_csv(out/'literature_values.csv', index=False)
    bp = read(BASE/'plan.json')
    paths = [BASE/'plan.json', BASE/'prepared_fem.mat', BASE/'contact_qc.csv',
             PIPE/bp['observations_source'].replace('\\','/'),
             LIT/'literature_selection.json', LIT/'literature_anchors.csv', raw,
             PIPE/'src/run_lateral_array_design.m']
    assert sha(paths[3]) == bp['observations_sha256']
    write(out/'plan.json', dict(status='exploratory_hypothesis_not_validated',
        task='Direct CEM comparison at a priori literature points; no parameter fitting',
        models=list(MODELS), sizes_mm=bp['sizes_mm'], frequency_hz=frequency_hz,
        frequency_status='literature_scenario_not_new_instrument_verification',
        frequency_change_scope='real tissue conductivities only; contact impedance held fixed; displacement current omitted',
        diameter_mm=5, contact_ohm_m2=bp['contact_ohm_m2'],
        electrode_count_per_measurement=4, simultaneous_all_sizes=False,
        geometry='fixed inspiratory CT, exact accepted C01 montage; two scalar conductive materials',
        mapping={'inhale':'inflated','exhale':'deflated'},
        mapping_status='hypothesis: database inflated/deflated not established as volunteer inhale/exhale',
        rho1_status='muscle value assigned to entire nonlung background; no mixture calibration',
        sources=[source(p) for p in paths], observations=source(paths[3]),
        points_sha256=sha(out/'points.csv'), literature_values_sha256=sha(out/'literature_values.csv'),
        literature_doi='10.13099/VIP21000-05-0', literature_urls=selection['selected_source_urls'],
        preparation_script_sha256=sha(__file__), acceptance=bp['acceptance']))
    return points


def collect(out=OUT):
    """Archive the existing worker's completed output without changing its frozen code."""
    out = Path(out).resolve()
    assert out.is_relative_to(PIPE.resolve())
    plan = read(out/'plan.json')
    origin = BASE/'points_responses.csv'
    execution = read(str(origin)+'.execution.json')
    assert execution['complete'] and execution['rows']==36
    assert execution['identity']['requests_sha256']==plan['points_sha256']
    assert execution['responses_sha256']==sha(origin)
    pairs=[(Path(str(origin)+suffix), out/('points_responses.csv'+suffix))
           for suffix in ('','.identity.json','.execution.json')]
    assert all(a.is_file() and not b.exists() for a,b in pairs)
    for a,b in pairs:
        a.rename(b)


def validate(out=OUT):
    out = Path(out)
    plan = read(out/'plan.json')
    for s in plan['sources']:
        assert sha(PIPE/s['path']) == s['sha256'], s['path']
    assert sha(out/'points.csv') == plan['points_sha256']
    assert sha(out/'literature_values.csv') == plan['literature_values_sha256']
    result = out/'points_responses.csv'
    execution = read(str(result)+'.execution.json')
    assert execution['complete'] and execution['rows'] == 36
    assert execution['responses_sha256'] == sha(result)
    identity = execution['identity']
    assert identity == read(str(result)+'.identity.json')
    assert identity['requests_sha256'] == plan['points_sha256']
    assert identity['prepared_identity'] == read(BASE/'preparation_qc.json')['identity']
    df = pd.read_csv(result)
    points = pd.read_csv(out/'points.csv').set_index('point_id')
    assert len(df)==36 and not df.duplicated(['point_id','model','L_mm']).any()
    assert set(df.model)==set(MODELS) and set(df.point_id)==set(points.index)
    assert np.isfinite(df[['rho1','rho2','Z','relative_residual','reciprocity_abs']]).all().all()
    for (pid, model), g in df.groupby(['point_id','model']):
        assert sorted(g.L_mm)==plan['sizes_mm']
        np.testing.assert_allclose(g[['rho1','rho2']].to_numpy(),
            np.tile(points.loc[pid,['rho1','rho2']].to_numpy(float),(9,1)), rtol=1e-12)
    assert df.relative_residual.max()<plan['acceptance']['system_relative_residual_max']
    assert df.reciprocity_abs.max()<plan['acceptance']['reciprocity_abs_ohm_max']
    obs = pd.read_csv(PIPE/plan['observations']['path'])
    assert sorted(obs.L_mm)==plan['sizes_mm'] and obs.L_mm.is_unique
    return plan, df, obs


def compare(out=OUT):
    out=Path(out)
    plan, df, obs=validate(out)
    comparisons=[]
    for state, pid in plan['mapping'].items():
        g=df[df.point_id==pid].merge(obs[['L_mm',f'Z_{state}_hold_ohm']], on='L_mm', validate='many_to_one')
        g=g.rename(columns={f'Z_{state}_hold_ohm':'Z_experiment'})
        g['state']=state
        g['residual_ohm']=g.Z-g.Z_experiment
        g['relative_residual_pct']=100*g.residual_ohm/g.Z_experiment
        comparisons.append(g)
    comparison=pd.concat(comparisons,ignore_index=True)
    comparison.to_csv(out/'comparison.csv', index=False)
    metrics=[]
    for (state,model),g in comparison.groupby(['state','model']):
        metrics.append(dict(state=state,model=model,rmse_ohm=np.sqrt(np.mean(g.residual_ohm**2)),
            mean_bias_ohm=g.residual_ohm.mean(),mape_pct=g.relative_residual_pct.abs().mean(),
            minimum_relative_residual_pct=g.relative_residual_pct.min(),
            maximum_relative_residual_pct=g.relative_residual_pct.max()))
    pd.DataFrame(metrics).to_csv(out/'metrics.csv', index=False)
    wide=comparison.pivot(index=['state','L_mm'],columns='model',values='Z')
    maxgeo=float((wide.reference-wide.variable_transverse).abs().max())
    contrast=obs.copy()
    contrast['experimental_contrast_ohm']=contrast.Z_inhale_hold_ohm-contrast.Z_exhale_hold_ohm
    for model in MODELS:
        a=df[df.model==model].pivot(index='L_mm',columns='point_id',values='Z')
        contrast[model+'_contrast_ohm']=contrast.L_mm.map(a.inflated-a.deflated)
    contrast.to_csv(out/'breathing_contrast.csv',index=False)
    write(out/'qc.json',dict(complete=True, direct_CEM_solves=36, fitted_parameters=0,
        max_model_geometry_difference_ohm=maxgeo,
        max_system_relative_residual=float(df.relative_residual.max()),
        max_reciprocity_error_ohm=float(df.reciprocity_abs.max()),
        global_spatial_convergence=False, physical_validation=False,
        source_sha256={n:sha(out/n) for n in ['plan.json','points.csv','literature_values.csv','points_responses.csv']},
        output_sha256={n:sha(out/n) for n in ['comparison.csv','metrics.csv','breathing_contrast.csv']},
        analysis_script_sha256=sha(__file__)))
    return pd.DataFrame(metrics)


def figure(out=OUT):
    out=Path(out)
    validate(out)
    qc=read(out/'qc.json')
    for n,h in {**qc['source_sha256'],**qc['output_sha256']}.items(): assert sha(out/n)==h
    df=pd.read_csv(out/'comparison.csv')
    fig=make_subplots(rows=1,cols=2,shared_yaxes=True,horizontal_spacing=.08,
        subplot_titles=['Вдох: сопоставление с inflated','Выдох: условное сопоставление с deflated'])
    colors={'reference':'#2166ac','variable_transverse':'#18a0ac'}
    for col,state in enumerate(['inhale','exhale'],1):
        g=df[(df.state==state)&(df.model=='reference')].sort_values('L_mm')
        fig.add_trace(go.Scatter(x=g.L_mm,y=g.Z_experiment,mode='lines+markers',name='Эксперимент',
            line=dict(color='#d64b31',width=2),marker=dict(size=8),legendgroup='experiment',showlegend=col==1),row=1,col=col)
        for model,name in MODELS.items():
            g=df[(df.state==state)&(df.model==model)].sort_values('L_mm')
            fig.add_trace(go.Scatter(x=g.L_mm,y=g.Z,mode='lines+markers',name=name,
                line=dict(color=colors[model],width=2.5,dash='solid' if model=='reference' else 'dash'),
                marker=dict(size=6),legendgroup=model,showlegend=col==1,
                customdata=g[['rho1','rho2']],
                hovertemplate='L=%{x} мм<br>Z=%{y:.3f} Ом<br>ρ₁=%{customdata[0]:.4f}; ρ₂=%{customdata[1]:.4f} Ом·м<extra>%{fullData.name}</extra>'),row=1,col=col)
        r1,r2=g[['rho1','rho2']].iloc[0]
        fig.add_annotation(x=.5,y=.97,xref='x domain' if col==1 else 'x2 domain',
            yref='y domain' if col==1 else 'y2 domain',showarrow=False,
            text=f'ρ₁ = {r1:.3f}; ρ₂ = {r2:.3f} Ом·м'.replace('.',','),font=dict(size=13))
    fig.update_xaxes(title='Размер сборки L, мм',tickvals=[50,70,90,110,130,140])
    fig.update_yaxes(range=[0,115],title='Импеданс Z, Ом',row=1,col=1)
    fig.update_yaxes(range=[0,115],row=1,col=2)
    fig.update_layout(template='plotly_white',width=1250,height=570,
        font=dict(family='Arial',size=14),margin=dict(l=70,r=25,t=85,b=100),
        title=dict(text=f"Прямая задача: литературные сопротивления, {read(out/'plan.json')['frequency_hz']/1000:g} кГц",x=.5,xanchor='center'),
        legend=dict(orientation='h',x=.5,xanchor='center',y=-.2),hovermode='x unified')
    return fig


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--collect',action='store_true')
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--frequency-hz',type=int,choices=[50000,100000],default=50000)
    args=parser.parse_args()
    if args.prepare:
        print(prepare(args.out,args.frequency_hz).to_string(index=False))
    else:
        if args.collect: collect(args.out)
        print(compare(args.out).to_string(index=False))
        fig=figure(args.out)
        fig.write_html(args.out/'forward_comparison.html',include_plotlyjs=True,config={'displaylogo':False})
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True)
            page=browser.new_page(viewport={'width':1280,'height':620},device_scale_factor=1.5)
            page.goto((args.out/'forward_comparison.html').resolve().as_uri(),wait_until='networkidle')
            page.locator('.js-plotly-plot').screenshot(path=str(args.out/'forward_comparison.png'))
            browser.close()
