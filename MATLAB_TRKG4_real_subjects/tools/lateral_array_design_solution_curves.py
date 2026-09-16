"""Isoimpedance curves for each measured array, using the finite-contact library.

No FEM solve, extrapolation, observation copy, or measurement-error assumption.
Run from any directory; --out selects the existing 33.09 calculation directory.
"""
from pathlib import Path
from itertools import combinations
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lateral_array_design_analysis import (
    load_plan, load_response_tables, SplineSurrogate,
    read_declared_observations, normalise_observations, sha256_file, save_json,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'MATLAB_TRKG4_real_subjects/output/exploratory/lateral_array_design_20260916'
STATES = {'inhale': 'Вдох', 'exhale': 'Выдох'}
COLORS = ['#0072b2', '#e69f00', '#009e73', '#d55e00', '#cc79a7',
          '#56b4e9', '#7158b5', '#795548', '#333333']


def calculate(out):
    plan = load_plan(out / 'plan.json')
    data = load_response_tables(out / 'responses.csv', plan)
    model = SplineSurrogate(plan, data['grid_z'])
    raw, source, source_sha = read_declared_observations(ROOT, plan)
    obs = normalise_observations(raw, plan['sizes'])
    log_bounds = np.log(plan['bounds'])
    rho2_axis = np.linspace(*plan['bounds'][1], 601)
    # Numerical branch check over the whole rectangle, not just near fitted rho.
    x, y = np.meshgrid(np.linspace(*log_bounds[0], 401), np.log(rho2_axis), indexing='ij')
    min_dx = min(float(s.ev(x.ravel(), y.ravel(), dx=1).min()) for s in model.splines)
    if min_dx <= 0:
        raise ValueError('Single-branch numerical check failed; examine multiple roots explicitly.')

    rows, crossings, summary = [], [], {}
    for state in STATES:
        observed = obs[f'Z_{state}_hold_ohm'].to_numpy()

        def root(i, rho2):
            sy = np.log(rho2)
            def f(sx):
                return float(model.splines[i].ev(sx, sy)) - observed[i]
            if f(log_bounds[0, 0]) * f(log_bounds[0, 1]) > 0:
                raise ValueError(f'No in-range root: {state}, L={plan["sizes"][i]}, rho2={rho2}')
            return float(np.exp(brentq(f, *log_bounds[0], xtol=1e-13)))

        curves = np.array([[root(i, r2) for i in range(9)] for r2 in rho2_axis])
        for j, r2 in enumerate(rho2_axis):
            for i, length in enumerate(plan['sizes']):
                r1 = curves[j, i]
                residual = float(model.splines[i].ev(np.log(r1), np.log(r2))) - observed[i]
                rows.append(dict(state=state, L_mm=int(length), rho2=r2, rho1=r1,
                                 residual_ohm=residual))
        before = len(crossings)
        for i, k in combinations(range(9), 2):
            delta = curves[:, i] - curves[:, k]
            brackets = np.flatnonzero(delta[:-1] * delta[1:] < 0)
            candidates = [float(rho2_axis[j]) for j in np.flatnonzero(delta == 0)]
            for j in brackets:
                candidates.append(brentq(lambda r2: root(i, r2) - root(k, r2),
                                         rho2_axis[j], rho2_axis[j+1], xtol=1e-10))
            for r2 in sorted(set(candidates)):
                r1 = root(i, r2)
                errors = model.predict(np.log([r1, r2])) - observed
                crossings.append(dict(state=state, L1_mm=int(plan['sizes'][i]),
                                      L2_mm=int(plan['sizes'][k]), rho1=r1, rho2=r2,
                                      pair_max_residual_ohm=float(np.max(np.abs(errors[[i,k]]))),
                                      all_nine_rmse_ohm=float(np.sqrt(np.mean(errors**2)))))

        def spread(r2):
            return float(np.ptp([root(i, r2) for i in range(9)]))
        best_index = int(np.argmin(np.ptp(curves, axis=1)))
        local = minimize_scalar(spread, method='bounded', bounds=(
            rho2_axis[max(0, best_index-1)], rho2_axis[min(600, best_index+1)]))
        candidates = [float(rho2_axis[best_index]), float(local.x), *plan['bounds'][1]]
        best_r2 = min(candidates, key=spread)
        summary[state] = dict(
            pair_crossings_found=len(crossings)-before,
            minimum_sampled_and_refined_rho1_spread=spread(best_r2), rho2_at_minimum=best_r2,
            sampled_rho1_50_max=float(curves[:, 0].max()),
            sampled_rho1_70_min=float(curves[:, 2].min()),
            sampled_50_70_envelope_gap=float(curves[:, 2].min()-curves[:, 0].max()),
        )
    frame = pd.DataFrame(rows)
    pair_frame = pd.DataFrame(crossings, columns=[
        'state','L1_mm','L2_mm','rho1','rho2','pair_max_residual_ohm','all_nine_rmse_ohm'])
    assert frame.residual_ohm.abs().max() < 1e-8
    assert pair_frame.empty or pair_frame.pair_max_residual_ohm.max() < 1e-8
    frame.to_csv(out/'solution_curves.csv', index=False, float_format='%.14g')
    pair_frame.to_csv(out/'solution_curve_intersections.csv', index=False, float_format='%.14g')
    qc = dict(status='exploratory_hypothesis_not_validated', model='variable_transverse',
              finite_diameter_mm=5, new_FEM_solutions=0, rho_display_axes_start_at_zero=True,
              extrapolated=False, bounds_ohm_m=plan['bounds'].tolist(),
              rho2_samples=601, log_rho1_derivative_samples=401,
              min_sampled_dZ_dlogrho1=min_dx, max_root_residual_ohm=float(frame.residual_ohm.abs().max()),
              intersection_method='sign-changing brackets on 601 rho2 values, refined by Brent; tangencies not exhaustively certified',
              spread_method='601 samples, local refinement around best sample, and endpoints; not an interval proof',
              observations_source=source, observations_sha256=source_sha,
              input_sha256={n:sha256_file(out/n) for n in ['plan.json','responses.csv','real_fits.csv','best_by_count.csv']},
              code_sha256={Path(__file__).name:sha256_file(Path(__file__)),
                           'lateral_array_design_analysis.py':sha256_file(Path(__file__).with_name('lateral_array_design_analysis.py'))},
              states=summary,
              output_sha256={n:sha256_file(out/n) for n in ['solution_curves.csv','solution_curve_intersections.csv']})
    save_json(out/'solution_curves_qc.json', qc)
    make_figure(out).write_html(out/'solution_curves.html', include_plotlyjs=True,
                               config={'responsive':True,'displaylogo':False})
    for k in range(2,9):
        make_figure(out,k).write_html(out/f'solution_curves_k{k}.html', include_plotlyjs=True,
                                     config={'responsive':True,'displaylogo':False})
    make_optimal_figure(out).write_html(out/'optimal_solution_curves.html', include_plotlyjs=True,
                                       config={'responsive':True,'displaylogo':False})
    print(summary)
    return qc


def make_figure(out, k=9):
    curves = pd.read_csv(out/'solution_curves.csv')
    pairs = pd.read_csv(out/'solution_curve_intersections.csv')
    fits = pd.read_csv(out/'real_fits.csv')
    best = pd.read_csv(out/'best_by_count.csv')
    chosen = best[best.k==k]
    if len(chosen)!=1:
        raise ValueError('Select a declared best set with 2 to 9 sizes.')
    lengths = [int(x) for x in chosen.iloc[0].sizes_mm.split('|')]
    assert len(lengths)==k
    palette = dict(zip(sorted(curves.L_mm.unique()), COLORS))
    rho1_close_range=[float(curves.rho1.min())-.18,float(curves.rho1.max())+.18]
    curves = curves[curves.L_mm.isin(lengths)]
    pairs = pairs[pairs.L1_mm.isin(lengths)&pairs.L2_mm.isin(lengths)]
    bounds = load_plan(out/'plan.json')['bounds']
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.08,
                        subplot_titles=list(STATES.values()))
    for col, state in enumerate(STATES, 1):
        for length, d in curves[curves.state==state].groupby('L_mm', sort=True):
            fig.add_trace(go.Scatter(x=d.rho2, y=d.rho1, mode='lines',
                name=f'{length} мм', legendgroup=str(length), showlegend=col==1,
                line=dict(color=palette[length],width=2.8),
                meta=dict(k=k,state=state,role='curve',L_mm=int(length)),
                hovertemplate='ρ₂ = %{x:.4f} Ом·м<br>ρ₁ = %{y:.4f} Ом·м<extra>%{fullData.name}</extra>'), row=1,col=col)
        d = pairs[pairs.state==state]
        fig.add_trace(go.Scatter(x=d.rho2,y=d.rho1,mode='markers',
            name='Пересечение двух кривых',legendgroup='pairs',showlegend=col==1,
            marker=dict(symbol='circle-open',color='#555',size=13,line=dict(width=2)),
            meta=dict(k=k,state=state,role='pairs'),
            customdata=d[['L1_mm','L2_mm','all_nine_rmse_ohm']].to_numpy(),
            hovertemplate='Пара %{customdata[0]:.0f}/%{customdata[1]:.0f} мм<br>ρ₂ = %{x:.4f}<br>ρ₁ = %{y:.4f} Ом·м<br>Невязка всех 9: %{customdata[2]:.3f} Ом<extra></extra>'),row=1,col=col)
        d = fits[(fits.state==state)&(fits.k==k)]
        assert len(d)==1 and d.iloc[0].sizes_mm==chosen.iloc[0].sizes_mm
        fig.add_trace(go.Scatter(x=d.rho2_hat,y=d.rho1_hat,mode='markers',
            name='Подгонка выбранного набора',legendgroup='fit',showlegend=col==1,
            marker=dict(symbol='diamond',color='black',size=13,line=dict(color='white',width=1.5)),
            meta=dict(k=k,state=state,role='fit'),
            customdata=d[['fit_resid_rmse_ohm']].to_numpy(),
            hovertemplate='Подгонка выбранного набора<br>ρ₂ = %{x:.4f}<br>ρ₁ = %{y:.4f} Ом·м<br>Невязка набора: %{customdata[0]:.3f} Ом<extra></extra>'),row=1,col=col)
        fig.add_vrect(x0=0,x1=bounds[1,0],fillcolor='#eceff2',opacity=0.65,line_width=0,layer='below',row=1,col=col)
        fig.add_vrect(x0=bounds[1,1],x1=bounds[1,1]+.55,fillcolor='#eceff2',opacity=0.65,line_width=0,layer='below',row=1,col=col)
        fig.add_hrect(y0=0,y1=bounds[0,0],fillcolor='#eceff2',opacity=0.65,line_width=0,layer='below',row=1,col=col)
        for b in bounds[1]:fig.add_vline(x=b,line_dash='dot',line_color='#999',row=1,col=col)
        fig.add_hline(y=bounds[0,0],line_dash='dot',line_color='#999',row=1,col=col)
        fig.add_annotation(x=bounds[1,0]/2,y=8.8,text='Вне расчётного<br>диапазона',showarrow=False,
                           font=dict(size=12,color='#747c85'),row=1,col=col)
        fig.update_xaxes(title='ρ₂ лёгкого, Ом·м',range=[0,bounds[1,1]+.55],row=1,col=col)
    fig.update_yaxes(range=[0,bounds[0,1]],showticklabels=True)
    fig.update_yaxes(title='ρ₁ мягких тканей, Ом·м',row=1,col=1)
    # Keep the close-up scale identical across all selected subsets.
    zoom=rho1_close_range
    zero={'xaxis.range':[0,bounds[1,1]+.55],'xaxis2.range':[0,bounds[1,1]+.55],
          'yaxis.range':[0,bounds[0,1]],'yaxis2.range':[0,bounds[0,1]]}
    close={'xaxis.range':[bounds[1,0]-.3,bounds[1,1]+.55],
           'xaxis2.range':[bounds[1,0]-.3,bounds[1,1]+.55],
           'yaxis.range':zoom,'yaxis2.range':zoom}
    names=', '.join(str(x) for x in lengths)
    fig.update_layout(template='plotly_white',height=660,font=dict(family='Arial',size=14),
        title=dict(text=f'Набор из {k}: {names} мм<br><sup>Модель 4 · фиксированная КТ-геометрия · CEM, диаметр контакта 5 мм</sup>',x=.05,y=.94,yanchor='top'),
        margin=dict(l=80,r=30,t=155,b=125),separators=', ',
        legend=dict(orientation='h',x=0,y=-.20,groupclick='togglegroup'),
        updatemenus=[dict(type='buttons',direction='right',x=1,xanchor='right',y=1.23,
            buttons=[dict(label='От нуля',method='relayout',args=[zero]),
                     dict(label='Крупнее',method='relayout',args=[close])])])
    return fig


def make_optimal_figure(out):
    figures={k:make_figure(out,k) for k in range(2,10)}
    combined=go.Figure(layout=figures[2].layout)
    for k, fig in figures.items():
        for trace in fig.data:
            trace.visible=(k==2)
            combined.add_trace(trace)
    buttons=[]
    for k, fig in figures.items():
        word='размера' if k<5 else 'размеров'
        buttons.append(dict(label=f'{k} {word}',method='update',
            args=[{'visible':[t.meta['k']==k for t in combined.data]},
                  {'title.text':fig.layout.title.text}]))
    combined.update_layout(updatemenus=[*combined.layout.updatemenus,
        dict(type='dropdown',direction='down',x=0,xanchor='left',y=1.23,active=0,buttons=buttons)])
    return combined


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--png', action='store_true', help='Also export the HTML figure with local Playwright Chromium.')
    args=parser.parse_args()
    out=args.out.resolve()
    calculate(out)
    if args.png:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True)
            page=browser.new_page(viewport={'width':1400,'height':900},device_scale_factor=1.5)
            for name in ['solution_curves','optimal_solution_curves']:
                page.goto((out/(name+'.html')).as_uri(),wait_until='networkidle')
                page.locator('.plotly-graph-div.js-plotly-plot').screenshot(path=str(out/(name+'.png')))
            browser.close()
