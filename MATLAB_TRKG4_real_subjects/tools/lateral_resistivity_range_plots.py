"""Reader views for newly calculated positive-resistivity CEM solutions."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PIPE = Path(__file__).resolve().parents[1]
OUT = PIPE / 'output/exploratory/lateral_resistivity_range_20260916'
SIZES = [50, 60, 70, 80, 90, 110, 120, 130, 140]
COLORS = ['#0072b2', '#e69f00', '#009e73', '#cc79a7', '#d55e00',
          '#6d58a0', '#4d4d4d', '#00a6a6', '#a6761d']
STATES = {'inhale': 'Вдох', 'exhale': 'Выдох'}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load(out):
    out = Path(out)
    plan = json.loads((out / 'plan.json').read_text(encoding='utf-8'))
    for name in ['interpolation_qc.json', 'direct_check_qc.json']:
        qc = json.loads((out / name).read_text(encoding='utf-8'))
        if not qc['passed']:
            raise ValueError(f'Numerical acceptance failed: {name}')
    curve_qc = json.loads((out / 'solution_curves_qc.json').read_text(encoding='utf-8'))
    for name, digest in {**curve_qc['input_sha256'], **curve_qc['output_sha256']}.items():
        if sha(out / name) != digest:
            raise ValueError(f'Stale curve result: {name}')
    for name, digest in curve_qc['code_sha256'].items():
        if sha(PIPE / 'tools' / name) != digest:
            raise ValueError(f'Changed numerical source: {name}')
    spec = plan['sources']['best_sets']
    path = PIPE / spec['path']
    if sha(path) != spec['sha256']:
        raise ValueError('Changed source of the literature-range best subsets')
    return (plan, pd.read_csv(path), pd.read_csv(out / 'solution_curves.csv'),
            pd.read_csv(out / 'solution_curve_intersections.csv'),
            pd.read_csv(out / 'real_fits.csv'))


def title_for(k, sizes):
    prefix = 'Все девять размеров' if k == 9 else f'Лучший набор: {k} размера' if k < 5 else f'Лучший набор: {k} размеров'
    return prefix + '<br><sup>' + ', '.join(map(str, sizes)) + ' мм · расширенный расчёт CEM, контакт Ø 5 мм</sup>'


def make_figure(out=OUT, k=9):
    plan, best, curves, pairs, fits = load(out)
    selected = [int(x) for x in str(best.loc[best.k == k, 'sizes_mm'].iloc[0]).split('|')]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=.12,
                        subplot_titles=['Вдох', 'Выдох'])
    for col, (state, state_name) in enumerate(STATES.items(), 1):
        for size in selected:
            d = curves[(curves.state == state) & (curves.L_mm == size)]
            branches = sorted(d.branch_index.dropna().unique()) if 'branch_index' in d else [0]
            for n, branch in enumerate(branches):
                q = d[d.branch_index == branch] if 'branch_index' in d else d
                # Reindex each branch against the common grid: missing roots must break lines.
                x = np.sort(d.rho2.unique())
                y = q.drop_duplicates('rho2').set_index('rho2').rho1.reindex(x).to_numpy()
                fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f'{size} мм',
                    legendgroup=f'L{size}', showlegend=col == 1 and n == 0,
                    connectgaps=False, line=dict(color=COLORS[SIZES.index(size)], width=2.5),
                    meta=dict(role='curve', k=k, state=state, L_mm=size, branch=int(branch)),
                    hovertemplate='ρ₂ = %{x:.4f} Ом·м<br>ρ₁ = %{y:.4f} Ом·м<extra>%{fullData.name}</extra>'), row=1, col=col)
        p = pairs[(pairs.state == state) & pairs.L1_mm.isin(selected) & pairs.L2_mm.isin(selected)]
        fig.add_trace(go.Scatter(x=p.rho2, y=p.rho1, mode='markers', name='Пересечения пар',
            legendgroup='pairs', showlegend=col == 1,
            marker=dict(size=9, color='white', line=dict(color='#333333', width=1.5)),
            customdata=p[['L1_mm', 'L2_mm']].to_numpy(),
            meta=dict(role='pairs', k=k, state=state),
            hovertemplate='Пара %{customdata[0]:.0f}/%{customdata[1]:.0f} мм<br>ρ₂ = %{x:.4f}<br>ρ₁ = %{y:.4f} Ом·м<extra></extra>'), row=1, col=col)
        f = fits[(fits.state == state) & (fits.k == k)].iloc[0]
        fig.add_trace(go.Scatter(x=[f.rho2_hat], y=[f.rho1_hat], mode='markers',
            name='Совместная подгонка', legendgroup='fit', showlegend=col == 1,
            marker=dict(symbol='diamond', size=13, color='black', line=dict(color='white', width=1)),
            meta=dict(role='fit', k=k, state=state),
            customdata=[[f.fit_resid_rmse_ohm, bool(f.within_literature_bounds)]],
            hovertemplate='Подгонка выбранного набора<br>ρ₂ = %{x:.4f}<br>ρ₁ = %{y:.4f} Ом·м<br>RMSE = %{customdata[0]:.4f} Ом<br>В литературной области: %{customdata[1]}<extra></extra>'), row=1, col=col)
        (r1lo, r1hi), (r2lo, r2hi) = plan['literature_bounds_ohm_m']
        fig.add_shape(type='rect', x0=r2lo, x1=r2hi, y0=r1lo, y1=r1hi,
            fillcolor='rgba(70,140,70,0.10)', line=dict(color='#558855', width=1.3, dash='dash'),
            layer='below', row=1, col=col)
        (a1, b1), (a2, b2) = plan['bounds_ohm_m']
        fig.add_shape(type='rect', x0=a2, x1=b2, y0=a1, y1=b1,
            line=dict(color='#8c929b', width=1, dash='dot'), fillcolor='rgba(0,0,0,0)',
            layer='below', row=1, col=col)
    full = {'xaxis.range': [0, b2 * 1.02], 'xaxis2.range': [0, b2 * 1.02],
            'yaxis.range': [0, b1 * 1.02], 'yaxis2.range': [0, b1 * 1.02]}
    phys = {'xaxis.range': [r2lo, r2hi], 'xaxis2.range': [r2lo, r2hi],
            'yaxis.range': [r1lo, r1hi], 'yaxis2.range': [r1lo, r1hi]}
    finite = curves.rho1[np.isfinite(curves.rho1)]
    lo, hi = finite.min(), finite.max()
    close = {**full, 'yaxis.range': [max(0, lo - .1 * (hi-lo)), min(b1*1.02, hi+.1*(hi-lo))],
             'yaxis2.range': [max(0, lo-.1*(hi-lo)), min(b1*1.02, hi+.1*(hi-lo))]}
    fig.update_xaxes(title_text='ρ₂ — лёгкое, Ом·м', range=[0, b2*1.02], zerolinecolor='#a0a0a0')
    fig.update_yaxes(title_text='ρ₁ — мягкие ткани, Ом·м', range=[0, b1*1.02], zerolinecolor='#a0a0a0')
    fig.update_layout(template='plotly_white', height=690,
        title=dict(text=title_for(k, selected), x=.5, y=.94, yanchor='top'),
        font=dict(family='Arial', size=14), margin=dict(l=85, r=30, t=160, b=125),
        hovermode='closest', legend=dict(orientation='h', x=0, y=-.22, groupclick='togglegroup'),
        updatemenus=[dict(type='buttons', direction='right', x=0, xanchor='left', y=1.17, active=0,
            buttons=[dict(label='Весь диапазон', method='relayout', args=[full]),
                     dict(label='Кривые крупнее', method='relayout', args=[close]),
                     dict(label='Литературная область', method='relayout', args=[phys])])])
    return fig


def make_optimal_figure(out=OUT):
    figs = {k: make_figure(out, k) for k in range(2, 10)}
    fig = go.Figure(layout=figs[2].layout)
    for k, f in figs.items():
        for trace in f.data:
            trace.visible = k == 2
            fig.add_trace(trace)
    buttons = []
    for k, f in figs.items():
        label = f'{k} размера' if k < 5 else f'{k} размеров'
        buttons.append(dict(label=label, method='update', args=[
            {'visible': [t.meta['k'] == k for t in fig.data]}, {'title.text': f.layout.title.text}]))
    menus = list(fig.layout.updatemenus)
    menus.append(dict(type='dropdown', direction='down', x=1, xanchor='right', y=1.17,
                      active=0, buttons=buttons))
    fig.update_layout(updatemenus=menus)
    return fig


def export(out=OUT, png=False):
    out = Path(out)
    figures = {'solution_curves': make_figure(out), 'optimal_solution_curves': make_optimal_figure(out)}
    figures.update({f'solution_curves_k{k}': make_figure(out, k) for k in range(2, 9)})
    for name, fig in figures.items():
        fig.write_html(out / f'{name}.html', include_plotlyjs=True, full_html=True,
                       config={'responsive': True, 'displaylogo': False})
    if png:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={'width': 1550, 'height': 780}, device_scale_factor=1.5)
            for name in ['solution_curves', 'optimal_solution_curves']:
                page.goto((out / f'{name}.html').resolve().as_uri(), wait_until='networkidle')
                page.locator('.js-plotly-plot').screenshot(path=str(out / f'{name}.png'))
            browser.close()
    report = {'source_sha256': sha(__file__), 'inputs_sha256': {
        name: sha(out / name) for name in ['plan.json', 'solution_curves.csv',
            'solution_curve_intersections.csv', 'real_fits.csv']},
        'artifacts_sha256': {p.name: sha(p) for p in out.iterdir() if p.suffix in ['.html', '.png'] and not p.name.startswith('visual_')}}
    (out / 'visual_qc_exports.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'exports': list(figures), 'png': png}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--png', action='store_true')
    args = parser.parse_args()
    export(args.out, args.png)
