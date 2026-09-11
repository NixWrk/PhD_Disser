"""Visualize an existing, verified inverse search. Does not evaluate FEM.

Run export_search_geometry.m first. Colours are observed objective values;
no interpolation, inference of untested values, or conversion of J to ohms.
All embedded geometry stays in a local standalone HTML file.
"""
from pathlib import Path
import argparse
import hashlib
import json
import html

import numpy as np
import pandas as pd
from scipy.io import loadmat
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
XYZ = ['centre_x_mm', 'centre_y_mm', 'centre_z_mm']
NORMAL = ['normal_x', 'normal_y', 'normal_z']
UV = ['centre_u_mm', 'centre_v_mm']


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def mesh_trace(mesh, name, color, opacity, visible=True):
    p = mesh.points
    f = mesh.faces.reshape(-1, 4)[:, 1:]
    return go.Mesh3d(x=p[:, 0], y=p[:, 1], z=p[:, 2],
                     i=f[:, 0], j=f[:, 1], k=f[:, 2], name=name,
                     color=color, opacity=opacity, visible=visible,
                     hoverinfo='skip', showlegend=True, flatshading=False,
                     lighting=dict(ambient=.7, diffuse=.7, specular=.1))


def point_text(row):
    return (f"Оценка №{int(row.evaluation)} · J = {row.objective:.3f}"
            f"<br>u = {row.centre_u_mm:.3f} мм; v = {row.centre_v_mm:.3f} мм"
            f"<br>φ = {row.phi_deg:.3f}°; ρ₁ = {row.rho_soft_ohm_m:.3f}; "
            f"ρ₂ = {row.rho_lungs_ohm_m:.3f} Ом·м"
            f"<br>h до вершины лёгкого = {row.h_centre_mm:.3f} мм"
            f"<br>XYZ = ({row.centre_x_mm:.3f}, {row.centre_y_mm:.3f}, "
            f"{row.centre_z_mm:.3f}) мм"
            f"<br>В этом центре: {int(row.valid_count)} допустимых; "
            f"{int(row.invalid_count)} отклонённых оценок")


def main(tag):
    contract_path = ROOT / 'output' / f'nik_trkg4_inverse_inhale_contract_{tag}.json'
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    assert contract['result_tag'] == tag and contract['geometry_qc_passed']
    checked = []
    for entry in contract['inputs'] + contract['artifacts']:
        p = ROOT / entry['path']
        assert sha(p) == entry['sha256'], f'Changed contracted file: {p}'
        checked.append(entry['path'])
    data_path = HERE / f'search_points_{tag}.csv'
    df = pd.read_csv(data_path)
    raw = pd.read_csv(ROOT / 'output' / f'nik_trkg4_inverse_direct_search_history_{tag}.csv')
    pd.testing.assert_frame_equal(df[raw.columns], raw, check_exact=False, rtol=1e-13, atol=1e-12)
    # Collapse sub-ulp differences in original CSV parameters (e.g. 4e-16 mm).
    # Eight decimals is solely a duplicate-key tolerance, not position accuracy.
    df[UV] = df[UV].round(8)
    assert set(df.status) <= {'evaluated', 'invalid_geometry'}
    assert df.engine_fingerprint.nunique() == 1 and df.engine_fingerprint.iloc[0] == contract['fingerprint']
    valid = df[df.status == 'evaluated'].copy()
    assert np.isfinite(valid.objective).all()
    best = valid.loc[valid.objective.idxmin()]
    fit = pd.read_csv(ROOT / 'output' / f'nik_trkg4_inverse_inhale_fit_summary_{tag}.csv').iloc[0]
    assert np.max(np.abs(best[XYZ].to_numpy(float) - fit[XYZ].to_numpy(float))) < 1e-8
    centres = valid.loc[valid.groupby(UV).objective.idxmin()].copy()
    counts = df.groupby(UV).status.agg(valid_count=lambda s: (s == 'evaluated').sum(),
                                     invalid_count=lambda s: (s == 'invalid_geometry').sum())
    centres = centres.merge(counts, on=UV).sort_values('objective')
    failed = df.merge(counts, on=UV).query('valid_count == 0').drop_duplicates(UV)
    assert len(centres) + len(failed) == len(counts)
    # One actual surface centre per requested (u,v); phi changes the axis only.
    assert df.groupby(UV)[XYZ].agg(lambda s: float(s.max()-s.min())).to_numpy().max() < 1e-9
    assert len(df[XYZ].round(8).drop_duplicates()) == len(counts)
    geo = loadmat(HERE / f'search_surface_{tag}.mat')
    nodes = geo['nodes_mm']
    faces = geo['boundary_faces'].astype(int) - 1
    body = pv.PolyData(nodes, np.column_stack([np.full(len(faces), 3), faces]))
    layers = {}
    layer_inputs = []
    for key, filename in [('lungs', 'lungs_solid_v3_volume_fill.stl'), ('bones', 'bones.stl')]:
        p = ROOT / 'data/nik/stl' / filename
        m = pv.read(p).triangulate().clean()
        original_faces = m.n_cells
        # Topology-preserving reduction is visualization-only; skin is exact.
        if m.n_cells > 45000:
            m = m.decimate_pro(1 - 45000 / m.n_cells, preserve_topology=True)
        layers[key] = m
        layer_inputs.append(dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p),
                                 source_triangles=original_faces, display_triangles=m.n_cells))
    centre = best[XYZ].to_numpy(float)
    normal = best[NORMAL].to_numpy(float)
    # CT z is kept unchanged. Camera up follows positive z, with equal xyz scale.
    camera = dict(eye=dict(zip('xyz', (1.4*normal).tolist())),
                  up=dict(x=0, y=0, z=1), projection=dict(type='orthographic'))
    cxyz = centres[XYZ].to_numpy(float)
    cn = centres[NORMAL].to_numpy(float)
    shown_xyz = cxyz + .8 * cn  # Prevent z fighting; underlying data is unchanged.
    fxyz = failed[XYZ].to_numpy(float) + .8 * failed[NORMAL].to_numpy(float)
    vmin, vmax = centres.objective.min(), centres.objective.max()
    colours = dict(color=centres.objective.to_numpy(), colorscale='RdYlBu_r', cmin=vmin, cmax=vmax)
    text = [point_text(r) for r in centres.itertuples()]
    badtext = [f"u={r.centre_u_mm:.3f}; v={r.centre_v_mm:.3f} мм<br>"
               f"Все {int(r.invalid_count)} проверенных варианта отклонены по геометрии."
               '<br>Другие углы здесь могли не проверяться.' for r in failed.itertuples()]
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'xy'}]],
                        column_widths=[.60, .40], horizontal_spacing=.06,
                        subplot_titles=['3D: участок боковой поверхности', 'Те же центры в координатах поиска'])
    fig.add_trace(mesh_trace(body, 'Поверхность FEM', '#c9c5be', .32), 1, 1)
    fig.add_trace(mesh_trace(layers['lungs'], 'Лёгкие', '#91cbdc', .25), 1, 1)
    fig.add_trace(mesh_trace(layers['bones'], 'Рёбра / кости', '#baa47d', .48, 'legendonly'), 1, 1)
    fig.add_trace(go.Scatter3d(x=shown_xyz[:, 0], y=shown_xyz[:, 1], z=shown_xyz[:, 2],
                             mode='markers', marker=dict(size=6, **colours, line=dict(color='#182430', width=1)),
                             text=text, hovertemplate='%{text}<extra></extra>', name='Допустимые центры'), 1, 1)
    fig.add_trace(go.Scatter3d(x=fxyz[:, 0], y=fxyz[:, 1], z=fxyz[:, 2], mode='markers',
                             marker=dict(size=6, color='#777777', symbol='x'), text=badtext,
                             hovertemplate='%{text}<extra></extra>', name='Только отклонённые варианты'), 1, 1)
    bshow = centre + 1.3*normal
    fig.add_trace(go.Scatter3d(x=[bshow[0]], y=[bshow[1]], z=[bshow[2]], mode='markers+text',
                             text=[f'Лучшая №{int(best.evaluation)}'],
                             textposition='top center', marker=dict(size=11, color='#111827', symbol='diamond-open'),
                             hoverinfo='skip', name='Лучший проверенный вариант'), 1, 1)
    electrodes_path = ROOT/'output'/f'nik_trkg4_inverse_inhale_electrodes_{tag}.csv'
    el = pd.read_csv(electrodes_path).query('L_mm == 140')
    ep = el[['requested_x_mm','requested_y_mm','requested_z_mm']].to_numpy(float)
    fig.add_trace(go.Scatter3d(x=ep[:, 0], y=ep[:, 1], z=ep[:, 2], mode='markers',
                             marker=dict(size=6, color='#7541ac'), text=['I+', 'V+', 'V−', 'I−'],
                             hovertemplate='%{text}<extra>Заданный центр на коже</extra>',
                             name='Электроды 140 мм: лучшая поза', visible='legendonly'), 1, 1)
    fig.add_trace(go.Scatter(x=centres.centre_u_mm, y=centres.centre_v_mm, mode='markers',
                            marker=dict(size=13, **colours, line=dict(color='#182430', width=1), showscale=True,
                                        colorbar=dict(title='J: меньше — лучше', x=1.03, len=.72)),
                            text=text, hovertemplate='%{text}<extra></extra>', showlegend=False), 1, 2)
    fig.add_trace(go.Scatter(x=failed.centre_u_mm, y=failed.centre_v_mm, mode='markers',
                            marker=dict(size=12, color='#777777', symbol='x'), text=badtext,
                            hovertemplate='%{text}<extra></extra>', showlegend=False), 1, 2)
    fig.add_trace(go.Scatter(x=[best.centre_u_mm], y=[best.centre_v_mm], mode='markers',
                            marker=dict(size=22, color='#111827', symbol='star-open', line=dict(width=2)),
                            hoverinfo='skip', showlegend=False), 1, 2)
    fig.update_xaxes(title='u, мм', range=[df.centre_u_mm.min()-4, df.centre_u_mm.max()+4], row=1, col=2)
    fig.update_yaxes(title='v, мм', range=[df.centre_v_mm.min()-4, df.centre_v_mm.max()+4],
                     scaleanchor='x', scaleratio=1, row=1, col=2)
    def scene_ranges(lo, hi):
        return {f'scene.{axis}axis.range': [float(lo[k]), float(hi[k])] for k, axis in enumerate('xyz')}
    close = scene_ranges(centre - 32, centre + 32)
    montage = scene_ranges(centre - [55, 85, 95], centre + [55, 85, 95])
    whole = scene_ranges(nodes.min(axis=0)-8, nodes.max(axis=0)+8)
    fig.update_layout(template='plotly_white', height=670, margin=dict(l=0,r=105,t=60,b=100),
                      font=dict(family='Arial', size=13),
                      legend=dict(orientation='h', y=-.14, x=0, font=dict(size=12)),
                      scene=dict(aspectmode='data', camera=camera,
                                 **{f'{a}axis': dict(title=f'{a.upper()}, мм', range=close[f'scene.{a}axis.range']) for a in 'xyz'}))
    for ranges in (close, montage, whole):
        ranges['scene.camera'] = camera
    fig.update_layout(updatemenus=[dict(type='buttons', direction='right', x=0, y=1.14, buttons=[
        dict(label='Точки крупно', method='relayout', args=[close]),
        dict(label='Область сборки', method='relayout', args=[montage]),
        dict(label='Всё тело', method='relayout', args=[whole])])])
    plot_html = fig.to_html(full_html=False, include_plotlyjs=True, div_id='search-map',
                            config=dict(responsive=True, displaylogo=False, scrollZoom=True))
    table = centres[['evaluation', *UV, 'phi_deg', 'rho_soft_ohm_m', 'rho_lungs_ohm_m',
                     'objective', 'h_centre_mm', 'valid_count', 'invalid_count']].rename(columns={
        'evaluation':'№ оценки', 'centre_u_mm':'u, мм', 'centre_v_mm':'v, мм', 'phi_deg':'φ, °',
        'rho_soft_ohm_m':'ρ₁, Ом·м', 'rho_lungs_ohm_m':'ρ₂, Ом·м', 'objective':'J',
        'h_centre_mm':'h, мм', 'valid_count':'Допустимо', 'invalid_count':'Отклонено'})
    page = f'''<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>TRKG4 — точки поиска</title>
<style>body{{font:16px/1.5 Arial,sans-serif;color:#172636;background:#f5f7fa;margin:0}}
main{{max-width:1450px;margin:auto;padding:24px}}h1{{font-size:27px;margin:0 0 8px}}
p{{max-width:1150px;margin:8px 0}}.stats{{display:flex;gap:30px;flex-wrap:wrap;margin:20px 0}}
.stats b{{display:block;font-size:27px}}.plot{{background:white;border:1px solid #d8e0e8;border-radius:8px;padding:12px 8px}}
.note{{padding:12px 16px;border-left:4px solid #397c9d;background:#eaf3f8}}table{{border-collapse:collapse;font-size:13px}}
th,td{{padding:7px 10px;border-bottom:1px solid #ddd;text-align:right}}summary{{cursor:pointer;font-weight:bold;margin:16px 0}}
.muted{{color:#536477;font-size:13px}}a{{color:#175e94}}</style><main>
<h1>Где уже искали место боковой сборки</h1>
<p>Результаты локальной подгонки FEM к эксперименту Ника на вдохе. Тег: <b>{html.escape(tag)}</b>.</p>
<div class="stats"><div><b>{len(df)}</b>оценок: {len(valid)} допустимых, {len(df)-len(valid)} отклонённых</div>
<div><b>{len(counts)}</b>различных центров: {len(centres)} с допустимыми вариантами</div>
<div><b>{fit.rms_residual_ohm:.2f} Ом</b>RMSE к эксперименту у лучшего по J варианта</div></div>
<div class="note"><b>Цвет — лучший J среди реально проверенных вариантов в этом центре.</b>
Синий означает меньший J, красный — больший. J объединяет расхождение с экспериментом и штрафы;
это не ошибка в Омах и не вероятность. В разных точках проверены разные ρ₁, ρ₂ и φ.
Эта карта не показывает сравнение двуслойной модели с FEM при фиксированных сопротивлениях.</div>
<div class="plot">{plot_html}</div>
<p>Вращайте модель мышью, приближайте колесом. Кнопки меняют область обзора;
нажатие названия слоя в легенде включает или скрывает его. Наведение на цветную точку показывает её параметры.</p>
<p>Чёрный контур отмечает лучший вариант. Серые кресты: все варианты, проверенные в этих центрах,
были отклонены по геометрии площадок; другие углы могли не проверяться.
Между точками значения не рассчитывались и не интерполировались.</p>
<p><b>Лучший проверенный вариант:</b> оценка №{int(best.evaluation)}, J={best.objective:.3f};
u={best.centre_u_mm:.3f} мм, v={best.centre_v_mm:.3f} мм, φ={best.phi_deg:.3f}°;
ρ₁={best.rho_soft_ohm_m:.3f}, ρ₂={best.rho_lungs_ohm_m:.3f} Ом·м.
Геометрическая глубина до ближайшей вершины лёгкого: {best.h_centre_mm:.3f} мм.</p>
<p>Поиск остановлен после {len(df)} оценок по бюджету. Оптимальность места и точность его координат не доказаны.
Сопротивления и угол подбирались совместно с положением; все центры исследованы неодинаково подробно.</p>
<details><summary>Лучший проверенный вариант в каждом из {len(centres)} допустимых центров</summary>
{table.to_html(index=False, float_format=lambda x:f'{x:.3f}', border=0)}</details>
<details><summary>Все {len(df)} оценок, включая отклонённые</summary>
<p>Для отклонённых вариантов J не показан: служебный штраф не является FEM-результатом.</p>
{df.assign(objective=df.objective.where(df.status=='evaluated'))[['evaluation',*UV,'phi_deg','rho_soft_ohm_m','rho_lungs_ohm_m','objective','status']].to_html(index=False,float_format=lambda x:f'{x:.3f}',na_rep='—',border=0)}</details>
<p class="muted">Локальная визуализация существующего математического расчёта. FEM не пересчитывался.
Поверхность кожи — все {len(faces):,} граней исходной расчётной сетки, без упрощения.
Лёгкие и кости упрощены только для отображения. Маркеры подняты над кожей на 0,8 мм для видимости;
координаты в подсказках и таблице исходные. Совпадающие центры объединены с округлением u и v до 10⁻⁸ мм; это устранение численных дублей, не точность положения. u и v — смещения в опорной касательной плоскости до проекции,
не геодезические расстояния. h может включать кость и не равен эффективной толщине двуслойной модели.</p>
</main></html>'''
    out = HERE / f'search_map_{tag}.html'
    out.write_text(page, encoding='utf-8')

    # Static scientific preview: two real 3D views plus the readable uv map.
    # VTK renders geometry; matplotlib only lays out the resulting panels.
    cmap = plt.get_cmap('RdYlBu_r')
    norm = Normalize(vmin, vmax)
    rgb = np.round(cmap(norm(centres.objective.to_numpy()))[:, :3]*255).astype(np.uint8)
    screenshots = []
    for local in (False, True):
        pl = pv.Plotter(off_screen=True, window_size=(800, 820))
        pl.set_background('white')
        if local:
            viewbody = body.clip_box([centre[0]-40,centre[0]+40,centre[1]-40,centre[1]+40,centre[2]-40,centre[2]+40], invert=False)
            pl.add_mesh(viewbody, color='#d5d0c5', opacity=.53, smooth_shading=True)
            viewlung = layers['lungs'].clip_box([centre[0]-45,centre[0]+45,centre[1]-40,centre[1]+40,centre[2]-40,centre[2]+40], invert=False)
            if viewlung.n_cells:
                pl.add_mesh(viewlung, color='#b1d9e5', opacity=.22, smooth_shading=True)
            focus = centre
        else:
            pl.add_mesh(body, color='#cec9c0', opacity=.32, smooth_shading=True)
            pl.add_mesh(layers['lungs'], color='#91cbdc', opacity=.22, smooth_shading=True)
            focus = (nodes.min(axis=0)+nodes.max(axis=0))/2
        cloud = pv.PolyData(shown_xyz)
        cloud['rgb'] = rgb
        pl.add_mesh(cloud, scalars='rgb', rgb=True, style='points',
                    point_size=19 if local else 9, render_points_as_spheres=True)
        if len(fxyz):
            # 3D static crosses built from lines, with constant size in mm.
            u = geo['reference_u'].ravel(); v = geo['reference_v'].ravel()
            for p in fxyz:
                for direction in (u+v,u-v):
                    pl.add_lines(np.vstack([p-.7*direction,p+.7*direction]), color='#60676c', width=3)
        pl.add_mesh(pv.PolyData([bshow]), color='#111827', point_size=24 if local else 11,
                    style='points', render_points_as_spheres=False)
        # Mark best by a white centre over the larger dark point.
        pl.add_mesh(pv.PolyData([bshow+.1*normal]), color='#ffffff', point_size=10 if local else 4,
                    style='points', render_points_as_spheres=False)
        pl.camera_position = [focus + normal*900, focus, [0,0,1]]
        pl.enable_parallel_projection()
        pl.camera.parallel_scale = 38 if local else 280
        pl.enable_depth_peeling()
        screenshot_path = HERE / f'{"local" if local else "body"}_view_{tag}.png'
        pl.screenshot(str(screenshot_path)); pl.close()
        screenshots.append(screenshot_path)
    f = plt.figure(figsize=(16,6.5), layout='constrained')
    gs = f.add_gridspec(1,3,width_ratios=[1,1,1.1])
    for k,p in enumerate(screenshots):
        ax = f.add_subplot(gs[0,k]);ax.imshow(plt.imread(p));ax.axis('off')
        ax.set_title(['Положение участка на теле','Участок 3D, увеличено'][k],fontsize=14)
    ax = f.add_subplot(gs[0,2])
    scatter = ax.scatter(centres.centre_u_mm,centres.centre_v_mm,c=centres.objective,
                         cmap=cmap,norm=norm,s=95,edgecolors='#172636',linewidths=.7,zorder=3)
    ax.scatter(failed.centre_u_mm,failed.centre_v_mm,c='#777777',marker='x',s=80,linewidths=2,zorder=4)
    ax.scatter([best.centre_u_mm],[best.centre_v_mm],facecolors='none',edgecolors='black',marker='*',s=340,zorder=5)
    ax.annotate(f'Лучшая №{int(best.evaluation)}\nJ={best.objective:.2f}',(best.centre_u_mm,best.centre_v_mm),
                xytext=(best.centre_u_mm-10,best.centre_v_mm-8),arrowprops=dict(arrowstyle='->'),fontsize=11)
    ax.set(xlabel='u, мм',ylabel='v, мм',title='Цвет: лучший проверенный J',aspect='equal')
    ax.grid(alpha=.2);ax.set_xlim(df.centre_u_mm.min()-4,df.centre_u_mm.max()+4)
    ax.set_ylim(df.centre_v_mm.min()-5,df.centre_v_mm.max()+5)
    f.colorbar(scatter,ax=ax,shrink=.6,pad=.03,label='J · меньше — лучше')
    f.suptitle(f'Локальный поиск: {len(df)} оценок → {len(counts)} центра → {len(centres)} с допустимыми вариантами',fontsize=17)
    f.supxlabel('Цвета показывают совпадение с экспериментом с учётом штрафов. ρ₁, ρ₂ и φ менялись.\n'
                f'Серые кресты — только отклонённые варианты. Между точками нет расчёта. RMSE лучшего: {fit.rms_residual_ohm:.2f} Ом.',fontsize=12)
    preview = HERE / f'search_map_{tag}.png'
    f.savefig(preview,dpi=160,facecolor='white')
    f.savefig(HERE / f'search_map_preview_{tag}.png',dpi=110,facecolor='white')
    plt.close(f)
    # Derived provenance is separate: the published inverse contract is immutable.
    report = dict(result_tag=tag, evaluations=len(df), valid_evaluations=len(valid),
                  rejected_evaluations=len(df)-len(valid), centres=len(counts), valid_centres=len(centres),
                  rejected_only_centres=len(failed), distinct_poses=len(df[UV+['phi_deg']].drop_duplicates()),
                  best_evaluation=int(best.evaluation), best_objective=float(best.objective),
                  best_rmse_ohm=float(fit.rms_residual_ohm),
                  search_span_uv_mm=(df[UV].max()-df[UV].min()).to_dict(),
                  colour_range=[float(vmin),float(vmax)], colour_metric='minimum_observed_composite_objective_per_uv',
                  interpolation=False, fixed_rho_spatial_scan=False, new_fem_evaluations=0,
                  duplicate_key_rounding_mm=1e-8, original_exact_float_uv_keys=len(raw[UV].drop_duplicates()),
                  verified_contract_files=checked, geometry_checks=json.loads((HERE/f'geometry_checks_{tag}.json').read_text()),
                  anatomical_display_layers=layer_inputs,
                  files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in
                         [contract_path,data_path,HERE/f'search_surface_{tag}.mat',Path(__file__),out,preview]])
    (HERE / f'search_map_checks_{tag}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['verified_contract_files','files','anatomical_display_layers']},ensure_ascii=False,indent=2))
    print(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default='auditfix_20260908')
    args = parser.parse_args()
    if not args.tag.replace('_','').replace('-','').isalnum():
        parser.error('Invalid tag')
    main(args.tag)
