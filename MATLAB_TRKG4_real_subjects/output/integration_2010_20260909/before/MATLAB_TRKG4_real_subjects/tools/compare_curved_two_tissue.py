"""Reproduce section 11.3 and compare explicit two-material geometries.

Run prepare after MATLAB Stage=export; finish after Stage=solve. Nothing is fit.
"""
from pathlib import Path
import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import j0
from scipy.io import loadmat, savemat

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / 'notebooks/TRKG4_inverse_inhale_results.ipynb'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def image_kernel(d_mm, h_mm, rho1, rho2):
    h = np.asarray(h_mm, dtype=float)
    n = np.arange(1, 257).reshape((-1,) + (1,) * h.ndim)
    k = (rho2 - rho1) / (rho2 + rho1)
    return 1000 / d_mm + 2000 * np.sum(k**n / np.hypot(d_mm, 2 * n * h), axis=0)


def slab_green(d_mm, h_mm, lung_mm, rho1, rho2):
    """Exact planar point-source Green function for rho1/rho2/rho1.

    First two layers are finite; final rho1 region is a half-space.
    Subtract the homogeneous singularity before Hankel integration.
    """
    r, h, t = np.asarray([d_mm, h_mm, lung_mm], dtype=float) / 1000
    if not (r > 0 and h > 0 and t >= 0 and rho1 > 0 and rho2 > 0):
        raise ValueError('Positive r, h, rho and nonnegative lung thickness required')
    primary = rho1 / (2 * np.pi * r)
    if t == 0 or rho1 == rho2:
        return primary

    def correction(lam):
        th2 = np.tanh(lam * t)
        impedance2 = rho2 * (rho1 + rho2 * th2) / (rho2 + rho1 * th2)
        th1 = np.tanh(lam * h)
        impedance1 = rho1 * (impedance2 + rho1 * th1) / (rho1 + impedance2 * th1)
        return (impedance1 - rho1) * j0(lam * r) / (2 * np.pi)

    value, err = quad(correction, 0, 24 / h, epsabs=2e-9, epsrel=2e-10, limit=800)
    if err > 1e-6:
        raise ArithmeticError(f'Hankel integration error {err}')
    return primary + value


def tetrapolar(L, green):
    a, m, n, b = -L / 2, -L / 4, L / 4, L / 2
    return green(a, m) - green(b, m) - green(a, n) + green(b, n)



def stack_green(d_mm, boundaries_mm, rho1, rho2):
    """Exact planar alternating soft/lung stack, final soft half-space."""
    depths = np.asarray(boundaries_mm, dtype=float)
    if len(depths) % 2 or not np.all(np.diff(np.r_[0., depths]) > 0):
        raise ValueError('Ordered paired lung entries/exits required')
    if not len(depths):
        return rho1 * 1000 / (2 * np.pi * d_mm)
    thickness = np.diff(np.r_[0., depths]) / 1000
    rhos = np.resize([rho1, rho2], len(depths))
    r = d_mm / 1000
    def correction(lam):
        impedance = rho1
        for h, rho in zip(thickness[::-1], rhos[::-1]):
            th = np.tanh(lam * h)
            impedance = rho * (impedance + rho * th) / (rho + impedance * th)
        return (impedance - rho1) * j0(lam * r) / (2 * np.pi)
    value, err = quad(correction, 0, 24 / thickness[0], epsabs=2e-9, epsrel=2e-10, limit=800)
    if err > 1e-6:
        raise ArithmeticError(f'Stack Hankel integration error {err}')
    return rho1 / (2 * np.pi * r) + value


def all_interval_curve(g, ns, step_mm):
    """Heuristic path average of LOCAL planar stacks; not curved-domain PDE."""
    s_grid = np.arange(-70., 70. + step_mm / 2, step_mm)
    boundaries = []
    for s in s_grid:
        skin_hits = ns['contour_intersections_at_s'](ns['section_segments']['body'], s)
        skin = min(skin_hits, key=abs)
        lung_hits = ns['contour_intersections_at_s'](ns['section_segments']['lungs'], s)
        depths = np.asarray(ns['unique_sorted_depths']([x for x in lung_hits if x > skin + 1.])) - skin
        if len(depths) % 2:
            raise ValueError(f'Odd number of section intersections at s={s}')
        boundaries.append(depths)
    rho1, rho2 = g['rho_soft_ohm_m'], g['rho_lung_ohm_m']
    result = []
    for L in g['L_mm']:
        kernels = {d: np.array([stack_green(d, depths, rho1, rho2) for depths in boundaries])
                   for d in [L / 4, 3 * L / 4]}
        def pair(a, b):
            d = abs(a - b)
            x = np.r_[min(a, b), s_grid[(s_grid > min(a, b)) & (s_grid < max(a, b))], max(a, b)]
            return np.trapezoid(np.interp(x, s_grid, kernels[d]), x) / d
        result.append(tetrapolar(L, pair))
    return np.array(result)

def section_namespace(tag):
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['TRKG4_RESULT_TAG'] = tag
    os.environ['TRKG4_REPORT_SKIP_DICOM'] = '1'
    namespace = {'__name__': '__two_tissue_section__'}
    previous = Path.cwd()
    try:
        os.chdir(NB.parent)
        notebook = json.loads(NB.read_text(encoding='utf-8'))
        with contextlib.redirect_stdout(io.StringIO()):
            for cell in notebook['cells']:
                if cell['cell_type'] != 'code':
                    continue
                source = ''.join(cell['source'])
                exec(compile(source, str(NB), 'exec'), namespace)
                namespace['display'] = lambda *a, **k: None
                namespace['show_plotly'] = lambda *a, **k: None
                if 'def z_adapted(' in source:
                    break
            else:
                raise RuntimeError('Section 11.3 code not found')
    finally:
        os.chdir(previous)
    return namespace


def classify_lung(centres, stl, centre, axis, inward):
    """Closed STL occupancy and union of rays after first lung entry.

    Extension is inside the existing body mesh. It removes ALL exits along one
    global inward direction, not exclusively the anatomical diaphragm.
    """
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(stl))
    reader.Update()
    surface = reader.GetOutput()
    if not vtk.vtkSelectEnclosedPoints.IsSurfaceClosed(surface):
        raise ValueError('Lung STL is not closed')
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(centres), deep=True))
    cloud = vtk.vtkPolyData()
    cloud.SetPoints(points)
    enclosed = vtk.vtkSelectEnclosedPoints()
    enclosed.SetInputData(cloud)
    enclosed.SetSurfaceData(surface)
    enclosed.SetTolerance(1e-8)
    enclosed.SetCheckSurface(True)
    enclosed.Update()
    closed = vtk_to_numpy(enclosed.GetOutput().GetPointData().GetArray('SelectedPoints')).astype(bool)
    normal = np.cross(axis, inward)
    frame = np.column_stack([axis, normal, inward])
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-10)
    lung_coords = (vtk_to_numpy(surface.GetPoints().GetData()) - centre) @ frame
    coords = (centres - centre) @ frame
    lower, upper = lung_coords.min(axis=0), lung_coords.max(axis=0)
    candidates = np.flatnonzero((~closed) & (coords[:, 2] >= lower[2]) &
        np.all(coords[:, :2] >= lower[:2], axis=1) & np.all(coords[:, :2] <= upper[:2], axis=1))
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()
    extended = closed.copy()
    value, sub_id, cell_id = vtk.reference(0.), vtk.reference(0), vtk.reference(0)
    hit, param = [0., 0., 0.], [0., 0., 0.]
    print(f'Closed lung: {closed.sum()} elements; extension queries: {len(candidates)}', flush=True)
    for count, i in enumerate(candidates):
        end = centres[i]
        start = end + (lower[2] - 10 - coords[i, 2]) * inward
        extended[i] = bool(locator.IntersectWithLine(start, end, 1e-7, value, hit, param, sub_id, cell_id))
        if count and count % 100000 == 0:
            print(f'Extension {count}/{len(candidates)}', flush=True)
    assert closed.any() and np.all(extended[closed]) and np.any(extended & ~closed)
    return closed, extended


def prepare(out, tag):
    before = out / 'before'
    before.mkdir(exist_ok=True)
    for file in [NB, NB.with_name('build_trkg4_inverse_report.py')]:
        if not (before / file.name).exists():
            shutil.copy2(file, before / file.name)
    ns = section_namespace(tag)
    s = np.asarray(ns['s_profile'])
    skin = np.asarray(ns['skin_profile'])
    branches = np.asarray(ns['lung_branch_profiles'])
    h1, t = branches[:, 0] - skin, branches[:, 1] - branches[:, 0]
    use = np.abs(s) <= 70
    assert np.all(np.isfinite(h1[use])) and np.all(h1[use] > 0)
    assert np.all(np.isfinite(t[use])) and np.all(t[use] > 0)
    profile = pd.DataFrame({'s_mm': s, 'skin_depth_mm': skin,
        'entry_depth_mm': branches[:, 0], 'exit_depth_mm': branches[:, 1],
        'h_entry_mm': h1, 'lung_thickness_mm': t,
        'crossing_count': np.isfinite(branches).sum(axis=1)})
    for i in range(branches.shape[1]):
        profile[f'boundary_{i + 1}_depth_mm'] = branches[:, i]
    profile.to_csv(out / 'profile.csv', index=False)
    electrode_file = ROOT / 'output' / f'nik_trkg4_inverse_inhale_electrodes_{tag}.csv'
    stl = ROOT / 'data/nik/stl/lungs_solid_v3_volume_fill.stl'
    g = {'schema_version': 1, 'result_tag': tag,
        'rho_soft_ohm_m': ns['rho1_fem'], 'rho_lung_ohm_m': ns['rho2_fem'],
        'L_mm': np.asarray(ns['L_values']).tolist(),
        'centre_mm': np.asarray(ns['centre']).tolist(),
        'axis': np.asarray(ns['axis']).tolist(), 'inward': np.asarray(ns['inward']).tolist(),
        'h_surface_mm': ns['h_centre_surface_mm'],
        'polynomial_coefficients': np.asarray(ns['coefficients']).tolist(),
        'original_adapted_Z_ohm': np.asarray(ns['Z_adapt_direct']).tolist(),
        'notebook_at_extraction_sha256': sha(NB), 'electrodes_sha256': sha(electrode_file),
        'lung_stl_sha256': sha(stl), 'profile_sha256': sha(out / 'profile.csv'),
        'assumptions': ['All body outside lung is soft, including bone, heart and below diaphragm.',
            'Fixed resistivities inherited from the four-material fit; no refitting.',
            'Finite analytic variant uses the FIRST lung interval only.',
            'Extended FEM removes all distal exits along the section inward direction; not a diaphragm-only intervention.',
            'Lung material assigned by tetrahedron centroid inside closed STL.']}
    mesh = loadmat(out / 'mesh_samples.mat', simplify_cells=True)
    closed, extended = classify_lung(mesh['centres_mm'], stl,
        np.asarray(g['centre_mm']), np.asarray(g['axis']), np.asarray(g['inward']))
    old = np.asarray(mesh['old_lung_mask'], dtype=bool)
    vol = mesh['element_volume_mm3']
    g['mesh_mask_diagnostics'] = {'element_count': len(closed),
        'closed_count': int(closed.sum()), 'extended_count': int(extended.sum()),
        'closed_lung_litres': float(vol[closed].sum() / 1e6),
        'extended_lung_litres': float(vol[extended].sum() / 1e6),
        'old_lung_disagreement_count': int(np.count_nonzero(closed != old)),
        'old_lung_disagreement_litres': float(vol[closed != old].sum() / 1e6),
        'vtk_closed_surface_check': True, 'vtk_enclosed_tolerance_fraction': 1e-8,
        'vtk_extension_ray_tolerance_mm': 1e-7}
    savemat(out / 'lung_masks.mat', {'closed_lung_mask': closed, 'extended_lung_mask': extended,
        'mesh_sha256': mesh['mesh_sha256']})
    (out / 'geometry.json').write_text(json.dumps(g, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(g, ensure_ascii=False, indent=2), flush=True)


def analytic_curves(g, profile, samples=401):
    rho1, rho2 = g['rho_soft_ohm_m'], g['rho_lung_ohm_m']
    prefactor = rho1 / (2 * np.pi)
    h0 = float(np.interp(0, profile.s_mm, profile.h_entry_mm))
    t0 = float(np.interp(0, profile.s_mm, profile.lung_thickness_mm))

    def pair(a, b, mode):
        d = abs(a - b)
        s = np.linspace(min(a, b), max(a, b), samples)
        h = np.polynomial.polynomial.polyval(s / 70, g['polynomial_coefficients'])
        if mode == 'adapted_mean_h':
            return prefactor * image_kernel(d, np.trapezoid(h, s) / d, rho1, rho2)
        if mode == 'adapted_mean_kernel':
            return prefactor * np.trapezoid(image_kernel(d, h, rho1, rho2), s) / d
        if mode == 'adapted_finite_interval':
            hi = np.interp(s, profile.s_mm, profile.h_entry_mm)
            ti = np.interp(s, profile.s_mm, profile.lung_thickness_mm)
            return slab_green(d, np.trapezoid(hi, s) / d, np.trapezoid(ti, s) / d, rho1, rho2)
        if mode == 'planar_centre_finite':
            return slab_green(d, h0, t0, rho1, rho2)
        hfix = g['h_surface_mm'] if mode == 'planar_surface_h' else h0
        return prefactor * image_kernel(d, hfix, rho1, rho2)

    result = {'L_mm': g['L_mm']}
    for mode in ['planar_surface_h', 'adapted_mean_h', 'adapted_mean_kernel',
                 'planar_centre_infinite', 'planar_centre_finite', 'adapted_finite_interval']:
        result[mode] = [tetrapolar(L, lambda a, b: pair(a, b, mode)) for L in g['L_mm']]
    return pd.DataFrame(result)


def finish(out):
    g = json.loads((out / 'geometry.json').read_text(encoding='utf-8'))
    contract = json.loads((out / 'fem_contract.json').read_text(encoding='utf-8'))
    for key, name in [('geometry_sha256', 'geometry.json'), ('mask_sha256', 'lung_masks.mat'),
                      ('fem_csv_sha256', 'fem.csv')]:
        assert sha(out / name) == contract[key], f'Stale {name}'
    assert sha(out / 'profile.csv') == g['profile_sha256']
    profile = pd.read_csv(out / 'profile.csv')
    curves = analytic_curves(g, profile)
    finer = analytic_curves(g, profile, samples=801)
    sample_diff = float(np.max(np.abs(curves.iloc[:, 1:].values - finer.iloc[:, 1:].values)))
    assert sample_diff < 0.002
    reproduction_error = float(np.max(np.abs(curves.adapted_mean_h - g['original_adapted_Z_ohm'])))
    assert reproduction_error < 1e-6
    curves = curves.merge(pd.read_csv(out / 'fem.csv'), on='L_mm', validate='one_to_one')
    # The accepted nine-record size series used by section 11.3; cross-check below.
    ns = section_namespace(g['result_tag'])
    baseline = ns['comparison'][['L_mm', 'Z_experiment_ohm', 'Z_FEM_ohm']].copy()
    baseline = baseline.rename(columns={'Z_FEM_ohm': 'old_four_material_FEM_ohm'})
    curves = curves.merge(baseline, on='L_mm', validate='one_to_one')
    # Exact planar stack at the centre and a heuristic using every section crossing.
    central = profile.loc[np.isclose(profile.s_mm, 0)].iloc[0]
    depths = np.array([central[c] - central.skin_depth_mm for c in profile.columns
                       if c.startswith('boundary_') and np.isfinite(central[c])])
    curves['planar_centre_all_intervals'] = [tetrapolar(L, lambda a, b:
        stack_green(abs(a-b), depths, g['rho_soft_ohm_m'], g['rho_lung_ohm_m'])) for L in g['L_mm']]
    print('All-interval section kernels, 2 mm and 1 mm sampling...', flush=True)
    all_coarse = all_interval_curve(g, ns, 2.)
    all_fine = all_interval_curve(g, ns, 1.)
    all_interval_difference = float(np.max(np.abs(all_fine - all_coarse)))
    assert all_interval_difference < 0.05, f'Refine section sampling: {all_interval_difference}'
    curves['adapted_all_intervals'] = all_fine
    names = {
        'planar_surface_h': 'Плоская 2 слоя: h до ближайшей поверхности',
        'adapted_mean_h': '11.3: среднее h по паре',
        'adapted_mean_kernel': 'Среднее ядро по h(s)',
        'planar_centre_infinite': 'Плоская 2 слоя: h по центральному лучу',
        'planar_centre_finite': 'Плоская 3 слоя: первый интервал в центре',
        'adapted_finite_interval': '3 слоя: средние h и толщина лёгкого по паре',
        'planar_centre_all_intervals': 'Плоская: все интервалы центрального луча',
        'adapted_all_intervals': 'Среднее ядро: все интервалы сечения',
        'Z_closed_lung_CEM_ohm': 'FEM: замкнутое лёгкое, две ткани',
        'Z_extended_lung_CEM_ohm': 'FEM: лёгкое продолжено после первого входа',
    }
    metrics = []
    for key, name in names.items():
        residual = curves[key] - curves.Z_closed_lung_CEM_ohm
        experiment_residual = curves[key] - curves.Z_experiment_ohm
        metrics.append({'key': key, 'model': name,
            'RMSE_to_closed_FEM_ohm': float(np.sqrt(np.mean(residual**2))),
            'max_abs_to_closed_FEM_ohm': float(np.max(np.abs(residual))),
            'RMSE_to_experiment_ohm': float(np.sqrt(np.mean(experiment_residual**2)))})
    curves.to_csv(out / 'comparison.csv', index=False)
    pd.DataFrame(metrics).to_csv(out / 'metrics.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.close('all')
    plt.rcParams.update({'font.size': 11})
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={'height_ratios': [1.4, 1]})
    display_keys = ['Z_closed_lung_CEM_ohm', 'Z_extended_lung_CEM_ohm', 'planar_surface_h',
                    'adapted_mean_h', 'adapted_mean_kernel', 'adapted_finite_interval', 'adapted_all_intervals']
    for key in display_keys:
        style = 'o-' if key.startswith('Z_') else '--'
        line, = axes[0].plot(curves.L_mm, curves[key], style, label=names[key], linewidth=1.8)
        if key != 'Z_closed_lung_CEM_ohm':
            axes[1].plot(curves.L_mm, curves[key]-curves.Z_closed_lung_CEM_ohm,
                         style, color=line.get_color())
    axes[0].plot(curves.L_mm, curves.Z_experiment_ohm, 'kx', ms=8, label='Измерение: отдельный ориентир')
    axes[0].set(ylabel='Z, Ом', title='Пример 11.3: одинаковые два сопротивления, без повторной подгонки')
    axes[0].legend(fontsize=9)
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].set(xlabel='Номинальный размер сборки L, мм', ylabel='Z модели − Z двухтканного FEM, Ом')
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / 'comparison.png', dpi=150)
    plt.close(fig)
    # Rasterize actual paired contour crossings; do not join changing branch indices.
    fig, ax = plt.subplots(figsize=(12, 6))
    sx = np.arange(-80., 80.01, .5)
    dz = np.arange(0., 370.01, .5)
    labels = np.full((len(dz), len(sx)), np.nan)
    for j, x in enumerate(sx):
        skin_hits = np.asarray(ns['contour_intersections_at_s'](ns['section_segments']['body'], x))
        skin = min(skin_hits, key=abs)
        far_skin = np.max(skin_hits) - skin
        inside_body = dz <= far_skin
        hits = ns['contour_intersections_at_s'](ns['section_segments']['lungs'], x)
        depths = np.asarray(ns['unique_sorted_depths']([v for v in hits if v > skin+1.])) - skin
        inside_lung = np.sum(dz[:,None] > depths[None,:],axis=1) % 2 == 1
        labels[inside_body,j] = inside_lung[inside_body].astype(float)
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    ax.imshow(labels, origin='lower', extent=[-80,80,0,370], aspect='auto',
              interpolation='nearest', cmap=ListedColormap(['#ecd8bc','#a6d2e9']), vmin=0, vmax=1)
    ax.axvline(-70, color='gray', linestyle=':'); ax.axvline(70, color='gray', linestyle=':')
    ax.set(xlim=(-80,80), ylim=(370,-5), xlabel='Координата вдоль сборки s, мм',
           ylabel='Глубина вдоль inward от местной кожи, мм',
           title='Реальное сечение 11.3 после выпрямления кожи для графика\nВсе ткани вне лёгочной маски объединены; белое — вне тела')
    ax.legend(handles=[Patch(color='#ecd8bc',label='Мягкие ткани'),Patch(color='#a6d2e9',label='Лёгкое')])
    fig.tight_layout(); fig.savefig(out / 'finite_lung_profile.png', dpi=150); plt.close(fig)
    summary = {'schema_version': 1, 'status': 'model_comparison_not_validation',
        'geometry': g, 'models': metrics,
        'reproduced_11_3_max_difference_ohm': reproduction_error,
        'path_samples_401_to_801_max_difference_ohm': sample_diff,
        'all_intervals_step_2_to_1_mm_max_difference_ohm': all_interval_difference,
        'central_first_entry_mm': float(central.h_entry_mm),
        'central_first_lung_thickness_mm': float(central.lung_thickness_mm),
        'extended_minus_closed_FEM_ohm': (curves.Z_extended_lung_CEM_ohm-curves.Z_closed_lung_CEM_ohm).tolist(),
        'central_finite_minus_infinite_ohm': (curves.planar_centre_finite-curves.planar_centre_infinite).tolist(),
        'old_four_material_minus_new_two_material_ohm': (curves.old_four_material_FEM_ohm-curves.Z_closed_lung_CEM_ohm).tolist(),
        'reciprocity_checks': contract['reciprocity_checks'],
        'source_sha256': sha(__file__),
        'artifact_sha256': {name: sha(out/name) for name in ['geometry.json','profile.csv','fem_contract.json','fem.csv',
                           'comparison.csv','metrics.csv','comparison.png','finite_lung_profile.png']},
        'limitations': [
            'Analytic electrodes are nominal point contacts on a plane; FEM has finite CEM contacts on the actual body.',
            'Averaging pair depths or local kernels is a heuristic, not a curved-interface solution.',
            'Only the closed/extended FEM pair isolates the stated geometry intervention with identical contacts.',
            'Extension changes all downstream regions along one direction, not solely the diaphragm.',
            'One existing mesh; mesh/domain/contact convergence has not been established.',
            'Fixed real scalar resistivities; no complex admittivity or anisotropy in this experiment.',
            'Experimental residual is descriptive, not evidence that two-tissue parameters have been identified or validated.'
        ]}
    (out / 'comparison_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(curves.round(4).to_string(index=False), flush=True)
    print(pd.DataFrame(metrics).round(4).to_string(index=False), flush=True)
    print('All-interval 2 to 1 mm max difference:', all_interval_difference, flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'finish', 'matlab-export', 'matlab-solve'])
    parser.add_argument('--tag', default='auditfix_20260908')
    args = parser.parse_args()
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.tag):
        parser.error('Invalid tag')
    out = ROOT / 'output' / f'two_tissue_{args.tag}'
    out.mkdir(parents=True, exist_ok=True)
    if args.stage.startswith('matlab-'):
        import subprocess
        stage = args.stage.split('-')[1]
        command = f"run_pipeline('Profile','two_tissue_comparison','ResultTag','{args.tag}','Stage','{stage}');"
        with (out / f'matlab_{stage}.log').open('w', encoding='utf-8') as log:
            process = subprocess.run(['matlab', '-batch', command], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        print('MATLAB', stage, 'exit', process.returncode, flush=True)
        raise SystemExit(process.returncode)
    if args.stage == 'prepare':
        prepare(out, args.tag)
    else:
        finish(out)


if __name__ == '__main__':
    main()
