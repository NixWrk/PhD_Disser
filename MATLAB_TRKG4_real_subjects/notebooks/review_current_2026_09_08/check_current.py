"""Isolated re-review: execute current report and check layered-model mathematics."""
from pathlib import Path
import ast
import contextlib
import hashlib
import io
import json
import os
import runpy
import time
import traceback

import numpy as np
from scipy.integrate import quad
from scipy.special import j0

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.chdir(ROOT)
os.environ['MPLBACKEND'] = 'Agg'
os.environ.pop('KALMYKOV_DICOM_DIR', None)
os.environ['TRKG4_RESULT_TAG'] = 'auditfix_20260908'
os.environ['TRKG4_REPORT_TARGET'] = str(HERE / 'generated.ipynb')
target = ROOT / 'notebooks/TRKG4_inverse_inhale_results.ipynb'
original = json.loads(target.read_text(encoding='utf-8'))
with contextlib.redirect_stdout(io.StringIO()):
    runpy.run_path(str(ROOT / 'notebooks/build_trkg4_inverse_report.py'))
fresh = json.loads((HERE / 'generated.ipynb').read_text(encoding='utf-8'))
result = {'scope': 'current notebook, without DICOM or FEM optimisation',
          'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [target, ROOT / 'notebooks/build_trkg4_inverse_report.py']},
          'generator_sources_match': [(c['cell_type'], c['source']) for c in original['cells']]
          == [(c['cell_type'], c['source']) for c in fresh['cells']], 'cells': []}
ns = {'__name__': '__review__'}
for index, cell in enumerate(original['cells']):
    if cell['cell_type'] != 'code':
        continue
    start = time.perf_counter()
    record = {'cell_index': index, 'status': 'passed'}
    try:
        source = ''.join(cell['source'])
        ast.parse(source)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source, f'cell_{index}', 'exec'), ns)
        for name in ['display', 'show_plotly']:
            if name in ns:
                ns[name] = lambda *a, **kw: None
        if 'plt' in ns:
            ns['plt'].close('all')
    except Exception:
        record.update(status='failed', traceback=traceback.format_exc())
    record['seconds'] = time.perf_counter() - start
    result['cells'].append(record)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    if record['status'] == 'failed':
        break


def transform(lam, rho, thickness_m):
    """Resistivity transform of isotropic planar layers; last layer infinite."""
    value = float(rho[-1])
    for r, h in zip(rho[-2::-1], thickness_m[::-1]):
        t = np.tanh(lam * h)
        value = r * (value + r * t) / (r + value * t)
    return value


def layered_z(L_m, rho, thickness_m):
    """Subtract homogeneous primary term; quadrature of decaying correction.

    Intended as an independent benchmark for positive finite top thickness,
    moderate contrast, and the explicitly tested geometries, not production.
    """
    a, b = L_m / 4, 3 * L_m / 4
    if len(rho) == 1:
        return rho[0] / np.pi * (1 / a - 1 / b), 0.
    cutoff = 24 / thickness_m[0]
    correction, err = quad(lambda lam: (transform(lam, rho, thickness_m)-rho[0])
                           * (j0(lam*a)-j0(lam*b)), 0, cutoff,
                           epsabs=1e-10, epsrel=1e-10, limit=1000)
    return (rho[0]*(1/a-1/b)+correction)/np.pi, err/np.pi


if all(c['status'] == 'passed' for c in result['cells']):
    for key in ['h_centre_surface_mm', 'h_centre_vertex_mm', 'h_fit_rmse']:
        if key in ns:
            result[key] = float(ns[key])
    for key in ['coefficients', 'delta_rho12']:
        if key in ns:
            result[key] = np.asarray(ns[key]).tolist()
    for key in ['direct_error_table', 'calibrated_table', 'objective_parts']:
        if key in ns:
            result[key] = ns[key].to_dict(orient='records')
    checks = []
    for L in [50., 90., 140.]:
        for r1, r2 in [(4., 20.), (20., 4.), (5., 5.)]:
            ref = float(ns['z_two_layer'](L, r1, r2, 25))
            z2, err = layered_z(L/1000, [r1, r2], [.025])
            # Equal adjacent layers merge exactly; use two independent merges.
            z3a, _ = layered_z(L/1000, [r1, r1, r2], [.010, .015])
            z3b, _ = layered_z(L/1000, [r1, r2, r2], [.025, .012])
            e = max(abs(z2-ref), abs(z3a-ref), abs(z3b-ref))
            assert e < 1e-8, (L, r1, r2, e)
            checks.append({'L_mm': L, 'rho_ohm_m': [r1, r2],
                           'max_2L_3L_series_difference_ohm': e,
                           'quadrature_estimate_ohm': err})
    # Removing the middle layer recovers the outer two layers.
    z3, _ = layered_z(.09, [5., 12., 24.], [.025, 0.])
    z2, _ = layered_z(.09, [5., 24.], [.025])
    assert abs(z3-z2) < 1e-10
    result['flat_layer_checks'] = checks
    result['zero_middle_layer_error_ohm'] = abs(z3-z2)
    result['synthetic_three_layer_example'] = {
        'status': 'synthetic isotropic half-space; not fitted tissue parameters',
        'rho_ohm_m': [5., 12., 24.], 'thickness_mm': [15., 10.],
        'Z_ohm': {str(L): layered_z(L/1000, [5., 12., 24.], [.015, .010])[0]
                  for L in [50., 90., 140.]}}
    # A change away from the montage plane affects Z, despite unchanged h(s).
    # This is a local first-order conductivity kernel, not a finite inclusion solve.
    p = np.array([0., .025, .025])
    sources = np.array([[-.035, 0., 0.], [.035, 0., 0.],
                        [-.0175, 0., 0.], [.0175, 0., 0.]])
    def grad_dipole(a, b, rho=5.):
        ra, rb = p-a, p-b
        return rho/(2*np.pi)*(-ra/np.linalg.norm(ra)**3+rb/np.linalg.norm(rb)**3)
    kernel = -float(grad_dipole(*sources[:2]) @ grad_dipole(*sources[2:]))
    result['off_plane_sensitivity_counterexample'] = {
        'point_m': p.tolist(), 'rho_ohm_m': 5., 'L_mm': 70.,
        'kernel_ohm_per_S_per_m2': kernel,
        'meaning': 'deltaZ = integral(kernel * delta_sigma dV); unit currents; nonzero outside y=0'}
    assert abs(kernel) > 0

(HERE / 'current_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
assert all(c['status'] == 'passed' for c in result['cells'])
assert result['generator_sources_match']
print('PASS', len(result['cells']), 'cells; independent 2L/3L planar checks', flush=True)
