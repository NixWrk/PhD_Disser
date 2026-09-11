"""Read-only numerical audit; writes only into this audit directory."""
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
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.chdir(ROOT)
os.environ['MPLBACKEND'] = 'Agg'
os.environ.pop('KALMYKOV_DICOM_DIR', None)  # no clinical imagery needed for code audit
target = ROOT / 'notebooks/TRKG4_inverse_inhale_results.ipynb'
notebook = json.loads(target.read_text(encoding='utf-8'))
result = {'scope': 'Code execution with display suppressed; no DICOM; no FEM optimisation',
          'sha256': {str(target.relative_to(ROOT)): hashlib.sha256(target.read_bytes()).hexdigest()},
          'cells': []}
generated = HERE / 'generated_source_check.ipynb'
os.environ['TRKG4_REPORT_TARGET'] = str(generated)
with contextlib.redirect_stdout(io.StringIO()):
    runpy.run_path(str(ROOT / 'notebooks/build_trkg4_inverse_report.py'))
fresh = json.loads(generated.read_text(encoding='utf-8'))
result['generator_sources_match'] = (
    [(c['cell_type'], ''.join(c['source'])) for c in notebook['cells']] ==
    [(c['cell_type'], ''.join(c['source'])) for c in fresh['cells']])
generated.unlink()  # remove the temporary source-comparison notebook
namespace = {'__name__': '__audit__'}
for index, cell in enumerate(notebook['cells']):
    if cell['cell_type'] != 'code':
        continue
    start = time.perf_counter()
    source = ''.join(cell['source'])
    ast.parse(source)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source, f'notebook_cell_{index}', 'exec'), namespace)
        if index == 3:
            namespace['display'] = lambda *args, **kwargs: None
            namespace['show_plotly'] = lambda *args, **kwargs: None
        namespace['plt'].close('all')
        record = {'index_zero_based': index, 'status': 'passed', 'seconds': time.perf_counter()-start}
    except Exception:
        record = {'index_zero_based': index, 'status': 'failed', 'traceback': traceback.format_exc()}
    result['cells'].append(record)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    if record['status'] == 'failed':
        break

if all(row['status'] == 'passed' for row in result['cells']):
    z = namespace['z_two_layer']
    checks = {}
    checks['homogeneous_relative_error'] = max(abs(z(L, 5, 5, 20)/(8*5/(3*np.pi*(L/1000)))-1)
                                              for L in [50, 90, 140])
    checks['image_series_100_vs_1000_max_abs_ohm'] = max(
        abs(namespace['image_kernel'](d,h,r1,r2,100)-namespace['image_kernel'](d,h,r1,r2,1000))*r1/np.pi
        for d in [12.5,35,105] for h in [3,15,60] for r1,r2 in [(2,40),(5,25),(10,8)])
    tri = np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]])
    checks['triangle_distance_errors'] = [
        abs(namespace['nearest_triangle_surface_distance'](p,tri)-expected)
        for p,expected in [([.2,.2,2.],2.),([1.,1.,0.],np.sqrt(.5)),([-1.,0.,0.],1.)]]
    result['math_checks'] = checks
    for key in ['h_centre_surface_mm','h_centre_vertex_mm','h_fit_rmse','delta_rho2_only']:
        result[key] = float(namespace[key])
    for key in ['delta_rho12','singular_values','std','coefficients']:
        result[key] = np.asarray(namespace[key]).tolist()
    result['h_polynomial_range_mm'] = [float(np.min(namespace['h_approx'])),float(np.max(namespace['h_approx']))]
    for key in ['direct_error_table','calibrated_table','objective_parts']:
        result[key] = namespace[key].to_dict(orient='records')
    # Geometric skin-to-lung distance includes rib/bone portions if encountered.
    bone_intersections = []
    for s,skin,lung in zip(namespace['s_profile'],namespace['skin_profile'],namespace['lung_profile']):
        if not (np.isfinite(skin) and np.isfinite(lung)):
            continue
        hits = namespace['unique_sorted_depths'](namespace['contour_intersections_at_s'](
            namespace['section_segments']['bones'],s))
        inside = hits[(hits>skin)&(hits<lung)]
        if len(inside):
            bone_intersections.append({'s_mm':float(s),'h_skin_lung_mm':float(lung-skin),
                                       'bone_crossings_in_wall':inside.tolist()})
    result['bone_intersections_between_skin_and_lung'] = bone_intersections
    # Check frozen input table against the current expert labels, without exporting raw signals.
    levels = pd.read_csv(ROOT/'data/nik/experimental/nik_breath_hold_levels.csv')
    mismatch=[]
    for _,row in levels.iterrows():
        p=ROOT.parent/'Colab Notebooks/timestamps'/f'{int(row.L_mm)}nik.json'
        current=json.loads(p.read_text(encoding='utf-8'))['hold_levels']
        for phase,column in [('вдох','Z_inhale_hold_ohm'),('выдох','Z_exhale_hold_ohm')]:
            if not np.isclose(float(current[phase]),row[column],rtol=0,atol=1e-9):
                mismatch.append({'L_mm':int(row.L_mm),'phase':phase,'delta_ohm':float(current[phase])-row[column]})
    result['manifest_timestamp_mismatches'] = mismatch

(HERE/'report_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
print('CHECKS',json.dumps({k:v for k,v in result.items() if k!='cells'},ensure_ascii=False,default=str),flush=True)
