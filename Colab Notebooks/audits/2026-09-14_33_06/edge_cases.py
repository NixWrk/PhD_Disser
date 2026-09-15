"""Numerical probes; original notebook and input artifacts are read-only."""
import ast
import json
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from exploratory_analysis import dynamic_two_layer_inverse
from two_layer_model import evaluate, geometry_from_size
from ttrkg_analysis import ensemble_waveform
nb = json.loads(next(ROOT.glob('33.06*.ipynb')).read_text(encoding='utf-8'))
cell14 = ast.parse(''.join(nb['cells'][14]['source']))
function = next(node for node in cell14.body if isinstance(node, ast.FunctionDef) and node.name == 'fit_one_rho2')
exec(compile(ast.Module(body=[function], type_ignores=[]), '<33.06 cell 14>', 'exec'))
result = {}
sizes = np.array([.05, .09, .14])
for r1 in [.01, .1, .5]:
    try:
        evaluate(r1, 999., .001, *geometry_from_size(.14))
        direct_status = 'converged'
    except RuntimeError as exc:
        direct_status = str(exc)
    fit = fit_one_rho2(sizes, np.array([100., 80., 60.]), r1, .001, previous=999.)
    result[f'penalty_rho1_{r1}'] = {'forward_status': direct_status, 'fit': fit}
for h in [1e-8, 1e-5, .001]:
    fixed = {'h_m': h, 'rho1_ohm_m': 5., 'rho2_inhale_ohm_m': 5.}
    wave = np.outer(np.array([1., .7, .5]), np.sin(np.linspace(0, np.pi, 20))) * .001
    inverse = dynamic_two_layer_inverse(sizes, wave, fixed, 'inhale')
    result[f'dynamic_h_{h}'] = {'rank': inverse['operator_rank'], 'finite_outputs': bool(np.isfinite(inverse['delta_rho1_ohm_m']).all())}
t = np.arange(0, 8, .005)
ensemble = ensemble_waveform(t, np.sin(t), [3., 3., 3.], [1., 7.], np.arange(-.15, .705, .005), [-.12, -.02])
result['duplicate_rpeaks'] = {'accepted_n_beats': ensemble['n_beats'], 'maximum_reported_se': float(np.max(ensemble['se_within_record']))}
(HERE / 'edge_cases.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
