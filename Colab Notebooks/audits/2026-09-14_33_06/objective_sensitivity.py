"""Sensitivity of the h profile to reasonable, explicitly diagnostic losses."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from exploratory_analysis import _static_prediction

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True)
args = parser.parse_args()
cfg = json.loads(args.config.read_text(encoding='utf-8'))
artifact = json.loads((Path(cfg['derived_root'])/'exp02/exploratory/33.06_side_arrays_exploratory.json').read_text(encoding='utf-8'))
h_grid = np.r_[np.arange(1, 31), 40, 60, 80] / 1000
summary = {}

for subject_id, subject in artifact['subjects'].items():
    sizes = np.asarray(subject['sizes_mm']) / 1000
    observed = np.column_stack([subject['observed_z_inhale_ohm'], subject['observed_z_exhale_ohm']]).ravel()
    saved_profile = subject['static_h_profile']['profile']
    subject_rows = {}
    for scenario in ('absolute_l2', 'relative_l2', 'absolute_soft_l1_2ohm', 'absolute_soft_l1_5ohm'):
        profile = []
        for h in h_grid:
            prior = min(saved_profile, key=lambda row: abs(row['h_m'] - h))
            seeds = [
                [prior['rho1_ohm_m'], prior['rho2_inhale_ohm_m'], prior['rho2_exhale_ohm_m']],
                [1, 20, 15], [5, 20, 15], [15, 5, 7],
            ]
            scale = observed if scenario == 'relative_l2' else np.ones_like(observed)
            loss = 'soft_l1' if 'soft_l1' in scenario else 'linear'
            f_scale = 2.0 if scenario.endswith('2ohm') else 5.0
            if loss == 'linear':
                f_scale = 1.0
            def residual(log_rho):
                return (_static_prediction(log_rho, sizes, h)[0] - observed) / scale
            candidates = [least_squares(residual, np.log(seed), bounds=(np.log(.01), np.log(1000)),
                                        loss=loss, f_scale=f_scale, method='trf') for seed in seeds]
            fit = min(candidates, key=lambda candidate: float(candidate.cost))
            raw = _static_prediction(fit.x, sizes, h)[0] - observed
            profile.append({'h_mm': float(h*1000), 'objective_cost': float(fit.cost),
                            'raw_rms_ohm': float(np.sqrt(np.mean(raw**2))),
                            'rho': np.exp(fit.x).tolist(), 'success': bool(fit.success)})
        best = min(profile, key=lambda row: row['objective_cost'])
        subject_rows[scenario] = {'best_on_grid': best, 'profile': profile}
        print(subject_id, scenario, best['h_mm'], best['raw_rms_ohm'], flush=True)
    summary[subject_id] = subject_rows

(HERE/'objective_sensitivity.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
