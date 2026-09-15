"""Diagnostic profile for the within-record inhale-minus-exhale contrast."""
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
result = {}
for sid, subject in artifact['subjects'].items():
    sizes = np.asarray(subject['sizes_mm'])/1000
    observed = np.asarray(subject['observed_z_inhale_ohm']) - np.asarray(subject['observed_z_exhale_ohm'])
    saved = subject['static_h_profile']['profile']
    rows=[]
    for h_mm in list(range(1,31))+[40,60,80]:
        h=h_mm/1000
        prior=min(saved,key=lambda row:abs(row['h_m']-h))
        seeds=[[prior['rho1_ohm_m'],prior['rho2_inhale_ohm_m'],prior['rho2_exhale_ohm_m']],
               [1,20,15],[5,20,15],[15,25,15],[15,5,7]]
        def residual(log_rho):
            prediction=_static_prediction(log_rho,sizes,h)[0].reshape(-1,2)
            return prediction[:,0]-prediction[:,1]-observed
        candidates=[]
        for seed in seeds:
            try:
                candidates.append(least_squares(residual,np.log(seed),bounds=(np.log(.01),np.log(1000)),method='trf'))
            except RuntimeError:
                pass
        fit=min(candidates,key=lambda candidate:np.dot(candidate.fun,candidate.fun))
        rows.append({'h_mm':h_mm,'rms_delta_ohm':float(np.sqrt(np.mean(fit.fun**2))),
                     'rho':np.exp(fit.x).tolist(),'success':bool(fit.success)})
    best=min(rows,key=lambda row:row['rms_delta_ohm'])
    result[sid]={'best_on_grid':best,'profile':rows}
    print(sid,best,flush=True)
(HERE/'paired_difference.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
