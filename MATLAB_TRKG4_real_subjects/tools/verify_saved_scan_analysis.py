"""Independent numerical verification of saved common-h fits, without FEM."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/exploratory/scan_localization_20260910'

def main():
    out=OUT;d=np.load(out/'scan_arrays.npz');f=np.load(out/'layered_fits.npz');q=pd.read_csv(out/'candidate_requests.csv')
    path=ROOT/'notebooks/audit_fixes_2026_09_08/compare_fixed_rho_h.py'
    spec=importlib.util.spec_from_file_location('independent_planar',path);ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    checks=ref.math_checks();sizes=d['sizes'];r1,r2=d['rho'];asymptote=8*r1/(3*np.pi*sizes*.001)
    def model(h):return ref.planar_z(sizes,r1,r2,h,256)[0] if np.isfinite(h) else asymptote
    lookup={int(k):i for i,k in enumerate(d['flat'])};rng=np.random.default_rng(20260910)
    rows=np.unique(np.r_[[lookup[int(k)] for k in q.flat_index],rng.choice(len(d['flat']),32,replace=False),np.argmin(f['h_mm']),np.argmax(f['h_mm']),np.argmax(f['rmse_ohm'])])
    grid=np.unique(np.r_[np.linspace(0,100,1001),np.geomspace(100,1e7,301)])
    library=np.array([model(h) for h in grid]);max_gap=0.;prediction_error=0.;fits=0
    for row in rows:
        target=d['Z_fem'][row];prediction_error=max(prediction_error,float(np.max(abs(model(f['h_mm'][row])-f['prediction'][row]))))
        for omitted in [-1,0,4,8]:
            mask=np.ones(9,bool)
            if omitted>=0:mask[omitted]=False
            values=np.mean((library[:,mask]-target[mask])**2,axis=1)
            def loss(h):return np.mean((model(h)[mask]-target[mask])**2)
            ix=np.flatnonzero((values[1:-1]<=values[:-2])&(values[1:-1]<=values[2:]))+1
            trials=[values[0],values[-1],loss(np.inf)]
            trials.extend(minimize_scalar(loss,bounds=(grid[i-1],grid[i+1]),method='bounded',options={'xatol':1e-9}).fun for i in ix)
            h=f['h_mm'][row] if omitted<0 else f['loo_h_mm'][row,omitted]
            gap=np.sqrt(loss(h))-np.sqrt(min(trials));max_gap=max(max_gap,float(gap));fits+=1
            assert gap<=.001+1e-7
    assert prediction_error<1e-10
    error=np.sqrt(np.mean((d['Z_fem']-f['prediction'])**2,axis=1))
    np.testing.assert_allclose(error,f['rmse_ohm'],atol=1e-12,rtol=0)
    loo=np.sqrt(np.mean((d['Z_fem']-f['loo_predictions'])**2,axis=1))
    np.testing.assert_allclose(loo,f['loo_rmse_ohm'],atol=1e-12,rtol=0)
    # Reconstructed electrode centres remain close to the saved finite patches;
    # these are distinct conventions, so record the difference rather than force equality.
    e=pd.read_csv(out/'candidate_electrodes.csv')
    assert len(e)==len(q)*9*4 and not e.duplicated(['candidate_id','L_mm','electrode']).any()
    assert (e.contact_area_mm2>0).all() and (e.contact_faces>0).all()
    delta=np.linalg.norm(e[['x_mm','y_mm','z_mm']].to_numpy()-e[['patch_centroid_x_mm','patch_centroid_y_mm','patch_centroid_z_mm']].to_numpy(),axis=1)
    summary=dict(status='passed',independent_formula_checks=checks,sampled_real_poses=len(rows),independent_global_fits=fits,
        max_RMSE_suboptimality_observed_ohm=max_gap,max_prediction_difference_independent_series_ohm=prediction_error,
        all_saved_RMSE_and_LOO_recomputed=True,curves=len(d['flat']),electrode_coordinate_rows=len(e),
        max_requested_to_finite_patch_centroid_mm=float(delta.max()),
        independent_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        physical_validation=False,new_FEM_solutions=0)
    (out/'numerical_verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
