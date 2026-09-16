"""Validate the nonlinear synthetic organ-replacement fits; never infer subject truth."""
from pathlib import Path
import numpy as np
import pandas as pd
from lateral_literature_forward import PIPE,sha,read,write
from lateral_tissue_ablation import OUT


def analyze(out=OUT):
    q=read(out/'synthetic_inverse_qc.json');assert q['complete'] and q['fits']==6 and q['synthetic_only']
    assert q['fits_sha256']==sha(out/'synthetic_inverse.csv')
    assert q['predictions_sha256']==sha(out/'synthetic_inverse_predictions.csv')
    assert q['identity']['worker_sha256']==sha(PIPE/'src/run_lateral_tissue_inverse_probe.m')
    assert q['identity']['forward_execution_sha256']==sha(out/'execution.json')
    assert q['identity']['prepared_sha256']==sha(out/'material_matrices.mat')
    assert q['identity']['base_prepared_sha256']==sha(PIPE/'output/exploratory/lateral_array_design_20260916/prepared_fem.mat')
    a=pd.read_csv(out/'synthetic_inverse.csv');b=pd.read_csv(out/'synthetic_inverse_predictions.csv')
    assert len(a)==6 and len(b)==54 and not a.duplicated(['state','comparison']).any()
    assert not b.duplicated(['state','comparison','L_mm']).any()
    assert np.isfinite(a.select_dtypes('number')).all().all() and np.isfinite(b.select_dtypes('number')).all().all()
    d=pd.read_csv(out/'responses.csv')
    max_agreement=float(a.two_start_log_agreement.max())
    assert max_agreement<1e-3 and (a.exitflag>0).all()
    for r in a.itertuples():
        g=b[(b.state==r.state)&(b.comparison==r.comparison)].sort_values('L_mm')
        t=d[(d.state==r.state)&(d.model==r.target_model)&(d.frequency_hz==50000)&(d.kind=='literature')].sort_values('L_mm')
        assert g.L_mm.tolist()==t.L_mm.tolist()
        np.testing.assert_allclose(g.target_Z,t.Z,rtol=1e-12,atol=1e-11)
        np.testing.assert_allclose(g.predicted_Z-g.target_Z,g.residual_Z,rtol=1e-10,atol=1e-10)
        assert abs(np.sqrt(np.mean(g.residual_Z**2))-r.fit_rmse_ohm)<1e-10
        for j in [1,2]:
            actual=100*(getattr(r,'estimated_rho'+str(j))/getattr(r,'truth_rho'+str(j))-1)
            assert abs(actual-getattr(r,'error_rho'+str(j)+'_pct'))<1e-9
    write(out/'synthetic_inverse_validation.json',dict(complete=True,synthetic_only=True,
        global_optimum_certified=False,max_two_start_log_difference=max_agreement,
        inputs_sha256={n:sha(out/n) for n in ['synthetic_inverse_qc.json','synthetic_inverse.csv','synthetic_inverse_predictions.csv','responses.csv']},
        analysis_source_sha256=sha(__file__)))
    print(a[['state','comparison','truth_rho1','truth_rho2','estimated_rho1','estimated_rho2','error_rho1_pct','error_rho2_pct','fit_rmse_ohm','bound_hit','two_start_log_agreement']].to_string(index=False))
    return a


if __name__=='__main__':analyze()

