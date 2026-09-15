"""Numerical acceptance of the new C01 FEM outputs before any inverse ranking."""
import json
import numpy as np
import pandas as pd
from scipy.interpolate import CubicHermiteSpline
from c01_baseline_study import ROOT,SOURCE,OUT,sha,write
from c01_baseline_analysis import checked

def main():
    g=checked();e=json.loads((OUT/'execution_pem.json').read_text('utf-8'));identity=e['identity']
    assert e['pem_complete'] and e['reference_holdouts_complete']
    assert identity['geometry_sha256']==sha(OUT/'geometry.json') and identity['masks_sha256']==sha(OUT/'geometry_masks.mat')
    assert identity['source_prepared_sha256']==sha(SOURCE/'prepared_fem.mat')
    assert identity['worker_sha256']==sha(ROOT/'src/run_c01_baseline_models.m')
    for file,key in [('pem_library.csv','pem_library_sha256'),('reference_holdouts.csv','reference_holdouts_sha256')]:assert sha(OUT/file)==e[key]
    raw=pd.read_csv(OUT/'pem_library.csv');ref=pd.read_csv(OUT/'reference_holdouts.csv');old=pd.read_csv(SOURCE/'pem_library.csv')
    assert not raw.duplicated(['model','ratio','L_mm']).any();assert not ref.duplicated(['ratio','L_mm']).any()
    assert np.isfinite(raw.select_dtypes('number').to_numpy()).all() and np.isfinite(ref.select_dtypes('number').to_numpy()).all()
    assert raw.relative_residual.max()<1e-7 and raw.reciprocity_abs.max()<1e-7
    assert ref.residual.max()<1e-7 and ref.reciprocity.max()<1e-7
    sizes=np.array(g['sizes_mm']);checks=[];hom=old[(old.model=='reference')&np.isclose(old.ratio,1)].sort_values('L_mm').f.to_numpy()
    for name,t in raw.groupby('model'):
        def at(q):
            x=t[np.isclose(t.ratio,q,rtol=1e-12,atol=1e-12)].sort_values('L_mm');np.testing.assert_array_equal(x.L_mm,sizes);return x
        homogeneous=float(np.max(abs(at(1).f.to_numpy()-hom)))
        fd=(at(4*np.exp(.001)).f.to_numpy()-at(4*np.exp(-.001)).f.to_numpy())/.002
        deriv=at(4).df_dlogq.to_numpy();derivative_error=float(np.max(abs(fd-deriv))/np.max(abs(deriv)))
        knots=np.array(g['knots']);f=np.stack([at(q).f for q in knots]);df=np.stack([at(q).df_dlogq for q in knots])
        sp=CubicHermiteSpline(np.log(knots),f,df,axis=0,extrapolate=False)
        interp=max(float(np.max(abs(sp(np.log(q))-at(q).f.to_numpy()))) for q in g['checks'])
        checks.append(dict(model=name,homogeneous_max_unit_error=homogeneous,derivative_relative_error=derivative_error,max_interpolation_unit_error=interp))
        assert homogeneous<1e-8 and derivative_error<1e-5 and interp<1e-3,(name,homogeneous,derivative_error,interp)
    recheck=[]
    for q,t in ref.groupby('ratio'):
        previous=old[(old.model=='reference')&np.isclose(old.ratio,q,rtol=1e-12,atol=1e-12)].sort_values('L_mm')
        if len(previous)==9:recheck.append(float(np.max(abs(t.sort_values('L_mm').f.to_numpy()-previous.f.to_numpy()))))
    assert recheck and max(recheck)<1e-8
    pd.DataFrame(checks).to_csv(OUT/'new_fem_qc.csv',index=False)
    write(OUT/'fem_acceptance.json',dict(geometry_sha256=sha(OUT/'geometry.json'),source_sha256=sha(__file__),
        outputs_sha256={f:sha(OUT/f) for f in ['pem_library.csv','reference_holdouts.csv','execution_pem.json','pem_identity.json']},
        max_existing_reference_reproduction_unit_error=max(recheck),checks=checks,status='passed_discrete_numerical_checks_not_physical_validation'))
    print(json.dumps({'qc':checks,'reproduction':max(recheck)},indent=2))
if __name__=='__main__':main()
