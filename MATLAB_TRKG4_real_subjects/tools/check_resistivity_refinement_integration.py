"""Smoke-test all analytical/report branches with labelled manufactured fields.

No synthetic values are written to the real analysis directory.
"""
from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import analyze_resistivity_refinement as a
from resistivity_grid_convergence import audit_grid,compare_nested_scorecards
import resistivity_refinement_report as report
import numpy as np
import pandas as pd

real_base=a.BASE
plan,profile=a.read_plan()
rows=[]
for item in plan['states']:
    s,h,l=[item['rho'][k] for k in a.PARAMS]
    f=.85+.02*plan['montages'].index(item['montage'])
    Z=f*(30*s+s*s+h+.1*h*h+2*l+.1*l*l+.01*s*h+.02*s*l)
    S=f*np.asarray([30+2*s+.01*h+.02*l,1+.2*h+.01*s,2+.2*l+.02*s])
    sig=1/np.asarray([s,h,l])
    row=dict(montage=item['montage'],state=item['state'],role=item['role'],origin='manufactured_test_not_fem',Z_ohm=Z,max_relative_residual=0.,reciprocity_absolute_ohm=0.)
    for j,k in enumerate(a.PARAMS):row['sigma_'+k]=sig[j];row['dZ_dsigma_'+k]=-S[j]/sig[j]**2
    rows.append(row)
frame=a.add_metrics(pd.DataFrame(rows),profile)
axes,grid,points,direct=a.to_arrays(plan,frame)
widths=np.asarray([np.ptp(profile['bounds'][k]) for k in a.PARAMS])
audit=audit_grid(axes,grid,points,direct,widths,plan['criteria'])
assert audit['all_derivative_pass'] and audit['all_pair_S_pass'] and audit['all_pair_Q_pass']
assert np.max(audit['derivative_error'])<1e-10
comparison=compare_nested_scorecards(axes,grid,widths,extra_values=direct,criteria=plan['criteria'])
cards,stability,preferred=a.score_tables(plan,comparison)
assert stability['pass'].all() and all(x['stable'] for x in preferred)
base=real_base/'report/synthetic_smoke'
base.mkdir(exist_ok=True)
for name in ['refinement_plan.json','refinement_plan.sha256','rho_profile.json']:(base/name).write_bytes((real_base/name).read_bytes())
a.BASE=base
summary=a.write_analysis(plan,profile,frame,{'MANUFACTURED':'NOT_REAL_FEM'},axes,grid,points,direct,audit,cards,stability,preferred)
assert summary['adequate_on_tested_states']
report.BASE=base
r=report.load_report()
# Avoid flooding logs with display objects while preserving generated images.
report.display=lambda *args,**kwargs:None
report.status(r);report.design(r);report.assembly_table(r);report.derivative_ranges(r)
report.error_maps(r,'derivative',3);report.error_maps(r,'pairs',4);report.pair_matrix(r,5)
report.minima(r,'min_B',6);report.minima(r,'min_Q',7);report.selection(r);report.error_table(r);report.excursions(r)
assert len(list(r['fig'].glob('figure_*.png')))==8
validation={'type':'manufactured_not_fem','states':len(frame),'derivative_max_error':float(audit['derivative_error'].max()),'figures':8,'all_integration_branches_pass':True,'no_real_analysis_directory_written':not (real_base/'analysis').exists()}
(base/'smoke_validation.json').write_text(json.dumps(validation,indent=2)+'\n',encoding='utf-8')
print(json.dumps(validation,indent=2))
