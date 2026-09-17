"""Verify completed pilot outputs and nominal regression against frozen TEPC pilot."""
from pathlib import Path
import csv,json,sys,hashlib
out=Path(__file__).resolve().parent
root=out.parents[2];sys.path.insert(0,str(root/'tools'))
import heart_replacement_electrical_pilot as pilot
p=pilot.load_plan(out);runner=pilot.configure(Path(p['runtime']['deps']),p['threads'])
base=Path(p['source_prepared']).parents[2]
rows=[]
for mid in pilot.MONTAGES:
 new=pilot.completed(out,p,'nominal',mid)
 if new is None:continue
 old=base/'pilot'/(mid+'__pilot_00')
 runner.verify_completed_output(old,{'identity.json','source_used.zip','channel_results.csv'},13,0,'exploratory_hypothesis_not_validated')
 identity=pilot.read(old/'identity.json');study=identity['study']
 assert study['source']['prepared_sha256']==p['source_prepared_sha256']
 state=next(s for s in study['states'] if s['id']=='pilot_base')
 assert all(abs(state['conductivity'][k]-1/v)<1e-14 for k,v in p['rho_ohm_m'].items())
 assert study['montages'][0]['contact_sha256']==new['contact_sha256']
 with (old/'channel_results.csv').open(newline='',encoding='utf-8') as f:oldrow=next(r for r in csv.DictReader(f) if r['state']=='pilot_base')
 z=float(oldrow['Z_ohm']);diff=new['Z_ohm']-z
 differences={k:new['dZ_dsigma_'+k]-float(oldrow['dZ_dsigma_'+k]) for k in p['rho_ohm_m']}
 assert abs(diff)<1e-7
 assert all(abs(d)<1e-7*max(abs(float(oldrow['dZ_dsigma_'+k])),1) for k,d in differences.items())
 rows.append({'montage':mid,'new_Z_ohm':new['Z_ohm'],'previous_Z_ohm':z,'difference_ohm':diff,
              'dZ_dsigma_differences':differences,'previous_completion_sha256':pilot.sha(old/'completion.json')})
result={'status':'passed' if len(rows)==6 else 'partial','scope':'Independent reassembly reproduces prior same-mesh nominal solution; internal regression, not physical validation','rows':rows}
pilot.write(out/'nominal_regression.json',result)
records=[]
for name in pilot.GEOMETRIES:
 for mid in pilot.MONTAGES:
  if (out/name/mid/'completion.json').exists():records.append(pilot.completed(out,p,name,mid))
pilot.write(out/'verification.json',{'status':'passed_all_18' if len(records)==18 else 'partial',
 'completed':len(records),'max_relative_residual':max(r['max_relative_residual'] for r in records),
 'max_current_error_A':max(r['max_current_error_A'] for r in records),
 'max_ground_residual_A':max(r['max_ground_residual_A'] for r in records),
 'max_reciprocity_absolute_ohm':max(r['reciprocity_absolute_ohm'] for r in records),
 'nominal_regression_max_abs_ohm':max(abs(r['difference_ohm']) for r in rows),
 'implementation_sha256':pilot.sha(__file__),
 'plan_sha256':pilot.sha(out/'plan.json')})
print(json.dumps(pilot.read(out/'verification.json'),indent=2))
