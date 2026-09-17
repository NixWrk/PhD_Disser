from pathlib import Path
import csv, hashlib, json, shutil
root=Path(r"D:\Аспа\Kalmykov_PhD\MATLAB_TRKG4_real_subjects")
baseline=root/'output/exploratory/tepc_preparation_20260911'
selection_path=root/'output/exploratory/tepc_volume_convergence_20260914/state_selection.json'
smoke=root/'output/exploratory/tepc_volume_analysis_smoke_20260914_01'
assert not smoke.exists(); smoke.mkdir(parents=True)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def writej(p,x): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
selection=json.loads(selection_path.read_text(encoding='utf-8-sig')); selected={x['state']:x for x in selection['selected']}
shutil.copyfile(selection_path,smoke/'state_selection.json')
with (baseline/'analysis/states_rho.csv').open(encoding='utf-8-sig',newline='') as f: source=list(csv.DictReader(f)); fields=list(source[0])
source=[r for r in source if r['state'] in selected]
assert len(source)==240
profile=json.loads((baseline/'rho_profile.json').read_text(encoding='utf-8-sig')); bone_sigma=1/profile['bone_fixed_rho_ohm_m']
basevol=json.loads((baseline/'refinement/refinement.json').read_text())['group_volume_after_mm3'][1:]
surface_sha=json.loads((baseline/'manifest.json').read_text(encoding='utf-8-sig'))['source']['surface_sha256']
for level_id,trunk,factor,volfactor in [('L09',9,1.001,1.0004),('L06',6,1.0012,1.0005)]:
  level=smoke/level_id.lower(); (level/'results').mkdir(parents=True)
  shutil.copyfile(baseline/'rho_profile.json',level/'rho_profile.json')
  mesh=level/'mesh'/f'body_trunk{trunk}mm.msh'; mesh.parent.mkdir(parents=True); mesh.write_text(level_id)
  report={'status':'candidate_mesh_requires_external_quality_and_matlab_qc','source_sha256':surface_sha,'target':str(mesh.resolve()),'arm_size_mm':1,'electrode_size_mm':1,'trunk_size_mm':trunk,'transition_mm':5,'algorithm_3d':1,'nodes':1000,'volume_elements':5000}
  writej(level/'mesh/gmsh_build.json',report)
  writej(level/'classification/classification.json',{'level_id':level_id,'mesh_sha256':sha(mesh),'group_volumes_mm3':[v*volfactor for v in basevol]})
  jobs=[]
  for montage in [f'tepc_{i}' for i in range(2,8)]:
    states=[]
    for item in selection['selected']:
      rho=item['rho']; states.append({'id':item['state'],'conductivity':{'soft':1/rho['soft'],'heart':1/rho['heart'],'lung':1/rho['lung'],'bone':bone_sigma}})
    study={'states':states}; study_path=level/'studies'/f'{montage}.json'; writej(study_path,study)
    output=level/'results'/montage; output.mkdir()
    subset=[dict(r) for r in source if r['montage']==montage]
    for row in subset:
      for name in ['Z_ohm','reciprocal_Z_ohm','dZ_dsigma_soft','dZ_dsigma_heart','dZ_dsigma_lung']:
        row[name]=str(float(row[name])*factor)
      row['max_relative_residual']='1e-12'
    with (output/'channel_results.csv').open('w',encoding='utf-8',newline='') as f:
      w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(subset)
    jobs.append({'montage':montage,'study':f'studies/{montage}.json','output':f'results/{montage}','states':40})
  plan={'level_id':level_id,'total_states':240,'jobs':jobs}; writej(level/'validation_plan.json',plan)
  writej(level/'verification.json',{'status':'completed_numerical_checks_passed','plan_sha256':sha(level/'validation_plan.json')})
print(smoke)