from pathlib import Path
import csv, hashlib, itertools, json
root=Path(r"D:\Аспа\Kalmykov_PhD\MATLAB_TRKG4_real_subjects")
baseline=root/'output/exploratory/tepc_preparation_20260911'
run=root/'output/exploratory/tepc_volume_convergence_report_smoke_20260914_01'
assert not run.exists()
analysis=run/'analysis'; analysis.mkdir(parents=True)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def writej(p,x): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
montages=[f'tepc_{i}' for i in range(2,8)]; params=['soft','heart','lung']
selected=[]
for i in range(40):
    selected.append({'state':f'rho_grid_n5_{i:03d}','role':'grid5','rho':{'soft':2+i/40,'heart':3+i/30,'lung':8+i/20},'reasons':['smoke']})
selection={'schema':'tepc_volume_convergence_state_selection_v1','status':'selected_before_new_mesh_results','montages':montages,'states_per_montage':40,'selected':selected,'physical_validation':False}
writej(run/'state_selection.json',selection)
levels={'L12':str(baseline.resolve())}; builds={}
for level_id,trunk,nodes,tets in [('L09',9,900000,5000000),('L06',6,1200000,7000000)]:
    level=run/level_id.lower(); meshdir=level/'mesh'; meshdir.mkdir(parents=True)
    mesh=meshdir/f'body_trunk{trunk}mm.msh'; mesh.write_bytes((level_id+' mesh').encode())
    report={'status':'candidate_mesh_requires_external_quality_and_matlab_qc','source_sha256':'a'*64,'target':str(mesh.resolve()),'arm_size_mm':1.0,'electrode_size_mm':1.0,'trunk_size_mm':trunk,'transition_mm':5.0,'algorithm_3d':1,'nodes':nodes-20000,'volume_elements':tets-50000}
    writej(meshdir/'gmsh_build.json',report)
    refinement={'counts':{'nodes':nodes,'tetrahedra':tets,'boundary':200000,'added_nodes':20000,'added_tetrahedra':50000},'min_tetra_volume_mm3':1e-6}
    writej(level/'refinement/refinement.json',refinement)
    levels[level_id]=str(level.resolve())
    builds[level_id]={'report':'mesh/gmsh_build.json','report_sha256':sha(meshdir/'gmsh_build.json'),'mesh':f'mesh/{mesh.name}','mesh_sha256':sha(mesh),'source_surface_sha256':'a'*64,'arm_size_mm':1.0,'electrode_size_mm':1.0,'trunk_size_mm':float(trunk),'transition_mm':5.0,'algorithm_3d':1,'nodes':nodes-20000,'volume_elements':tets-50000}
def writecsv(name,fields,rows):
    with (analysis/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
metric=[]
for transition in ['L12_to_L09','L09_to_L06']:
  for m in montages:
    for state in selected:
      for met,comp in [('Z','all')]+[(x,c) for x in ['S','B','Q'] for c in params]:
        tol=.01 if met=='Q' else .02
        ratio=.3 + .05*(int(m[-1])-2)
        metric.append({'transition':transition,'coarse_level':'L12','fine_level':'L09','montage':m,'state':state['state'],'metric':met,'compartment':comp,'coarse_value':1,'fine_value':1.001,'absolute_change':tol*ratio,'relative_change':.001,'global_scale':1,'tolerance':tol,'criterion':'smoke','passed':True})
writecsv('metric_convergence.csv',list(metric[0]),metric)
top=[]
for state in selected:
  for met in ['S','B','Q']:
    for c in params:
      winner={'soft':'tepc_2','heart':'tepc_4','lung':'tepc_3'}[c]
      top.append({'state':state['state'],'metric':met,'compartment':c,'top_L12':winner,'top_L09':winner,'top_L06':winner,'top_stable':True,'fine_top':winner,'fine_second':'tepc_5','fine_gap':.1,'fine_mesh_uncertainty_sum':.01,'fine_top_gap_resolved':True})
writecsv('top_stability.csv',list(top[0]),top)
pairs=[]
for state in selected:
  for met in ['S','B','Q']:
    for c in params:
      for a,b in itertools.combinations(montages,2):
        pairs.append({'state':state['state'],'metric':met,'compartment':c,'assembly_a':a,'assembly_b':b,'difference_L12':.1,'difference_L09':.1,'difference_L06':.1,'preference_stable':True,'fine_gap':.1,'fine_mesh_uncertainty_sum':.01,'fine_pair_resolved':True,'acceptable_stable_or_unresolved':True})
writecsv('pair_stability.csv',list(pairs[0]),pairs)
vol=[]
for transition in ['L12_to_L09','L09_to_L06']:
  for region,tol in [('body_total',.001),('soft_trunk',.02),('soft_arms',.02),('heart',.02),('lungs',.02),('bones',.02)]:
    vol.append({'transition':transition,'region':region,'coarse_volume_mm3':100,'fine_volume_mm3':100.01,'relative_change':tol*.2,'tolerance':tol,'passed':True})
writecsv('volume_convergence.csv',list(vol[0]),vol)
writecsv('level_results.csv',['dummy'],[{'dummy':1}]); writecsv('rankings.csv',['dummy'],[{'dummy':1}])
source_files={'gmsh_mesh_builder':root/'tools/gmsh_arm_parameter_mesh.py','mesh_classification':root/'src/trkg4_classify_volume_convergence_mesh.m','contact_refinement':root/'tools/refine_surface_contact_mesh.py','model_export':root/'src/trkg4_export_volume_convergence_level.m'}
outputs={name:sha(analysis/name) for name in ['level_results.csv','metric_convergence.csv','rankings.csv','top_stability.csv','pair_stability.csv','volume_convergence.csv']}
checks={'all_final_metric_changes_within_tolerance':True,'all_selected_top_montages_stable_across_three_levels':True,'all_fine_top_gaps_exceed_mesh_uncertainty':True,'all_resolved_pair_preferences_stable':True,'all_final_volume_changes_within_tolerance':True}
summary={'schema':'tepc_independent_volume_mesh_convergence_v1','status':'passed_selected_state_independent_volume_mesh_convergence','scope':'smoke','levels':levels,'independent_mesh_builds':builds,'selection_sha256':sha(run/'state_selection.json'),'states_per_montage':40,'rows_compared':len(metric),'criteria':{},'checks':checks,'failed_metric_rows':0,'unstable_top_rows':0,'unresolved_top_rows':0,'resolved_pair_reversals':0,'failed_volume_rows':0,'worst_metric_cases':{},'interpretation':'smoke','physical_validation':False,'source_sha256':sha(root/'tools/tepc_volume_convergence.py'),'producer_source_sha256':{k:sha(v) for k,v in source_files.items()},'outputs_sha256':outputs,'input_sha256':{'selection':sha(run/'state_selection.json'),'baseline_analysis':'x','level09_verification':'x','level06_verification':'x'}}
writej(analysis/'summary.json',summary); (analysis/'summary.sha256').write_text(sha(analysis/'summary.json')+'\n')
print(run)