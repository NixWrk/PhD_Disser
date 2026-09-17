from pathlib import Path
import json,hashlib,re,ast
import numpy as np,pandas as pd,nbformat
from scipy.io import loadmat
from bs4 import BeautifulSoup
ROOT=Path.cwd();MODEL=ROOT/'MATLAB_TRKG4_real_subjects';OUT=MODEL/'output/exploratory/transverse_consistency_20260914';OLD=MODEL/'output/exploratory/transverse_geometry_20260911';CANON=ROOT/'Colab Notebooks'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
record={'status':'recalculated_and_consistent','clinical_or_mesh_convergence_validation':False,'candidate':'C01','new_halfspace_models_computed':False,'numerical_comparisons':{}}
a=json.loads((OLD/'input_contract.json').read_text());b=json.loads((OUT/'input_contract.json').read_text());assert a==b;record['input_contract_matches_previous']=True
for file in ['central_profile.csv','geometry_holdout.csv','geometry_volumes.csv','curvature_fit.csv']:
 assert pd.read_csv(OLD/file).equals(pd.read_csv(OUT/file)),file
record['geometry_profiles_and_metrics_exactly_reproduced']=True
x=loadmat(OLD/'geometry_masks.mat',simplify_cells=True);y=loadmat(OUT/'geometry_masks.mat',simplify_cells=True);assert np.array_equal(x['masks'],y['masks']);record['tissue_masks_exactly_reproduced']=True
files=['pem_library.csv','cem_baseline.csv','inverse_recovery.csv','inverse_predictions.csv','cem_inverse.csv','cem_inverse_predictions.csv','linked_scenarios.csv','perturbations.csv','pair_diagnostics.csv','depth_sensitivity_summary.csv','depth_sensitivity_curves.csv']
for file in files:
 x=pd.read_csv(OLD/file);y=pd.read_csv(OUT/file);assert list(x.columns)==list(y.columns) and x.shape==y.shape,file
 cols=[c for c in x.select_dtypes(include='number') if not any(k in c.lower() for k in ['second','elapsed'])]
 for c in cols:np.testing.assert_array_equal(x[c],y[c],err_msg=file+':'+c)
 record['numerical_comparisons'][file]={'rows':len(y),'numeric_columns_equal':cols,'max_absolute_difference_excluding_time':0.0}
x=loadmat(OLD/'depth_sensitivity.mat',simplify_cells=True);y=loadmat(OUT/'depth_sensitivity.mat',simplify_cells=True);np.testing.assert_array_equal(x['sensitivity'],y['sensitivity']);record['elementwise_sensitivities_exactly_reproduced']=True
for file in ['execution_solve.json','execution_cem.json','cem_inverse_summary.json','depth_execution.json']:
 assert (OUT/file).exists(),file
record['pem_max_reciprocity_error']=float(pd.read_csv(OUT/'pem_library.csv').reciprocity_abs.max());record['cem_max_reciprocity_error']=float(pd.read_csv(OUT/'cem_baseline.csv').reciprocity_abs.max())
record['notebooks']={}
for prefix in ['20.10','20.16']:
 path=next(CANON.glob(prefix+'*.ipynb'));n=nbformat.read(path,4);nbformat.validate(n)
 errors=[o for c in n.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.output_type=='error'];assert not errors
 soup=BeautifulSoup(path.with_suffix('.html').read_text(encoding='utf8'),'html.parser');assert not soup.select('.jp-CodeCell .jp-InputArea, .code_cell .input_area')
 record['notebooks'][prefix]={'sha256':sha(path),'html_sha256':sha(path.with_suffix('.html')),'schema_valid':True,'error_outputs':0,'code_inputs_in_html':0}
 if prefix=='20.16':
  code=[c for c in n.cells if c.cell_type=='code'];assert len(code)==16 and [c.execution_count for c in code]==list(range(1,17))
  text='\n\n'.join(c.source for c in n.cells if c.cell_type=='markdown');figs=re.findall(r'\*\*Рисунок (\d+)\.',text);assert figs==list(map(str,range(1,8))),figs
  for name in ['01_ct_sections.png','02_depth_profile.png','04_transverse_profiles_3d.png']:assert not any(name in c.source for c in code)
  assert len(soup.find_all('img'))==6
  expected=next(node.value for node in ast.walk(ast.parse(code[0].source)) if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='expected' for t in node.targets));expected=ast.literal_eval(expected)
  for name,digest in expected.items():assert sha(OUT/name)==digest,name
  record['notebooks'][prefix].update(code_cells_reexecuted=16,figure_numbers=figs,removed_original_figures=[1,2,4],reader_sha256=hashlib.sha256(text.encode('utf8')).hexdigest())
 else:
  current=[c for c in n.cells if c.metadata.get('shared_transverse_current')];assert len(current)==7
  record['notebooks'][prefix]['new_section_code_cells_executed']=2
  # Previously existing numerical outputs are retained, not recalculated under a different pose.
  oldnb=nbformat.read(OUT/'documentation_before'/path.name,4);byid={c.id:c for c in n.cells}
  for c in oldnb.cells:
   if c.cell_type=='code':assert c.outputs==byid[c.id].outputs,c.id
  record['notebooks'][prefix]['historical_outputs_preserved']=True
m=json.loads((OUT/'shared_section/manifest.json').read_text(encoding='utf8'));assert sha(MODEL/'tools/shared_transverse_section.py')==m['source_sha256']
for name,digest in m['sources'].items():assert sha(OUT/name)==digest
record['shared_section_same_verified_sources_as_2016']=True
record['source_sha256']={str(p.relative_to(MODEL)):sha(p) for p in [MODEL/'tools/shared_transverse_section.py',MODEL/'notebooks/synchronize_transverse_sections.py',MODEL/'tools/transverse_depth_analysis.py',MODEL/'notebooks/build_transverse_geometry_report.py',MODEL/'notebooks/build_trkg4_inverse_report.py']}
(OUT/'verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in record.items() if k not in ['numerical_comparisons','source_sha256','notebooks']},ensure_ascii=False,indent=2));print('Verified numerical tables:',len(files))
