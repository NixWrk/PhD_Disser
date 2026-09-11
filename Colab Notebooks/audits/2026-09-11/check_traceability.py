import ast,json,re,sys
from pathlib import Path
from urllib.parse import unquote,urlsplit
ROOT=Path(__file__).resolve().parents[3]
N=ROOT/'Colab Notebooks'; OUT=Path(__file__).resolve().parent
summary=[];bad=[];missing=[];html=[]
active=sorted(N.glob('*.ipynb'))+sorted((N/'cardiac_4d_segmentation').glob('*.ipynb'))
for p in active+sorted(N.glob('*.md'))+sorted((N/'cardiac_4d_segmentation').glob('*.md')):
 if p.suffix=='.ipynb':
  nb=json.loads(p.read_text(encoding='utf-8-sig'));cells=nb['cells']
  text='\n'.join(''.join(c['source']) for c in cells if c['cell_type']=='markdown')
  codes=[(i,c) for i,c in enumerate(cells) if c['cell_type']=='code' and ''.join(c['source']).strip()]
  summary.append(dict(path=p.relative_to(ROOT).as_posix(),cells=len(cells),code_cells=len(codes),outputs=sum(len(c.get('outputs',[])) for c in cells)))
  for i,c in codes:
   s=''.join(c['source'])
   if any(l.lstrip().startswith(('%','!')) for l in s.splitlines()):continue
   try:ast.parse(s)
   except SyntaxError as e:bad.append(dict(path=p.relative_to(ROOT).as_posix(),cell=i,cell_id=c.get('id'),line=e.lineno,error=e.msg))
 else:text=p.read_text(encoding='utf-8-sig')
 text=re.sub(r'```.*?```','',text,flags=re.S)
 for m in re.finditer(r'(?<!!)\[[^\]\n]+\]\(([^)\n]+)\)',text):
  target=m.group(1).strip().strip('<>');url=urlsplit(target)
  if url.scheme or url.netloc or not url.path:continue
  if not (p.parent/unquote(url.path)).exists():missing.append(dict(path=p.relative_to(ROOT).as_posix(),target=target))
from bs4 import BeautifulSoup
for p in sorted(N.glob('*.html')):
 soup=BeautifulSoup(p.read_text(encoding='utf8'),'html.parser')
 inputs=[n for n in soup.select('.jp-CodeCell .jp-InputArea, .code_cell .input_area') if n.get_text(strip=True)]
 if inputs:html.append(dict(path=p.relative_to(ROOT).as_posix(),nonempty_code_inputs=len(inputs)))
sys.path.insert(0,str(N))
import numpy as np
from ttrkg_analysis import matrix_diagnostics
nb=json.loads(next(N.glob('50.01*.ipynb')).read_text(encoding='utf8'))
c=next(c for c in nb['cells'] if c.get('id')=='5001-core')
fn=next(n for n in ast.parse(''.join(c['source'])).body if isinstance(n,ast.FunctionDef) and n.name=='analyze_design_operator')
ns=dict(np=np,matrix_diagnostics=matrix_diagnostics);exec(compile(ast.Module(body=[fn],type_ignores=[]),'audited-function','exec'),ns)
item=dict(parameter_names=['target','nuisance','unseen'],parameter_units=['mL','1','1'],parameter_scales=[1,1,1],jacobian=[[1,0,0],[0,1,0],[0,0,0]],measurement_covariance=np.eye(3).tolist(),measurement_configurations=[dict(row_id=str(i)) for i in range(3)],acceptance=dict(relative_rank_tolerance=1e-10,target_parameter='target',target_nullspace_tolerance=1e-10,max_nonzero_subspace_condition=10,max_conditional_target_std=3))
a=ns['analyze_design_operator'](item)
item.update(jacobian=item['jacobian'][:2],measurement_covariance=np.eye(2).tolist(),measurement_configurations=item['measurement_configurations'][:2])
b=ns['analyze_design_operator'](item)
def compact(r):return {k:('infinity' if isinstance(v,float) and not np.isfinite(v) else v) for k,v in r.items() if k in ['rank','target_estimable','conditional_design_target_std','nonzero_subspace_condition','target_design_pass']}
result=dict(active_notebooks=summary,python_cell_parse_errors=bad,missing_local_link_candidates=missing,html_with_input_elements=html,identifiability_counterexample=dict(with_zero_row=compact(a),without_zero_row=compact(b)))
(OUT/'checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
print(json.dumps(dict(active_notebooks=len(summary),parse_errors=bad,missing_link_candidates=len(missing),html_inputs=html,counterexample=result['identifiability_counterexample']),ensure_ascii=False,indent=2))
