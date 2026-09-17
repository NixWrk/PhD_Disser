from pathlib import Path
import nbformat,sys,copy,json
from nbclient import NotebookClient
r=Path(__file__).resolve().parents[4];out=Path(__file__).resolve().parent
sys.path.insert(0,str(r/'MATLAB_TRKG4_real_subjects/notebooks'))
from revise_2010 import apply
n=apply(nbformat.read(next((r/'Colab Notebooks').glob('20.10*.ipynb')),4))
last=next(i for i,c in enumerate(n.cells) if c.metadata.get('revise_2010_geometry_section_display'))
selected=n.cells[:last+1]+[c for c in n.cells[last+1:] if c.metadata.get('revise_2010_pem_role') in ['cell_45','cell_47']]
t=nbformat.v4.new_notebook(cells=copy.deepcopy(selected),metadata=copy.deepcopy(n.metadata))
try:
 NotebookClient(t,timeout=240,kernel_name='python3',resources={'metadata':{'path':str(r/'Colab Notebooks')}}).execute()
 result={'status':'pass','executed_code_cells':sum(c.cell_type=='code' for c in t.cells),'scope':'Initializers and changed geometry/phi/PEM cells only; no FEM'}
except Exception as e:
 result={'status':'failed','error':str(e)[-3000:]}
nbformat.write(t,out/'changed_cells_runtime_check.ipynb')
(out/'changed_cells_runtime_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(result,ensure_ascii=False,indent=2))
if result['status']!='pass':raise SystemExit(1)
