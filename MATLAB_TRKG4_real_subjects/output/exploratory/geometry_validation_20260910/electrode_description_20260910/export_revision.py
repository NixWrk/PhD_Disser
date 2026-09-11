"""Rebuild prose, preserve existing cell results, export HTML without execution."""
from pathlib import Path
import hashlib
import json
import runpy
import nbformat
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup

REV=Path(__file__).resolve().parent
ROOT=REV.parents[3]
PROJECT=ROOT.parent
OUT=REV.parent
NB=next((PROJECT/'Colab Notebooks').glob('20.14*.ipynb'))
old=nbformat.read(REV/'before'/NB.name,as_version=4)
before=json.loads((REV/'before_identity.json').read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,h in before.items():assert sha(ROOT/name)==h,name
runpy.run_path(str(ROOT/'notebooks/build_geometry_validation_report.py'),run_name='__main__')
new=nbformat.read(NB,as_version=4)
a=[c for c in old.cells if c.cell_type=='code']
b=[c for c in new.cells if c.cell_type=='code']
assert len(a)==len(b)==7
for source,target in zip(a,b):
    assert source.source==target.source,'Calculation cell changed during prose revision'
    target.update(source)
assert a==b
new.metadata['kalmykov_corpus']['reader_revision']=str(REV.relative_to(ROOT))
nbformat.validate(new);nbformat.write(new,NB)
e=HTMLExporter(template_name='lab')
e.exclude_input=True;e.exclude_input_prompt=True;e.exclude_output_prompt=True
html,_=e.from_notebook_node(new)
NB.with_suffix('.html').write_text(html,encoding='utf-8')
soup=BeautifulSoup(html,'html.parser');body=soup.select_one('.jp-Notebook')
assert not soup.select('.jp-CodeCell .jp-InputArea')
for tag in body.find_all(['script','style']):tag.decompose()
reader=body.get_text('\n',strip=True)
(REV/'reader_visible_text.txt').write_text(reader,encoding='utf-8')
figures=[o.data['image/png'] for c in b for o in c.outputs if 'data' in o and 'image/png' in o.data]
checks={'technical_status':'passed','computational_status':'preserved_verified_previous_calculations',
        'scientific_status':'exploratory_conditional_comparison_not_in_vivo_accuracy_validation',
        'code_cells_unchanged':len(b),'all_saved_outputs_unchanged':True,'FEM_executed_this_revision':False,
        'notebook_cells_executed_this_revision':0,'figures':len(figures),'tables':len(body.find_all('table')),
        'reader_characters':len(reader),'figure_payload_sha256':[hashlib.sha256(v.encode()).hexdigest() for v in figures],
        'html_has_no_code_inputs':True,'frozen_calculation_files_unchanged':len(before),
        'notebook_sha256':sha(NB),'html_sha256':sha(NB.with_suffix('.html')),
        'reader_sha256':sha(REV/'reader_visible_text.txt')}
assert checks['figures']==5 and checks['tables']==17
(REV/'editorial_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in checks.items() if 'sha256' not in k}))
