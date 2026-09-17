from pathlib import Path
import json,hashlib,re,datetime,base64,ast
import nbformat
from bs4 import BeautifulSoup
root=Path.cwd(); p=root/'MATLAB_TRKG4_real_subjects'
out=p/'output/exploratory/heart_sphere_methods_20260914'
nbp=root/'Colab Notebooks/40.15_Эквивалентная_сфера_сердца_методики_и_проверка.ipynb';hp=nbp.with_suffix('.html')
nb=nbformat.read(nbp,as_version=4);nbformat.validate(nb)
soup=BeautifulSoup(hp.read_text(encoding='utf-8'),'html.parser');nodes=soup.select('.jp-RenderedMarkdown')
assert len(nodes)==len(nb.cells)==23
assert not any(c.cell_type=='code' for c in nb.cells)
assert not soup.select('.jp-InputArea,.jp-InputPrompt,.jp-OutputPrompt,.input_area')
assert len(soup.select('img[src^="data:image/png"]'))==6
count=0
for c,node in zip(nb.cells,nodes):
 visible=re.sub(r'\s+','',node.get_text())
 for block,inline in re.findall(r'\$\$(.*?)\$\$|\$([^$\n]+)\$',c.source,re.S):
  assert re.sub(r'\s+','',block or inline) in visible
  count+=1
text='\n\n'.join(c.source for c in nb.cells)
assert '17.8. Ближайшие расчётные задачи' in text
assert 'Общая для всех людей траектория без индивидуальных входов не предполагается установленной.' in text
assert not re.search(r'[CDZ]:[\\\/]',text)
for dest in re.findall(r'\]\(([^)]+)\)',text):
 if dest.startswith(('attachment:','zotero:','http:','https:','#')):continue
 assert (nbp.parent/dest.split('#')[0]).resolve().exists(),dest
record=json.loads((out/'before_joint_question_artifact_checks.json').read_text(encoding='utf-8'))
expected={f['sha256'] for f in record['figure_visual_review']['figures']}
actual={hashlib.sha256(base64.b64decode(a['image/png'])).hexdigest() for c in nb.cells for a in c.get('attachments',{}).values()}
assert expected==actual
assert all(b'\r\n' not in x.read_bytes() for x in (nbp,hp))
ast.parse((p/'tools/build_heart_sphere_methods_report.py').read_text(encoding='utf-8'))
assert hashlib.sha256((p/'docs/archive/heart_sphere_independent_pool_20260914.md').read_bytes()).hexdigest()=='d57b97a4cb1285b7d5ddedc79ece6d3e38a643d9ac1fa8ad6c860f4a7a26301b'
for item in record['artifacts']:
 data=(root/item['path']).read_bytes();item.update(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
record['checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
record['artifact_purpose']='Joint physiological/electrical reduced-model proposal, alternative representations and movement laws; no new geometric FEM'
record['technical_checks']['math_expressions_preserved_in_html']=count
record['technical_checks']['all_relative_notebook_links']='passed'
record['technical_checks']['figure_bytes_match_previous_visual_review']='passed'
record['technical_checks']['LF_serialization_and_reader_equivalence']='passed'
record['supporting_literature']={'selection':'MATLAB_TRKG4_real_subjects/docs/HEART_REDUCED_MODELS_SOURCES_20260914.json','scope':'Selected primary-source passages and official dataset/code pages; not a systematic review','public_data_downloaded':False,'weights_or_atlases_executed':False}
(p/'docs/HEART_SPHERE_METHODS.artifact-checks.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
spec={'message':'docs: frame joint physiological and electrical heart model selection','paths':['MATLAB_TRKG4_real_subjects/tools/heart_sphere_methods_ru.md',nbp.relative_to(root).as_posix(),hp.relative_to(root).as_posix(),'MATLAB_TRKG4_real_subjects/docs/HEART_SPHERE_METHODS.language-review.json','MATLAB_TRKG4_real_subjects/docs/HEART_SPHERE_METHODS.artifact-checks.json','MATLAB_TRKG4_real_subjects/docs/HEART_REDUCED_MODELS_SOURCES_20260914.json']}
(out/'joint_question_commit.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({'notebook_valid':True,'cells':23,'figures_unchanged':6,'formulae_checked':count,'relative_links':'all resolve','new_FEM':False},ensure_ascii=False))
