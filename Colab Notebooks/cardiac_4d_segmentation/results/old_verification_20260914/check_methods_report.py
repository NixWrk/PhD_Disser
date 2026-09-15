from pathlib import Path
import json,hashlib,re,base64
import nbformat
from bs4 import BeautifulSoup
root=Path.cwd();docs=root/'MATLAB_TRKG4_real_subjects/docs';p=docs/'HEART_SPHERE_METHODS.artifact-checks.json'
r=json.loads(p.read_text(encoding='utf8'));nbp=root/'Colab Notebooks/40.15_Эквивалентная_сфера_сердца_методики_и_проверка.ipynb'
nb=nbformat.read(nbp,as_version=4);nbformat.validate(nb);soup=BeautifulSoup(nbp.with_suffix('.html').read_text(encoding='utf8'),'html.parser')
assert len(nb.cells)==23 and all(c.cell_type=='markdown' for c in nb.cells)
assert not soup.select('.jp-CodeCell .jp-InputArea,.jp-InputPrompt,.jp-OutputPrompt')
expected={f['sha256'] for f in r['figure_visual_review']['figures']}
actual={hashlib.sha256(base64.b64decode(a['image/png'])).hexdigest() for c in nb.cells for a in c.get('attachments',{}).values()}
assert expected==actual
for c in nb.cells:
 for dest in re.findall(r'\]\(([^)]+)\)',c.source):
  if dest.startswith(('http:','https:','zotero:','attachment:','#')):continue
  assert (nbp.parent/dest.split('#')[0]).resolve().exists(),dest
count=0
for c,node in zip(nb.cells,soup.select('.jp-RenderedMarkdown')):
 text=re.sub(r'\s+','',node.get_text())
 for block,inline in re.findall(r'\$\$(.*?)\$\$|\$([^$\n]+)\$',c.source,re.S):
  assert re.sub(r'\s+','',block or inline) in text;count+=1
for a in r['artifacts']:
 b=(root/a['path']).read_bytes();a.update(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
from datetime import datetime,timezone
r['checked_at']=datetime.now(timezone.utc).isoformat();r['technical_checks']['math_expressions_preserved_in_html']=count
r['historical_ct_preparation']={'scope':'413 OLD Enhanced CT headers; no independent reference masks available per author','full_20_phase_groups':17,'incomplete_18_phase_groups':1,'contrast_administered_tag_yes':413,'anatomical_accuracy_evaluated':False,'image_visual_review':'blocked_by_automatic_approval_review'}
r['methodological_review']['status']='historical_review_retained_not_repeated_for_new_revision'
r['methodological_review']['scope']+='; refers to the previous source_sha256, not the current revised section'
p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
print(json.dumps({'notebook_cells':23,'figures_unchanged':6,'formulae_preserved':count,'links':'passed'}))
