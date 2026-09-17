"""Execute notebook 20.13 and export a code-free reader HTML with QC."""
from pathlib import Path
import base64
import hashlib
import json
import re
import time
from urllib.parse import unquote, urlsplit

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent
OUT=ROOT/'output/exploratory/geometry_reduction_20260910'
NB=PROJECT/'Colab Notebooks/20.13_Геометрия_лёгкого_и_ошибка_восстановления_сопротивлений.ipynb'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def export(path):
    nb=nbformat.read(path,as_version=4)
    exporter=HTMLExporter(template_name='lab')
    exporter.exclude_input=True;exporter.exclude_input_prompt=True;exporter.exclude_output_prompt=True
    html,_=exporter.from_notebook_node(nb)
    path.with_suffix('.html').write_text(html,encoding='utf-8')
    return BeautifulSoup(html,'html.parser')


start=time.perf_counter()
nb=nbformat.read(NB,as_version=4)
NotebookClient(nb,timeout=180,resources={'metadata':{'path':str(NB.parent)}}).execute()
nbformat.validate(nb);nbformat.write(nb,NB)
soup=export(NB)
assert not soup.select('.jp-CodeCell .jp-InputArea'), 'Code inputs leaked into reader HTML'
body=soup.select_one('.jp-Notebook')
assert body is not None
for element in body.find_all(['script','style']):element.decompose()
reader=body.get_text('\n',strip=True)
(OUT/'reader_visible_text.txt').write_text(reader,encoding='utf-8')
qa=OUT/'visual_qa';qa.mkdir(exist_ok=True)
image_files=[]
for c in nb.cells:
    for output in c.get('outputs',[]):
        if 'image/png' in output.get('data',{}):
            p=qa/f'figure_{len(image_files)+1}.png'
            p.write_bytes(base64.b64decode(output['data']['image/png']));image_files.append(p)
        assert output.output_type!='error'
links=[]
for link in body.find_all('a',href=True):
    url=urlsplit(link['href'])
    if url.scheme or url.netloc or not url.path:continue
    target=(NB.parent/unquote(url.path)).resolve()
    assert target.exists(),str(target)
    links.append(str(target.relative_to(PROJECT)))
neighbor=PROJECT/'Colab Notebooks/20.12_Критерий_Lmax_плоскослоистой_модели.ipynb'
export(neighbor)
result={'code_cells_executed':sum(c.cell_type=='code' for c in nb.cells),
        'markdown_cells':sum(c.cell_type=='markdown' for c in nb.cells),
        'errors':0,'elapsed_seconds':time.perf_counter()-start,'html_code_inputs':0,
        'figure_count':len(image_files),'html_tables':len(body.find_all('table')),
        'local_links_checked':len(links),'reader_characters':len(reader),
        'notebook_sha256':digest(NB),'html_sha256':digest(NB.with_suffix('.html')),
        'reader_sha256':digest(OUT/'reader_visible_text.txt'),
        'figure_sha256':{p.name:digest(p) for p in image_files}}
(OUT/'notebook_execution.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
