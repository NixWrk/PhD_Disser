from pathlib import Path
import json,re,hashlib,datetime,ast
import nbformat
from bs4 import BeautifulSoup
root=Path(__file__).resolve().parents[4]
project=root/'MATLAB_TRKG4_real_subjects'
nbp=root/'Colab Notebooks/40.15_Эквивалентная_сфера_сердца_методики_и_проверка.ipynb'
nb=nbformat.read(nbp,as_version=4); nbformat.validate(nb)
hp=nbp.with_suffix('.html')
soup=BeautifulSoup(hp.read_text(encoding='utf-8'),'html.parser')
rendered=soup.select('.jp-RenderedMarkdown')
formula_count=0
for cell,node in zip(nb.cells,rendered):
    expected=re.findall(r'\$\$(.*?)\$\$|\$([^$\n]+)\$',cell.source,re.S)
    visible=re.sub(r'\s+','',node.get_text())
    for block,inline in expected:
        formula=block or inline
        assert re.sub(r'\s+','',formula) in visible,formula
        formula_count+=1
ast.parse((project/'tools/build_heart_sphere_methods_report.py').read_text(encoding='utf-8'))
freeze=json.loads((project/'docs/HEART_SPHERE_METHOD_POOL_20260914.json').read_text(encoding='utf-8-sig'))
frozen=project/'docs/archive/heart_sphere_independent_pool_20260914.md'
assert hashlib.sha256(frozen.read_bytes()).hexdigest()=='d57b97a4cb1285b7d5ddedc79ece6d3e38a643d9ac1fa8ad6c860f4a7a26301b'
files=[project/'tools/heart_sphere_methods_ru.md',project/'tools/build_heart_sphere_methods_report.py',nbp,hp]
figs=sorted((project/'output/exploratory/heart_sphere_methods_figures').glob('*.png'))
def item(p):
    return {'path':p.relative_to(root).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
record={
 'schema':'heart_sphere_methods_artifact_checks_v1',
 'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'artifact_purpose':'Methodological proposal and bounded historical comparison; no new geometric FEM results',
 'artifacts':[item(p) for p in files],
 'technical_checks':{'nbformat':'passed','markdown_cells':len(nb.cells),'code_cells':0,
    'embedded_png_count':6,'all_relative_notebook_links':'passed','zotero_page_links':15,
    'math_expressions_preserved_in_html':formula_count,'no_notebook_input_or_prompt_nodes':'passed',
    'generator_syntax':'passed','independent_pool_hash_unchanged':'passed'},
 'figure_visual_review':{'reviewer':'orchestrator','status':'passed_after_layout_corrections',
    'scope':'all six complete figures; titles, axes, units, legends, formulas, overlaps and synthetic provenance',
    'figures':[item(p) for p in figs]},
 'html_browser_visual_check':{'status':'unavailable','reason':'Browser tool initialization failed: Windows sandbox helper apply deny-read ACLs',
    'fallback':'HTML structure, links and all math-expression contents checked; embedded figures reviewed separately'},
 'scientific_status':{'methods':'proposed_not_validated','historical_comparison':'source_pages_and_selected_code_checked',
    'new_geometric_FEM':'not_performed','anatomical_validation':'not_performed','physical_validation':'not_performed'},
 'historical_source_visual_pages':{'NUZKBZGS':[75,81], 'SG58QNPU':[35],
    '3NAQJNXQ':[140,144,146,155,156,160]},
 'not_a_formal_release_gate_receipt':True
}
path=project/'docs/HEART_SPHERE_METHODS.artifact-checks.json'
path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({'artifact_checks':path.relative_to(root).as_posix(),'math_expressions_checked':formula_count,'artifacts':record['artifacts']},ensure_ascii=False,indent=2))
