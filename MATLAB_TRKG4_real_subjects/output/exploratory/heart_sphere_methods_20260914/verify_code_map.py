from pathlib import Path
import json,hashlib,datetime,ast,re
import nbformat
from bs4 import BeautifulSoup
root=Path.cwd()
p=root/'MATLAB_TRKG4_real_subjects'
out=p/'output/exploratory/heart_sphere_methods_20260914'
nbp=root/'Colab Notebooks/40.15_Эквивалентная_сфера_сердца_методики_и_проверка.ipynb'
nb=nbformat.read(nbp,as_version=4)
nbformat.validate(nb)
hp=nbp.with_suffix('.html')
soup=BeautifulSoup(hp.read_text(encoding='utf-8'),'html.parser')
rendered=soup.select('.jp-RenderedMarkdown')
assert len(nb.cells)==len(rendered)==23
count=0
for cell,node in zip(nb.cells,rendered):
    visible=re.sub(r'\s+','',node.get_text())
    for block,inline in re.findall(r'\$\$(.*?)\$\$|\$([^$\n]+)\$',cell.source,re.S):
        assert re.sub(r'\s+','',block or inline) in visible,block or inline
        count+=1
assert not soup.select('.jp-InputArea,.jp-InputPrompt,.jp-OutputPrompt,.input_area')
assert len(soup.select('img[src^="data:image/png"]'))==6
assert not any(b'\r\n' in x.read_bytes() for x in (nbp,hp))
checks=json.loads((out/'before_code_map_artifact_checks.json').read_text(encoding='utf-8'))
for fig in checks['figure_visual_review']['figures']:
    assert hashlib.sha256((root/fig['path']).read_bytes()).hexdigest()==fig['sha256']
# Each regenerated embedded PNG is byte-identical to the already visually reviewed file.
import base64
expected={f['sha256'] for f in checks['figure_visual_review']['figures']}
actual=set()
for cell in nb.cells:
    for a in cell.get('attachments',{}).values():
        actual.add(hashlib.sha256(base64.b64decode(a['image/png'])).hexdigest())
assert expected==actual
for item in checks['artifacts']:
    data=(root/item['path']).read_bytes()
    item.update(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
checks['checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
checks['technical_checks']['math_expressions_preserved_in_html']=count
checks['technical_checks']['LF_serialization_and_reader_equivalence']='passed'
checks['technical_checks']['figure_bytes_match_previous_visual_review']='passed'
checks['scientific_status']['historical_comparison']='source_pages_and_expanded_code_map_checked_static_only'
checks['code_audit_record']='MATLAB_TRKG4_real_subjects/docs/HEART_SPHERE_CODE_AUDIT_20260914.json'
(p/'docs/HEART_SPHERE_METHODS.artifact-checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
files=[
'Tikhomirov/cardio-model-py/src/cardio_model/models.py',
'Tikhomirov/cardio-model-py/src/cardio_model/sphere_fit.py',
'Tikhomirov/cardio-model-py/src/cardio_model/finders.py',
'Tikhomirov/cardio-model-py/src/cardio_model/radial.py',
'Tikhomirov/cardio-model-py/src/cardio_model/pipeline.py',
'Tikhomirov/cardio-model-py/src/cardio_model/geometry.py',
'Tikhomirov/cardio-model-py/src/cardio_model/volume.py',
'Tikhomirov/cardio-stl-repair/src/cardio_stl_repair/repair.py',
'Tikhomirov/cardio-stl-repair/tests/test_import.py',
'Tikhomirov/mathcad/sphere_mathcad_sketch.py',
'Tikhomirov/mathcad/sphere_mathcad_model.md',
'Tikhomirov/WolframMath-master/Kernel/part1/SphereMovingFunction.m',
'Tikhomirov/WolframMath-master/Kernel/part1/SphereMovingOptimization.m',
'Tikhomirov/WolframMath-master/Kernel/part1/EqualSphereMoveModellingLib.m',
'Tikhomirov/WolframMath-master/Kernel/Project/RFFI.m',
'Tikhomirov/WolframMath-master/implem/sphereMove/SphereMoveModelling.nb']
record={'schema':'heart_sphere_code_audit_v1','date':'2026-09-14','scope':'Static inspection of selected historical source modules and stored notebook cells; not a full reproduction of the dissertation calculations.',
'reviewers':[{'role':'orchestrator','scope':'selected source statements, Python R+h parameterization, STL implementation status and Mathcad caveat'},{'role':'executor','model':'gpt-5.6-luna','reasoning_effort':'max','scope':'Python port and existing assertions; no execution'}],
'code_changed':False,'historical_tests_executed':False,'historical_wolfram_executed':False,'new_geometric_FEM_executed':False,
'findings':[
{'id':'depth_parameterization','path':files[0],'symbol':'sphere_influence','code':'Rh = R + h','finding':'Computed normal depth is R+h, whereas the docstring calls h center depth. Radius variation at fixed h changes depth too; fixed-center radius variation requires h=c_perp-R.'},
{'id':'contour_fit','path':files[1],'symbol':'eq_sph_nm_center_radius','finding':'Geometric fitting of cx,cy,r to a 2D contour; not joint electrical inversion.'},
{'id':'radial_inverse','paths':files[2:5],'finding':'Separate scalar searches per channel and externally supplied center movement; no joint cx,cy,cz,r electrical inversion found in inspected Python src.'},
{'id':'original_shape_comparison','finding':'No complete Z(t) original arbitrary 3D shape vs sphere comparison found in inspected Python source modules. This does not exclude another historical implementation outside the inspected scope.'},
{'id':'stl_stub','path':files[7],'finding':'Docstring only; import tests do not establish mesh processing functionality.'},
{'id':'mathcad_formula','path':files[9],'symbol':'dro_fl','finding':'Preserves subtraction 0.05**2 - 0.025**3; provenance and dimensional consistency require investigation.'}],
'code_inventory':[{'path':name,'sha256':hashlib.sha256((root/name).read_bytes()).hexdigest()} for name in files],
'notebook_section':'40.15, sections 16.6-16.7',
'status':'implementation_map_not_physical_validation'}
(p/'docs/HEART_SPHERE_CODE_AUDIT_20260914.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({'cells':23,'figures_unchanged':6,'math_expressions_checked':count,'new_code_map_files':len(files),'code_executed':False},ensure_ascii=False))
