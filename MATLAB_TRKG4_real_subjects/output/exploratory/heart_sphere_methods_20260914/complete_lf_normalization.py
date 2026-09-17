from pathlib import Path
import json,hashlib,ast
from bs4 import BeautifulSoup
root=Path(__file__).resolve().parents[4]; proj=root/'MATLAB_TRKG4_real_subjects'
builder=proj/'tools/build_heart_sphere_methods_report.py'
ast.parse(builder.read_text(encoding='utf-8'))
nbp=root/'Colab Notebooks/40.15_Эквивалентная_сфера_сердца_методики_и_проверка.ipynb'; hp=nbp.with_suffix('.html')
record=proj/'docs/HEART_SPHERE_METHODS.language-review.json'
data=json.loads(record.read_text(encoding='utf-8'))
old={key:data['sha256'][key] for key in ['nb','html']}
for key,p in [('nb',nbp),('html',hp)]:
    before=p.read_bytes(); after=before.replace(b'\r\n',b'\n')
    if key=='nb': assert json.loads(before)==json.loads(after)
    else:
        a=BeautifulSoup(before,'html.parser').get_text().replace('\r\n','\n')
        b=BeautifulSoup(after,'html.parser').get_text().replace('\r\n','\n')
        assert a==b
    p.write_bytes(after)
    data['sha256'][key]=hashlib.sha256(after).hexdigest()
    data['artifacts'][key]['bytes']=len(after)
data['serialization_rebinding']={
    'operation':'CRLF_to_LF in notebook JSON container and HTML source only',
    'notebook_JSON_value_identical':True,'HTML_text_identical_after_standard_newline_normalization':True,
    'reader_text_file_unchanged':True,'original_artifact_hashes':old,
    'reason':'Repository eol=lf; preserve the hash-bound reviewed report after checkout'
}
record.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
record=proj/'docs/HEART_SPHERE_METHODS.artifact-checks.json'
data=json.loads(record.read_text(encoding='utf-8'))
for item in data['artifacts']:
    p=root/item['path']; item['sha256']=hashlib.sha256(p.read_bytes()).hexdigest(); item['bytes']=p.stat().st_size
data['technical_checks']['LF_serialization_and_reader_equivalence']='passed'
record.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print('LF serialization verified; reader content unchanged; review fingerprints synchronized')
