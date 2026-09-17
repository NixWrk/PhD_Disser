from pathlib import Path
import json,hashlib,ast,datetime
from bs4 import BeautifulSoup
root=Path(__file__).resolve().parents[4]; proj=root/'MATLAB_TRKG4_real_subjects'
builder=proj/'tools/build_heart_sphere_methods_report.py'
s=builder.read_text(encoding='utf-8')
assert 'nbformat.write(notebook, notebook_path)' in s
s=s.replace('nbformat.write(notebook, notebook_path)',
            'notebook_path.write_text(nbformat.writes(notebook), encoding="utf-8", newline="\\n")')
s=s.replace('html_path.write_text(body, encoding="utf-8")',
            'html_path.write_text(body, encoding="utf-8", newline="\\n")')
builder.write_text(s,encoding='utf-8',newline='\n'); ast.parse(s)
nbp=root/'Colab Notebooks/40.15_Эквивалентная_сфера_сердца_методики_и_проверка.ipynb'
hp=nbp.with_suffix('.html')
old={}
for p in [nbp,hp]:
    before=p.read_bytes(); after=before.replace(b'\r\n',b'\n')
    old[p.suffix]=hashlib.sha256(before).hexdigest()
    if p.suffix=='.ipynb': assert json.loads(before)==json.loads(after)
    else:
        assert BeautifulSoup(before,'html.parser').get_text()==BeautifulSoup(after,'html.parser').get_text()
    p.write_bytes(after)
record=proj/'docs/HEART_SPHERE_METHODS.language-review.json'
data=json.loads(record.read_text(encoding='utf-8'))
for key,p in [('nb',nbp),('html',hp)]:
    data['sha256'][key]=hashlib.sha256(p.read_bytes()).hexdigest()
    data['artifacts'][key]['bytes']=p.stat().st_size
data['serialization_rebinding']={
    'operation':'CRLF_to_LF in notebook JSON container and HTML source only',
    'notebook_JSON_value_identical':True,'HTML_visible_text_identical':True,
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
print('LF serialization fixed; notebook JSON and HTML reader-visible text are unchanged')
