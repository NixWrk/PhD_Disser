"""One bounded pass for saved data of report figures 12,14,19; read only.

Search HTML/notebook/JSON once, excluding tool environments and own outputs.
Inspect matching files and the companion archive without executing any content.
PNG pixels, image data and Plotly libraries are never used to infer a curve.
"""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys
import zipfile

REV=Path(__file__).resolve().parents[1]
ROOT=REV.parents[3]
sys.stdout.reconfigure(encoding='utf-8')
sha=lambda b:hashlib.sha256(b).hexdigest()
groups={
 12:['anchor_volume_change_ml','anchor_displacement_error_mm','cycle_0_dynamics_1','Изменение объёма от опоры R–R','Ошибка перемещения от опоры'],
 14:['shape_comparison','negative_probe_mohm','positive_probe_mohm','heart_material_scenarios','Одинаковое ΔV;','stl_electrical_comparison'],
 19:['RR0_guarded_blocks','predicted_volume_mL','guarded_chamber_predictions','Одна опора: 14%','КТ: объёмы камер'],
}
rx=re.compile('|'.join(re.escape(s) for values in groups.values() for s in values))
plotrx=re.compile(r'Plotly\.newPlot\(\s*("[^"\n]+")\s*,\s*')
decoder=json.JSONDecoder()

def json_at(s,pos):
    while s[pos].isspace() or s[pos]==',':pos+=1
    return decoder.raw_decode(s,pos)

def scenes(s):
    for match in plotrx.finditer(s):
        try:
            data,end=json_at(s,match.end());layout,end=json_at(s,end)
            yield dict(data=data,layout=layout,div=json.loads(match[1]))
        except (ValueError,IndexError):continue

def number(x):return isinstance(x,(int,float)) and not isinstance(x,bool)

def walk(x):
    if isinstance(x,dict):
        yield x
        for v in x.values():yield from walk(v)
    elif isinstance(x,list):
        for v in x:yield from walk(v)

def has_coordinates(trace):
    def array(x):return isinstance(x,list) and any(number(v) for v in x) or isinstance(x,dict) and bool(x.get('bdata'))
    return isinstance(trace,dict) and array(trace.get('x')) and array(trace.get('y'))

def inspect(label,blob):
    text=blob.decode('utf-8-sig',errors='replace')
    if not rx.search(text):return None
    hits={n:[m for m in markers if m in text] for n,markers in groups.items()}
    result=dict(path=label,sha256=sha(blob),bytes=len(blob),hits={str(n):v for n,v in hits.items() if v},
                numeric_payloads=[],plotly_candidates=[],raster_embedded=('image/png' in text or 'data:image/' in text))
    # For notebooks/JSON, only real object fields count as saved result data.
    try:structured=json.loads(text)
    except ValueError:structured=None
    roots=[] if structured is None else [structured]
    ss=list(scenes(text))
    if structured is not None:
        for obj in walk(structured):
            p=obj.get('application/vnd.plotly.v1+json')
            if isinstance(p,dict):ss.append(p)
            html=obj.get('text/html')
            if isinstance(html,list):ss.extend(scenes(''.join(html)))
    for scene in ss:
        roots.append(scene)
        data=scene.get('data',[])
        st=json.dumps(scene.get('layout',{}),ensure_ascii=False)
        names=[str(t.get('name','')) for t in data if isinstance(t,dict)]
        related=[]
        if all(x in names for x in ['T0','I1','I2']) and ('Dice' in st or 'опоры' in st):related.append(12)
        if all(x in names for x in ['STL','G1','G2','G3a','G3b','L1','L2']) and ('ΔV' in st or 'сценари' in st):related.append(14)
        if 'КТ: объёмы камер' in names and ('желудочек' in st or 'отложенн' in st):related.append(19)
        if related:result['plotly_candidates'].append(dict(figures=related,div=scene.get('div'),names=names,
            traces_with_saved_xy=sum(has_coordinates(t) for t in data)))
    for root in roots:
        for obj in walk(root):
            if obj.get('method_id') in ['T0','I1','I2','I3','G1','G2','G3a'] and number(obj.get('anchor_volume_change_ml')) and number(obj.get('dice')):
                result['numeric_payloads'].append(dict(figure=12,kind='normalized_geometry_row'));break
            rows=obj.get('shape_comparison')
            if isinstance(rows,list) and any(isinstance(r,dict) and number(r.get('negative_probe_mohm')) and number(r.get('positive_probe_mohm')) for r in rows):
                result['numeric_payloads'].append(dict(figure=14,kind='shape_comparison',rows=len(rows)));break
            if obj.get('evaluation')=='RR0_guarded_blocks' and obj.get('target') in ['LV','RV'] and number(obj.get('predicted_volume_mL')):
                result['numeric_payloads'].append(dict(figure=19,kind='guarded_prediction_row'));break
    result['classification']='potential_saved_data' if result['numeric_payloads'] or any(x['traces_with_saved_xy'] for x in result['plotly_candidates']) else 'text_code_reference_or_raster_only'
    return result

def main():
    command=['rg','--files','--hidden','--no-ignore','-g','*.html','-g','*.htm','-g','*.ipynb','-g','*.json']
    for pattern in ['**/.git/**','**/.venv/**','**/venv/**','**/node_modules/**','**/vendor*/**','**/cache/**','**/.pytest_cache/**','**/audit/**','**/revision_20261029_local_edits/**']:
        command+=['-g','!'+pattern]
    names=subprocess.run(command,cwd=ROOT,check=True,capture_output=True,encoding='utf-8').stdout.splitlines()
    findings=[];total_bytes=0;files=0
    for name in names:
        p=ROOT/name
        blob=p.read_bytes();total_bytes+=len(blob);files+=1
        found=inspect(p.relative_to(ROOT).as_posix(),blob)
        if found:findings.append(found)
    archive=REV.parent/'revision_20261029_author_review/deliverables/Материалы_научной_редакции_20261029.zip'
    archive_blob=archive.read_bytes();entries=[]
    with zipfile.ZipFile(archive) as z:
        for n in z.namelist():
            if Path(n).suffix.lower() not in {'.html','.htm','.ipynb','.json','.csv'}:continue
            blob=z.read(n);entries.append(dict(name=n,bytes=len(blob)))
            found=inspect(archive.relative_to(ROOT).as_posix()+'!'+n,blob)
            if found:findings.append(found)
    potential=[x for x in findings if x['classification']=='potential_saved_data']
    result=dict(schema='targeted_embedded_figure_sources_v1',figures=[12,14,19],passes=1,
        scope='one inventory/content pass over repository HTML/HTM/IPYNB/JSON, excluding environments/caches/audit/own revision; one pass over companion archive HTML/HTM/IPYNB/JSON/CSV entries',
        repo_files_examined=files,repo_bytes_examined=total_bytes,
        archive=dict(path=archive.relative_to(ROOT).as_posix(),sha256=sha(archive_blob),entries_examined=entries),
        candidates=findings,potential_saved_data_files=len(potential),
        conclusion='no_target_saved_numeric_payload_found' if not potential else 'inspect_potential_payloads_before_concluding',
        no_png_digitization=True,no_execution_of_notebooks_or_HTML=True)
    (REV/'tools/embedded_figure_source_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(files=files,bytes=total_bytes,archive_entries=len(entries),matched_files=len(findings),potential=potential),ensure_ascii=False))

if __name__=='__main__':main()
