from pathlib import Path
from collections import Counter
import csv, difflib, hashlib, io, json, re, zipfile

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent/'revision_20261029_author_review'
REPO=ROOT.parents[3]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]; diffs=[]
for p in sorted((ROOT/'drafts').glob('*.md')):
 old=(BASE/'drafts'/p.name).read_text(encoding='utf-8-sig');new=p.read_text(encoding='utf-8')
 pattern=r'(?<!\w)[−+±]?\d+(?:[,.]\d+)?'
 a=Counter(re.findall(pattern,old));b=Counter(re.findall(pattern,new))
 checks.append(dict(file=p.name,removed_numbers=dict(a-b),added_numbers=dict(b-a),
  display_equations_unchanged=re.findall(r'\$\$.*?\$\$',old,re.S)==re.findall(r'\$\$.*?\$\$',new,re.S),
  citations_preserved=not(Counter(re.findall(r'\[\[C:[^\]]+\]\]',old))-Counter(re.findall(r'\[\[C:[^\]]+\]\]',new)))))
 diffs.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='approved/'+p.name,tofile='local/'+p.name))
assert all(not x['removed_numbers'] and x['display_equations_unchanged'] and x['citations_preserved'] for x in checks),checks
source=json.loads((ROOT/'audit/source_manifest.json').read_text(encoding='utf-8'))
assert sha(Path(source['path']))==source['sha256'],'Approved Word changed during work; do not overwrite or silently incorporate.'
(ROOT/'audit/content_diff.patch').write_text(''.join(diffs),encoding='utf-8')
(ROOT/'audit/numeric_preservation.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')

# Bounded supplement: no raw measurements, DICOM, full photographs or external files.
payload={}
def add(name,path):
 assert name not in payload
 payload[name]=Path(path).read_bytes()
add('README.md',ROOT/'Изменения_и_замечания.md')
for name in ['rho_evidence.md','rho_addition.json','numeric_preservation.json','review_resolutions.json','local_language_review.md','figure_edits.md','final_checks.json','visual_qa.json']:
 add('audit/'+name,ROOT/'audit'/name)
add('figures/figure_edits_mapping.json',ROOT/'figures/figure_edits_mapping.json')
mapping=json.loads((ROOT/'figures/figure_edits_mapping.json').read_text(encoding='utf-8'))
for f in mapping['asset_mapping'].values():
 p=REPO/f['new_path'];add('figures/'+p.name,p)
for name in ['rebuild_ecg_display.cjs','rebuild_material_labels.py','rebuild_materials_anatomy.cjs','ecg_display_check.json','material_label_checks.json']:
 p=ROOT/'tools'/name
 if p.exists():add('tools/'+name,p)
add('data/tissue_literature_v2.json',REPO/'MATLAB_TRKG4_real_subjects/configs/tissue_literature_v2.json')
add('tools/build_tissue_literature_registry.py',REPO/'MATLAB_TRKG4_real_subjects/tools/build_tissue_literature_registry.py')
text=(ROOT/'drafts/03_models.md').read_text(encoding='utf-8')
block=text.split('[[TABLE:tissue_electrical_properties|',1)[1].split(']]',1)[1].strip().split('\n\n',1)[0]
rows=[[c.strip() for c in line.strip().strip('|').split('|')] for line in block.splitlines()]
rows=[row for row in rows if not all(re.fullmatch(r'[-: ]+',x) for x in row)]
s=io.StringIO();csv.writer(s).writerows(rows);payload['data/tissue_resistivity_table_ohm_m.csv']=s.getvalue().encode('utf-8-sig')
manifest={'scope':'Supplement to earlier companion archive, not the complete medical research dataset',
 'word':json.loads((ROOT/'audit/final_checks.json').read_text(encoding='utf-8')),
 'files':[{'path':n,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in sorted(payload.items())]}
payload['manifest.json']=json.dumps(manifest,ensure_ascii=False,indent=2).encode('utf-8')
out=ROOT/'deliverables/Дополнение_к_материалам_20261029_точечная_правка.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for name,data in sorted(payload.items()):z.writestr(name,data)
with zipfile.ZipFile(out) as z:assert z.testzip() is None
print(json.dumps({'numeric_checks':'passed','original_word_unchanged':True,'supplement':str(out),'files':len(payload),'archive_sha256':sha(out)},ensure_ascii=False,indent=2))
