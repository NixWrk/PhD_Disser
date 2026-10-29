"""Check prior input hashes, equations, figure retention and contract coverage."""
from pathlib import Path
from collections import Counter
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'revision_20261028'
MAP={'05_experiments.md':'07_experiments.md','06_anatomy.md':'05_anatomy.md','07_forward.md':'06_forward.md'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def formulas(t):
    return Counter(re.sub(r'\s+','',re.sub(r'\\tag\{[^}]+\}','',x)) for x in re.findall(r'\$\$(.*?)\$\$',t,re.S))
snapshot=json.loads((ROOT/'reviews/input_snapshot.json').read_text(encoding='utf-8'))
hash_errors=[x['file'] for x in snapshot['sources'] if sha(ROOT.parent/x['file'])!=x['sha256']]
equations=[]
for p in sorted((OLD/'chapters').glob('*.md')):
    new=ROOT/'chapters'/MAP.get(p.name,p.name)
    a,b=formulas(p.read_text(encoding='utf-8')),formulas(new.read_text(encoding='utf-8'))
    missing=list((a-b).elements())
    equations.append({'old':p.name,'new':new.name,'old_display_formulas':sum(a.values()),'new_display_formulas':sum(b.values()),'missing_or_changed':missing})
req=json.loads((ROOT/'contract_requirements.json').read_text(encoding='utf-8'))['requirements']
matrix=(ROOT/'contract_matrix.md').read_text(encoding='utf-8')
appendix=(ROOT/'appendices.md').read_text(encoding='utf-8')
requirement_errors=[x['id'] for x in req if x['requirement'] not in matrix or x['requirement'] not in appendix]
receipt=json.loads((ROOT/'build_manifest.json').read_text(encoding='utf-8'))
before=json.loads((ROOT/'reviews/source_build_manifest_20261028.json').read_text(encoding='utf-8'))
after_hashes=Counter(x['sha256'] for x in receipt['figures'])
prior_hashes=Counter(x['sha256'] for x in before['figures'])
missing_figures=dict(prior_hashes-after_hashes)
result={'prior_files_changed':hash_errors,'formulas':equations,'requirements':len(req),'missing_requirements':requirement_errors,'prior_figure_positions':sum(prior_hashes.values()),'new_figure_positions':sum(after_hashes.values()),'missing_prior_figure_positions':missing_figures,'scope':'Deterministic preservation checks; changed prose and figure interpretation require semantic review.'}
(ROOT/'deliverables/preservation_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='formulas'}|{'missing_or_changed_formulas':sum(len(x['missing_or_changed']) for x in equations)},ensure_ascii=False))
raise SystemExit(bool(hash_errors or requirement_errors or missing_figures or any(x['missing_or_changed'] for x in equations)))
