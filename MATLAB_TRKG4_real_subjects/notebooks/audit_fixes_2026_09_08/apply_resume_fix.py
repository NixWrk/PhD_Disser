"""Apply only the reviewed driver and tests, after the frozen run has finished."""
import hashlib,json
from pathlib import Path
here=Path(__file__).resolve().parent
root=here.parents[1]
candidate=here/'resume_fix_candidate'
manifest=json.loads((candidate/'manifest.json').read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
summary=json.loads((root/'output/nik_trkg4_inverse_direct_search_summary_auditfix_20260908.json').read_text(encoding='utf-8'))
assert summary['finalization']['status']=='complete'
contract=json.loads((root/'output/nik_trkg4_inverse_inhale_contract_auditfix_20260908.json').read_text(encoding='utf-8'))
for record in contract['artifacts']:
    assert sha(root/record['path'])==record['sha256'],record['path']
changes=[]
for name,relative in manifest['apply_map'].items():
    target=(root/relative).resolve();proposed=candidate/name
    assert target.is_relative_to(root)
    expected_base=manifest['base_sha256'][relative]
    assert sha(target)==expected_base,relative+' changed since review'
    assert sha(proposed)==manifest['candidate_sha256'][name],name+' changed since tests'
    changes.append((target,proposed,relative,expected_base))
for target,proposed,relative,old_hash in changes:
    target.write_bytes(proposed.read_bytes())
record={'status':'applied_after_frozen_run_completed','changes':[
    {'path':relative,'before_sha256':old_hash,'after_sha256':sha(target)}
    for target,proposed,relative,old_hash in changes],
    'FEM_sources_changed':False,'search_identities_rewritten':False,
    'numerical_run_source_archive':'source_used_auditfix_20260908.zip'}
(here/'resume_fix_applied.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(record,ensure_ascii=False))
