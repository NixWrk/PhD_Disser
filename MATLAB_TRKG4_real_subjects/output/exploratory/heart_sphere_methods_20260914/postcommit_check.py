from pathlib import Path
import json,hashlib,subprocess
root=Path.cwd()
project=root/'MATLAB_TRKG4_real_subjects'
checks=json.loads((project/'docs/HEART_SPHERE_METHODS.artifact-checks.json').read_text(encoding='utf-8'))
for item in checks['artifacts']:
    current=(root/item['path']).read_bytes()
    committed=subprocess.check_output(['git','show','HEAD:'+item['path']],cwd=root)
    assert hashlib.sha256(current).hexdigest()==item['sha256']
    assert current==committed,item['path']
for stem in ['TEPC_VALIDATION_PLAN','TEPC_PHYSICAL_VALIDATION_PROTOCOL']:
    d=json.loads((project/f'docs/{stem}.language-review.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((project/f'docs/{stem}.md').read_bytes()).hexdigest()==d['artifact_sha256'].lower()
    print(stem, d.get('review_status'), [k for k,v in d.items() if isinstance(v,str) and ('finding' in v or 'clarification' in v)])
run=project/'output/exploratory/tepc_volume_convergence_20260914_r2'
d=json.loads((run/'detached_status.json').read_text(encoding='utf-8-sig'))
print(json.dumps({'reader_bytes_match_commit':True,'background_stage':d['stage'],'background_status':d['status'],'updated':d.get('updated')},ensure_ascii=False))
