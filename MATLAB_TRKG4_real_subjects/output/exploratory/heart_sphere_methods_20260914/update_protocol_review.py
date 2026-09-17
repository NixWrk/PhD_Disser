from pathlib import Path
import json,hashlib,datetime
root=Path(__file__).resolve().parents[4]
p=root/'MATLAB_TRKG4_real_subjects/docs/TEPC_PHYSICAL_VALIDATION_PROTOCOL.md'
r=p.with_suffix('.language-review.json')
d=json.loads(r.read_text(encoding='utf-8-sig'))
d['artifact_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
d['artifact_bytes']=p.stat().st_size
d['reviewed_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
d['review_status']='passed'
if 'language_review' in d and isinstance(d['language_review'],dict): d['language_review']['status']='passed'
if 'checks' in d:
    d['checks']['language.ru.claim-status-and-limitations']='passed'
d['attestation']['full_text_read_sequentially_after_last_byte_change']=True
d['attestation']['open_language_findings']=0
d['language_findings']=[]
d['resolved_findings']=[{'line':156,'resolution':'Input checker output is explicitly limited to implemented structural/hash/run_id checks; independence and temporal provenance remain separate unresolved verification requirements.'}]
d['conclusion']='После уточнения области программной проверки весь текст (строки 1–162) последовательно вычитан оркестратором. Языковая проверка не подтверждает полноту программного контракта или выполнение физической валидации.'
d['physical_input_validator_status']='preparation_only_contract_incomplete'
d['physical_validation_status']='not_performed'
d['computational_validation_status']='volume_run_in_progress'
d['line_ending_normalization']['followed_by_claim_scope_correction']=True
r.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
for stem in ['TEPC_VALIDATION_PLAN','TEPC_PHYSICAL_VALIDATION_PROTOCOL']:
    doc=root/f'MATLAB_TRKG4_real_subjects/docs/{stem}.md'
    record=json.loads(doc.with_suffix('.language-review.json').read_text(encoding='utf-8-sig'))
    assert hashlib.sha256(doc.read_bytes()).hexdigest()==record['artifact_sha256'].lower()
print('Both TEPC document reviews match their final bytes')
