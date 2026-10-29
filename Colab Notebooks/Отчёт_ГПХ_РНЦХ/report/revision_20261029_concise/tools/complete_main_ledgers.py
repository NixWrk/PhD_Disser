"""Link the editor's final paragraphs to frozen evidence sections."""
from pathlib import Path
import json,re,hashlib
ROOT=Path(__file__).resolve().parents[1]
def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
maps={
'02':{
 'chapters/02_method.md':{'2.1':['4.1','4.2'],'2.2':['4.3','4.4','4.5','2.2'],'2.3':['4.1','4.3'],'2.4':['4.5','4.6','4.7','4.8','4.9']},
 'supplement/02_method.md':[['4.7'],['4.4','4.5','4.6'],['4.7','4.9','12.7']]},
'07':{
 'chapters/07_validation.md':{'7.1':['11.1','11.4','11.5','11.6','11.9'],'7.2':['11.2','11.6','11.7','12.4','12.5'],'7.3':['11.8','12.7','12.8'],'7.4':['11.7','11.9','12.1','12.8']},
 'supplement/07_validation.md':[['11.1','11.4'],['11.2'],['11.4'],['11.5','11.6']]}}
for key,files in maps.items():
 d=read(f'reviews/{key}_ledger.json')
 if key=='02':
  other=read('reviews/07_ledger.json')
  d['sources']=[x for x in d['sources'] if x['source_id']!='OLD_12']+[x for x in other['sources'] if x['source_id']=='OLD_12']
  d['evidence']=[x for x in d['evidence'] if x['source_id']!='OLD_12']+[x for x in other['evidence'] if x['source_id']=='OLD_12' and x['locator'].startswith('12.7 ')]
  # Added editorial paragraph on the actual precardial branch has its own
  # frozen experimental / direct-FEM evidence, rather than only the aim chapter.
  for fn,sid in [('06_forward.md','OLD_06_PC'),('07_experiments.md','OLD_07_PC')]:
   rel='revision_20261029_contract/chapters/'+fn
   path=ROOT.parent/rel
   d['sources']=[x for x in d['sources'] if x['source_id']!=sid]+[{'source_id':sid,'file':rel,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}]
   d['evidence']=[x for x in d['evidence'] if x['source_id']!=sid]
   section=''
   for j,block in enumerate(re.split(r'\n\s*\n',path.read_text(encoding='utf-8'))):
    if block.startswith('## '):section=block.splitlines()[0][3:]
    if 'прекарди' in block.lower() or 'ТЕПС-2' in block:
     d['evidence'].append({'evidence_id':sid+'-'+str(j),'source_id':sid,'locator':'2.2 additional; source '+section+', block '+str(j),'fragment':block,'verification_status':'extracted','support_type':'context'})
 claims=[]
 for name,mapping in files.items():
  text=(ROOT/name).read_text(encoding='utf-8')
  sections=re.split(r'(?m)^(## .+)$',text)
  for i in range(1,len(sections),2):
   title=sections[i][3:]
   if title.startswith('Источники'):break
   section_index=(i-1)//2
   prefixes=mapping[section_index] if isinstance(mapping,list) else mapping[title.split()[0]]
   evidence=[x for x in d['evidence'] if any(x['locator'].startswith(s+' ') for s in prefixes)]
   assert evidence,(name,title,prefixes)
   source_ids=list(dict.fromkeys(x['source_id'] for x in evidence))
   for block in re.split(r'\n\s*\n',sections[i+1]):
    block=block.strip()
    if not block or block.startswith(('<details','</details','<summary','![')):continue
    rows=[x for x in block.splitlines() if x.startswith('|') and not re.match(r'^\|[\s|:-]+$',x)] if block.startswith('|') else [block]
    for row in rows:
     claims.append({'id':f'C{key}-{len(claims)+1:03d}','text':row,'type':'method_or_bounded_result','status':'bounded','source_ids':source_ids,'evidence_ids':[x['evidence_id'] for x in evidence],'locator':title,'boundary':'Редакционное сокращение замороженной версии; условия, вид проверки и недоступность части первичных материалов сохраняются. Ссылки на разделы не заменяют ручную сверку численных результатов.','destination':name,'output_section':title})
 d['claims']=claims
 d['semantic_review']='High-impact quantities manually checked against old11 tables11.1/11.6 and §11.4–11.6; independent review reviews/02_07_independent_review.md. Source-section links preserve context; not claimed as new primary validation.'
 (ROOT/f'reviews/{key}_ledger.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
 print(key,len(claims))
