from pathlib import Path
import json,hashlib,nbformat
from bs4 import BeautifulSoup
from datetime import datetime,timezone
root=Path('Colab Notebooks/cardiac_4d_segmentation');w=root/'results/old_verification_20260914'
p=root/'21.08_preparation_checks.json';r=json.loads(p.read_text(encoding='utf8'))
r['checks']['runner_review']['regression_tests']={'passed':3,'file':'tests/test_historical_cardiac_runner.py','scope':'QC rejection persists across resume; extra input phase blocks inference; changed weights block phase acceptance'}
for n in list(r['artifact_hashes'])+['tests/test_historical_cardiac_runner.py']:
 r['artifact_hashes'][n]=hashlib.sha256((root/n).read_bytes()).hexdigest()
r['checks']['russian_protocol_review']={'status':'passed','reviewer':'orchestrator after final pilot-QC addition; preceding draft reviewed by Luna Max','full_sequential_read':True,'headings_terminology_definitions_and_scope_checked':True,'path':'21.08_Историческая_контрастная_КТ_объёмы_камер.md','sha256':r['artifact_hashes']['21.08_Историческая_контрастная_КТ_объёмы_камер.md']}
r['pilot_QC']=json.loads((w/'pilot_boundary_qc.json').read_text(encoding='utf8'))
npth=root/'21.08_Историческая_контрастная_КТ_объёмы_камер.ipynb';nb=nbformat.read(npth,as_version=4);nbformat.validate(nb)
soup=BeautifulSoup(npth.with_suffix('.html').read_text(encoding='utf8'),'html.parser')
assert not soup.select('.jp-CodeCell .jp-InputArea,.jp-InputPrompt,.jp-OutputPrompt')
assert nb.metadata['summary_phase_count']==4
r['pilot_report']={'status':'preliminary_automatic_report','phases':4,'nbformat':'passed','html_has_no_code':True,'full_reader_text_read':True,'language_status':'draft_with_remaining_editorial_findings','editorial_followup':'Expand boundary-contact discussion in the generated report and polish the sentence about transfer to weakly contrasted CT; detailed current QC is in the protocol Markdown.','visual_review':'not_performed','hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [npth,npth.with_suffix('.html')]},'may_be_regenerated_by_running_full_sequence':True}
r['checked_at']=datetime.now(timezone.utc).isoformat();p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
paths=['MATLAB_TRKG4_real_subjects/tools/heart_sphere_methods_ru.md','Colab Notebooks/40.15_Эквивалентная_сфера_сердца_методики_и_проверка.ipynb','Colab Notebooks/40.15_Эквивалентная_сфера_сердца_методики_и_проверка.html','MATLAB_TRKG4_real_subjects/docs/HEART_SPHERE_METHODS.language-review.json','MATLAB_TRKG4_real_subjects/docs/HEART_SPHERE_METHODS.artifact-checks.json']
paths += [(root/n).as_posix() for n in ['enhanced_cardiac_ct.py','tests/test_enhanced_cardiac_ct.py','tests/test_historical_cardiac_runner.py','run_historical_cardiac_sequence.py','analyze_historical_cardiac_sequence.py','21.08_Историческая_контрастная_КТ_объёмы_камер.md','21.08_Историческая_контрастная_КТ_объёмы_камер.ipynb','21.08_Историческая_контрастная_КТ_объёмы_камер.html','21.08_preparation_checks.json','README.md','21.00_Карта_4D_сердца_и_RR.md']]
(w/'commit_spec.json').write_text(json.dumps({'message':'feat: prepare historical cardiac CT sequence and reference-free volume pilot','paths':paths},ensure_ascii=False,indent=2),encoding='utf8')
print('Prepared scoped commit for',len(paths),'files; primary images and masks excluded.')
