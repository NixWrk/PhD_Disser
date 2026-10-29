"""Trace the scientific/editorial delta; existing chapter ledgers remain sources."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'revision_20261028'
REPO=ROOT.parents[3]
def source(sid,p,title,kind):
    return {'source_id':sid,'title':title,'content_hash':hashlib.sha256(p.read_bytes()).hexdigest(),'representation':kind,'local_ref':p.relative_to(REPO).as_posix()}
sources=[source('contract',ROOT/'contract_requirements.json','Переданные требования и решение об основной цели УО','data'),source('prior_appendices',OLD/'appendices.md','Границы исходной редакции и состав поставки','markdown'),source('prior_ch10',OLD/'chapters/10_chambers.md','Камерные проверки и смысл УО','markdown')]
evidence=[];claims=[];results=[]
def add(sid,loc,text,section,kind='interpretive',boundary='Вывод ограничен названным источником и выполненной проверкой.',value=None,unit=None):
    n=len(claims)+1;eid=f'EV-{n:03d}';cid=f'CL-{n:03d}';rid=f'RES-{n:03d}'
    evidence.append({'evidence_id':eid,'source_id':sid,'locator':loc,'claim':text,'support_type':'limitation' if kind=='limitation' else 'direct','verification_status':'verified','limitations':boundary})
    if value is not None:
        results.append({'result_id':rid,'source_id':sid,'locator':loc,'value':value,'unit':unit,'version':next(x['content_hash'] for x in sources if x['source_id']==sid),'analysis':'Сохранённый результат; при редакционной переработке не пересчитывается.'})
    claims.append({'claim_id':cid,'text':text,'output_section':section,'claim_type':'numeric' if value is not None else kind,'certainty':'direct','evidence_ids':[eid],'result_ids':[rid] if value is not None else [],'status':'bounded','disposition':'keep_with_boundary','boundary':boundary,'causal_basis':None})
add('contract','primary_endpoint; additional_direction','УО является основной целью; ФВ сохраняется как дополнительное направление.','Введение и глава 4',boundary='Решение автора от 29.10.2026; изменение целей не повышает доказательный статус расчётов.')
add('prior_appendices','Приложение А, строки программной реализации и количественной работоспособности','Единый подтверждённый экспериментальный расчёт конечного УО по всей последовательности отсутствует.','Матрица договора','limitation')
add('prior_appendices','Приложение А, презентационные материалы; Приложение Б, статус исторической описи','Полный исследовательский архив не перепроверен; презентация не включена в текущую редакционную поставку.','Приложения А и Б','limitation')
add('prior_ch10','Таблица 10.5, ЛЖ, блоки первого интервала','В данной проверке электрические признаки не улучшают постоянную опору по RMSE камерного объёма.','Глава 11',boundary='Один участник, межсессионное сопоставление; это ошибки камерного объёма, а не УО.',value='25,84; 29,57; 31,96',unit='мл')
add('prior_ch10','Таблица 10.6, ЛЖ, КТ и отложенные блоки','Малая ошибка размаха сопровождается ошибочным положением максимума и не подтверждает точный УО.','Глава 10',boundary='Неполное окно и составная последовательность четырёх калибровок.',value='+0,168; 43; 92',unit='мл; % R–R; % R–R')
add('prior_ch10','Формула 10.20 и пояснение','Общее смещение может сократиться в межфазной разности; абсолютный КДО дополнительно нужен для ФВ, а масштаб миллилитров остаётся существенным для УО.','Глава 4 и дополнительное направление ФВ',boundary='Алгебраическое свойство общих ошибок; не утверждение, что все реальные ошибки являются общими.')
figs=json.loads((ROOT/'figures/html_exports/manifest.json').read_text(encoding='utf-8'))
figure_ids=[]
for i,f in enumerate(figs['figures'],1):
    if i in (1,2):continue  # Named rasters were excluded from the publication.
    sid=f'HTML-{i:02d}';p=REPO/f['source_path'];sources.append(source(sid,p,f['source_figure'],'html'));figure_ids.append(sid)
    add(sid,f"img[{f['image_index_zero_based']}]; {f['source_figure']}",f['caption'],f['recommended_chapter_before_renumbering'],boundary=f['status']+' '+' '.join(f['limitations']))
bundle={'schema_version':'1.0','bundle_id':'REPORT-CONTRACT-SV-20261029-DELTA','task':{'mode':'manuscript','request':'Изменить приоритет на УО, сохранить ФВ как дополнительное направление, построить матрицу договора и включить проверяемые иллюстрации. Реестр охватывает редакционное изменение; полные основания прежних результатов сохранены в рецензиях глав.','input_scope':[s['source_id'] for s in sources],'language':'ru','audience':'Автор и научный рецензент','figure_mode':'with_figures','formatting_mode':'section_only','figure_source_ids':figure_ids},'sources':sources,'evidence':evidence,'results':results,'claims':claims}
(ROOT/'reviews/evidence_delta.json').write_text(json.dumps(bundle,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'sources':len(sources),'claims':len(claims),'results':len(results)},ensure_ascii=False))
