from pathlib import Path
import fitz, hashlib, json, re, datetime
repo=Path.cwd()
out=repo/'MATLAB_TRKG4_real_subjects/output/exploratory/heart_sphere_methods_20260914/source_extracts'
out.mkdir(parents=True, exist_ok=True)
storage=Path(r'C:\PC\Zotero\Zotero_Heart_n_Lung_Data\storage')
sources=[
('tikhomirov_dissertation','UHVBNS3Z','NUZKBZGS','Тихомиров - Компьютерная система многоканальной электроимпедансной кардиографии.pdf'),
('tikhomirov_abstract','BUPQP59M','FIW7DWQH','Автореферат Тихомиров АН.pdf'),
('rfbr_2021','X9VNDF2Z','3NAQJNXQ','Отчет РФФИ 2021 v2.1.pdf'),
('rfbr_2020','GUT5IB6Q','SG58QNPU','Отчет РФФИ 2020.pdf')]
records=[]
for sid,parent,key,name in sources:
    path=storage/key/name
    records.append(dict(source_id=sid,zotero_item_key=parent,attachment_key=key,filename=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),selection_reason='author-requested predecessor for heart-sphere methods; primary local PDF'))
manifest=dict(schema='heart_sphere_selected_sources_v1',selection_mode='fixed',frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_authority='user-authorized local Zotero primary texts',research_questions=['definition of equivalent sphere','origin and frame of motion coordinates','geometric vs electrical validation','limits of transfer to current CT/EIDORS setting'],sources=records)
(repo/'MATLAB_TRKG4_real_subjects/docs/HEART_SPHERE_SOURCE_CORPUS_20260914.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
for spec,record in zip(sources,records):
    sid,parent,key,name=spec
    path=storage/key/name
    doc=fitz.open(path)
    pages=[dict(pdf_page=i+1,text=p.get_text()) for i,p in enumerate(doc)]
    (out/f'{sid}.pages.json').write_text(json.dumps(pages,ensure_ascii=False),encoding='utf8')
    (out/f'{sid}.txt').write_text('\n\n'.join(f"=== PDF PAGE {p['pdf_page']} ===\n{p['text']}" for p in pages),encoding='utf8')
    hits=[]
    for p in pages:
        text=p['text']
        matches=list(re.finditer('сфер',text,re.I))
        if matches:
            hits.append(dict(pdf_page=p['pdf_page'],count=len(matches),contexts=[re.sub(r'\s+',' ',text[max(0,m.start()-100):m.start()+200]) for m in matches[:2]]))
    print(json.dumps(dict(source_id=sid,pages=len(pages),sha256=record['sha256'],cover=pages[0]['text'][:2100],sphere_pages=hits),ensure_ascii=False))
