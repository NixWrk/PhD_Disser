"""Audit this editorial delivery, without claiming a new scientific validation."""
from pathlib import Path
import argparse, hashlib, json, re, zipfile
from lxml import etree

ap=argparse.ArgumentParser()
ap.add_argument('--render',required=True,type=Path)
args=ap.parse_args()
root=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
errors=[]
snapshot=read(root/'reviews/input_snapshot.json')
preserved=[]
for s in snapshot['sources']:
    p=root.parent/s['file']
    ok=p.is_file() and sha(p)==s['sha256']
    preserved.append({'file':s['file'],'unchanged':ok})
    if not ok: errors.append('Changed original: '+str(p))
manifest=read(root/'build_manifest.json')
for s in manifest['sources']:
    if sha(root/s['file'])!=s['sha256']:errors.append('Build source changed: '+s['file'])
old=read(root.parent/'revision_20261029_contract/contract_requirements.json')['requirements']
new=read(root/'contract_requirements.json')['requirements']
contract_equal=[(x['id'],x['requirement']) for x in old]==[(x['id'],x['requirement']) for x in new]
if len(new)!=41 or not contract_equal:errors.append('Contract text/count differs')
if (root/'appendices.md').read_bytes()!=(root/'contract_matrix.md').read_bytes():errors.append('Appendix A differs from matrix')
if any(not x.get('evidence') or not x.get('status') for x in new):errors.append('Incomplete matrix row')
source_fig=read(root/'reviews/source_build_manifest.json')['figures']
allowed={x['sha256'] for x in source_fig}
for f in manifest['figures']:
    if sha(root/f['file'])!=f['sha256'] or f['sha256'] not in allowed:errors.append('Unverified figure: '+f['file'])
assembled=(root/'report_for_word.md').read_text(encoding='utf-8')
tags=re.findall(r'\\tag\{([^}]+)\}',assembled)
if len(tags)!=len(set(tags)):errors.append('Duplicate equation labels')
main='\n'.join((root/f['file']).read_text(encoding='utf-8') for f in manifest['sources'] if f['file'].startswith('chapters/'))
main_tags=re.findall(r'\\tag\{([^}]+)\}',main)
for path in sorted((root/'chapters').glob('*.md')):
    tags_ch=re.findall(r'\\tag\{(\d+)\.(\d+)\}',path.read_text(encoding='utf-8'))
    if tags_ch and [int(b) for a,b in tags_ch]!=list(range(1,len(tags_ch)+1)):errors.append('Nonconsecutive equations '+path.name)
docx=root/'deliverables/Отчёт_ГПХ_РНЦХ_20261029_сокращённый_для_рецензирования.docx'
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
with zipfile.ZipFile(docx) as z:
    if z.testzip():errors.append('DOCX ZIP integrity')
    settings=etree.fromstring(z.read('word/settings.xml'))
    xml=etree.fromstring(z.read('word/document.xml'))
    tracked=bool(settings.xpath('//w:trackRevisions',namespaces=ns))
    revisions=len(xml.xpath('//w:ins|//w:del|//w:moveFrom|//w:moveTo',namespaces=ns))
    if not tracked or revisions:errors.append('Track changes not clean/enabled')
    drawing_count=len(xml.xpath('//w:drawing',namespaces=ns))
    if drawing_count!=len(manifest['figures']):errors.append('Image count differs')
    text=' '.join(xml.xpath('//w:t/text()|//m:t/text()',namespaces=ns))
    if re.search(r'\\(?:frac|begin|tag|mathrm)\b|\[\[TOC\]\]',text):errors.append('Unconverted source markup')
pages=sorted(args.render.glob('page-*.txt'))
heads={}
for p in pages:
    # Word's PDF text stream emits the footer page number before body text.
    t=re.sub(r'^\d+\s*\n\s*','',p.read_text(encoding='utf-8').strip())
    for heading in ['Введение','Список использованных источников','Приложение А','Приложение Б','Приложение И']:
        if t.startswith(heading):heads[heading]=int(p.stem.split('-')[1])
if not {'Введение','Список использованных источников'}<=set(heads):errors.append('Main page boundaries not found')
main_pages=None
if 'Введение' in heads and 'Список использованных источников' in heads:
    main_pages=heads['Список использованных источников']-heads['Введение']
    if not 60<=main_pages<=80:errors.append('Main body outside requested 60–80 pages')
pdf=read(root/'deliverables/layout_final.json')
if any(p['overflow'] for p in pdf['pages']):errors.append('PDF margin overflow')
ledger_counts={}
for p in sorted((root/'reviews').glob('[0-9][0-9]_ledger.json')):
    d=read(p); ledger_counts[p.name]=len(d.get('claims',[]))
result={'scope':'Editorial provenance, contract coverage and document structure. No new primary-data validation; visual inspection is separate.',
    'errors':errors,'word_file':str(docx),'word_sha256':sha(docx),'pdf_sha256':sha(args.render/'report.pdf'),
    'preserved_originals':preserved,'literal_contract_requirements':len(new),'contract_text_unchanged':contract_equal,
    'total_pages':len(pages),'main_body_pages':main_pages,'page_boundaries':heads,
    'selected_figures':drawing_count,'all_figures_match_prior_manifest':not any(e.startswith('Unverified figure') for e in errors),
    'main_display_equations':len(main_tags),'total_equation_labels':len(tags),'native_math_objects':len(xml.xpath('//m:oMath',namespaces=ns)),
    'tables':len(xml.xpath('//w:tbl',namespaces=ns)),'track_changes_enabled':tracked,'existing_revisions':revisions,
    'bibliography_entries':manifest['bibliography_entries'],'unavailable_source_references_annotated':len(manifest['unavailable_references']),
    'claim_ledger_entries':ledger_counts,'final_source_hashes':manifest['sources']}
(root/'deliverables/release_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:result[k] for k in ['errors','total_pages','main_body_pages','page_boundaries','selected_figures','main_display_equations','tables','track_changes_enabled']},ensure_ascii=False))
raise SystemExit(bool(errors))
