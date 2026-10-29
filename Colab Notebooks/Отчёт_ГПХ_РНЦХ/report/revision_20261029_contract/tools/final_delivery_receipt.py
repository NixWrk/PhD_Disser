"""Record the reviewed delivery without modifying the Word document."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import zipfile
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[3]
DOCX = ROOT / 'deliverables/Отчёт_ГПХ_РНЦХ_20261029_УО_договор_для_рецензирования.docx'
PDF = REPO / 'output/gph_report_20261029/render_handoff/report.pdf'
OLD = ROOT.parent / 'revision_20261028/deliverables/Отчёт_ГПХ_РНЦХ_20261028_для_рецензирования.docx'
EXPECTED_OLD = '64bb37e36f439f5b7ad0a4ef24e1f29a0b38d785e2f6b6ea9c0ee088ef01ea4c'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8-sig'))

manifest = read('build_manifest.json')
structure = read('structure_check.json')
preservation = read('deliverables/preservation_audit.json')
layout = read('deliverables/layout_audit.json')
docx_audit = read('deliverables/docx_audit.json')
evidence = read('reviews/evidence_validation.json')
reviews = [f'reviews/visual_final_{a}_{b}.md' for a,b in [(1,77),(78,154),(155,231),(232,308)]]
assert all((ROOT/p).is_file() for p in reviews), 'Final visual reviews still pending'
assert not structure['errors'] and not docx_audit['errors']
assert not preservation['prior_files_changed'] and not preservation['missing_requirements']
assert not preservation['missing_prior_figure_positions']
assert not any(x['missing_or_changed'] for x in preservation['formulas'])
assert not any(p['overflow'] for p in layout['pages'])
assert evidence['valid']
assert all(sha(ROOT/x['file']) == x['sha256'] for x in manifest['sources'])
assert all(sha(ROOT/x['file']) == x['sha256'] for x in manifest['figures'])
assert sha(DOCX) == docx_audit['docx_sha256']
assert sha(OLD) == EXPECTED_OLD

ns = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
with zipfile.ZipFile(DOCX) as z:
    settings = etree.fromstring(z.read('word/settings.xml'))
    document = etree.fromstring(z.read('word/document.xml'))
    track = settings.xpath('//w:trackRevisions', namespaces=ns)
    assert len(track) == 1 and track[0].get('{'+ns['w']+'}val','true') not in ('0','false','off')
    revisions = len(document.xpath('//w:ins|//w:del|//w:moveFrom|//w:moveTo', namespaces=ns))
    assert revisions == 0

text = '\n'.join(p.read_text(encoding='utf-8') for p in (ROOT/'chapters').glob('*.md'))
captions = re.findall(r'Рисунок\s+(\d+\.\d+)\s*[—–-]', text)
references = re.findall(r'рисунк[аеуиов]*\s+(\d+\.\d+)', text)
assert len(captions) == len(set(captions)) == 30
assert not set(references)-set(captions)

receipt = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'document': DOCX.name,
    'document_sha256': sha(DOCX),
    'qa_pdf': PDF.relative_to(REPO).as_posix(),
    'qa_pdf_sha256': sha(PDF),
    'pages': len(layout['pages']),
    'chapters': len(structure['chapters']),
    'contract_requirements': preservation['requirements'],
    'figures': len(captions),
    'retained_prior_figure_positions': preservation['prior_figure_positions'],
    'additional_html_figures': 6,
    'tables': docx_audit['tables'],
    'numbered_equations': sum(x['equations'] for x in structure['chapters']),
    'native_math_objects': docx_audit['native_math_objects'],
    'bibliography_entries': manifest['bibliography_entries'],
    'track_changes_enabled': True,
    'existing_revisions': revisions,
    'old_document_unchanged': True,
    'old_document_sha256': sha(OLD),
    'source_hashes_match_build': True,
    'visual_reviews': reviews,
    'review_scope': 'All final pages inspected; layout checks do not constitute independent scientific validation.',
    'language_review': 'reviews/language_manual_review.md; heuristic findings retained, not reported as an automated pass',
    'unavailable_historical_references': len(structure['unavailable_historical_references']),
    'verified_cached_figure_references': len(structure['verified_cached_figures']),
    'limits': [
        'Stroke volume is the primary development objective, not a claimed independently validated endpoint.',
        'Ejection fraction is a supplementary research direction.',
        'Full medical/segmentation/STL/data archives and a presentation are not newly assembled or verified by this editorial delivery.',
        'No new physiological measurements, FEM runs or clinical validation were performed.',
    ],
}
(ROOT/'deliverables/review_delivery.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k:receipt[k] for k in ['document','pages','figures','contract_requirements','track_changes_enabled','old_document_unchanged']}, ensure_ascii=False))
