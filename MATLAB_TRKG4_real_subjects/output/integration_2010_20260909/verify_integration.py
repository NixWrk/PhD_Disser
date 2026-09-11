"""Read-only migration checks plus reader HTML export for Markdown contracts."""
from pathlib import Path
import hashlib
import json
import re
import shutil
from urllib.parse import unquote, urlsplit

import nbformat
from bs4 import BeautifulSoup
from nbconvert import HTMLExporter

PROJECT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parent
BEFORE = RUN / 'before'
CORPUS = PROJECT / 'Colab Notebooks'
MODEL = PROJECT / 'MATLAB_TRKG4_real_subjects'
STEM = '20.10_КТ_FEM_локализация_боковых_матриц_МГТУ'
NEW = CORPUS / (STEM + '.ipynb')
OLD_REL = Path('MATLAB_TRKG4_real_subjects/notebooks/TRKG4_inverse_inhale_results.ipynb')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(cell):
    return ''.join(cell['source'])


def read_nb(path):
    return json.loads(path.read_text(encoding='utf-8'))


def payloads(nb, mime):
    return [o['data'][mime] for c in nb['cells'] for o in c.get('outputs', [])
            if mime in o.get('data', {})]


before = read_nb(BEFORE / OLD_REL)
after = read_nb(NEW)
assert [c['cell_type'] for c in before['cells']] == [c['cell_type'] for c in after['cells']]
code_changes = []
mapping = []
preserved_outputs = 0
for i, (old, new) in enumerate(zip(before['cells'], after['cells'])):
    mapping.append({'index': i, 'type': old['cell_type'], 'old_id': old.get('id'), 'new_id': new.get('id')})
    if old['cell_type'] != 'code':
        continue
    assert old.get('id') == new.get('id'), i
    assert old.get('execution_count') == new.get('execution_count'), i
    if source(old) != source(new):
        code_changes.append(i)
    old_outputs = [o for o in old.get('outputs', []) if 'text/markdown' not in o.get('data', {})]
    new_outputs = [o for o in new.get('outputs', []) if 'text/markdown' not in o.get('data', {})]
    assert old_outputs == new_outputs, f'Changed numerical/graphical output in cell {i}'
    preserved_outputs += len(old_outputs)
assert code_changes == [3, 32], code_changes
assert payloads(before, 'image/png') == payloads(after, 'image/png')
assert payloads(before, 'text/html') == payloads(after, 'text/html')
assert len(payloads(after, 'image/png')) == 18
pointer = read_nb(PROJECT / OLD_REL)
assert len(pointer['cells']) == 1 and pointer['cells'][0]['cell_type'] == 'markdown'
assert STEM in unquote(source(pointer['cells'][0]))
manuscript = MODEL / 'notebooks/TRKG4_inverse_inhale_scientific.md'
assert sha(manuscript) == '69bb88d864c82306a8e139f69f9606b962ddc8ff0f8176180b3284888200edac'

summary = json.loads((MODEL / 'output/two_tissue_auditfix_20260908/comparison_summary.json').read_text(encoding='utf-8'))
producer_sha = summary['source_sha256']
producer = MODEL / 'output/two_tissue_auditfix_20260908/source_used' / f'compare_curved_two_tissue_{producer_sha}.py'
assert sha(producer) == producer_sha

exported = []
for prefix in ('20.11_', '20.12_'):
    notebook = next(CORPUS.glob(prefix + '*.ipynb'))
    nb = nbformat.read(notebook, as_version=4)
    assert all(c.cell_type == 'markdown' for c in nb.cells)
    nbformat.validate(nb)
    target = notebook.with_suffix('.html')
    backup = BEFORE / target.relative_to(PROJECT)
    if target.exists() and not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup)
    exporter = HTMLExporter(template_name='lab')
    exporter.exclude_input = True
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    body, _ = exporter.from_notebook_node(nb)
    target.write_text(body, encoding='utf-8')
    exported.append(str(target.relative_to(PROJECT)))
nbformat.validate(nbformat.read(NEW, as_version=4))


def markdown_links(text):
    return [a or b for a, b in re.findall(r'\[[^\]\n]*\]\((?:<([^>]+)>|([^\s)]+))\)', text)]


def links_in(path):
    text = path.read_text(encoding='utf-8')
    if path.suffix == '.ipynb':
        nb = json.loads(text)
        return markdown_links('\n'.join(source(c) for c in nb['cells'] if c['cell_type'] == 'markdown'))
    if path.suffix == '.html':
        soup = BeautifulSoup(text, 'html.parser')
        return [a['href'] for a in soup.find_all('a', href=True)]
    return markdown_links(text)


def broken_links(content_path, resolve_from):
    broken = set()
    local_count = 0
    for link in links_in(content_path):
        parsed = urlsplit(link)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        local_count += 1
        target = (resolve_from.parent / unquote(parsed.path)).resolve()
        if not target.exists():
            broken.add(link)
    return broken, local_count


files = [NEW, NEW.with_suffix('.html'), NEW.with_suffix('.md'), PROJECT / OLD_REL,
         (PROJECT / OLD_REL).with_suffix('.html')]
files += [next(CORPUS.glob(prefix + '*' + ext)) for prefix in ('20.11_', '20.12_') for ext in ('.ipynb', '.html')]
files += [CORPUS / name for name in (
    'README.md', '00.00_Карта_проекта.md', '00.02_Реестр_доказательств_и_ограничений.md',
    'TODO.md', 'ЖУРНАЛ_РЕШЕНИЙ.md', 'ПЛАН_РЕОРГАНИЗАЦИИ_И_ПЕРЕИМЕНОВАНИЯ.md',
    '34.03_КТ_как_референс_разработки.md', '30.01_Прямая_двуслойная_модель_боковой_сборки.md',
    'AGENTS.md')]
files += [MODEL / 'docs/PIPELINE_FOR_NOTEBOOKS.md', MODEL / 'AGENTS.md']
link_checks = []
for path in files:
    assert path.exists(), path
    relative = path.relative_to(PROJECT)
    broken, count = broken_links(path, path)
    baseline = BEFORE / relative
    old_broken = broken_links(baseline, path)[0] if baseline.exists() else set()
    introduced = sorted(broken - old_broken)
    link_checks.append({'file': str(relative), 'local_links': count,
                        'preexisting_broken': sorted(broken & old_broken), 'introduced_broken': introduced})
    assert not introduced, (str(relative), introduced)

result = {
    'status': 'passed', 'canonical_notebook': str(NEW.relative_to(PROJECT)),
    'cells': len(after['cells']), 'code_cells': 25,
    'code_changes': {'3': 'root discovery only', '32': 'hash-verified historical producer source lookup'},
    'unchanged_code_cell_sources': 23, 'all_code_ids_and_execution_counts_preserved': True,
    'non_markdown_outputs_preserved_exactly_against_pre_migration': preserved_outputs,
    'png_outputs_preserved_exactly': 18, 'all_html_outputs_preserved_exactly': True,
    'scientific_source_sha256_unchanged': sha(manuscript),
    'historical_producer_source_sha256': producer_sha,
    'old_notebook_is_one_markdown_pointer': True,
    'neighbor_notebooks_are_markdown_contracts_without_execution': True,
    'neighbor_html_exported': exported, 'links': link_checks,
    'numerical_recalculation_performed': False,
    'language_review_scope': 'Structural and navigation edits; prior full scientific review remains bound to its original hashes.',
    'sha256': {str(p.relative_to(PROJECT)): sha(p) for p in files},
}
(RUN / 'cell_mapping_all.json').write_text(json.dumps(mapping, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
(RUN / 'integration_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k not in ('sha256', 'links')}, ensure_ascii=False, indent=2))
print('Local links checked:', sum(x['local_links'] for x in link_checks))
print('Files with pre-existing broken links:', [x['file'] for x in link_checks if x['preexisting_broken']])
