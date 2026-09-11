from pathlib import Path
import difflib
import hashlib
import json
import re
import subprocess
from urllib.parse import unquote, urlsplit

ROOT = Path.cwd()
ARC = ROOT / 'Colab Notebooks/archive/private/cardiac_docs_before_20260910'
OUT = ARC / 'proposed_commit'
PATHS = json.loads((ARC / 'inventory.json').read_text(encoding='utf-8'))

def git(*args):
    return subprocess.check_output(['git', *args]).decode('utf-8')

def section(text, heading):
    pattern = r'(?ms)^' + re.escape(heading) + r'\n.*?(?=^## |\Z)'
    hits = list(re.finditer(pattern, text))
    assert len(hits) == 1, (heading, len(hits))
    return hits[0].group().rstrip() + '\n'

def upsert_section(text, desired, heading):
    new = section(desired, heading)
    if heading in text:
        return text.replace(section(text, heading).rstrip(), new.rstrip(), 1)
    return text.rstrip() + '\n\n' + new

def line(text, prefix):
    hits = [s for s in text.splitlines() if s.startswith(prefix)]
    assert len(hits) == 1, (prefix, len(hits))
    return hits[0]

def replace_line(text, desired, prefix, old_prefix=None):
    return text.replace(line(text, old_prefix or prefix), line(desired, prefix), 1)

def paragraph(text, prefix):
    hits = [p for p in text.split('\n\n') if p.startswith(prefix)]
    assert len(hits) == 1, (prefix, len(hits))
    return hits[0]

def replace_paragraph(text, desired, prefix):
    return text.replace(paragraph(text, prefix), paragraph(desired, prefix), 1)

start_head = git('rev-parse', 'HEAD').strip()
assert not git('diff', '--cached', '--name-only').strip(), 'Shared index is not empty'
diffs = []
hashes = {}
for path in PATHS:
    head = git('show', 'HEAD:' + path).replace('\r\n', '\n')
    work = (ROOT / path).read_text(encoding='utf-8')
    if '/cardiac_4d_segmentation/' in path:
        base = (ARC / path).read_text(encoding='utf-8')
        assert head == base, 'Canonical documentation changed concurrently: ' + path
        target = work
    elif path == 'README.md':
        heading = '## 4D-сердце: геометрия и условные оценки УО/ФВ'
        if heading in head:
            target = upsert_section(head, work, heading)
        else:
            anchor = '## Подтверждённая последовательность исследования'
            assert head.count(anchor) == 1
            target = head.replace(anchor, section(work, heading) + '\n' + anchor, 1)
    elif path == 'Colab Notebooks/README.md':
        target = replace_line(head, work, '| `21`')
        if 'Для ветки сердца:' in head:
            target = replace_paragraph(target, work, 'Для ветки сердца:')
        else:
            target = target.rstrip() + '\n\n' + paragraph(work, 'Для ветки сердца:') + '\n'
    elif path.endswith('00.00_Карта_проекта.md'):
        target = replace_line(head, work, '| `21`')
        target = replace_paragraph(target, work, 'Серия `21`')
    elif path.endswith('/TODO.md'):
        target = upsert_section(head, work, '## Серия 21: уточнение условных УО/ФВ — 10.09.2026')
    elif path.endswith('/AGENTS.md'):
        target = head
        for heading in [
            '## Единая папка 4D-сердца и R–R',
            '## Контраст крови и миокарда в 4D-КТ — уточнение автора 10.09.2026',
            '## Роль новых ручных масок 4D-сердца — уточнение автора 10.09.2026',
            '## Работа без нового ручного референса — последнее решение автора 10.09.2026',
            '## Индексация объёмов и документация серии 21 — 10.09.2026',
        ]:
            target = upsert_section(target, work, heading)
    elif path.endswith('00.02_Реестр_доказательств_и_ограничений.md'):
        target = head
        for prefix in ['| TotalSegmentator 2.18.0', '| Выходной класс `heart_myocardium`', '| Межмодельный коэффициент Dice']:
            target = replace_line(target, work, prefix)
        target = replace_line(target, work, '| В серии `21` рассчитаны', '| Объёмные кривые камер серии `21`')
        anchor = line(target, '| В серии `21` рассчитаны')
        additions = [line(work, p) for p in ['| Совместный анализ ЛЖ/ПЖ', '| Деление объёмов на общую BSA']]
        target = target.replace(anchor, '\n'.join([anchor, *additions]), 1)
        target = target.rstrip() + '\n\n' + paragraph(work, 'Уточнение серии `21` от 10.09.2026:') + '\n'
    else:
        raise AssertionError(path)
    assert target != head, 'No changes: ' + path
    dest = OUT / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(target, encoding='utf-8', newline='\n')
    hashes[path] = {'working_sha256': hashlib.sha256((ROOT/path).read_bytes()).hexdigest(), 'proposed_sha256': hashlib.sha256(dest.read_bytes()).hexdigest()}
    delta = ''.join(difflib.unified_diff(head.splitlines(True), target.splitlines(True), fromfile='a/'+path, tofile='b/'+path, n=3))
    diffs.append(delta)
    print(path, 'added', sum(s.startswith('+') and not s.startswith('+++') for s in delta.splitlines()), 'removed', sum(s.startswith('-') and not s.startswith('---') for s in delta.splitlines()))

assert git('rev-parse', 'HEAD').strip() == start_head
(ARC/'proposed_documentation.diff').write_text('\n'.join(diffs), encoding='utf-8')
(ARC/'commit_scope.json').write_text(json.dumps({'head':start_head,'paths':PATHS,'hashes':hashes},ensure_ascii=False,indent=2),encoding='utf-8')

links=[]
for path in PATHS:
    for raw in re.findall(r'\]\(([^)]+)\)', (ROOT/path).read_text(encoding='utf-8')):
        url=raw.strip().strip('<>')
        if not url or url.startswith('#') or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',url):
            continue
        relative=unquote(urlsplit(url).path)
        if relative:
            target=(ROOT/path).parent/relative
            links.append({'source':path,'target':raw,'exists':target.exists()})
missing=[item for item in links if not item['exists']]
assert not missing, missing
c=ROOT/'Colab Notebooks/cardiac_4d_segmentation'
frozen={
    '21.03_Вычитание_крови_и_объёмы_камер_RR.ipynb':'d15ba0988f676e5e086c8896908ce399208ab22c9dea0ec9158e5e1dbb2339cb',
    '21.03_Вычитание_крови_и_объёмы_камер_RR.html':'c53233da6c7f9908c0484ce74ae0a5bf9cd1454e1ba76845c1cf35f944e97337',
}
for name,expected in frozen.items():
    assert hashlib.sha256((c/name).read_bytes()).hexdigest()==expected,name
(ARC/'documentation_checks.json').write_text(json.dumps({'local_links_checked':len(links),'missing_links':missing,'notebook_html_unchanged':True,'frozen_artifacts_sha256':frozen,'documentation_hashes':hashes},ensure_ascii=False,indent=2),encoding='utf-8')
print('Links checked:',len(links),'; missing:',len(missing),'; notebook/HTML unchanged; index untouched.')
