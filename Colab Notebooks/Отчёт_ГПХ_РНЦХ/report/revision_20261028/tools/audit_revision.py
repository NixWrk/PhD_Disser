"""Structural checks for the report sources; does not validate the science."""
from __future__ import annotations
import hashlib
import argparse
import json
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument('--availability-manifest', type=Path)
args = ap.parse_args()
availability = json.loads(args.availability_manifest.read_text(encoding='utf-8-sig')) if args.availability_manifest else {}
declared = {(x['chapter'], x['target']) for x in availability.get('unavailable_references', [])}
cached = {x['source']: x for x in availability.get('figures', []) if x.get('verified_previous_build_cache')}
unavailable, cached_used = [], []
rows, errors = [], []
numbers = {}
for path in sorted((ROOT / 'chapters').glob('[0-9][0-9]_*.md')):
    text = path.read_text(encoding='utf-8-sig')
    tags = re.findall(r'\\tag\{([^}]+)\}', text)
    for tag in tags:
        if tag in numbers:
            errors.append(f'Duplicate equation {tag}: {path.name}, {numbers[tag]}')
        numbers[tag] = path.name
    missing = []
    for target in re.findall(r'\]\(([^)]+)\)', text):
        if re.match(r'(https?://|#|mailto:)', target):
            continue
        file = (path.parent / unquote(target.strip('<>').split('#')[0])).resolve()
        if not file.exists():
            if (path.name, target) in declared:
                unavailable.append({'chapter': path.name, 'target': target})
                continue
            relative = file.relative_to(ROOT.parents[3]).as_posix() if file.is_relative_to(ROOT.parents[3]) else ''
            entry = cached.get(relative)
            if entry and (ROOT / entry['file']).is_file() and hashlib.sha256((ROOT / entry['file']).read_bytes()).hexdigest() == entry['sha256']:
                cached_used.append({'chapter': path.name, 'target': target, 'cache': entry['file']})
                continue
            missing.append(target)
    errors.extend(f'{path.name}: missing {x}' for x in missing)
    if text.count('$$') % 2:
        errors.append(f'{path.name}: odd math delimiters')
    if text.count('<details>') != text.count('</details>'):
        errors.append(f'{path.name}: mismatched details blocks')
    if re.search(r'(?i)TODO|TBD|здесь вставить|текст будет добавлен', text):
        errors.append(f'{path.name}: editorial placeholder')
    if re.search(r'[CDZ]:[\\/]', text):
        errors.append(f'{path.name}: absolute local path')
    rows.append({'file': path.name, 'bytes': path.stat().st_size,
                 'words': len(text.split()), 'equations': len(tags),
                 'figures': len(re.findall(r'^!\[', text, re.M)),
                 'tables': len(re.findall(r'^\|\s*:?-+', text, re.M)),
                 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
result = {'chapters': rows, 'errors': errors,
          'unavailable_historical_references': unavailable, 'verified_cached_figures': cached_used,
          'scope': 'Структура, формулы по номерам, локальные ссылки и отсутствие заглушек; не научная валидация'}
(ROOT / 'structure_check.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'chapters': len(rows), 'equations': len(numbers), 'errors': errors,
                  'unavailable_historical_references': len(unavailable), 'verified_cached_figures': len(cached_used)}, ensure_ascii=False))
raise SystemExit(bool(errors))
