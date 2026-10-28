"""Structural checks for the report sources; does not validate the science."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
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
          'scope': 'Структура, формулы по номерам, локальные ссылки и отсутствие заглушек; не научная валидация'}
(ROOT / 'structure_check.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'chapters': len(rows), 'equations': len(numbers), 'errors': errors}, ensure_ascii=False))
raise SystemExit(bool(errors))
