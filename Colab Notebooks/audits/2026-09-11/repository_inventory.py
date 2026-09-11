"""Read-only inventory of a research workspace; does not execute notebooks."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SKIP = {'.git', '.git-recovery', 'venv', '.venv', '.venv-registration',
        '__pycache__', 'node_modules', '.pytest_cache', '.mypy_cache', '.ruff_cache'}

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)

def paths(*args):
    return [p.decode('utf-8') for p in git(*args).split(b'\0') if p]

def main():
    tracked = set(paths('ls-files', '-z'))
    changed = set(paths('diff', '--name-only', '-z', 'HEAD'))
    untracked = set(paths('ls-files', '--others', '--exclude-standard', '-z'))
    staged = set(paths('diff', '--cached', '--name-only', '-z'))
    files, notebooks, duplicates = [], [], defaultdict(list)
    errors = []
    for base, dirs, names in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in SKIP and not d.startswith('.venv') and d != 'edge_profile')
        for name in sorted(names):
            p = Path(base) / name
            rel = p.relative_to(ROOT).as_posix()
            try:
                row = dict(path=rel, size=p.stat().st_size, tracked=rel in tracked,
                           changed=rel in changed, staged=rel in staged,
                           untracked_visible=rel in untracked)
                files.append(row)
                if p.suffix != '.ipynb':
                    continue
                nb = json.loads(p.read_text(encoding='utf-8-sig'))
                cells = nb.get('cells', [])
                source = '\n'.join(c.get('cell_type', '') + '\n' + ''.join(c.get('source', [])) for c in cells)
                source_hash = hashlib.sha256(source.encode()).hexdigest()
                duplicates[source_hash].append(rel)
                markdown = '\n'.join(''.join(c.get('source', [])) for c in cells if c.get('cell_type') == 'markdown')
                code = '\n'.join(''.join(c.get('source', [])) for c in cells if c.get('cell_type') == 'code')
                outputs = [o for c in cells for o in c.get('outputs', [])]
                notebooks.append(dict(**row, cells=len(cells),
                    markdown_cells=sum(c.get('cell_type') == 'markdown' for c in cells),
                    code_cells=sum(c.get('cell_type') == 'code' for c in cells),
                    markdown_chars=len(markdown), code_chars=len(code),
                    output_count=len(outputs), output_types=dict(Counter(o.get('output_type') for o in outputs)),
                    error_names=[o.get('ename') for o in outputs if o.get('output_type') == 'error'],
                    null_execution_counts=sum(c.get('execution_count') is None for c in cells if c.get('cell_type') == 'code'),
                    source_sha256=source_hash,
                    headings=[s.strip() for s in markdown.splitlines() if s.startswith('#')][:100]))
            except (OSError, ValueError) as exc:
                errors.append(dict(path=rel, error=type(exc).__name__))
    result = dict(head=git('rev-parse', 'HEAD').decode().strip(),
                  excluded_directory_names=sorted(SKIP),
                  files=files, notebooks=notebooks,
                  identical_notebook_sources=[v for v in duplicates.values() if len(v)>1], errors=errors)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'inventory.local.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    top = Counter(r['path'].split('/')[0] for r in files)
    print(json.dumps(dict(files=len(files), notebooks=len(notebooks), tracked=len(tracked),
        changed=len(changed), staged=len(staged), untracked=len(untracked),
        top_level=dict(top), notebook_source_duplicate_groups=len(result['identical_notebook_sources']),
        errors=errors), ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
