"""Write a separate JSON configuration with paths mapped to an extracted archive.

Only exact known paths and directories containing selected files are mapped.
Unresolved absolute paths are retained and reported, never silently guessed.
"""
import argparse
import json
import re
from pathlib import Path


def normalized(value):
    return value.replace('\\', '/').rstrip('/').casefold()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, action='append', required=True)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists() or args.output.resolve() == args.input.resolve():
        raise FileExistsError('Choose a new output file; originals are preserved')
    paths = {}
    for manifest in args.manifest:
        data = json.loads(manifest.read_text(encoding='utf-8-sig'))
        for e in data['entries']:
            old, new = e['source_path'].replace('\\', '/'), e['archive_path']
            paths[normalized(old)] = new
            # A shared prefix is mapped only when its mapping is unambiguous.
            while '/' in new and '/' in old and not re.fullmatch(r'[A-Za-z]:', old):
                old, new = old.rsplit('/', 1)[0], new.rsplit('/', 1)[0]
                key = normalized(old)
                if key in paths and paths[key] != new:
                    paths[key] = None
                elif key not in paths:
                    paths[key] = new
    changed, unresolved = [], []
    def visit(value, key='$'):
        if isinstance(value, dict):
            return {k: visit(v, key + '.' + k) for k, v in value.items()}
        if isinstance(value, list):
            return [visit(v, f'{key}[{i}]') for i, v in enumerate(value)]
        if isinstance(value, str) and re.match(r'^(?:[A-Za-z]:[/\\]|/)', value):
            target = paths.get(normalized(value))
            if target:
                result = str((args.root / target).resolve())
                changed.append({'key': key, 'source': value, 'target': result,
                                'exists': Path(result).exists()})
                return result
            unresolved.append({'key': key, 'value': value})
        return value
    result = visit(json.loads(args.input.read_text(encoding='utf-8-sig')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    log = {'input': str(args.input), 'output': str(args.output), 'changed': changed,
           'unresolved_absolute_paths': unresolved, 'research_executed': False,
           'status': 'needs_review' if unresolved or any(not x['exists'] for x in changed) else 'paths_resolved'}
    args.output.with_suffix('.relocation.json').write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'changed': len(changed), 'unresolved': len(unresolved), 'status': log['status']}))


if __name__ == '__main__':
    main()
