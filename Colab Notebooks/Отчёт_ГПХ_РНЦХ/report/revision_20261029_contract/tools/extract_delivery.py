"""Extract selected verified volumes into a NEW directory without losing versions."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_path(root, name):
    rel = PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts or '\\' in name or ':' in name:
        raise ValueError(f'Unsafe archive path: {name}')
    target = root.joinpath(*rel.parts).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(name)
    return target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archives', type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--with-ct4d', action='store_true')
    ap.add_argument('--with-recompute', action='store_true')
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a NEW output directory; existing work is never overwritten')
    manifest = json.loads((args.archives / 'DELIVERY_MANIFEST.json').read_text(encoding='utf-8'))
    selected = ['core']
    if args.with_ct4d: selected.append('optional_ct4d')
    if args.with_recompute: selected.append('optional_recompute')
    selected += ['researchstate', 'report_and_sources']
    receipts = {r['tier']: r for r in manifest['volumes']}
    for tier in selected:
        file = args.archives / receipts[tier]['file']
        if digest(file) != receipts[tier]['sha256']:
            raise ValueError(f'ZIP checksum mismatch: {file.name}')
    args.output.mkdir(parents=True)
    records, preserved = [], []
    controls = {'README.md', 'FILES_ANNOTATED.csv', 'ANNOTATED_MANIFEST.json'}
    for tier in selected:
        count = 0
        with zipfile.ZipFile(args.archives / receipts[tier]['file']) as archive:
            for item in archive.infolist():
                if item.is_dir(): continue
                name = item.filename
                if name in controls:
                    name = f'manifests/{tier}/{name}'
                target = safe_path(args.output, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    # Keep both historical and report-time versions. The latter
                    # is deliberately extracted last and its provenance is logged.
                    old_hash = digest(target)
                    saved = safe_path(args.output, f'versions/before_{tier}/{name}')
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    target.rename(saved)
                    preserved.append({'path': name, 'previous_sha256': old_hash,
                                      'preserved_as': saved.relative_to(args.output).as_posix()})
                with archive.open(item) as source, target.open('xb') as output:
                    shutil.copyfileobj(source, output, 8 * 1024 * 1024)
                count += 1
        records.append({'tier': tier, 'extracted_files': count})
    shutil.copy2(args.archives / 'README.md', args.output / 'README.md')
    result = {'status': 'extracted_verified_zip_crc', 'volumes': records,
              'historical_versions_preserved': preserved, 'research_executed': False}
    (args.output / 'EXTRACTION_REPORT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': result['status'], 'volumes': len(records), 'preserved': len(preserved)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
