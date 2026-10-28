"""Add the reviewed report to frozen data volumes and verify the final local ZIP."""
import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

REV = Path(__file__).resolve().parents[1]
REPO = REV.parents[3]
TIERS = ['core', 'optional_ct4d', 'optional_recompute', 'researchstate']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--delivery', type=Path, required=True)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--docx', type=Path, required=True)
    p.add_argument('--pdf', type=Path, required=True)
    p.add_argument('--outer', type=Path, required=True)
    args = p.parse_args()
    supplement = args.delivery / 'report_and_sources.zip'
    if supplement.exists() or args.outer.exists():
        raise FileExistsError('Use a fresh delivery destination for a new revision')
    build = load(REV / 'build_manifest.json')
    if build['partial_build']:
        raise ValueError('A partial report cannot be delivered')
    for entry in build['sources']:
        if sha(REV / entry['file']) != entry['sha256']:
            raise ValueError('Source changed after Word build: ' + entry['file'])
    frozen, volumes = {}, []
    for tier in TIERS:
        receipt = load(args.delivery / f'{tier}.verification.json')
        file = args.delivery / f'{tier}.zip'
        if receipt['status'] != 'verified' or sha(file) != receipt['sha256']:
            raise ValueError('Invalid frozen volume: ' + tier)
        volumes.append({'tier': tier, 'file': file.name, 'sha256': receipt['sha256'],
                        'bytes': file.stat().st_size, 'payload_files': receipt['payload_files']})
        with zipfile.ZipFile(file) as z:
            for entry in json.loads(z.read('ANNOTATED_MANIFEST.json'))['entries']:
                frozen[entry['archive_path']] = entry['sha256']
    files = {}
    def add(path, name, purpose):
        path = path.resolve()
        if not path.is_file(): raise FileNotFoundError(path)
        files[name] = (path, purpose)
    add(args.docx, 'Отчёт_ГПХ_РНЦХ.docx', 'Основной редактируемый отчёт; формулы OMML, обновлённое оглавление')
    add(args.pdf, 'Отчёт_ГПХ_РНЦХ.pdf', 'Отрисованная Word версия для контроля пагинации')
    assets = {x['file']: x['description'] for x in build['figures']}
    for item in build['figures']:
        source = REPO / item['source']
        if source.is_relative_to(REV):
            assets[source.relative_to(REV).as_posix()] = item['description']
    for file in REV.rglob('*'):
        if not file.is_file(): continue
        relative = file.relative_to(REV).as_posix()
        if '__pycache__' in file.parts or file.suffix in {'.pyc', '.docx', '.pdf'}:
            continue
        if relative.startswith('figures/') and relative not in assets:
            continue
        purpose = assets.get(relative)
        if purpose is None:
            if relative.startswith('chapters/'): purpose = 'Научный текст главы; ссылки, формулы и условия результатов'
            elif relative.startswith('reviews/'): purpose = 'Адресная рецензия: охват проверки, источники и ограничения'
            elif relative.startswith('tools/'): purpose = 'Средство подготовки, проверки или воспроизведения оформления; не физический решатель'
            else: purpose = 'План, правила, сводный текст или паспорт окончательной редакции'
        add(file, 'workspace/' + file.relative_to(REPO).as_posix(), purpose)
    # Every direct local reference in the reader document must resolve in the
    # merged package. A differing current version is explicitly retained last.
    links = re.findall(r'\]\(<(workspace/[^>]+)>\)', (REV/'report_for_word.md').read_text(encoding='utf-8'))
    link_audit = []
    for link in sorted(set(links)):
        path = REPO / unquote(link.removeprefix('workspace/'))
        if path.is_dir():
            link_audit.append({'path': link, 'status': 'directory_reference'})
            continue
        value = sha(path)
        state = 'in_frozen_volume' if frozen.get(link) == value else 'included_in_report_supplement'
        if state != 'in_frozen_volume': add(path, link, 'Прямой локальный источник утверждения или таблицы отчёта; сохранённая версия')
        link_audit.append({'path': link, 'status': state, 'sha256': value})
    # Retain the exact arrays and plotting sources identified by the figure
    # authors, even when they are not direct hyperlinks in the reader text.
    figure_dependencies = {}
    def collect_dependencies(value):
        if isinstance(value, dict):
            path = value.get('path')
            if value.get('path_from_project'):
                path = (REV.parents[1] / value['path_from_project']).relative_to(REPO).as_posix()
            if path and value.get('sha256'):
                figure_dependencies[path] = value['sha256']
            for item in value.values(): collect_dependencies(item)
        elif isinstance(value, list):
            for item in value: collect_dependencies(item)
    for name in ['figure_edits.json', 'motion_figure_edits.json', 'chamber_figure_edit.json']:
        collect_dependencies(load(REV / 'reviews' / name))
    dependency_audit = []
    for relative, expected in sorted(figure_dependencies.items()):
        name = 'workspace/' + relative
        file = (REPO / relative).resolve()
        if not file.is_relative_to(REPO): raise ValueError(relative)
        if frozen.get(name) == expected and name not in files:
            state = 'in_frozen_volume'
        else:
            if sha(file) != expected:
                raise ValueError('Figure dependency changed: ' + relative)
            add(file, name, 'Точный источник данных или программы редакционного рисунка; контрольная сумма из паспорта рисунка')
            state = 'included_in_report_supplement'
        dependency_audit.append({'path': name, 'sha256': expected, 'status': state})
    add(args.inventory, 'manifests/inventory/archive_inventory.json', 'Полная частная аннотированная инвентаризация данных и зависимостей')
    notes = args.inventory.with_name('archive_inventory_notes.md')
    if notes.exists(): add(notes, 'manifests/inventory/archive_inventory_notes.md', 'Протокол отбора, отсутствующие источники и ограничения воспроизводимости')
    entries = []
    for name, (file, purpose) in sorted(files.items()):
        entries.append({'archive_path': name, 'source_path': str(file), 'sha256': sha(file),
                        'size_bytes': file.stat().st_size, 'purpose': purpose,
                        'previous_frozen_sha256': frozen.get(name),
                        'scientific_status': 'report_revision_not_clinical_validation'})
    manifest = {'schema': 'gph_report_supplement_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'entries': entries, 'local_reference_audit': link_audit,
                'figure_dependency_audit': dependency_audit,
                'privacy': 'local delivery; primary data and private manifests are not published to Git'}
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=['archive_path','size_bytes','sha256','purpose','scientific_status'])
    writer.writeheader()
    writer.writerows({k:e[k] for k in writer.fieldnames} for e in entries)
    with zipfile.ZipFile(supplement, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as z:
        z.writestr('ANNOTATED_MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        z.writestr('FILES_ANNOTATED.csv', '\ufeff'+stream.getvalue())
        z.write(REV/'DELIVERY_README.md', 'README.md')
        for e in entries: z.write(files[e['archive_path']][0], e['archive_path'])
    with zipfile.ZipFile(supplement) as z:
        if z.testzip() is not None: raise ValueError('Supplement CRC failed')
        for e in entries:
            h = hashlib.sha256()
            with z.open(e['archive_path']) as f:
                for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
            if h.hexdigest() != e['sha256']: raise ValueError('Supplement content changed: '+e['archive_path'])
    volumes.append({'tier':'report_and_sources','file':supplement.name,'sha256':sha(supplement),
                    'bytes':supplement.stat().st_size,'payload_files':len(entries)})
    final = {'schema':'gph_delivery_v1', 'created_utc':datetime.now(timezone.utc).isoformat(),
             'volumes':volumes,'local_reference_count':len(link_audit),
             'primary_data_published':False, 'research_recalculated':False}
    (args.delivery/'DELIVERY_MANIFEST.json').write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copy2(REV/'DELIVERY_README.md',args.delivery/'README.md')
    shutil.copy2(REV/'tools/extract_delivery.py',args.delivery/'extract_delivery.py')
    args.outer.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.outer,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for v in volumes: z.write(args.delivery/v['file'],v['file'])
        for name in ['README.md','extract_delivery.py','DELIVERY_MANIFEST.json']:
            z.write(args.delivery/name,name)
    with zipfile.ZipFile(args.outer) as z:
        if z.testzip() is not None: raise ValueError('Outer ZIP CRC failed')
    receipt = {'status':'verified','file':args.outer.name,'bytes':args.outer.stat().st_size,
               'sha256':sha(args.outer),'volumes':len(volumes),'supplement_payload_files':len(entries),
               'local_reference_count':len(link_audit),'crc_verified':True,
               'supplement_payload_sha256_verified':True}
    args.outer.with_suffix('.verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__ == '__main__': main()
