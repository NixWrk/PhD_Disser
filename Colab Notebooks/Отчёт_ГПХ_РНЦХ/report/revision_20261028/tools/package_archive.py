"""Stream the selected, annotated research snapshot to local ZIP64 volumes.

Does not run research code or modify source data. Every payload is hashed on
write and again through the ZIP reader. Changed entries are explicitly excluded.
The input inventory is private and must never be committed with medical data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import time
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

CHUNK = 4 * 1024 * 1024


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(CHUNK), b''):
            h.update(block)
    return h.hexdigest()


def safe_name(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
        raise ValueError(f'Unsafe archive path: {name}')
    return p.as_posix()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--tier', required=True,
                        choices=['core', 'optional_ct4d', 'optional_recompute', 'researchstate'])
    args = parser.parse_args()
    data = json.loads(args.inventory.read_text(encoding='utf-8'))
    entries = [e for e in data['entries'] if e['delivery_tier'] == args.tier]
    args.outdir.mkdir(parents=True, exist_ok=True)
    dest = args.outdir / (args.tier + '.zip')
    temporary = dest.with_suffix('.zip.partial')
    if dest.exists():
        raise FileExistsError(dest)
    included, excluded, names = [], [], set()
    started = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as archive:
        for i, e in enumerate(entries):
            source = Path(e['source_path'])
            name = safe_name(e['archive_path'])
            if name in names:
                raise ValueError('Duplicate destination: ' + name)
            names.add(name)
            try:
                before = source.stat()
            except OSError as exc:
                excluded.append({'entry': e, 'reason': str(exc)})
                continue
            if before.st_size != e['size_bytes'] or before.st_mtime_ns != e['source_mtime_ns']:
                # Metadata changes alone are harmless if all bytes still match.
                if digest(source) != e['sha256']:
                    excluded.append({'entry': e, 'reason': 'source_changed_since_inventory'})
                    continue
                before = source.stat()
            info = zipfile.ZipInfo(name, time.localtime(before.st_mtime)[:6])
            info.compress_type = (zipfile.ZIP_STORED if source.suffix.lower()
                                  in {'.gz', '.png', '.jpg', '.jpeg', '.zip', '.docx', '.pdf'}
                                  else zipfile.ZIP_DEFLATED)
            info._compresslevel = 1
            h = hashlib.sha256()
            with source.open('rb') as src, archive.open(info, 'w', force_zip64=True) as dst:
                for block in iter(lambda: src.read(CHUNK), b''):
                    h.update(block)
                    dst.write(block)
            after = source.stat()
            if (h.hexdigest() != e['sha256'] or after.st_size != before.st_size
                    or after.st_mtime_ns != before.st_mtime_ns):
                raise RuntimeError('Unstable source; partial archive is not deliverable: ' + str(source))
            included.append(e)
            if i % 100 == 0:
                print(json.dumps({'tier': args.tier, 'written': i + 1, 'total': len(entries)}), flush=True)
        manifest = {
            'schema': 'gph_packaged_snapshot_v1', 'tier': args.tier,
            'started_utc': started, 'inventory_sha256': digest(args.inventory),
            'entries': included, 'excluded_changed_or_unreadable': excluded,
            'missing_sources': data['missing_sources'],
            'held_entries': [e for e in data['entries'] if e['delivery_tier'].startswith('hold_')],
            'limitations': data['limitations'], 'external_dependencies': data['external_dependencies'],
            'commands': data['commands'], 'atomic_snapshot': False,
        }
        archive.writestr('ANNOTATED_MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        rows = io.StringIO(newline='')
        fields = ['archive_path', 'group', 'type', 'purpose', 'units', 'protocol', 'subject', 'sha256', 'size_bytes', 'verified_status']
        writer = csv.DictWriter(rows, fieldnames=fields)
        writer.writeheader()
        for e in included:
            row = {k: e.get(k, '') for k in fields}
            row['verified_status'] = json.dumps(row['verified_status'], ensure_ascii=False)
            writer.writerow(row)
        archive.writestr('FILES_ANNOTATED.csv', '\ufeff' + rows.getvalue())
        readme = f'''# Локальный исследовательский архив ГПХ РНЦХ

Том: {args.tier}. Начало упаковки: {started}. Файлов: {len(included)}.
Архив содержит медицинские данные и предназначен для локальной передачи.

## Состав и чтение

FILES_ANNOTATED.csv — пофайловая аннотация назначения, единиц, протокола,
статуса проверки и SHA-256. ANNOTATED_MANIFEST.json дополнительно сохраняет
происхождение, исторические паспорта, зависимости, команды, пропуски и отказы.
Каждый исходник помещён без редактирования. Аннотации не заменяют научную
проверку и не повышают статус расчёта. Отказы также сохранены как результаты.

Том core содержит основной экспериментальный корпус, сегментации, результаты,
протоколы и код. optional_ct4d дополняет КТ-референс; optional_recompute содержит
крупные производные и сетки. researchstate хранит незавершённую текущую ветку.
Все тома можно извлечь в один пустой каталог: пути согласованы по описи.
Служебные ANNOTATED_MANIFEST.json и FILES_ANNOTATED.csv каждого тома следует
сохранить отдельно перед объединением; они описывают именно свой том.

## Применимость

Сегментация и подготовка FEM — программные процедуры с собственными проверками.
Они не подтверждают клиническую точность УО/ФВ. Ручная сегментация Inobitec,
автоматические маски и исследовательская 4D-динамика имеют разные статусы.
Четырёхмерные целевые объёмы не являются входом операционной задачи одной КТ.
Исторические конфигурации сохранены как свидетельства происхождения; абсолютные
пути требуют переназначения по source_path → archive_path из JSON. Установка
MATLAB/EIDORS/Gmsh, лицензии и модели внешних программ в этот архив не входят.

## Полнота и контроль

Срез пофайловый, не атомарный. Изменившиеся после описи файлы исключены явно:
{len(excluded)}. Неполученные исторические файлы и несовпавшие хеши перечислены
в манифесте; подмена похожим файлом не выполнялась. Точная редакция части
исторических генераторов геометрии не восстановлена. Это ограничивает полный
повтор прежних вычислений даже при наличии современных рабочих скриптов.
При упаковке проверены SHA-256 исходных потоков; после упаковки каждая запись
повторно прочитана через ZIP с проверкой CRC и SHA-256. Внешний файл
{args.tier}.verification.json фиксирует результат этой проверки.
'''
        archive.writestr('README.md', readme)
    with zipfile.ZipFile(temporary) as archive:
        for i, e in enumerate(included):
            h = hashlib.sha256()
            with archive.open(e['archive_path']) as src:
                for block in iter(lambda: src.read(CHUNK), b''):
                    h.update(block)
            if h.hexdigest() != e['sha256']:
                raise RuntimeError('Archive hash mismatch: ' + e['archive_path'])
            if i % 500 == 0:
                print(json.dumps({'tier': args.tier, 'verified': i + 1, 'total': len(included)}), flush=True)
        for name in ['README.md', 'ANNOTATED_MANIFEST.json', 'FILES_ANNOTATED.csv']:
            archive.read(name)
    os.replace(temporary, dest)
    result = {'status': 'verified', 'archive': dest.name, 'sha256': digest(dest),
              'archive_bytes': dest.stat().st_size, 'payload_files': len(included),
              'payload_bytes': sum(e['size_bytes'] for e in included),
              'excluded_count': len(excluded), 'excluded': excluded,
              'groups': dict(Counter(e['group'] for e in included)),
              'crc_and_payload_sha256_verified': True}
    dest.with_suffix('.verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ['excluded','groups']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
