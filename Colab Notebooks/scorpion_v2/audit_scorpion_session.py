"""Inventory an imported Scorpion session without assigning physical channels.

Input: ignored metadata/archive_inventory.local.json and immutable raw files.
Output: ignored inventory and native-unit numerical file QC. Filename-derived
montages remain distinct from electrical channel identities and annotations.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


SUBJECTS = {'Ник': ('nik', 'N'), 'Ярослав': ('yaroslav', 'Y')}
COLUMNS = ['COUNT', 'EMG1mV', 'EMG2mV', 'BEI1ohm_100kHz', 'BEI2ohm_100kHz',
           'BPH1grad_100kHz', 'BPH2grad_100kHz', 'CE1ohm_100kHz', 'CE2ohm_100kHz',
           'CPH1grad_100kHz', 'CPH2grad_100kHz']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf-8')


def load_txt(path):
    # Data rows have a trailing twelfth empty field; do not infer a row index.
    frame = pd.read_csv(path, sep='\t', decimal=',', usecols=range(11),
                        index_col=False, encoding='utf-8-sig')
    if frame.columns.tolist() != COLUMNS:
        raise ValueError('Unexpected columns: ' + Path(path).name)
    if frame.empty or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError('Empty or nonfinite signal: ' + Path(path).name)
    counter = frame.COUNT.to_numpy(float)
    if np.any(counter != np.floor(counter)) or np.any((counter < 0) | (counter > 255)):
        raise ValueError('COUNT is not a byte counter')
    return frame


def name_context(name):
    name = name.lower()
    kind = ('reciprocity' if 'взаимност' in name else 'switching' if 'попеременн' in name
            else 'ttrkg_only' if 'только' in name and 'ттркг' in name
            else 'lateral_only' if 'только' in name else 'both')
    size = re.search(r'\b(60|100|140)\b', name)
    variant = re.search(r'([12])[хx×]([12])', name)
    return dict(record_kind=kind, lateral_size_mm=int(size[0]) if size else None,
                ttrkg_variant=variant[1] + 'x' + variant[2] if variant else None,
                source='Имя файла; смысл сборок подтверждён автором 23.10.2026',
                physical_channel_roles=None)


def file_qc(frame):
    counter = frame.COUNT.to_numpy(dtype=int)
    increments = np.diff(counter) % 256
    fields = {}
    for column in COLUMNS[1:]:
        values = frame[column].to_numpy(float)
        fields[column] = dict(minimum=float(values.min()), maximum=float(values.max()),
            median=float(np.median(values)), q01=float(np.quantile(values, .01)),
            q25=float(np.quantile(values, .25)), q75=float(np.quantile(values, .75)),
            q99=float(np.quantile(values, .99)), standard_deviation=float(values.std()),
            distinct_values=int(np.unique(values).size))
    return dict(rows=len(frame), columns=COLUMNS, all_fields_finite=True,
                count_first=int(counter[0]), count_last=int(counter[-1]),
                count_nonunit_steps=int(np.count_nonzero(increments != 1)),
                count_step_histogram={str(int(v)): int(n) for v, n in
                                      zip(*np.unique(increments, return_counts=True))},
                count_is_not_absolute_time=True, fields=fields,
                physiological_signal_quality_accepted=False)


def identity_by_source(identity, imported):
    """Frozen content identities; filesystem times must never rename a record."""
    if identity['source_archive_sha256'] != imported['archive_sha256']:
        raise ValueError('Record identities refer to a different archive')
    records=identity['records']
    by_path={r['raw_relative_path']:r for r in records}
    if len(by_path)!=len(records) or len({r['record_id'] for r in records})!=len(records):
        raise ValueError('Duplicate frozen record identity')
    actual={'raw/'+r['relative_path']:r['sha256'] for r in imported['files']
            if Path(r['relative_path']).suffix.lower()=='.txt'}
    if set(actual)!=set(by_path):raise ValueError('Frozen TXT inventory differs from import')
    if any(by_path[p]['sha256']!=digest for p,digest in actual.items()):
        raise ValueError('Frozen identity refers to another signal')
    return by_path


def run(experiment):
    experiment = Path(experiment).resolve()
    archive_manifest = experiment / 'metadata/archive_inventory.local.json'
    original = read(archive_manifest)
    identity_path=experiment/'record_identity.json'
    identities=identity_by_source(read(identity_path),original)
    grouped = defaultdict(list)
    binaries = []
    for item in original['files']:
        path = experiment / 'raw' / item['relative_path']
        if sha(path) != item['sha256']:
            raise ValueError('Imported raw hash changed: ' + path.name)
        if path.suffix.lower() == '.bin':
            binaries.append((item, path))
        elif path.suffix.lower() == '.txt':
            subject = next((part for part in path.parts if part in SUBJECTS), None)
            if subject is None:
                raise ValueError('Subject is not established by import context')
            grouped[subject].append((item, path))
    rows, files, hashes = [], [], defaultdict(list)
    for subject in SUBJECTS:
        ordered = sorted(grouped[subject], key=lambda pair:
                         identities[pair[1].relative_to(experiment).as_posix()]['record_id'])
        for number, (item, path) in enumerate(ordered, 1):
            alias, code = SUBJECTS[subject]
            frozen=identities[path.relative_to(experiment).as_posix()]
            if frozen['subject_id']!=alias:raise ValueError('Frozen participant mismatch')
            rid = frozen['record_id']
            frame = load_txt(path); qc = file_qc(frame); context = name_context(path.name)
            entry = dict(record_id=rid, subject_id=alias, source_name=path.name,
                raw_relative_path=path.relative_to(experiment).as_posix(),
                sha256=item['sha256'], bytes=item['bytes'], signal_input_candidate=True,
                export_mtime_utc=frozen['export_mtime_utc'],
                ordering_basis='Закреплённые путь и SHA-256; время архива сохранено только как метаданные',
                text_qc=qc, montage_context=context)
            files.append(entry); hashes[item['sha256']].append(rid)
            rows.append(dict(record_id=rid, subject=subject, file=path.name, rows=len(frame),
                nominal_duration_at_1000hz_s=(len(frame) - 1) / 1000,
                counter_nonunit_steps=qc['count_nonunit_steps'],
                BEI1_median_ohm=qc['fields']['BEI1ohm_100kHz']['median'],
                BEI2_median_ohm=qc['fields']['BEI2ohm_100kHz']['median']))
    for number, (item, path) in enumerate(sorted(binaries, key=lambda pair: pair[1].name), 1):
        files.append(dict(record_id=f'SCV2-BIN-{number:03d}', source_name=path.name,
            raw_relative_path=path.relative_to(experiment).as_posix(),
            bytes=item['bytes'], sha256=item['sha256'], signal_input_candidate=False,
            subject_assignment='Принадлежность участнику не выводится из имени архивной папки Георгия'))
    duplicates = [ids for ids in hashes.values() if len(ids) > 1]
    result = dict(schema='scorpion_multisubject_inventory_v1',
        status='file_inventory_not_physiological_acceptance', import_date='2026-10-23',
        session_date_from_directory='2026-10-22', source_archive_sha256=original['archive_sha256'],
        import_manifest_sha256=sha(archive_manifest), producer_sha256=sha(__file__),
        frozen_record_identity_sha256=sha(identity_path),
        probing_frequency_hz_in_headers=100000, sampling_frequency_confirmed_from_header=False,
        files=files, byte_identical_record_groups=duplicates)
    write(experiment / 'metadata/inventory.local.json', result)
    out = experiment / 'derived/session_audit'; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / 'native_file_qc.local.csv', index=False, encoding='utf-8-sig')
    write(out / 'contract.local.json', dict(producer_sha256=sha(__file__),
        inventory_sha256=sha(experiment / 'metadata/inventory.local.json'),
        physiological_signal_quality_accepted=False, annotations_created=False))
    print(pd.DataFrame(rows).to_string(index=False))
    print('Byte-identical groups:', duplicates)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, type=Path)
    run(parser.parse_args().experiment)
