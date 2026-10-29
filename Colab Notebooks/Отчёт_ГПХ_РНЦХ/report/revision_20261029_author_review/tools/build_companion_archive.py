#!/usr/bin/env python3
"""Build the bounded local companion, without executing research programs.

Only explicitly selected text, report tables and manifest-bound figures enter
the ZIP. Original documents, mixed protocols, notebook outputs and external
medical inputs are never recursively copied. Python standard library only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import zipfile
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath

BASE = Path(__file__).resolve().parents[1]
REPO = next(p for p in BASE.parents if (p / '.git').exists())
BASE_REL = BASE.relative_to(REPO).as_posix()
SCORPION = 'Colab Notebooks/scorpion_v2/'
ARCHIVE_NAME = 'Материалы_научной_редакции_20261029.zip'
LIMIT = 150 * 1024 * 1024
EXCLUDED_FIGURES = {'photo_lateral_detail', 'photo_tt_wide', 'photo_belt'}
FIGURE_KEYS = {
    'materials_sequence', 'ct_segmentation', 'ct_model_preparation',
    'materials_electrodes', 'models_layered_geometry',
    'models_ct_fem_preparation', 'processing_cycles',
    'models_anatomy_contacts', 'models_geometry_comparison',
    'models_step_quadrature', 'models_electrical_factors',
    'models_finite_compensation', 'tissue_eit_baseline',
    'tissue_pulse_projection', 'tissue_geometry_mixing',
    'cardiac_phase_reversal',
}
TABLE_KEYS = (
    'tissue_lateral_baseline', 'cardiac_anatomical_transfer',
    'cardiac_electrical_errors', 'am_static_forms', 'am_volume_quadrature',
    'am_y_step', 'am_own_materials', 'am_finite_inverse',
    'am_finite_compensation', 'am_volume_transfer', 'am_pulse_projection',
    'am_false_correction', 'am_omission_errors', 'am_isolated_limits',
)
PROGRAMS = {
    'tools/segment_ct_to_stl.py': 'Получение анатомических поверхностей',
    'tools/prepare_ct_fem.py': 'Единая подготовка КТ/FEM',
    SCORPION + 'scorpion_qc.py': 'Фильтры и кандидаты QRS',
    SCORPION + 'session_20261022.py': 'Сессионные окна и ансамбли У1/У3',
    SCORPION + 'lateral_pilot.py': 'Двуслойная боковая модель',
    SCORPION + 'session_two_layer.py': 'Двуслойная проверка У1/У3',
    SCORPION + 'scorpion_measured_pulse.py': 'Диагностика готовых циклов',
    SCORPION + 'inspect_scorpion_bin.py': 'Чтение структуры BIN',
    SCORPION + 'match_scorpion_bin.py': 'Связь TXT и BIN',
    SCORPION + 'test_scorpion_qc.py': 'Синтетические проверки сигнального кода',
    SCORPION + 'test_scorpion_bin.py': 'Проверки чтения BIN',
    'MATLAB_TRKG4_real_subjects/run_pipeline.m': 'Каноническая точка входа электрического расчёта',
    'MATLAB_TRKG4_real_subjects/tools/heart_motion_electrical_analysis.py': 'Сохранённые диагностики движения У1',
    'MATLAB_TRKG4_real_subjects/tools/heart_montage_saved_analysis.py': 'Сохранённая монтажная серия У1',
    'MATLAB_TRKG4_real_subjects/tools/heart_montage_figures.py': 'Рисунки монтажной серии У1',
    'Colab Notebooks/cardiac_4d_segmentation/analyze_subtraction_and_rr.py': 'Камерные и R–R-референсы КТ',
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


class TableReader(HTMLParser):
    """Read cell strings only; no numeric conversion or recalculation."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self.row = []
        elif tag in ('td', 'th'):
            self.cell = []
        elif tag == 'br' and self.cell is not None:
            self.cell.append('\n')

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ('td', 'th'):
            self.row.append(''.join(self.cell))
            self.cell = None
        elif tag == 'tr':
            self.rows.append(self.row)
            self.row = None


class Builder:
    def __init__(self, strict_build_hashes=False):
        self.strict_build_hashes = strict_build_hashes
        self.cache = {}
        self.payload = {}
        self.files = []
        self.extractions = []
        self.dependencies = []
        self.figure_index = []
        self.table_index = []
        self.processed_tables = []

    def read(self, relative: str) -> bytes:
        relative = PurePosixPath(relative).as_posix()
        p = (REPO / relative).resolve()
        p.relative_to(REPO)  # Reject symlinks or paths outside the repository.
        if relative not in self.cache:
            self.cache[relative] = p.read_bytes()
        return self.cache[relative]

    def text(self, relative: str) -> str:
        return self.read(relative).decode('utf-8-sig').replace('\r\n', '\n')

    def add(self, name: str, data, kind: str, sources=None, note=None):
        p = PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or '\\' in name:
            raise ValueError(f'Unsafe member: {name}')
        if name in self.payload:
            raise ValueError(f'Duplicate member: {name}')
        if isinstance(data, str):
            data = data.encode('utf-8')
        self.payload[name] = data
        item = {'path': name, 'bytes': len(data), 'sha256': digest(data), 'kind': kind,
                'sources': sources or []}
        if note:
            item['note'] = note
        self.files.append(item)

    def locate(self, source, fragment, transform=None):
        original = self.text(source)
        if original.count(fragment) != 1:
            raise ValueError(f'Fragment not unique in {source}: {fragment[:70]}')
        start = original.index(fragment)
        result = transform(fragment) if transform else fragment
        rec = {
            'source_path': source, 'source_sha256': digest(self.read(source)),
            'start_line': original[:start].count('\n') + 1,
            'end_line': original[:start + len(fragment)].count('\n') + 1,
            'source_fragment_sha256': digest(fragment.encode('utf-8')),
            'output_fragment_sha256': digest(result.encode('utf-8')),
            'transformation': 'scoped_selection' if transform else 'verbatim',
            'text_normalization': 'UTF-8 decoded; BOM removed if present; CRLF to LF before fragment hashing',
            'verification_status': 'extracted',
        }
        self.extractions.append(rec)
        return result, rec

    def section(self, source, heading):
        s = self.text(source)
        marker = '\n' + heading + '\n'
        start = s.index(marker) + 1
        end = s.find('\n## ', start + len(heading))
        return s[start: end if end != -1 else len(s)].strip()

    def description(self, dest, title, fragments, note=''):
        records, parts = [], []
        for source, fragment, transform in fragments:
            result, record = self.locate(source, fragment, transform)
            records.append(record)
            parts.append(result)
        intro = f'# {title}\n\n'
        intro += ('Аннотированное извлечение для ограниченного сопроводительного архива. '
                  'Исходный документ сохранён в репозитории без изменений. '
                  'Это извлечение не является оригиналом полного протокола.\n\n')
        if note:
            intro += note + '\n\n'
        intro += ('Пути источников и номера строк приведены в конце файла. '
                  'Относительные ссылки внутри фрагментов относятся к их исходникам в репозитории.\n\n')
        foot = '\n\n## Источники извлечения\n\n'
        for r in records:
            foot += (f"- `{r['source_path']}`, строки {r['start_line']}–{r['end_line']}; "
                     f"SHA-256 исходника `{r['source_sha256']}`; "
                     f"способ: `{r['transformation']}`.\n")
        self.add(dest, intro + '\n\n'.join(parts) + foot, 'annotated_protocol_extract', records)

    def descriptions(self):
        ct = 'docs/CT_FEM_TOTALSEGMENTATOR_VERIFICATION_20261026.md'
        self.description('materials/02/ct_surface_preparation.md', 'Получение поверхностей',
                         [(ct, self.section(ct, '## Получение поверхностей'), None)])
        ep = 'Experiments/scorpion_v2_georg/PROTOCOL.md'
        tt = self.section(ep, '## ТТРКГ и площадь контактов')
        lateral = self.section(ep, '## Боковые сборки')
        # The source paragraph combines an in-scope channel fact with an
        # out-of-scope sentence. Only that sentence is omitted and disclosed.
        def lateral_scope(s):
            return s.replace(' Номера ТЕПС/АЕПС пока не установлены.', '')
        confirmation = self.text(ep).split('## Подтверждение подключения от 14.10.2026\n\n', 1)[1]
        confirmation = confirmation.split('Снятие проводов само по себе', 1)[0].rstrip()
        self.description('materials/02/electrode_protocol.md', 'Протокол электродных подключений', [
            (ep, confirmation, None), (ep, tt, None), (ep, lateral, lateral_scope)],
            'Область У2 ограничена записями 001–011. Из раздела о боковых сборках '
            'опущено одно предложение об иных монтажах. Исторические слова «в архиве» '
            'внутри фрагментов относятся к исходному экспериментальному набору; '
            'сами физиологические записи в настоящий ZIP не включены.')
        paired = 'Experiments/scorpion_20261022/PROTOCOL.md'
        initial = self.text(paired).split('\n## ', 1)[0].strip()
        self.description('materials/02/paired_recording_protocol.md', 'Протокол парных регистраций У1 и У3',
                         [(paired, initial, lambda s: s.replace(
                             'В новых данных нет 1×4, крестов, ТЕПС, АЕПС, 80 и 120 мм. ', ''))],
                         'Из раннего описания опущено одно предложение с перечислением '
                         'незаписанных монтажей. Утверждение о неизвестной '
                         'фазе источников имеет более позднее уточнение: сводка '
                         '`Colab Notebooks/scorpion_v2/_assets/fem_smart_20261026.summary.json`, '
                         '`author_overlay`, фиксирует подтверждённое 26.10.2026 временное '
                         'разделение источников; относительная фаза к такому режиму неприменима. '
                         'Это уточнение не делает разновременные регистрации синхронными.')
        summary = SCORPION + '_assets/fem_smart_20261026.summary.json'
        overlay = json.loads(self.read(summary))['author_overlay']
        if overlay.get('mode') != 'time_division' or overlay.get('phase_not_applicable') is not True:
            raise ValueError('Author source-mode clarification changed; review description')
        self.add('materials/02/source_mode_clarification.json',
                 json_bytes({'source_path': summary, 'source_sha256': digest(self.read(summary)),
                                  'json_pointer': '/author_overlay', 'value': overlay}),
                 'selected_protocol_metadata', [summary + '#/author_overlay'])
        ts = SCORPION + '10.20_Настройки_и_бинарные_потоки_Scorpion_v2.md'
        block = self.section(ts, '## Рабочая интерпретация временной шкалы')
        def scope_time(s):
            s = s.replace('`007–018`', '`007–011`')
            start = s.index('Для концов `007/018`')
            end = s.index('Время изменения файла', start)
            return s[:start] + s[end:]
        self.description('materials/02/time_scale.md', 'Рабочая интерпретация временной шкалы',
                         [(ts, block, scope_time)],
                         'Область извлечения — записи У2 001–011. Диапазон 007–018 '
                         'сужен до 007–011; техническое сравнение с концом записи 018 '
                         'опущено. Все остальные значения сохранены. Интерпретация периода '
                         'остаётся рабочей; независимо подтверждённая точность часов не заявлена.')

    def report_tables(self, build):
        source = BASE_REL + '/report_for_word.md'
        text = self.text(source)
        labels = {x['key']: x['number'] for x in build['labels'] if x['kind'] == 'tab'}
        for key in TABLE_KEYS:
            n = labels[key]
            found = re.search(rf'^Таблица XSEQtab{n}X — (.+)\n\n((?:\|.*\n)+)', text, re.M)
            if not found:
                raise ValueError(f'Report table {n} not found')
            caption, md = found.groups()
            fragment, record = self.locate(source, found.group(0))
            rows = []
            for line in md.splitlines():
                cells = [v.strip() for v in re.split(r'(?<!\\)\|', line)[1:-1]]
                if cells and all(re.fullmatch(r':?-+:?', c) for c in cells):
                    continue
                rows.append(cells)
            if len({len(row) for row in rows}) != 1:
                raise ValueError(f'Non-rectangular report table {n}')
            dest = f'tables/report/{n:02d}_{key}'
            stream = io.StringIO(newline='')
            csv.writer(stream, lineterminator='\n').writerows(rows)
            self.add(dest + '.csv', stream.getvalue(), 'report_table_transcription', [record],
                     'Строковые ячейки; численные значения не пересчитывались. Источник — отчёт, не первичный CSV.')
            # Include the full enclosing subsection for interpretation, with
            # scientific limits intact, rather than detached numeric cells.
            preceding = list(re.finditer(r'^## .+$', text[:found.start()], re.M))
            start = preceding[-1].start()
            nxt = re.search(r'^#{1,2} ', text[found.end():], re.M)
            end = found.end() + nxt.start() if nxt else len(text)
            context = text[start:end].strip()
            _, context_record = self.locate(source, context)
            note = (f'# {caption}\n\nАннотированное извлечение таблицы {n} и её подраздела '
                    'из собранного научного текста. Это представление авторских результатов, '
                    'а не первичный расчётный файл. Числа не пересчитывались; '
                    'маркеры XREF/XSEQ сохранены как в источнике.\n\n')
            note += (f"Источник: `{source}`, строки {context_record['start_line']}–"
                     f"{context_record['end_line']}; SHA-256 `{digest(self.read(source))}`.\n\n")
            self.add(dest + '.md', note + context + '\n', 'report_table_with_context', [context_record])
            self.table_index.append({'key': key, 'number_at_build': n, 'caption': caption,
                                     'csv': dest + '.csv', 'context': dest + '.md',
                                     'origin': 'report_transcription_not_primary_output',
                                     'source': record, 'rows_without_header': len(rows) - 1})

    def observations(self):
        paths = list((REPO / SCORPION).glob('33.31_*.md'))
        if len(paths) != 1:
            raise ValueError('Ambiguous source for lateral observation table')
        source = paths[0].relative_to(REPO).as_posix()
        s = self.text(source)
        fragment = re.search(r'<details><summary>Таблица 3\..*?</details>', s, re.S).group(0)
        _, record = self.locate(source, fragment)
        parser = TableReader()
        parser.feed(fragment)
        rows = parser.rows
        if len(rows) != 25 or any(not 1 <= int(row[0]) <= 11 for row in rows[1:]):
            raise ValueError('Unexpected lateral observations or out-of-scope record')
        stream = io.StringIO(newline='')
        csv.writer(stream, lineterminator='\n').writerows(rows)
        self.add('tables/lateral_u2/observed_levels_002_011.csv', stream.getvalue(),
                 'processed_observations_extract', [record])
        desc = ('# Боковые наблюдения У2: записи 002–011\n\n'
                'Только таблица 3 из сохранённого отчёта 33.31: 24 уровня, 12 пар '
                'кандидатных дыхательных интервалов. Извлечены строковые ячейки HTML '
                'без численных преобразований. Полный исходный документ остаётся '
                'в репозитории; его электрические проверки в ZIP не перенесены.\n\n'
                'Разметка кандидатная. Запись 001 не входит в эту таблицу. '
                'Боковая проверка не является оценкой сердечной функции.\n\n'
                f"Источник: `{source}`, строка {record['start_line']}; "
                f"SHA-256 `{record['source_sha256']}`.\n\n" + fragment + '\n')
        self.add('tables/lateral_u2/README.md', desc, 'observation_table_context', [record])

    def ct_processed_tables(self):
        base = 'Colab Notebooks/cardiac_4d_segmentation/heart_rr_analysis/'
        subjects = {'nix': 'У1', 'georg': 'У2'}
        for filename in ['chamber_volumes_by_rr.csv', 'chamber_volume_changes_measured_cycles.csv']:
            source = base + filename
            data = self.read(source)
            reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig')))
            columns = reader.fieldnames
            rows = list(reader)
            selected, source_rows = [], []
            for i, row in enumerate(rows, 2):
                if row['subject'] in subjects:
                    row = dict(row)
                    row['subject'] = subjects[row['subject']]
                    selected.append(row)
                    source_rows.append(i)
            expected = 27 if filename == 'chamber_volumes_by_rr.csv' else 108
            if len(selected) != expected:
                raise ValueError(f'Unexpected CT processed-table selection: {filename}')
            stream = io.StringIO(newline='')
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
            writer.writeheader()
            writer.writerows(selected)
            record = {'source_path': source, 'source_sha256': digest(data),
                      'source_csv_rows_including_header': source_rows,
                      'transformation': 'filter subjects nix/georg; relabel subject to У1/У2; numeric strings unchanged',
                      'verification_status': 'extracted'}
            dest = 'tables/ct_processed/' + filename
            self.add(dest, stream.getvalue(), 'processed_ct_table_subset', [record])
            self.processed_tables.append({'archive_path': dest, 'rows_without_header': len(selected),
                                          'source': record})
        self.add('tables/ct_processed/README.md',
                 '# Обработанные фазовые объёмы КТ\n\n'
                 'Два извлечения из сохранённых обработанных CSV: 27 фазовых строк '
                 'целого сердца и камер, 108 строк изменений четырёх камер на '
                 'доступных интервалах. Отобраны только исходные коды `nix` и '
                 '`georg`, заменённые на У1 и У2. Исторический дополнительный '
                 'участник не включён и не переименован в У3. Остальные строковые '
                 'ячейки, включая численные значения и статусы, сохранены без пересчёта.\n\n'
                 'Это таблицы обработки, а не DICOM или временные ряды ЭКГ. '
                 'Не включены таблицы искусственного замыкания кардиоцикла и '
                 'исходные метаданные DICOM. Неполные R–R остаются неполными. '
                 'Наблюдаемый размах не объявлен валидированным ударным объёмом. '
                 '`heart_minus_blood_ml` обозначает разность масок, а не независимо '
                 'сегментированный миокард. Фазовое сопоставление с импедансными '
                 'данными не означает одновременную регистрацию.\n\n'
                 'Хэши исходников и номера отобранных строк приведены в '
                 '`tables/processed_index.json` и реестре архива.\n', 'processed_table_context')

    def figures(self, build):
        seen = set()
        for f in build['figures']:
            key = f['key']
            if key not in FIGURE_KEYS | EXCLUDED_FIGURES:
                raise ValueError(f'New figure needs explicit scope review: {key}')
            p = Path(f['path'])
            # Current build manifests may use a machine's absolute workspace.
            # Resolve the repository suffix only, never an external data root.
            normalized = str(p).replace('\\', '/')
            marker = BASE_REL + '/'
            if marker not in normalized:
                raise ValueError(f'Figure is outside this edition: {key}')
            relative = marker + normalized.split(marker, 1)[1]
            meta = {'key': key, 'number_at_build': f['number'], 'caption': f['caption'],
                    'repository_path': relative, 'expected_sha256': f['sha256']}
            if key in EXCLUDED_FIGURES:
                meta.update(status='not_packaged', reason='Фотография подготовки: персональное изображение; исходная копия не дублируется.')
            else:
                data = self.read(relative)
                matches = digest(data) == f['sha256']
                if not matches and self.strict_build_hashes:
                    raise ValueError(f'Figure changed since Word build: {key}; rebuild Word manifest first')
                if not data.startswith(b'\x89PNG\r\n\x1a\n'):
                    raise ValueError(f'Expected PNG: {key}')
                name = 'figures/' + relative.split('/figures/', 1)[1]
                self.add(name, data, 'edition_figure', [relative])
                meta.update(status='packaged', archive_path=name, sha256=digest(data),
                            matches_build_manifest=matches)
                if not matches:
                    meta['note'] = ('Включён текущий сохранённый файл этой редакции. '
                                    'Его хэш отличается от ранее собранного Word; '
                                    'после общей сборки требуется повторная упаковка.')
                seen.add(key)
            self.figure_index.append(meta)
        if seen != FIGURE_KEYS:
            raise ValueError(f'Missing approved figure keys: {FIGURE_KEYS - seen}')

    def references(self):
        entries = []
        for p, purpose in PROGRAMS.items():
            data = self.read(p)
            entries.append({'repository_path': p, 'sha256': digest(data), 'bytes': len(data),
                            'purpose': purpose, 'status': 'reference_only',
                            'scope': 'Назначение относится только к разрешённой постановке. Полный код может поддерживать другие ветки; они не включены в научный комплект.'})
        self.add('programs/references.json', json_bytes(entries), 'exact_program_references')
        self.add('programs/README.md',
                 '# Программы\n\nПрограммы представлены точными путями относительно корня '
                 'репозитория и SHA-256 доступных файлов в `references.json`. Сам код '
                 'не скопирован. Это позволяет не переносить смешанные программы и '
                 'исторические ветки вместе с ограниченным научным комплектом. Хэш '
                 'относится к прочитанной рабочей копии, а не обязательно к коммиту Git.\n\n'
                 'Исследовательские программы при упаковке не запускались. Для расчётов '
                 'нужны их исходные зависимости, MATLAB/EIDORS/Gmsh либо Python-пакеты '
                 'соответствующей процедуры и отдельно разрешённые индивидуальные данные. '
                 'Наличие ссылки не означает автономную воспроизводимость.\n', 'readme')

    def document_references(self):
        documents = []
        for p, purpose in [
            ('configs/scorpion_canonical_stl.json', 'Канонический реестр происхождения выбранных анатомических STL'),
            ('docs/CT_FEM_PREPARATION_STANDARD.md', 'Общий стандарт подготовки КТ и FEM-геометрии'),
            ('MATLAB_TRKG4_real_subjects/docs/ELECTRODE_MODEL_STANDARD.md', 'Стандарт конечных контактных площадок'),
        ]:
            data = self.read(p)
            documents.append({'repository_path': p, 'sha256': digest(data), 'bytes': len(data),
                              'purpose': purpose, 'status': 'reference_only'})
        self.add('materials/repository_references.json', json_bytes(documents), 'exact_document_references')

    def missing(self):
        report = 'Colab Notebooks/Отчёт_ГПХ_РНЦХ/'
        items = [
            ('execution_20261026/', 'Пакет обработки исходных сигналов', 'directory'),
            ('pulmovista_check_20261028/PATIENT_RESULT.md', 'Индивидуальный поясной отчёт', 'file'),
            ('pulmovista_check_20261028/local/patient_v3/tissue_parameters_conditional.csv', 'Условные тканевые параметры', 'file'),
            ('pulmovista_check_20261028/local/patient_v3/tt_transfer_operator.csv', 'Оператор переноса в ТТРКГ', 'file'),
            ('pulmovista_check_20261028/local/patient_v3/band_projection_summary.csv', 'Пульсовые проекции', 'file'),
            ('pulmovista_check_20261028/local/composition_pulse_v1/common_metric_comparison.csv', 'Сравнение нормировок', 'file'),
            ('pulmovista_check_20261028/local/shape_electrical_comparison_v1/target_chain_effective_geometry.csv', 'Диагностика геометрического смешения', 'file'),
            ('results/montage_sensitivity_v1/tables/derivatives.csv', 'Первичные таблицы монтажной серии', 'file'),
        ]
        for relative, purpose, kind in items:
            p = report + relative
            present = (REPO / p).is_dir() if kind == 'directory' else (REPO / p).is_file()
            self.dependencies.append({'repository_path': p, 'purpose': purpose,
                                      'status': 'present_not_packaged' if present else 'missing_local_checkout',
                                      'verification': 'existence_only',
                                      'substitute': 'Извлечение таблицы отчёта не заменяет первичный расчётный файл.'})
        self.dependencies.extend([
            {'purpose': 'DICOM, нативные маски, внешние STL и полные физиологические ряды',
             'status': 'excluded_by_scope', 'verification': 'not_accessed',
             'note': 'Внешний диск недоступен по условию задания. Сетевые копии и зеркало не использовались.'},
            {'purpose': 'Поясной внешний решатель', 'status': 'external_reference_not_verified_here',
             'url': 'https://github.com/ILoveBTSM/CompartmentPipeline',
             'recorded_commit': 'a382dbf8fd5f23bba9e69fcf8c6169a1627bc102',
             'source': 'AGENTS.md, PulmoVista как источник тканевых поправок',
             'note': 'Коммит указан по локальной инструкции; внешний checkout не читался и не загружался.'},
        ])

    def build(self):
        # Source maps are inventoried, not copied: they contain excluded scope.
        for p in ['drafts/02_04_sources.md', 'drafts/sidecar03_05_06_sources.md',
                  'drafts/01_07_08_sources.md', 'AUTHOR_CONFIRMATIONS.md']:
            self.read(BASE_REL + '/' + p)
        build = json.loads(self.read(BASE_REL + '/build_manifest.json'))
        self.descriptions()
        self.report_tables(build)
        self.observations()
        self.ct_processed_tables()
        self.figures(build)
        self.references()
        self.document_references()
        self.missing()
        self.add('tables/index.json', json_bytes(self.table_index), 'table_index')
        self.add('tables/processed_index.json', json_bytes(self.processed_tables), 'processed_table_index')
        self.add('figures/index.json', json_bytes(self.figure_index), 'figure_index')
        self.add('provenance/extractions.json', json_bytes(self.extractions), 'extraction_locators')
        self.add('provenance/uncovered_dependencies.json', json_bytes(self.dependencies), 'dependency_register')
        self.add('README.md', self.readme(), 'readme')
        # Hash the builder itself as an input but keep it outside the ZIP.
        self.read(Path(__file__).relative_to(REPO).as_posix())
        source_index = [{'repository_path': p, 'sha256': digest(data), 'bytes': len(data),
                         'copied_whole': any(e['sources'] == [p] for e in self.files)}
                        for p, data in sorted(self.cache.items())]
        self.add('provenance/source_index.json', json_bytes(source_index), 'source_hash_register')
        self.validate_payload()
        manifest = {
            'schema': 'bounded-scientific-companion/1', 'edition': '20261029_author_review',
            'scope': 'Ограниченные сопроводительные материалы научной редакции; не полный договорный архив.',
            'base_repository_path': BASE_REL,
            'source_build_manifest_sha256': digest(self.read(BASE_REL + '/build_manifest.json')),
            'source_report_sha256': digest(self.read(BASE_REL + '/report_for_word.md')),
            'figure_mode': 'with_figures', 'formatting_mode': 'section_only',
            'u2_record_scope': '001–011', 'u2_observation_table_scope': '002–011; lateral only',
            'files': sorted(self.files, key=lambda x: x['path']),
            'counts': {'payload_files': len(self.files), 'scientific_figures': len(FIGURE_KEYS),
                       'omitted_preparation_photos': len(EXCLUDED_FIGURES),
                       'report_tables': len(self.table_index), 'lateral_observation_rows': 24},
            'processed_ct_tables': {'files': 2, 'phase_rows': 27, 'chamber_change_rows': 108,
                                    'scope': 'У1/У2 only; historical extra participant omitted'},
            'figure_hash_mismatches_with_word_build': [f['key'] for f in self.figure_index
                                                     if f.get('matches_build_manifest') is False],
            'uncovered_dependencies': self.dependencies,
            'integrity_convention': 'files excludes archive_manifest.json and CHECKSUMS.sha256 to avoid self-reference; CHECKSUMS includes the internal manifest. The external manifest also hashes the complete ZIP.',
            'scientific_recalculation_performed': False,
            'validation_scope': 'Проверены структура ZIP, CRC, хэши, состав и точность извлечения. Это не научная или физиологическая валидация результатов.',
        }
        internal = json_bytes(manifest)
        self.payload['archive_manifest.json'] = internal
        checksums = ''.join(f'{digest(data)}  {name}\n' for name, data in sorted(self.payload.items()))
        self.payload['CHECKSUMS.sha256'] = checksums.encode('utf-8')
        if sum(map(len, self.payload.values())) > LIMIT:
            raise ValueError('Bounded package exceeds 150 MiB; use a reference register instead')
        # Do not silently combine versions changed by a parallel editor.
        for p, data in self.cache.items():
            if digest((REPO / p).read_bytes()) != digest(data):
                raise ValueError(f'Source changed during packaging; rerun: {p}')
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for name, data in sorted(self.payload.items()):
                info = zipfile.ZipInfo(name, (2026, 10, 29, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                z.writestr(info, data)
        archive = output.getvalue()
        validation = verify(archive)
        manifest['archive'] = {'name': ARCHIVE_NAME, 'bytes': len(archive), 'sha256': digest(archive)}
        manifest['internal_manifest_sha256'] = digest(internal)
        manifest['checksums_sha256'] = digest(self.payload['CHECKSUMS.sha256'])
        manifest['validation'] = validation
        return archive, manifest

    def validate_payload(self):
        for name, data in self.payload.items():
            if PurePosixPath(name).suffix not in {'.md', '.json', '.csv', '.png'}:
                raise ValueError(f'Non-allowlisted extension: {name}')
            if name.endswith('.png'):
                continue
            text = data.decode('utf-8')
            # Source references can mention the exclusion class; there must be
            # no experimental descriptions of the excluded branches in payload.
            if re.search(r'ТЕПС|АЕПС|TEPC|TEPS|AEPC|прекарди', text, re.I):
                raise ValueError(f'Excluded scientific branch in payload: {name}')
            if re.search(r'(?i)(?<![a-z])[a-z]:[\\/]', text):
                raise ValueError(f'Absolute machine path in portable payload: {name}')
            if re.search(r'SCV2-G-0(?:1[2-9]|[2-9][0-9])\b', text):
                raise ValueError(f'Out-of-scope U2 record: {name}')

    def readme(self):
        return '''# Материалы научной редакции 29.10.2026

Ограниченный локальный сопроводительный комплект редакции
`revision_20261029_author_review`. Пакет предназначен для чтения и проверки
происхождения представленных материалов. Это не полный договорный архив.
Наличие материалов не означает завершённой валидации методики.

## Описания, на которые ссылаются главы 2–3

| Раздел в научном тексте | Файл внутри ZIP |
|---|---|
| Получение поверхностей | [ct_surface_preparation.md](materials/02/ct_surface_preparation.md) |
| ТТРКГ и площадь контактов | [electrode_protocol.md](materials/02/electrode_protocol.md) |
| Боковые сборки | [electrode_protocol.md](materials/02/electrode_protocol.md) |
| Рабочая интерпретация временной шкалы | [time_scale.md](materials/02/time_scale.md) |

Дополнительный протокол парных регистраций У1/У3 находится в
`materials/02/paired_recording_protocol.md`; позднее уточнение о временном
разделении источников сохранено отдельно. Описания являются аннотированными
извлечениями с точными путями к исходникам, строками и SHA-256. Исходники
в репозитории не изменены. Нельзя выдавать извлечение за оригинал протокола.

## Таблицы и рисунки

В `tables/report/` представлены 14 таблиц результатов собранного научного текста:
строковые CSV и соответствующие подразделы с условиями и ограничениями.
Это перенос уже приведённых результатов из отчёта, без пересчёта и без
заявления о наличии первичных расчётных CSV. Их происхождение, номера
текущей сборки и хэши источников приведены в `tables/index.json`.

В `tables/lateral_u2/` отдельно сохранены 24 обработанных боковых уровня
записей 002–011 из таблицы 3 отчёта 33.31. Разметка дыхательных интервалов
кандидатная. Запись 001 в эту таблицу не входит. Другие электрические
расчёты У2 из исходного отчёта 33.31 в пакет не переносились.

В `tables/ct_processed/` находятся два доступных обработанных CSV:
27 фазовых строк объёмов и 108 строк камерных изменений У1/У2. Это
аннотированные извлечения: исходные коды участников заменены на У1/У2,
прочие ячейки сохранены без пересчёта. Дополнительный исторический
участник исключён; динамические данные У3 не добавлены. Неполные R–R
не замыкаются, разность масок не принимается за отдельную маску миокарда.
Точные номера строк и хэши — `tables/processed_index.json`.

В `figures/` включены 16 научных PNG из текущего `build_manifest.json`.
Список с подписями, текущими номерами и хэшами — `figures/index.json`.
Три фотографии подготовки не продублированы: их персональные исходные
копии требуют отдельного отбора изображений для передачи. Это ограничение
не означает отсутствия фотографий в авторском отчёте. Обработанные
иллюстрации КТ являются рисунками отчёта; DICOM и геометрические массивы
в пакет не включены.

## Программы и непокрытые зависимости

В `programs/references.json` указаны точные пути и SHA-256 доступных
программ. Код и его зависимости не скопированы; программы не запускались.
Для полного расчёта нужен репозиторий и соответствующие индивидуальные
входы. Смешанные программы могут поддерживать более широкий круг задач,
чем разрешённая область этого комплекта.

В `provenance/uncovered_dependencies.json` перечислены отсутствующие
локальные пакеты обработки, первичные таблицы поясной и монтажной серий
и внешний решатель. Доступность установлена только для указанных путей;
отсутствие файла здесь не доказывает отсутствия выполненного расчёта.
Диск внешних данных недоступен по условию задания. Зеркало и сетевые
копии не использовались. Исходные DICOM, маски, внешние STL, BIN, EIT,
полные физиологические TXT/CSV, фотографии с лицами, Word, драфты,
матрица соответствия и локальные конфигурации в ZIP не включены.
Область У2 ограничена записями 001–011; остальные ветки исключены.

## Контроль целостности и повторная упаковка

`archive_manifest.json` внутри ZIP содержит хэши всех полезных файлов,
кроме самого реестра и `CHECKSUMS.sha256`. В `CHECKSUMS.sha256` также
включён хэш внутреннего реестра. Внешний `archive_manifest.json` рядом
с ZIP дополнительно содержит SHA-256 всего ZIP и результат проверки.
Такая схема исключает круговую ссылку контрольных сумм.

Сборщик находится в `tools/build_companion_archive.py` данной редакции
в репозитории. Запуск из любой рабочей папки:

```text
python -B -X utf8 <путь_к_редакции>/tools/build_companion_archive.py
python -B -X utf8 <путь_к_редакции>/tools/build_companion_archive.py --verify-only
```

Для обновления сначала завершить общую сборку редакции и её манифест,
затем повторить упаковку. Если текущий рисунок отличается от последней
сборки Word, ожидаемый и фактический хэши указаны отдельно в реестре:
включён текущий сохранённый файл этой же редакции. Параметр
`--strict-build-hashes` запрещает такую промежуточную упаковку.
При новом неизвестном ключе рисунка или изменении источника во время
чтения сборщик останавливается.
Рекурсивного копирования каталогов и обращения к внешним хранилищам нет.
Проверяются CRC, хэши, пути и состав ZIP. Эти проверки не устанавливают
анатомическую, метрологическую или физиологическую точность результатов.
'''


def verify(data: bytes):
    if len(data) > LIMIT:
        raise ValueError('ZIP exceeds the size limit')
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        if z.testzip() is not None:
            raise ValueError('ZIP CRC test failed')
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP members')
        for name in names:
            p = PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name:
                raise ValueError('Unsafe ZIP path')
        m = json.loads(z.read('archive_manifest.json'))
        if sum(e.file_size for e in z.infolist()) > LIMIT:
            raise ValueError('Uncompressed payload exceeds the size limit')
        expected = {e['path'] for e in m['files']} | {'archive_manifest.json', 'CHECKSUMS.sha256'}
        if set(names) != expected:
            raise ValueError('Manifest member set mismatch')
        for e in m['files']:
            content = z.read(e['path'])
            if len(content) != e['bytes'] or digest(content) != e['sha256']:
                raise ValueError(f'Payload hash mismatch: {e["path"]}')
            if PurePosixPath(e['path']).suffix not in {'.md', '.json', '.csv', '.png'}:
                raise ValueError(f'Non-allowlisted extension: {e["path"]}')
            if e['path'].endswith('.png'):
                if not content.startswith(b'\x89PNG\r\n\x1a\n'):
                    raise ValueError('Non-PNG image')
            else:
                text = content.decode('utf-8')
                if re.search(r'ТЕПС|АЕПС|TEPC|TEPS|AEPC|прекарди', text, re.I):
                    raise ValueError(f'Excluded branch in payload: {e["path"]}')
                if re.search(r'(?i)(?<![a-z])[a-z]:[\\/]', text):
                    raise ValueError('Absolute machine path in payload')
                if re.search(r'SCV2-G-0(?:1[2-9]|[2-9][0-9])\b', text):
                    raise ValueError('Out-of-scope U2 record')
        figure_index = json.loads(z.read('figures/index.json'))
        table_index = json.loads(z.read('tables/index.json'))
        if {t['key'] for t in table_index} != set(TABLE_KEYS) or len(table_index) != len(TABLE_KEYS):
            raise ValueError('Report table allowlist mismatch')
        approved_images = {f['archive_path'] for f in figure_index if f['status'] == 'packaged'}
        actual_images = {n for n in names if n.endswith('.png')}
        if approved_images != actual_images or len(actual_images) != len(FIGURE_KEYS):
            raise ValueError('Figure allowlist mismatch')
        if {f['key'] for f in figure_index if f['status'] == 'packaged'} != FIGURE_KEYS:
            raise ValueError('Figure key mismatch')
        observed = list(csv.reader(io.StringIO(z.read(
            'tables/lateral_u2/observed_levels_002_011.csv').decode('utf-8'))))
        if len(observed) != 25 or any(not 1 <= int(r[0]) <= 11 for r in observed[1:]):
            raise ValueError('Lateral observation scope mismatch')
        for filename, count in [('chamber_volumes_by_rr.csv', 27),
                                ('chamber_volume_changes_measured_cycles.csv', 108)]:
            ct = list(csv.DictReader(io.StringIO(z.read(
                'tables/ct_processed/' + filename).decode('utf-8'))))
            if len(ct) != count or {r['subject'] for r in ct} != {'У1', 'У2'}:
                raise ValueError('Processed CT table scope mismatch')
        sums = {}
        for line in z.read('CHECKSUMS.sha256').decode('utf-8').splitlines():
            h, name = line.split('  ', 1)
            if name in sums or digest(z.read(name)) != h:
                raise ValueError(f'Checksum mismatch: {name}')
            sums[name] = h
        if set(sums) != set(names) - {'CHECKSUMS.sha256'}:
            raise ValueError('Checksum member set mismatch')
    return {'zip_test': 'passed', 'member_hashes': 'passed', 'checksums': 'passed',
            'member_count': len(names), 'scope_allowlist': 'passed',
            'primary_medical_arrays_included': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--strict-build-hashes', action='store_true',
                        help='Require current PNG hashes to match the last Word build')
    args = parser.parse_args()
    dest = BASE / 'deliverables'
    archive_path = dest / ARCHIVE_NAME
    manifest_path = dest / 'archive_manifest.json'
    if args.verify_only:
        data = archive_path.read_bytes()
        m = json.loads(manifest_path.read_text('utf-8'))
        if digest(data) != m['archive']['sha256'] or len(data) != m['archive']['bytes']:
            raise ValueError('Whole ZIP hash mismatch')
        checks = verify(data)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            if digest(z.read('archive_manifest.json')) != m['internal_manifest_sha256']:
                raise ValueError('Internal manifest mismatch')
        print(json.dumps(checks, ensure_ascii=False))
        return
    data, manifest = Builder(strict_build_hashes=args.strict_build_hashes).build()
    dest.mkdir(exist_ok=True)
    # Replacement is confined to the two requested outputs, with no cleanup
    # of other deliverables belonging to parallel editors.
    for path, content in [(archive_path, data), (manifest_path, json_bytes(manifest))]:
        temporary = path.with_name(path.name + '.companion.tmp')
        temporary.write_bytes(content)
        temporary.replace(path)
    print(json.dumps({'archive': str(archive_path), 'bytes': len(data),
                      'sha256': digest(data), 'counts': manifest['counts'],
                      'validation': manifest['validation']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError, StopIteration, AttributeError) as error:
        print(f'Companion archive: {error}', file=sys.stderr)
        sys.exit(1)
