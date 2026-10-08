"""Build a local reader report without changing signals or accepted annotations."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    out = experiment / 'derived/scorpion_bin_review'
    inventory = json.loads((experiment / 'metadata/inventory.local.json').read_text(encoding='utf-8'))
    structure_path = out / 'binary_structure.local.json'
    matches_path = out / 'txt_binary_matches.local.json'
    structure = json.loads(structure_path.read_text(encoding='utf-8'))
    matching = json.loads(matches_path.read_text(encoding='utf-8'))
    entries = {entry['record_id']: entry for entry in inventory['files']}
    matches = {match['record_id']: match for match in matching['matches']}
    if len(matches) != len(matching['matches']):
        raise ValueError('Multiple BIN matches require explicit resolution')
    rates = []
    rows = []
    for record_id in inventory['signal_record_ids']:
        entry = entries[record_id]
        match = matches.get(record_id)
        if match:
            controls = match['settings_states_in_record']
            if len(controls) != 1 or controls[0]['control'] is None:
                raise ValueError('A record crosses configuration states')
            word = controls[0]['control']['u16_at_2']
            if word not in (1000, 2500):
                raise ValueError('Unreviewed configuration value')
            fs = int(1_000_000 / word)
            status = 'supported_by_BIN_alignment_and_period_interpretation_units_not_documented'
            basis = f'{match["bin_id"]}; поле +2 = {word}'
        else:
            word = None
            fs = inventory['author_reported_default_sample_rate_hz']
            status = 'author_default_only_no_corresponding_BIN_segment_found'
            basis = 'Настройка по умолчанию; соответствующий BIN не найден'
        n = entry['text_qc']['rows']
        rates.append({'record_id': record_id, 'txt_sha256': entry['sha256'],
                      'working_sample_rate_hz': fs, 'sample_period_word_at_offset_2': word,
                      'duration_s_N_over_fs': n / fs, 'status': status,
                      'independent_device_protocol_confirmation': False,
                      'bin_match': match})
        name = Path(entry['raw_relative_path']).name
        rows.append(f'<tr><td>{record_id}</td><td>{escape(name)}</td><td>{basis}</td>'
                    f'<td>{fs}</td><td>{n / fs:.4f}</td></tr>')
    timing = []
    for before, after in [('SCV2-G-005', 'SCV2-G-006'), ('SCV2-G-007', 'SCV2-G-018')]:
        a, b = matches[before], matches[after]
        if a['bin_id'] != b['bin_id']:
            raise ValueError('Time comparison needs one continuous BIN')
        gap = b['end_data_index_exclusive'] - a['end_data_index_exclusive']
        seconds = (datetime.fromisoformat(entries[after]['zip_mtime_without_timezone'])
                   - datetime.fromisoformat(entries[before]['zip_mtime_without_timezone'])).total_seconds()
        timing.append({'before_record_id': before, 'after_record_id': after,
                       'BIN_data_packet_delta': gap, 'TXT_zip_mtime_delta_s': seconds,
                       'packet_delta_divided_by_zip_mtime_delta_hz': gap / seconds,
                       'limitation': 'ZIP_mtime_is_export_proxy_not_acquisition_timestamp'})
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    profile = {'status': 'working_timebase_device_protocol_confirmation_pending',
               'generated_utc': datetime.now(timezone.utc).isoformat(),
               'assumption': 'u16_at_decoded_offset_2_is_sample_period_in_microseconds',
               'support': 'same_text_record_located_in_binary_and_timestamp_proxy_and_signal_period',
               'code_sha256': code_hash,
               'inventory_sha256_at_report': hashlib.sha256((experiment / 'metadata/inventory.local.json').read_bytes()).hexdigest(),
               'structure_sha256': hashlib.sha256(structure_path.read_bytes()).hexdigest(),
               'matches_sha256': hashlib.sha256(matches_path.read_bytes()).hexdigest(),
               'records': rates, 'timestamp_proxy_comparisons': timing}
    (out / 'working_sampling_rates.local.json').write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    settings_rows = []
    hex_blocks = []
    file_rows = []
    for file in structure['files']:
        file_rows.append(f'<tr><td>{file["file_id"]}</td><td>{escape(file["source_name"])}</td>'
                         f'<td>{file["bytes"]:,}</td><td>{file.get("data_packets", 0):,}</td></tr>')
        for state in file.get('control_variants', []):
            values = ' / '.join(str(state[f'u16_at_{i}']) for i in (2, 4, 6, 8, 10))
            groups = ', '.join(f'{a}:{b}' for a, b in state['groups_from_14_u32_u16'])
            settings_rows.append(f'<tr><td>{file["file_id"]}:{state["state_id"]}</td>'
                                 f'<td>{state["decoded_length"]}</td><td>{values}</td>'
                                 f'<td>{state["byte_12"]} / {state["byte_13"]}</td>'
                                 f'<td>{groups}</td><td>{state["occurrences"]}</td></tr>')
            hex_blocks.append(f'<details><summary>{file["file_id"]}:{state["state_id"]} — исходные байты сообщения 0x11</summary>'
                              f'<pre>{state["decoded_hex"]}</pre></details>')
    page = '''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Scorpion v2: BIN и настройки</title>
<style>body{font:15px system-ui;line-height:1.5;margin:28px;max-width:1400px;color:#172033}table{border-collapse:collapse;width:100%;margin:16px 0}td,th{border:1px solid #d8e0ea;padding:8px;text-align:left}th{background:#edf2f7}pre{white-space:pre-wrap;word-break:break-word}.note{background:#fff4ce;padding:16px;border-radius:8px}a{color:#2459b5}.wide{overflow:auto}</style>
<h1>Scorpion v2 — бинарные потоки и параметры регистрации</h1>
<p>Георг · BMSTU · 05.10.2026. Разбор от 07.10.2026. Первичные байты сохранены; дыхательная и QRS-разметка не принималась автоматически.</p>
<p class="note"><strong>Новая информация:</strong> три непустых BIN содержат поток измерительных и служебных пакетов. Прежнее описание «только настройки, для анализа не нужны» уточнено. Непрерывные участки для 16 из 18 TXT найдены по самим данным. Рабочие частоты: 400 Гц для 005/006; 1000 Гц для остальных. Единицу поля периода требуется подтвердить у разработчика; для 001/002 BIN-основания нет.</p>
<h2>Состав</h2><table><tr><th>BIN</th><th>Имя</th><th>Байты</th><th>Пакеты измерений</th></tr>'''
    page += ''.join(file_rows) + '''</table>
<p>Разделитель пакетов — 00; структура совместима с COBS. Сообщение 0x10 содержит циклический COUNT, два знаковых 24-битных поля ЭКГ и блоки импедансных данных; длина 15 + 22 × число блоков, 37 или 191 байт. 0x11 содержит повторяющиеся параметры. В текстовом сообщении присутствует <code>version: 0.001, compiled: Sep 24 2026 18:55:17</code>. Принадлежность версии прошивке или другому компоненту ещё не подтверждена документацией.</p>
<h2>Соответствие TXT и BIN</h2>
<p>Поиск использует 1024 отсчёта ЭКГ, затем проверяется <strong>вся длина записи</strong>: COUNT совпадает в каждой строке, оба поля ЭКГ соответствуют знаковым 24-битным числам BIN × 0,001 мВ с отклонением не более 0,005 мВ, то есть половины шага TXT 0,01 мВ. Преобразование эмпирическое; это проверка соответствия файлов, а не независимая калибровка. TXT читаются непосредственно из raw с проверкой SHA-256.</p>
<table><tr><th>TXT</th><th>Исходное имя</th><th>Основание</th><th>Рабочая fs, Гц</th><th>N/fs, с</th></tr>'''
    page += ''.join(rows) + '''</table>
<h2>Временная шкала</h2>
<p>В 005 и 006 действует поле +2 = 2500, а в 003, 004 и 007–018 — 1000. Интерпретация как периода в мкс даёт fs = 10⁶ / поле: 400 и 1000 Гц. Внутри каждого найденного TXT поле не меняется. Длинный BIN фиксирует переход 2500 → 1000 между записью 006 и 007.</p>
<p>Дополнительная проверка: между концами 005/006 в BIN 58 472 пакета, разность времён TXT в ZIP 146 с, отношение 400,49 Гц. Между концами 007/018 — 3 123 286 пакетов и 3124 с, отношение 999,77 Гц. Время изменения файла — приближённый ориентир экспорта, имеет ограниченную точность и не является измеренной шкалой прибора. Эти сравнения поддерживают 400/1000 Гц независимо от назначения спектральной линии сетевой помехе, но не проверяют точность тактового генератора.</p>
<h2>Остальные параметры: значения без выдуманной расшифровки</h2>
<p>Смещения указаны от первого байта декодированного сообщения, начиная с нуля; числа little-endian. Задокументированного протокола нет. Байту +12 соответствует число блоков данных (8 или 1). Остальные назначения остаются открытыми.</p>
<div class="wide"><table><tr><th>BIN:состояние</th><th>Байты</th><th>u16 на +2 / +4 / +6 / +8 / +10</th><th>Байты +12 / +13</th><th>Группы u32:u16 от +14</th><th>Повторы</th></tr>'''
    page += ''.join(settings_rows) + '''</table></div>
<p>В одноблочном режиме группа имеет значения 100 000 000 и 7000. Если первое число задано в мГц, оно соответствует подтверждённому автором зондированию 100 кГц; единица пока не установлена. Значение 7000 не названо амплитудой тока или усилением. Поля +4 и +10 меняются вместе с +2: 1000/1300 при 2500 и 450/550 при 1000. Это факт о байтах; значение интеграции, фильтра или паузы не установлено.</p>
<h2>Ограничения и вопросы разработчику</h2>
<ul><li>Подтвердить формат 0x11, значение и единицу u16 +2, поля +4/+6/+8/+10, байта +13 и пары u32:u16.</li><li>Уточнить алгоритм двух последних байтов сообщения: контрольная сумма предполагается, но не проверена.</li><li>В длинном BIN есть частичный пакет в начале и один внутренний блок, не прошедший COBS-разбор. Переход COUNT с ненулевым разрывом расположен вне найденных TXT; источник разрыва не установлен. Это не дефект ZIP: первичные хэши совпадают.</li><li>Возможные аппаратные фильтры, межканальные задержки, шкала и физический смысл BPH/CE/CPH из этого разбора не установлены.</li><li>Дыхательные команды не идентифицированы; дополнительные участки BIN не включены автоматически в физиологический анализ.</li></ul>
<h2>Байты и воспроизводимость</h2>'''
    page += ''.join(hex_blocks) + '''<p><a href="binary_structure.local.json">Структура и переходы параметров</a> · <a href="txt_binary_matches.local.json">Полное соответствие TXT/BIN</a> · <a href="working_sampling_rates.local.json">Рабочая временная шкала</a> · <a href="../scorpion_review/index.html">Просмотр сигналов</a></p></html>'''
    (out / 'settings_review.html').write_text(page, encoding='utf-8')
    (out / 'report_source.local.py').write_bytes(Path(__file__).read_bytes())
    print(out / 'settings_review.html')


if __name__ == '__main__':
    main()
