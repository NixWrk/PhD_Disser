"""Export every available rho(L)/pulse-delta-rho(L) pair from the frozen CSV.

This reader does not filter signals, fit parameters or run FEM. The previous
summary figures and their verification receipt remain unchanged.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import html
from io import StringIO
import json
import math
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parent
SERIES = {'Scorpion': '«Скорпион»', 'BMSTU': 'Реокардиомонитор МГТУ'}
PARTICIPANTS = {'P01': 'Георгий', 'P02': 'Ник', 'P03': 'Ярослав'}
STYLES = {
    '': ('o', '#6748a1', 'Боковая сборка МГТУ'),
    '1x1': ('o', '#2563eb', 'ТТРКГ 1×1'),
    '2x2': ('s', '#ac5400', 'ТТРКГ 2×2'),
    '1x4': ('^', '#148260', 'ТТРКГ 1×4'),
}
METRICS = {
    'baseline': ('rho0_ohm_m', 1.0, 'Базовое кажущееся ρ', 'Кажущееся ρ, Ом·м'),
    'pulse': ('delta_rho_median_ohm_m', 1000.0, 'Пульсовое кажущееся Δρ',
              'Медиана размахов Δρ, 10⁻³ Ом·м'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows():
    source = ROOT / 'summary.csv'
    verification = json.loads((ROOT / 'verification.json').read_text(encoding='utf-8'))
    if sha(source) != verification['summary_sha256']:
        raise ValueError('summary.csv отличается от проверенного снимка')
    with source.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != verification['data_rows']:
        raise ValueError('Число строк не соответствует паспорту данных')
    seen = set()
    for index, row in enumerate(rows, 2):
        row['_csv_row'] = index
        key = tuple(row[k] for k in ('series', 'participant_code', 'size_mm', 'ttrkg_variant'))
        if key in seen:
            raise ValueError(f'Повтор наблюдения: {key}')
        seen.add(key)
        if key[0] not in SERIES or key[1] not in PARTICIPANTS or key[3] not in STYLES:
            raise ValueError(f'Неизвестная серия, участник или монтаж: {key}')
        values = [float(row[k]) for k in ('rho0_ohm_m', 'delta_rho_median_ohm_m',
                                          'delta_rho_q25_ohm_m', 'delta_rho_q75_ohm_m')]
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError(f'Некорректная величина: строка {index}')
        if not values[2] <= values[1] <= values[3]:
            raise ValueError(f'Квартили не охватывают медиану: строка {index}')
        if int(row['n_cycles']) < 1:
            raise ValueError(f'Нет циклов: строка {index}')
        if row['weak_qrs_agreement'] not in ('', 'true', 'false'):
            raise ValueError(f'Неизвестная отметка ЭКГ: строка {index}')
        if row['series'] == 'Scorpion':
            weak = float(row['qrs_agreement_f1']) < .65
            if weak != (row['weak_qrs_agreement'] == 'true'):
                raise ValueError(f'Отметка ЭКГ не согласована с F₁: строка {index}')
    for series, receipt in verification['series'].items():
        group = [r for r in rows if r['series'] == series]
        if len(group) != receipt['records'] or sum(int(r['n_cycles']) for r in group) != receipt['cycles']:
            raise ValueError(f'Неполный набор записей/циклов: {series}')
    return rows, sha(source)


def ru(value, digits=5):
    return f'{value:.{digits}g}'.replace('.', ',')


def weak(row, metric):
    return metric == 'pulse' and row['weak_qrs_agreement'] == 'true'


def x_display(row, rows):
    # Only separate repeated nominal sizes; all axes/tables retain nominal L.
    same_size = sorted(r['ttrkg_variant'] for r in rows if r['size_mm'] == row['size_mm'])
    return float(row['size_mm']) + 2.5 * (same_size.index(row['ttrkg_variant']) - (len(same_size) - 1) / 2)


def draw(ax, rows, metric, upper=None):
    key, scale, _, ylabel = METRICS[metric]
    for variant in STYLES:
        group = sorted((r for r in rows if r['ttrkg_variant'] == variant), key=lambda r: int(r['size_mm']))
        if not group:
            continue
        marker, color, label = STYLES[variant]
        # A weak-QRS pulse observation remains visible but breaks the guide line.
        ax.plot([x_display(r, rows) for r in group],
                [math.nan if weak(r, metric) else scale * float(r[key]) for r in group],
                linestyle=':', linewidth=1.2, color=color, alpha=.8, zorder=1)
        for i, row in enumerate(group):
            value = float(row[key])
            error = None
            if metric == 'pulse':
                error = [[scale * (value - float(row['delta_rho_q25_ohm_m']))],
                         [scale * (float(row['delta_rho_q75_ohm_m']) - value)]]
            ax.errorbar(x_display(row, rows), scale * value, yerr=error, fmt=marker,
                        color=color, markerfacecolor='white' if weak(row, metric) else color,
                        markeredgewidth=1.4, markersize=6.5, capsize=3,
                        label=label if i == 0 else None, zorder=3)
    if upper is None:
        upper = 1.14 * scale * max(float(r['delta_rho_q75_ohm_m'] if metric == 'pulse' else r[key]) for r in rows)
    ax.set(xlim=(44, 148), ylim=(0, upper), xlabel='Номинальный размер боковой сборки L, мм', ylabel=ylabel)
    ax.set_xticks(sorted({int(r['size_mm']) for r in rows}))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: ru(v, 4)))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(alpha=.22)
    ax.spines[['top', 'right']].set_visible(False)


def legend(ax, rows, metric):
    variants = {r['ttrkg_variant'] for r in rows}
    handles = [Line2D([], [], marker=marker, color=color, linestyle='none', label=label)
               for variant, (marker, color, label) in STYLES.items() if variant in variants]
    labels = [h.get_label() for h in handles]
    if any(weak(r, metric) for r in rows):
        handles.append(Line2D([], [], marker='o', color='#555', markerfacecolor='white', linestyle='none'))
        labels.append('F₁ ЭКГ < 0,65')
    ax.legend(handles, labels, frameon=False, loc='best', fontsize=9)


def footer(fig, metric, scorpion):
    lines = []
    if metric == 'pulse':
        lines.append('Точка — медиана полных размахов циклов; отрезок — квартили 25 и 75 %.')
    if scorpion:
        lines.append('Дыхательная и ЭКГ-разметка кандидатные. Разные варианты ТТРКГ показаны отдельно.')
    lines.extend(['Пунктир соединяет наблюдения одного монтажа и служит ориентиром для чтения.',
                  'При совпадении L маркеры разнесены по горизонтали; размеры в таблице номинальные.'])
    fig.text(.13, .025, '\n'.join(lines), va='bottom', fontsize=8.5, linespacing=1.5)


def save(fig, stem):
    paths = []
    for extension in ('png', 'svg'):
        path = ROOT / 'figures' / f'{stem}.{extension}'
        if extension == 'svg':
            buffer = StringIO()
            fig.savefig(buffer, format='svg', metadata={'Date': None})
            path.write_text('\n'.join(line.rstrip() for line in buffer.getvalue().splitlines()) + '\n',
                            encoding='utf-8', newline='\n')
        else:
            fig.savefig(path, dpi=180, metadata={'Software': 'Matplotlib'})
        paths.append(path)
    plt.close(fig)
    return paths


def pack_png(assets):
    archive = ROOT / 'plots_png.zip'
    entries = {}
    with ZipFile(archive, 'w', compression=ZIP_DEFLATED) as bundle:
        for path in assets:
            if path.suffix != '.png':
                continue
            series, suffix = path.stem.split('_', 1)
            parts = suffix.split('_')
            prefix = 'Скорпион' if series == 'scorpion' else 'МГТУ'
            if parts[0].upper() in PARTICIPANTS:
                participant = PARTICIPANTS[parts[0].upper()]
                metric = parts[1]
            else:
                participant = 'все_испытуемые'
                metric = parts[0]
            quantity = 'ро_кажущееся' if metric == 'baseline' else 'дельта_ро_пульсовое'
            name = f'{prefix}_{participant}_{quantity}.png'
            info = ZipInfo(name, date_time=(2026, 11, 10, 12, 0, 0))
            info.compress_type = ZIP_DEFLATED
            bundle.writestr(info, path.read_bytes())
            entries[name] = path.relative_to(ROOT).as_posix()
    return archive, entries


def table(rows):
    parts = ['<details><summary>Числовые значения и показатели отбора циклов</summary>',
             '<table><thead><tr><th>Строка CSV</th><th>L, мм</th><th>ТТРКГ</th><th>ρ, Ом·м</th>',
             '<th>Δρ, Ом·м</th><th>Q25–Q75, Ом·м</th><th>Циклов</th><th>F₁ ЭКГ</th></tr></thead><tbody>']
    for r in sorted(rows, key=lambda r: (int(r['size_mm']), r['ttrkg_variant'])):
        values = [r['_csv_row'], r['size_mm'], r['ttrkg_variant'].replace('x', '×') or '—',
                  ru(float(r['rho0_ohm_m'])), ru(float(r['delta_rho_median_ohm_m'])),
                  f"{ru(float(r['delta_rho_q25_ohm_m']))}–{ru(float(r['delta_rho_q75_ohm_m']))}",
                  r['n_cycles'], ru(float(r['qrs_agreement_f1']), 3) if r['qrs_agreement_f1'] else '—']
        parts.append('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in values) + '</tr>')
    return '\n'.join(parts + ['</tbody></table></details>'])


def build_html(groups):
    parts = ['<!doctype html><html lang="ru"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             '<title>Кажущееся ρ и пульсовое Δρ по размерам сборки</title>',
             '<style>body{font:16px/1.55 Arial,sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;color:#20242a} '
             'h1{font-size:29px}h2{margin-top:45px}h3{margin-top:35px}a{color:#174ead} '
             '.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}img{width:100%;height:auto} '
             'figure{margin:0}figcaption{font-size:14px}table{border-collapse:collapse;width:100%;font-size:14px} '
             'td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}details{margin:20px 0;overflow:auto} '
             'summary{cursor:pointer}nav{display:flex;gap:20px;flex-wrap:wrap} '
             '@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>',
             '<h1>Кажущееся ρ и пульсовое Δρ от размера электродной сборки</h1>',
             '<p>Задержка глубокого вдоха. Пересчёт по индивидуальной однородной КТ/FEM-модели. '
             'Представлены все 34 наблюдения сохранённого расчёта: 15 для «Скорпиона» и 19 для реокардиомонитора МГТУ.</p>',
             '<nav><a href="#scorpion">«Скорпион»: три испытуемых</a><a href="#bmstu">МГТУ: два испытуемых</a>'
             '<a href="summary.csv">Сводная таблица CSV</a><a href="README.md">Источники и ограничения</a></nav>',
             '<p>Δρ — медиана полных размахов отдельных циклов; отрезки показывают квартили внутри записи. '
             'Это разброс циклов, а не доверительный интервал оценки. Пустые маркеры сохраняют четыре '
             'наблюдения со слабым согласием ЭКГ (F₁ &lt; 0,65). Кажущееся ρ относится к однородной '
             'эквивалентной области; отдельные свойства тканей здесь не определяются.</p>']
    for series, name in SERIES.items():
        parts.append(f'<h2 id="{series.lower()}">{name}</h2>')
        if series == 'BMSTU':
            parts.append('<p>Для Ярослава в этом расчётном снимке нет записей МГТУ. '
                         'Для Ника отсутствует независимая запись 100 мм; пропуск сохранён.</p>')
        for (s, participant), rows in groups.items():
            if s != series:
                continue
            title = f'{participant} · {PARTICIPANTS[participant]}'
            parts.append(f'<h3 id="{s.lower()}-{participant.lower()}">{title}</h3><div class="pair">')
            for metric in METRICS:
                stem = f'{series.lower()}_{participant.lower()}_{metric}'
                encoded = base64.b64encode((ROOT / 'figures' / f'{stem}.png').read_bytes()).decode('ascii')
                parts.append(f'<figure><img src="data:image/png;base64,{encoded}" '
                             f'alt="{html.escape(METRICS[metric][2])}: {html.escape(title)}">'
                             f'<figcaption><a href="figures/{stem}.png">PNG</a> · '
                             f'<a href="figures/{stem}.svg">SVG</a></figcaption></figure>')
            parts.extend(['</div>', table(rows)])
    parts.append('<p>Изображения встроены в HTML; для его просмотра дополнительные файлы не нужны. '
                 'Ссылки на CSV, SVG и отдельные PNG требуют соответствующих файлов из репозитория.</p></html>')
    (ROOT / 'index.html').write_text('\n'.join(parts) + '\n', encoding='utf-8', newline='\n')


def main():
    rows, source_hash = read_rows()
    groups = {(series, participant): [r for r in rows if r['series'] == series and r['participant_code'] == participant]
              for series in SERIES for participant in PARTICIPANTS
              if any(r['series'] == series and r['participant_code'] == participant for r in rows)}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none',
                         'svg.hashsalt': 'apparent-resistivity-by-participant-20261110'})
    (ROOT / 'figures').mkdir(exist_ok=True)
    assets, plots = [], []
    for (series, participant), group in groups.items():
        for metric in METRICS:
            fig, ax = plt.subplots(figsize=(8.2, 5.5))
            fig.subplots_adjust(left=.13, right=.97, top=.80, bottom=.29)
            draw(ax, group, metric)
            legend(ax, group, metric)
            fig.suptitle(f'{PARTICIPANTS[participant]} ({participant}) · {SERIES[series]}', fontsize=15, y=.96)
            fig.text(.5, .88, METRICS[metric][2] + ' · задержка глубокого вдоха', ha='center', fontsize=11)
            footer(fig, metric, series == 'Scorpion')
            stem = f'{series.lower()}_{participant.lower()}_{metric}'
            paths = save(fig, stem)
            assets.extend(paths)
            plots.append({'series': series, 'participant_code': participant, 'metric': metric,
                          'csv_rows': [r['_csv_row'] for r in group], 'points': len(group),
                          'weak_qrs_markers': sum(weak(r, metric) for r in group),
                          'files': [p.relative_to(ROOT).as_posix() for p in paths]})
    for series in SERIES:
        subset = [(p, g) for (s, p), g in groups.items() if s == series]
        all_rows = [r for r in rows if r['series'] == series]
        for metric, (key, scale, title, _) in METRICS.items():
            upper = 1.14 * scale * max(float(r['delta_rho_q75_ohm_m'] if metric == 'pulse' else r[key]) for r in all_rows)
            fig, axes = plt.subplots(1, len(subset), figsize=(5.6 * len(subset), 5.7), sharey=True)
            fig.subplots_adjust(left=.065, right=.985, top=.79, bottom=.29, wspace=.24)
            for ax, (participant, group) in zip(axes, subset):
                draw(ax, group, metric, upper)
                legend(ax, group, metric)
                ax.set_title(f'{PARTICIPANTS[participant]} ({participant})', fontsize=12)
            fig.suptitle(SERIES[series] + ' · ' + title, fontsize=16, y=.96)
            fig.text(.5, .89, 'Задержка глубокого вдоха · одинаковая вертикальная шкала', ha='center', fontsize=11)
            footer(fig, metric, series == 'Scorpion')
            assets.extend(save(fig, f'{series.lower()}_{metric}_all_participants'))
    build_html(groups)
    png_archive, png_entries = pack_png(assets)
    manifest = {'schema': 'apparent_resistivity_plot_coverage_v1', 'source': 'summary.csv',
                'source_sha256': source_hash, 'records': len(rows), 'participant_groups': len(groups),
                'participant_labels': PARTICIPANTS,
                'participant_label_sources': ['../../SCORPION_PULSE_RESISTIVITY_METHOD.md',
                                              '../../../Colab Notebooks/scorpion_v2/plot_apparent_resistivity.py',
                                              '../../../Colab Notebooks/plot_bmstu_apparent_resistivity.py'],
                'individual_plots': plots, 'image_files': len(assets),
                'images_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in assets},
                'png_bundle': {'path': png_archive.name, 'sha256': sha(png_archive),
                               'files': png_entries},
                'html_sha256': sha(ROOT / 'index.html'),
                'interpretation': 'Redraw of saved aggregates only; no new signal processing, FEM or physiological validation.'}
    (ROOT / 'plot_coverage.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    if sha(ROOT / 'summary.csv') != source_hash:
        raise RuntimeError('Исходная таблица изменилась при построении')
    print(json.dumps({'records': len(rows), 'groups': len(groups), 'individual_plots': len(plots),
                      'image_files': len(assets), 'output': str(ROOT / 'index.html')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
