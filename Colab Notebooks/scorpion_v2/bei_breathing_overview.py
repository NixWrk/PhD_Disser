"""Display-only BEI review from the existing 11.20 cache and candidate intervals."""
from pathlib import Path
import json
import numpy as np
from plotly.offline import get_plotlyjs
from lateral_pilot import EXPERIMENT, PRIMARY, candidate_windows, read, save, sha
from experiment_signal_display import phase_intervals
from scorpion_qc import MODES, MODE_NAMES

FOLDER = Path(__file__).parent
OUT = EXPERIMENT / 'derived/exploratory/33.21_bei_breathing_overview_20261009'
REVIEW = EXPERIMENT / 'derived/scorpion_review'
ANNOTATIONS = EXPERIMENT / 'annotations/scorpion_review/annotations_candidates.local.json'
VIEWER = EXPERIMENT / 'derived/exploratory/10.00_experiment_view_20261009'
BUCKETS = 600


def extrema_indices(values, buckets=BUCKETS):
    """Retain every bucket's extrema and both sides of each nonfinite gap."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.array([], dtype=int)
    keep = {0, len(values)-1}
    for left, right in zip(np.linspace(0, len(values), min(buckets, len(values))+1, dtype=int)[:-1],
                           np.linspace(0, len(values), min(buckets, len(values))+1, dtype=int)[1:]):
        finite = np.flatnonzero(np.isfinite(values[left:right])) + left
        if len(finite):
            keep.update((int(finite[np.argmin(values[finite])]), int(finite[np.argmax(values[finite])])))
    invalid = ~np.isfinite(values)
    changes = np.flatnonzero(np.diff(np.r_[False, invalid, False]))
    for left, right in zip(changes[::2], changes[1::2]):
        keep.update(i for i in (left-1, left, right-1, right) if 0 <= i < len(values))
    return np.array(sorted(keep), dtype=int)


def segment(time, values, start, end):
    """Half-open sample interval; statistics use all samples, not the display subset."""
    values = np.asarray(values[start:end], dtype=float)
    time = np.asarray(time[start:end], dtype=float)
    finite = values[np.isfinite(values)]
    stats = dict(n=len(values), nonfinite=int(np.sum(~np.isfinite(values))),
                 below_zero=int(np.sum(finite < 0)))
    for key, value in zip(('minimum', 'q25', 'median', 'q75', 'maximum'),
                          np.quantile(finite, [0, .25, .5, .75, 1]) if len(finite) else [None]*5):
        stats[key] = None if value is None else float(value)
    idx = extrema_indices(values)
    return dict(samples=[int(start), int(end)], start=float(time[0]), end=float(time[-1]),
                t=np.round(time[idx], 6).tolist(),
                y=[float(v) if np.isfinite(v) else None for v in values[idx]], stats=stats)


def inputs():
    summary = read(REVIEW/'summary.local.json')
    annotations = read(ANNOTATIONS)
    if summary['code_sha256'] != sha(FOLDER/'scorpion_qc.py'):
        raise ValueError('11.20 signal producer changed')
    contract = read(VIEWER/'contract.local.json')
    if sha(VIEWER/'contract.local.json') != read(VIEWER/'manifest.local.json')['outputs']['contract.local.json']:
        raise ValueError('10.00 signal provenance changed')
    if contract['summary_sha256'] != sha(REVIEW/'summary.local.json') or contract['annotations_sha256'] != sha(ANNOTATIONS):
        raise ValueError('10.00 and 11.20 inputs differ')
    sources = {r['record_id']: r for r in contract['sources']}
    records = []
    for r in summary['records']:
        fs = 400 if r['record_id'][-3:] in ('005', '006') else 1000
        if r['fs_hz_scenario'] != fs:
            continue
        source = sources[r['record_id']]
        cache = REVIEW/(r['key']+'.npz')
        if sha(cache) != source['cache_sha256'] or sha(EXPERIMENT/r['source_relative_path']) != source['source_sha256']:
            raise ValueError('Recorded BEI or 11.20 cache changed: '+r['record_id'])
        a = next(a for a in annotations['records'] if (a['record_id'], a['fs_hz_scenario']) == (r['record_id'], fs))
        if a['source_sha256'] != r['source_sha256'] or r['source_sha256'] != source['source_sha256']:
            raise ValueError('Annotation refers to a different record')
        records.append((r, a, cache))
    if len(records) != 18:
        raise ValueError('Expected eighteen independent recordings')
    return records


def run(out=OUT):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    records, sources = [], []
    for r, a, cache in inputs():
        fs = r['fs_hz_scenario']
        with np.load(cache) as arrays:
            t = arrays['time_s']
            np.testing.assert_allclose(t, np.arange(len(t))/fs, atol=1e-10, rtol=0)
            breathing = a['breathing']
            phases = phase_intervals(breathing, fs, len(t)/fs)
            if not phases:
                phases = [dict(start=0., end=len(t)/fs, mode='unlabelled', label='Без разметки')]
            windows = candidate_windows(breathing, fs, len(t))
            channels = []
            for ch in (1, 2):
                y = arrays[f'bei_raw_{ch}']
                parts = []
                for p in phases:
                    start, end = int(round(p['start']*fs)), int(round(p['end']*fs))
                    part = {**p, 'full': segment(t, y, start, end), 'analysis': None}
                    state = {'inspiration_hold':'inhale', 'expiration_hold':'exhale'}.get(p['mode'])
                    if state in windows:
                        left, right = windows[state]
                        if not start <= left < right <= end:
                            raise ValueError('Analysis window outside its breathing stage')
                        part['analysis'] = segment(t, y, left, right)
                    parts.append(part)
                channels.append(dict(channel=ch, phases=parts))
        records.append(dict(record_id=r['record_id'], label=r['record_id'][-3:]+' · '+r['montage_label'],
                            primary=r['record_id'][-3:] in PRIMARY, fs=fs, timebase_status=r['timebase_status'],
                            channels=channels, duration=len(t)/fs,
                            ambiguity_s=(np.asarray(breathing.get('boundary_uncertainty_ranges_samples', []))/fs).tolist(),
                            annotation_accepted=breathing.get('author_accepted', False)))
        sources.append(dict(record_id=r['record_id'], cache=r['key']+'.npz', cache_sha256=sha(cache),
                            source_relative_path=r['source_relative_path'], source_sha256=r['source_sha256']))
    payload = dict(records=records, modes=MODES+['unlabelled'], mode_labels=MODE_NAMES+['Без разметки'])
    save(out/'display.local.json', payload)
    save(out/'contract.local.json', dict(summary_sha256=sha(REVIEW/'summary.local.json'),
         annotations_sha256=sha(ANNOTATIONS), producer_sha256=sha(__file__),
         dependencies={name:sha(FOLDER/name) for name in ['scorpion_qc.py', 'experiment_signal_display.py', 'lateral_pilot.py', 'bei_breathing_view.js']},
         sources=sources, buckets_per_stage=BUCKETS, filters_applied=False, annotations_modified=False,
         statistics_use_full_samples=True, negative_threshold_is_visual_only=True))
    save(out/'manifest.local.json', dict(outputs={name:sha(out/name) for name in ['display.local.json', 'contract.local.json']}))
    return out


def verify(out=OUT):
    out = Path(out)
    for name, digest in read(out/'manifest.local.json')['outputs'].items():
        if sha(out/name) != digest:
            raise ValueError('BEI display output changed')
    c = read(out/'contract.local.json')
    if c['producer_sha256'] != sha(__file__):
        raise ValueError('BEI display producer changed')
    for name, digest in c['dependencies'].items():
        if sha(FOLDER/name) != digest:
            raise ValueError('BEI display dependency changed')
    if c['summary_sha256'] != sha(REVIEW/'summary.local.json') or c['annotations_sha256'] != sha(ANNOTATIONS):
        raise ValueError('BEI summary or candidate intervals changed')
    for r in c['sources']:
        if sha(REVIEW/r['cache']) != r['cache_sha256'] or sha(EXPERIMENT/r['source_relative_path']) != r['source_sha256']:
            raise ValueError('BEI input changed')
    return c


def widget(out=OUT):
    verify(out)
    payload = read(Path(out)/'display.local.json')
    ui = '''<section id="bei-review" style="border:1px solid #aaa;padding:14px">
<h3>Обзор BEI до расчёта сопротивлений</h3>
<p><label>Этап дыхания: <select data-control="phase"><option value="all">Все этапы / вся запись</option></select></label>
<label>Поля: <select data-control="channel"><option value="both">BEI1 и BEI2</option><option value="1">Только BEI1</option><option value="2">Только BEI2</option></select></label></p>
<p><label>Шкала кривых: <select data-control="scale"><option value="absolute">Исходный BEI, Ом</option><option value="centred">Отклонение от медианы этапа, Ом</option></select></label>
<label><input type="checkbox" data-control="analysis"> Только интервалы расчёта уровней 33.21</label></p>
<p><label>Размах на графике уровней: <select data-control="spread"><option value="range">Минимум–максимум</option><option value="iqr">Квартили 25–75%</option><option value="none">Только медиана</option></select></label>
<label><input type="checkbox" data-control="dips" checked> Выделить нисходящие отклонения</label>
<label>Порог ниже медианы этапа, Ом: <input type="number" data-control="threshold" value="1" min="0" step="0.1" style="width:75px"></label></p>
<p><button type="button" data-control="all">Все 18 записей</button> <button type="button" data-control="main">Пять размеров основного ряда</button>
<button type="button" data-control="none">Снять выбор</button> <button type="button" data-control="reset">Сбросить масштаб</button></p>
<div data-control="records" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:6px"></div>
<p data-control="status"></p><p data-control="phase-key"></p>
<p>Цвет обозначает запись; сплошная линия — BEI1, штриховая — BEI2. Красные треугольники отмечают показанные значения ниже медианы своего этапа на выбранный порог. Порог задаёт визуальное выделение и не устанавливает дефект записи. Чёрная точечная линия — нулевой уровень выбранной шкалы.</p>
<div id="bei-time"></div>
<p>На нижнем графике точки — медианы этапов в исходных омах. Вертикальные отрезки показывают выбранный размах или межквартильный диапазон. Это вариабельность записанных значений, не доверительный интервал. Линии соединяют этапы одной записи и не означают непрерывную временную шкалу.</p>
<div id="bei-levels"></div>
<p>Таблица использует все отсчёты выбранных интервалов; масштабирование графика её не пересчитывает. «Ниже нуля» относится к исходному BEI. Неконечные значения не соединяются линией.</p>
<div data-control="line-key"></div>
<div data-control="table" style="max-height:360px;overflow:auto"></div>
</section>'''
    return ui+'<script>'+get_plotlyjs()+'</script><script>const beiReviewPayload='+json.dumps(payload, separators=(',', ':'), allow_nan=False)+';'+(FOLDER/'bei_breathing_view.js').read_text(encoding='utf-8')+'</script>'


if __name__ == '__main__':
    run()
