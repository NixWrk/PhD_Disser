"""Measured montage comparisons and non-identifying ECG evidence, October 2026."""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lateral_pilot import prepare, EXPERIMENT, save, read, sha, candidate_windows, cycle_ensemble
from scorpion_protocol import channel_assignment, REVISION

OUT = EXPERIMENT / 'derived/exploratory/protocol_revision_20261012'
STATES = {'inhale': 'Глубокий вдох', 'exhale': 'Глубокий выдох'}
COLORS = ['#176aaf', '#d97706', '#8b5cf6', '#25866c', '#dc4460', '#495867',
          '#009fb7', '#946138', '#6d8a37', '#bd3aab', '#548bc3', '#d67791',
          '#3e776c', '#7d7dbe', '#ad8640', '#6b4e71', '#339299', '#586b22']


def short_name(record_id, role, configuration):
    rid = int(record_id[-3:])
    if role == 'ttrkg':
        label = configuration['variant'].replace('x', '×')
        if configuration['status'] != 'author_confirmed': label += '?'
        return f'{rid:03d} · ТТРКГ {label}'
    if role == 'tepc': return f'{rid:03d} · ТЕПС-{rid-10}'
    if role == 'aepc': return f'{rid:03d} · АЕПС-5'
    size = 140 if rid <= 6 else {7:120, 8:100, 9:80, 10:60, 11:60}[rid]
    axis = 'продольная' if role == 'lateral_horizontal' else 'поперечная'
    suffix = ' · крест' if rid in (6,11) else ''
    return f'{rid:03d} · {size} мм · {axis}{suffix}'


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    frame, ensembles, contract = prepare(EXPERIMENT)
    summary = read(EXPERIMENT / 'derived/scorpion_review/summary.local.json')
    annotations = read(EXPERIMENT / 'annotations/scorpion_review/annotations_candidates.local.json')
    rows, traces, ecg = [], [], []
    for record in summary['records']:
        rid = record['record_id']; fs = record['fs_hz_scenario']
        if fs != (400 if rid.endswith(('005','006')) else 1000): continue
        assignment = channel_assignment(rid)
        cache = EXPERIMENT / 'derived/scorpion_review' / (record['key']+'.npz')
        if sha(EXPERIMENT/record['source_relative_path'])!=record['source_sha256']:
            raise ValueError('Raw source changed: '+rid)
        annotation = next(a for a in annotations['records'] if a['record_id']==rid and a['fs_hz_scenario']==fs)
        windows = candidate_windows(annotation['breathing'], fs, record['rows'])
        with np.load(cache) as data:
            stride = max(1, round(fs/20))
            for channel in (1,2):
                role = assignment['physical_channel_roles'][str(channel)]
                label = short_name(rid, role, assignment['ttrkg'])
                selected = frame[(frame.record_id==rid)&(frame.channel==channel)]
                for row in selected.to_dict('records'):
                    start, end = (int(round(row[k]*fs)) for k in ('start_s','end_s'))
                    raw = data[f'bei_raw_{channel}'][start:end]
                    row.update(role=role, label=label, variant=assignment['ttrkg']['variant'] if role=='ttrkg' else None,
                        variant_status=assignment['ttrkg']['status'] if role=='ttrkg' else None,
                        raw_q25_ohm=float(np.quantile(raw,.25)), raw_q75_ohm=float(np.quantile(raw,.75)))
                    rows.append(row)
                if rid.endswith('001'):
                    # This short test contains no reconstructed deep breath holds.
                    save(OUT/'quiet_test.local.json', dict(record_id=rid, deep_hold_medians_available=False))
                traces.append(dict(record_id=rid, channel=channel, role=role, label=label,
                    time_s=data['time_s'][::stride].tolist(),
                    slow_ohm=data[f'bei_slow_{channel}'][::stride].tolist()))
                if role != 'ttrkg': continue
                peaks = data[f'qrs_samples_{record["selected_ecg_channel_candidate"]}'] if record['selected_ecg_channel_candidate'] else []
                for state,(start,end) in windows.items():
                    wave = data[f'ecg_filtered_{channel}']
                    ensemble = cycle_ensemble(wave, peaks, start, end, fs)
                    ensemble={key.replace('_ohm','_mV').replace('_mohm','_uV'):value
                        for key,value in ensemble.items()}
                    raw = data[f'ecg_raw_{channel}'][start:end]
                    segment = wave[start:end]
                    ecg.append(dict(record_id=rid,channel=channel,state=state,label=label,
                        variant_status=assignment['ttrkg']['status'], variant=assignment['ttrkg']['variant'],
                        filtered_p05_p95_mV=float(np.ptp(np.quantile(segment,[.05,.95]))),
                        raw_p05_p95_mV=float(np.ptp(np.quantile(raw,[.05,.95]))),
                        ensemble=ensemble))
    measured = pd.DataFrame(rows)
    measured.to_csv(OUT/'measured_levels.local.csv',index=False,encoding='utf-8-sig')
    save(OUT/'comparison_traces.local.json',traces)
    save(OUT/'ecg_evidence.local.json',ecg)
    save(OUT/'ensembles.local.json',ensembles)
    # Both normalized morphology and absolute ECG scale are supporting observations.
    # Contact, physiological state and parallel leads make the configuration non-unique.
    correlations = []
    for item in ecg:
        if not 7<=int(item['record_id'][-3:])<=10: continue
        wave = np.array(item['ensemble'].get('median_mV',[]))
        if not len(wave) or np.linalg.norm(wave-wave.mean())==0: continue
        for ref in ecg:
            if ref['record_id'][-3:] not in ('002','003','004') or ref['state']!=item['state']: continue
            other=np.array(ref['ensemble'].get('median_mV',[]))
            if not len(other) or np.linalg.norm(other-other.mean())==0: continue
            correlations.append(dict(record_id=item['record_id'],state=item['state'],reference_record=ref['record_id'],
                reference_variant=ref['variant'],pearson_without_shift=float(np.corrcoef(wave,other)[0,1]),
                filtered_scale_ratio=item['filtered_p05_p95_mV']/ref['filtered_p05_p95_mV']))
    pd.DataFrame(correlations).to_csv(OUT/'late_ecg_comparison.local.csv',index=False,encoding='utf-8-sig')
    save(OUT/'contract.local.json',dict(revision=REVISION,status='exploratory_hypothesis_not_validated',
        protocol_sha256=sha(Path(__file__).with_name('scorpion_protocol.py')),producer_sha256=sha(__file__),
        input_contract=contract,deep_hold_annotations_accepted=False,deep_hold_source='11.20 candidate intervals, unchanged',
        comparison_display_hz=20,median_source='all raw BEI samples in trimmed candidate hold windows',
        ecg_rule='filtered 0.5-40 Hz and 50/100/150 Hz notches; existing candidate QRS; 200 RR phases; no shifts, sign changes or gains',
        late_block_variant_identified=False,late_block_scenarios=['1x1','1x4'],
        sources=[dict(path=str(cache.relative_to(EXPERIMENT)),sha256=sha(cache)) for cache in sorted((EXPERIMENT/'derived/scorpion_review').glob('*.npz'))],
        outputs={name:sha(OUT/name) for name in ['measured_levels.local.csv','comparison_traces.local.json','ecg_evidence.local.json','ensembles.local.json','late_ecg_comparison.local.csv']}))
    return measured


def verify():
    c=read(OUT/'contract.local.json')
    if c['producer_sha256']!=sha(__file__) or c['protocol_sha256']!=sha(Path(__file__).with_name('scorpion_protocol.py')):
        raise ValueError('Comparison producer or protocol changed')
    for name,digest in c['outputs'].items():
        if sha(OUT/name)!=digest: raise ValueError('Comparison output changed: '+name)
    for item in c['sources']:
        if sha(EXPERIMENT/item['path'])!=item['sha256']: raise ValueError('Comparison signal cache changed')
    from lateral_pilot import ROOT
    c0=c['input_contract']
    for path,key in [(EXPERIMENT/'derived/scorpion_review/summary.local.json','review_summary_sha256'),
                     (EXPERIMENT/'annotations/scorpion_review/annotations_candidates.local.json','annotations_sha256')]:
        if sha(path)!=c0[key]:raise ValueError('Candidate inputs changed')
    for item in c0['provenance']:
        record=next(r for r in read(EXPERIMENT/'derived/scorpion_review/summary.local.json')['records']
                    if r['record_id']==item['record_id'] and r['fs_hz_scenario']==item['working_fs_hz'])
        if sha(EXPERIMENT/record['source_relative_path'])!=item['source_sha256']:raise ValueError('Raw source changed')
    return c


def comparison_figure(group):
    verify()
    allowed={'ttrkg'} if group=='ttrkg' else {'lateral_horizontal','lateral_vertical'}
    frame=pd.read_csv(OUT/'measured_levels.local.csv')
    frame=frame[frame.role.isin(allowed)]
    traces=[t for t in read(OUT/'comparison_traces.local.json') if t['role'] in allowed]
    fig=make_subplots(rows=1,cols=2,subplot_titles=('Базовый BEI во всех записях','Медианы общего BEI на задержках'),column_widths=[.58,.42])
    for i,t in enumerate(traces):
        color=COLORS[i%len(COLORS)]
        fig.add_trace(go.Scattergl(x=t['time_s'],y=t['slow_ohm'],name=t['label'],legendgroup=t['label'],line=dict(color=color)),row=1,col=1)
    labels=list(dict.fromkeys(frame.label))
    for state,color,symbol in [('inhale','#176aaf','circle'),('exhale','#d97706','diamond')]:
        values=frame[frame.state==state].copy()
        fig.add_trace(go.Scatter(x=values.label,y=values.baseline_raw_median_ohm,mode='markers',name=STATES[state],
            marker=dict(color=color,symbol=symbol,size=9),customdata=values[['start_s','end_s','raw_q25_ohm','raw_q75_ohm']].values,
            hovertemplate='%{x}<br>Медиана %{y:.5f} Ом<br>Окно %{customdata[0]:.2f}–%{customdata[1]:.2f} с<br>25–75% отсчётов: %{customdata[2]:.4f}–%{customdata[3]:.4f} Ом<extra>%{fullData.name}</extra>'),row=1,col=2)
    fig.update_xaxes(title_text='Время от начала каждой записи, с',row=1,col=1)
    fig.update_xaxes(categoryorder='array',categoryarray=labels,tickangle=-65,row=1,col=2)
    fig.update_yaxes(title_text='BEI, Ом')
    fig.update_layout(template='plotly_white',height=690,margin=dict(b=250),legend=dict(orientation='h',y=-.60),
        title='Сравнение ТТРКГ' if group=='ttrkg' else 'Сравнение боковых сборок',uirevision='scorpion-comparisons')
    return fig


def levels_table(group):
    verify();f=pd.read_csv(OUT/'measured_levels.local.csv')
    allowed={'ttrkg'} if group=='ttrkg' else {'lateral_horizontal','lateral_vertical'}
    f=f[f.role.isin(allowed)]
    return f.pivot(index='label',columns='state',values='baseline_raw_median_ohm').reindex(columns=['inhale','exhale']).rename(
        columns={'inhale':'Глубокий вдох, Ом','exhale':'Глубокий выдох, Ом'}).rename_axis('Запись и монтаж').reset_index()


def ecg_figure():
    verify();items=read(OUT/'ecg_evidence.local.json')
    fig=make_subplots(rows=1,cols=2,subplot_titles=('Медианные ЭКГ на вдохе: исходная шкала','Медианные ЭКГ на вдохе: форма'))
    for i,item in enumerate(t for t in items if t['state']=='inhale'):
        e=item['ensemble'];wave=np.array(e.get('median_mV',[]))
        if not len(wave):continue
        color=COLORS[i%len(COLORS)]
        fig.add_trace(go.Scatter(x=e['phase_rr'],y=wave,name=item['label'],legendgroup=item['label'],line=dict(color=color)),row=1,col=1)
        scale=np.max(np.abs(wave))
        if scale:
            fig.add_trace(go.Scatter(x=e['phase_rr'],y=wave/scale,name=item['label'],legendgroup=item['label'],showlegend=False,line=dict(color=color)),row=1,col=2)
    fig.update_xaxes(title_text='Фаза кандидатного интервала QRS–QRS')
    fig.update_yaxes(title_text='ЭКГ, мВ',row=1,col=1);fig.update_yaxes(title_text='ЭКГ / максимум модуля',row=1,col=2)
    fig.update_layout(template='plotly_white',height=650,margin=dict(b=160),legend=dict(orientation='h',y=-.3))
    return fig


if __name__=='__main__':
    f=run()
    print(f[['record_id','role','state','baseline_raw_median_ohm']].to_string(index=False))
