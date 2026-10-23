"""Independent comparison inputs for Nik, without Georg data or matrices."""
from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from audit_scorpion_session import read, sha, write
from session_20261022 import ROOT, EXPERIMENT, STATUS


def legacy_four(cycles, start, stop):
    candidates = []
    for i in range(len(cycles)-3):
        block = cycles[i:i+4]
        if not all(c['eligible_conservative'] and .45 <= c['rr_s'] <= 1.8 for c in block):
            continue
        if block[0]['start_r_s'] < start or block[-1]['stop_r_s'] > stop:
            continue
        if any(abs(a['stop_r_s']-b['start_r_s']) > 1e-9 for a,b in zip(block,block[1:])):
            continue
        candidates.append([c['start_r_s'] for c in block]+[block[-1]['stop_r_s']])
    return min(candidates, key=lambda v:abs(v[0]+v[-1]-start-stop)) if candidates else None


def prepare(experiment=EXPERIMENT):
    config_path = Path(os.environ.get('KALMYKOV_EXP02_CONFIG',
                       ROOT/'Colab Notebooks/config/exp02_paths.local.json'))
    config = read(config_path)
    derived = Path(config['derived_root'])/'exp02'
    manifest_path = derived/'qc/10.01_record_manifest.candidate.json'
    qc_path = derived/'qc/12.01_signal_qc.candidate.json'
    manifest, qc = read(manifest_path), read(qc_path)
    by_sha = {r['input_sha256']:r for r in qc['records']}
    levels, segments, provenance = [], [], []
    for r in manifest['records']:
        if r['subject_id'] != 'exp02_nik' or r.get('side_size_mm') not in (60,140) or not r['include']:
            continue
        digest = r['input_sha256']; size = r['side_size_mm']
        raw = Path(config['data_root'])/r['relative_path']
        bp = derived/'annotations/breathing'/(digest[:16]+'.json')
        ep = derived/'annotations/ecg'/(digest[:16]+'.json')
        breathing, ecg = read(bp), read(ep)
        if sha(raw)!=digest or any(a['input']['sha256']!=digest for a in (breathing,ecg)):
            raise ValueError('Legacy data/annotation mismatch')
        if any(a['qc']['status']!='accepted' for a in (breathing,ecg)):
            raise ValueError('Legacy annotation not accepted')
        if ecg['upstream_breathing']['sidecar_sha256']!=sha(bp):
            raise ValueError('Legacy ECG belongs to different breathing revision')
        q=by_sha[digest]; data=pd.read_csv(raw,encoding='utf-8-sig')
        t=data['TIME_s'].to_numpy(float); fs=1/np.median(np.diff(t))
        if not np.isclose(fs,200) or np.max(abs(np.diff(t)*fs-1))>1e-6:
            raise ValueError('Legacy time is not uniform 200 Hz')
        for role,c in [('ttrkg',1),('lateral',2)]:
            for state,mode in [('inhale','задержка_вдох'),('exhale','задержка_выдох')]:
                a,b=breathing['accepted_modes'][mode]; mask=(t>=a+.5)&(t<=b-.5)
                v=data[f'BASE_{c}_Ω'].to_numpy(float)[mask]
                cached=q['static_levels'][f'BASE{c}_{state}']
                if len(v)!=cached['sample_count'] or not np.isclose(np.median(v),cached['median'],rtol=0,atol=1e-10):
                    raise ValueError('Legacy level fails reconstruction')
                levels.append(dict(subject='nik',device='mgtu',role=role,size_mm=size,
                    record_id=r['record_id'],state=state,usable=bool(q['component_use'][f'BASE{c}']),
                    baseline_ohm=float(np.median(v)),q05_q95_ohm=np.quantile(v,[.05,.95]).tolist(),
                    start_s=a+.5,end_s=b-.5,annotation_status='accepted',variant='не установлена'))
            mode='задержка_вдох';a,b=breathing['accepted_modes'][mode]
            events=legacy_four([c for c in q['cycles'] if c['mode']==mode],a+.5,b-.5)
            if events is None or not q['component_use'][f'RHEO{c}']:
                continue
            if not all(np.any(np.isclose(ecg['rpeaks_s'],p,atol=1e-9)) for p in events):
                raise ValueError('Legacy cycle events not in accepted ECG')
            rheo=data[f'RHEO_{c}_mΩ'].to_numpy(float)
            filtered=sosfiltfilt(butter(4,[.7,12],fs=fs,btype='bandpass',output='sos'),rheo)
            mask=(t>=events[0]-1e-9)&(t<=events[-1]+1e-9)
            segments.append(dict(role=role,size_mm=size,record_id=r['record_id'],
                time_s=(t[mask]-events[0]).tolist(),pulse_mohm=filtered[mask].tolist(),
                reported_rheo_mohm=rheo[mask].tolist(),ecg_mV=(data['ECG_V'].to_numpy(float)[mask]*1000).tolist(),
                event_time_s=(np.array(events)-events[0]).tolist(),annotation_status='accepted'))
        provenance.append(dict(record_id=r['record_id'],raw_path=str(raw),sha256=digest,
            breathing_sha256=sha(bp),ecg_sha256=sha(ep),
            qc_record_id=q['record_id']))
    if sorted(set(x['size_mm'] for x in levels))!=[60,140]:
        raise ValueError('Two independent Nik size matches required')
    out=Path(experiment)/'derived/exploratory/session_3340'
    result=dict(status=STATUS,producer_sha256=sha(__file__),config_sha256=sha(config_path),
        manifest_sha256=sha(manifest_path),qc_sha256=sha(qc_path),
        common_independent_sizes_mm=[60,140],
        omitted_size_100_reason='100 мм МГТУ — исключённая копия записи 90 мм; независимым совпадением размеров не является.',
        yaroslav_status='В проверенном манифесте эксперимента МГТУ Ярослава нет.',
        previous_device_ttrkg_variant='не установлена',
        sign_gain_offset='Тождественное преобразование записанных единиц; подгонка шкалы не выполняется.',
        recordings_simultaneous=False,provenance=provenance,levels=levels,segments=segments)
    write(out/'equipment.local.json',result)
    print('Nik equipment comparison:',len(levels),'levels,',len(segments),'four-cycle segments')
    return result


if __name__=='__main__':
    prepare()
