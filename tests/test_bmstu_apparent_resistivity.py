"""Проверки принятой разметки, BASE2 и независимого ряда МГТУ."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import pytest

spec=importlib.util.spec_from_file_location('bmstu',Path(__file__).resolve().parents[1]/'MATLAB_TRKG4_real_subjects/tools/bmstu_apparent_resistivity.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d),encoding='utf-8')


@pytest.fixture
def series(tmp_path):
    data=tmp_path/'raw';derived=tmp_path/'derived';observed=[];provenance=[]
    for alias,extra in [('georg',0),('nik',1000)]:
        for size in range(50,141,10):
            if alias=='nik' and size==100:continue
            z=extra+size;relative=f'{alias}/{size}{alias}.csv';source=data/relative;source.parent.mkdir(parents=True,exist_ok=True)
            rows=['TIME_s,BASE_1_Ω,BASE_2_Ω,RHEO_2_mΩ']
            for i in range(21):
                t=i*.25;value=z if 1.5<=t<=3.5 else -10000
                rows.append(f'{t},-99999,{value},77777')
            source.write_text('\n'.join(rows)+'\n',encoding='utf-8');h=worker.sha(source);rid=h[:16]
            a=dict(subject_id='exp02_'+alias,size_mm=size,qc={'status':'accepted'},
                input={'relative_path':relative,'sha256':h},accepted_modes={'задержка_вдох':[1,4]},
                candidate_modes={'задержка_вдох':[0,1]})
            ap=derived/f'exp02/annotations/breathing/{rid}.json';save(ap,a)
            observed.append(dict(subject_id='exp02_'+alias,size_mm=size,z_inhale_ohm=z))
            provenance.append(dict(subject_id='exp02_'+alias,size_mm=size,record_id=rid,
                breathing_sidecar_sha256=worker.sha(ap),source_csv_sha256=h))
    shutil.copyfile(data/'nik/90nik.csv',data/'nik/100nik.csv')
    snapshot=derived/'exp02/exploratory/33.06_side_arrays_exploratory.json'
    save(snapshot,dict(static_observations=observed,input_provenance=provenance))
    return data,derived,tmp_path/'model',tmp_path/'out'


def test_inspiration_uses_accepted_window_BASE2_and_excludes_only_known_duplicate(series):
    plan=worker.read(worker.collect(*series))
    g,n=plan['subjects'];assert len(g['observations'])==10 and len(n['observations'])==9
    assert {o['size_mm'] for o in g['observations']}==set(range(50,141,10))
    assert 100 not in {o['size_mm'] for o in n['observations']}
    for s in plan['subjects']:
        for o in s['observations']:
            assert o['Z_ohm']==o['size_mm']+(1000 if s['subject_alias']=='nik' else 0)
            assert o['selected_window_s']==[1.5,3.5]
            assert o['channel']==2 and o['annotation_status']=='accepted'
    assert {o['size_mm'] for o in g['observations'] if o['channel1_anomaly_does_not_exclude_BASE2']}=={90,100}


def test_changed_raw_data_are_rejected(series):
    data,_,_,_=series
    with (data/'georg/50georg.csv').open('a',encoding='utf-8') as f:f.write('6,1,1,1\n')
    with pytest.raises(ValueError,match='не соответствует разметке'):worker.collect(*series)


def test_unaccepted_annotation_is_not_promoted(series):
    _,derived,_,_=series;snapshot=derived/'exp02/exploratory/33.06_side_arrays_exploratory.json'
    d=worker.read(snapshot);p=d['input_provenance'][0];ap=derived/f"exp02/annotations/breathing/{p['record_id']}.json"
    a=worker.read(ap);a['qc']['status']='pending_manual_review';save(ap,a)
    p['breathing_sidecar_sha256']=worker.sha(ap);save(snapshot,d)
    with pytest.raises(ValueError,match='Непринятая'):worker.collect(*series)
