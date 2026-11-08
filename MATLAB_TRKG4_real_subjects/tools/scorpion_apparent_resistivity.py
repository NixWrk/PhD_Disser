"""Prepare/recheck Scorpion inspiration levels; run the canonical uniform CT/CEM fit.

Primary files and historical FEM artifacts are read-only. Local plans contain
private source paths and must remain in the ignored output/exploratory tree.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import subprocess

ROOT=Path(__file__).resolve().parents[2]
SUBJECTS=('georg','nik','yaroslav')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_impedance(source,channel):
    import pandas as pd
    if channel not in (1,2):raise ValueError('A physical BEI channel 1 or 2 is required')
    # Export rows have a trailing tab beyond the 11-column header.
    # Pin all positions, as in the canonical Scorpion reader: otherwise pandas
    # can infer an index and shift BEI into a neighbouring phase column.
    column=f'BEI{channel}ohm_100kHz'
    return pd.read_csv(source,sep='\t',decimal=',',usecols=range(11))[column].to_numpy(float)


def prepare(model_root,out):
    import numpy as np
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    subjects=[];excluded=[]
    for alias in SUBJECTS:
        model=Path(model_root).resolve()/alias
        original=model/'electrical_inputs.local.json';d=read(original)
        if d['subject_alias']!=alias:raise ValueError('Subject alias mismatch')
        summary_path=Path(d['summary_source']['path'])
        if sha(summary_path)!=d['summary_source']['sha256']:raise ValueError('Summary hash mismatch')
        summary=read(summary_path);observations=[]
        sources={x['sha256']:Path(x['path']) for x in d['source_files'] if x['path'].lower().endswith('.txt')}
        for o in d['observations']:
            if o['role']!='lateral' or o['state']!='inhale':continue
            if o['status']=='excluded':
                excluded.append(dict(subject_alias=alias,record_id=o['record_id'],reason=o['exclusion_reason']));continue
            if o['status']!='eligible_conditional':raise ValueError('Unexpected observation status')
            source=sources[o['source_sha256']]
            if sha(source)!=o['source_sha256']:raise ValueError('Raw source hash mismatch')
            recs=[r for r in summary['records'] if r['record_id']==o['record_id']]
            fs=int(o['source_qc']['fs_hz_scenario']) if alias=='georg' else int(o['window']['annotation']['fs_hz'])
            rec=next(r for r in recs if r.get('fs_hz_scenario',r.get('fs'))==fs)
            if rec['source_sha256']!=o['source_sha256']:raise ValueError('Wrong record identity')
            if alias=='georg':
                bounds=[round(o['window'][k]*fs) for k in ('start_s','end_s')]
            else:bounds=o['window']['annotation']['inner_windows_samples_half_open']['inhale']
            v=read_impedance(source,o['channel'])
            start,end=map(int,bounds)
            if not 0<=start<end<=len(v):raise ValueError('Invalid inspiration window')
            segment=v[start:end]
            if not np.isfinite(segment).all():raise ValueError('Nonfinite impedance')
            median=float(np.median(segment))
            if abs(median-o['Z_ohm'])>1e-8:
                raise ValueError(f"Median does not reproduce snapshot {o['record_id']}: {median} vs {o['Z_ohm']}")
            match=re.fullmatch(r'L(\d+)_TT(.+)',o['case_id'])
            if not match:raise ValueError('Invalid montage case')
            q05,q95=map(float,np.quantile(segment,[.05,.95]))
            observations.append(dict(record_id=o['record_id'],case_id=o['case_id'],size_mm=int(match[1]),
                ttrkg_variant=match[2],channel=o['channel'],Z_ohm=median,
                raw_q05_ohm=q05,raw_q95_ohm=q95,window_samples_half_open=[start,end],
                fs_hz=fs,sampling_status=rec.get('timebase_status',summary.get('sampling',{})),
                source_path=str(source),source_sha256=o['source_sha256'],
                annotation_status=o['annotation_status'],manual_annotation_accepted=o['manual_annotation_accepted']))
            print(f"{o['record_id']} L={match[1]} mm Z={median:.8f} Ohm raw median verified",flush=True)
        if not observations:raise ValueError('No eligible inspiration levels')
        subjects.append(dict(subject_alias=alias,model_directory=str(model),
            input_snapshot_sha256=sha(original),summary_sha256=sha(summary_path),
            mesh_sha256=sha(model/'materials/mesh.mat'),cem_sha256=sha(model/'cem/cem_contacts.mat'),
            prepared_sha256=sha(model/'electrical/prepared_materials.mat'),observations=observations))
    plan=dict(schema='scorpion_uniform_rho_input_v1',subjects=subjects,excluded_observations=excluded,
        producer_sha256=sha(__file__),frequency_hz=100000,
        status='exploratory_hypothesis_not_validated',primary_files_modified=False,
        homogeneous_material_assumption='All five compartments share one positive rho; own CT geometry and finite fixed CEM contacts remain.',
        global_mesh_convergence_proven=False,breathing_annotation_accepted=False)
    (out/'prepare_worker_snapshot.py').write_bytes(Path(__file__).read_bytes())
    target=out/'input.local.json';target.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    return target


def run(plan,out,matlab,*,pulse_plan=None):
    plan=Path(plan).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError('Use a fresh FEM output directory')
    def quote(p):return str(p).replace("'","''")
    pulse_arg=f",'PulsePlan','{quote(Path(pulse_plan).resolve())}'" if pulse_plan else ''
    script=f"cd('{quote(ROOT/'MATLAB_TRKG4_real_subjects')}');run_pipeline('Profile','scorpion_apparent_resistivity','InputFile','{quote(plan)}','OutputDir','{quote(out)}'{pulse_arg});"
    log=out.parent/(out.name+'.matlab.log')
    if log.exists():raise FileExistsError('Use a fresh launch log')
    preferences=out.parent/(out.name+'.matlab_preferences');preferences.mkdir(exist_ok=False)
    environment=os.environ.copy();environment['MATLAB_PREFDIR']=str(preferences)
    if pulse_plan:
        temporary=out.parent/(out.name+'.matlab_temp');temporary.mkdir(exist_ok=False)
        environment.update(TEMP=str(temporary),TMP=str(temporary))
    with log.open('x',encoding='utf-8') as f:
        result=subprocess.run([str(matlab),'-batch',script],stdout=f,stderr=subprocess.STDOUT,
            env=environment,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if result.returncode:raise RuntimeError(f'MATLAB exited {result.returncode}; inspect {log}')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--model-root',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a=sub.add_parser('run');a.add_argument('--plan',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--matlab',type=Path,required=True)
    a.add_argument('--pulse-plan',type=Path)
    a=p.parse_args()
    if a.command=='prepare':print(prepare(a.model_root,a.out),flush=True)
    elif a.command=='run':run(a.plan,a.out,a.matlab,pulse_plan=a.pulse_plan)


if __name__=='__main__':main()
