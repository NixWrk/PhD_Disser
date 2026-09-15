"""Run the declared historical cardiac sequence with resumable phase-level provenance."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,hashlib
from pathlib import Path
from datetime import datetime,timezone

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(4194304),b''):h.update(c)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',type=Path,required=True);a=ap.parse_args()
    cfg=json.loads(a.config.read_text(encoding='utf-8-sig'));root=Path(__file__).resolve().parent
    seq=Path(cfg['sequence_dir']);out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    conversion=seq/'phases'/'conversion_manifest.json';m=json.loads(conversion.read_text(encoding='utf8'))
    phases=sorted(m['phases'],key=lambda x:x['phase_id'])
    for phase in phases:
        if digest(seq/'phases'/phase['nifti'])!=phase['sha256']:raise RuntimeError('Converted CT changed')
    percent={p['phase_percent']:i for i,p in enumerate(phases)}
    pilot=[percent[k] for k in [0,25,50,75]];order=pilot+[i for i in range(len(phases)) if i not in pilot]
    signatures={'conversion':digest(conversion),'segmenter':digest(root/'segment_totalseg_v2.py'),'analyzer':digest(root/'analyze_historical_cardiac_sequence.py')}
    state_path=out/'run_status.json'
    if state_path.exists():
        state=json.loads(state_path.read_text(encoding='utf8'))
        if state['input_signatures']!=signatures:raise RuntimeError('Inputs/code changed; use a new run directory')
    else:state={'status':'prepared','completed_phase_ids':[],'input_signatures':signatures,'accuracy_evaluated':False,'reference_masks_available':False,'pid':os.getpid()}
    def save(**extra):
        state.update(extra);state['updated_at']=datetime.now(timezone.utc).isoformat();state['pid']=os.getpid()
        temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8');temp.replace(state_path)
    def run(cmd):
        subprocess.run([str(x) for x in cmd],check=True,env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONUNBUFFERED':'1'},cwd=root)
    def report(partial):
        cmd=[cfg['analysis_python'],'-B','-X','utf8',root/'analyze_historical_cardiac_sequence.py','--sequence-dir',seq,'--mask-dir',out/'automatic_masks_v2','--output-dir',out/'analysis','--notebook',cfg['notebook']]
        if partial:cmd.append('--allow-partial')
        run(cmd)
    try:
        save(status='running')
        for i in order:
            p=phases[i];pid=p['phase_id']
            if pid in state['completed_phase_ids']:
                for rel,expected in state['phase_hashes'][pid].items():
                    if digest(out/rel)!=expected:raise RuntimeError('Completed output changed; refusing reuse')
                continue
            phase_dir=out/'automatic_masks_v2'/pid
            if phase_dir.exists() and any(phase_dir.rglob('*.nii.gz')):
                raise RuntimeError('Unrecorded partial phase outputs exist; use a fresh run directory')
            save(stage='pilot_segmentation' if i in pilot else 'full_sequence_segmentation',current_phase_id=pid,current_phase_percent=p['phase_percent'])
            run([cfg['segmentation_python'],'-B','-X','utf8',root/'segment_totalseg_v2.py','--input-dir',seq/'phases','--output-dir',out/'automatic_masks_v2','--gpu',cfg.get('gpu','0'),'--phase-start',i,'--phase-stop',i+1,'--quiet'])
            seg_manifest=out/'automatic_masks_v2'/f"segmentation_manifest_gpu{cfg.get('gpu','0')}_p{i}_{i+1}.json"
            sm=json.loads(seg_manifest.read_text(encoding='utf8'))
            if sm['backend_version']!=cfg['expected_totalsegmentator_version']:
                raise RuntimeError('Unexpected TotalSegmentator version')
            state.setdefault('phase_hashes',{})[pid]={str(f.relative_to(out)):digest(f) for f in [*phase_dir.rglob('*.nii.gz'),seg_manifest]}
            state['completed_phase_ids'].append(pid);save()
            if all(phases[j]['phase_id'] in state['completed_phase_ids'] for j in pilot) and not state.get('pilot_report_written'):
                report(True);s=json.loads((out/'analysis/automatic_sequence_summary.json').read_text(encoding='utf8'))
                if any(r['empty'] for r in s['rows']):raise RuntimeError('Pilot contains an empty target mask; inspect before full sequence')
                save(pilot_report_written=True)
        report(False);save(status='completed',stage='automatic_volume_analysis_complete',current_phase_id=None)
    except Exception as e:
        save(status='failed',error=f'{type(e).__name__}: {e}');raise
if __name__=='__main__':main()
