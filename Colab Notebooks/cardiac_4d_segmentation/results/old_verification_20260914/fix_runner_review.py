from pathlib import Path
p=Path('Colab Notebooks/cardiac_4d_segmentation/run_historical_cardiac_sequence.py')
s=p.read_text(encoding='utf8')
s=s.replace("percent={p['phase_percent']:i for i,p in enumerate(phases)}", """expected_names=[p['nifti'] for p in phases]
    actual_names=[p.name for p in sorted((seq/'phases').glob('phase_*.nii.gz'))]
    if expected_names!=actual_names:raise RuntimeError('Phase files differ from the exact conversion manifest list')
    percent={p['phase_percent']:i for i,p in enumerate(phases)}""")
s=s.replace("signatures={'conversion':digest(conversion),", "signatures={'runner':digest(Path(__file__)), 'config':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),'conversion':digest(conversion),")
s=s.replace("    try:\n        save(status='running')", """    def check_pilot():
        report(True)
        summary=json.loads((out/'analysis/automatic_sequence_summary.json').read_text(encoding='utf8'))
        if any(row['empty'] for row in summary['rows']):raise RuntimeError('Empty target mask; inspect before continuing')
        save(pilot_report_written=True)
    pilot_checked=False
    try:
        save(status='running')""")
s=s.replace("            p=phases[i];pid=p['phase_id']", """            p=phases[i];pid=p['phase_id']
            if i not in pilot and not pilot_checked:
                if not all(phases[j]['phase_id'] in state['completed_phase_ids'] for j in pilot):raise RuntimeError('Pilot is incomplete')
                check_pilot();pilot_checked=True""")
s=s.replace("            if sm['backend_version']!=cfg['expected_totalsegmentator_version']:", """            expected_mask_names=['total_heart/heart.nii.gz']+['heartchambers_highres/'+name+'.nii.gz' for name in ['heart_myocardium','heart_atrium_left','heart_ventricle_left','heart_atrium_right','heart_ventricle_right','aorta','pulmonary_artery']]
            if not all((phase_dir/name).is_file() for name in expected_mask_names):raise RuntimeError('Incomplete phase output')
            if len(sm['runs'])!=1 or sm['runs'][0]['phase_id']!=pid:raise RuntimeError('Segmentation manifest phase mismatch')
            if not sm.get('model_files'):raise RuntimeError('Model weight provenance is empty')
            model_signature=hashlib.sha256(json.dumps({k:sm[k] for k in ['backend_version','torch','cuda_runtime','models','model_files']},sort_keys=True).encode()).hexdigest()
            if state.get('model_signature',model_signature)!=model_signature:raise RuntimeError('Model weights or environment changed between phases')
            state['model_signature']=model_signature
            if sm['backend_version']!=cfg['expected_totalsegmentator_version']:""")
start=s.index("            if all(phases[j]['phase_id'] in state['completed_phase_ids'] for j in pilot) and not state.get('pilot_report_written'):")
end=s.index("        report(False);save",start)
s=s[:start]+s[end:]
s=s.replace("        report(False);save(status='completed'", """        report(False)
        final_summary=json.loads((out/'analysis/automatic_sequence_summary.json').read_text(encoding='utf8'))
        if any(row['empty'] for row in final_summary['rows']):raise RuntimeError('Full sequence contains an empty target mask')
        save(status='completed'""")
p.write_text(s,encoding='utf8',newline='\n')
print('Fixed mandatory pilot gate, exact phase list, model consistency, runtime/config/runner fingerprints.')
