from pathlib import Path
root=Path('Colab Notebooks/cardiac_4d_segmentation')
for name in ['analyze_historical_cardiac_sequence.py','run_historical_cardiac_sequence.py']:
 p=root/name;s=p.read_text(encoding='utf-8-sig')
 s=s.replace("sequence/'conversion_manifest.json'","sequence/'phases'/'conversion_manifest.json'").replace("seq/'conversion_manifest.json'","seq/'phases'/'conversion_manifest.json'")
 if name.startswith('analyze'):
  s=s.replace("ct=nib.load(ct_path)","ct_hash=sha(ct_path)\n        if ct_hash!=p['sha256']: raise ValueError('CT hash differs from conversion manifest')\n        ct=nib.load(ct_path)")
  s=s.replace("'ct_sha256':sha(ct_path)","'ct_sha256':ct_hash")
 else:
  s=s.replace("phases=sorted(m['phases'],key=lambda x:x['phase_id']);percent", "phases=sorted(m['phases'],key=lambda x:x['phase_id'])\n    for phase in phases:\n        if digest(seq/'phases'/phase['nifti'])!=phase['sha256']:raise RuntimeError('Converted CT changed')\n    percent")
 p.write_text(s,encoding='utf8',newline='\n')
print('Adapter contract integrated; converted CT hashes are verified before use.')
