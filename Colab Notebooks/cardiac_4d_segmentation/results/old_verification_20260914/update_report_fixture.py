from pathlib import Path
p=Path('Colab Notebooks/cardiac_4d_segmentation/results/old_verification_20260914/check_report.py')
s=p.read_text(encoding='utf-8-sig')
s=s.replace("from analyze_historical_cardiac_sequence import analyze,notebook,MASKS","from analyze_historical_cardiac_sequence import analyze,notebook,MASKS,sha")
s=s.replace("(seq/'conversion_manifest.json').write_text", "\nfor phase in ps: phase['sha256']=sha(seq/'phases'/f\"{phase['phase_id']}.nii.gz\")\n(seq/'phases'/'conversion_manifest.json').write_text")
p.write_text(s,encoding='utf8')
