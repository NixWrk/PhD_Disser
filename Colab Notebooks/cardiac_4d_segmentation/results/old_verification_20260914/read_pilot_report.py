from pathlib import Path
import json
root=Path('Colab Notebooks/cardiac_4d_segmentation');work=root/'results/old_verification_20260914'
s=json.loads((work/'run_01/run_status.json').read_text(encoding='utf8'))
print({k:s.get(k) for k in ['status','stage','current_phase_percent','pilot_report_written','error']},'completed',len(s['completed_phase_ids']))
p=root/'21.08_Историческая_контрастная_КТ_объёмы_камер.ipynb'
if p.exists():
 import nbformat
 nb=nbformat.read(p,as_version=4)
 text='\n\n'.join(c.source for c in nb.cells)
 (work/'pilot_reader.md').write_text(text,encoding='utf8')
 print(text)
