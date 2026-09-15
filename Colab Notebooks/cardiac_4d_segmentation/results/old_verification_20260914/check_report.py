from pathlib import Path
import sys,json,numpy as np,nibabel as nib
root=Path('Colab Notebooks/cardiac_4d_segmentation').resolve();sys.path.insert(0,str(root))
from analyze_historical_cardiac_sequence import analyze,notebook,MASKS,sha
work=root/'results/old_verification_20260914/synthetic_report_check';seq=work/'sequence';maskdir=work/'masks'
(seq/'phases').mkdir(parents=True,exist_ok=True)
ps=[]
for i,percent in enumerate([0,25,50,75]):
 pid=f'phase_{i:02}';ps.append({'phase_id':pid,'phase_percent':percent})
 ct=nib.Nifti1Image(np.zeros((8,9,10),np.int16),np.diag([2,3,4,1]));ct.header.set_xyzt_units('mm');nib.save(ct,seq/'phases'/f'{pid}.nii.gz')
 a=np.zeros(ct.shape,np.uint8);a[2:4+i%2,2:5,3:6]=1
 for rel,_ in MASKS.values():
  p=maskdir/pid/rel;p.parent.mkdir(parents=True,exist_ok=True);nib.save(nib.Nifti1Image(a,ct.affine,ct.header),p)

for phase in ps: phase['sha256']=sha(seq/'phases'/f"{phase['phase_id']}.nii.gz")
(seq/'phases'/'conversion_manifest.json').write_text(json.dumps({'case_id':'synthetic','sequence_id':'test','phases':ps}),encoding='utf8')
r=analyze(seq,maskdir,work/'analysis');notebook(r,work/'synthetic_report.ipynb')
assert r['completed_phases']==4 and not r['accuracy_evaluated']
assert abs(r['rows'][5]['delta_from_previous_ml']-0.216)<1e-12
from bs4 import BeautifulSoup
soup=BeautifulSoup((work/'synthetic_report.html').read_text(encoding='utf8'),'html.parser')
assert not soup.select('.jp-CodeCell .jp-InputArea,.jp-InputPrompt,.jp-OutputPrompt')
assert '\\varepsilon_' in soup.get_text()
assert len(soup.select('img[src^="data:image/png"]'))==1
(work/'checks.json').write_text(json.dumps({'synthetic_only':True,'known_volume_delta_ml':0.216,'delta_check':'passed','html_no_code':'passed','formula_preserved':'passed'}),encoding='utf8')
print('Synthetic sequence: volume delta, report generation, HTML formula/no-code checks passed.')
