from pathlib import Path
import json,re,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pydicom
code=Path('Colab Notebooks/cardiac_4d_segmentation').resolve();sys.path.insert(0,str(code))
from enhanced_cardiac_ct import read_phase,sha256_file
prior=code/'results/old_verification_20260914'
work=code/'results/old_reference_20260915/coverage_probes';work.mkdir(exist_ok=True)
inventory=json.loads((prior/'dicom_inventory.local.json').read_text(encoding='utf-8'))
summary=json.loads((prior/'header_summary.json').read_text(encoding='utf-8'))
plans=[]
for case in summary:
 for group in case['groups']:
  if group['objects']!=20 or group['thickness_mm']!=5 or group['phases']!=list(range(0,100,5)): continue
  original=next(x for x in inventory if x['case']==case['case'])
  files=[]
  for series in original['series']:
   if series['series_number']!=group['series']:continue
   for f in series['files']:
    path=Path(original['folder'])/f
    d=pydicom.dcmread(path,stop_before_pixels=True)
    match=re.search(r'(\d+)%',str(d.SeriesDescription))
    if match:files.append({'source_path':str(path),'phase_percent':int(match.group(1))})
  phase=next(v for v in files if v['phase_percent']==0)
  plans.append({'case_id':case['case'],'series_number':group['series'],'source_path':phase['source_path'],
                'phases':sorted(files,key=lambda r:r['phase_percent'])})
(work/'probe_plan.local.json').write_text(json.dumps(plans,ensure_ascii=False,indent=2),encoding='utf-8')
rows=[]
for plan in plans:
 identity=f"{plan['case_id']}_series{plan['series_number']}"
 dest=work/identity;dest.mkdir(exist_ok=True)
 nifti=dest/'phase_00.nii.gz'
 if not nifti.exists():
  image,metadata=read_phase(Path(plan['source_path']))
  nib.save(image,nifti)
  metadata.update(phase_percent=0,phase_id='phase_00',nifti=nifti.name,sha256=sha256_file(nifti))
  (dest/'phase_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
 else:image=nib.load(nifti)
 array=np.asanyarray(image.dataobj)
 fig,axes=plt.subplots(1,4,figsize=(16,4.5),dpi=100)
 for ax,k in zip(axes,[0,2,5,10]):
  ax.imshow(array[:,:,k],cmap='gray',vmin=-150,vmax=550,interpolation='nearest')
  ax.set_title(f'{identity}: axial k={k}');ax.set_xticks([0,128,256,384,511]);ax.set_yticks([0,128,256,384,511])
 fig.tight_layout();fig.savefig(dest/'inferior_coverage.jpg',pil_kwargs={'quality':80});plt.close(fig)
 rows.append({'id':identity,'shape':list(image.shape),'nifti':str(nifti),'preview':str(dest/'inferior_coverage.jpg'),
               'first_slice_z_ras_mm':float(image.affine[2,3]),'anatomical_review':'pending'})
 (work/'probe_index.local.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
 print(identity,flush=True)
print('Completed '+str(len(rows))+' phase-zero coverage probes',flush=True)
