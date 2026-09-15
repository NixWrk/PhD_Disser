"""Refine transported search domains with target CT intensity.
OLD_REFERENCE_ROOT selects local OLD dataset; existing receipts are preserved.
"""
from pathlib import Path
import json,os,hashlib,sys
import numpy as np,SimpleITK as sitk
from scipy import ndimage
common=Path(os.environ['OLD_REFERENCE_ROOT']).resolve();root=common/'whole_heart_dynamic_hu_v01';root.mkdir(exist_ok=True);os.chdir(common)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
io=lambda p:os.path.relpath(p)
sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(4)
domain_path=common/'whole_chamber_review_v01/phase00_search_domains_v01/chamber_domains.nii.gz';domain=sitk.ReadImage(io(domain_path),sitk.sitkUInt8)
completed=[]
for receipt_path in sorted((common/'whole_heart_dynamic_v01').glob('phase_*/receipt.json')):
 r=json.loads(receipt_path.read_text(encoding='utf-8'));pid=r['phase_id'];out=root/pid
 if (out/'receipt.json').exists():completed.append(pid);continue
 out.mkdir(exist_ok=True);ct_path=common/'sequence_02/phases'/(pid+'.nii.gz');ct=sitk.ReadImage(io(ct_path),sitk.sitkFloat32);a=sitk.GetArrayFromImage(ct)
 transform=sitk.Transform() if pid=='phase_00' else sitk.ReadTransform(io(receipt_path.parent/'target_to_reference.h5'))
 warped=sitk.Resample(domain,ct,transform,sitk.sitkNearestNeighbor,0,sitk.sitkUInt8);labels=sitk.GetArrayFromImage(warped)
 sitk.WriteImage(warped,io(out/'transported_search_domains.nii.gz'),True)
 records={}
 for hu in [170,190,210]:
  for i,ch in enumerate(['LV','RV','LA','RA'],1):
   mask=(labels==i)&(a>=hu);cc,n=ndimage.label(mask);sizes=np.bincount(cc.ravel());sizes[0]=0
   if n:mask=cc==sizes.argmax()
   img=sitk.GetImageFromArray(mask.astype('uint8'));img.CopyInformation(ct);path=out/f'{ch}_HU{hu}.nii.gz';sitk.WriteImage(img,io(path),True)
   records[f'{ch}_HU{hu}']={'file':path.name,'volume_ml':float(mask.sum()*np.prod(ct.GetSpacing())/1000),'sha256':sha(path)}
 result={'phase_id':pid,'phase_percent':r['phase_percent'],'source_ct_sha256':r['source_ct_sha256'],'transport_receipt_sha256':sha(receipt_path),'source_domains_sha256':sha(domain_path),'method':'CT-only BSpline transport of authored search domains then target-phase HU filtering and largest6connectedcomponent','accepted':False,'whole_chamber_confirmed':False,'registration':r['registration'],'records':records,'automatic_comparator_used':False}
 (out/'receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');completed.append(pid);print(pid,'HU candidates saved',flush=True)
(root/'status.json').write_text(json.dumps({'status':'completed' if len(completed)==20 else 'partial','completed':completed,'accepted':False},indent=2),encoding='utf-8')
