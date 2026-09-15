"""CT-only candidate transport. OLD_TRANSPORT_ROOT contains config.local.json.
Not an independent manual reference. Local completed receipts enable resume.
"""
from pathlib import Path
import json,sys,hashlib,os,time
from datetime import datetime,timezone
import numpy as np,SimpleITK as sitk
root=Path(os.environ['OLD_TRANSPORT_ROOT']).resolve();os.chdir(root.parent)
def io_path(p):return os.path.relpath(p)
sys.path.insert(0,str(Path(__file__).resolve().parent))
from temporal_registration import register,assert_same_geometry
sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(4)
s=json.loads((root/'config.local.json').read_text(encoding='utf-8'));seq=Path(s['sequence_dir'])
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for ch,p in s['reference_masks'].items():
 if sha(p)!=s['reference_mask_hashes'][ch]:raise ValueError('Reference mask changed')
ref=sitk.ReadImage(io_path(seq/(s['reference_phase']+'.nii.gz')),sitk.sitkFloat32)
ref_masks={ch:sitk.ReadImage(io_path(p),sitk.sitkUInt8) for ch,p in s['reference_masks'].items()}
for ch,m in ref_masks.items():assert_same_geometry(ref,m,ch)
def downsample(img):
 c=sitk.RegionOfInterest(img,s['roi_size'],s['roi_index']);spacing=s['registration_spacing_mm'];size=[int(np.ceil((n-1)*sp/new))+1 for n,sp,new in zip(c.GetSize(),c.GetSpacing(),spacing)]
 return sitk.Resample(c,size,sitk.Transform(),sitk.sitkLinear,c.GetOrigin(),spacing,c.GetDirection(),-1000,sitk.sitkFloat32)
refsmall=downsample(ref);state={'status':'running','pid':os.getpid(),'completed':[],'failed':[],'manual_reference':False,'config_sha256':sha(root/'config.local.json'),'code_sha256':sha(Path(__file__))}
status=root/'status.json'
def save():
 state['updated_utc']=datetime.now(timezone.utc).isoformat();tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2),encoding='utf-8');tmp.replace(status)
save()
for idx in s['phase_order']:
 pid=f'phase_{idx:02d}';out=root/pid
 if (out/'receipt.json').exists():
  state['completed'].append(pid);save();continue
 state['current']=pid;save();out.mkdir(exist_ok=True);start=time.time()
 fixed=sitk.ReadImage(io_path(seq/(pid+'.nii.gz')),sitk.sitkFloat32);assert_same_geometry(ref,fixed,pid)
 diag={};transform=sitk.Transform()
 if idx:
  fixedsmall=downsample(fixed)
  transform,diag=register(fixedsmall,refsmall,s['control_spacing_mm'],s['iterations'],s['sampling_fraction'],s['seed']+idx)
  field=sitk.TransformToDisplacementField(transform,sitk.sitkVectorFloat64,fixedsmall.GetSize(),fixedsmall.GetOrigin(),fixedsmall.GetSpacing(),fixedsmall.GetDirection())
  jac=sitk.DisplacementFieldJacobianDeterminant(field);j=np.asarray(sitk.GetArrayFromImage(jac));diag['jacobian_min']=float(j.min());diag['jacobian_nonpositive_fraction']=float((j<=0).mean())
  before=sitk.GetArrayFromImage(refsmall);target=sitk.GetArrayFromImage(fixedsmall);warped=sitk.GetArrayFromImage(sitk.Resample(refsmall,fixedsmall,transform,sitk.sitkLinear,-1000,sitk.sitkFloat32))
  tissue=(target>-150)&(target<700);diag['mae_before_hu']=float(np.mean(abs(before[tissue]-target[tissue])));diag['mae_after_hu']=float(np.mean(abs(warped[tissue]-target[tissue])))
  sitk.WriteTransform(transform,io_path(out/'target_to_reference.h5'))
  if diag['jacobian_nonpositive_fraction']>0:
   state['failed'].append({'phase':pid,'reason':'nonpositive_jacobian','diagnostics':diag});save();continue
 records={};union=None
 for ch,m in ref_masks.items():
  warped=sitk.Resample(m,fixed,transform,sitk.sitkNearestNeighbor,0,sitk.sitkUInt8);arr=sitk.GetArrayFromImage(warped)>0
  if ch in ['LV','RV','LA','RA']:
   if union is None:union=arr.astype('uint8')
   else:union+=arr.astype('uint8')
  path=out/(ch+'.nii.gz');sitk.WriteImage(warped,io_path(path),True);records[ch]={'file':path.name,'sha256':sha(path),'volume_ml':float(arr.sum()*np.prod(fixed.GetSpacing())/1000)}
 if (union>1).any():raise ValueError('Transport produced chamber overlap')
 receipt={'phase_id':pid,'phase_percent':idx*5,'source_ct_sha256':sha(seq/(pid+'.nii.gz')),'reference_ct_sha256':sha(seq/(s['reference_phase']+'.nii.gz')),'reference_mask_hashes':s['reference_mask_hashes'],'method':s['method'],'accepted':False,'whole_chamber_coverage':'candidate_not_confirmed','records':records,'registration':diag,'elapsed_s':time.time()-start,'automatic_comparator_used':False}
 (out/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8');state['completed'].append(pid);save();print(pid,'saved',round(time.time()-start,1),flush=True)
state['status']='completed' if len(state['completed'])==20 else 'partial_with_registration_failures';state['current']=None;save()
