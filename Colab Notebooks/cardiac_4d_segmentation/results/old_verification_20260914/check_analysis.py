from pathlib import Path
p=Path('Colab Notebooks/cardiac_4d_segmentation/run_historical_cardiac_sequence.py')
s=p.read_text(encoding='utf-8-sig')
s=s.replace("if pid in state['completed_phase_ids']:continue", """if pid in state['completed_phase_ids']:
                for rel,expected in state['phase_hashes'][pid].items():
                    if digest(out/rel)!=expected:raise RuntimeError('Completed output changed; refusing reuse')
                continue
            phase_dir=out/'automatic_masks_v2'/pid
            if phase_dir.exists() and any(phase_dir.rglob('*.nii.gz')):
                raise RuntimeError('Unrecorded partial phase outputs exist; use a fresh run directory')""")
s=s.replace("state['completed_phase_ids'].append(pid);save()", """seg_manifest=out/'automatic_masks_v2'/f\"segmentation_manifest_gpu{cfg.get('gpu','0')}_p{i}_{i+1}.json\"
            sm=json.loads(seg_manifest.read_text(encoding='utf8'))
            if sm['backend_version']!=cfg['expected_totalsegmentator_version']:
                raise RuntimeError('Unexpected TotalSegmentator version')
            state.setdefault('phase_hashes',{})[pid]={str(f.relative_to(out)):digest(f) for f in [*phase_dir.rglob('*.nii.gz'),seg_manifest]}
            state['completed_phase_ids'].append(pid);save()""")
p.write_text(s,encoding='utf8',newline='\n')
# Meaningful numerical check of affine volume and center; no medical input.
import sys,tempfile,numpy as np,nibabel as nib
sys.path.insert(0,str(p.parent.resolve()))
from analyze_historical_cardiac_sequence import mask_summary
A=np.array([[0,-3,0,10],[2,0,0,20],[0,0,4,30],[0,0,0,1]],float)
a=np.zeros((6,7,8),np.uint8);a[1:3,2:5,3:5]=1
ct=nib.Nifti1Image(np.zeros(a.shape),A);ct.header.set_xyzt_units('mm')
m=nib.Nifti1Image(a,A);r=mask_summary(m,ct)
assert abs(r['volume_ml']-12*24/1000)<1e-12
assert np.allclose(r['center_ras_mm'],[1,23,44])
assert not r['touches_image_boundary'] and not r['empty']
bad=A.copy();bad[0,3]+=1
try:mask_summary(nib.Nifti1Image(a,bad),ct)
except ValueError:pass
else:raise AssertionError('Grid mismatch not rejected')
print('Numerical check passed: oblique affine volume/center and grid mismatch rejection')
