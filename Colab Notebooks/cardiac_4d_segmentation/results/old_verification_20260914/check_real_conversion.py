from pathlib import Path
import json,numpy as np,nibabel as nib,pydicom
root=Path('Colab Notebooks/cardiac_4d_segmentation/results/old_verification_20260914')
selection=json.loads((root/'selected_sequence.local.json').read_text(encoding='utf8'))
manifest=json.loads((root/'sequence_01/phases/conversion_manifest.json').read_text(encoding='utf8'))
checks=[]
for k in [0,5,10,15,19]:
 p=manifest['phases'][k];img=nib.load(root/'sequence_01/phases'/p['nifti']);a=np.asanyarray(img.dataobj)
 d=pydicom.dcmread(selection['phases'][k]['source_path']);raw=d.pixel_array;sh=d.SharedFunctionalGroupsSequence[0]
 for sl in [0,49,98]:
  original=p['source_order'][sl];fr=d.PerFrameFunctionalGroupsSequence[original]
  pos=np.array(fr.PlanePositionSequence[0].ImagePositionPatient,dtype=float)
  expected=pos*np.array([-1,-1,1]);actual=nib.affines.apply_affine(img.affine,[0,0,sl])
  assert np.allclose(expected,actual,atol=1e-4,rtol=0)
  trans=getattr(fr,'PixelValueTransformationSequence',getattr(sh,'PixelValueTransformationSequence',None))[0]
  hu=raw[original].astype(float)*float(trans.RescaleSlope)+float(trans.RescaleIntercept)
  assert np.array_equal(a[:,:,sl],hu.astype(np.float32))
 checks.append({'phase_percent':p['phase_percent'],'coordinate_planes_checked':3,'pixel_planes_checked':3,'result':'passed'})
report={'scope':'5 phases, first/middle/last spatial planes; all pixels in selected planes and physical plane origins','checks':checks,'image_visual_review':'not_performed','anatomical_accuracy':'not_evaluated','conversion_phases':20}
(root/'conversion_numeric_qc.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps({'phases_checked':5,'coordinate_planes':15,'HU_planes':15,'result':'passed'}))
