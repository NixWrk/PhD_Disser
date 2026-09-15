from pathlib import Path
import json,re,sys,hashlib
import pydicom
import nibabel as nib
code=Path('Colab Notebooks/cardiac_4d_segmentation').resolve();sys.path.insert(0,str(code))
from enhanced_cardiac_ct import read_phase,sha256_file
root=code/'results/old_reference_20260915'
inventory=json.loads((code/'results/old_verification_20260914/dicom_inventory.local.json').read_text(encoding='utf-8'))
case=next(x for x in inventory if x['case']=='old_01')
selected=json.loads((root/'selected_sequence02.local.json').read_text(encoding='utf-8'))
coarse=pydicom.dcmread(selected['phases'][0]['source_path'],stop_before_pixels=True)
for series in case['series']:
    if series['series_number']!='10':continue
    for name in series['files']:
        source=Path(case['folder'])/name
        ds=pydicom.dcmread(source,stop_before_pixels=True)
        m=re.search(r'(\d+)%',str(ds.SeriesDescription))
        if not m or int(m.group(1))!=0:continue
        dest=root/'thin_phase00_series10';dest.mkdir(exist_ok=True)
        checks={tag+'_equal':getattr(ds,tag,None)==getattr(coarse,tag,None) for tag in ['FrameOfReferenceUID','StudyInstanceUID','AcquisitionDateTime','AcquisitionNumber','ImageOrientationPatient']}
        checks.update({'thin_series_number':'10','coarse_series_number':'12','same_nominal_phase_percent':True,'not_registration_or_motion_equivalence':True})
        (dest/'metadata_comparison.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
        print(json.dumps(checks),flush=True)
        ct_path=dest/'phase_00.nii.gz'
        if not ct_path.exists():
            ct,meta=read_phase(source);nib.save(ct,ct_path)
            meta.update(phase_id='phase_00',phase_percent=0,nifti=ct_path.name,sha256=sha256_file(ct_path))
            (dest/'phase_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
        print('Thin phase zero converted for independent anatomical inspection',flush=True)
