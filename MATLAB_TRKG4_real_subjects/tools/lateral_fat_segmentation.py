"""Full static CT conversion and local TotalSegmentator v2 masks for lateral CEM.

Inputs live in an excluded local JSON configuration; no raw CT is uploaded.
The existing FEM surface and electrode positions are never changed here.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time

PIPE = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def convert(cfg):
    import numpy as np
    import pydicom
    import nibabel as nib
    module_dir = PIPE.parent/'Colab Notebooks/cardiac_4d_segmentation'
    sys.path.insert(0, str(module_dir))
    from cardiac4d import dicom_phase_to_nifti
    out = Path(cfg['output']); out.mkdir(parents=True, exist_ok=True)
    target = out/'static_ct.nii.gz'
    assert not target.exists(), 'Frozen CT export already exists'
    records = []
    tags = ['SeriesNumber','SeriesInstanceUID','FrameOfReferenceUID','ImageOrientationPatient',
            'ImagePositionPatient','PixelSpacing','Rows','Columns','SOPInstanceUID']
    for path in Path(cfg['dicom_directory']).rglob('*.dcm'):
        ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=tags)
        if int(ds.get('SeriesNumber', -1)) == cfg['series_number']:
            records.append((path, ds))
    assert len(records) >= 100
    assert len({str(d.SeriesInstanceUID) for _,d in records}) == 1
    frames = {str(d.FrameOfReferenceUID) for _,d in records}; assert len(frames)==1
    details = dicom_phase_to_nifti(records, target)
    image = nib.load(target)
    # The shared converter writes int16 HU; require the exact acquisition scaling used here.
    assert details['rescale_slope_values'] == [1.0]
    assert all(float(x).is_integer() for x in details['rescale_intercept_values'])
    details.update(series_number=cfg['series_number'], source_dicom_count=len(records),
        frame_sha256=hashlib.sha256(next(iter(frames)).encode()).hexdigest(),
        ct_sha256=sha(target), converter_sha256=sha(module_dir/'cardiac4d.py'),
        runner_sha256=sha(__file__), ras_to_fem_mm=np.diag([-1.,-1.,1.,1.]).tolist(),
        role='static CT for the fixed FEM anatomy; not a cardiac phase',
        hu_min=int(np.min(image.dataobj)), hu_max=int(np.max(image.dataobj)))
    write(out/'conversion.json', details)
    print('Static CT converted:',image.shape,details['voxel_spacing_mm'],flush=True)


def segment(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.get('gpu',0))
    import numpy as np
    import nibabel as nib
    from totalsegmentator.python_api import totalsegmentator
    from totalsegmentator.map_to_binary import class_map
    out=Path(cfg['output']);ct=out/'static_ct.nii.gz'
    conversion=json.loads((out/'conversion.json').read_text(encoding='utf-8'))
    assert sha(ct)==conversion['ct_sha256']
    lungs=[n for n in class_map['total'].values() if n.startswith('lung_') and 'lobe' in n]
    exact={'sacrum','humerus_left','humerus_right','scapula_left','scapula_right',
           'clavicula_left','clavicula_right','femur_left','femur_right','hip_left','hip_right','skull','sternum'}
    bones=[n for n in class_map['total'].values() if n.startswith(('rib_','vertebrae_')) or n in exact]
    groups={'lung':lungs,'bone':bones,'heart':['heart'],'fat':['subcutaneous_fat','torso_fat']}
    assert len(lungs)==5 and 'heart' in class_map['total'].values()
    start=time.monotonic()
    for task,classes in [('total',lungs+bones+['heart']),('tissue_types',groups['fat']+['skeletal_muscle'])]:
        folder=out/('raw_'+task);folder.mkdir(exist_ok=True)
        if not all((folder/(x+'.nii.gz')).exists() for x in classes):
            print('Segmenting',task,flush=True)
            totalsegmentator(ct,folder,task=task,roi_subset=classes if task=='total' else None,
                device='gpu',nr_thr_resamp=1,nr_thr_saving=1,quiet=True,no_derived_masks=True)
    reference=nib.load(ct);inventory={}
    for name,classes in groups.items():
        mask=np.zeros(reference.shape,dtype=bool);sources=[]
        for label in classes:
            path=out/('raw_tissue_types' if name=='fat' else 'raw_total')/(label+'.nii.gz')
            image=nib.load(path);assert image.shape==reference.shape
            np.testing.assert_allclose(image.affine,reference.affine,rtol=0,atol=1e-5)
            mask |= np.asanyarray(image.dataobj)>0
            sources.append({'file':path.relative_to(out).as_posix(),'sha256':sha(path)})
        assert mask.any(),name
        target=out/(name+'.nii.gz');im=nib.Nifti1Image(mask.astype(np.uint8),reference.affine)
        im.header.set_xyzt_units('mm');nib.save(im,target)
        inventory[name]={'classes':classes,'sha256':sha(target),'source_masks':sources,
            'voxels':int(mask.sum()),'volume_ml':float(mask.sum()*abs(np.linalg.det(reference.affine[:3,:3]))/1000)}
    write(out/'segmentation.json',{'complete':True,'backend':'TotalSegmentator',
        'version':importlib.metadata.version('TotalSegmentator'),'ct_sha256':sha(ct),
        'runner_sha256':sha(__file__),'products':inventory,'elapsed_seconds':time.monotonic()-start,
        'scope':'full static CT; unchanged original FEM boundary and montage; automatic masks not manually validated'})
    print('Segmentation complete:',{k:round(v['volume_ml'],2) for k,v in inventory.items()},flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['convert','segment'])
    ap.add_argument('--config',type=Path,required=True);args=ap.parse_args()
    cfg=json.loads(args.config.read_text(encoding='utf-8'))
    (convert if args.stage=='convert' else segment)(cfg)
