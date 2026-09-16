"""Read-only DICOM frame/affine and extra-cardiac surface alignment audit.

Distances describe agreement of segmentations; they are not an independently
measured registration error. No transform is optimized from the heart masks.
"""
from pathlib import Path
import argparse
import hashlib
import json
import heart_replacement_electrical_pilot as core


def run(config_path):
    cfg=core.read(config_path); runner=core.configure(Path(cfg['deps']),1)
    import numpy as np
    import nibabel as nib
    import pydicom
    import trimesh
    from scipy.ndimage import binary_erosion
    from scipy.spatial import cKDTree
    root=Path(cfg['dicom_directory']); series={}
    for path in root.rglob('*.dcm'):
        ds=pydicom.dcmread(path,stop_before_pixels=True,specific_tags=['SeriesNumber','FrameOfReferenceUID','ImageOrientationPatient','ImagePositionPatient'])
        number=int(ds.get('SeriesNumber',-1))
        if number not in (301,307):continue
        record=series.setdefault(number,{'count':0,'frame_hashes':set(),'orientations':set()})
        record['count']+=1
        uid=str(ds.get('FrameOfReferenceUID',''));assert uid
        record['frame_hashes'].add(hashlib.sha256(uid.encode()).hexdigest())
        record['orientations'].add(tuple(float(x) for x in ds.ImageOrientationPatient))
    assert set(series)=={301,307}
    assert len(series[301]['frame_hashes'])==1 and series[301]['frame_hashes']==series[307]['frame_hashes']
    result={'schema':'cardiac_coordinate_audit_v1','same_dicom_patient_frame':True,
      'dicom_series':{str(k):{'count':v['count'],'frame_hashes':sorted(v['frame_hashes']),'orientations':sorted(v['orientations'])} for k,v in series.items()},
      'transform_ras_mm_to_fem_mm':np.diag([-1.,-1.,1.,1.]).tolist(),'additional_registration':False,'inputs':[],
      'limitations':['FEM/STL axis agreement is supported by surface distances; STL contains no DICOM frame metadata.',
       'Distance combines segmentation, sampling, respiration and possible motion; not independent registration uncertainty.',
       'Static series 301 and cardiac series 307 share the patient frame; common origin alone would not establish anatomical alignment.']}
    source=Path(cfg['prepared_fem'])
    with runner.h5py.File(source) as f:
        nodes=np.asarray(f['fmdl_m/nodes']).T*1000; bd=np.asarray(f['fmdl_m/boundary']).astype(int)-1
    for name,file in [('bones','bones_in_scan_fov'),('body','body_in_scan_fov')]:
        path=Path(cfg['anatomy_products'])/(file+'.nii.gz');im=nib.load(path);m=np.asanyarray(im.dataobj)>0
        surf=m&~binary_erosion(m);surf[:,:,:8]=False;surf[:,:,-8:]=False
        idx=np.column_stack(np.where(surf));xyz=nib.affines.apply_affine(im.affine,idx)*[-1,-1,1]
        if name=='bones':
            mesh=trimesh.load(cfg['bones_stl'],process=False);v=mesh.vertices
            v=v[(v[:,2]>xyz[:,2].min()+5)&(v[:,2]<xyz[:,2].max()-5)]
            v=np.unique(v,axis=0)[::5]
        else:
            v=nodes[np.unique(bd)];v=v[(abs(v[:,0])<130)&(v[:,2]>-220)&(v[:,2]<-95)]
        distance=cKDTree(xyz).query(v)[0]
        result[name]={'sample_count':len(v),'distance_p50_p90_p95_mm':np.percentile(distance,[50,90,95]).tolist(),
                      'distance_direction':'static surface sample to nearest 4D boundary voxel centre; CT end caps excluded'}
        result['inputs'].append({'role':file,'sha256':core.sha(path)})
    result['inputs'] += [{'role':'prepared_fem','sha256':core.sha(source)},{'role':'static_bones_stl','sha256':core.sha(cfg['bones_stl'])}]
    result['code_sha256']=core.sha(__file__)
    core.write(Path(cfg['output']),result)
    print(json.dumps({k:result[k] for k in ('same_dicom_patient_frame','bones','body')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    run(parser.parse_args().config)
