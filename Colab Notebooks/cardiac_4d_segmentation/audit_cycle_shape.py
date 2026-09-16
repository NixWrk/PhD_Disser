"""Shape descriptors and connected-component QC of the same frozen phase masks."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import nibabel as nib
from scipy.ndimage import label
from run_reduced_heart_geometry import load_crop, serial
from heart_reduced_geometry import mask_moments, _eigenframe


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def run(config,output):
    cfg=json.loads(Path(config).read_text(encoding='utf8'));root=Path(cfg['derived_subject'])
    geometry=json.loads((Path(cfg['output'])/'geometry.json').read_text(encoding='utf8'))
    manifest=json.loads((root/'phases/conversion_manifest.json').read_text(encoding='utf8'));rows=[]
    for source,state in zip(manifest['phases'],geometry['states']):
        maskpath=root/'products_v2'/state['id']/'whole_heart.nii.gz'
        assert sha(maskpath)==state['mask_sha256']
        mask,affine,provenance=load_crop(maskpath,source,nib.load(root/'phases'/source['nifti']))
        moments=mask_moments(mask,np.asarray(cfg['ras_mm_to_fem_mm'])@affine)
        assert np.isclose(moments['volume_ml'],state['mask_volume_ml'],rtol=1e-12)
        assert np.allclose(moments['centroid_mm'],state['center_fem_mm'],atol=1e-8)
        eigenvalues,axes,ambiguous=_eigenframe(moments['covariance_mm2'],1e-6)
        labels,count=label(mask,structure=np.ones((3,3,3),bool));sizes=np.bincount(labels.ravel())[1:]
        rows.append({'id':state['id'],'mask_sha256':state['mask_sha256'],'moments':moments,
            'principal_axes_columns_fem':axes,'principal_variances_mm2':eigenvalues,
            'orientation_degenerate':ambiguous,'connected_components_26':count,
            'largest_component_fraction':float(sizes.max()/sizes.sum()),'touches_ct_boundary':provenance['touches_source_image_boundary']})
    result={'schema':'cardiac_cycle_shape_qc_v1','rows':rows,'geometry_sha256':sha(Path(cfg['output'])/'geometry.json'),
        'implementation':[{'name':name,'sha256':sha(Path(__file__).parent/name)} for name in ('audit_cycle_shape.py','run_reduced_heart_geometry.py','heart_reduced_geometry.py')],
        'axis_convention':'descending eigenvalues; deterministic signs, det +1; no temporal rotation estimate',
        'physical_validation':False,'orientation_trajectory_validated':False}
    Path(output).write_text(json.dumps(serial(result),ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
    print('phases',len(rows),'max_components',max(r['connected_components_26'] for r in rows),'min_largest_fraction',min(r['largest_component_fraction'] for r in rows))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.config,args.output)
