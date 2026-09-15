"""Render an explicitly authored CT contour draft; never an accepted reference.

Contours use original NIfTI index coordinates: polygon points are [j, i] in
axial slice k. Intermediate envelopes interpolate signed 2D distances. Optional
HU filtering and largest-component selection are explicit assisted operations.
"""
from pathlib import Path
import argparse
import json
import re
from datetime import datetime, timezone
import numpy as np
import nibabel as nib
from scipy import ndimage
from matplotlib.path import Path as Polygon
from prepare_old_reference_workspace import sha256, binary_nifti, write_json


def checked_vertices(shape, vertices):
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('Finite [j, i] polygon vertices required')
    if len(points) > 1 and np.array_equal(points[0], points[-1]):
        points = points[:-1]
    if len(points) < 3 or len(np.unique(points, axis=0)) != len(points):
        raise ValueError('At least three distinct polygon vertices required')
    if np.any(points < 0) or np.any(points > np.array(shape[::-1]) - 1):
        raise ValueError('Polygon vertices must lie inside the source image')
    following = np.roll(points, -1, axis=0)
    area = np.sum(points[:, 0] * following[:, 1] - points[:, 1] * following[:, 0])
    if abs(area) < 1e-8:
        raise ValueError('Polygon has zero signed area')
    def cross(a, b, c):
        ab, ac = b-a, c-a
        return ab[0]*ac[1]-ab[1]*ac[0]
    for i, (a, b) in enumerate(zip(points, following)):
        for j in range(i+1, len(points)):
            if j == i+1 or (i == 0 and j == len(points)-1):
                continue
            c, d = points[j], following[j]
            if np.any(np.maximum(a, b) < np.minimum(c, d)) or np.any(np.maximum(c, d) < np.minimum(a, b)):
                continue
            if cross(a,b,c)*cross(a,b,d) <= 0 and cross(c,d,a)*cross(c,d,b) <= 0:
                raise ValueError('Polygon self-intersects')
    return points


def polygon_mask(shape, vertices):
    vertices = checked_vertices(shape, vertices)
    ii, jj = np.indices(shape)
    return Polygon(np.asarray(vertices,dtype=float)).contains_points(np.column_stack((jj.ravel(),ii.ravel()))).reshape(shape)


def render_envelope(shape, contours, spacing=(1., 1.)):
    if len(shape) != 3 or any(n <= 0 for n in shape):
        raise ValueError('Three-dimensional source grid required')
    if len(spacing) != 2 or not np.isfinite(spacing).all() or np.any(np.asarray(spacing) <= 0):
        raise ValueError('Positive finite in-plane spacing required')
    if len(contours)<2:
        raise ValueError('At least two independently specified anchor contours required')
    anchors=sorted((int(k),v) for k,v in contours.items())
    if len({k for k,_ in anchors})!=len(anchors) or anchors[0][0]<0 or anchors[-1][0]>=shape[2]:
        raise ValueError('Invalid anchor slice indices')
    if any(len(vertices)<3 for _,vertices in anchors):
        raise ValueError('Each contour needs at least three vertices')
    fields=[]
    for k,vertices in anchors:
        mask=polygon_mask(shape[:2],vertices)
        if not mask.any():
            raise ValueError('Anchor polygon contains no voxel centres')
        fields.append((k,ndimage.distance_transform_edt(mask, sampling=spacing)-ndimage.distance_transform_edt(~mask, sampling=spacing)))
    envelope=np.zeros(shape,dtype=bool)
    for (ka,a),(kb,b) in zip(fields[:-1],fields[1:]):
        for k in range(ka,kb+1):
            alpha=(k-ka)/(kb-ka)
            envelope[:,:,k]=((1-alpha)*a+alpha*b)>0
    return envelope


def build(spec_path):
    spec_path=Path(spec_path)
    spec=json.loads(spec_path.read_text(encoding='utf-8-sig'))
    reviewed = spec.get('images_actually_reviewed')
    reviewed_ok = reviewed is True or (isinstance(reviewed, list) and bool(reviewed)
        and all(isinstance(item, str) and item.strip() for item in reviewed))
    if not reviewed_ok or spec.get('reference_accepted') is not False:
        raise ValueError('Reviewed-image provenance required; this tool cannot accept references')
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', spec['structure']):
        raise ValueError('Structure must be a safe anatomical label, not a path')
    if not isinstance(spec.get('author'), str) or not spec['author'].strip():
        raise ValueError('Explicit draft author required')
    thresholds = spec['lower_hu_scenarios']
    if (not isinstance(thresholds, list) or not thresholds
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in thresholds)
        or len(set(thresholds)) != len(thresholds)):
        raise ValueError('Unique finite HU scenarios required')
    output=Path(spec['output_dir'])
    if output.exists():
        raise FileExistsError('Output directory must be new')
    ct_path=Path(spec['ct_path'])
    if sha256(ct_path)!=spec['ct_sha256']:
        raise ValueError('Source CT hash mismatch')
    image=nib.load(ct_path)
    if image.header.get_xyzt_units()[0]!='mm':
        raise ValueError('Explicit mm units required')
    linear=image.affine[:3,:3]
    spacing=np.linalg.norm(linear,axis=0)
    if (len(image.shape) != 3 or not np.isfinite(linear).all() or np.any(spacing <= 0)
        or not np.allclose((linear/spacing).T@(linear/spacing),np.eye(3),atol=1e-5)):
        raise ValueError('Finite orthogonal physical voxel axes required; shear is unsupported')
    uncertain_slices=spec.get('uncertain_slices',[])
    if any(isinstance(k,bool) or not isinstance(k,int) or k < 0 or k >= image.shape[2] for k in uncertain_slices):
        raise ValueError('Uncertain slices must be valid original integer indices')
    envelope=render_envelope(image.shape,spec['contours_ijk'],spacing=spacing[:2])
    data=np.asanyarray(image.dataobj)
    if not np.isfinite(data).all():
        raise ValueError('Nonfinite CT intensities')
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'authored_spec.local.json',spec)
    results={}
    for threshold in thresholds:
        candidate=envelope & (data>=threshold)
        components,count=ndimage.label(candidate,structure=ndimage.generate_binary_structure(3,1))
        if count:
            sizes=np.bincount(components.ravel());sizes[0]=0
            candidate=components==int(np.argmax(sizes))
        label=str(threshold).replace('.', 'p').replace('-', 'minus')
        path=output/f"{spec['structure']}_agent_HU{label}.nii.gz"
        binary_nifti(candidate,image,path)
        results[str(threshold)]={'file':path.name,'sha256':sha256(path),'voxel_count':int(np.count_nonzero(candidate)),
            'volume_ml':float(np.count_nonzero(candidate)*abs(np.linalg.det(image.affine[:3,:3]))/1000),
            'components_before_selection':int(count),'accepted':False}
    binary_nifti(envelope,image,output/'authored_envelope.nii.gz')
    uncertain=np.zeros(image.shape,dtype=bool)
    for k in spec.get('uncertain_slices',[]):
        uncertain[:,:,k]=envelope[:,:,k]
    binary_nifti(uncertain,image,output/'uncertain_basal_region.nii.gz')
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'spec_sha256':sha256(spec_path),'renderer_sha256':sha256(__file__),
        'source_ct_sha256':spec['ct_sha256'],'structure':spec['structure'],'phase_id':spec['phase_id'],
        'author':spec['author'],'status':'agent_assisted_draft_pending_anatomical_review','accepted':False,
        'contour_method':'authored_axial_anchors_and_physical_signed_distance_interpolation',
        'images_actually_reviewed':reviewed,'source_spec_file':'authored_spec.local.json',
        'intensity_method':'lower_HU_scenario_with_largest_6_connected_component',
        'automatic_comparator_used':False,'reference_volume_accuracy_established':False,
        'rules':spec['rules'],'uncertain_slices':spec.get('uncertain_slices',[]),'masks':results}
    write_json(output/'draft_provenance.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',required=True,type=Path)
    args=p.parse_args();build(args.spec)
