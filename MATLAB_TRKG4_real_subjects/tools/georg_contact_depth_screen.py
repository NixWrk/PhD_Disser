"""Read-only CT ray profiles at actual lateral contacts; no effective h claim."""
from pathlib import Path
import argparse,json
import nibabel as nib
import numpy as np
import trimesh
from scipy.io import loadmat
from georg_quality_comparison import read,sha
from georg_quality_selected_refinement import candidate_row


def screen(candidate_file,parent,out,max_depth=80.,step=.5):
    candidate_file,parent,out=map(Path,(candidate_file,parent,out))
    c=candidate_row(read(candidate_file));plan=read(parent/'plan.local.json')
    images={}
    for name in ('static_ct','lung','bone','heart','fat'):
        found=[r for r in plan['sources'] if Path(r['path']).name==name+'.nii.gz']
        if len(found)!=1 or sha(found[0]['path'])!=found[0]['sha256']:
            raise ValueError('CT or material mask identity changed')
        images[name]=nib.load(found[0]['path'])
    ct=images['static_ct'];labels=np.zeros(ct.shape,np.uint8)
    for name,label in [('fat',4),('heart',3),('bone',2),('lung',1)]:
        image=images[name]
        if image.shape!=ct.shape or not np.array_equal(image.affine,ct.affine):
            raise ValueError('CT and tissue grids differ')
        labels[np.asanyarray(image.dataobj)>0]=label
    meshfile=Path(plan['pose_reference_mesh']);m=loadmat(meshfile,variable_names=['nodes_mm','boundary'])
    surface=trimesh.Trimesh(m['nodes_mm'],m['boundary']-1,process=False);surface.fix_normals()
    if not surface.is_volume:raise ValueError('A consistently oriented closed surface is required')
    transform=np.linalg.inv(ct.affine)@np.diag([-1.,-1.,1.,1.])
    depths=np.arange(0,max_depth+step/2,step);rows=[]
    for block in c['pose']['electrodes']:
        for label,xyz in zip(('I+','U+','U-','I-'),block['centres_lps_mm']):
            nearest,distance,triangle=surface.nearest.on_surface([xyz])
            inward=-surface.face_normals[triangle[0]]
            points=nearest[0]+depths[:,None]*inward
            vox=np.floor(nib.affines.apply_affine(transform,points)+.5).astype(int)
            valid=np.all((vox>=0)&(vox<np.array(ct.shape)),axis=1)
            tissue=np.full(len(depths),-1,dtype=int)
            tissue[valid]=labels[tuple(vox[valid].T)]
            hit=np.flatnonzero(tissue==1)
            rows.append(dict(size_mm=block['size_mm'],axis=block['axis'],contact=label,
                source_centre_lps_mm=xyz,ray_origin_lps_mm=nearest[0].tolist(),
                projection_distance_mm=float(distance[0]),inward_unit_xyz=inward.tolist(),
                depth_mm=depths.tolist(),tissue_label=tissue.tolist(),
                first_lung_depth_mm=float(depths[hit[0]]) if len(hit) else None,
                outside_CT_samples=int((~valid).sum())))
    result=dict(schema='georg_contact_CT_ray_profiles_v1',contacts=rows,
        max_ray_depth_mm=max_depth,step_mm=step,
        label_order=['soft_background','lung','bone','whole_heart','fat'],
        negative_label_means_outside_CT=True,
        scope='first lung-mask sample along each local inward surface normal',
        effective_h_not_defined=True,segmentation_independently_validated=False,
        candidate_sha256=sha(candidate_file),model_plan_sha256=sha(parent/'plan.local.json'),
        pose_reference_mesh_sha256=sha(meshfile),producer_sha256=sha(__file__))
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    finite=[r['first_lung_depth_mm'] for r in rows if r['first_lung_depth_mm'] is not None]
    print(dict(contacts=len(rows),lung_found=len(finite),depth_min_mm=min(finite) if finite else None,
               depth_max_mm=max(finite) if finite else None))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','parent','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();screen(a.candidate,a.parent,a.out)
