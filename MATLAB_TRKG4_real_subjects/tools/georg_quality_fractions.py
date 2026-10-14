"""Checkpointed mask integration and exact coefficient inheritance."""
from pathlib import Path
import argparse
import shutil

import nibabel as nib
import numpy as np
from scipy.io import loadmat, savemat
from scipy.stats import qmc

from georg_ct_model import PIPE
from georg_lateral_quadrature import uniform_boxes
from georg_mesh_quality import volumes, inherited_field_energy_check
from georg_physical_contacts import prepare as physical_contacts
from scorpion_georg_fem_prepare import read, write, sha

SEG=PIPE/'output/exploratory/scorpion_georg_fem_20261008/segmentation'
MATERIALS=['soft_background','lung','bone','heart','fat']


def map_masks(base,points=(1024,4096),parent_base=None,parent_maps=None):
    base=Path(base);points=tuple(points)
    mesh=loadmat(base/'mesh.mat');nodes=mesh['nodes_mm'];tets=mesh['elems'].astype(np.int32)-1
    ct=nib.load(SEG/'static_ct.nii.gz');labels=np.zeros(ct.shape,np.uint8)
    sources=[]
    for name,label in [('fat',4),('heart',3),('bone',2),('lung',1)]:
        image=nib.load(SEG/(name+'.nii.gz'))
        if image.shape!=ct.shape or not np.array_equal(image.affine,ct.affine):raise ValueError('Mask grid mismatch')
        labels[np.asanyarray(image.dataobj)>0]=label;sources.append(dict(path=str((SEG/(name+'.nii.gz')).resolve()),sha256=sha(SEG/(name+'.nii.gz'))))
    transform=np.linalg.inv(ct.affine)@np.diag([-1.,-1.,1.,1.]);vox=nib.affines.apply_affine(transform,nodes)
    low=np.empty((len(tets),3),np.int32);high=low.copy()
    for start in range(0,len(tets),100000):
        p=vox[tets[start:start+100000]];low[start:start+len(p)]=np.floor(p.min(1)+.5).astype(np.int32)-1
        high[start:start+len(p)]=np.floor(p.max(1)+.5).astype(np.int32)+1
    print('Certifying material boxes',len(tets),flush=True)
    certified=uniform_boxes(labels,low,high);del low,high
    reused=np.zeros(len(tets),bool);parent_ids=None;parent_fractions={}
    if parent_base is not None:
        parent_base=Path(parent_base);maps=loadmat(parent_maps);parent_ids=maps['element_parent'].ravel().astype(np.int32)-1
        counts=np.bincount(parent_ids)
        reused=counts[parent_ids]==1
        previous=loadmat(parent_base/'mesh.mat')
        np.testing.assert_array_equal(tets[reused],previous['elems'][parent_ids[reused]]-1)
        np.testing.assert_array_equal(nodes[:len(previous['nodes_mm'])],previous['nodes_mm'])
        for q in points:parent_fractions[q]=loadmat(parent_base/f'fractions_q{q}.mat')['fractions']
        del previous
    mixed=np.flatnonzero((certified<0)&~reused)
    cuts=np.sort(qmc.Sobol(3,scramble=True,seed=20261008).random_base2(int(np.log2(max(points)))),axis=1)
    bary=np.diff(np.c_[np.zeros(len(cuts)),cuts,np.ones(len(cuts))],axis=1)
    identity=dict(mesh_sha256=sha(base/'mesh.mat'),mask_sources=sources,points=list(points),seed=20261008,
                  producer_sha256=sha(__file__),voxel_certifier_sha256=sha(PIPE/'tools/georg_lateral_quadrature.py'))
    if parent_base is not None:
        identity['unchanged_parent_reuse']=dict(parent_mesh_sha256=sha(parent_base/'mesh.mat'),
            parent_maps_sha256=sha(parent_maps),fraction_sources=[sha(parent_base/f'fractions_q{q}.mat') for q in points])
    checkpoint=base/'fraction_checkpoint.local.json';progress=0;arrays={}
    if checkpoint.exists():
        saved=read(checkpoint)
        if saved['identity']!=identity:raise ValueError('Material mapping checkpoint input changed')
        progress=saved['mixed_cells_completed']
    for q in points:
        path=base/f'fractions_work_q{q}.npy'
        arrays[q]=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.float64,shape=(len(tets),5))
        if progress==0:
            arrays[q][:]=0
            fixed=np.flatnonzero(certified>=0);arrays[q][fixed,certified[fixed]]=1
            if parent_base is not None:arrays[q][reused]=parent_fractions[q][parent_ids[reused]]
    print('Mixed material cells',len(mixed),'resume',progress,flush=True)
    for start in range(progress,len(mixed),256):
        ids=mixed[start:start+256]
        p=np.einsum('qj,bjk->bqk',bary,vox[tets[ids]],optimize=True)
        index=np.floor(p+.5).astype(np.int32);valid=np.all((index>=0)&(index<np.asarray(labels.shape)),axis=2)
        sampled=np.zeros(valid.shape,np.uint8);ijk=index[valid];sampled[valid]=labels[ijk[:,0],ijk[:,1],ijk[:,2]]
        for q,array in arrays.items():
            array[ids]=np.column_stack([(sampled[:,:q]==j).mean(1) for j in range(5)])
        stop=start+len(ids)
        if stop%10240==0 or stop==len(mixed):
            for array in arrays.values():array.flush()
            write(checkpoint,dict(identity=identity,mixed_cells_completed=stop,total_mixed_cells=len(mixed)))
            print('Material mapping',stop,'/',len(mixed),flush=True)
    v=volumes(nodes,tets);inventories=[]
    for q,array in arrays.items():
        if not np.array_equal(array.sum(1),np.ones(len(tets))) or np.any(array<0):raise ValueError('Invalid material fractions')
        savemat(base/f'fractions_q{q}.mat',dict(fractions=array),do_compression=False)
        inventories.append(dict(points=q,volume_ml=(v@array/1000).tolist()))
    write(base/'material_inventory.local.json',dict(materials=MATERIALS,inventory=inventories,
          homogeneous_cells=int(np.count_nonzero(certified>=0)),mixed_cells=len(mixed),
          unchanged_parent_elements_reused=int(reused.sum()),
          quadrature_is_nested=True,masks_priority=['lung','bone','heart','fat','background'],
          outside_CT_is_soft_background_assumption=True,identity=identity))


def plan(base,geometry,meshroot,points=(1024,4096),inherit=False):
    base,geometry,meshroot=map(Path,[base,geometry,meshroot])
    if not read(base/'mesh_quality.local.json')['passed']:raise ValueError('Shape QC failed before operator preparation')
    contactfile=geometry/'fem_contacts.local.json'
    if not contactfile.exists():physical_contacts(geometry,meshroot/'body_meshing.stl',contactfile)
    registry=PIPE/'configs/tissue_literature_v2.json';book=read(registry)
    refs=[book['tissues'][name]['gabriel_at_frequency']['100000']['rho_ohm_m'] for name in
          ['Muscle','Lung (Inflated)','Bone (Cortical)','Heart Muscle','Fat']]
    write(base/'reference_materials.local.json',dict(materials=MATERIALS,rho_ohm_m=refs,frequency_hz=100000,
          registry_sha256=sha(registry),soft_as_muscle_assumption=True,heart_as_myocardium_assumption=True,
          bone_as_cortical_assumption=True,individual_resistivities_measured=False))
    paths=[base/'mesh.mat',base/'mesh_quality.local.json',base/'reference_materials.local.json',contactfile,
           geometry/'contour_qc.local.json',geometry/'geometry.local.json',meshroot/'surface_qc.local.json',
           *(base/f'fractions_q{q}.mat' for q in points),*(SEG/(n+'.nii.gz') for n in ['static_ct','fat','heart','bone','lung'])]
    if inherit:paths += [base/'inheritance.local.json',meshroot/'fine/parent_maps.mat']
    qc=read(base/'mesh_quality.local.json')
    data=dict(schema='georg_shared_ct_fem_v1',subject_alias='georg',frequency_hz=100000,
          material_order=MATERIALS,numerical_quadrature_points=list(points),integration_rules=list(points),
          nodes=qc['nodes'],tetrahedra=qc['tetrahedra'],volume_ml=qc['volume_ml'],
          geometry_directory=str(geometry.resolve()),pose_reference_mesh=str((meshroot/'coarse/mesh.mat').resolve()),
          contacts_file=str(contactfile.resolve()),shape_QC_passed=True,shared_geometry_changed=True,
          fraction_origin='exact parent inheritance' if inherit else 'CT mask integration',
          exact_contact_parent_mesh=str((meshroot/'coarse/mesh.mat').resolve()),
          boundary_parent_map='' if sha(base/'mesh.mat')==sha(meshroot/'coarse/mesh.mat') else str((meshroot/'fine/parent_maps.mat').resolve()),
          constant_physical_contact_layout=True,
          contact_impedance_ohm_m2=2.5e-7*.05/(np.pi*.01**2/4),contact_impedance_is_measured=False,
          source_rule='own_drive',geometry_accuracy_is_independently_validated=False,
          quantitative_inversion_allowed=False,sources=[dict(path=str(p.resolve()),sha256=sha(p)) for p in paths],
          producer_sha256=sha(__file__))
    write(base/'plan.local.json',data)


def inherit(meshroot):
    meshroot=Path(meshroot);coarse=meshroot/'coarse';fine=meshroot/'fine';out=meshroot/'fine_inherited'
    out.mkdir(exist_ok=False)
    for name in ['mesh.mat','mesh_quality.local.json']:shutil.copy2(fine/name,out/name)
    maps=loadmat(fine/'parent_maps.mat');parent=maps['element_parent'].ravel().astype(np.int32)-1
    sources=[]
    for q in (1024,4096):
        src=coarse/f'fractions_q{q}.mat';fractions=loadmat(src)['fractions']
        savemat(out/f'fractions_q{q}.mat',dict(fractions=fractions[parent]),do_compression=False)
        sources.append(dict(points=q,source_sha256=sha(src)))
    c=loadmat(coarse/'mesh.mat');f=loadmat(fine/'mesh.mat');rho=read(coarse/'reference_materials.local.json')['rho_ohm_m']
    sigma=loadmat(coarse/'fractions_q4096.mat')['fractions']@(1/np.asarray(rho))
    energy=inherited_field_energy_check(c['nodes_mm'],c['elems'].astype(np.int32)-1,
        f['nodes_mm'],f['elems'].astype(np.int32)-1,maps['midpoint_edges'].astype(np.int32)-1,parent,sigma,
        maps['center_parent_ids'].ravel().astype(np.int32)-1,maps['center_barycentric_weights'])
    if not energy['passed']:raise ValueError('Parent-field energy changed under coefficient inheritance')
    write(out/'inheritance.local.json',dict(parent_fraction_sources=sources,parent_mesh_sha256=sha(coarse/'mesh.mat'),
          fine_mesh_sha256=sha(fine/'mesh.mat'),parent_maps_sha256=sha(fine/'parent_maps.mat'),energy_check=energy,
          no_tissue_resampling=True,no_geometry_change=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path)
    p.add_argument('--meshroot',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--stage',choices=['map','inherit','plan'],required=True);p.add_argument('--points',type=int,nargs=2,default=[1024,4096])
    p.add_argument('--reuse-parent',type=Path);p.add_argument('--parent-maps',type=Path)
    a=p.parse_args()
    if a.stage=='map':map_masks(a.base,a.points,a.reuse_parent,a.parent_maps);plan(a.base,a.geometry,a.meshroot,a.points)
    elif a.stage=='inherit':inherit(a.meshroot);plan(a.meshroot/'fine_inherited',a.geometry,a.meshroot,inherit=True)
    else:plan(a.base,a.geometry,a.meshroot,a.points,inherit=a.base.name=='fine_inherited')
