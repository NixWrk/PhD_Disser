"""Resolve finite electrodes over the whole scanned right-torso search domain.

Conforming edge bisection preserves the CT polyhedron and arm joins exactly.
Both nested material rules are remapped for new children; unchanged cells
retain their previously verified fractions byte for byte.
"""
from pathlib import Path
import argparse
import shutil
import nibabel as nib
import numpy as np
import trimesh
from scipy.io import loadmat, savemat
from scipy.stats import qmc
from georg_ct_model import PIPE
from georg_model_contract import resolve
from georg_ct_fem_prepare import validate_tetrahedra
from georg_lateral_quadrature import uniform_boxes
from scorpion_georg_fem_prepare import read, write, sha
import refine_surface_contact_mesh as refine
refine.np=np


def search_edges(nodes, boundary, bounds, target):
    points=nodes[boundary];lo=points.min(1);hi=points.max(1)
    intersects=np.all((hi>=bounds[0]) & (lo<=bounds[1]),axis=1)
    # Flat closures at the cropped CT ends are numerical caps, not skin on
    # the ribs. Keep these faces in the volume but exclude contact refinement.
    caps=(hi[:,2]-lo[:,2]<1e-6)&((np.abs(lo[:,2]-bounds[0,2])<2)|(np.abs(hi[:,2]-bounds[1,2])<2))
    selected=intersects & (lo[:,0]<(bounds[0,0]+bounds[1,0])/2) & ~caps
    faces=boundary[selected]
    edges=np.unique(np.sort(np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[0,2]]]),axis=1),axis=0)
    return edges[np.linalg.norm(nodes[edges[:,0]]-nodes[edges[:,1]],axis=1)>target*(1+1e-12)]


def fractions(nodes,tets,seg):
    ct=nib.load(seg/'static_ct.nii.gz');labels=np.zeros(ct.shape,np.uint8)
    for name,label in [('fat',4),('heart',3),('bone',2),('lung',1)]:
        image=nib.load(seg/(name+'.nii.gz'));assert image.shape==ct.shape
        np.testing.assert_allclose(image.affine,ct.affine,atol=1e-5,rtol=0)
        labels[np.asanyarray(image.dataobj)>0]=label
    vox=nib.affines.apply_affine(np.linalg.inv(ct.affine)@np.diag([-1.,-1.,1.,1.]),nodes)
    low=np.empty((len(tets),3),np.int32);high=low.copy()
    for start in range(0,len(tets),100000):
        p=vox[tets[start:start+100000]]
        low[start:start+len(p)]=np.floor(p.min(1)+.5).astype(np.int32)-1
        high[start:start+len(p)]=np.floor(p.max(1)+.5).astype(np.int32)+1
    certified=uniform_boxes(labels,low,high);fixed=certified>=0;mixed=np.flatnonzero(~fixed)
    arrays={q:np.zeros((len(tets),5)) for q in (1024,4096)}
    for a in arrays.values():a[fixed]=np.eye(5)[certified[fixed]]
    cut=np.sort(qmc.Sobol(3,scramble=True,seed=20261008).random_base2(12),axis=1)
    bary=np.diff(np.c_[np.zeros(4096),cut,np.ones(4096)],axis=1)
    print('New cells: certified',int(fixed.sum()),'sampled',len(mixed),flush=True)
    for start in range(0,len(mixed),256):
        cells=mixed[start:start+256];xyz=np.einsum('qj,bjk->bqk',bary,vox[tets[cells]],optimize=True)
        index=np.floor(xyz+.5).astype(np.int32);valid=np.all((index>=0)&(index<np.array(labels.shape)),axis=2)
        sampled=np.zeros(valid.shape,np.uint8);ijk=index[valid];sampled[valid]=labels[ijk[:,0],ijk[:,1],ijk[:,2]]
        for q,a in arrays.items():a[cells]=np.column_stack([(sampled[:,:q]==j).mean(1) for j in range(5)])
        if start%10240==0:print('Search-mesh integration',start+len(cells),'/',len(mixed),flush=True)
    return arrays


def prepare(base,out,target=1.):
    model,current=resolve();base=base.resolve();out=out.resolve();parentplan=read(base/'plan.local.json')
    for item in parentplan['sources']:
        if sha(item['path'])!=item['sha256']:raise ValueError('Parent input changed')
    out.mkdir(parents=True,exist_ok=False)
    skin=trimesh.load(model/'body_ct_surface.stl',force='mesh');bounds=skin.bounds.copy()
    source=loadmat(base/'mesh.mat');nodes=source['nodes_mm'];original_nodes=nodes.copy()
    tets=source['elems'].astype(np.int64)-1;original=tets.copy();boundary=source['boundary'].astype(np.int64)-1
    parent=np.arange(len(tets));volume=refine.volumes(nodes,tets).sum();history=[]
    for iteration in range(18):
        edges=search_edges(nodes,boundary,bounds,target)
        if not len(edges):break
        nn=len(nodes);tets,p=refine.split_simplices(tets,edges,nn);parent=parent[p]
        boundary,_=refine.split_simplices(boundary,edges,nn)
        nodes=np.vstack([nodes,(nodes[edges[:,0]]+nodes[edges[:,1]])/2])
        row=dict(round=iteration+1,edges=len(edges),nodes=len(nodes),tetrahedra=len(tets),boundary=len(boundary))
        history.append(row);print(row,flush=True)
    else:raise ValueError('Whole search surface did not reach edge target')
    v=refine.volumes(nodes,tets);flip=v<0;tets[flip]=tets[flip][:,[1,0,2,3]];v=refine.volumes(nodes,tets)
    tiny=validate_tetrahedra(nodes,tets,v);np.testing.assert_array_equal(nodes[:len(original_nodes)],original_nodes)
    assert abs(v.sum()/volume-1)<1e-10
    print('Verifying conforming boundary',flush=True);refine.verify_boundary(tets,boundary)
    changed=np.any(tets!=original[parent],axis=1);print('New children',int(changed.sum()),flush=True)
    child=fractions(nodes,tets[changed],PIPE/current['segmentation_directory'])
    for q in (1024,4096):
        old=loadmat(base/f'fractions_q{q}.mat')['fractions'];a=old[parent];a[changed]=child[q]
        np.testing.assert_array_equal(a[~changed],old[parent[~changed]])
        assert np.array_equal(a.sum(1),np.ones(len(a)))
        savemat(out/f'fractions_q{q}.mat',dict(fractions=a),do_compression=False)
    savemat(out/'mesh.mat',dict(nodes_mm=nodes,elems=tets+1,boundary=boundary+1),do_compression=False)
    for name in ['fem_contacts.local.json','reference_materials.local.json']:shutil.copy2(base/name,out/name)
    files=[out/name for name in ['mesh.mat','fractions_q1024.mat','fractions_q4096.mat','fem_contacts.local.json','reference_materials.local.json']]
    files += [base/'plan.local.json',base/'mesh.mat',base/'fractions_q1024.mat',base/'fractions_q4096.mat',model/'body_ct_surface.stl',model/'body_embedded_arms_370mm.stl']
    files += [PIPE/current['segmentation_directory']/(s+'.nii.gz') for s in ['static_ct','heart','lung','bone','fat']]
    files += [PIPE/'tools/georg_lateral_quadrature.py',PIPE/'tools/refine_surface_contact_mesh.py']
    plan=dict(parentplan,parent_directory=str(base),contacts_file=str(out/'fem_contacts.local.json'),
        nodes=len(nodes),tetrahedra=len(tets),surface_refinement_history=history,
        search_surface_edge_target_mm=target,search_surface_scope='right scanned lateral wall, excluding flat artificial CT end caps; caps remain in the volume',
        code_sha256=sha(__file__),sources=[dict(path=str(p),sha256=sha(p)) for p in files],
        mesh_unchanged=False,shared_geometry_changed=False,all_tetrahedra_positive=True,
        conforming_boundary_verified=True,tiny_tetrahedra_qc=tiny,min_tetrahedron_volume_mm3=float(v.min()),
        numerical_quadrature_points=[1024,4096],unchanged_fractions_reused_exactly=True,
        newly_mapped_children=int(changed.sum()),quantitative_inversion_allowed=False,
        scope='whole right CT torso contact search; spatial convergence remains unproved')
    write(out/'plan.local.json',plan);print('Prepared whole right-torso search mesh',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--target',type=float,default=1.);a=p.parse_args();prepare(a.base,a.out,a.target)
