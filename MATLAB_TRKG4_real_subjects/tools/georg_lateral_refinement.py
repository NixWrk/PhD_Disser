"""Nested volume refinement for Georg lateral numerical convergence controls.

Preserve CT exterior and original vertices. No inverse fit is allowed here.
Local contact neighbourhoods do not establish whole-surface convergence.
"""
from pathlib import Path
import argparse
import time

import numpy as np
from scipy.io import loadmat, savemat

from georg_model_contract import resolve
from scorpion_georg_fem_prepare import sha, read, write, map_fractions
from georg_ct_model import PIPE
from georg_ct_fem_prepare import validate_tetrahedra
import refine_surface_contact_mesh as refine

refine.np=np


def volume_edges(nodes,tets,centres,radius,target):
    near=np.zeros(len(tets),bool)
    for start in range(0,len(tets),100000):
        points=nodes[tets[start:start+100000]]
        lo,hi=points.min(axis=1),points.max(axis=1)
        selected=np.zeros(len(points),bool)
        for centre in centres:
            distance=np.maximum(np.maximum(lo-centre,centre-hi),0)
            selected |= np.einsum('ij,ij->i',distance,distance)<=radius**2
        near[start:start+len(points)]=selected
    cells=tets[near]
    edges=np.unique(np.sort(np.vstack([cells[:,pair] for pair in refine.PAIRS]),axis=1),axis=0)
    return edges[np.linalg.norm(nodes[edges[:,0]]-nodes[edges[:,1]],axis=1)>target*(1+1e-12)]


def prepare(target=3.,radius=20.,out=None):
    model,current=resolve();base=PIPE/current['fem_directory']
    out=out or model/f'lateral_nested_h{target:g}_r{radius:g}'
    out.mkdir(parents=True,exist_ok=False)
    source=base/'mesh.mat';a=loadmat(source)
    nodes=a['nodes_mm'];tets=a['elems'].astype(np.int64)-1;boundary=a['boundary'].astype(np.int64)-1
    original_nodes=len(nodes)
    contacts=read(model/'fem_contacts.local.json')
    controls=[r for r in contacts['scenarios'] if r['id'] in ['reference_60_lateral','reference_140_lateral']]
    centres=np.unique(np.vstack([np.vstack(r['groups_mm'][4:]) for r in controls]),axis=0)
    volume=refine.volumes(nodes,tets).sum();history=[];start=time.perf_counter()
    for iteration in range(16):
        edges=volume_edges(nodes,tets,centres,radius,target)
        if not len(edges):break
        nn=len(nodes)
        tets,_=refine.split_simplices(tets,edges,nn)
        # Volume-only edges may leave every boundary triangle unchanged.
        # The historical surface helper expects at least one affected face.
        keys=np.sort(edges[:,0]*nn+edges[:,1])
        hit=False
        for i,j in [(0,1),(0,2),(1,2)]:
            key=np.minimum(boundary[:,i],boundary[:,j])*nn+np.maximum(boundary[:,i],boundary[:,j])
            indices=np.searchsorted(keys,key);valid=indices<len(keys)
            if np.any(keys[indices[valid]]==key[valid]):hit=True;break
        if hit:boundary,_=refine.split_simplices(boundary,edges,nn)
        nodes=np.vstack([nodes,(nodes[edges[:,0]]+nodes[edges[:,1]])/2])
        row=dict(round=iteration+1,edges=len(edges),nodes=len(nodes),tetrahedra=len(tets),boundary=len(boundary))
        history.append(row);print(row,flush=True)
    else:raise ValueError('Volume refinement did not reach its declared edge target')
    volumes=refine.volumes(nodes,tets);flip=volumes<0;tets[flip]=tets[flip][:,[1,0,2,3]]
    volumes=refine.volumes(nodes,tets);tiny=validate_tetrahedra(nodes,tets,volumes)
    if abs(volumes.sum()/volume-1)>1e-10:raise ValueError('Volume changed')
    np.testing.assert_array_equal(nodes[:original_nodes],a['nodes_mm'])
    refine.verify_boundary(tets,boundary)
    savemat(out/'mesh.mat',dict(nodes_mm=nodes,elems=tets+1,boundary=boundary+1),do_compression=False)
    seg=PIPE/current['segmentation_directory'];geometry=map_fractions(nodes,tets,seg,out)
    write(out/'fem_contacts.local.json',dict(contacts,scenarios=controls))
    write(out/'reference_materials.local.json',read(base/'reference_materials.local.json'))
    old=read(base/'plan.local.json')
    sources=[out/'mesh.mat',out/'fractions_q256.mat',out/'fractions_q1024.mat',
        out/'fem_contacts.local.json',out/'reference_materials.local.json',source,
        model/'body_embedded_arms_370mm.stl',*(seg/(s+'.nii.gz') for s in ['static_ct','heart','lung','bone','fat'])]
    plan=dict(old,nodes=len(nodes),tetrahedra=len(tets),geometry=geometry,
        contacts_file=str((out/'fem_contacts.local.json').resolve()),
        sources=[dict(path=str(p.resolve()),sha256=sha(p)) for p in sources],
        code_sha256=sha(__file__),interior_target_mm=target,
        min_tetrahedron_volume_mm3=float(volumes.min()),tiny_tetrahedra_qc=tiny,
        refinement_history=history,refinement_scope='contact neighbourhood volume, not all candidate poses',
        volume_neighbourhood_radius_mm=radius,original_vertices_preserved=True,
        quantitative_inversion_allowed=False,shared_geometry_changed=False,
        seconds=time.perf_counter()-start)
    write(out/'plan.local.json',plan);print('Prepared nested control',out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',type=float,default=3.)
    p.add_argument('--radius',type=float,default=20.)
    p.add_argument('--out',type=Path);a=p.parse_args()
    if a.target<=0 or a.radius<=0:p.error('Positive lengths required')
    prepare(a.target,a.radius,a.out)


if __name__=='__main__':main()
