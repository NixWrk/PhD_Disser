"""Resolve finite contacts for all lateral source controls on the common CT.

This prepares an illustrative centre, not a recovered electrode position.
Unchanged tetrahedra retain exactly their existing quadrature arrays. New
children are sampled again in the same masks by the canonical Sobol rule.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import trimesh
from scipy.io import loadmat, savemat

from georg_model_contract import resolve
from georg_ct_model import PIPE
from georg_lateral_inverse_prepare import montage_rows, pose_points
from georg_ct_fem_prepare import validate_tetrahedra
from scorpion_georg_fem_prepare import read, write, sha, map_fractions, MATERIALS
import refine_surface_contact_mesh as refine
refine.np=np


def control_montages(pose, contacts):
    arms={r['id'].removeprefix('TTRKG_'):r['groups_mm'] for r in contacts['scenarios'] if r['family']=='TTRKG'}
    arrays={(r['size_mm'],r['axis']):r['centres_lps_mm'] for r in pose['electrodes']}
    rows=[]
    for r in montage_rows():
        if r['record_id'].endswith('001'):continue
        if r['is_cross']:
            groups=[[p] for p in arrays[r['size_mm'],'longitudinal']+arrays[r['size_mm'],'perpendicular']]
            first=1 if r['axis']=='longitudinal' else 5
        else:
            groups=arms[r['arm_montage']]+[[p] for p in arrays[r['size_mm'],r['axis']]]
            first=5
        rows.append(dict(id=r['record_id']+'_'+r['axis'],family='lateral_source_control',
            montage=str(r['size_mm']),recording=r['record_id'],perturbation='nominal',shift_mm=0.,
            groups_mm=groups,drive=[first,first+3],measure=[first+1,first+2],
            pose_is_measured=False,axis=r['axis'],is_cross=r['is_cross']))
    return rows


def prepare(base, out):
    model,current=resolve();base=base.resolve();out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    old=read(base/'plan.local.json')
    for s in old['sources']:
        if sha(Path(s['path']))!=s['sha256']:raise ValueError('Parent model changed')
    surface=trimesh.load(model/'body_ct_surface.stl',force='mesh')
    pose=pose_points(surface,0.,-210.)
    contacts=read(model/'fem_contacts.local.json');rows=control_montages(pose,contacts)
    centres=np.unique(np.vstack([np.vstack(r['groups_mm']) for r in rows]),axis=0)
    a=loadmat(base/'mesh.mat');nodes=a['nodes_mm'];original_nodes=nodes.copy()
    tets=a['elems'].astype(np.int64)-1;original=tets.copy();boundary=a['boundary'].astype(np.int64)-1
    parent=np.arange(len(tets));volume=refine.volumes(nodes,tets).sum();history=[]
    for iteration in range(16):
        edges=refine.selected_edges(nodes,boundary,centres,8.,.7)
        if not len(edges):break
        nn=len(nodes);tets,p=refine.split_simplices(tets,edges,nn);parent=parent[p]
        boundary,_=refine.split_simplices(boundary,edges,nn)
        nodes=np.vstack([nodes,(nodes[edges[:,0]]+nodes[edges[:,1]])/2])
        history.append(dict(round=iteration+1,edges=len(edges),nodes=len(nodes),tetrahedra=len(tets)))
        print(history[-1],flush=True)
    else:raise ValueError('Surface contact refinement incomplete')
    v=refine.volumes(nodes,tets);flip=v<0;tets[flip]=tets[flip][:,[1,0,2,3]];v=refine.volumes(nodes,tets)
    tiny=validate_tetrahedra(nodes,tets,v)
    np.testing.assert_array_equal(nodes[:len(original_nodes)],original_nodes)
    if abs(v.sum()/volume-1)>1e-10:raise ValueError('Volume changed')
    refine.verify_boundary(tets,boundary)
    changed=np.any(tets!=original[parent],axis=1);print('New quadrature cells',changed.sum(),flush=True)
    partial=out/'new_children';partial.mkdir()
    geometry=map_fractions(nodes,tets[changed],PIPE/current['segmentation_directory'],partial)
    inventory=[]
    for q in (256,1024):
        f=loadmat(base/f'fractions_q{q}.mat')['fractions'][parent]
        f[changed]=loadmat(partial/f'fractions_q{q}.mat')['fractions']
        assert np.array_equal(f.sum(1),np.ones(len(f)))
        savemat(out/f'fractions_q{q}.mat',dict(fractions=f),do_compression=False)
        for j,name in enumerate(MATERIALS):
            inventory.append(dict(quadrature_points=q,material=name,volume_ml=float(v@f[:,j]/1000),
                                  nonzero_elements=int(np.count_nonzero(f[:,j]))))
    pd.DataFrame(inventory).to_csv(out/'material_inventory.csv',index=False)
    savemat(out/'mesh.mat',dict(nodes_mm=nodes,elems=tets+1,boundary=boundary+1),do_compression=False)
    write(out/'fem_contacts.local.json',dict(contacts,scenarios=rows))
    write(out/'reference_materials.local.json',read(base/'reference_materials.local.json'))
    files=[out/'mesh.mat',out/'fractions_q256.mat',out/'fractions_q1024.mat',out/'fem_contacts.local.json',
           out/'reference_materials.local.json',base/'mesh.mat',base/'fractions_q256.mat',base/'fractions_q1024.mat',
           base/'plan.local.json',model/'body_embedded_arms_370mm.stl',model/'body_ct_surface.stl',
           PIPE/'tools/georg_lateral_inverse_prepare.py']
    files.extend(PIPE/current['segmentation_directory']/(s+'.nii.gz') for s in ['static_ct','heart','lung','bone','fat'])
    plan=dict(old,nodes=len(nodes),tetrahedra=len(tets),code_sha256=sha(Path(__file__)),
              contacts_file=str(out/'fem_contacts.local.json'),
              sources=[dict(path=str(p.resolve()),sha256=sha(p)) for p in files],
              parent_directory=str(base),surface_refinement_history=history,
              surface_contact_edge_mm=.7,surface_neighbourhood_radius_mm=8.,
              quadrature_recomputed_children=int(changed.sum()),quadrature_unchanged_parent_reused=True,
              new_children_geometry=geometry,min_tetrahedron_volume_mm3=float(v.min()),tiny_tetrahedra_qc=tiny,
              shared_geometry_changed=False,quantitative_inversion_allowed=False,
              scope='all lateral and cross forward source controls at illustrative y=0 z=-210 only')
    write(out/'plan.local.json',plan);print('Prepared all lateral source controls',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();prepare(a.base,a.out)
