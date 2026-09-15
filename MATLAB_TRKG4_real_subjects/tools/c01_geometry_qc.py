"""Geometric diagnostics of actual tetrahedral masks at C01; no electrical fitting."""
import json
import numpy as np
import pandas as pd
import vtk
from vtk.util.numpy_support import numpy_to_vtk,numpy_to_vtkIdTypeArray
from scipy.io import loadmat
from c01_baseline_study import SOURCE,OUT,sha,write

def main():
    g=loadmat(SOURCE/'geometry_input.mat',simplify_cells=True);m=loadmat(OUT/'geometry_masks.mat',simplify_cells=True)
    nodes=g['nodes_mm'];elems=np.asarray(g['elems'],np.int64)-1;faces=np.asarray(g['boundary_faces'],np.int64)-1
    original=loadmat(SOURCE/'geometry_masks.mat',simplify_cells=True)['masks'][:,:3].astype(bool)
    masks=np.column_stack([original,m['masks'].astype(bool)]);names=np.r_[['reference','uniform_transverse','variable_transverse'],m['model_names']]
    n=len(nodes);codes=[];owners=[]
    for ids in [[0,1,2],[0,1,3],[0,2,3],[1,2,3]]:
        f=np.sort(elems[:,ids],axis=1);codes.append((f[:,0]*n+f[:,1])*n+f[:,2]);owners.append(np.arange(len(elems)))
    codes=np.concatenate(codes);owners=np.concatenate(owners);order=np.argsort(codes);codes=codes[order];owners=owners[order]
    sf=np.sort(faces,axis=1);keys=(sf[:,0]*n+sf[:,1])*n+sf[:,2];pos=np.searchsorted(codes,keys)
    np.testing.assert_array_equal(codes[pos],keys);face_owner=owners[pos]
    assert np.all(np.searchsorted(codes,keys,'right')-pos==1),'Body boundary has nonunique volume owner'
    areas=np.linalg.norm(np.cross(nodes[faces[:,1]]-nodes[faces[:,0]],nodes[faces[:,2]]-nodes[faces[:,0]]),axis=1)/2
    pts=vtk.vtkPoints();pts.SetData(numpy_to_vtk(np.ascontiguousarray(nodes),deep=True));ca=vtk.vtkCellArray()
    ca.SetData(numpy_to_vtkIdTypeArray(np.arange(0,4*len(elems)+1,4,dtype=np.int64),deep=True),numpy_to_vtkIdTypeArray(elems.ravel(),deep=True))
    grid=vtk.vtkUnstructuredGrid();grid.SetPoints(pts);grid.SetCells(vtk.VTK_TETRA,ca)
    locator=vtk.vtkStaticCellLocator();locator.SetDataSet(grid);locator.BuildLocator()
    contacts=pd.read_csv(SOURCE/'contacts.csv');rows=[];ray_rows=[];origin=g['centre'];direction=g['frame'][:,2]
    samples=np.arange(.125,400.,.25)
    origins=[('centre',origin)]+[(str(i),nodes[int(row.point_node)-1]) for i,row in enumerate(contacts.itertuples())]
    for key,p in origins:
        ids=np.array([locator.FindCell(p+d*direction) for d in samples]);inside=ids>=0
        for j,name in enumerate(names):
            material=np.zeros(len(ids),bool);material[inside]=masks[ids[inside],j]
            crossings=samples[np.r_[material[0],material[1:]!=material[:-1]]]
            ray_rows.append(dict(ray=key,model=str(name),first_lung_mm=float(samples[material][0]) if np.any(material) else np.nan,crossings_mm=json.dumps(crossings.tolist()),sampling_step_mm=.25))
    rays=pd.DataFrame(ray_rows);rays.to_csv(OUT/'mask_ray_qc.csv',index=False)
    for j,name in enumerate(names):
        touching=masks[face_owner,j]
        contact_lung=0
        for node in contacts.point_node:
            inc=np.any(elems==int(node)-1,axis=1);contact_lung+=bool(np.any(masks[inc,j]))
        first=rays[(rays.model==name)&(rays.ray=='centre')].first_lung_mm.iloc[0]
        rows.append(dict(model=str(name),skin_lung_area_mm2=float(areas@touching),contacts_adjacent_to_lung=contact_lung,
                         central_discrete_entry_d_mm=first,sampling_step_mm=.25,
                         classification='diagnostic_lung_reaches_skin' if np.any(touching) else 'no_lung_on_skin'))
    pd.DataFrame(rows).to_csv(OUT/'geometry_mask_qc.csv',index=False)
    write(OUT/'geometry_qc_identity.json',dict(geometry_sha256=sha(OUT/'geometry.json'),masks_sha256=sha(OUT/'geometry_masks.mat'),source_sha256=sha(__file__),
        inputs=['same C01 body boundary and tetrahedra','same36 original PEM nodes'],interpretation='Surface contact of artificial lung is a model defect, not CT failure; all variants retained as diagnostics'))
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
