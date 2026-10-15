"""A lightweight display copy of the actual calculation boundary, never FEM."""
from pathlib import Path
import argparse
import numpy as np
from scipy.io import loadmat
import pymeshlab
from georg_quality_selected_refinement import read,sha,write_new


def prepare(model,out):
    model,out=Path(model).resolve(),Path(out).resolve()
    if out.exists():raise ValueError('Fresh immutable display output required')
    source=model/'mesh.mat';qcfile=model/'mesh_quality.local.json';qc=read(qcfile)
    if not qc['passed'] or qc['mesh_sha256']!=sha(source):raise ValueError('Mesh shape/identity failed')
    raw=loadmat(source,variable_names=['nodes_mm','boundary'])
    boundary=raw['boundary'].astype(np.int64)-1
    ids,inverse=np.unique(boundary.ravel(),return_inverse=True)
    mesh=pymeshlab.MeshSet()
    mesh.add_mesh(pymeshlab.Mesh(raw['nodes_mm'][ids],inverse.reshape(boundary.shape).astype(np.int32)))
    mesh.meshing_decimation_quadric_edge_collapse(targetfacenum=12000,preservenormal=True,preservetopology=True)
    result=mesh.current_mesh();vertices=result.vertex_matrix();faces=result.face_matrix()
    if (not np.isfinite(vertices).all() or faces.min()<0 or faces.max()>=len(vertices)):
        raise ValueError('Invalid display indexing')
    write_new(out,dict(schema='georg_calculation_boundary_display_v1',vertices_lps_mm=vertices.tolist(),
        faces_zero_based=faces.tolist(),calculation_mesh_sha256=qc['mesh_sha256'],
        original_boundary_faces=int(len(boundary)),display_boundary_faces=int(len(faces)),
        display_only=True,calculation_geometry_changed=False,display_distance_error_bound_established=False,
        FEM_was_run=False,sources=[dict(path=str(p),sha256=sha(p)) for p in (source,qcfile,Path(__file__))]))
    print(dict(original_faces=len(boundary),display_faces=len(faces)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();prepare(args.model,args.out)
