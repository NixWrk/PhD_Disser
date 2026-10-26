"""Geometric diagnostics of an explicit mesher input; never repairs or accepts it."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import trimesh


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def statistics(values):
    return dict(zip(('minimum','q001','q01','q05','median','q95','maximum'),
        map(float,np.quantile(values,[0,.001,.01,.05,.5,.95,1]))))


def audit(path):
    mesh=trimesh.load(path,force='mesh')
    points=mesh.triangles
    sides=np.stack([points[:,1]-points[:,0],points[:,2]-points[:,1],points[:,0]-points[:,2]],axis=1)
    length=np.linalg.norm(sides,axis=2)
    cross=np.cross(sides[:,0],-sides[:,2]);twice_area=np.linalg.norm(cross,axis=1)
    angles=[]
    for i in range(3):
        a=-sides[:,i];b=sides[:,(i+1)%3]
        cosine=np.einsum('ij,ij->i',a,b)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1))
        angles.append(np.degrees(np.arccos(np.clip(cosine,-1,1))))
    min_angle=np.min(angles,axis=0)
    altitude=twice_area/length.max(axis=1)
    ratio=2*np.sqrt(3)*twice_area/np.sum(length**2,axis=1)
    fp32=np.asarray(mesh.vertices,dtype=np.float32).astype(float)
    q=fp32[mesh.faces]
    area32=np.linalg.norm(np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),axis=1)
    report=dict(source_sha256=sha(path),vertices=len(mesh.vertices),faces=len(mesh.faces),
        watertight=bool(mesh.is_watertight),positive_volume=bool(mesh.is_volume),euler=int(mesh.euler_number),
        volume_ml=float(mesh.volume/1000),surface_area_mm2=float(mesh.area),
        edge_mm=statistics(mesh.edges_unique_length),face_area_mm2=statistics(twice_area/2),
        triangle_min_angle_deg=statistics(min_angle),triangle_min_altitude_mm=statistics(altitude),
        triangle_mean_ratio=statistics(ratio),
        short_unique_edges={str(t):int(np.count_nonzero(mesh.edges_unique_length<t)) for t in (1e-5,1e-4,.001,.01,.1)},
        small_angle_faces={str(t):int(np.count_nonzero(min_angle<t)) for t in (.001,.01,.1,1,5)},
        zero_area_after_float32=int(np.count_nonzero(area32==0)),
        diagnostics_are_not_acceptance_thresholds=True,FEM_was_run=False)
    worst=np.argsort(ratio)[:40]
    report['worst_faces']=[dict(face=int(i),min_angle_deg=float(min_angle[i]),altitude_mm=float(altitude[i]),
        centroid_lps_mm=points[i].mean(axis=0).tolist()) for i in worst]
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--surface',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();result=audit(Path(a.surface));out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('vertices','faces','short_unique_edges','small_angle_faces')},ensure_ascii=False))
