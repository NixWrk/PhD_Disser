"""Check actual physical CEM patch coverage by the nested boundary refinement.

Coverage of contact faces alone is not a bulk convergence certificate.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.io import loadmat
from georg_quality_comparison import read,sha


def patch_coverage(coarse_forward,coarse_mesh,parent_maps):
    forward=read(coarse_forward);mesh=loadmat(coarse_mesh);maps=loadmat(parent_maps)
    nodes=mesh['nodes_mm'];boundary=mesh['boundary'].astype(np.int64)-1
    areas=np.linalg.norm(np.cross(nodes[boundary[:,1]]-nodes[boundary[:,0]],
                                 nodes[boundary[:,2]]-nodes[boundary[:,0]]),axis=1)/2
    counts=np.bincount(maps['boundary_parent'].ravel().astype(np.int64)-1,minlength=len(boundary))
    if not np.all(counts>=1):raise ValueError('A parent boundary face is missing')
    rows=[]
    for record in forward['patches']:
        for index,face in enumerate(record['faces']):
            ids=np.atleast_1d(face).astype(np.int64)-1
            ratio=float(areas[ids[counts[ids]>1]].sum()/areas[ids].sum())
            rows.append(dict(record_id=record['record_id'],physical_contact=index+1,
                electrical_group=record['gids'][index],area_mm2=float(areas[ids].sum()),
                refined_boundary_area_fraction=ratio,all_faces_refined=bool(np.all(counts[ids]>1))))
    return dict(contacts=rows,all_actual_contact_faces_refined=all(r['all_faces_refined'] for r in rows),
        minimum_refined_boundary_area_fraction=min(r['refined_boundary_area_fraction'] for r in rows),
        scope='actual finite contact boundary faces; not bulk or whole search-window convergence',
        global_mesh_convergence_proven=False,coarse_forward_sha256=sha(coarse_forward),
        coarse_mesh_sha256=sha(coarse_mesh),parent_maps_sha256=sha(parent_maps),producer_sha256=sha(__file__))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('coarse-forward','coarse-mesh','parent-maps','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();result=patch_coverage(a.coarse_forward,a.coarse_mesh,a.parent_maps)
    if a.output.exists():raise ValueError('Use a fresh immutable coverage output')
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('all_actual_contact_faces_refined','minimum_refined_boundary_area_fraction')}))
