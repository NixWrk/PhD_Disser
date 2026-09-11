"""Conforming local edge bisection near surface contacts, preserving polyhedral geometry.

Every selected edge is bisected in ALL adjacent tetrahedra and boundary faces.
Existing vertices and tissue/group labels are preserved. No smoothing/projection.
"""
from pathlib import Path
import argparse,hashlib,itertools,json,sys
ROOT=Path(__file__).resolve().parents[1]
PAIRS=list(itertools.combinations(range(4),2))

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def volumes(nodes,tets):
    result=np.empty(len(tets))
    for off in range(0,len(tets),100000):
        p=nodes[tets[off:off+100000]]
        result[off:off+len(p)]=np.einsum('ij,ij->i',np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),p[:,3]-p[:,0])/6
    return result

def selected_edges(nodes,faces,centres,radius,target):
    from trimesh.triangles import closest_point
    tri=nodes[faces];lo=tri.min(1);hi=tri.max(1);near=np.zeros(len(faces),bool)
    for c in centres:
        d=np.maximum(np.maximum(lo-c,c-hi),0)
        cand=np.flatnonzero(np.einsum('ij,ij->i',d,d)<=radius**2)
        points=closest_point(tri[cand],np.broadcast_to(c,(len(cand),3)))
        near[cand[np.linalg.norm(points-c,axis=1)<=radius]]=True
    f=faces[near]
    e=np.sort(np.vstack([f[:,[0,1]],f[:,[0,2]],f[:,[1,2]]]),axis=1)
    e=np.unique(e,axis=0)
    return e[np.linalg.norm(nodes[e[:,0]]-nodes[e[:,1]],axis=1)>target*(1+1e-12)]

def split_simplices(simplices,edges,nn):
    """Return child simplices and parent index, using one global edge order."""
    keys=edges[:,0].astype(np.int64)*nn+edges[:,1]
    keys.sort()
    affected=np.zeros(len(simplices),bool)
    for a,b in itertools.combinations(range(simplices.shape[1]),2):
        x=np.minimum(simplices[:,a],simplices[:,b]).astype(np.int64)*nn+np.maximum(simplices[:,a],simplices[:,b])
        idx=np.searchsorted(keys,x);ok=idx<len(keys);hit=np.zeros(len(x),bool);hit[ok]=keys[idx[ok]]==x[ok];affected|=hit
    keep=np.flatnonzero(~affected);work=np.flatnonzero(affected)
    edge_map={tuple(map(int,e)):nn+i for i,e in enumerate(edges)}
    children=[];parents=[]
    for index in work:
        original=simplices[index].tolist();pieces=[original]
        present=sorted(e for e in itertools.combinations(sorted(original),2) if e in edge_map)
        for a,b in present:
            new=[];mid=edge_map[(a,b)]
            for piece in pieces:
                if a in piece and b in piece:
                    left=piece.copy();right=piece.copy();left[left.index(b)]=mid;right[right.index(a)]=mid;new.extend([left,right])
                else:new.append(piece)
            pieces=new
        children.extend(pieces);parents.extend([index]*len(pieces))
    return np.vstack([simplices[keep],np.asarray(children,dtype=simplices.dtype)]),np.concatenate([keep,np.asarray(parents,dtype=np.int64)])

def refine(nodes,tets,boundary,labels,groups,centres,radius=8.,target=1.,max_rounds=12):
    history=[]
    for round_ in range(max_rounds):
        edges=selected_edges(nodes,boundary,centres,radius,target)
        if not len(edges):return nodes,tets,boundary,labels,groups,history
        nn=len(nodes)
        tets,parents=split_simplices(tets,edges,nn)
        labels=labels[parents];groups=groups[parents]
        boundary,_=split_simplices(boundary,edges,nn)
        nodes=np.vstack([nodes,(nodes[edges[:,0]]+nodes[edges[:,1]])/2])
        history.append(dict(round=round_+1,edges=len(edges),nodes=len(nodes),tetrahedra=len(tets),boundary=len(boundary)))
        print(history[-1],flush=True)
    raise ValueError('Contact edges still too long after refinement limit')

def verify_boundary(tets,boundary):
    faces=np.concatenate([tets[:,ix] for ix in [[0,1,2],[0,1,3],[0,2,3],[1,2,3]]])
    faces.sort(axis=1)
    unique,counts=np.unique(faces,axis=0,return_counts=True)
    if not np.isin(counts,[1,2]).all():raise ValueError('Nonmanifold volume faces')
    outer=unique[counts==1]
    b=np.sort(boundary,axis=1);order=np.lexsort((b[:,2],b[:,1],b[:,0]));b=b[order]
    if not np.array_equal(outer,b):raise ValueError('Tetrahedral boundary mismatch / hanging faces')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',required=True,type=Path);p.add_argument('--deps',required=True,type=Path);p.add_argument('--arm-report',required=True,type=Path);a=p.parse_args()
    sys.path.insert(0,str(a.deps.resolve()))
    global np
    import numpy as np,h5py
    from scipy.io import savemat
    from prepare_surface_sensitivity import verify_manifest
    base=a.base.resolve();m=verify_manifest(base);out=base/'refinement';out.mkdir(exist_ok=False)
    source=base/m['source']['prepared_fem']
    ident=json.loads((base/m['source']['model_identity']).read_text())
    expected=next(x['sha256'] for x in ident['inputs'] if x['name']==a.arm_report.name)
    if sha(a.arm_report)!=expected:raise ValueError('Arm/body group provenance mismatch')
    arm=json.loads(a.arm_report.read_text())
    if isinstance(arm,list):arm=next(x for x in arm if x.get('name')=='body')
    arms=arm['arm_extension']
    with h5py.File(source) as f:
        nodes=f['fmdl_mm/nodes'][:].T;tets=f['fmdl_mm/elems'][:].T.astype(np.int32)-1;boundary=f['fmdl_mm/boundary'][:].T.astype(np.int32)-1;labels=f['tissue_id'][:].ravel().astype(np.int32)
    old_nodes=nodes.copy();old_count=len(tets);oldv=volumes(nodes,tets)
    if (oldv<=0).any():raise ValueError('Parent tetrahedra not all positive')
    xc=nodes[tets,0].mean(1);distal=(xc>arms['right']['cut_x'])|(xc<arms['left']['cut_x'])
    groups=np.select([labels==1,labels==3,labels==2,labels==4],[1,3,4,5],default=0).astype(np.int32);groups[(labels==1)&distal]=2
    if set(groups)!={1,2,3,4,5}:raise ValueError('Unsupported tissue/group labels')
    before=np.bincount(groups,weights=oldv,minlength=6)
    centres=np.array([v for item in m['inputs'] for v in json.loads((base/item['json']).read_text())['preliminary_electrode_centers_xyz_mm'].values()])
    nodes,tets,boundary,labels,groups,history=refine(nodes,tets,boundary,labels,groups,centres)
    v=volumes(nodes,tets);after=np.bincount(groups,weights=v,minlength=6)
    if (v<=0).any() or not np.array_equal(nodes[:len(old_nodes)],old_nodes) or not np.allclose(before,after,rtol=1e-10,atol=1e-5):raise ValueError('Refinement changed geometry/volumes or inverted tetrahedra')
    print('Verifying complete tetrahedral boundary',flush=True);verify_boundary(tets,boundary)
    path=out/'refined_mesh.mat'
    savemat(path,dict(nodes_mm=nodes,elems=tets+1,boundary=boundary+1,tissue_id=labels[:,None],group_id=groups[:,None]),do_compression=False)
    report=dict(status='passed_geometry_preserving_local_refinement',parent_prepared_sha256=sha(source),manifest_sha256=sha(base/'manifest.json'),refined_mesh_sha256=sha(path),arm_report_sha256=sha(a.arm_report),target_edge_mm=1.,contact_neighbourhood_radius_mm=8.,rounds=history,
        counts=dict(nodes=len(nodes),tetrahedra=len(tets),boundary=len(boundary),added_nodes=len(nodes)-len(old_nodes),added_tetrahedra=len(tets)-old_count),
        checks=dict(original_vertices_unchanged=True,all_positive_tetrahedra=True,group_volumes_preserved=True,conforming_boundary_verified=True,contact_neighbourhood_edges_le_1mm=True),group_volume_before_mm3=before.tolist(),group_volume_after_mm3=after.tolist(),min_tetra_volume_mm3=float(v.min()),physical_validation=False,spatial_solution_convergence=False,source_sha256=sha(Path(__file__)))
    (out/'refinement.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
