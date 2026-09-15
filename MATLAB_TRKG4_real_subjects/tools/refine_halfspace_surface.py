"""Conforming planar subdivision to resolve field variation without refitting CT."""
import numpy as np
from scipy.spatial import cKDTree


def refine(vertices,faces,sources,max_edge_to_distance=.35,max_rounds=8):
    v=np.array(vertices,float);f=np.array(faces,int);tree=cKDTree(np.asarray(sources).reshape(-1,3))
    for iteration in range(max_rounds):
        edges=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
        edges=np.unique(edges,axis=0);mid=v[edges].mean(axis=1);length=np.linalg.norm(v[edges[:,1]]-v[edges[:,0]],axis=1)
        distance=tree.query(mid)[0];chosen=edges[length>max_edge_to_distance*distance]
        if not len(chosen):break
        midpoints={tuple(e):len(v)+i for i,e in enumerate(chosen)}
        v=np.concatenate([v,v[chosen].mean(axis=1)])
        nf=[]
        for a,b,c in f:
            ab=midpoints.get(tuple(sorted((a,b))));bc=midpoints.get(tuple(sorted((b,c))));ca=midpoints.get(tuple(sorted((c,a))))
            count=sum(x is not None for x in [ab,bc,ca])
            if count==0:nf.append([a,b,c])
            elif count==3:nf.extend([[a,ab,ca],[ab,b,bc],[ca,bc,c],[ab,bc,ca]])
            elif count==1:
                if ab is not None:nf.extend([[a,ab,c],[ab,b,c]])
                elif bc is not None:nf.extend([[b,bc,a],[bc,c,a]])
                else:nf.extend([[c,ca,b],[ca,a,b]])
            else:
                if ca is None:nf.extend([[ab,b,bc],[a,ab,bc],[a,bc,c]])
                elif ab is None:nf.extend([[bc,c,ca],[b,bc,ca],[b,ca,a]])
                else:nf.extend([[ca,a,ab],[c,ca,ab],[c,ab,b]])
        f=np.array(nf,int)
    # Each directed edge must have exactly one opposite orientation.
    e=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);_,count=np.unique(np.sort(e,axis=1),axis=0,return_counts=True)
    assert np.all(count==2),'Subdivision broke surface closure'
    signs=np.where(e[:,0]<e[:,1],1,-1);_,inverse=np.unique(np.sort(e,axis=1),axis=0,return_inverse=True)
    assert np.all(np.bincount(inverse,weights=signs)==0),'Subdivision broke orientation'
    return v,f

def main():
    """Reproduce both adaptive surfaces; preserve identical existing files."""
    import argparse,json,hashlib
    from pathlib import Path
    from halfspace_study import OUT,electrodes
    ap=argparse.ArgumentParser()
    ap.add_argument('--tolerances',nargs='+',type=float,default=[.25,.15])
    args=ap.parse_args()
    source=OUT/'m4h_surface_fine.npz'
    z=np.load(source,allow_pickle=False)
    records=[]
    for tol in args.tolerances:
        if not 0<tol<=1:raise ValueError('Tolerance must lie in (0,1]')
        vertices,faces=refine(z['vertices'],z['faces'],electrodes()[1],tol)
        dest=OUT/f'm4h_field{int(round(tol*100)):03d}_surface.npz'
        if dest.exists():
            old=np.load(dest,allow_pickle=False)
            if not(np.array_equal(old['vertices'],vertices) and np.array_equal(old['faces'],faces)):
                raise RuntimeError(f'Recomputed surface differs; existing file preserved: {dest}')
            status='existing_arrays_reproduced_exactly'
        else:
            np.savez_compressed(dest,vertices=vertices,faces=faces,units='m',model='M4-H',
                                source=source.name,max_edge_to_distance=tol)
            status='created'
        records.append({'file':dest.name,'faces':len(faces),'tolerance':tol,'status':status,
                        'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
    report={'source':source.name,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'surfaces':records,'geometry_refitted':False}
    (OUT/'surface_reproduction.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
