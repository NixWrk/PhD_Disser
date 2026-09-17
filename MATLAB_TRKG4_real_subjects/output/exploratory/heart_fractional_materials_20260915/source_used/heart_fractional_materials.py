"""Deterministic nested integration of heart fractions in unchanged P1 tetrahedra.

No FEM solve. Every input tetrahedron is tested by a safe vertex-AABB broad
phase, NEVER by its centroid. Within a tetrahedron affine normalized q(x)
is bounded by the vertex coordinate box: distance(box,0)>1 is safely outside;
all four vertices inside the convex ellipsoid is safely inside. Others use
nested deterministic scrambled Sobol(3) -> sorted order spacings, uniform
in a tetrahedron in the continuous sampling limit. Fixed seed / shared points
reduce sampling differences between states but do not certify convergence.

P1: B_k = F_volume.T @ diag(repeat(f[:,k],3)) @ F_volume;
K = sum_k sigma_k B_k + C. Because P1 gradients are constant, this is the exact
Galerkin element integral IF the material fractions are exact. It does NOT
make the coarse P1 trial space resolve a material interface inside a cell.
The original heart is soft background; bone and lung backgrounds are immutable.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import time
import numpy as np
from scipy.stats import qmc

LEVELS=(64,512,4096)
SEED=20260915

def require(value,message):
    if not value:raise ValueError(message)

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)

def uniform_tet_barycentric(count=4096,seed=SEED):
    require(isinstance(count,int) and count>0 and count&(count-1)==0,'Point count must be a power of two')
    # Fixed digital scrambling avoids the degenerate all-zero Sobol first node.
    samples=qmc.Sobol(d=3,scramble=True,seed=seed).random_base2(int(math.log2(count)))
    ordered=np.sort(samples,axis=1)
    return np.diff(np.column_stack((np.zeros(count),ordered,np.ones(count))),axis=1)

def checked_candidate(candidate):
    c=np.asarray(candidate['centre_m'],float);a=np.asarray(candidate['semiaxes_m'],float)
    R=np.asarray(candidate['axes_columns'],float)
    require(c.shape==(3,) and a.shape==(3,) and R.shape==(3,3),'Invalid ellipsoid parameter shapes')
    require(np.isfinite(c).all() and np.isfinite(a).all() and np.isfinite(R).all() and np.all(a>0),'Invalid ellipsoid parameters')
    require(np.allclose(R.T@R,np.eye(3),atol=1e-10,rtol=0),'Axes must be orthonormal')
    return c,a,R

def candidate_bounds(candidate):
    c,a,R=checked_candidate(candidate);extent=np.linalg.norm(R*a,axis=1)
    return c-extent,c+extent

def safe_box_overlap(vertices,lower,upper):
    # Includes a conservative roundoff margin; false positives are acceptable.
    margin=64*np.finfo(float).eps*max(float(np.max(np.abs(vertices))),float(np.max(np.abs(lower))),float(np.max(np.abs(upper))),1.)
    return np.all((vertices.max(axis=1)>=lower-margin)&(vertices.min(axis=1)<=upper+margin),axis=1)

def integrate_fractions(vertices,candidate,levels=LEVELS,barycentric=None,batch_size=32):
    """Return fractions [level,tet] and per-tet full/error-free classification.

    Only quadrature error is varied. Rows flagged outside/full are exact up to
    conservative floating-point margins. An unresolved tiny intersection may
    still get quadrature estimate zero; it remains classified 'quadrature'.
    """
    v=np.asarray(vertices,float);levels=tuple(levels)
    require(v.ndim==3 and v.shape[1:]==(4,3) and np.isfinite(v).all(),'Expected finite Nx4x3 vertices')
    require(levels and tuple(sorted(set(levels)))==levels and all(isinstance(n,int) and n>0 and n&(n-1)==0 for n in levels),'Levels must be increasing powers of two')
    require(isinstance(batch_size,int) and batch_size>0,'Positive batch size required')
    c,a,R=checked_candidate(candidate)
    q=((v-c)@R)/a
    # q is affine over each convex tetrahedron; coordinate box is an enclosure.
    low=q.min(axis=1);high=q.max(axis=1)
    near=np.maximum(np.maximum(low,-high),0.)
    eps=128*np.finfo(float).eps*np.maximum(1.,np.max(np.abs(q),axis=(1,2))**2)
    outside=np.einsum('ij,ij->i',near,near)>1+eps
    inside=np.max(np.einsum('tvi,tvi->tv',q,q),axis=1)<1-eps
    require(not np.any(inside&outside),'Inconsistent conservative classification')
    partial=~(inside|outside);indices=np.flatnonzero(partial)
    answer=np.zeros((len(levels),len(v)),dtype=np.float32);answer[:,inside]=1
    bary=uniform_tet_barycentric(max(levels)) if barycentric is None else np.asarray(barycentric,float)
    require(bary.shape==(max(levels),4) and np.isfinite(bary).all() and np.all(bary>=0) and np.allclose(bary.sum(axis=1),1,rtol=0,atol=1e-14),'Invalid barycentric rule')
    for start in range(0,len(indices),batch_size):
        ids=indices[start:start+batch_size]
        pts=np.einsum('pv,tvc->tpc',bary,q[ids],optimize=True)
        membership=np.einsum('tpi,tpi->tp',pts,pts)<=1.
        cumulative=np.cumsum(membership,axis=1,dtype=np.int32)
        for k,n in enumerate(levels):answer[k,ids]=cumulative[:,n-1]/n
    require(np.isfinite(answer).all() and np.all((answer>=0)&(answer<=1)),'Fraction outside [0,1]')
    return answer,{'inside':inside,'outside':outside,'quadrature':partial}

def material_fractions(heart_fraction,background):
    """Columns soft/lung/heart/bone; no mutation and no unknown tissue recovery."""
    f=np.asarray(heart_fraction,float);b=np.asarray(background)
    require(f.ndim==1 and b.shape==f.shape and np.isfinite(f).all() and np.all((f>=0)&(f<=1)),'Invalid heart fraction')
    require(np.isin(b,[1,2,4]).all(),'Background must contain only soft,lung,bone (no heart)')
    fractions=np.zeros((len(f),4));fractions[:,2]=f
    for material in (1,2,4):fractions[:,material-1]=(1-f)*(b==material)
    require(np.allclose(fractions.sum(axis=1),1,rtol=0,atol=1e-15),'Fractions must partition every tet')
    return fractions

def p1_element_matrix(vertices,fractions,conductivities):
    """Small reference implementation for tests; production blocks use EIDORS F."""
    v=np.asarray(vertices,float);f=np.asarray(fractions,float);sigma=np.asarray(conductivities,float)
    require(v.shape==(4,3) and f.shape==sigma.shape and f.ndim==1,'Invalid P1 reference arguments')
    require(np.isfinite(v).all() and np.isfinite(f).all() and np.isfinite(sigma).all() and np.all(f>=0) and np.all(sigma>0) and np.isclose(f.sum(),1,atol=1e-14,rtol=0),'Invalid physical fractions or conductivity')
    volume=abs(np.linalg.det(v[1:]-v[0]))/6
    require(volume>0,'Degenerate tetrahedron')
    G=np.linalg.inv(np.column_stack((np.ones(4),v)))[1:,:]
    return float(f@sigma)*volume*(G.T@G)

def expand_heart_fraction(element_count,indices,values):
    require(indices.ndim==1 and values.shape==indices.shape and np.all(indices>=0) and np.all(indices<element_count) and len(np.unique(indices))==len(indices),'Invalid compact element indices')
    full=np.zeros(element_count,dtype=values.dtype);full[indices]=values
    return full

def selected_mesh(mesh,states,chunk_size=65536):
    """Scan ALL tets; a vertex-AABB separation certifies every excluded tet."""
    import h5py
    bounds=[candidate_bounds(s) for s in states]
    lower=np.min([p[0] for p in bounds],axis=0);upper=np.max([p[1] for p in bounds],axis=0)
    selected=[];ids=[];volumes=[];background=[]
    total_volume=0.;total_bg=np.zeros(4)
    with h5py.File(mesh) as f:
        nodes=f['fmdl_m/nodes'][:].T;elems=f['fmdl_m/elems'];n=elems.shape[1]
        labels=f['tissue_id'][:].ravel().astype(np.uint8);require(len(labels)==n,'Label length mismatch')
        bg=labels.copy();bg[bg==3]=1;require(np.isin(bg,[1,2,4]).all(),'Invalid background')
        for start in range(0,n,chunk_size):
            e=elems[:,start:start+chunk_size].T.astype(np.int64)-1;v=nodes[e]
            det=np.einsum('ij,ij->i',v[:,1]-v[:,0],np.cross(v[:,2]-v[:,0],v[:,3]-v[:,0]))
            require(np.all(det>0),'Nonpositive tet Jacobian')
            vol=det/6*1e6;keep=safe_box_overlap(v,lower,upper)
            selected.append(v[keep]);ids.append(start+np.flatnonzero(keep));volumes.append(vol[keep]);background.append(bg[start:start+len(e)][keep])
            total_volume+=float(vol.sum());total_bg+=np.bincount(bg[start:start+len(e)],weights=vol,minlength=5)[1:]
    return np.concatenate(selected),np.concatenate(ids),np.concatenate(volumes),np.concatenate(background),{
        'all_input_tetrahedra_scanned':n,'selected_tetrahedra':sum(len(i) for i in ids),
        'excluded_tetrahedra_certified_by':'vertex-AABB separation from union of all candidate AABBs, with conservative tolerance',
        'centroid_filter_used':False,'background_volume_ml_by_tissue_id':total_bg.tolist(),
        'total_domain_volume_ml':total_volume,'lower_m':lower.tolist(),'upper_m':upper.tolist()}

def derivatives(rows,params,levels):
    result=[];lookup={(r['geometry'],r['parameter'],r['signed_step'],r['points_per_tet']):r for r in rows}
    for name,p in params.items():
        for level in levels:
            for parameter,steps in [('x_mm',(.5,1.,2.,4.)),('y_mm',(.5,1.,2.,4.)),('z_mm',(.5,1.,2.,4.)),('volume_fraction',(.005,.01,.02,.04))]:
                previous=None
                for step in steps:
                    plus=lookup[name,parameter,step,level];minus=lookup[name,parameter,-step,level]
                    denominator=2*step*(p['analytic_volume_ml'] if parameter=='volume_fraction' else 1.)
                    d=(plus['material_volume_ml']-minus['material_volume_ml'])/denominator
                    result.append({'geometry':name,'parameter':parameter,'step':step,'points_per_tet':level,
                        'material_volume_derivative':d,'expected_analytic_volume_derivative':1. if parameter=='volume_fraction' else 0.,
                        'units':'ml/ml' if parameter=='volume_fraction' else 'ml/mm',
                        'difference_from_half_step':None if previous is None else d-previous})
                    previous=d
    lookup={(d['geometry'],d['parameter'],d['step'],d['points_per_tet']):d for d in result}
    for d in result:
        i=levels.index(d['points_per_tet']);prev=None if i==0 else lookup[d['geometry'],d['parameter'],d['step'],levels[i-1]]
        d['difference_from_previous_quadrature_level']=None if prev is None else d['material_volume_derivative']-prev['material_volume_derivative']
    return result

def run(args):
    if args.deps:sys.path.insert(0,str(args.deps.resolve()))
    import h5py,scipy,heart_geometry_perturbation_audit as audit
    audit.np=np # Existing probes() uses np initialized only by its own CLI.
    source=args.materials.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    pre=read(source/'preflight.json');record=pre['sources']['prepared_fem'];mesh=(source/record['path']).resolve()
    require(sha(mesh)==record['sha256'],'Mesh hash mismatch')
    params=read(source/'candidate_parameters.json')['candidates'];states=audit.probes(params)
    require(len(states)==66,'Expected existing 66-state probe contract')
    old=read(args.centroid_audit)
    require(old['mesh_sha256']==sha(mesh) and old['candidate_parameters_sha256']==sha(source/'candidate_parameters.json'),'Centroid comparison belongs to another mesh or candidates')
    bary=uniform_tet_barycentric(4096);identity={'schema':'heart_fractional_materials_v1','mesh_sha256':sha(mesh),
       'candidate_parameters_sha256':sha(source/'candidate_parameters.json'),'centroid_audit_sha256':sha(args.centroid_audit),
       'implementation_sha256':sha(__file__),'probes_implementation_sha256':sha(audit.__file__),
       'background_sha256':sha(source/'background_labels.npy'),'states':states,'levels':list(LEVELS),'seed':SEED,
       'barycentric_sha256':hashlib.sha256(bary.tobytes()).hexdigest(),'scipy_version':scipy.__version__,
       'numpy_version':np.__version__,'h5py_version':h5py.__version__,
       'sampling':'Fixed-seed scrambled Sobol3, sort3 -> four spacings; identical nested points across states/tets'}
    if (out/'identity.json').exists():require(read(out/'identity.json')==identity,'Cannot resume changed inputs/code/quadrature')
    else:write(out/'identity.json',identity)
    plan_sha=sha(out/'identity.json');lock=out/'run.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    started=time.time()
    try:
        write(out/'status.json',{'status':'running','stage':'all_tet_broad_phase','pid':os.getpid()})
        vertices,indices,vol,bg,scan=selected_mesh(mesh,states)
        with h5py.File(mesh) as f:original=f['tissue_id'][:].ravel().astype(np.uint8)
        background=np.load(source/'background_labels.npy')
        require(np.array_equal(background,np.where(original==3,1,original)),'Frozen background does not match rule')
        require(np.array_equal(bg,background[indices]),'Selected background mismatch')
        np.save(out/'selected_element_indices_zero_based.npy',indices)
        np.save(out/'selected_background_labels.npy',bg)
        np.save(out/'barycentric_4096.npy',bary)
        scan['selected_indices_sha256']=sha(out/'selected_element_indices_zero_based.npy')
        write(out/'mesh_scan.json',scan)
        rows=[];receipts=[]
        for i,state in enumerate(states):
            stem=f'state_{i:03d}';receipt=out/(stem+'.json');array_file=out/(stem+'.npz')
            if receipt.exists():
                saved=read(receipt)
                require(saved['identity_sha256']==plan_sha and saved['array_sha256']==sha(array_file),'Corrupt/incompatible state cache')
                rows.extend(saved['rows']);receipts.append({'file':receipt.name,'sha256':sha(receipt)});continue
            tick=time.time();fraction,classification=integrate_fractions(vertices,state,barycentric=bary,batch_size=args.batch_size)
            state_rows=[]
            for j,level in enumerate(LEVELS):
                f=fraction[j];tissue_f=material_fractions(f,bg);v=float(vol@f)
                replaced={str(t):float(vol[bg==t]@f[bg==t]) for t in (1,2,4)}
                whole=np.array(scan['background_volume_ml_by_tissue_id']);whole[2]=v
                for t in (1,2,4):whole[t-1]-=replaced[str(t)]
                require(abs(whole.sum()-scan['total_domain_volume_ml'])<1e-7*scan['total_domain_volume_ml'],'Material volume partition failed')
                require(np.allclose(tissue_f.sum(axis=1),1,atol=1e-15,rtol=0),'Material partition failed')
                state_rows.append({'state_index':i,'geometry':state['name'],'parameter':state['parameter'],
                    'step':state['step'],'signed_step':state['signed_step'],'points_per_tet':level,
                    'analytic_volume_ml':state['analytic_volume_ml'],'material_volume_ml':v,
                    'volume_error_percent':100*(v/state['analytic_volume_ml']-1),
                    'replaced_background_ml':replaced,'full_domain_material_volume_ml_by_tissue_id':whole.tolist(),
                    'quadrature_tets':int(classification['quadrature'].sum()),'exact_full_tets':int(classification['inside'].sum()),
                    'conservative_outside_tets':int(classification['outside'].sum()),
                    'fractional_tets':int(np.count_nonzero((f>0)&(f<1))),
                    'quadrature_zero_estimate_tets':int(np.count_nonzero(classification['quadrature']&(f==0)))})
            np.savez_compressed(array_file,heart_fraction=fraction,levels=np.array(LEVELS))
            write(receipt,{'identity_sha256':plan_sha,'state':state,'rows':state_rows,'elapsed_s':time.time()-tick,
                          'array_sha256':sha(array_file),'representation':'fractions[level,selected_global_index]; all other global tets exactly fheart=0 by broad-phase certificate'})
            rows.extend(state_rows);receipts.append({'file':receipt.name,'sha256':sha(receipt)})
            write(out/'status.json',{'status':'running','stage':'quadrature','completed_states':i+1,'required_states':66,
                   'pid':os.getpid(),'elapsed_s':time.time()-started,'last_state_seconds':time.time()-tick})
            print(f'{i+1}/66 {state["name"]} {state["parameter"]} {state["signed_step"]}: '+str([r['volume_error_percent'] for r in state_rows]),flush=True)
        ds=derivatives(rows,params,LEVELS)
        oldlookup={(d['geometry'],d['parameter'],d['step']):d for d in old['derivatives']}
        for d in ds:d['centroid_material_volume_derivative']=oldlookup[d['geometry'],d['parameter'],d['step']]['material_volume_derivative']
        metrics=[]
        for level in LEVELS:
            rr=[r for r in rows if r['points_per_tet']==level];dd=[d for d in ds if d['points_per_tet']==level]
            metrics.append({'points_per_tet':level,'max_abs_volume_error_percent':max(abs(r['volume_error_percent']) for r in rr),
                'max_abs_translation_dV_ml_per_mm':max(abs(d['material_volume_derivative']) for d in dd if d['parameter']!='volume_fraction'),
                'max_abs_volume_derivative_minus_one':max(abs(d['material_volume_derivative']-1) for d in dd if d['parameter']=='volume_fraction')})
        require(sha(mesh)==identity['mesh_sha256'] and sha(__file__)==identity['implementation_sha256'] and sha(audit.__file__)==identity['probes_implementation_sha256'],'Input/code changed during quadrature')
        result={'schema':'heart_fractional_materials_result_v1','status':'completed_geometry_quadrature_not_FEM',
            'identity_sha256':plan_sha,'states':66,'rows':rows,'derivatives':ds,'level_metrics':metrics,
            'elapsed_s':time.time()-started,'receipts':receipts,'FEM_solved':False,
            'physical_validation':False,'spatial_solution_convergence':False,'reference_individual_shape_included':False,
            'limitations':['Finite nested quadrature; level agreement is not a rigorous continuous-domain error bound',
              'Fixed quadrature induces correlated errors across states and tets; no independent stochastic confidence interval',
              'Small unresolved intersections may have zero sampled fraction; they are not classified geometrically outside',
              'Exact material integration would not remove coarse P1 field discretization error',
              'Automatic target-phase mask versus individual reference-shape comparison is not included']}
        write(out/'results.json',result)
        write(out/'completion.json',{'identity_sha256':plan_sha,'results_sha256':sha(out/'results.json'),
              'mesh_scan_sha256':sha(out/'mesh_scan.json'),'selected_background_sha256':sha(out/'selected_background_labels.npy'),
              'barycentric_sha256':sha(out/'barycentric_4096.npy'),'status':'completed_geometry_checks_only'})
        write(out/'status.json',{'status':'complete','states':66,'elapsed_s':time.time()-started,'FEM_solved':False})
        print(json.dumps(metrics,indent=2))
    finally:lock.unlink(missing_ok=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--materials',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--centroid-audit',type=Path,required=True)
    p.add_argument('--deps',type=Path);p.add_argument('--batch-size',type=int,default=32);run(p.parse_args())
if __name__=='__main__':main()
