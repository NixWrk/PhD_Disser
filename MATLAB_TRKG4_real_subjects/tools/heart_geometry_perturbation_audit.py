"""Audit geometric perturbation resolution on a frozen tetrahedral mesh.

This computes no impedance and establishes no clinical motion ranges. The
analytic candidates and old immutable material background are unchanged.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,copy,json,hashlib
from pathlib import Path
import sys
from prepare_static_heart_replacement_pilot import sha256

def probes(parameters):
    result=[]
    for name,p in parameters.items():
        base=copy.deepcopy(p);base['name']=name;base['parameter']='baseline';base['step']=0.;base['signed_step']=0.
        result.append(base)
        for axis,tag in enumerate(('x_mm','y_mm','z_mm')):
            for h in (.5,1.,2.,4.):
                for sign in (-1,1):
                    state=copy.deepcopy(base);state.update(parameter=tag,step=h,signed_step=sign*h)
                    state['centre_m'][axis]+=sign*h/1000
                    result.append(state)
        for f in (.005,.01,.02,.04):
            for sign in (-1,1):
                state=copy.deepcopy(base);state.update(parameter='volume_fraction',step=f,signed_step=sign*f)
                state['semiaxes_m']=(np.asarray(base['semiaxes_m'])*np.cbrt(1+sign*f)).tolist()
                state['analytic_volume_ml']=base['analytic_volume_ml']*(1+sign*f)
                result.append(state)
    return result

def run(a):
    if a.deps:sys.path.insert(0,str(Path(a.deps).resolve()))
    global np,h5py
    import numpy as np
    import h5py
    source=Path(a.materials);pre=json.loads((source/'preflight.json').read_text(encoding='utf-8'))
    record=pre['sources']['prepared_fem'];mesh=(source/record['path']).resolve()
    if sha256(mesh)!=record['sha256']:raise ValueError('Mesh hash mismatch')
    params=json.loads((source/'candidate_parameters.json').read_text(encoding='utf-8'))['candidates']
    states=probes(params);lo=[];hi=[]
    for p in states:
        extent=np.linalg.norm(np.asarray(p['axes_columns'])*np.asarray(p['semiaxes_m']),axis=1)
        c=np.asarray(p['centre_m']);lo.append(c-extent);hi.append(c+extent)
    lower=np.min(lo,axis=0);upper=np.max(hi,axis=0)
    centroids=[];volumes=[];labels=[]
    with h5py.File(mesh,'r') as f:
        nodes=f['fmdl_m/nodes'][:].T;elems=f['fmdl_m/elems'];tissues=f['tissue_id'][:].ravel().astype(np.uint8)
        for start in range(0,elems.shape[1],65536):
            conn=elems[:,start:start+65536].T.astype(np.int64)-1
            v=nodes[conn];c=v.mean(axis=1);keep=np.all((c>=lower)&(c<=upper),axis=1)
            v=v[keep];c=c[keep];e=v[:,1:]-v[:,:1]
            vol=np.einsum('ni,ni->n',np.cross(e[:,0],e[:,1]),e[:,2])/6
            if np.any(vol<=0):raise ValueError('Nonpositive tetrahedron')
            centroids.append(c);volumes.append(vol*1e6);labels.append(tissues[start:start+65536][keep])
    points=np.concatenate(centroids);v=np.concatenate(volumes);bg=np.concatenate(labels);bg[bg==3]=1
    rows=[];masks={}
    for p in states:
        c=np.asarray(p['centre_m']);axes=np.asarray(p['axes_columns']);a_axes=np.asarray(p['semiaxes_m'])
        q=(points-c)@axes/a_axes;keep=np.sum(q*q,axis=1)<=1+1e-12
        total=float(v[keep].sum());actual=(v[keep,None]*points[keep]).sum(axis=0)/total
        key=(p['name'],p['parameter'],p['signed_step']);masks[key]=keep
        rows.append({'geometry':p['name'],'parameter':p['parameter'],'step':p['step'],'signed_step':p['signed_step'],
            'analytic_volume_ml':p['analytic_volume_ml'],'material_volume_ml':total,
            'volume_error_percent':100*(total/p['analytic_volume_ml']-1),
            'analytic_center_mm':(c*1000).tolist(),'material_centroid_mm':(actual*1000).tolist(),
            'heart_tetrahedra':int(keep.sum()),'replaced_background_ml':{str(t):float(v[keep&(bg==t)].sum()) for t in (1,2,4)}})
    lookup={(r['geometry'],r['parameter'],r['signed_step']):r for r in rows};derivatives=[]
    for name in params:
        base=lookup[(name,'baseline',0.)]
        for parameter,steps in [('x_mm',(.5,1.,2.,4.)),('y_mm',(.5,1.,2.,4.)),('z_mm',(.5,1.,2.,4.)),('volume_fraction',(.005,.01,.02,.04))]:
            previous=None
            for step in steps:
                plus=lookup[(name,parameter,step)];minus=lookup[(name,parameter,-step)]
                denominator=2*step*(params[name]['analytic_volume_ml'] if parameter=='volume_fraction' else 1.)
                deriv=(plus['material_volume_ml']-minus['material_volume_ml'])/denominator
                expected=1. if parameter=='volume_fraction' else 0.
                d={'geometry':name,'parameter':parameter,'step':step,
                    'material_volume_derivative':deriv,'expected_analytic_volume_derivative':expected,
                    'units':'ml/ml' if parameter=='volume_fraction' else 'ml/mm',
                    'changed_tets_plus_minus':int(np.count_nonzero(masks[(name,parameter,step)]!=masks[(name,parameter,-step)])),
                    'difference_from_half_step':None if previous is None else deriv-previous,
                    'relative_difference_defined':False if expected==0 else True}
                derivatives.append(d);previous=deriv
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    result={'schema':'heart_geometric_perturbation_resolution_v1','status':'exploratory_diagnostic',
        'mesh_sha256':record['sha256'],'code_sha256':sha256(Path(__file__)),
        'candidate_parameters_sha256':sha256(source/'candidate_parameters.json'),
        'evaluated_tetrahedra_in_full_candidate_envelope':len(points),'rows':rows,'derivatives':derivatives,
        'FEM_solved':False,'physiological_amplitudes_claimed':False,
        'reference_individual_shape_perturbations_included':False,
        'domain_boundary_rechecked':False,
        'limitations':['centroid whole-tetrahedron material assignment','no spatial refinement of mesh',
            'known continuous volumes provide a numerical check, not anatomical validation',
            'changes in geometric volume do not directly quantify impedance error']}
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'states':len(rows),'derivatives':len(derivatives),'selected_tets':len(points)}))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--deps');p.add_argument('--materials',required=True);p.add_argument('--output',required=True);run(p.parse_args())
