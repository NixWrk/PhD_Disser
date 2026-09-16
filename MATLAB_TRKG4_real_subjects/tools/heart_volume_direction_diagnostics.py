"""Account for tissue exchange during uniform scaling of the individual heart.

Signed volume accounting is not a causal decomposition of the impedance.
The input is an existing sealed geometry/electrical stage; no FEM is run.
"""
from pathlib import Path
import argparse
import numpy as np
import heart_replacement_electrical_pilot as core
import heart_fractional_electrical_pilot as fractional


def run(source,output):
    source=source.resolve();p=fractional.load_plan(source)
    core.require(p['geometry']=='individual' and p['parameter']=='volume','Individual volume design required')
    runtime=core.configure(Path(p['runtime']['deps']),p['threads'])
    base=next(s for s in p['states'] if s['parameter']=='baseline')
    def load(s):
        path=source/s['geometry_npz']
        core.require(core.sha(path)==s['geometry_npz_sha256'],'Geometry array changed')
        with np.load(path) as a:
            level=list(a['levels']).index(s['points_per_tet'])
            return a['element_indices'],a['fractions'][level]
    ids,baseline=load(base)
    with runtime.h5py.File(p['source_prepared']) as f:
        nodes=f['fmdl_m/nodes'][:].T
        elems=f['fmdl_m/elems'][:,ids].T.astype(np.int64)-1
        labels=f['tissue_id'][:].ravel()[ids]
    v=nodes[elems]
    ml=np.einsum('ij,ij->i',v[:,1]-v[:,0],np.cross(v[:,2]-v[:,0],v[:,3]-v[:,0]))/6*1e6
    core.require(np.all(ml>0),'Nonpositive selected volume')
    core.require(np.array_equal(baseline,labels==3),'Baseline does not reproduce original materials')
    rows=[]
    for state in p['states']:
        current_ids,f=load(state);core.require(np.array_equal(current_ids,ids),'Different selected elements')
        delta=f-baseline;total=float(ml@delta)
        bymaterial={str(k):float(ml[labels==k]@delta[labels==k]) for k in [1,2,3,4]}
        core.require(np.isclose(sum(bymaterial.values()),total,rtol=1e-12,atol=1e-10),'Accounting failed')
        analytic=state['volume_fraction']*p['nominal_volume_ml']
        rows.append(dict(volume_fraction=state['volume_fraction'],delta_heart_volume_ml=total,
                         analytic_delta_volume_ml=analytic,
                         one_sided_material_volume_ratio=None if analytic==0 else total/analytic,
                         changes_by_original_material_ml=bymaterial))
    core.write(output,dict(source_plan_sha256=core.sha(source/'plan.json'),implementation_sha256=core.sha(__file__),
        geometry_hashes={s['geometry_npz']:s['geometry_npz_sha256'] for s in p['states']},
        quadrature=base['points_per_tet'],original_materials={'1':'soft','2':'lung','3':'heart (background soft)','4':'bone'},
        rows=rows,interpretation='Signed material-volume accounting, not causal decomposition of impedance.',
        nonnested_shapes_possible=True,physical_validation=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source,a.output)
