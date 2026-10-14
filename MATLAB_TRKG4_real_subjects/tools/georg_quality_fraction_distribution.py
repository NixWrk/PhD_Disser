"""Describe per-element tissue changes on an identical saved mesh; no FEM."""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.io import loadmat
from georg_quality_comparison import read,sha


def describe(inherited,remapped,points=(1024,4096)):
    inherited,remapped=Path(inherited).resolve(),Path(remapped).resolve()
    mesh_hash=sha(inherited/'mesh.mat')
    if mesh_hash!=sha(remapped/'mesh.mat'):raise ValueError('This comparison requires identical mesh bytes')
    plans=[read(d/'plan.local.json') for d in (inherited,remapped)]
    if plans[0]['material_order']!=plans[1]['material_order'] or len(plans[0]['material_order'])!=5:
        raise ValueError('Material ordering changed')
    rows=[];sources=[dict(path=str(d/'mesh.mat'),sha256=mesh_hash) for d in (inherited,remapped)]
    for q in points:
        paths=[d/f'fractions_q{q}.mat' for d in (inherited,remapped)]
        a,b=[loadmat(p,variable_names=['fractions'])['fractions'] for p in paths]
        if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=5:raise ValueError('Fraction arrays have different shapes')
        for array in (a,b):
            if not np.all(np.isfinite(array)) or np.any(array<0) or not np.array_equal(array.sum(1),np.ones(len(array))):
                raise ValueError('Invalid tissue partition')
        delta=b-a;changed=delta!=0
        rows.append(dict(points=q,tetrahedra=len(a),changed_elements=int(np.count_nonzero(changed.any(1))),
            changed_elements_by_material=np.count_nonzero(changed,axis=0).tolist(),
            maximum_absolute_fraction_change=np.max(np.abs(delta),axis=0).tolist(),
            positive_changes_by_material=np.count_nonzero(delta>0,axis=0).tolist(),
            negative_changes_by_material=np.count_nonzero(delta<0,axis=0).tolist(),
            fraction_sum_preserved_exactly=True))
        sources.extend(dict(path=str(p),sha256=sha(p)) for p in paths)
    sources.extend(dict(path=str(p),sha256=sha(p)) for p in
                   [inherited/'plan.local.json',remapped/'plan.local.json',Path(__file__)])
    return dict(schema='georg_same_mesh_tissue_fraction_distribution_v1',
        materials=plans[0]['material_order'],mesh_sha256=mesh_hash,rules=rows,sources=sources,
        bulk_mesh_identical=True,coordinate_changes=False,FEM_was_run=False,
        compartment_impedance_contribution_not_computed=True,
        independent_segmentation_accuracy_not_established=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('inherited','remapped','output'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise ValueError('Use a fresh immutable distribution output')
    result=describe(a.inherited,a.remapped)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result['rules']))
