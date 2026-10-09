"""Active bounded scalar solution surfaces for all recorded lateral montages."""
from pathlib import Path
import argparse
from itertools import combinations

import numpy as np
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits

from lateral_pilot import ROOT,EXPERIMENT,DiskModel,prepare,read,save,sha
from lateral_identifiability import KernelBank,SIZES,IDS,vector
from lateral_admissible_sets import bounded_fit,profile

OUT=EXPERIMENT/'derived/exploratory/33.21_solution_surfaces_20261009'
H=np.r_[.5,np.arange(1.,101.)]


def definition(channel):
    """3: author-confirmed transition; 1/2: former constant-number counterfactuals."""
    items=[]
    default=1 if channel==3 else channel
    def add(record,size,label,reference=False,source_channel=None):
        ch=source_channel or default
        items.append(dict(key=record+'_'+str(ch),record_id='SCV2-G-'+record,
            size_mm=size,label=label,reference=reference,channel=ch))
    for size,rid in zip(SIZES,IDS):
        ch=2 if channel==3 and rid.endswith('004') else default
        add(rid[-3:],int(size),f'{int(size)} мм · {rid[-3:]} · основной ряд',True,ch)
    for record,label in [('002','1×1 на руках'),('003','2×2 на руках')]:
        add(record,140,f'140 мм · {record} · {label} · дополнительное наблюдение',source_channel=2 if channel==3 else default)
    add('005',140,'140 мм · 005 · вертикальная боковая, BEI2',source_channel=2)
    for record,size in [('006',140),('011',60)]:
        ch=1 if channel==3 else default
        add(record,size,f'{size} мм · {record} · горизонтальная ось креста',source_channel=ch)
        add(record,size,f'{size} мм · {record} · вертикальная ось креста',source_channel=3-ch)
    return items


def pair_intersections(model,q,f,observations,sizes,bounds,h):
    """Bracket intersections on contrast grid and refine actual forward roots."""
    rows=[]
    for a,b in combinations(range(len(observations)),2):
        if sizes[a]==sizes[b]:continue
        ia,ib=[int(np.where(SIZES==sizes[i])[0][0]) for i in (a,b)]
        ya,yb=observations[a],observations[b]
        differences=ya*f[ib]-yb*f[ia]
        brackets=np.flatnonzero(differences[:-1]*differences[1:]<0)
        roots=[]
        for j in brackets:
            def residual(contrast):
                z=model.evaluate([1.,contrast])[0]
                return ya*z[ib]-yb*z[ia]
            roots.append(brentq(residual,q[j],q[j+1],xtol=1e-12))
        roots.extend(q[np.flatnonzero(differences==0)].tolist())
        for contrast in np.unique(roots):
            unit=model.evaluate([1.,contrast])[0]
            rho1=ya/unit[ia];rho2=contrast*rho1
            p=np.array([rho1,rho2])
            if not np.all((p>=bounds[:,0]-1e-9)&(p<=bounds[:,1]+1e-9)):continue
            z=model.evaluate(p)[0]
            pair_error=max(abs(z[ia]-ya),abs(z[ib]-yb))
            if pair_error>1e-7:raise ValueError('Intersection root failed forward substitution')
            reference_error=float(np.max(abs(z-observations[:5])/observations[:5])*100)
            rows.append([a,b,rho1,rho2,h,pair_error,reference_error])
    return rows


def run(out=OUT):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    observations,_,input_contract=prepare(EXPERIMENT)
    input_contract['assumptions'] = [a for a in input_contract['assumptions']
        if not a.startswith('Two global numerical channel assignments')]
    input_contract['assumptions'].append(
        'Author-confirmed 001–007 roles and 004–007 transition; 008–011 continue the late-block protocol. '
        'Former constant-number cases are counterfactual diagnostics.')
    registry=ROOT/'MATLAB_TRKG4_real_subjects/configs/tissue_literature_v2.json'
    compartments=read(registry)['screening_profiles']['100000']['compartments']
    bounds=np.array([compartments[k]['rho_bounds_ohm_m'] for k in ('soft','lung_unspecified')])
    q=np.geomspace(bounds[1,0]/bounds[0,1],bounds[1,1]/bounds[0,0],481)
    cases=[]
    with threadpool_limits(3):
        bank=KernelBank(h=H,q=q)
        np.savez_compressed(out/'kernel.local.npz',h=H,q=q,f=bank.f,bounds=bounds)
        for channel in (3,1,2):
            items=definition(channel)
            for state in ('inhale','exhale'):
                y=[]
                for item in items:
                    selected=observations.query('record_id==@item["record_id"] and channel==@item["channel"] and state==@state')
                    if len(selected)!=1:raise ValueError('Missing or duplicate observation: '+item['key']+' '+state)
                    y.append(float(selected.iloc[0].baseline_raw_median_ohm))
                y=np.array(y);fitted=bounded_fit(y[:5],bank,bounds);p=profile(bank,y[:5],bounds)
                rows=[]
                for ih,h in enumerate(H):
                    model=DiskModel(SIZES,h,radial=2,angular=8,terms=2048)
                    rows.extend(pair_intersections(model,q,bank.f[ih],y,[i['size_mm'] for i in items],bounds,h))
                key=f'channel_{channel}_{state}'
                intersections=np.array(rows).reshape(-1,7)
                np.savez_compressed(out/(key+'.npz'),observations=y,intersections=intersections,
                    profile_rho1=p['rho1'],profile_rho2=p['rho2'],profile_rss=p['h_rss'])
                cases.append(dict(key=key,channel=channel,state=state,assemblies=items,fit=fitted,
                    intersection_points=len(rows),maximum_pair_residual_ohm=float(intersections[:,5].max()) if len(rows) else 0.))
                print(key,'bounded fit',fitted['parameters'],'intersections',len(rows),flush=True)
    save(out/'results.local.json',cases)
    save(out/'input_contract.local.json',dict(input_contract,
        active_inverse_bounds_ohm_m=bounds.tolist(),wide_technical_inverse_is_historical=True,
        physiological_bounds_status='literature-informed working hypothesis; not individual validated norms',
        jointly_fitted_reference_records=IDS,all_lateral_montages_displayed=True,
        auxiliary_observations_used_in_joint_fit=False,h_grid_mm=H.tolist(),q_grid=q.tolist(),
        surface_rule='Z_L=rho1*f_L(rho2/rho1,h); rho1=Zobs/f_L; clipped to both material bounds',
        pair_intersections_rule='sign-changing contrast roots refined by brentq at each h; tangent or between-h roots may be missed',
        default_slice_h_mm=20.,slice_h_is_anatomical_measurement=False,
        registry_sha256=sha(registry),producer_sha256=sha(__file__),
        dependencies={n:sha(Path(__file__).with_name(n)) for n in ['lateral_pilot.py','lateral_identifiability.py','lateral_admissible_sets.py']},
        annotations_accepted=False,parameters_transferred_to_cardiac_function=False))
    contract=read(out/'input_contract.local.json')
    contract.update(role_source='author confirmation 2026-10-09',
        reference_channels={'004':2,'007':1,'008':1,'009':1,'010':1},
        direct_author_role_records=['001','002','003','004','005','006','007'],
        late_block_number_continuity_is_protocol_assumption=True,
        previous_constant_number_cases_are_counterfactual=True)
    save(out/'input_contract.local.json',contract)
    numerical_names=['kernel.local.npz','results.local.json','input_contract.local.json']
    numerical_names.extend(case['key']+'.npz' for case in cases)
    # Reader files have their own manifest and change on re-export.
    save(out/'manifest.local.json',dict(outputs={name:sha(out/name) for name in numerical_names}))
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT)
    run(parser.parse_args().out)
