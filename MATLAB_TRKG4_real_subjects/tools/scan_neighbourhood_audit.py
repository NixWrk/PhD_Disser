"""Audit how the parameter-grid neighbourhood maps to actual CT coordinates."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/exploratory/scan_localization_20260910'

def main(out=OUT):
    d=np.load(out/'scan_arrays.npz');m=np.load(out/'analysis_maps.npz');g=pd.read_csv(out/'centres_geometry.csv');q=pd.read_csv(out/'candidate_requests.csv')
    A=len(d['phi']);U=len(d['u']);V=len(d['v']);lookup=np.full(d['status'].size,-1,int);lookup[d['flat']]=np.arange(len(d['flat']))
    xyz=g[['x_mm','y_mm','z_mm']].to_numpy();normal=g[['nx','ny','nz']].to_numpy();records=[]
    ci_valid=np.flatnonzero(np.any(d['status']==1,axis=1));unique=np.full(len(g),-1,int);spread=np.full(len(g),np.nan)
    for ci in ci_valid:
        v,u=divmod(ci,U)
        if v<2 or v>=V-2 or u<2 or u>=U-2:continue
        idx=np.array([vv*U+uu for vv in range(v-2,v+3) for uu in range(u-2,u+3)])
        points=xyz[idx];unique[ci]=len(np.unique(np.round(points,6),axis=0));centered=points-points.mean(axis=0)
        spread[ci]=np.linalg.svd(centered,compute_uv=False)[1]/np.sqrt(25)
    distinct=(unique==25)&(spread>1e-3)
    extra=[]
    for metric in ['robust_experiment','robust_joint']:
        z=np.where(distinct[:,None],m[metric],np.nan)
        if np.isfinite(z).any():extra.append((int(np.nanargmin(z)),'minimum_'+metric+'_distinct_centres'))
    f=np.load(out/'layered_fits.npz')
    extra.extend((int(k),'both_RMSE_below_5_ohm') for k in d['flat'][(d['rmse_experiment']<=5)&(f['rmse_ohm']<=5)])
    pose_table=pd.read_csv(out/'pose_analysis.csv').set_index('flat_index',drop=False)
    for flat,reason in extra:
        if flat in q.flat_index.values:continue
        row=pose_table.loc[flat].to_dict();row.update(candidate_id=f'C{len(q)+1:02d}',reasons=reason)
        row.update(dict(zip(['x_mm','y_mm','z_mm'],xyz[flat//A])))
        q=pd.concat([q,pd.DataFrame([row])],ignore_index=True)
    q.to_csv(out/'candidate_requests.csv',index=False)
    analysis=json.loads((out/'analysis_summary.json').read_text(encoding='utf-8'));analysis['candidates']=len(q)
    (out/'analysis_summary.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2),encoding='utf-8')
    for _,c in q.iterrows():
        ci=int(c.centre_index)-1;v,u=divmod(ci,U);ai=np.flatnonzero(d['phi']==c.phi_deg)[0]
        flats=np.array([(vv*U+uu)*A+aa for vv in range(max(0,v-2),min(V,v+3)) for uu in range(max(0,u-2),min(U,u+3)) for aa in range(max(0,ai-2),min(A,ai+3))])
        state=d['status'].ravel()[flats];valid=flats[state==1];rows=lookup[valid];idx=np.unique(flats//A)
        records.append(dict(candidate_id=c.candidate_id,neighbours_in_grid=len(flats),valid=int((state==1).sum()),
            qc_invalid=int((state==2).sum()),excluded=int((state==3).sum()),missing=int((state==0).sum()),
            unique_projected_centres=int(len(np.unique(np.round(xyz[idx],6),axis=0))),
            second_spatial_std_mm=float(spread[ci]),max_centre_shift_mm=float(np.linalg.norm(xyz[idx]-xyz[ci],axis=1).max()),
            max_normal_change_deg=float(np.degrees(np.arccos(np.clip(normal[idx]@normal[ci],-1,1))).max()),
            rmse_min_valid_ohm=float(d['rmse_experiment'][rows].min()),rmse_max_valid_ohm=float(d['rmse_experiment'][rows].max()),
            worst_flat_index=int(valid[np.argmax(d['rmse_experiment'][rows])])) )
    pd.DataFrame(records).to_csv(out/'candidate_neighbourhood_diagnostics.csv',index=False)
    pd.DataFrame(dict(centre_index=ci_valid+1,unique_projected_centres_in_5x5=unique[ci_valid],second_spatial_std_mm=spread[ci_valid])).to_csv(out/'centre_neighbourhood_geometry.csv',index=False)
    # A transparent extra scenario: every nominal centre is spatially distinct,
    # and the 25 points span two directions (numerical rank tolerance 1e-3 mm).
    distinct=(unique==25)&(spread>1e-3);filtered=np.where(distinct[:,None],m['robust_experiment'],np.nan)
    ok=np.isfinite(filtered);best=int(np.nanargmin(filtered)) if ok.any() else None
    np.savez_compressed(out/'neighbourhood_geometry.npz',unique_centres=unique,second_spatial_std_mm=spread,robust_experiment_distinct=filtered)
    result=dict(coordinate_rounding_mm=1e-6,valid_grid_centres=len(ci_valid),
        distinct_projected_centres=int(len(np.unique(np.round(xyz[ci_valid],6),axis=0))),
        collapsed_5x5_centres=int((unique[ci_valid]==1).sum()),
        full_125_valid_windows=int(np.isfinite(m['robust_experiment']).sum()),
        full_valid_windows_with_25_distinct_centres=int(ok.sum()),
        best_distinct_flat_index=best,best_distinct_robust_rmse_ohm=float(filtered.ravel()[best]) if best is not None else None,
        interpretation='A parameter step is not a fixed physical displacement. Collapsed projected centres do not test spatial robustness.',
        distinct_scenario='25 different projected centres, rank-2 spread > 0.001 mm; still not an isotropic millimetre placement tolerance.')
    (out/'neighbourhood_geometry_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
