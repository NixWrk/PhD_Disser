"""Summarize executed, conditional STL/FEM volume probes, never clinical SV.

The seven shapes receive the same absolute volume increments. A three-point
quadratic is an interpolation of forward solutions, not an AI predictor.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(reports, output):
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for path in reports:
        result=json.loads(path.read_text(encoding='utf-8-sig'))
        assert result['status']=='conditional_STL_geometry_electrical_calculations_completed'
        rows.extend(result['rows'])
    byid={r['id']:r for r in rows}
    assert len(byid)==len(rows), 'Duplicate executed identifiers'
    ref=byid['heart4_STL_v100_q64']
    reference_volume=ref['target_volume_ml']
    delta=np.array([-.05,0,.05])*reference_volume
    methods=['STL','G1','G2','G3a','G3b','L1','L2']
    fits={}; table=[]
    for method in methods:
        triplet=[byid[f'heart4_{method}_v{factor:03d}_q64'] for factor in [95,100,105]]
        z=np.array([r['Ztt_ohm'] for r in triplet])
        q=np.polyfit(delta,z,2)
        fits[method]=(q,z-z[1])
        table.append({'method':method,'nominal_target_volume_ml':triplet[1]['target_volume_ml'],
                      'nominal_volume_error_pct':100*(triplet[1]['target_volume_ml']/reference_volume-1),
                      'Z0_ohm':z[1],'dZ_dV_mohm_per_ml':q[1]*1000,
                      'quadratic_coefficient_mohm_per_ml2':q[0]*1000,
                      'positive_probe_mohm':1000*(z[2]-z[1]),
                      'negative_probe_mohm':1000*(z[0]-z[1]),
                      'max_material_volume_error_ml':max(abs(r['volume_ml'][3]-r['target_volume_ml']) for r in triplet)})
    for r in table:
        curve=fits[r['method']][1]
        r['electrical_delta_RMSE_mohm_vs_STL']=float(np.sqrt(np.mean((curve-fits['STL'][1])**2))*1000)
        r['slope_relative_difference_pct_vs_STL']=float(100*(fits[r['method']][0][1]/fits['STL'][0][1]-1))
    convergence=[]
    for q in [64,256]:
        for step in [2,5]:
            minus=byid[f'heart4_STL_v{100-step:03d}_q{q}']
            plus=byid[f'heart4_STL_v{100+step:03d}_q{q}']
            dx=plus['target_volume_ml']-minus['target_volume_ml']
            dz=plus['Ztt_ohm']-minus['Ztt_ohm']
            actualdx=plus['volume_ml'][3]-minus['volume_ml'][3]
            convergence.append({'q':q,'step_pct':step,'dZ_dV_target_mohm_per_ml':1000*dz/dx,
                'dZ_dV_integrated_mohm_per_ml':1000*dz/actualdx,
                'target_volume_span_ml':dx,'integrated_volume_span_ml':actualdx})
    scenarios=[]
    for label in ['heart4','blood135','conditional02']:
        rr=[byid[f'{label}_STL_v{x:03d}_q256'] for x in [95,100,105]]
        slope=(rr[2]['Ztt_ohm']-rr[0]['Ztt_ohm'])/(.1*reference_volume)
        scenarios.append({'scenario':label,'heart_rho_ohm_m':rr[1]['rho_ohm_m'][3],
                          'dZ_dV_mohm_per_ml':1000*slope,'Z0_ohm':rr[1]['Ztt_ohm'],
                          'eit_relative_residual_pct':rr[1]['eit_relative_residual_pct'],
                          'Ktt_ohm_per_S_m':rr[1]['Ktt_ohm_per_S_m']})
    result={'status':'conditional_forward_comparison_not_SV_validation','sources':{str(p):sha(p) for p in reports},
            'implementation_sha256':sha(__file__),'reference_STL_volume_ml':reference_volume,
            'reference_volume_probe_ml':delta.tolist(),'shape_comparison':table,
            'local_quadrature_and_step_check':convergence,'heart_material_scenarios':scenarios,
            'assumptions':['same background CT and montage across shapes','heart replaces intersected background',
              'heart rho=4 is diagnostic, not measured effective whole-heart resistivity',
              'blood135 homogeneous heart is an endpoint scenario, not a pure-blood anatomical mask',
              'no cardiac-motion registration or independent clinical SV reference',
              'soft/lung pulse contribution is not set to zero; reported derivatives support its explicit estimation']}
    (output/'summary.local.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (output/'shape_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    fig,axs=plt.subplots(1,3,figsize=(15,4.8))
    for method in methods:
        axs[0].plot(delta,1000*fits[method][1],marker='o',label=method)
    axs[0].set(xlabel='Изменение заданного объёма, мл',ylabel='Изменение ТТРКГ, мОм',title='Одинаковое ΔV; ρ сердца = 4 Ом·м')
    axs[0].legend(fontsize=8,ncol=2)
    axs[1].bar(methods,[r['electrical_delta_RMSE_mohm_vs_STL'] for r in table])
    axs[1].set(ylabel='RMSE приращений относительно STL, мОм',title='Расхождение при одинаковом ρ')
    # Three separate material scenarios, not a sampled continuous rho law.
    axs[2].scatter([x['heart_rho_ohm_m'] for x in scenarios],[x['dZ_dV_mohm_per_ml'] for x in scenarios])
    axs[2].axhline(0,color='black',lw=.6)
    axs[2].set(xlabel='Заданное ρ целого сердца, Ом·м',ylabel='dZ/dV, мОм/мл',title='Три сценария; одна STL и монтаж')
    for ax in axs:ax.grid(alpha=.2)
    fig.suptitle('Условные FEM-расчёты на КТ Ника; пригодность сферы и точность УО не установлены',fontsize=12)
    fig.tight_layout();fig.savefig(output/'electrical_approximation.png',dpi=170);plt.close(fig)
    print(json.dumps({'shape_comparison':table,'convergence':convergence,'scenarios':scenarios},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reports',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.reports,a.output)
