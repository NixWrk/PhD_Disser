"""Verify real-model map integrals and controlled equal-volume perturbations."""
from pathlib import Path
import argparse
import csv
import json
import time
import run_electrode_sensitivity as engine


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',required=True,type=Path)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--reference',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--deps',type=Path)
    parser.add_argument('--finite-differences',action='store_true')
    args=parser.parse_args()
    engine.initialize(args.deps,4)
    import numpy as np
    from sensitivity_probes import select_volume_probe,local_stiffness
    from electrode_sensitivity_contract import load_study
    study=load_study(args.study)
    if len(study['montages'])!=1 or len(study['states'])!=1:
        raise ValueError('Preparation control expects one montage and one state')
    montage,state=study['montages'][0],study['states'][0]
    run_identity=json.loads((args.run/'identity.json').read_text(encoding='utf-8'))
    if run_identity['study']!=study:raise ValueError('Study differs from the solved study')
    for source_path,expected in run_identity['inputs'].items():
        if engine.digest(source_path)!=expected:raise ValueError('Changed input '+source_path)
    completion=json.loads((args.run/'completion.json').read_text(encoding='utf-8'))
    for name,expected in completion['output_sha256'].items():
        if engine.digest(args.run/name)!=expected:raise ValueError('Changed result '+name)
    out=args.output
    if out.exists() and any(out.iterdir()):raise ValueError('Use an empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    source_files=[Path(__file__),Path(__file__).with_name('sensitivity_probes.py'),Path(engine.__file__),Path(__file__).with_name('electrode_sensitivity_contract.py')]
    with engine.zipfile.ZipFile(out/'source_used.zip','w',engine.zipfile.ZIP_DEFLATED) as archive:
        for source in source_files:archive.write(source,source.name)
    model=engine.load_model(args.study.parent/study['source']['prepared_fem'])
    with np.load(args.run/'map_geometry.npz') as f:
        geometry={k:f[k] for k in f.files}
    names={1:'soft_tissue',2:'lungs',3:'heart',4:'bones'}
    # This pilot uses the frozen Nix labels. Fail rather than transfer silently.
    if model['tissue_names']!=list(names.values()):raise ValueError('Unexpected pilot anatomy')
    with (args.run/'channel_results.csv').open(encoding='utf-8') as f:results=list(csv.DictReader(f))
    maps={}
    for row in results:
        stem=f"{row['montage']}__{row['state']}__{row['channel']}"
        with np.load(args.run/(stem+'_sensitivity.npz')) as f:maps[row['channel']]=f['dZ_dsigma']
    mapping={'soft_tissue':'soft','lungs':'lung','heart':'heart','bones':'bone'}
    checks=[]
    for row in results:
        values=maps[row['channel']]
        for label,tissue in names.items():
            got=float(values[geometry['tissue_id']==label].sum())
            expected=float(row['dZ_dsigma_'+mapping[tissue]])
            error=abs(got-expected)
            passed=error<=1e-8+1e-8*abs(expected)
            checks.append(dict(check='map_integral',channel=row['channel'],parameter=mapping[tissue],
                               actual=got,reference=expected,absolute_error=error,passed=passed))
    standard=next(r for r in results if r['channel']=='standard')
    with (args.reference/(montage['id']+'_results.csv')).open(encoding='utf-8') as f:old=list(csv.DictReader(f))
    old=next(r for r in old if all(np.isclose(float(r['sigma_'+t]),state['conductivity'][t],rtol=1e-12,atol=1e-14) for t in ['soft','heart','lung']))
    for key,newkey in [('Z_direct_ohm','Z_ohm')]+[(f'dZ_dsigma_{t}',f'dZ_dsigma_{t}') for t in ['soft','heart','lung']]:
        got,expected=float(standard[newkey]),float(old[key])
        error=abs(got-expected)
        checks.append(dict(check='frozen_40_05_regression',channel='standard',parameter=key,
                           actual=got,reference=expected,absolute_error=error,
                           passed=error<=1e-8+1e-7*abs(expected)))
    if set(maps)=={'standard','adjacent','cross'}:
        discrepancy=maps['cross']-maps['standard']-maps['adjacent']
        scale=max(float(np.max(np.abs(x))) for x in maps.values())
        error=float(np.max(abs(discrepancy)))
        checks.append(dict(check='four_electrode_channel_identity',channel='cross=standard+adjacent',parameter='element_derivatives',
                           actual=error,reference=0.,absolute_error=error,passed=error<=1e-8+1e-7*scale))
    probes={}
    probe_rows=[]
    for label in [1,3,2]:
        probe=select_volume_probe(geometry['centroid_m'],geometry['volume_m3'],geometry['tissue_id'],label,10.)
        probes[label]=probe
        np.savez(out/f'probe_tissue_{label}.npz',indices=probe['indices'],weights=probe['weights'],centre_m=probe['centre_m'])
        for channel,values in maps.items():
            D=float(values[probe['indices']]@probe['weights'])
            probe_rows.append(dict(channel=channel,tissue=names[label],effective_volume_ml=probe['effective_volume_ml'],
                                   support_volume_ml=probe['support_volume_ml'],element_count=probe['element_count'],
                                   terminal_weight=probe['terminal_weight'],centre_x_mm=probe['centre_m'][0]*1000,
                                   centre_y_mm=probe['centre_m'][1]*1000,centre_z_mm=probe['centre_m'][2]*1000,
                                   dZ_dsigma=D,delta_sigma_S_per_m=.001,predicted_delta_Z_mohm=D,
                                   representation=probe['representation']))
    engine.write_csv(out/'equal_volume_probes.csv',probe_rows)
    fd_rows=[]
    if args.finite_differences:
        nn=len(model['nodes']);blocks=engine.parameter_blocks(model,study['parameters'],len(montage['electrodes']))
        K=engine.load_contact(args.study.parent/montage['contact_matrix'],nn,len(montage['electrodes']),model['ground'],study['source']['prepared_sha256'])
        for key,B in blocks.items():K=K+state['conductivity'][key]*B
        probe=probes[3]
        B=local_stiffness(model['nodes'],model['elements'],probe['indices'],probe['weights'],K.shape[0])
        ch=next(c for c in montage['channels'] if c['id']=='standard')
        q=engine.channel_rhs(nn,dict(electrodes=montage['electrodes'],channels=[ch]))
        stem=f"{montage['id']}__{state['id']}__standard"
        with np.load(args.run/(stem+'_fields.npz')) as f:u,v=f['u'],f['v']
        derivative=float(-v@(B@u))
        from_map=float(maps['standard'][probe['indices']]@probe['weights'])
        checks.append(dict(check='local_matrix_vs_map',channel='standard',parameter='heart_local',
                           actual=derivative,reference=from_map,absolute_error=abs(derivative-from_map),
                           passed=abs(derivative-from_map)<=1e-9+1e-8*abs(derivative)))
        z0=float(q[:,1]@u)
        for h in [.001,.0005]:
            zz=[]
            for sign in [-1,1]:
                start=time.perf_counter()
                fields,residuals=engine.solve_system(K+sign*h*B,q[:,:1],model['ground'])
                z=float(q[:,1]@fields[:,0]);zz.append(z)
                delta=z-z0;linear=sign*h*derivative
                fd_rows.append(dict(delta_sigma_S_per_m=sign*h,Z_ohm=z,delta_Z_ohm=delta,
                                     linear_delta_Z_ohm=linear,linear_absolute_error_ohm=abs(delta-linear),
                                     relative_residual=float(max(residuals)),seconds=time.perf_counter()-start))
                print('Local heart perturbation',sign*h,'delta Z',delta,flush=True)
            finite=(zz[1]-zz[0])/(2*h)
            error=abs(finite-derivative)
            checks.append(dict(check='local_central_difference',channel='standard',parameter=f'heart_local_h_{h}',
                               actual=finite,reference=derivative,absolute_error=error,
                               passed=error<=1e-5+.005*abs(derivative)))
        engine.write_csv(out/'local_finite_differences.csv',fd_rows)
    engine.write_csv(out/'verification.csv',checks)
    report=dict(status='passed' if all(r['passed'] for r in checks) else 'failed',checks=len(checks),
                finite_difference_solutions=len(fd_rows),effective_probe_volume_ml=10.,
                perturbation_status='technical_scale_not_physiological_amplitude',
                physical_validation='not_evaluated',mesh_convergence='not_evaluated',
                source_sha256={p.name:engine.digest(p) for p in source_files},
                inputs_sha256={str(args.run/'completion.json'):engine.digest(args.run/'completion.json')})
    report['output_sha256']={p.name:engine.digest(p) for p in out.iterdir() if p.is_file()}
    engine.write_json(out/'checks.json',report)
    if report['status']!='passed':raise ArithmeticError('Preparation checks failed')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
