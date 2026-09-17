"""Four literature scenarios on twelve Nix phases and seven unchanged CEM montages.

Adds a fifth fat material. No source plan or sealed implementation is modified.
Every channel gets an independent factorization and direct/reciprocal solve.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import os
from pathlib import Path
import sys
import time
import heart_replacement_electrical_pilot as core
import heart_cardiac_cycle_electrical as cycle

ROOT=Path(__file__).resolve().parents[1]


def record(path):
    p=Path(path).resolve();return {'path':str(p),'sha256':core.sha(p)}


def prepare(args):
    out=Path(args.output).resolve();cfg=core.read(args.config)
    core.require(not (out/'plan.json').exists(),'New plan directory required')
    old=Path(cfg['old_run']);base=cycle.load_plan(old/'electrical')
    ext=core.read(old/'ttrkg_electrical/plan.json')
    geometry_path=Path(cfg['output'])/'geometry.json';geo=core.read(geometry_path)
    core.verify_files(geo['inputs']);core.verify_files(geo['implementation'])
    registry_path=ROOT/'configs/tissue_literature_v2.json';registry=core.read(registry_path)
    core.require(geo['source_prepared_sha256']==base['source_prepared_sha256']==ext['source_prepared_sha256'],'Different mesh')
    contacts=base['contacts']+[ext['contact']]
    core.require(len(contacts)==7 and len({c['id'] for c in contacts})==7,'Expected seven montages')
    for c in contacts:core.require(core.sha(c['path'])==c['sha256'],'Contact changed')
    tissue=registry['tissues'];rho=lambda k:tissue[k]['gabriel_at_frequency']['50000']['rho_ohm_m']
    scenarios=[]
    for lung_source,lung in [('gabriel',rho('Lung (Inflated)')),('lf_mean',1/tissue['Lung (Inflated)']['low_frequency_up_to_1MHz']['mixed']['mean_S_m'])]:
        for heart_source in ['myocardium','blood']:
            heart=rho('Heart Muscle' if heart_source=='myocardium' else 'Blood')
            values=[5.,lung,heart,rho('Bone (Cortical)'),rho('Fat')]
            key=lung_source+'_'+heart_source
            scenarios.append({'id':key,'lung_source':lung_source,'heart_source':heart_source,
                'rho_by_material':values,'conductivity_by_material':[1/x for x in values],
                'rationale':{'muscle_background':'User-selected 5 ohm m within literature range; not the Gabriel point or a measured individual value',
                'lung':'Gabriel at 50 kHz' if lung_source=='gabriel' else 'Reciprocal of LF pooled mean conductivity; applying it at 50 kHz is an uncertainty scenario',
                'heart':'Whole observed mask assigned '+heart_source+' properties; not a chamber-only geometry',
                'fat':'Gabriel Fat at 50 kHz; static subcutaneous_fat union torso_fat',
                'bone':'Gabriel cortical bone at 50 kHz'}})
    profile=registry['screening_profiles']['50000']
    for s in scenarios:
        for key,value in zip(['muscle','lung_inflated',s['heart_source'],'bone_cortical','fat'],s['rho_by_material']):
            lo,hi=profile['compartments'][key]['rho_bounds_ohm_m'];core.require(lo<=value<=hi,'Outside literature range')
    states=[]
    for s in geo['states']:
        old_state=next(v for v in base['states'] if v['id']==s['id'])
        export=cycle._checked_export(old/'electrical',base,old_state)
        states.append(dict(s,fractions_path=str(geometry_path.parent/s['fraction_mat']),
            old_blocks_path=str(old/'electrical'/s['id']/'blocks.mat'),old_blocks_sha256=export['blocks_sha256']))
    inputs=geo['inputs']+[record(geometry_path),record(old/'electrical/plan.json'),record(old/'ttrkg_electrical/plan.json'),record(registry_path)]
    for s in states:inputs.append(record(s['fractions_path']))
    implementation=geo['implementation']+[record(__file__),record(cycle.__file__),record(ROOT/'src/run_cardiac_cycle_fat_export.m')]
    out.mkdir(parents=True,exist_ok=True)
    plan={k:base[k] for k in ('runtime','threads','memory_limit_gib','source_prepared','source_prepared_sha256','node_count','element_count','ground_zero_based','criteria','contact_parameters')}
    plan.update(schema='cardiac_cycle_fat_electrical_v1',frequency_hz=50000,geometry_path=str(geometry_path),
        background_path=str(geometry_path.parent/'background.mat'),background_sha256=geo['background_mat_sha256'],
        inputs=inputs,implementation=implementation,states=states,contacts=contacts,scenarios=scenarios,
        material_order=geo['material_order'],registry_id=registry['registry_id'],registry_sha256=core.sha(registry_path),
        profile_id=profile['id'],required_solutions=12*7*4,physical_validation=False,spatial_convergence=False,
        limitations=['Observed whole-heart mask touches CT boundary; no cap extrapolation',
        'Muscle conductivity assigned to remaining background; not anatomical pure muscle',
        'Fat mask is static and automatic; outside acquired CT prior background is retained',
        'Whole-heart blood scenario is not chamber-only segmentation',
        'Four nominal scenarios are not a full literature-range sensitivity sweep'])
    core.write(out/'plan.json',plan)
    (out/'plan.sha256').write_text(core.sha(out/'plan.json')+'\n',encoding='utf-8')
    print('Prepared',plan['required_solutions'],'solutions',flush=True)


def plan_at(out):
    p=core.read(out/'plan.json')
    core.require(core.sha(out/'plan.json')==(out/'plan.sha256').read_text().strip(),'Plan changed')
    core.verify_files(p['implementation'])
    return p


def worker(args):
    out=Path(args.output).resolve();p=plan_at(out)
    s=next(x for x in p['states'] if x['id']==args.phase)
    scenario=next(x for x in p['scenarios'] if x['id']==args.scenario)
    contact=next(x for x in p['contacts'] if x['id']==args.montage)
    export=core.read(out/s['id']/'export.json');blockpath=out/s['id']/'blocks.mat'
    core.require(export['status']=='passed' and export['plan_sha256']==core.sha(out/'plan.json'),'Invalid export')
    core.require(core.sha(blockpath)==export['blocks_sha256'],'Blocks changed')
    core.require(core.sha(contact['path'])==contact['sha256'],'Contact changed')
    target=out/'solutions'/scenario['id']/s['id']/contact['id']
    core.require(not target.exists(),'Incomplete result requires inspection');target.mkdir(parents=True)
    runner=core.configure(Path(p['runtime']['deps']),p['threads'])
    nn=p['node_count'];ground=p['ground_zero_based']
    C=runner.load_contact(contact['path'],nn,4,ground,p['source_prepared_sha256'],list(cycle.ELECTRODES))
    K=C.copy()
    with runner.h5py.File(blockpath) as f:
        refs=f['blocks'][:].ravel();core.require(len(refs)==5,'Expected five materials')
        core.require(int(f['nn'][0,0])==nn and int(f['ground'][0,0])-1==ground,'Node/ground mismatch')
        for sigma,ref in zip(scenario['conductivity_by_material'],refs):
            g=f[ref];size=int(g.attrs['MATLAB_sparse']);core.require(size==nn+4,'Block shape mismatch')
            B=runner.sp.csc_matrix((g['data'][:],g['ir'][:],g['jc'][:]),shape=(size,size)).tocsr()
            core.require(not B[nn:,:].nnz and not B[:,nn:].nnz,'Electrode entries in volume matrix')
            K=K+sigma*B
    q=runner.channel_rhs(nn,cycle.montage_spec())
    fields,row=core.solve_checked(K,C,q,ground,p['criteria'],p['memory_limit_gib'],target)
    runner.np.savez_compressed(target/'fields.npz',potentials_ohm=fields)
    row.update(id=s['id'],montage=contact['id'],scenario=scenario['id'],rr_percent=s['rr_percent'],
        rho_by_material=scenario['rho_by_material'],plan_sha256=core.sha(out/'plan.json'),
        blocks_sha256=export['blocks_sha256'],contact_sha256=contact['sha256'])
    core.write(target/'result.json',row)
    core.write(target/'completion.json',{'plan_sha256':row['plan_sha256'],'status':'passed',
        'hashes':{name:core.sha(target/name) for name in ['fields.npz','result.json','factorization_memory.json']}})


def completed(out,p,scenario,phase,montage):
    target=out/'solutions'/scenario/phase/montage
    if not (target/'completion.json').exists():return None
    c=core.read(target/'completion.json')
    core.require(c['status']=='passed' and c['plan_sha256']==core.sha(out/'plan.json'),'Completion plan mismatch')
    for name,digest in c['hashes'].items():core.require(core.sha(target/name)==digest,'Result changed')
    row=core.read(target/'result.json')
    core.require((row['scenario'],row['id'],row['montage'])==(scenario,phase,montage),'Result identity mismatch')
    return row


def run(args):
    out=Path(args.output).resolve();p=plan_at(out);core.verify_files(p['inputs'])
    core.configure(Path(p['runtime']['deps']),p['threads'])
    lock=out/'run.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        quote=lambda s:"'"+str(s).replace('\\','/').replace("'","''")+"'"
        code='run('+quote(p['runtime']['eidors_startup'])+');addpath('+quote(ROOT/'src')+');run_cardiac_cycle_fat_export('+quote(out/'plan.json')+');'
        core.monitored([p['runtime']['matlab'],'-batch',code],out,'export',p['memory_limit_gib'])
        jobs=[(s['id'],phase['id'],c['id']) for phase in p['states'] for s in p['scenarios'] for c in p['contacts']]
        todo=[j for j in jobs if completed(out,p,*j) is None]
        def execute(job):
            scenario,phase,montage=job;stage='_'.join(job)
            monitor=out/'monitors'/stage;monitor.mkdir(parents=True,exist_ok=True)
            command=[p['runtime']['python'],'-B','-X','utf8',str(Path(__file__).resolve()),'worker','--output',str(out),'--phase',phase,'--scenario',scenario,'--montage',montage]
            core.monitored(command,monitor,'solve',p['memory_limit_gib'])
            return completed(out,p,*job)
        # Two processes, each guarded; no shared factorization across montages.
        with ThreadPoolExecutor(max_workers=2) as pool:
            queue=iter(todo);pending=set();count=len(jobs)-len(todo)
            for _ in range(2):
                job=next(queue,None)
                if job is not None:pending.add(pool.submit(execute,job))
            while pending:
                done,pending=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    row=future.result();count+=1
                    core.write(out/'progress.json',{'completed':count,'required':len(jobs),'last':[row['scenario'],row['id'],row['montage']]})
                    print('Completed',count,'/',len(jobs),flush=True)
                    job=next(queue,None)
                    if job is not None:pending.add(pool.submit(execute,job))
        rows=[completed(out,p,*j) for j in jobs];core.require(all(r is not None for r in rows),'Missing solution')
        core.verify_files(p['inputs']);core.verify_files(p['implementation'])
        core.write(out/'summary.json',{'status':'complete','completed':len(rows),'required':len(jobs),
            'plan_sha256':core.sha(out/'plan.json'),'scenarios':p['scenarios'],'solutions':rows,'limitations':p['limitations']})
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);sp=ap.add_subparsers(dest='command',required=True)
    x=sp.add_parser('prepare');x.add_argument('--config',required=True);x.add_argument('--output',required=True)
    x=sp.add_parser('run');x.add_argument('--output',required=True)
    x=sp.add_parser('worker');x.add_argument('--output',required=True);x.add_argument('--phase',required=True);x.add_argument('--scenario',required=True);x.add_argument('--montage',required=True)
    args=ap.parse_args();{'prepare':prepare,'run':run,'worker':worker}[args.command](args)
