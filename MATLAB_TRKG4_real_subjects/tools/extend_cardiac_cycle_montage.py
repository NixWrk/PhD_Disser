"""Add one independently prepared CEM montage to an immutable phase-block run.

No source run file is modified. Preparation binds existing verified phase blocks,
new contacts, producer code and the original plan. Each worker checks its inputs.
"""
from pathlib import Path
import argparse
import sys
import json
import heart_cardiac_cycle_electrical as cycle
import heart_replacement_electrical_pilot as core


def record(path):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':core.sha(path)}


def verify(records):
    for r in records:
        core.require(core.sha(r['path'])==r['sha256'],'Changed sealed input: '+r['path'])


def prepare(args):
    source=Path(args.source).resolve(); output=Path(args.output).resolve()
    core.require(not output.exists() or not any(output.iterdir()),'Output must be empty')
    p=cycle.load_plan(source)
    prep_path=Path(args.preparation).resolve(); prep=core.read(prep_path)
    core.require(prep['status']=='passed','Contact preparation failed')
    core.require(prep['prepared_sha256']==p['source_prepared_sha256'],'Different FEM geometry')
    core.require(prep['electrode_ids']==list(cycle.ELECTRODES),'Wrong contact order')
    core.require(prep['electrode_order']==['I_plus','V_plus','V_minus','I_minus'],'Wrong role order')
    entries=prep['montages'];entries=entries if isinstance(entries,list) else [entries]
    c=next(x for x in entries if x['id']==args.montage)
    core.require(c['status']=='passed' and c['matrix_error']<1e-12,'Contact assembly failed')
    contact=prep_path.parent/Path(c['contact_matrix']).name
    core.require(core.sha(contact)==c['contact_sha256'],'Contact hash differs')
    manifest=prep_path.parent.parent/'manifest.json'
    core.require(core.sha(manifest)==prep['manifest_sha256'],'Contact manifest differs')
    inputs=[record(source/'plan.json'),record(prep_path),record(manifest),record(contact)]
    md=core.read(manifest)
    for x in md['inputs']:
        raw=manifest.parent/x['json'];core.require(core.sha(raw)==x['sha256'],'Contact source changed')
        inputs.append(record(raw))
    states=[]
    for s in p['states']:
        e=cycle._checked_export(source,p,s)
        states.append(dict(s,blocks_path=str(source/s['id']/'blocks.mat'),blocks_sha256=e['blocks_sha256']))
    implementation=p['implementation']+[record(__file__)]
    manifest_producer=manifest.parent/'producer_hashes.json'
    core.require(manifest_producer.is_file(),'Contact producer code provenance is required')
    producer=core.read(manifest_producer);verify(producer)
    implementation+=producer
    result={k:p[k] for k in ('runtime','threads','memory_limit_gib','node_count','ground_zero_based',
        'source_prepared','source_prepared_sha256','conductivity_by_tissue_id','rho_ohm_m','criteria','frequency_hz')}
    result.update(schema='cardiac_cycle_montage_extension_v1',source_run=str(source),
        source_plan_sha256=core.sha(source/'plan.json'),inputs=inputs,implementation=implementation,
        states=states,montage=args.montage,required_solutions=len(states),
        contact=dict(id=args.montage,path=str(contact),sha256=c['contact_sha256'],
            areas_mm2=c['areas_mm2'],actual_centres_xyz_mm=c['actual_centres_xyz_mm']),
        contact_parameters=prep['contact'],mask_policy=p['mask_policy'],scientific_status=cycle.STATUS)
    output.mkdir(parents=True,exist_ok=True);core.write(output/'plan.json',result)
    (output/'plan.sha256').write_text(core.sha(output/'plan.json')+'\n',encoding='utf8')
    print('Prepared',len(states),'additional solves')


def load(output):
    path=output/'plan.json'
    core.require(core.sha(path)==(output/'plan.sha256').read_text().strip(),'Plan seal mismatch')
    p=core.read(path);verify(p['implementation']);verify(p['inputs'])
    core.require(p['schema']=='cardiac_cycle_montage_extension_v1','Wrong extension schema')
    core.require(len(p['states'])==p['required_solutions']==12,'Expected twelve phases')
    core.require(len({s['id'] for s in p['states']})==12,'Duplicate phase')
    return p


def completed(output,p,s):
    target=output/s['id'];receipt=target/'completion.json'
    if not receipt.is_file():return None
    d=core.read(receipt)
    core.require(d['plan_sha256']==core.sha(output/'plan.json'),'Wrong completion plan')
    for name,h in d['hashes'].items():core.require(core.sha(target/name)==h,'Changed result '+name)
    r=core.read(target/'result.json')
    core.require(r['blocks_sha256']==s['blocks_sha256'] and r['contact_sha256']==p['contact']['sha256'],'Wrong result provenance')
    core.require(r['state_id']==s['id'] and r['montage']==p['montage'],'Wrong result identity')
    return r


def worker(args):
    output=Path(args.output).resolve();p=load(output)
    s=next(x for x in p['states'] if x['id']==args.state)
    core.require(core.sha(s['blocks_path'])==s['blocks_sha256'],'Phase blocks changed')
    if completed(output,p,s):return
    target=output/s['id'];core.require(not target.exists(),'Incomplete state requires inspection')
    target.mkdir()
    runner=core.configure(Path(p['runtime']['deps']),p['threads'])
    nn=p['node_count'];gnd=p['ground_zero_based'];c=p['contact']
    blocks=cycle._load_fractional_blocks(runner,Path(s['blocks_path']),nn,gnd)
    C=runner.load_contact(c['path'],nn,4,gnd,p['source_prepared_sha256'],list(cycle.ELECTRODES))
    K=C.copy()
    for sig,b in zip(p['conductivity_by_tissue_id'],blocks):K=K+sig*b
    q=runner.channel_rhs(nn,cycle.montage_spec())
    fields,r=core.solve_checked(K,C,q,gnd,p['criteria'],p['memory_limit_gib'],target)
    r.update(id=s['id'],state_id=s['id'],montage=p['montage'],rr_percent=s['rr_percent'],
             blocks_sha256=s['blocks_sha256'],contact_sha256=c['sha256'],
             source_prepared_sha256=p['source_prepared_sha256'],plan_sha256=core.sha(output/'plan.json'),
             scientific_status=p['scientific_status'],frequency_hz=p['frequency_hz'])
    runner.np.savez_compressed(target/'fields.npz',potentials_ohm=fields)
    core.write(target/'result.json',r);verify(p['implementation']);verify(p['inputs'])
    core.write(target/'completion.json',dict(plan_sha256=core.sha(output/'plan.json'),
        hashes={name:core.sha(target/name) for name in ('fields.npz','result.json','factorization_memory.json')}))
    print(json.dumps({'phase':s['id'],'Z_ohm':r['Z_ohm']}))


def summarize(output,p=None):
    output=Path(output).resolve();p=p or load(output)
    rows=[r for s in p['states'] if (r:=completed(output,p,s)) is not None]
    d=dict(status='complete' if len(rows)==len(p['states']) else 'incomplete',
        completed=len(rows),required=len(p['states']),states=rows,contact=p['contact'],
        source_plan_sha256=p['source_plan_sha256'],plan_sha256=core.sha(output/'plan.json'))
    core.write(output/'summary.json',d);return d


def run(args):
    output=Path(args.output).resolve();p=load(output)
    core.configure(Path(p['runtime']['deps']),p['threads'])
    core.require(core.sha(p['source_prepared'])==p['source_prepared_sha256'],'Source mesh changed')
    for s in p['states']:
        if completed(output,p,s):continue
        core.monitored([p['runtime']['python'],'-B','-X','utf8',str(Path(__file__).resolve()),
            'worker','--output',str(output),'--state',s['id']],output,'solve_'+s['id'],p['memory_limit_gib'])
        summarize(output,p)
    verify(p['implementation']);verify(p['inputs'])
    for s in p['states']:core.require(core.sha(s['blocks_path'])==s['blocks_sha256'],'Phase blocks changed during run')
    d=summarize(output,p);core.require(d['status']=='complete','Incomplete extension')
    core.write(output/'status.json',dict(status='complete',completed=d['completed']))
    print('Complete:',d['completed'])


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--source',required=True);q.add_argument('--preparation',required=True);q.add_argument('--montage',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('run');q.add_argument('--output',required=True)
    q=sub.add_parser('worker');q.add_argument('--output',required=True);q.add_argument('--state',required=True)
    args=parser.parse_args();{'prepare':prepare,'run':run,'worker':worker}[args.command](args)
if __name__=='__main__':main()
