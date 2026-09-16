"""Controlled resistivity audit of the first TTRKG cardiac-cycle phase.

Keep phase material fractions, mesh, contacts and current unchanged; change
only soft-tissue conductivity, then all tissue conductivities to the old test.
Does not alter the sealed 84-solution cycle run.
"""
from pathlib import Path
import argparse
import sys
import heart_cardiac_cycle_electrical as cycle
import extend_cardiac_cycle_montage as ext
import heart_replacement_electrical_pilot as core
ROOT=Path(__file__).resolve().parents[1]
CASES=('old_soft_only','all_old_tissues')

def worker(args):
    out=Path(args.output).resolve();p=ext.load(Path(args.source).resolve())
    state=p['states'][0];core.require(state['id']=='phase_00','Wrong reference phase')
    core.require(core.sha(state['blocks_path'])==state['blocks_sha256'],'Changed phase blocks')
    oldpath=ROOT/'output/nik_trkg4_arm_electrode_model_tests_contract.json'
    old=core.read(oldpath);byname={x['name']:x['sigma_s_per_m'] for x in old['tissues']}
    sig=list(p['conductivity_by_tissue_id'])
    sig[0]=byname['soft_tissue']
    if args.case=='all_old_tissues':sig=[byname[x] for x in ('soft_tissue','lungs','heart','bones')]
    target=out/args.case;core.require(not target.exists(),'Refuse overwrite');target.mkdir(parents=True)
    runner=core.configure(Path(p['runtime']['deps']),p['threads'])
    nn=p['node_count'];gnd=p['ground_zero_based'];c=p['contact']
    blocks=cycle._load_fractional_blocks(runner,Path(state['blocks_path']),nn,gnd)
    C=runner.load_contact(c['path'],nn,4,gnd,p['source_prepared_sha256'],list(cycle.ELECTRODES))
    K=C.copy()
    for sigma,block in zip(sig,blocks):K=K+float(sigma)*block
    q=runner.channel_rhs(nn,cycle.montage_spec())
    fields,row=core.solve_checked(K,C,q,gnd,p['criteria'],p['memory_limit_gib'],target)
    row.update(case=args.case,conductivity_by_soft_lung_heart_bone=sig,
        rho_by_soft_lung_heart_bone=[1/x for x in sig],blocks_sha256=state['blocks_sha256'],
        contact_sha256=c['sha256'],cycle_plan_sha256=core.sha(Path(args.source)/'plan.json'),
        old_contract_sha256=core.sha(oldpath),audit_code_sha256=core.sha(__file__),
        geometry_and_contacts_unchanged=True,scientific_status='controlled_numerical_comparison_not_physical_validation')
    core.write(target/'result.json',row)
    print(args.case,row['Z_ohm'])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True);parser.add_argument('--case',choices=CASES);args=parser.parse_args()
    if args.case:worker(args);return
    out=Path(args.output).resolve();core.require(not out.exists(),'New audit directory required');out.mkdir(parents=True)
    p=ext.load(Path(args.source).resolve());core.configure(Path(p['runtime']['deps']),p['threads'])
    for case in CASES:
        core.monitored([sys.executable,'-B','-X','utf8',str(Path(__file__).resolve()),'--source',str(Path(args.source).resolve()),'--output',str(out),'--case',case],out,case,p['memory_limit_gib'])
    baseline=core.read(Path(args.source)/'phase_00/result.json')
    core.write(out/'summary.json',dict(status='complete',baseline_Z_ohm=baseline['Z_ohm'],
        baseline_result_sha256=core.sha(Path(args.source)/'phase_00/result.json'),
        cases=[core.read(out/case/'result.json') for case in CASES],criteria=p['criteria']))
    print('Controlled comparison complete')
if __name__=='__main__':main()
