"""Solve the unchanged individual-heart state on a newly refined CEM mesh.

A baseline comparison diagnoses field discretization; it does not validate
volume derivatives or certify spatial/physical convergence.
"""
from pathlib import Path
import argparse
import sys
import json
import numpy as np
import heart_replacement_electrical_pilot as core
import heart_fractional_electrical_pilot as fractional


def run(a):
    source=a.baseline.resolve();directory=a.refined_export.resolve();out=a.output.resolve()
    p=fractional.load_plan(source)
    core.require(p['geometry']=='individual','Original individual geometry required')
    runner=core.configure(Path(p['runtime']['deps']),p['threads'])
    report_path=directory/'heart_refinement_export.json';report=core.read(report_path)
    core.require(report['status']=='passed_heart_refinement_export','Refined export QC failed')
    core.require(report['source_prepared_sha256']==p['source_prepared_sha256'],'Different source anatomy')
    core.require(report['original_contact_sha256']==p['contact']['sha256'],'Different original contacts')
    for name in ['original_nodes_preserved','outer_boundary_preserved','contact_faces_preserved']:
        core.require(report[name] is True,'Geometry changed: '+name)
    core.require(report['contact_embedding_relative_error']<1e-12 and report['matrix_relative_error']<1e-12,'Algebraic checks failed')
    prepared=directory/report['prepared_file'];contact=directory/report['contact_file']
    core.require(core.sha(prepared)==report['prepared_sha256'] and core.sha(contact)==report['contact_sha256'],'Changed refined export')
    state=next(s for s in p['states'] if s['parameter']=='baseline')
    reference=fractional.checked_result(source,p,state)
    core.require(reference is not None,'Original baseline is incomplete')
    records=[{'path':str(v),'sha256':core.sha(v)} for v in [prepared,contact,report_path,Path(__file__),
             Path(core.__file__),Path(runner.__file__),source/'plan.json',source/state['id']/'result.json']]
    out.mkdir(parents=True,exist_ok=True)
    if not a.worker:
        core.require(not (out/'result.json').exists(),'Do not overwrite a previous comparison')
        core.monitored([sys.executable,'-B','-X','utf8',str(Path(__file__).resolve()),
                        '--baseline',str(source),'--refined-export',str(directory),'--output',str(out),'--worker'],
                       out,'baseline_solve',p['memory_limit_gib'])
        return
    core.require(not (out/'result.json').exists(),'Existing result requires review')
    model=runner.load_model(prepared);nn=len(model['nodes'])
    core.require(nn==report['nodes'] and len(model['elements'])==report['elements'],'Dimensions differ')
    core.require(model['ground']==report['ground_zero_based'] and len(model['blocks'])==4,'Unexpected exported model')
    parameters=[{'id':k,'block_indices':[i]} for i,k in enumerate(('soft','lung','heart','bone'))]
    blocks=runner.parameter_blocks(model,parameters,4)
    C=runner.load_contact(contact,nn,4,model['ground'],report['prepared_sha256'])
    K=C.copy()
    for name,B in blocks.items():K=K+B/p['rho_ohm_m'][name]
    q=np.zeros((nn+4,2));q[nn:]=np.array([[1,0],[0,1],[0,-1],[-1,0]])
    fields,row=core.solve_checked(K,C,q,model['ground'],p['criteria'],p['memory_limit_gib'],out)
    np.savez_compressed(out/'fields.npz',potentials_ohm=fields)
    change=row['Z_ohm']-reference['Z_ohm']
    row.update(geometry='individual',montage=p['contact']['id'],frequency_hz=p['frequency_hz'],rho_ohm_m=p['rho_ohm_m'],
               coarse_Z_ohm=reference['Z_ohm'],fine_Z_ohm=row['Z_ohm'],change_ohm=change,
               relative_change=change/reference['Z_ohm'],coarse_nodes=p['node_count'],fine_nodes=nn,
               coarse_elements=p['element_count'],fine_elements=report['elements'],
               source_plan_sha256=core.sha(source/'plan.json'),refined_export_sha256=core.sha(report_path),
               input_hashes={str(Path(r['path']).name):r['sha256'] for r in records},
               scientific_status='exploratory_not_validated',spatial_convergence=False,physical_validation=False,
               scope='Unchanged geometry and physical contacts; baseline only, no volume or translation derivative')
    core.verify_files(records);core.verify_files(p['inputs']);core.verify_files(p['implementation'])
    core.write(out/'result.json',row)
    core.write(out/'completion.json',dict(status='completed_numerical_checks_passed',
               hashes={name:core.sha(out/name) for name in ['result.json','fields.npz','factorization_memory.json']}))
    print(json.dumps(row,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',required=True,type=Path)
    parser.add_argument('--refined-export',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    run(parser.parse_args())
