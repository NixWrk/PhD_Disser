"""Controls on a mesh refined at the relocated selected contacts; no inversion."""
from pathlib import Path
import argparse
import datetime as dt
import os
import subprocess
import sys
import time

from georg_convergence_continue import (ConvergenceRun,PIPE,read,sha,write_new,
    compile_audit,compare_saved_controls,fixed_point_compare,nested_rule_compare,
    verify_independent_integrations)
from georg_convergence_contacts import prepare as contact_model
from georg_refined_contact_coverage import compute_coverage
from scorpion_georg_fem_prepare import write


class RelocatedChecks(ConvergenceRun):
    def publish(self,**values):
        self.state.update(values);self.state['updated_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        temp=self.work/'relocated_run_state_next.local.json';write(temp,self.state)
        os.replace(temp,self.work/'relocated_run_state.local.json')

    def await_file(self,path):
        while not path.is_file():
            if time.time()>=self.deadline:raise TimeoutError('Original hard deadline expired')
            self.publish(status='waiting_relocated_preparation',pending_file=path.name);time.sleep(15)

    def integrations(self,source):
        jobs=[]
        for seed in (20261008,20261015):
            directory=self.work/f'relocated_q16384_seed{seed}'
            if (directory/'plan.local.json').is_file():continue
            if directory.exists():raise ValueError('Unfinished independent integration output needs explicit recovery')
            log=(self.work/f'relocated_integration_seed{seed}_stdout.local.log').open('w',encoding='utf-8')
            command=[sys.executable,'-B','-X','utf8',str(PIPE/'tools/georg_integration_stability.py'),
                '--model',str(source),'--out',str(directory),'--points','4096','16384','--seed',str(seed)]
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            jobs.append((child,log,seed))
        try:
            while any(child.poll() is None for child,_,_ in jobs):
                if time.time()>=self.deadline:
                    for child,_,_ in jobs:
                        if child.poll() is None:child.kill()
                    raise TimeoutError('Independent integration original deadline')
                self.publish(status='preparing_relocated_independent_integrations',
                    integration_pids=[c.pid for c,_,_ in jobs if c.poll() is None]);time.sleep(15)
            if any(c.returncode for c,_,_ in jobs):raise RuntimeError('Relocated independent integration failed')
        finally:
            for _,log,_ in jobs:log.close()

    def run(self):
        try:
            self.verify();base=self.work/'relocated_contacts_r3_v2'
            self.await_file(base/'fine2/plan.local.json')
            self.await_file(self.work/'parent_direct_new_candidate.local.json')
            q=read(base/'fine2/mesh_quality.local.json')
            if not q['passed']:raise ValueError('Relocated mesh quality failed')
            directories=[self.work/name for name in ('relocated_inherited','relocated_remapped','relocated_direct')]
            for source,out in zip((base/'fine2_inherited',base/'fine2'),directories[:2]):
                if not out.exists():contact_model(source,out,self.work/'parent_direct/mesh.mat',base/'fine2/parent_maps.mat')
            if not directories[2].exists():contact_model(base/'fine2',directories[2],base/'fine2/mesh.mat',None)
            self.integrations(directories[2])
            reference=self.work/'reference_new_candidate.local.json';files=[self.work/'parent_direct_new_candidate.local.json']
            for directory in directories:files.append(self.forward(directory,directory.name+'_forward',reference=reference))
            models=[self.work/f'relocated_q16384_seed{seed}' for seed in (20261008,20261015)]
            high=[self.forward(model,model.name+'_forward',derivatives=k==0,reference=reference) for k,model in enumerate(models)]
            rows=[read(path) for path in high]
            audit=compile_audit(self.work/'parent_direct',*directories[:2],files[:3])
            contact=compare_saved_controls(*directories[1:3],*files[2:4],hdf5_library=self.config['hdf5_library'])
            identity=verify_independent_integrations(models,rows,self.config['hdf5_library'])
            comparisons=dict(
                spatial_respiratory=fixed_point_compare(read(files[0]),read(files[1]),relative=.005),
                tissue_respiratory=fixed_point_compare(read(files[1]),read(files[2]),relative=.005),
                contact_respiratory=fixed_point_compare(read(files[2]),read(files[3]),relative=.005),
                rule_growth=fixed_point_compare(read(files[3]),rows[0]),
                independent_seed=fixed_point_compare(*rows))
            for k,row in enumerate(rows):comparisons[f'nested_rules_seed_{k+1}']=nested_rule_compare(row)
            coverage=compute_coverage(high[0],models[0]/'mesh.mat',base/'fine2/parent_maps.mat')
            coveragefile=self.work/'relocated_fine_contact_coverage.local.json';self.save_receipt(coveragefile,coverage)
            passed=(audit['all_checks_passed'] and contact['passed'] and identity['passed'] and
                all(c['passed'] for c in comparisons.values()) and coverage['all_contact_faces_refined'] and
                all(r['system_checks_passed'] and r['quadrature_passed'] and r['patch_qc']['passed'] for r in rows) and
                len(rows[0]['derivatives'])==2 and all(r['passed'] for r in rows[0]['derivatives']))
            result=dict(schema='georg_relocated_selected_numerical_audit_v1',all_checks_passed=bool(passed),
                spatial_tissue_audit=audit,contact_comparison=contact,comparisons=comparisons,
                integration_identity=identity,actual_contact_coverage_passed=coverage['all_contact_faces_refined'],
                point_only=True,global_mesh_convergence_proven=False,quantitative_result_accepted=False,
                selected_candidate_sha256=sha(self.work/'selected_new_candidate.local.json'),
                reference_sha256=sha(reference),model_directories=[str(p.resolve()) for p in [self.work/'parent_direct',*directories,*models]],
                forward_files=[str(p.resolve()) for p in [*files,*high]],
                sources=[dict(path=str(p.resolve()),sha256=sha(p)) for p in [*files,*high,coveragefile,Path(__file__)]])
            self.save_receipt(self.work/'relocated_selected_audit.local.json',result)
            self.publish(status='relocated_selected_checks_complete',all_checks_passed=bool(passed))
        except BaseException as error:
            self.publish(status='failed',error_type=type(error).__name__,error=str(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    RelocatedChecks(parser.parse_args().config).run()
