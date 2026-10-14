"""Bounded joint search and selected-candidate controls on prepared models.

All FEM calls use run_pipeline; notebooks only read the immutable outputs.
Machine-specific executables belong in the local, excluded configuration.
"""
from pathlib import Path
import argparse
import datetime as dt
import json
import os
import subprocess
import time
import numpy as np
from georg_quality_comparison import read,sha,audit
from scorpion_georg_fem_prepare import write

PIPE=Path(__file__).resolve().parents[1]


def quote(value):
    return "'"+str(value).replace("'","''")+"'"


class QualityRun:
    def __init__(self,config):
        self.config=read(config);self.work=PIPE/self.config['work_directory']
        self.state=dict(status='starting',pid=os.getpid(),stages=[],child_pid=None,
            quantitative_result_accepted=False,common_model_pointer_changed=False)
        self.deadline=self.config['deadline_unix'];self.search_deadline=self.config['search_deadline_unix']
        self.state.update(deadline_unix=self.deadline,search_deadline_unix=self.search_deadline)

    def publish(self,**values):
        self.state.update(values);self.state['updated_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        temporary=self.work/'run_state_next.local.json';write(temporary,self.state)
        os.replace(temporary,self.work/'run_state.local.json')

    def verify(self):
        for row in self.config['sources']:
            if sha(PIPE/row['path'])!=row['sha256']:
                raise ValueError('Executed source changed: '+row['path'])

    def matlab(self,label,args,search=False):
        self.verify();label=label+'_'+self.config['stage_suffix'];script=self.work/(label+'.m')
        if script.exists():raise ValueError('Use a fresh immutable MATLAB stage')
        expression="mfilename('fullpath')"
        for _ in range(len(self.work.relative_to(PIPE).parts)+1):expression='fileparts('+expression+')'
        script.write_text('cd('+expression+');\n'
            +'result=run_pipeline('+','.join(args)+');\n',encoding='utf-8')
        env=os.environ.copy();env['MATLAB_PREFDIR']=str(self.work/'matlab_preferences')
        env['EIDORS_STARTUP']=self.config['eidors_startup']
        command=[self.config['matlab_executable'],'-batch','run('+quote(script.relative_to(PIPE).as_posix())+')',
                 '-logfile',str(self.work/(label+'.local.log'))]
        deadline=self.search_deadline if search else self.deadline
        row=dict(label=label,status='running',started_unix=time.time());self.state['stages'].append(row)
        with (self.work/(label+'_stdout.local.log')).open('w',encoding='utf-8') as log:
            child=subprocess.Popen(command,cwd=PIPE,env=env,stdout=log,stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            self.publish(status=label,child_pid=child.pid)
            timed_out=False
            while child.poll() is None:
                if time.time()>=deadline:
                    timed_out=True
                    if os.name=='nt':
                        subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=log,
                            stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW,check=False)
                    else:child.kill()
                    child.wait(timeout=30);break
                try:child.wait(timeout=min(30,max(.1,deadline-time.time())))
                except subprocess.TimeoutExpired:
                    if search:self.progress()
                    else:self.publish()
        row.update(status='deadline_stop' if timed_out else ('complete' if child.returncode==0 else 'failed'),
                   finished_unix=time.time(),return_code=child.returncode)
        self.publish(child_pid=None)
        if not timed_out and child.returncode:raise RuntimeError('MATLAB stage failed: '+label)
        return not timed_out

    def progress(self):
        run=self.work/self.config.get('input_directory','inverse_inputs')/('joint_own_drive_'+self.config['run_label'])
        completed=[];invalid=0
        for path in run.glob('evaluation_*.local.json'):
            try:row=read(path)
            except (OSError,ValueError):continue
            if row['status']=='numerical_candidate_not_validated':completed.append(row)
            elif row['status']=='invalid_geometry':invalid+=1
        if completed:
            best=min(completed,key=lambda row:row['mean_square_ohm2'])
            self.state['progress']=dict(completed_candidates=len(completed),invalid_geometry=invalid,
                best_RMSE_ohm=float(np.sqrt(best['mean_square_ohm2'])),
                best_rho_inhale=best['rho_inhale_ohm_m'],best_rho_exhale=best['rho_exhale_ohm_m'],
                best_centre_lps_mm=best['pose']['centre_lps_mm'])
        self.publish()

    def run(self):
        try:
            self.publish();initial=self.work/self.config.get('initial_audit_file','numerical_audit_author_v2.local.json')
            initial_names=self.config.get('initial_control_files',[p+'_author_v2.local.json' for p in
                          ('coarse_forward','fine_inherited','fine_remapped')])
            model_names=self.config.get('control_models',['mesh_v9/coarse_operators','mesh_v9/fine_inherited','mesh_v9/fine'])
            models=[self.work/name for name in model_names]
            def make_audit(files):
                if self.config.get('independent_contact_master',False):
                    from georg_quality_general_audit import compile_audit
                    return compile_audit(*models,files)
                return audit(self.work,*files)
            while not initial.exists():
                self.verify()
                if time.time()>=self.search_deadline:raise TimeoutError('Initial checks exceeded search deadline')
                if all((self.work/name).exists() for name in initial_names):
                    write(initial,make_audit([self.work/name for name in initial_names]));break
                self.publish(status='waiting_initial_controls');time.sleep(15)
            numerical=read(initial)
            required=['shape_QC_passed','energy_inheritance_passed','quadrature_passed','derivatives_passed','system_checks_passed']
            if not all(numerical[k] for k in required):raise ValueError('Initial numerical prerequisites failed')
            inputs=self.work/self.config.get('input_directory','inverse_inputs');model=models[-1]
            rel=lambda p:Path(p).relative_to(PIPE).as_posix()
            args=["'Profile'","'georg_lateral_inverse'","'OutputDir'",quote(rel(inputs)),
                "'ModelDir'",quote(rel(model)),"'ModelContractFile'",quote(rel(self.work/'candidate_model_contract.local.json')),
                "'NumericalPolicy'","'nix_preliminary'","'NumericalAuditFile'",quote(rel(initial)),
                "'CentreFile'",quote(rel(inputs/'author_centre_20261012.local.json')),
                "'CentreHalfWidthMM'","[30,30]","'AngleHalfWidthDeg'","20",
                "'MaxEvaluations'",str(self.config['max_evaluations']),"'RunLabel'",quote(self.config['run_label']),
                "'WarmStartFile'",quote(rel(self.work/'warm_seed.local.json')),
                "'StartIndices'",'['+','.join(map(str,self.config['start_indices']))+']']
            finished=self.matlab('joint_search_quality_v1',args,search=True)
            run=inputs/('joint_own_drive_'+self.config['run_label']);candidate=run/'execution.local.json'
            if not finished:
                rows=[]
                for path in run.glob('evaluation_*.local.json'):
                    try:row=read(path)
                    except (OSError,ValueError):continue
                    if row['status']=='numerical_candidate_not_validated':rows.append((path,row))
                if not rows:raise TimeoutError('No completed search candidate before deadline')
                source,best=min(rows,key=lambda item:item[1]['mean_square_ohm2'])
                candidate=run/'execution_time_budget.local.json'
                write(candidate,dict(best=best,best_parameters=best['parameters'],
                    best_mean_square_ohm2=best['mean_square_ohm2'],source_checkpoint_sha256=sha(source),
                    search_stopped_for_time=True,resolution_reached=False,quantitative_result_accepted=False))
            result=read(candidate);best=result['best'];selected=self.work/'selected_candidate.local.json'
            selected.write_bytes(candidate.read_bytes())
            reference=read(self.work/self.config.get('initial_reference_file','reference_author_v2.local.json'))
            reference.update(pose=best['pose'],parameters=best['parameters'],
                selected_candidate_sha256=sha(selected),common_parameters_not_fitted_here=False,
                optimized_on_fine_remapped_mesh=True)
            ref_file=self.work/'reference_selected_v1.local.json';write(ref_file,reference)
            self.publish(status='checking_selected_candidate',selected_candidate=rel(selected))
            for prefix,directory,derivatives in [('coarse_forward',models[0],True),
                ('fine_inherited',models[1],False),('fine_remapped',models[2],False)]:
                label=prefix+'_selected_v1'
                common=["'Profile'","'georg_quality_forward'","'ModelDir'",quote(rel(directory)),
                    "'InputDir'",quote(rel(inputs)),"'ReferenceFile'",quote(rel(ref_file)),
                    "'OutputFile'",quote(rel(self.work/(label+'.local.json')))]
                if derivatives:common += ["'CheckDerivatives'","true"]
                if not self.matlab(label,common):raise TimeoutError('Selected candidate controls did not finish')
            final=make_audit([self.work/(p+'_selected_v1.local.json') for p in
                       ('coarse_forward','fine_inherited','fine_remapped')])
            final['selected_candidate_sha256']=sha(selected)
            write(self.work/'numerical_audit_selected_v1.local.json',final)
            self.progress();self.publish(status='complete_with_explicit_numerical_results',
                selected_numerical_checks_passed=final['all_checks_passed'])
        except BaseException as error:
            self.publish(status='failed',error_type=type(error).__name__,error=str(error))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    args=p.parse_args();QualityRun(args.config).run()
