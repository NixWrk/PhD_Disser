"""Run the authorized Georg local joint search, then numerical checks.

One local process, sequential MATLAB/mesh stages and immutable source hashes.
No notebook executes FEM; no accepted-model pointer is modified.
"""
from pathlib import Path
import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import time

import numpy as np

from georg_ct_model import PIPE
from scorpion_georg_fem_prepare import read, write, sha
from georg_lateral_candidate_mesh import prepare


class BudgetExpired(RuntimeError):
    pass


def matlab_string(value):
    return "'"+str(value).replace("'", "''")+"'"


def mesh_comparison(coarse, fine):
    a, b = np.array(coarse['fine_Z_ohm']), np.array(fine['fine_Z_ohm'])
    delta = np.abs(a-b)
    tolerance = .02+.005*np.abs(b)
    return dict(coarse_model=coarse['identity']['model_plan_sha256'],
                fine_model=fine['identity']['model_plan_sha256'],
                difference_ohm=delta.tolist(), tolerance_ohm=tolerance.tolist(),
                maximum_ratio=float((delta/tolerance).max()),
                failed_values=int((delta>tolerance).sum()),
                all_passed=bool(np.all(delta<=tolerance)),
                compared_at_identical_parameters_and_electrode_centres=True,
                global_convergence_proven=False)


class Pipeline:
    def __init__(self, config):
        self.config = read(config)
        self.job = PIPE/self.config['job_directory']
        self.out = PIPE/self.config['output_directory']
        self.base = PIPE/self.config['search_model']
        self.state = dict(status='starting', pid=os.getpid(), child_pid=None, stages=[],
                          quantitative_result_accepted=False, common_model_changed=False)
        self.active_run = None
        self.child = None
        self.deadline = self.config.get('deadline_unix', float('inf'))
        self.search_deadline = self.config.get('search_deadline_unix', self.deadline)
        self.state.update(deadline_unix=self.config.get('deadline_unix'),
                          search_deadline_unix=self.config.get('search_deadline_unix'))

    def publish(self, **values):
        self.state.update(values)
        self.state['updated_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        path = self.job/'state.local.json'
        temp = self.job/'state_next.local.json'
        write(temp, self.state)
        os.replace(temp, path)

    def verify_sources(self):
        for source in self.config['code']:
            if sha(PIPE/source['path']) != source['sha256']:
                raise ValueError('Executed pipeline dependency changed: '+source['path'])

    def matlab(self, label, args):
        self.verify_sources()
        script = self.job/(label+'.m')
        text = 'cd('+matlab_string(PIPE.as_posix())+');\n'
        text += 'result=run_pipeline('+','.join(args)+');\n'
        script.write_text(text, encoding='utf-8')
        command = [self.config['matlab_executable'], '-batch',
                   'run('+matlab_string(script.as_posix())+')', '-logfile', str(self.job/(label+'.local.log'))]
        self.process(label, command, searching=label.startswith('search_'))

    def process(self, label, command, searching=False):
        self.verify_sources()
        deadline = min(self.deadline, self.search_deadline) if searching else self.deadline
        if time.time() >= deadline:
            raise BudgetExpired('No time remains for '+label)
        stage = dict(label=label, status='running', started_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        self.state['stages'].append(stage)
        timed_out = False
        with (self.job/(label+'_stdout.local.log')).open('w', encoding='utf-8') as stdout:
            self.child = subprocess.Popen(command,
                                          cwd=PIPE, stdout=stdout, stderr=subprocess.STDOUT,
                                          creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            self.publish(status=label, child_pid=self.child.pid)
            while True:
                left = deadline-time.time()
                if left <= 0 and self.child.poll() is None:
                    timed_out = True
                    # This PID is the live child launched above; kill its own
                    # MATLAB engine or mesh worker descendants as well.
                    if os.name == 'nt':
                        subprocess.run(['taskkill', '/PID', str(self.child.pid), '/T', '/F'],
                                       stdout=stdout, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW, check=False)
                    else:
                        self.child.kill()
                    code = self.child.wait(timeout=30)
                    break
                try:
                    code = self.child.wait(timeout=max(.1,min(30,left)))
                    break
                except subprocess.TimeoutExpired:
                    if self.active_run and self.active_run.is_dir():
                        checkpoints = sorted(self.active_run.glob('evaluation_*.local.json'))
                        completed = []
                        for p in checkpoints:
                            try:
                                row = read(p)
                            except (ValueError, OSError):
                                continue
                            if row.get('status') == 'numerical_candidate_not_validated':
                                completed.append(row)
                        if completed:
                            best = min(completed, key=lambda x:x['mean_square_ohm2'])
                            self.state['search_progress'] = dict(saved_files=len(checkpoints),
                                completed_fem=len(completed), best_RMSE_ohm=float(np.sqrt(best['mean_square_ohm2'])),
                                best_rho_inhale=best['rho_inhale_ohm_m'],best_rho_exhale=best['rho_exhale_ohm_m'],
                                best_centre_lps_mm=best['pose']['centre_lps_mm'])
                    self.publish()
        stage['return_code'] = code
        stage['status'] = 'deadline_stop' if timed_out else ('complete' if code == 0 else 'failed')
        stage['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        self.publish(child_pid=None)
        self.child = None
        if timed_out:
            raise BudgetExpired('Stage stopped at its deadline: '+label)
        if code:
            raise RuntimeError('Stage failed: '+label+'; see its saved logs')

    def search(self, halfwidth, warm=None):
        label = f"author_local{halfwidth}_"+self.config.get('run_suffix','v1')
        self.active_run = self.out/('joint_own_drive_'+label)
        args = ["'Profile'", "'georg_lateral_inverse'", "'NumericalPolicy'", "'nix_preliminary'",
                "'NumericalAuditFile'", matlab_string(self.config['numerical_audit']),
                "'ModelDir'", matlab_string(self.config['search_model']),
                "'OutputDir'", matlab_string(self.config['output_directory']),
                "'CentreFile'", matlab_string(self.config['centre_file']),
                "'CentreHalfWidthMM'", f'[{halfwidth},{halfwidth}]',
                "'MaxEvaluations'", str(self.config['max_evaluations']),
                "'RunLabel'", matlab_string(label)]
        if warm:
            args += ["'WarmStartFile'", matlab_string(warm.relative_to(PIPE).as_posix())]
        if self.config.get('start_indices'):
            args += ["'StartIndices'", '['+','.join(map(str,self.config['start_indices']))+']']
        if self.config.get('resume_from'):
            args += ["'ResumeFrom'", matlab_string(self.config['resume_from'])]
        result_file = self.active_run/'execution.local.json'
        try:
            self.matlab('search_'+str(halfwidth), args)
            result = read(result_file)
        except BudgetExpired:
            completed = []
            for f in self.active_run.glob('evaluation_*.local.json'):
                try:
                    row = read(f)
                except (ValueError,OSError):
                    continue
                if row.get('status') == 'numerical_candidate_not_validated':
                    completed.append((f,row))
            if not completed:
                raise
            source,best = min(completed,key=lambda x:x[1]['mean_square_ohm2'])
            result = dict(best=best,best_parameters=best['parameters'],
                          best_mean_square_ohm2=best['mean_square_ohm2'],
                          source_checkpoint_sha256=sha(source),
                          search_stopped_for_time=True,resolution_reached=False,
                          quantitative_result_accepted=False,global_minimum_certified=False)
            result_file = self.active_run/'execution_time_budget.local.json'
            write(result_file,result)
        if len(result['best']['target_Z_ohm']) != 24:
            raise ValueError('The joint fit omitted an observation')
        return result_file, result

    def fixed_check(self, model, candidate, label):
        self.active_run = None
        args = ["'Profile'", "'georg_lateral_inverse'", "'ModelDir'", matlab_string(model.relative_to(PIPE).as_posix()),
                "'OutputDir'", matlab_string(self.config['output_directory']),
                "'CheckCandidateFile'", matlab_string(candidate.relative_to(PIPE).as_posix()),
                "'RunLabel'", matlab_string(label)]
        self.matlab('check_'+label, args)
        result = read(self.out/('forward_check_own_drive_'+label+'.local.json'))
        if result['status'] != 'fixed_parameter_numerical_check' or not result['system_checks_passed']:
            raise ValueError('Numerical or contact QC failed: '+label)
        return result

    def variant(self, base, candidate, label, scope, points, target=4.):
        self.verify_sources()
        self.active_run = None
        free = shutil.disk_usage(self.job).free
        if free < self.config['minimum_free_bytes']:
            raise OSError('Insufficient free disk space before mesh preparation')
        out = self.job/label
        command = [self.config.get('python_executable', os.sys.executable), '-u',
                   str(PIPE/'tools/georg_lateral_candidate_mesh.py'),
                   '--base',str(base),'--out',str(out),'--candidate',str(candidate),
                   '--scope',scope,'--target',str(target),
                   '--radius',str(self.config['local_refinement_radius_mm']),
                   '--points',*map(str,points)]
        self.process('prepare_'+label, command)
        return out

    def run(self):
        try:
            self.publish()
            candidate, first = self.search(30)
            candidates = [dict(path=candidate.relative_to(PIPE).as_posix(), RMSE_ohm=float(np.sqrt(first['best_mean_square_ohm2'])),
                               pose_at_boundary=bool(np.any(np.isclose(np.abs(first['best_parameters'][6:8]),30))))]
            if candidates[0]['pose_at_boundary'] and self.config.get('expand_pose_window',True):
                second_path, second = self.search(60, candidate)
                candidates.append(dict(path=second_path.relative_to(PIPE).as_posix(),
                                       RMSE_ohm=float(np.sqrt(second['best_mean_square_ohm2'])),
                                       pose_at_boundary=bool(np.any(np.isclose(np.abs(second['best_parameters'][6:8]),60)))))
                if second['best_mean_square_ohm2'] < first['best_mean_square_ohm2']:
                    candidate, first = second_path, second
            # Freeze the selected result before subsequent variants bind its hash.
            selected = self.job/'selected_candidate.local.json'
            selected.write_bytes(candidate.read_bytes())
            self.publish(selected_candidate=selected.relative_to(PIPE).as_posix(), searches=candidates)
            self.validation = dict(selected_candidate_sha256=sha(selected),search_summaries=candidates,
                                   integration_checks=[],mesh_checks=[],spatial_comparisons=[],
                                   global_mesh_convergence_proven=False,physical_validation=False,
                                   quantitative_result_accepted=False,common_model_changed=False,
                                   tolerance_not_relaxed=True,status='checks_in_progress')
            self.validation['unfinished_checks'] = ['integration_4096_16384']
            if self.config.get('local_checks_first',False):
                self.validation['unfinished_checks'] += [
                    'local_volume_h'+str(t).replace('.','p')
                    for t in self.config.get('local_mesh_targets_mm',[2.,1.5])]
            write(self.job/'numerical_validation.local.json',self.validation)
            strong = self.variant(self.base, selected, 'quadrature_4096_16384', 'quadrature', [4096,16384])
            reference = self.fixed_check(strong, selected, 'author_q16384_v1')
            quad_checks = [reference]
            self.validation['integration_checks'] = quad_checks
            self.validation['unfinished_checks'].remove('integration_4096_16384')
            write(self.job/'numerical_validation.local.json',self.validation)
            points = [4096,16384]
            if not reference['quadrature_passed'] and self.config.get('escalate_quadrature',True):
                strong = self.variant(strong, selected, 'quadrature_16384_65536', 'quadrature', [16384,65536])
                reference = self.fixed_check(strong, selected, 'author_q65536_v1')
                quad_checks.append(reference);points = [16384,65536]
                write(self.job/'numerical_validation.local.json',self.validation)
            if self.config.get('local_checks_first',False):
                previous_model,previous = strong,reference
                checks,comparisons = [],[]
                for target in self.config.get('local_mesh_targets_mm',[2.,1.5]):
                    label='local_volume_h'+str(target).replace('.','p')
                    model=self.variant(previous_model,selected,label,'local',points,target)
                    result=self.fixed_check(model,selected,'author_'+label+'_v1')
                    checks.append(result);comparisons.append(mesh_comparison(previous,result))
                    self.validation['unfinished_checks'].remove(label)
                    self.validation.update(mesh_checks=checks,spatial_comparisons=comparisons)
                    write(self.job/'numerical_validation.local.json',self.validation)
                    previous_model,previous=model,result
                self.validation.update(status='planned_checks_completed',
                    final_local_mesh_comparison_passed=comparisons[-1]['all_passed'],
                    all_integration_checks_passed=all(c['quadrature_passed'] for c in [*quad_checks,*checks]),
                    full_volume_refinement_completed=False,
                    stronger_quadrature_pending=not reference['quadrature_passed'])
                write(self.job/'numerical_validation.local.json',self.validation)
                self.publish(status='complete_with_explicit_numerical_results',
                             final_local_mesh_comparison_passed=comparisons[-1]['all_passed'])
                return
            # Full-volume edge refinement, then two nested contact-volume levels.
            global_model = self.variant(strong, selected, 'global_volume_h4', 'global', points, 4.)
            global_result = self.fixed_check(global_model, selected, 'author_global_h4_v1')
            comparisons = [mesh_comparison(reference, global_result)]
            previous_model, previous = global_model, global_result
            checks = [global_result]
            for target in [2.,1.5]:
                label = 'local_volume_h'+str(target).replace('.','p')
                model = self.variant(previous_model, selected, label, 'local', points, target)
                result = self.fixed_check(model, selected, 'author_'+label+'_v1')
                comparisons.append(mesh_comparison(previous,result));checks.append(result)
                previous_model, previous = model, result
            if not comparisons[-1]['all_passed']:
                model = self.variant(previous_model, selected, 'local_volume_h1', 'local', points, 1.)
                result = self.fixed_check(model, selected, 'author_local_h1_v1')
                comparisons.append(mesh_comparison(previous,result));checks.append(result)
            report = dict(selected_candidate_sha256=sha(selected), search_summaries=candidates,
                          integration_checks=quad_checks, mesh_checks=checks, spatial_comparisons=comparisons,
                          final_local_mesh_comparison_passed=comparisons[-1]['all_passed'],
                          all_reported_spatial_comparisons_passed=all(c['all_passed'] for c in comparisons),
                          all_integration_checks_passed=all(c['quadrature_passed'] for c in [*quad_checks,*checks]),
                          global_mesh_convergence_proven=False, physical_validation=False,
                          quantitative_result_accepted=False, common_model_changed=False,
                          tolerance_not_relaxed=True)
            write(self.job/'numerical_validation.local.json', report)
            self.publish(status='complete_with_explicit_numerical_results',
                         final_local_mesh_comparison_passed=report['final_local_mesh_comparison_passed'],
                         all_integration_checks_passed=report['all_integration_checks_passed'])
        except BudgetExpired as error:
            if hasattr(self,'validation'):
                self.validation.update(status='time_budget_exhausted',
                                       unfinished_checks_are_not_passed=True)
                write(self.job/'numerical_validation.local.json',self.validation)
            self.publish(status='time_budget_exhausted',error=str(error),
                         unfinished_checks_are_not_passed=True)
        except BaseException as error:
            if hasattr(self,'validation'):
                self.validation.update(status='failed',error_type=type(error).__name__,error=str(error))
                write(self.job/'numerical_validation.local.json',self.validation)
            self.publish(status='failed', error_type=type(error).__name__, error=str(error))
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    a = p.parse_args()
    Pipeline(a.config).run()


if __name__ == '__main__':
    main()
