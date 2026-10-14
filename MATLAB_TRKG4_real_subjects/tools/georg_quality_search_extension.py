"""Continue an unchanged qualified search; then isolate selected contact effects.

All FEM calls remain in run_pipeline. The local configuration pins the inputs,
completed parent run, source hashes, inherited deadline and extended budget.
"""
from pathlib import Path
import argparse
import copy
import os
import shutil
import time

from georg_quality_rebuild_pipeline import QualityRun, PIPE, quote
from georg_quality_comparison import read, sha
from scorpion_georg_fem_prepare import write
from georg_quality_contact_variant import compare_saved_controls
from georg_quality_contact_coverage import patch_coverage


def preserve_resume_source(parent, inputs, resume_name):
    """Keep the exact worker bytes required by the completed search identity."""
    parent, inputs = Path(parent), Path(inputs)
    digest = read(inputs/resume_name/'plan.local.json')['identity']['worker_sha256']
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Invalid executed worker SHA256')
    target = inputs/'source_snapshots'/('run_georg_lateral_inverse_'+digest+'.m')
    if target.exists():
        if sha(target) != digest:
            raise ValueError('Existing resume archive differs from the executed worker')
        return target
    source = parent/'source_snapshots/src/run_georg_lateral_inverse.m'
    if not source.is_file() or sha(source) != digest:
        raise ValueError('Exact executed worker bytes are unavailable')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(source.read_bytes())
    if sha(target) != digest:
        raise ValueError('Resume archive changed during preparation')
    return target


def prepare_extension(parent, output, budget, direct, hdf5_library=None):
    parent, output, direct = map(lambda p: Path(p).resolve(), (parent, output, direct))
    if output.exists() or not output.is_relative_to(PIPE) or output.is_relative_to(parent/'inputs'):
        raise ValueError('Use a fresh extension directory inside the project, outside the inverse inputs')
    previous = read(parent/'run_config.local.json')
    if not previous['max_evaluations'] < budget <= previous['maximum_possible_evaluations_after_timed_review']:
        raise ValueError('Extended budget exceeds the previously declared possible budget')
    cfg = copy.deepcopy(previous)
    relative = lambda p: Path(os.path.relpath(p,output)).as_posix()
    cfg.update(work_directory=output.relative_to(PIPE).as_posix(),
        parent_work_directory=parent.relative_to(PIPE).as_posix(),max_evaluations=budget,
        resume_from='joint_own_drive_'+previous['run_label'],
        run_label=previous['run_label']+'_resume'+str(budget),stage_suffix='resume'+str(budget),
        direct_contact_directory=direct.relative_to(PIPE).as_posix(),hdf5_library=hdf5_library,
        deadline_not_reset_after_extension=True,
        input_directory=relative(parent/previous['input_directory']),
        initial_audit_file=relative(parent/previous['initial_audit_file']),
        initial_reference_file=relative(parent/previous['initial_reference_file']),
        initial_control_files=[relative(parent/p) for p in previous['initial_control_files']],
        initial_contact_file=relative(parent/'contact_discretization_initial.local.json'),
        control_models=[relative(parent/p) for p in previous['control_models']])
    for source in cfg['sources']:
        if sha(PIPE/source['path'])!=source['sha256']:
            raise ValueError('Parent executed dependency changed: '+source['path'])
    resume_source=preserve_resume_source(parent,parent/previous['input_directory'],cfg['resume_from'])
    additional=[resume_source,Path(__file__),PIPE/'tools/georg_quality_contact_variant.py',
                PIPE/'tools/georg_quality_contact_coverage.py',direct/'plan.local.json',
                direct/'contact_variant.local.json']
    for source in additional:
        path=source.resolve().relative_to(PIPE).as_posix()
        if not any(r['path']==path for r in cfg['sources']):
            cfg['sources'].append(dict(path=path,sha256=sha(source)))
    output.mkdir(parents=True,exist_ok=False)
    for name in ('candidate_model_contract.local.json','warm_seed.local.json'):
        shutil.copy2(parent/name,output/name)
    for source in cfg['sources']:
        snapshot=output/'source_snapshots'/source['path']
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(PIPE/source['path'],snapshot)
        source['snapshot']=snapshot.relative_to(PIPE).as_posix()
        if sha(snapshot)!=source['sha256']:raise ValueError('Extension snapshot changed')
    path=output/'run_config.local.json'
    write(path,cfg)
    return path


class ExtendedRun(QualityRun):
    def matlab(self, label, args, search=False):
        if search:
            args = [*args, "'ResumeFrom'", quote(self.config['resume_from'])]
        return super().matlab(label, args, search=search)

    def run(self):
        parent = PIPE / self.config['parent_work_directory']
        while True:
            self.verify()
            state = read(parent / 'run_state.local.json')
            if state['status'] == 'complete_with_explicit_numerical_results':
                break
            if state['status'] == 'failed':
                raise RuntimeError('The preceding search/control run failed')
            if time.time() >= self.search_deadline:
                raise TimeoutError('Parent controls did not finish before inherited search deadline')
            self.publish(status='waiting_completed_parent_controls')
            time.sleep(15)
        if not read(parent / 'numerical_audit_selected_v1.local.json')['all_checks_passed']:
            raise ValueError('Resolve the parent selected-point numerical failure before extending')
        super().run()
        try:
            rel = lambda p: Path(p).resolve().relative_to(PIPE).as_posix()
            base = (self.work / self.config['control_models'][-1]).resolve()
            direct = (PIPE / self.config['direct_contact_directory']).resolve()
            output = self.work / 'direct_contacts_selected.local.json'
            args = ["'Profile'", "'georg_quality_forward'", "'ModelDir'", quote(rel(direct)),
                    "'InputDir'", quote(rel(self.work / self.config['input_directory'])),
                    "'ReferenceFile'", quote(rel(self.work / 'reference_selected_v1.local.json')),
                    "'OutputFile'", quote(rel(output))]
            self.publish(status='checking_selected_contact_discretization')
            if not self.matlab('direct_contacts_selected', args):
                raise TimeoutError('Selected contact control did not finish before inherited deadline')
            comparison = compare_saved_controls(base, direct,
                self.work / 'fine_remapped_selected_v1.local.json', output,
                hdf5_library=self.config.get('hdf5_library'))
            write(self.work / 'contact_discretization_selected.local.json', comparison)
            coarse = (self.work / self.config['control_models'][0]).resolve()
            coverage = patch_coverage(self.work / 'coarse_forward_selected_v1.local.json',
                coarse / 'mesh.mat', base / 'parent_maps.mat')
            write(self.work / 'contact_coverage_selected.local.json', coverage)
            self.publish(status='complete_with_explicit_numerical_results',
                selected_contact_check_passed=comparison['passed'],
                all_selected_contact_faces_refined=coverage['all_actual_contact_faces_refined'],
                global_mesh_convergence_proven=False)
        except BaseException as error:
            self.publish(status='failed_selected_contact_control',error_type=type(error).__name__,error=str(error))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    ExtendedRun(parser.parse_args().config).run()
