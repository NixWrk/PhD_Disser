"""Fresh geometry-priority continuation of confirmed inputs, same FEM kernels.

Only the parameter-check order changes. Saved initial controls are verified,
the replaced execution and its source are preserved, original deadline remains.
"""
from pathlib import Path
import argparse, copy, shutil
from georg_quality_rebuild_pipeline import QualityRun, PIPE
from georg_quality_comparison import read, sha
from georg_quality_general_audit import compile_audit
from georg_quality_contact_coverage import patch_coverage
from scorpion_georg_fem_prepare import write


def prepare(work):
    work=Path(work).resolve();cfg=copy.deepcopy(read(work/'run_config_joint_only.local.json'))
    previous=read(work/'run_state.local.json')
    write(work/'controller_before_priority.local.json',previous)
    for field in ('status','child_pid'):
        if field=='child_pid':previous[field]=None
        else:previous[field]='stopped_for_explicit_geometry_priority'
    write(work/'run_state.local.json',previous)
    directory=work/'inputs'/('joint_own_drive_'+cfg['run_label'])
    started=[read(p).get('evaluation',0) for p in directory.glob('evaluation_*.local.json')]
    prior_count=max(started,default=0)
    cfg.update(run_label='confirmed_1x1_geometry_priority',stage_suffix='confirmed_priority',
        replaced_search_directory=directory.relative_to(PIPE).as_posix(),
        replaced_search_started_evaluations=prior_count,
        max_evaluations=max(16,cfg['max_evaluations']-prior_count),
        parameter_evaluation_order=[7,8,9,1,4,2,5,3,6],
        priority_reason='Remaining original time must include centre/angle trials and tissue trials')
    inverse=PIPE/'src/run_georg_lateral_inverse.m'
    archive=PIPE/'archive/executed_georg_continuation_20261014/before_priority/src/run_georg_lateral_inverse.m'
    changes=[]
    for row in cfg['sources']:
        if (PIPE/row['path']).resolve()==inverse:
            changes.append(dict(path=row['path'],previous_sha256=row['sha256'],new_sha256=sha(inverse),
                                executed_previous_snapshot=archive.relative_to(PIPE).as_posix()))
            if sha(archive)!=row['sha256']:raise ValueError('Prior executed inverse source is not preserved')
            row['sha256']=sha(inverse);row.pop('snapshot',None)
    if not changes:raise ValueError('Inverse source must already be frozen')
    write(work/'explicit_parameter_order_change.local.json',dict(changes=changes,
        default_order_remains_1_to_9=True,physical_kernels_changed=False,
        parameter_bounds_changed=False,observation_values_changed=False,
        old_predictions_reused=False,original_deadlines_retained=True))
    files=[Path(__file__).resolve(),PIPE/'src/trkg4_reordered_pattern_search.m',archive,
           work/'explicit_parameter_order_change.local.json']
    cfg['sources'] += [dict(path=p.relative_to(PIPE).as_posix(),sha256=sha(p)) for p in files]
    file=work/'run_config_priority.local.json';write(file,cfg);return file


class PriorityRun(QualityRun):
    def matlab(self,label,args,search=False):
        if search:args=args+["'JointParameterOrder'",'['+','.join(map(str,self.config['parameter_evaluation_order']))+']']
        return super().matlab(label,args,search)

    def run(self):
        models=[(self.work/p).resolve() for p in self.config['control_models']]
        files=[self.work/p for p in self.config['initial_control_files']]
        verified=compile_audit(*models,files)
        saved=read(self.work/self.config['initial_audit_file'])
        if verified!=saved:raise ValueError('Completed initial audit changed during orchestration handoff')
        super().run()
        coverage=patch_coverage(self.work/'coarse_forward_selected_v1.local.json',models[0]/'mesh.mat',models[-1]/'parent_maps.mat')
        write(self.work/'contact_coverage_final.local.json',coverage)
        self.publish(status='complete_with_explicit_numerical_results',
            all_final_contact_faces_refined=coverage['all_actual_contact_faces_refined'],global_mesh_convergence_proven=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',type=Path);p.add_argument('--config',type=Path);a=p.parse_args()
    if a.prepare:print(prepare(a.prepare))
    elif a.config:PriorityRun(a.config).run()
    else:p.error('Use prepare or config')
