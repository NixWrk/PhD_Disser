"""Restart only the joint search after explicit near-bound warm-start snapping.

Completed FEM data remain immutable. The original task deadlines and TOTAL
joint evaluation budget are retained. No tissue fit, observation or physical
kernel is changed. The new search has its own identity and fresh evaluations.
"""
from pathlib import Path
import argparse,copy,shutil
import numpy as np
from georg_quality_rebuild_pipeline import QualityRun,PIPE,quote
from georg_quality_comparison import read,sha
from georg_quality_contact_coverage import patch_coverage
from scorpion_georg_fem_prepare import write


def prepare(old,out):
    old,out=Path(old).resolve(),Path(out).resolve()
    state=read(old/'run_state.local.json')
    if state['status']!='failed' or state['child_pid'] is not None:
        raise ValueError('Stop the former search, preserving its failed/replaced state first')
    for name in ('inherited_initial_contact80','remapped_initial_contact80','fixed_pose_material_profile_contact80'):
        if not any(r['label']==name and r['status']=='complete' for r in state['stages']):
            raise ValueError('A completed inherited preparation stage is missing')
    if out.exists():raise ValueError('Use a fresh restart directory')
    cfg=read(old/'run_config.local.json');cfg=copy.deepcopy(cfg)
    run=old/'inputs'/('joint_own_drive_'+cfg['run_label'])
    checkpoints=sorted(run.glob('evaluation_*.local.json'))
    prior_started=max(read(p)['evaluation'] for p in checkpoints)
    remaining=cfg['max_evaluations']-prior_started
    if remaining<1:raise ValueError('No original joint evaluation budget remains')
    plan=read(run/'plan.local.json');lo=np.asarray(plan['lower'])[:6];hi=np.asarray(plan['upper'])[:6]
    warm=read(old/'warm_profile.local.json');params=np.asarray(warm['parameters'],float)
    before=params.copy();tolerance=np.array([.005,.01,.01,.005,.01,.01])*1e-6
    for k in range(6):
        if abs(params[k]-lo[k])<=tolerance[k]:params[k]=lo[k]
        elif abs(params[k]-hi[k])<=tolerance[k]:params[k]=hi[k]
    if not np.any(before!=params):raise ValueError('No near-bound parameter needs normalisation')
    if np.any(params[:6]<lo) or np.any(params[:6]>hi):raise ValueError('Normalised start leaves declared bounds')
    out.mkdir(parents=True)
    (out/'inputs/fits_own_drive').mkdir(parents=True)
    for name in ('inputs.local.json','observations.local.csv','author_centre_20261012.local.json','pose_grid.local.json'):
        shutil.copy2(old/'inputs'/name,out/'inputs'/name)
    shutil.copy2(old/'inputs/fits_own_drive/selected_previous_model.local.json',
        out/'inputs/fits_own_drive/selected_previous_model.local.json')
    for name in ('warm_seed.local.json','reference_initial.local.json','coarse_forward_initial.local.json',
        'inherited_initial.local.json','remapped_initial.local.json','initial_numerical_audit.local.json',
        'candidate_model_contract.local.json','coverage_author_selected.local.json',
        'fraction_distribution.local.json','contact_anatomical_depths.local.json'):
        shutil.copy2(old/name,out/name)
    identity={k:warm['identity'][k] for k in ('author_centre_sha256','observations_sha256')}
    normalized=dict(parameters=params.tolist(),identity=identity,
        pose=read(old/'warm_seed.local.json')['best']['pose'],
        status='explicit_near_bound_start_normalization_not_a_new_tissue_estimate',
        before_parameters=before.tolist(),normalization_logrho_tolerance=tolerance.tolist(),
        source_profile_sha256=sha(old/'inputs/fits_own_drive/selected_previous_model.local.json'),
        source_warm_start_sha256=sha(old/'warm_profile.local.json'),
        requested_bounds_from_executed_matlab_plan=True,
        maximum_absolute_logrho_change=float(np.max(np.abs(params-before))))
    write(out/'warm_profile.local.json',normalized)
    cfg['control_models']=[str((old/p).resolve().relative_to(out.parent)).replace('\\','/') for p in cfg['control_models']]
    # Both work roots are siblings; keep all selected models at original paths.
    cfg['control_models']=['../'+p for p in cfg['control_models']]
    cfg.update(work_directory=out.relative_to(PIPE).as_posix(),
        max_evaluations=remaining,run_label='all_contact_refined_normalized_restart',
        stage_suffix='contact_restart',initial_joint_evaluation_budget=cfg['max_evaluations'],
        prior_search_started_evaluations=prior_started,
        prior_work_directory=old.relative_to(PIPE).as_posix(),
        prepared_depth_root='../'+(old/'deep_mesh_v1').relative_to(out.parent).as_posix(),
        original_deadlines_retained=True,old_search_point_predictions_not_reused=True)
    files=[Path(__file__).resolve(),old/'run_config.initial.local.json',old/'run_config.local.json',
        old/'inputs/fits_own_drive/selected_previous_model.local.json',run/'plan.local.json',
        out/'warm_profile.local.json',*checkpoints]
    cfg['sources'] += [dict(path=p.relative_to(PIPE).as_posix(),sha256=sha(p)) for p in files]
    write(out/'run_config.local.json',cfg);write(out/'run_config.initial.local.json',cfg)
    return cfg


class RestartRun(QualityRun):
    def matlab(self,label,args,search=False):
        if search:
            old=quote((self.work/'warm_seed.local.json').relative_to(PIPE).as_posix())
            new=quote((self.work/'warm_profile.local.json').relative_to(PIPE).as_posix())
            if old not in args:raise ValueError('Expected explicit warm start')
            args=[new if x==old else x for x in args]
        return super().matlab(label,args,search)

    def run(self):
        super().run()
        models=[(self.work/p).resolve() for p in self.config['control_models']]
        coverage=patch_coverage(self.work/'coarse_forward_selected_v1.local.json',
            models[0]/'mesh.mat',models[-1]/'parent_maps.mat')
        write(self.work/'contact_coverage_final.local.json',coverage)
        self.publish(status='complete_with_explicit_numerical_results',
            all_final_contact_faces_refined=coverage['all_actual_contact_faces_refined'],
            global_mesh_convergence_proven=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prior',type=Path);p.add_argument('--out',type=Path);p.add_argument('--config',type=Path)
    a=p.parse_args()
    if a.config:RestartRun(a.config).run()
    else:
        if not a.prior or not a.out:p.error('Prepare requires --prior and --out')
        c=prepare(a.prior,a.out)
        print('Remaining evaluations:',c['max_evaluations'],'; original deadlines retained')
