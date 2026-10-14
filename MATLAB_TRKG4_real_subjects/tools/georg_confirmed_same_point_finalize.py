"""Bind already-completed corrected controls to an identical selected point.

No FEM is run or fabricated. Reuse requires exact parameters/pose/targets,
unchanged operator identities and a reproduced separated audit. Original files
remain immutable; copies retain their original reference/source identities.
"""
from pathlib import Path
import argparse, copy, shutil, time
import numpy as np
from georg_quality_rebuild_pipeline import QualityRun, PIPE
from georg_quality_comparison import read, sha
from georg_quality_general_audit import compile_audit
from georg_quality_contact_coverage import patch_coverage
from scorpion_georg_fem_prepare import write


def finalize(work):
    work=Path(work).resolve();cfg=read(work/'run_config_priority.local.json')
    job=QualityRun(work/'run_config_priority.local.json');job.verify()
    selected=read(work/'selected_candidate.local.json');best=selected['best']
    reference=read(work/'reference_initial.local.json')
    for key in ('parameters','pose'):
        if reference[key]!=best[key]:raise ValueError('Selected '+key+' differs; fresh controls are mandatory')
    models=[(work/p).resolve() for p in cfg['control_models']]
    sources=[work/p for p in cfg['initial_control_files']]
    original=compile_audit(*models,sources)
    if original!=read(work/cfg['initial_audit_file']):raise ValueError('Initial audit dependencies changed')
    rows=[read(p) for p in sources]
    for row in rows:
        if row['parameters']!=best['parameters'] or row['pose']!=best['pose']:
            raise ValueError('Existing control belongs to another point')
        if row['target_Z_ohm']!=best['target_Z_ohm']:raise ValueError('Targets changed')
    difference=float(np.max(np.abs(np.asarray(rows[-1]['fine_Z_ohm'])-np.asarray(best['predicted_Z_ohm']))))
    if difference>2e-8:raise ValueError('Independent inverse and fixed forward disagree')
    state=read(work/'run_state.local.json');write(work/'state_before_exact_point_reuse.local.json',state)
    for row in state['stages']:
        if row['status']=='running':row.update(status='stopped_redundant_identical_point',finished_unix=time.time())
    copies=[]
    for source,prefix in zip(sources,('coarse_forward','fine_inherited','fine_remapped'),strict=True):
        destination=work/(prefix+'_selected_v1.local.json')
        if destination.exists():raise ValueError('Cannot overwrite a completed selected control')
        shutil.copyfile(source,destination)
        if sha(destination)!=sha(source):raise ValueError('Control copy changed')
        copies.append(dict(source=source.relative_to(PIPE).as_posix(),source_sha256=sha(source),
                           selection_alias=destination.relative_to(PIPE).as_posix(),alias_sha256=sha(destination)))
    files=[work/(p+'_selected_v1.local.json') for p in ('coarse_forward','fine_inherited','fine_remapped')]
    audit=compile_audit(*models,files);audit.update(selected_candidate_sha256=sha(work/'selected_candidate.local.json'),
        completed_controls_reused_at_exactly_identical_point=True,
        original_initial_audit_sha256=sha(work/cfg['initial_audit_file']))
    write(work/'numerical_audit_selected_v1.local.json',audit)
    coverage=patch_coverage(files[0],models[0]/'mesh.mat',models[-1]/'parent_maps.mat')
    write(work/'contact_coverage_final.local.json',coverage)
    write(work/'selection_controls_reuse.local.json',dict(schema='exact_selected_point_control_reuse_v1',
        selected_candidate_sha256=sha(work/'selected_candidate.local.json'),
        reference_initial_sha256=sha(work/'reference_initial.local.json'),
        parameters_and_complete_pose_equal_exactly=True,targets_equal_exactly=True,
        inverse_fixed_forward_maximum_difference_ohm=difference,allowed_replay_difference_ohm=2e-8,
        original_controls_unchanged=True,copies=copies,FEM_rerun=False,
        negative_numerical_results_preserved=True,producer_sha256=sha(Path(__file__))))
    state.update(status='complete_with_explicit_numerical_results',child_pid=None,
        completed_controls_reused_at_exactly_identical_point=True,
        selected_numerical_checks_passed=audit['all_checks_passed'],
        all_final_contact_faces_refined=coverage['all_actual_contact_faces_refined'],
        quantitative_result_accepted=False,global_mesh_convergence_proven=False)
    write(work/'run_state.local.json',state)
    print(dict(selected_RMSE_ohm=float(np.sqrt(selected['best_mean_square_ohm2'])),
               maximum_replay_difference_ohm=difference,all_final_contact_faces_refined=coverage['all_actual_contact_faces_refined']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,required=True);a=p.parse_args();finalize(a.work)
