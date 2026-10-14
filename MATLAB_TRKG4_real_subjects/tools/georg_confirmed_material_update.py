"""Bounded analytic-Jacobian material update at the retained contact position.

This is a conditional six-parameter fit, kept separate from the truncated
joint search and its same-point controls. No new global convergence is claimed.
"""
from pathlib import Path
import argparse, copy
from georg_quality_rebuild_pipeline import QualityRun, PIPE, quote
from georg_quality_comparison import read, sha
from scorpion_georg_fem_prepare import write


def run(main):
    main=Path(main).resolve();cfg=copy.deepcopy(read(main/'run_config_priority.local.json'))
    selected=read(main/'selected_candidate.local.json')['best']
    warm=read(main/'warm_seed.local.json')
    for key in ('parameters','pose'):
        if selected[key]!=warm[key]:raise ValueError('A new conditional input pose is required')
    work=main/'material_update_v1';work.mkdir(exist_ok=False)
    cfg.update(work_directory=work.relative_to(PIPE).as_posix(),stage_suffix='conditional_materials',
        profile_max_evaluations=4,profile_max_iterations=3,
        fit_scope='six material parameters at retained pose; separate from limited joint search')
    cfg['sources'] += [dict(path=p.relative_to(PIPE).as_posix(),sha256=sha(p)) for p in
        (Path(__file__).resolve(),main/'selected_candidate.local.json',main/'initial_numerical_audit.local.json')]
    write(work/'run_config.local.json',cfg);job=QualityRun(work/'run_config.local.json')
    rel=lambda p:Path(p).resolve().relative_to(PIPE).as_posix()
    model=(main/cfg['control_models'][-1]).resolve()
    args=["'Profile'","'georg_lateral_inverse'","'OutputDir'",quote(rel(main/'inputs')),
        "'ModelDir'",quote(rel(model)),"'ModelContractFile'",quote(rel(main/'candidate_model_contract.local.json')),
        "'NumericalPolicy'","'nix_preliminary'","'NumericalAuditFile'",quote(rel(main/'initial_numerical_audit.local.json')),
        "'SearchMode'","'profile_grid'","'WarmStartFile'",quote(rel(main/'warm_seed.local.json')),
        "'StartIndices'",'[1]',"'ProfileMaxEvaluations'",'4',"'ProfileMaxIterations'",'3']
    try:
        finished=job.matlab('fixed_pose_material_update',args)
        file=main/'inputs/fits_own_drive/selected_previous_model.local.json'
        result=dict(status='completed_conditional_material_fit' if finished else 'interrupted_original_time_limit',
            maximum_evaluations=4,maximum_iterations=3,
            fit_file=str(file.relative_to(PIPE)) if file.exists() else None,
            fit_sha256=sha(file) if file.exists() else None,
            location_refitted_here=False,global_mesh_convergence_proven=False,
            quantitative_result_accepted=False)
        write(work/'result.local.json',result)
    except BaseException as error:
        write(work/'result.local.json',dict(status='failed',error_type=type(error).__name__,error=str(error),
            quantitative_result_accepted=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--main',type=Path,required=True);a=p.parse_args();run(a.main)
