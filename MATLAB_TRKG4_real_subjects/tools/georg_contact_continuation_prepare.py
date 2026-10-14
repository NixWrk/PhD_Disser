"""Fresh continuation inputs: measured values unchanged, technical rho bounds.

Local runtime settings and medical input paths remain in excluded JSON files.
"""
from pathlib import Path
import argparse, copy, json, shutil, subprocess, time
from georg_quality_comparison import read, sha
from scorpion_georg_fem_prepare import write
PIPE=Path(__file__).resolve().parents[1]


def prepare(work, prior, mesh, candidate, parent_forward, old_runtime):
    work,prior,mesh,candidate,parent_forward=map(lambda p:Path(p).resolve(),
        (work,prior,mesh,candidate,parent_forward))
    inputs=work/'inputs';inputs.mkdir(exist_ok=False)
    for name in ('observations.local.csv','author_centre_20261012.local.json'):
        shutil.copy2(prior/name,inputs/name)
    d=read(prior/'inputs.local.json')
    for r in d['sources']:
        if sha(r['path'])==r['sha256']:continue
        path=Path(r['path']).resolve()
        if path.name!='PROTOCOL.md':raise ValueError('Experimental dependency changed: '+str(path))
        relative=path.relative_to(PIPE.parent).as_posix()
        content=subprocess.check_output(['git','show','efc87b1:'+relative],cwd=PIPE.parent)
        import hashlib
        if hashlib.sha256(content).hexdigest()!=r['sha256']:
            raise ValueError('Exact prior protocol is unavailable')
        frozen=work/'source_snapshots/PROTOCOL_before_20261013.md'
        frozen.parent.mkdir(parents=True,exist_ok=True);frozen.write_bytes(content)
        r['path']=str(frozen)
    # Changing bounds is an explicit new problem; never rewrite the prior input.
    d.update(search_bounds_ohm_m=[[.1,100.]]*3,
        physiological_bounds_used_as_hard_constraints=False,
        bounds_are_individually_validated=False,automatic_bound_expansion_allowed=False,
        bounds_scope='author technical FEM inverse limits, 0.1–100 ohm m per free material',
        rho_index_scope='lateral only: soft background, lung, fat; whole-heart material fixed',
        parameter_names=['rho_soft','rho_lung','rho_fat'],
        previous_input_contract_sha256=sha(prior/'inputs.local.json'),
        new_input_producer_sha256=sha(__file__),annotation_acceptance_changed=False)
    d['sources'] += [dict(path=str(prior/'inputs.local.json'),sha256=sha(prior/'inputs.local.json'))]
    write(inputs/'inputs.local.json',d)
    selected=read(candidate)['best'];selected_pose=copy.deepcopy(selected['pose'])
    selected_pose['pose_id']='selected_previous_model'
    common=read(PIPE/'output/georg_current_model.local.json')
    geometry=PIPE/common['model_directory']
    write(inputs/'pose_grid.local.json',dict(schema='georg_joint_lateral_pose_grid_v1',
        surface_sha256=sha(geometry/'body_ct_surface.stl'),poses=[selected_pose],
        fixed_pose_from_previous_model=True,independent_placement_measurement=False))
    for name,source in [('warm_seed.local.json',candidate),
        ('reference_initial.local.json',parent_forward.parent/'reference_selected_v1.local.json'),
        ('coarse_forward_initial.local.json',parent_forward)]:shutil.copy2(source,work/name)
    files=[geometry/name for name in ('body_ct_surface.stl','fem_contacts.local.json')]
    files += [mesh/'fine2'/name for name in ('plan.local.json','mesh.mat','mesh_quality.local.json')]
    contract=dict(schema='georg_current_model_v1',status='geometry_reference_checked',
        model_directory=geometry.relative_to(PIPE).as_posix(),
        fem_directory=(mesh/'fine2').relative_to(PIPE).as_posix(),candidate_only=True,
        mesh_convergence_passed=False,integration_comparison_passed=False,
        quantitative_inversion_allowed=False,common_model_pointer_changed=False,
        files=[dict(path=p.relative_to(PIPE).as_posix(),sha256=sha(p)) for p in files])
    write(work/'candidate_model_contract.local.json',contract)
    runtime=read(old_runtime)
    begin=work.stat().st_ctime
    cfg=dict(work_directory=work.relative_to(PIPE).as_posix(),
        matlab_executable=runtime['matlab_executable'],eidors_startup=runtime['eidors_startup'],
        deadline_unix=begin+10*3600,search_deadline_unix=begin+8*3600,
        deadline_includes_preparation=True,input_directory='inputs',
        control_models=['parent_placeholder',
            'contact_mesh_v1/fine2_inherited','contact_mesh_v1/fine2'],
        independent_contact_master=True,max_evaluations=80,start_indices=[1],
        run_label='all_contact_refined_technical80',stage_suffix='contact80',
        initial_audit_file='initial_numerical_audit.local.json',
        initial_reference_file='reference_initial.local.json',
        initial_control_files=['coarse_forward_initial.local.json','inherited_initial.local.json','remapped_initial.local.json'],
        sources=[])
    # Path.relative_to cannot express the sibling parent; use an explicit relative path.
    import os
    cfg['control_models'][0]=Path(os.path.relpath(parent_forward.parent.parent/'base',work)).as_posix()
    source_paths=[*files,*inputs.glob('*.json'),inputs/'observations.local.csv',
        work/'warm_seed.local.json',work/'reference_initial.local.json',
        work/'coarse_forward_initial.local.json',Path(__file__)]
    source_paths += list((PIPE/'src').glob('*.m'))
    source_paths += list((PIPE/'tools').glob('georg*.py'))
    source_paths += [PIPE/'run_pipeline.m']
    for p in dict.fromkeys(source_paths):
        path=p.relative_to(PIPE).as_posix();snapshot=work/'source_snapshots'/path
        snapshot.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,snapshot)
        cfg['sources'].append(dict(path=path,sha256=sha(p),snapshot=snapshot.relative_to(PIPE).as_posix()))
    write(work/'run_config.local.json',cfg)
    return cfg


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('work','prior','mesh','candidate','parent-forward','old-runtime'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();prepare(a.work,a.prior,a.mesh,a.candidate,a.parent_forward,a.old_runtime)
