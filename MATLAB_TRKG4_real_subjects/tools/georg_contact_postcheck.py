"""Wait for the continued joint fit, then check depth and direct contact choice.

The initial task deadline is retained. Rebuild around the actual final pose
when it differs from the prepared pose. All FEM calls remain in run_pipeline.
"""
from pathlib import Path
import argparse,copy,time,datetime as dt,os,sys,subprocess
import numpy as np
from georg_quality_rebuild_pipeline import QualityRun,PIPE,quote
from georg_quality_comparison import read,sha
from georg_quality_general_audit import compile_audit
from georg_quality_contact_coverage import patch_coverage
from georg_quality_contact_variant import prepare as contact_variant,compare_saved_controls
from georg_contact_depth_screen import screen
from scorpion_georg_fem_prepare import write


def bounded_preparation(job,command):
    job.verify()
    with (job.work/'depth_preparation.local.log').open('w',encoding='utf-8') as log:
        child=subprocess.Popen(command,cwd=PIPE,stdout=log,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        job.publish(status='preparing_final_depth',child_pid=child.pid)
        while child.poll() is None:
            if time.time()>=job.deadline:
                if os.name=='nt':
                    subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=log,
                        stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW,check=False)
                else:child.kill()
                child.wait(timeout=30)
                raise TimeoutError('Original task deadline during depth preparation')
            try:child.wait(timeout=min(30,max(.1,job.deadline-time.time())))
            except subprocess.TimeoutExpired:job.publish()
        job.publish(child_pid=None)
        if child.returncode:raise RuntimeError('Depth preparation failed; inspect saved log')
    job.verify()


def run(main,hdf5_library=None):
    main=Path(main).resolve();bootstrap=read(main/'run_config.initial.local.json')
    while True:
        state=read(main/'run_state.local.json')
        if state['status']=='complete_with_explicit_numerical_results':break
        if state['status']=='failed':raise ValueError('Main continuation failed')
        if time.time()>=bootstrap['deadline_unix']:raise TimeoutError('Original task deadline')
        time.sleep(15)
    work=main/'postcheck_v1';work.mkdir(exist_ok=False)
    cfg=copy.deepcopy(bootstrap)
    cfg.update(work_directory=work.relative_to(PIPE).as_posix(),stage_suffix='depth_final',
        hdf5_library=str(hdf5_library) if hdf5_library else None,
        sources=[*read(main/'run_config.local.json')['sources'],
            dict(path=str(Path(__file__).resolve().relative_to(PIPE)),sha256=sha(__file__))])
    write(work/'run_config.local.json',cfg)
    job=QualityRun(work/'run_config.local.json')
    rel=lambda p:Path(p).resolve().relative_to(PIPE).as_posix()
    candidate=main/'selected_candidate.local.json';best=read(candidate)['best']
    old=read(main/'warm_seed.local.json')['best']
    parent=(main/bootstrap['control_models'][0]).resolve()
    # A new pose requires new contact and depth guides; old coverage is not reused.
    if best['pose']!=old['pose']:
        ray=work/'contact_CT_profiles_final.local.json'
        screen(candidate,parent,ray)
        deep=work/'deep_final'
        bounded_preparation(job,[sys.executable,'-X','utf8',
            str(PIPE/'tools/georg_quality_contact_depth_rebuild.py'),
            '--candidate',str(candidate),'--parent',str(parent),'--out',str(deep),
            '--parent-forward',str(main/'coarse_forward_selected_v1.local.json'),
            '--ray-profile',str(ray)])
    else:
        deep=(main/bootstrap.get('prepared_depth_root','deep_mesh_v1')).resolve()
    if time.time()>=bootstrap['deadline_unix']:
        raise TimeoutError('Original task deadline reached during depth preparation')
    reference=main/'reference_selected_v1.local.json';files=[main/'coarse_forward_selected_v1.local.json']
    job.publish(status='checking_final_depth',deep_model=rel(deep))
    for label,model in [('deep_inherited',deep/'fine2_inherited'),('deep_remapped',deep/'fine2')]:
        file=work/(label+'.local.json')
        args=["'Profile'","'georg_quality_forward'","'ModelDir'",quote(rel(model)),
            "'InputDir'",quote(rel(main/'inputs')),"'ReferenceFile'",quote(rel(reference)),
            "'OutputFile'",quote(rel(file))]
        if not job.matlab(label,args):raise TimeoutError('Deep comparison deadline')
        files.append(file)
    audit=compile_audit(parent,deep/'fine2_inherited',deep/'fine2',files)
    audit['selected_candidate_sha256']=sha(candidate)
    write(work/'deep_numerical_audit_final.local.json',audit)
    coverage=patch_coverage(files[0],parent/'mesh.mat',deep/'fine2/parent_maps.mat')
    write(work/'deep_contact_coverage_final.local.json',coverage)
    if not coverage['all_actual_contact_faces_refined']:
        raise ValueError('Actual final contacts lack complete depth-branch refinement')
    direct=work/'direct_contacts_final';contact_variant(deep/'fine2',direct)
    file=work/'direct_contacts_final.local.json'
    args=["'Profile'","'georg_quality_forward'","'ModelDir'",quote(rel(direct)),
        "'InputDir'",quote(rel(main/'inputs')),"'ReferenceFile'",quote(rel(reference)),
        "'OutputFile'",quote(rel(file))]
    if not job.matlab('direct_contacts_final',args):raise TimeoutError('Direct contact deadline')
    comparison=compare_saved_controls(deep/'fine2',direct,files[-1],file,
        hdf5_library=cfg['hdf5_library'])
    write(work/'direct_contact_comparison_final.local.json',comparison)
    job.publish(status='complete_with_explicit_numerical_results',
        deep_numerical_checks_passed=audit['all_checks_passed'],
        direct_contact_check_passed=comparison['passed'],
        all_final_contact_faces_refined=True,quantitative_result_accepted=False,
        global_mesh_convergence_proven=False)


def supervise(main,hdf5_library=None):
    """Bound the WHOLE worker, including Python hashing/copies/audits."""
    main=Path(main).resolve();deadline=read(main/'run_config.initial.local.json')['deadline_unix']
    if time.time()>=deadline:raise TimeoutError('Original task deadline before postcheck')
    command=[sys.executable,'-X','utf8',str(Path(__file__).resolve()),'--worker','--main',str(main)]
    if hdf5_library is not None:command+=['--hdf5-library',str(hdf5_library)]
    child=subprocess.Popen(command,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    write(main/'postcheck_supervisor.local.json',dict(pid=os.getpid(),worker_pid=child.pid,
        deadline_unix=deadline,entire_worker_bounded=True))
    while child.poll() is None:
        if time.time()>=deadline:
            if os.name=='nt':
                subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],
                    creationflags=subprocess.CREATE_NO_WINDOW,check=False)
            else:child.kill()
            child.wait(timeout=30)
            raise TimeoutError('Original task deadline for the entire postcheck worker')
        try:child.wait(timeout=min(30,max(.1,deadline-time.time())))
        except subprocess.TimeoutExpired:pass
    if child.returncode:raise RuntimeError('Postcheck worker failed; inspect its saved failure/log')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--main',type=Path,required=True)
    p.add_argument('--hdf5-library',type=Path)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    try:
        if a.worker:run(a.main,a.hdf5_library)
        else:supervise(a.main,a.hdf5_library)
    except BaseException as error:
        failure=dict(status='failed',error_type=type(error).__name__,error=str(error),
            pid=os.getpid(),updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            quantitative_result_accepted=False,global_mesh_convergence_proven=False)
        work=a.main/'postcheck_v1'
        if work.exists():
            state_file=work/'run_state.local.json'
            state=read(state_file) if state_file.exists() else {}
            state.update(failure);write(state_file,state)
        failure_file=a.main/'postcheck_failure.local.json'
        # Preserve a more specific worker error when the supervisor only sees
        # its nonzero exit code. A timeout always records the actual deadline.
        if a.worker or isinstance(error,TimeoutError) or not failure_file.exists():
            write(failure_file,failure)
        raise
