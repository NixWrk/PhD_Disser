"""Fresh final-depth retry after the preserved two-promotion-round failure.

The entire worker and all subprocesses retain the original numerical deadline.
Only the bounded number of green-parent promotions changes; no QC relaxation.
"""
from pathlib import Path
import argparse,copy,os,shutil,subprocess,sys,time
from georg_contact_postcheck import bounded_preparation
from georg_quality_rebuild_pipeline import QualityRun,PIPE,quote
from georg_quality_comparison import read,sha
from georg_quality_general_audit import compile_audit
from georg_quality_contact_coverage import patch_coverage
from georg_quality_contact_variant import prepare as contact_variant,compare_saved_controls
from scorpion_georg_fem_prepare import write


def worker(main,library):
    main=Path(main).resolve();work=main/'postcheck_v2'
    cfg=read(work/'run_config.local.json');job=QualityRun(work/'run_config.local.json')
    candidate=main/'selected_candidate.local.json';parent=(main/cfg['control_models'][0]).resolve()
    source=main/'postcheck_v1/contact_CT_profiles_final.local.json';ray=work/source.name
    if read(source)['candidate_sha256']!=sha(candidate):raise ValueError('Preserved rays belong to another pose')
    shutil.copyfile(source,ray);deep=work/'deep_final'
    bounded_preparation(job,[sys.executable,'-X','utf8',str(PIPE/'tools/georg_contact_depth_retry_prepare.py'),
        '--candidate',str(candidate),'--parent',str(parent),'--out',str(deep),
        '--parent-forward',str(main/'coarse_forward_selected_v1.local.json'),'--ray-profile',str(ray),
        '--promotion-rounds','6'])
    job.publish(status='checking_final_depth',deep_model=deep.relative_to(PIPE).as_posix())
    rel=lambda p:Path(p).resolve().relative_to(PIPE).as_posix()
    files=[main/'coarse_forward_selected_v1.local.json'];reference=main/'reference_selected_v1.local.json'
    for label,model in [('deep_inherited',deep/'fine2_inherited'),('deep_remapped',deep/'fine2')]:
        file=work/(label+'.local.json')
        if not job.matlab(label,["'Profile'","'georg_quality_forward'","'ModelDir'",quote(rel(model)),
            "'InputDir'",quote(rel(main/'inputs')),"'ReferenceFile'",quote(rel(reference)),"'OutputFile'",quote(rel(file))]):
            raise TimeoutError('Original deadline in depth retry')
        files.append(file)
    audit=compile_audit(parent,deep/'fine2_inherited',deep/'fine2',files)
    audit['selected_candidate_sha256']=sha(candidate);write(work/'deep_numerical_audit_final.local.json',audit)
    coverage=patch_coverage(files[0],parent/'mesh.mat',deep/'fine2/parent_maps.mat')
    write(work/'deep_contact_coverage_final.local.json',coverage)
    if not coverage['all_actual_contact_faces_refined']:raise ValueError('Incomplete actual-final-contact refinement')
    direct=work/'direct_contacts_final';contact_variant(deep/'fine2',direct);file=work/'direct_contacts_final.local.json'
    if not job.matlab('direct_contacts_final',["'Profile'","'georg_quality_forward'","'ModelDir'",quote(rel(direct)),
        "'InputDir'",quote(rel(main/'inputs')),"'ReferenceFile'",quote(rel(reference)),"'OutputFile'",quote(rel(file))]):
        raise TimeoutError('Original deadline in contact retry')
    comparison=compare_saved_controls(deep/'fine2',direct,files[-1],file,hdf5_library=library)
    write(work/'direct_contact_comparison_final.local.json',comparison)
    job.publish(status='complete_with_explicit_numerical_results',deep_numerical_checks_passed=audit['all_checks_passed'],
        direct_contact_check_passed=comparison['passed'],all_final_contact_faces_refined=True,
        quantitative_result_accepted=False,global_mesh_convergence_proven=False)


def supervise(main,library):
    main=Path(main).resolve();bootstrap=read(main/'run_config.initial.local.json');deadline=bootstrap['deadline_unix']
    if time.time()>=deadline:raise TimeoutError('Original deadline before final-depth retry')
    work=main/'postcheck_v2';work.mkdir(exist_ok=False)
    cfg=copy.deepcopy(bootstrap)
    cfg.update(work_directory=work.relative_to(PIPE).as_posix(),stage_suffix='depth_retry',
        sources=[*read(main/'run_config.local.json')['sources'],
            *[dict(path='tools/'+name,sha256=sha(PIPE/'tools'/name)) for name in
              ('georg_contact_postcheck_retry.py','georg_contact_depth_retry_prepare.py')]],
        prior_failed_attempt_directory='postcheck_v1',promotion_rounds=6,
        quality_limits_relaxed=False,maximum_fine_nodes=1000000)
    write(work/'run_config.local.json',cfg)
    command=[sys.executable,'-X','utf8',str(Path(__file__).resolve()),'--worker','--main',str(main),'--hdf5-library',str(library)]
    child=subprocess.Popen(command,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    write(work/'supervisor.local.json',dict(pid=os.getpid(),worker_pid=child.pid,deadline_unix=deadline,whole_worker_bounded=True))
    while child.poll() is None:
        if time.time()>=deadline:
            subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],creationflags=subprocess.CREATE_NO_WINDOW,check=False)
            child.wait(timeout=30);raise TimeoutError('Original numerical deadline for whole depth retry')
        try:child.wait(timeout=min(30,max(.1,deadline-time.time())))
        except subprocess.TimeoutExpired:pass
    if child.returncode:raise RuntimeError('Depth retry worker failed; inspect preserved evidence')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--main',type=Path,required=True)
    p.add_argument('--hdf5-library',type=Path,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args()
    try:
        worker(a.main,a.hdf5_library) if a.worker else supervise(a.main,a.hdf5_library)
    except BaseException as error:
        work=a.main/'postcheck_v2'
        if work.is_dir() and not (work/'failure.local.json').exists():
            write(work/'failure.local.json',dict(error_type=type(error).__name__,error=str(error),quantitative_result_accepted=False))
        raise
