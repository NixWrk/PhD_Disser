"""Optional fixed-bulk contact-discretization check within the original deadline.

Wait for the corrected joint candidate and its three separated controls.
Never reruns a fit. Missing time is reported as missing evidence, not a pass.
"""
from pathlib import Path
import argparse, copy, time
from georg_quality_rebuild_pipeline import QualityRun, PIPE, quote
from georg_quality_comparison import read, sha
from georg_quality_contact_variant import prepare, compare_saved_controls
from scorpion_georg_fem_prepare import write


def run(main, library, config_name='run_config_joint_only.local.json', output_name='contact_check_v1'):
    main = Path(main).resolve(); cfg = read(main / config_name)
    work = main / output_name; work.mkdir(exist_ok=False)
    deadline = cfg['deadline_unix']
    write(work / 'waiting.local.json', dict(status='waiting_corrected_joint_controls', deadline_unix=deadline))
    while time.time() < deadline:
        state = read(main / 'run_state.local.json')
        if state['status'] == 'failed':
            write(work / 'result.local.json', dict(status='not_executed_main_failed', quantitative_result_accepted=False)); return
        if state['status'] == 'complete_with_explicit_numerical_results': break
        time.sleep(min(15, max(.1, deadline - time.time())))
    remaining = deadline - time.time()
    if remaining < 12 * 60:
        write(work / 'result.local.json', dict(status='not_executed_original_time_limit', remaining_seconds=remaining,
            minimum_reserved_seconds=720, passed=None, quantitative_result_accepted=False)); return
    fine = (main / cfg['control_models'][-1]).resolve()
    cfg = copy.deepcopy(cfg)
    cfg.update(work_directory=work.relative_to(PIPE).as_posix(), stage_suffix='confirmed_direct_contacts')
    cfg['sources'] += [dict(path=p.relative_to(PIPE).as_posix(), sha256=sha(p)) for p in
        (Path(__file__).resolve(), main / 'selected_candidate.local.json', main / 'reference_selected_v1.local.json',
         main / 'fine_remapped_selected_v1.local.json')]
    write(work / 'run_config.local.json', cfg); job = QualityRun(work / 'run_config.local.json')
    variant = work / 'direct'; prepare(fine, variant)
    file = work / 'direct_forward.local.json'
    rel = lambda p: Path(p).resolve().relative_to(PIPE).as_posix()
    args = ["'Profile'", "'georg_quality_forward'", "'ModelDir'", quote(rel(variant)),
        "'InputDir'", quote(rel(main / 'inputs')), "'ReferenceFile'", quote(rel(main / 'reference_selected_v1.local.json')),
        "'OutputFile'", quote(rel(file))]
    if not job.matlab('direct_contact_check', args):
        write(work / 'result.local.json', dict(status='interrupted_original_time_limit', passed=None,
            quantitative_result_accepted=False)); return
    comparison = compare_saved_controls(fine, variant, main / 'fine_remapped_selected_v1.local.json', file,
                                       hdf5_library=library)
    write(work / 'comparison.local.json', comparison)
    write(work / 'result.local.json', dict(status='completed_fixed_bulk_contact_comparison',
        passed=comparison['passed'], comparison_sha256=sha(work / 'comparison.local.json'),
        quantitative_result_accepted=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--main', type=Path, required=True)
    p.add_argument('--hdf5-library', type=Path, required=True)
    p.add_argument('--config-name', default='run_config_joint_only.local.json')
    p.add_argument('--output-name', default='contact_check_v1'); a = p.parse_args()
    try: run(a.main, a.hdf5_library, a.config_name, a.output_name)
    except BaseException as error:
        work = a.main / a.output_name
        if work.is_dir(): write(work / 'result.local.json', dict(status='failed', error_type=type(error).__name__,
            error=str(error), passed=None, quantitative_result_accepted=False))
        raise
