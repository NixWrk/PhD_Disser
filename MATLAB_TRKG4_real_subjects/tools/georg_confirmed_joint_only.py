"""Continue corrected controls and fit all nine parameters without a profile.

The preserved fixed-stage child may be adopted after stopping only its Python
controller. No executed source is edited. Completed controls are verified by
the canonical separated audit; old wrong-boundary predictions are never used.
"""
from pathlib import Path
import argparse, copy, ctypes, time
from georg_quality_rebuild_pipeline import QualityRun, PIPE, quote
from georg_quality_comparison import read, sha
from georg_quality_general_audit import compile_audit
from georg_quality_contact_coverage import patch_coverage
from scorpion_georg_fem_prepare import write


def prepare(work):
    work = Path(work).resolve(); cfg = copy.deepcopy(read(work / 'run_config.local.json'))
    previous = read(work / 'run_state.local.json')
    write(work / 'previous_controller_stopped.local.json', previous)
    cfg.update(conditional_profile_omitted_for_original_time_budget=True,
        adopted_stage_pid=previous['child_pid'], adopted_stage_label=previous['status'],
        preceding_controller_pid=previous['pid'],
        warm_start_source='previous wrong-1x4 fit is an initial guess only; all predicted values recomputed')
    cfg['sources'].append(dict(path=Path(__file__).resolve().relative_to(PIPE).as_posix(), sha256=sha(Path(__file__))))
    file = work / 'run_config_joint_only.local.json'; write(file, cfg); return file


def wait_adopted(pid, deadline):
    # A handle prevents PID reuse during adoption. SYNCHRONIZE | QUERY_LIMITED_INFORMATION.
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x100000 | 0x1000, False, int(pid))
    if not handle: raise OSError('Preserved stage no longer exists; verify its output explicitly')
    try:
        while kernel.WaitForSingleObject(handle, 15000) == 258:
            if time.time() >= deadline: raise TimeoutError('Original deadline while adopting a control')
        code = ctypes.c_uint32()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)): raise OSError('Cannot read adopted-stage exit code')
        if code.value: raise RuntimeError('Adopted MATLAB stage failed: ' + str(code.value))
    finally: kernel.CloseHandle(handle)


class JointOnlyRun(QualityRun):
    def rel(self, p): return Path(p).resolve().relative_to(PIPE).as_posix()

    def run(self):
        try:
            self.publish(status='adopting_completed_or_live_initial_control')
            models = [(self.work / p).resolve() for p in self.config['control_models']]
            files = [self.work / p for p in self.config['initial_control_files']]
            label = self.config['adopted_stage_label']
            labels = ('coarse_initial', 'inherited_initial', 'remapped_initial')
            adopt_index = next(i for i, name in enumerate(labels) if label == name + '_' + self.config['stage_suffix'])
            if not files[adopt_index].exists():
                wait_adopted(self.config['adopted_stage_pid'], self.deadline)
            if not files[adopt_index].exists(): raise ValueError('Adopted stage exited without its saved control')
            for i, (model, file) in enumerate(zip(models, files, strict=True)):
                if file.exists():
                    self.state['stages'].append(dict(label=labels[i], status='saved_control_adopted', source_sha256=sha(file)))
                    continue
                args = ["'Profile'", "'georg_quality_forward'", "'ModelDir'", quote(self.rel(model)),
                    "'InputDir'", quote(self.rel(self.work / 'inputs')), "'ReferenceFile'", quote(self.rel(self.work / 'reference_initial.local.json')),
                    "'OutputFile'", quote(self.rel(file))]
                if i == 0: args += ["'CheckDerivatives'", 'true']
                if not self.matlab(labels[i], args): raise TimeoutError('Corrected initial controls deadline')
            initial = compile_audit(*models, files); write(self.work / self.config['initial_audit_file'], initial)
            required = ('shape_QC_passed', 'energy_inheritance_passed', 'quadrature_passed', 'derivatives_passed', 'system_checks_passed')
            if not all(initial[k] for k in required): raise ValueError('Corrected preliminary numerical prerequisites failed')
            self.publish(status='starting_joint_without_conditional_profile', initial_all_checks_passed=initial['all_checks_passed'])
            super().run()
            coverage = patch_coverage(self.work / 'coarse_forward_selected_v1.local.json', models[0] / 'mesh.mat', models[-1] / 'parent_maps.mat')
            write(self.work / 'contact_coverage_final.local.json', coverage)
            self.publish(status='complete_with_explicit_numerical_results',
                         all_final_contact_faces_refined=coverage['all_actual_contact_faces_refined'], global_mesh_convergence_proven=False)
        except BaseException as error:
            self.publish(status='failed', error_type=type(error).__name__, error=str(error)); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--prepare', type=Path); p.add_argument('--config', type=Path); a = p.parse_args()
    if a.prepare: print(prepare(a.prepare))
    elif a.config: JointOnlyRun(a.config).run()
    else: p.error('Use prepare or config')
