"""Finish selected-point checks after an interrupted search, without new search.

Uses the frozen convergence controller and canonical run_pipeline forwards.
The original hard deadline, observations, parameter bounds and numerical
tolerances remain in force. Interrupted/incomplete evaluations are retained.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np

from georg_convergence_continue import (
    ConvergenceRun, PIPE, quote, read, sha, write_new, compile_audit,
    compare_saved_controls, fixed_point_compare, nested_rule_compare,
    verify_independent_integrations)
from georg_refined_contact_coverage import compute_coverage


def completed_choice(directory, observations_sha):
    valid, inventory = [], []
    for path in sorted(Path(directory).glob('evaluation_*.local.json')):
        row=read(path)
        inventory.append(dict(path=str(path.resolve()),sha256=sha(path),status=row['status']))
        if row['status']!='numerical_candidate_not_validated':continue
        z=np.asarray(row['predicted_Z_ohm'],float)
        target=np.asarray(row['target_Z_ohm'],float)
        if (z.shape!=(24,) or target.shape!=(24,) or
                not np.isfinite(np.r_[z,target,row['parameters']]).all() or
                row['identity']['observations_sha256']!=observations_sha):
            raise ValueError('Completed checkpoint has invalid levels or observation identity')
        expected=float(np.mean((z-target)**2))
        if not np.isclose(row['mean_square_ohm2'],expected,rtol=1e-12,atol=1e-12):
            raise ValueError('Checkpoint objective is not the original unweighted criterion')
        valid.append((path,row))
    if not valid:raise ValueError('No complete numerical checkpoint')
    source,best=min(valid,key=lambda item:item[1]['mean_square_ohm2'])
    return dict(best=best,source_checkpoint_sha256=sha(source),source_checkpoint=str(source.resolve()),
        checkpoint_inventory=inventory,completed_numerical_evaluations=len(valid),
        search_interrupted=True,search_stop_cause='process absent; exact cause not established',
        search_stopped_for_time=False,search_not_resumed_after_search_deadline=True,
        resolution_reached=False,quantitative_result_accepted=False,physical_validation=False)


class SelectedChecks(ConvergenceRun):
    def run(self):
        try:
            self.verify()
            for pending in self.config.get('handoff_pending_selected_controls',[]):
                while not (self.work/pending).is_file():
                    if time.time()>=self.deadline:raise TimeoutError('Pending control hard deadline')
                    self.publish(status='waiting_preexisting_selected_control',pending_control=pending)
                    time.sleep(15)
            if time.time()<self.search_deadline:
                raise ValueError('This recovery mode is for the expired search window only')
            if time.time()>=self.deadline:raise TimeoutError('Original hard deadline expired')
            gate=read(self.work/'initial_convergence_gate.local.json')
            if not gate['passed']:raise ValueError('Original fixed-point numerical gate failed')
            directory=self.work/'inputs'/('joint_own_drive_'+self.config['run_label'])
            selected=completed_choice(directory,sha(self.work/'inputs/observations.local.csv'))
            target=self.work/'selected_new_candidate.local.json'
            self.save_receipt(target,selected)
            reference=read(self.work/'reference_selected_v1.local.json')
            reference.update(parameters=selected['best']['parameters'],pose=selected['best']['pose'],
                selected_candidate_sha256=sha(target))
            ref=self.work/'reference_new_candidate.local.json';self.save_receipt(ref,reference)
            self.publish(status='checking_new_selected_candidate_after_interruption',
                search_resumed=False,original_hard_deadline_preserved=True)
            model=self.work/'integration_q16384_seed20261008'
            patchfile=self.work/'selected_contact_geometry_probe.local.json'
            if not patchfile.exists():
                args=["'Profile'","'georg_quality_forward'","'ModelDir'",quote(self.rel(model)),
                    "'InputDir'",quote(self.rel(self.work/'inputs')),"'ReferenceFile'",quote(self.rel(ref)),
                    "'OutputFile'",quote(self.rel(patchfile)),"'Stage'","'contacts'"]
                if not self.matlab('selected_contact_geometry_probe',args):raise TimeoutError('Contact geometry deadline')
            probe=compute_coverage(patchfile,model/'mesh.mat',self.work/'mesh_level_3_r3/fine2/parent_maps.mat')
            self.save_receipt(self.work/'selected_contact_geometry_probe_coverage.local.json',probe)
            self.publish(actual_selected_contact_coverage_passed=probe['all_contact_faces_refined'])
            checked=[]
            for name in ('parent_direct','level3_inherited','level3_remapped','level3_direct'):
                checked.append(self.forward(self.work/name,name+'_new_candidate',
                    derivatives=name=='parent_direct',reference=ref))
            final=compile_audit(*[self.work/p for p in ('parent_direct','level3_inherited','level3_remapped')],checked[:3])
            contact=compare_saved_controls(self.work/'level3_remapped',self.work/'level3_direct',*checked[2:4],
                hdf5_library=self.config['hdf5_library'])
            self.save_receipt(self.work/'selected_new_numerical_audit.local.json',final)
            self.save_receipt(self.work/'selected_new_contact_comparison.local.json',contact)
            models=[self.work/name for name in ('integration_q16384_seed20261008','integration_q16384_seed20261015')]
            files=[self.forward(model,model.name+'_new_candidate',reference=ref,derivatives=k==0)
                for k,model in enumerate(models)]
            rows=[read(p) for p in files]
            identity=verify_independent_integrations(models,rows,self.config['hdf5_library'])
            integration=dict(
                spatial_respiratory=fixed_point_compare(read(checked[0]),read(checked[1]),relative=.005),
                tissue_respiratory=fixed_point_compare(read(checked[1]),read(checked[2]),relative=.005),
                contact_respiratory=fixed_point_compare(read(checked[2]),read(checked[3]),relative=.005),
                rule_growth=fixed_point_compare(read(checked[3]),rows[0]),
                independent_seed=fixed_point_compare(*rows))
            for k,row in enumerate(rows):integration[f'nested_rules_seed_{k+1}']=nested_rule_compare(row)
            integration['passed']=identity['passed'] and all(r['passed'] for r in integration.values()) and all(
                r['quadrature_passed'] and r['system_checks_passed'] and r['patch_qc']['passed'] for r in rows)
            integration.update(selected_candidate_sha256=sha(target),quantitative_result_accepted=False,
                integration_identity=identity)
            self.save_receipt(self.work/'selected_new_integration_stability.local.json',integration)
            coverage=compute_coverage(files[0],models[0]/'mesh.mat',self.work/'mesh_level_3_r3/fine2/parent_maps.mat')
            self.save_receipt(self.work/'selected_fine_actual_contact_coverage.local.json',coverage)
            self.publish(status='joint_finished_with_selected_checks',search_resumed=False,
                selected_numerical_checks_passed=final['all_checks_passed'] and contact['passed'] and integration['passed'],
                actual_selected_contact_coverage_passed=coverage['all_contact_faces_refined'],
                quantitative_result_accepted=False)
        except BaseException as error:
            self.publish(status='failed',error_type=type(error).__name__,error=str(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    SelectedChecks(parser.parse_args().config).run()
