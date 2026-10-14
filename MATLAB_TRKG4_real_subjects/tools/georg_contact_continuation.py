"""Initial mesh controls, analytic-Jacobian tissue fit, joint search and controls.

All electrical calculations use the canonical run_pipeline entry point.
The fixed-pose material fit is a diagnostic and an initial guess for the joint
search. It never establishes the location or tissue parameters independently.
"""
from pathlib import Path
import argparse, copy
import numpy as np
from georg_quality_rebuild_pipeline import QualityRun,PIPE,quote
from georg_quality_comparison import read,sha
from georg_quality_general_audit import compile_audit
from georg_quality_contact_coverage import patch_coverage
from scorpion_georg_fem_prepare import write


class ContactRun(QualityRun):
    def relative(self,p):return Path(p).resolve().relative_to(PIPE).as_posix()

    def matlab(self,label,args,search=False):
        if search:
            old=quote(self.relative(self.work/'warm_seed.local.json'))
            new=quote(self.relative(self.work/'warm_profile.local.json'))
            if old not in args:raise ValueError('Expected explicit warm-start argument')
            args=[new if value==old else value for value in args]
        return super().matlab(label,args,search)

    def run(self):
        try:
            self.publish()
            models=[(self.work/p).resolve() for p in self.config['control_models']]
            files=[self.work/p for p in self.config['initial_control_files']]
            for label,model,file in zip(('inherited_initial','remapped_initial'),models[1:],files[1:]):
                args=["'Profile'","'georg_quality_forward'","'ModelDir'",quote(self.relative(model)),
                    "'InputDir'",quote(self.relative(self.work/'inputs')),
                    "'ReferenceFile'",quote(self.relative(self.work/'reference_initial.local.json')),
                    "'OutputFile'",quote(self.relative(file))]
                if not self.matlab(label,args):raise TimeoutError('Initial control deadline')
            initial=compile_audit(*models,files)
            write(self.work/self.config['initial_audit_file'],initial)
            if not initial['all_checks_passed']:
                self.publish(status='initial_numerical_failure_preserved')
                raise ValueError('All-contact initial numerical comparisons failed; keep diagnostics')
            args=["'Profile'","'georg_lateral_inverse'","'OutputDir'",quote(self.relative(self.work/'inputs')),
                "'ModelDir'",quote(self.relative(models[-1])),
                "'ModelContractFile'",quote(self.relative(self.work/'candidate_model_contract.local.json')),
                "'NumericalPolicy'","'nix_preliminary'","'NumericalAuditFile'",quote(self.relative(self.work/self.config['initial_audit_file'])),
                "'SearchMode'","'profile_grid'","'WarmStartFile'",quote(self.relative(self.work/'warm_seed.local.json')),
                "'StartIndices'",'[1]',"'ProfileMaxEvaluations'",'18',"'ProfileMaxIterations'",'16']
            if not self.matlab('fixed_pose_material_profile',args,search=False):
                raise TimeoutError('Conditional material fit did not finish')
            profile=read(self.work/'inputs/fits_own_drive/selected_previous_model.local.json')
            prior=read(self.work/'warm_seed.local.json')['best']
            warm=copy.deepcopy(prior)
            warm['parameters'][:6]=np.log([*profile['rho_inhale_ohm_m'],*profile['rho_exhale_ohm_m']]).tolist()
            warm['status']='conditional_material_fit_for_new_joint_initialization'
            warm['profile_source_sha256']=sha(self.work/'inputs/fits_own_drive/selected_previous_model.local.json')
            warm.pop('predicted_Z_ohm',None);warm.pop('residual_ohm',None)
            warm['previous_model_predictions_not_transferred']=True
            file=self.work/'warm_profile.local.json';write(file,warm)
            self.config['sources'].append(dict(path=self.relative(file),sha256=sha(file)))
            self.config['actual_search_warm_start_file']=self.relative(file)
            write(self.work/'run_config.local.json',self.config)
            self.publish(status='starting_joint_after_material_profile',
                fixed_pose_material_RMSE_ohm=profile['RMSE_ohm'],
                fixed_pose_material_evaluations=profile['evaluations'])
            super().run()
            coverage=patch_coverage(self.work/'coarse_forward_selected_v1.local.json',
                models[0]/'mesh.mat',models[-1]/'parent_maps.mat')
            write(self.work/'contact_coverage_final.local.json',coverage)
            self.publish(status='complete_with_explicit_numerical_results',
                all_final_contact_faces_refined=coverage['all_actual_contact_faces_refined'],
                global_mesh_convergence_proven=False)
        except BaseException as error:
            self.publish(status='failed',error_type=type(error).__name__,error=str(error))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    ContactRun(p.parse_args().config).run()
