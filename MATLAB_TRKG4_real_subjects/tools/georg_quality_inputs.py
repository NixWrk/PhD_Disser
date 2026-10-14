"""Prepare a new candidate contract and immutable inverse inputs.

Retains measured levels, confirmed channels, candidate breath intervals and
literature bounds. Does not replace the common accepted-model pointer.
"""
from pathlib import Path
import argparse
import shutil
from georg_quality_comparison import read, sha
from scorpion_georg_fem_prepare import write

PIPE=Path(__file__).resolve().parents[1]


def prepare(work, prior_inputs, reference_file):
    work,prior_inputs,reference_file=map(Path,(work,prior_inputs,reference_file))
    out=work/'inverse_inputs';out.mkdir(exist_ok=False)
    for name in ('observations.local.csv','author_centre_20261012.local.json'):
        shutil.copy2(prior_inputs/name,out/name)
    inputs=read(prior_inputs/'inputs.local.json')
    inputs['sources'].append(dict(path=str((prior_inputs/'inputs.local.json').resolve()),
                                 sha256=sha(prior_inputs/'inputs.local.json')))
    inputs['previous_input_contract_sha256']=sha(prior_inputs/'inputs.local.json')
    inputs['new_input_producer_sha256']=sha(__file__)
    inputs['cross_forward_rule']='all eight lateral and twelve arm contacts physically present; unconnected arm pads are individually floating CEM (continuation assumption)'
    inputs['constant_arm_physical_contact_count']=12
    write(out/'inputs.local.json',inputs)
    reference=read(reference_file)
    grid=dict(schema='georg_joint_lateral_pose_grid_v1',
        surface_sha256=sha(work/'geometry_v5/body_ct_surface.stl'),
        poses=[dict(centre_lps_mm=reference['pose']['centre_lps_mm'])],
        purpose='single numerical initialization only; continuous local joint search is defined by the author centre file',
        inherited_historical_pose_grid=False,global_continuous_optimum_certified=False,
        axes_are_not_independent_anatomical_registration=True,
        fixed_control_reference_sha256=sha(reference_file),producer_sha256=sha(__file__))
    write(out/'pose_grid.local.json',grid)
    paths=[work/'geometry_v5'/name for name in ('body_ct_surface.stl','body_embedded_arms_370mm.stl',
        'geometry.local.json','contour_qc.local.json','fem_contacts.local.json')]
    paths += [work/'mesh_v9/fine'/name for name in ('mesh.mat','plan.local.json','mesh_quality.local.json')]
    paths += [work/'mesh_v9/body_meshing.stl',work/'mesh_v9/surface_qc.local.json',
              work/'mesh_v9/coarse/mesh.mat',work/'mesh_v9/fine/parent_maps.mat']
    contract=dict(schema='georg_current_model_v1',status='geometry_reference_checked',subject_alias='georg',
        model_directory=(work/'geometry_v5').relative_to(PIPE).as_posix(),
        fem_directory=(work/'mesh_v9/fine').relative_to(PIPE).as_posix(),
        mesh_convergence_passed=False,integration_comparison_passed=False,
        quantitative_inversion_allowed=False,independent_anatomical_validation=False,
        physiological_validation=False,placement_accuracy_is_known=False,
        candidate_only=True,common_model_pointer_changed=False,
        scope='quality-checked polyhedron and locally nested mesh; numerical checks remain separate',
        files=[dict(path=path.relative_to(PIPE).as_posix(),sha256=sha(path)) for path in paths])
    write(work/'candidate_model_contract.local.json',contract)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',required=True,type=Path);p.add_argument('--prior-inputs',required=True,type=Path)
    p.add_argument('--reference',required=True,type=Path)
    a=p.parse_args();print(prepare(a.work.resolve(),a.prior_inputs.resolve(),a.reference.resolve()))
