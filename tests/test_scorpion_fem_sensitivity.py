"""Known-matrix tests of postprocessing; not physiological/FEM validation."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SOURCE=Path(__file__).resolve().parents[1]/"MATLAB_TRKG4_real_subjects/tools/scorpion_fem_sensitivity.py"
SPEC=importlib.util.spec_from_file_location("scorpion_fem_sensitivity",SOURCE)
fem=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fem)


def baseline(cases=2):
    """A signed known Jacobian with two null directions in a single case."""
    answers=[]
    for state in ("inhale","exhale"):
        for rule in (1024,4096):
            for index in range(cases):
                t=np.eye(4)*100
                t[1,0]=t[0,1]=-10-index
                t[3,2]=t[2,3]=5+index
                j=np.zeros((4,4,4))
                j[1,0,:]=[-1,2,3,4] if index==0 else [2,0,1,0]
                j[3,2,:]=[1,-2,0,1] if index==0 else [0,1,0,0]
                if rule==1024:j*=.99
                answers.append(dict(state=state,rule=rule,case_id=f"case{index}",rho_ohm_m=[2,8,30,1.5,20],
                    transfer_matrix_ohm=t.tolist(),jacobian_logrho=j.tolist(),
                    ttrkg_Z_ohm=t[1,0],lateral_Z_ohm=t[3,2]))
    return dict(schema="scorpion_own_CT_forward_v1",subject_alias="nik",frequency_hz=100000,
        identity=dict(subject_alias="nik"),FEM_was_run=True,source_relative_phase="unknown",
        own_drive_is_working_assumption=True,answers=answers,integration_passed=True,
        derivative_checks=dict(case_id="case0",state="inhale",active_material_indices=[1,2,4,5],
            passed=True,relative_errors=[1e-7]*4,log_step=.001))


def analysis(report,subset="joint",state="inhale"):
    return next(a for a in fem.analyze_baseline(report)["analyses"] if a["subset"]==subset and a["state"]==state)


def test_underdetermined_matrix_has_structural_nullspace_and_infinite_condition():
    a=np.array([[1.,2.,0.,0.],[0.,0.,1.,1.]])
    result=fem.matrix_diagnostics(a)
    assert result["rank"]==2 and result["nullity"]==2
    assert result["condition_number_full_parameter_space"] is None
    assert result["condition_status"]=="rank_deficient_infinite"
    assert result["parameter_space_singular_values"][-2:]==[0,0]
    np.testing.assert_allclose(a@np.array(result["nullspace_basis"]).T,0,atol=1e-14)
    assert not result["full_column_rank"]


def test_normalization_and_rho_chain_rule_keep_signs_and_order():
    value=analysis(baseline(1))
    assert value["row_labels"]==[dict(case_id="case0",role="lateral"),dict(case_id="case0",role="ttrkg")]
    jlog=np.array([[1,-2,0,1],[-1,2,3,4]])
    np.testing.assert_allclose(value["matrices"]["dZ_dlogrho"]["matrix"],jlog)
    np.testing.assert_allclose(value["matrices"]["dZ_drho"]["matrix"],jlog/[2,8,1.5,20])
    np.testing.assert_allclose(value["matrices"]["dimensionless_signed_elasticity"]["matrix"],jlog/np.array([[5],[-10]]))
    assert value["matrices"]["dimensionless_signed_elasticity"]["matrix"][1][1]<0


def test_rank_is_computed_for_lateral_tt_and_joint_and_never_pools_states():
    report=fem.analyze_baseline(baseline())
    assert len(report["analyses"])==6
    assert analysis(baseline(),"lateral")["matrices"]["dZ_drho"]["rank"]==2
    assert analysis(baseline(),"ttrkg")["matrices"]["dZ_drho"]["rank"]==2
    assert analysis(baseline())["matrices"]["dZ_drho"]["rank"]==4
    assert all(len(a["row_labels"])<8 for a in report["analyses"])
    assert analysis(baseline(),state="exhale")["exhale_is_fixed_inhale_CT_conditional_scenario"]
    assert not report["practically_identifiable"]


def test_negative_or_zero_Z_has_no_abs_or_arbitrary_floor():
    source=baseline(1)
    for a in source["answers"]:
        a["transfer_matrix_ohm"][1][0]=a["ttrkg_Z_ohm"]=0
    value=analysis(source)
    assert value["normalization_status"]=="undefined_zero_Z"
    assert value["matrices"]["dimensionless_signed_elasticity"] is None
    assert value["matrices"]["dZ_drho"]["rank"]==2
    json.dumps(fem.analyze_baseline(source),allow_nan=False)


def test_quadrature_control_is_signed_observed_difference_and_fd_scope_is_local():
    source=baseline()
    value=analysis(source)
    fine=np.array(value["matrices"]["dZ_dlogrho"]["matrix"])
    control=value["quadrature_control"]["matrices"]["dZ_dlogrho"]
    np.testing.assert_allclose(control["signed_difference"],fine*.01,atol=1e-14)
    assert control["difference_is_not_error_bound_or_noise_variance"]
    assert value["finite_difference_control_applies_to_at_least_one_case"]
    assert not value["finite_difference_control_covers_all_cases"]
    assert not analysis(source,state="exhale")["finite_difference_control_applies_to_at_least_one_case"]
    assert not fem.analyze_baseline(source)["noise_variance_assigned"]


def test_rank_threshold_is_explicit_and_zero_matrix_is_serializable():
    a=np.diag([1,1e-10,0,0])
    assert fem.matrix_diagnostics(a)["rank"]==2
    assert fem.matrix_diagnostics(a,rank_rtol=1e-8)["rank"]==1
    zero=fem.matrix_diagnostics(np.zeros((1,4)))
    assert zero["rank"]==0 and zero["nullity"]==4
    assert zero["condition_number_on_resolved_subspace"] is None
    json.dumps(zero,allow_nan=False)


def test_rank_metadata_is_native_float_after_geometry_import_in_a_fresh_process():
    import subprocess
    import sys
    root=SOURCE.parents[2]
    script="""
import importlib.util, json, sys
from pathlib import Path
root=Path.cwd();sys.path.insert(0,str(root/'tools'))
import ct_fem_core
spec=importlib.util.spec_from_file_location('sensitivity',root/'MATLAB_TRKG4_real_subjects/tools/scorpion_fem_sensitivity.py')
fem=importlib.util.module_from_spec(spec);spec.loader.exec_module(fem)
import numpy as np
for rtol in (None,np.longdouble('1e-10')):
    result=fem.matrix_diagnostics(np.zeros((1,4)),rank_rtol=rtol)
    assert type(result['rank_relative_tolerance']) is float
    json.dumps(result,allow_nan=False)
json.dumps(fem.analyze_baseline(json.loads(sys.stdin.read())),allow_nan=False)
"""
    completed=subprocess.run([sys.executable,"-c",script],input=json.dumps(baseline()),text=True,
        cwd=root,capture_output=True,timeout=60)
    assert completed.returncode==0,completed.stderr


@pytest.mark.parametrize("mutation",[
    lambda r:r.update(subject_alias="georg"),
    lambda r:r.update(frequency_hz=50000),
    lambda r:r.update(FEM_was_run=False),
    lambda r:r.update(source_relative_phase="zero"),
    lambda r:r["derivative_checks"].update(active_material_indices=[1,2,3,4]),
    lambda r:r["answers"].append(deepcopy(r["answers"][0])),
    lambda r:r["answers"][0].update(jacobian_logrho=[1,2,3,4]),
    lambda r:r["answers"][0].update(ttrkg_Z_ohm=999),
    lambda r:r["answers"][0].update(rho_ohm_m=[2,9,30,1.5,20]),
    lambda r:r["answers"][0].update(subject_alias="yaroslav"),
    lambda r:r["answers"].pop(),
])
def test_refuses_mixed_subjects_changed_order_wrong_units_unpaired_or_corrupt_data(mutation):
    report=baseline();mutation(report)
    with pytest.raises(ValueError):fem.analyze_baseline(report,subject="nik")


def test_absent_fd_is_not_manufactured():
    report=baseline();report.pop("derivative_checks")
    result=fem.analyze_baseline(report)
    assert not result["numerical_controls_complete"]
    assert not any(a["finite_difference_control_applies_to_at_least_one_case"] for a in result["analyses"])


def test_partial_fd_pass_does_not_become_complete_control():
    report=fem.analyze_baseline(baseline())
    assert report["baseline_reported_numerical_checks_passed"]
    assert not report["numerical_controls_complete"]


def test_completed_v2_limited_report_retains_numbers_without_inventing_quadrature_or_fd():
    report=baseline(1)
    report.update(schema="scorpion_own_CT_forward_v2",active_material_indices=[1,2,4,5],
        limited_run=True,comparison=[],integration_passed=False,
        derivative_checks=dict(performed=False,passed=False))
    report["answers"]=[a for a in report["answers"] if a["state"]=="inhale" and a["rule"]==4096]
    report["system_checks"]=[dict(system_relative_residual=1e-12,reciprocity_ohm=1e-12)]
    result=fem.analyze_baseline(report)
    assert result["baseline_numbers_available"] and result["baseline_limited_run"]
    assert len(result["analyses"])==3
    assert all(a["quadrature_control"]["status"]=="not_run_both_rules" for a in result["analyses"])
    assert not result["numerical_controls_complete"]
    singleton=deepcopy(report)
    singleton["answers"]=singleton["answers"][0]
    singleton["system_checks"]=singleton["system_checks"][0]
    single_result=fem.analyze_baseline(singleton)
    assert single_result["analyses"][0]["Z_ohm"]==result["analyses"][0]["Z_ohm"]
    report.pop("comparison")
    with pytest.raises(ValueError,match="Incomplete v2"):fem.analyze_baseline(report)


def test_q1024_only_is_explicit_and_not_mislabelled_as_q4096():
    report=baseline(1);report["answers"]=[a for a in report["answers"] if a["rule"]==1024]
    result=fem.analyze_baseline(report)
    assert all(a["rule"]==1024 for a in result["analyses"])
    assert not result["numerical_controls_complete"]
    assert not any(a["finite_difference_control_applies_to_at_least_one_case"] for a in result["analyses"])


def test_changed_bone_or_material_order_is_refused():
    report=baseline(1)
    for a in report["answers"]:
        if a["state"]=="exhale":a["rho_ohm_m"][2]=31
    with pytest.raises(ValueError,match="Bone"):fem.analyze_baseline(report)
    report=baseline(1);report["identity"]["material_order"]=["soft_background","lung","heart","bone","fat"]
    with pytest.raises(ValueError,match="material ordering"):fem.analyze_baseline(report)


def test_cli_uses_only_given_subject_and_fresh_outputs(tmp_path):
    model=tmp_path/"nik";(model/"electrical").mkdir(parents=True)
    path=model/"electrical/forward.local.json"
    path.write_text(json.dumps(baseline()),encoding="utf-8")
    output=tmp_path/"report"
    args=["--model-root",str(model),"--output-dir",str(output),"--subject","nik"]
    fem.main(args)
    saved=json.loads((output/"sensitivity.local.json").read_text(encoding="utf-8"))
    assert saved["source"]["sha256"]==fem.sha256(path)
    assert not saved["placement_test_complete"]
    with pytest.raises(ValueError,match="fresh"):fem.main(args)
    with pytest.raises(ValueError,match="Subject"):fem.main(args[:-1]+["georg"])


def test_placement_deltas_are_signed_and_scenarios_not_statistical_errors():
    source=baseline(1)["answers"][0]
    shifted=deepcopy(source);shifted["scenario_id"]="Y_plus5mm"
    shifted["placement"]=dict(requested_offset_mm=5,requested_axis_LPS=[0,1,0])
    shifted["transfer_matrix_ohm"][3][2]-=2
    shifted["jacobian_logrho"][3][2][1]-=3
    result=fem.summarize_placement(dict(schema="scorpion_contact_sensitivity_v1",FEM_was_run=True,
        subject_alias="nik",case_id="L140_TT2x2",baseline=source,answers=[shifted],frequency_hz=100000,
        active_material_indices=[1,2,4,5],source_rule="own_drive",source_relative_phase="unknown",
        CT_mesh_materials_unchanged=True))
    lateral=result["scenarios"][0]["roles"]["lateral"]
    assert lateral["signed_delta_Z_ohm"]==-2
    assert lateral["signed_delta_jacobian_logrho"][1]==pytest.approx(-3)
    assert not result["scenario_offsets_are_confidence_intervals"]
    assert not result["both_requested_scenarios_complete"]
