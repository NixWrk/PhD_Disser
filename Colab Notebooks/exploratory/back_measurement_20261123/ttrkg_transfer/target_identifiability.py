"""Target-specific identifiability of tissue contribution to the arm lead.

Run: python -B target_identifiability.py
Inputs: ../small_arrays/tissue_fields.csv (primary, gabriel100_proxy, 100 kHz)
and arm_fields.csv. Output: target_identifiability.csv ONLY, next to this file.
No waveform fitting, new FEM, physiological pulse recovery, AI, motion, cardiac
volume or stroke-volume calculation is performed. Requires NumPy.

x=(delta ln rho_soft, delta ln rho_fat, delta ln rho_lung). The soft compartment
excludes the separately modelled fat; lungs are one electrical compartment.
Bone is fixed. Signed J=dZ/d ln rho is in ohms. Target t=b*x is the surrounding
tissue contribution to the arm lead, EXCLUDING the heart term. The primary
scenario holds effective heart rho fixed and assumes ||x||_2 <= epsilon=.001.
The same compartment changes act in both leads: in particular soft rho also
changes the assumed homogeneous arm extensions. Separate local arm changes
are not identifiable from this three-compartment parameterisation or bounded
by the present calculation; spatial homogeneity remains an extra assumption.
This equal-weight Euclidean ball is a stipulated scenario, not a physiological
distribution, empirical uncertainty, per-tissue maximum, or noise model.

For posterior rows A and noiseless measurement y=A*x:
  P=A^+ A, r=(I-P)b, w=(A^T)^+ b;
  b*x = w*y + r*x;
  max_{A*x=0, ||x||2<=epsilon} |b*x| = epsilon*||r||2;
  x_star=epsilon*r/||r||2, absent measurement max=epsilon*||b||2.
Two opposite admissible perturbations have the same y=0 and differ in target
by twice the reported one-sided maximum. The target alone is identifiable iff
r=0; rank deficiency of the individual parameters is not the target criterion.

Heart sensitivity c is retained separately. The second scenario uses
z=(x,h), h=delta ln rho_effective_heart, ||z||2<=epsilon, A4=[A,c], b4=[b,0].
The SAME norm radius is used, not independent .001 bounds for each tissue.
The reported posterior heart probe is c*.001; arm heart probe is J_arm_heart*.001.
If one applied the heart-fixed transfer w to y=A*x+c*h, its spurious target
term would be (w*c)*h. We report that signed term at h=.001. No motion or volume
effect is included. Without ANY bound on h, nonzero c removes the single-row
constraint on x algebraically; this statement is not extrapolated to finite
physiology outside the local linear model.

Checks: null-space/SVD optimum, explicit maximiser, row-space identity,
Pythagorean identity, deterministic random probes, and three analytic edge
cases (parallel/perpendicular/fully observed targets). All directions, signed
coefficients, numerical residuals and source hashes are retained in the CSV.
Optional two-row designs are hypothetical common-state designs; their pulses
are not asserted simultaneous with TRKG. No noise or mesh uncertainty is fitted.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
POSTERIOR = HERE.parent/'small_arrays/tissue_fields.csv'
ARM = HERE/'arm_fields.csv'
TARGET = ('soft_background', 'fat', 'lung')
ALL_TISSUES = TARGET + ('heart', 'bone')
EPSILON = .001
SCENARIO = 'gabriel100_proxy'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def projection_bound(a, b):
    a, b = np.atleast_2d(np.asarray(a, float)), np.asarray(b, float)
    _, singular, vt = np.linalg.svd(a, full_matrices=True)
    tol = np.finfo(float).eps*max(a.shape)*singular[0]
    rank = int(np.sum(singular > tol))
    null_basis = vt[rank:].T
    hidden = null_basis @ (null_basis.T @ b)
    w = np.linalg.lstsq(a.T, b, rcond=None)[0]
    observed = a.T @ w
    hidden_norm = np.linalg.norm(hidden)
    identifiable = hidden_norm <= 1e-12*max(1., np.linalg.norm(b))
    direction = np.zeros_like(b) if identifiable else hidden/hidden_norm
    delta = EPSILON*direction
    maximum = EPSILON*hidden_norm
    # Independent projection expression and actual attainable witness.
    p = np.linalg.pinv(a) @ a
    np.testing.assert_allclose(hidden, (np.eye(len(b))-p) @ b, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(a @ delta, 0, atol=1e-12)
    np.testing.assert_allclose(b @ delta, maximum, atol=1e-12)
    assert np.linalg.norm(delta) <= EPSILON*(1+1e-12)
    assert maximum <= EPSILON*np.linalg.norm(b)*(1+1e-12)
    rng = np.random.default_rng(20261123)
    probes = rng.normal(size=(128, len(b)))
    identity_error = np.max(np.abs(probes @ b - (probes @ a.T) @ w - probes @ hidden))
    pythagorean_error = abs(b @ b - observed @ observed - hidden @ hidden)
    assert identity_error < 1e-10*max(1., np.linalg.norm(b))
    assert pythagorean_error < 1e-10*max(1., b @ b)
    if null_basis.shape[1]:
        null_probes = (probes @ null_basis) @ null_basis.T
        null_probes *= EPSILON / np.maximum(np.linalg.norm(null_probes, axis=1), 1e-30)[:, None]
        assert np.max(np.abs(null_probes @ b)) <= maximum+1e-12
    return dict(rank=rank, nullity=len(b)-rank, w=w, direction=direction, delta=delta,
                residual=hidden, maximum=maximum,
                unconstrained=EPSILON*np.linalg.norm(b), identifiable=identifiable,
                null_response_error=np.max(np.abs(a @ delta)),
                identity_error=identity_error, pythagorean_error=pythagorean_error,
                optimum_error=abs(b @ delta-maximum),
                smallest_nonzero_singular=singular[rank-1])


def selfcheck():
    aligned = projection_bound([[1, 0, 0]], [2, 0, 0])
    perpendicular = projection_bound([[1, 0, 0]], [0, 2, 0])
    fully = projection_bound(np.eye(3), [1, 2, 3])
    assert aligned['identifiable'] and fully['identifiable']
    np.testing.assert_allclose(perpendicular['maximum'], .002)
    heart = projection_bound([[1, 0, 0, 1]], [2, 0, 0, 0])
    np.testing.assert_allclose(heart['maximum'], .001*np.sqrt(2))


def vector(rows, names=TARGET):
    by_tissue = {r['tissue']: r for r in rows}
    if len(by_tissue) != len(rows) or set(by_tissue) != set(ALL_TISSUES):
        raise ValueError('Each field group must contain exactly the same five tissues')
    values = np.array([float(by_tissue[t]['J_logrho_ohm']) for t in names])
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite sensitivity')
    return values


def packed(values):
    return ';'.join(format(float(v), '.15g') for v in np.atleast_1d(values))


def result(participant, posterior_groups, arm_rows, design, arm_label):
    b = vector(arm_rows)
    arm_by_tissue = {r['tissue']: r for r in arm_rows}
    a = np.stack([vector(group) for group in posterior_groups])
    c = np.array([vector(group, ('heart',))[0] for group in posterior_groups])
    for group in posterior_groups:
        for r in group:
            ar = arm_by_tissue[r['tissue']]
            np.testing.assert_allclose(float(r['rho_ohm_m']), float(ar['rho_ohm_m']), rtol=1e-11)
            for field in ('frequency_hz', 'scenario'):
                if r[field] != ar[field]:
                    raise ValueError(f'Incompatible {field}')
            for field in ('registry_sha256', 'kernel_sha256'):
                if r[field] != ar[field]:
                    raise ValueError(f'Incompatible {field}')
            if r['size_mm']!='200' and r['mesh_sha256']!=ar['mesh_sha256']:
                raise ValueError('Unexpected mesh mismatch for posterior 60/100/140')
    fixed = projection_bound(a, b)
    joint = projection_bound(np.column_stack([a, c]), np.r_[b, 0.])
    assert joint['maximum']+1e-12 >= fixed['maximum']
    max0 = fixed['unconstrained']*1000
    row = dict(participant=participant, scenario=SCENARIO, frequency_hz=100000,
        arm_field=arm_label, design=design,
        posterior_rows=';'.join(f"{g[0]['placement']}:{g[0]['size_mm']}" for g in posterior_groups),
        posterior_sizes_mm=';'.join(g[0]['size_mm'] for g in posterior_groups),
        n_posterior=len(a), norm='unweighted_L2_delta_ln_rho', norm_radius=EPSILON,
        target='arm_soft_plus_fat_plus_single_lung_excluding_heart',
        bone_delta_ln_rho=0.,
        posterior_mesh_families=';'.join(g[0]['mesh_family'] for g in posterior_groups),
        posterior_mesh_sha256=';'.join(g[0]['mesh_sha256'] for g in posterior_groups),
        arm_mesh_sha256=arm_rows[0]['mesh_sha256'],
        arm_result_sha256=arm_rows[0]['result_sha256'],
        arm_run_plan_sha256=arm_rows[0]['run_plan_sha256'],
        posterior_result_sha256=';'.join(g[0]['result_sha256'] for g in posterior_groups),
        registry_sha256=arm_rows[0]['registry_sha256'],
        kernel_sha256=arm_rows[0]['kernel_sha256'],
        posterior_arm_same_mesh=all(g[0]['mesh_sha256']==arm_rows[0]['mesh_sha256'] for g in posterior_groups),
        arm_scenario_status=arm_rows[0]['scenario_status'],
        posterior_scenario_status=posterior_groups[0][0]['scenario_status'],
        arm_z_contact_ohm_m2=float(arm_rows[0]['z_contact_ohm_m2']),
        posterior_z_contact_ohm_m2=';'.join(g[0]['z_contact_ohm_m2'] for g in posterior_groups),
        same_contact_impedance=all(np.isclose(float(g[0]['z_contact_ohm_m2']),
            float(arm_rows[0]['z_contact_ohm_m2']), rtol=1e-10, atol=0) for g in posterior_groups),
        soft_region_includes_assumed_homogeneous_arm_extensions=arm_rows[0]['soft_region_includes_arms'],
        arm_regional_sensitivity_available=arm_rows[0]['matching_regional_arm_J_available'],
        arm_polarity_to_BEI2_verified=arm_rows[0]['polarity_to_BEI2_verified'],
        shared_global_tissue_changes_assumed=True,
        posterior_to_arms_spatial_homogeneity_validated=False,
        independent_local_arm_tissue_changes_included=False,
        target_identifiable_heart_fixed=fixed['identifiable'],
        rank_heart_fixed=fixed['rank'], nullity_heart_fixed=fixed['nullity'],
        max_without_posterior_mohm=max0,
        max_hidden_heart_fixed_mohm=1000*fixed['maximum'],
        hidden_fraction_heart_fixed=fixed['maximum']/fixed['unconstrained'],
        bound_reduction_percent=100*(1-fixed['maximum']/fixed['unconstrained']),
        indistinguishable_pair_target_difference_mohm=2000*fixed['maximum'],
        posterior_to_target_weights=packed(fixed['w']),
        posterior_measurement_error_operator_L2_norm=float(np.linalg.norm(fixed['w'])),
        posterior_smallest_nonzero_singular_ohm=fixed['smallest_nonzero_singular'],
        exact_noiseless_posterior_constraint=True,
        heart_probe_delta_ln_rho=EPSILON,
        heart_probe_posterior_mohm=packed(1000*EPSILON*c),
        heart_probe_arm_mohm=1000*EPSILON*float(arm_by_tissue['heart']['J_logrho_ohm']),
        heart_fixed_transfer_spurious_target_mohm=1000*EPSILON*(fixed['w'] @ c),
        heart_joint_norm=EPSILON, rank_heart_joint=joint['rank'], nullity_heart_joint=joint['nullity'],
        max_hidden_heart_joint_mohm=1000*joint['maximum'],
        hidden_fraction_heart_joint=joint['maximum']/joint['unconstrained'],
        added_hidden_bound_due_to_heart_mohm=1000*(joint['maximum']-fixed['maximum']),
        posterior_to_target_weights_heart_joint=packed(joint['w']),
        heart_unbounded_single_posterior_algebraically_no_constraint=bool(len(a)==1 and c[0]!=0),
        numerical_null_response_error_mohm=1000*fixed['null_response_error'],
        numerical_identity_error_ohm=fixed['identity_error'],
        numerical_pythagorean_error_ohm2=fixed['pythagorean_error'],
        numerical_optimum_error_mohm=1000*fixed['optimum_error'],
        numerical_heart_joint_null_error_mohm=1000*joint['null_response_error'],
        numerical_heart_joint_identity_error_ohm=joint['identity_error'],
        numerically_checked=True,
        placement_status='scenario_not_verified_19November_skin_contacts',
        time_basis='single_lead_fixed_CT_model' if len(a)==1 else 'calculated_design_hypothetical_common_state',
        actual_19Nov_simultaneous_right_arm_record_exists=bool(len(a)==1),
        posterior_arm_joint_model_validated_to_recording=False,
        morphology_motion_or_volume_included=False,
        local_linear_scenario_not_accuracy_or_physiological_bound=True,
        individual_rho_measured=False, complex_measurement_mapping_validated=False,
        global_mesh_convergence_proven=False)
    for j, name in enumerate(TARGET):
        row[f'arm_J_{name}_ohm'] = b[j]
        row[f'posterior_J_{name}_ohm'] = packed(a[:, j])
        row[f'unit_null_direction_{name}'] = fixed['direction'][j]
        row[f'maximiser_delta_ln_rho_{name}'] = fixed['delta'][j]
        row[f'heart_joint_maximiser_delta_ln_rho_{name}'] = joint['delta'][j]
    row['heart_joint_maximiser_delta_ln_rho_heart'] = joint['delta'][3]
    return row


def main():
    selfcheck()
    posterior_hash, arm_hash = sha(POSTERIOR), sha(ARM)
    posterior = [r for r in read(POSTERIOR) if r['scenario']==SCENARIO
                 and r['frequency_hz']=='100000' and r['role']=='primary']
    arms = [r for r in read(ARM) if r['scenario']==SCENARIO and r['frequency_hz']=='100000'
            and r['role']=='primary' and r['posture']=='fixed_ct'
            and r['placement']=='arm_arm_top_1x1_step100']
    rows = []
    for participant in ('U1', 'U2'):
        selected = [r for r in arms if r['participant']==participant]
        # A single agreed arm placement/material base is required; duplicate
        # tissue rows fail rather than silently choosing a model after results.
        vector(selected)
        groups = {}
        for r in posterior:
            if r['participant']==participant:
                groups.setdefault((r['placement'], int(r['size_mm'])), []).append(r)
        for size in (60, 100, 140, 200):
            rows.append(result(participant, [groups['back_right', size]], selected,
                               'single_right_posterior', selected[0].get('placement', 'arm')))
        for name, keys in [('right_100_plus_200_design', [('back_right', 100), ('back_right', 200)]),
                           ('bilateral_100_design', [('back_right', 100), ('back_left', 100)])]:
            rows.append(result(participant, [groups[k] for k in keys], selected, name,
                               selected[0].get('placement', 'arm')))
    for row in rows:
        row.update(code_sha256=sha(Path(__file__)), posterior_csv_sha256=posterior_hash,
                   arm_csv_sha256=arm_hash, numpy_version=np.__version__)
    if sha(POSTERIOR)!=posterior_hash or sha(ARM)!=arm_hash:
        raise RuntimeError('Source CSV changed during calculation; rerun on the completed export')
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (HERE/'target_identifiability.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(f'{len(rows)} designs; analytic and numerical identities passed')


if __name__ == '__main__':
    main()
