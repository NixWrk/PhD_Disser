"""Verify a fixed replay before rebuilding a mesh around a previous-model fit.

This does not relabel the previous fit as a fit of its coarser material parent.
The replay must preserve parameters, coordinates, observations and contacts.
"""
from pathlib import Path
from georg_quality_comparison import read, sha, verify_saved_identity


def verify_parent_replay(candidate, parent_dir, template, lineages, forward_file):
    parent_dir, forward_file = Path(parent_dir), Path(forward_file)
    row = read(forward_file)
    verify_saved_identity(row, parent_dir,
        Path(__file__).resolve().parents[1]/'src/run_georg_quality_forward.m')
    if row.get('parameters_optimized') is not False:
        raise ValueError('Require a fixed-parameter parent replay, not a new optimization')
    for field in ('parameters', 'pose'):
        if row[field] != candidate[field]:
            raise ValueError('Parent replay changes selected '+field)
    identity = candidate['identity']
    if (row['identity']['observations_sha256'] != identity['observations_sha256'] or
        row['identity']['physical_contacts_sha256'] != identity['contact_file_sha256'] or
        identity['pose_reference_mesh_sha256'] != lineages['pose_sha256'] or
        sha(template['contacts_file']) != identity['contact_file_sha256']):
        raise ValueError('Parent replay changes observations, contact inventory or pose reference')
    if not (row['quadrature_passed'] and row['system_checks_passed'] and row['patch_qc']['passed']):
        raise ValueError('Fixed parent replay numerical prerequisites failed')
    if len(row['derivatives']) != 2 or not all(r['passed'] for r in row['derivatives']):
        raise ValueError('Both cross derivatives are required in the parent replay')
    return dict(candidate_role='previous_model_fit_with_verified_fixed_parent_replay',
        fitted_in_current_parent_model=False,
        source_model_plan_sha256=identity['model_plan_sha256'],
        parent_replay_file=str(forward_file.resolve()), parent_replay_sha256=sha(forward_file),
        additional_sources=[forward_file])
