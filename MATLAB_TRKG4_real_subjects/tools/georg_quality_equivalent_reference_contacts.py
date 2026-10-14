"""Contact comparison with strict, explicit physical-reference equivalence.

The original comparator is retained unchanged. This variant differs only in a
reference gate: different hashes are allowed solely for proven provenance-only
JSON changes. All actual reference hashes and source identities are retained.
No FEM values, identities, inputs or metadata are rewritten.
"""
from pathlib import Path
from georg_quality_contact_variant import (
    ROOT, FILES, POINTS, read, sha, contact_difference, verify_reused_ancestry)


def verify_reference_equivalence(rows, files):
    if len(rows)!=2 or len(files)!=2:raise ValueError('Two actual references are mandatory')
    references=[]
    for row,file in zip(rows,files,strict=True):
        if sha(file)!=row['identity']['reference_sha256']:raise ValueError('Actual reference hash differs')
        ref=read(file)
        if ref.get('parameters')!=row['parameters'] or ref.get('pose')!=row['pose']:
            raise ValueError('Reference physical point differs from its actual forward')
        if ref.get('observations_sha256')!=row['identity']['observations_sha256']:
            raise ValueError('Reference observations differ from their forward')
        references.append(ref)
    allowed={'selected_candidate_sha256','common_parameters_not_fitted_here','optimized_on_fine_remapped_mesh'}
    physical=[{k:v for k,v in r.items() if k not in allowed} for r in references]
    if physical[0]!=physical[1]:raise ValueError('References differ outside allowed provenance-only fields')
    changed=sorted(k for k in allowed if references[0].get(k)!=references[1].get(k))
    return dict(schema='exact_physical_reference_equivalence_v1',
        reference_sha256=[sha(p) for p in files],changed_provenance_fields=changed,
        physical_reference_contents_equal_exactly=True,reference_hashes_not_rewritten=True)


def compare_equivalent_reference_controls(base, variant, inherited_forward, direct_forward, reference_files, hdf5_library=None):
    """Read-only contact comparison of two qualified saved fixed-point forwards.

    Verify parent-prepared ancestry from the actual MAT identities. No material
    matrices are loaded. An unavailable metadata reader fails closed.
    """
    from georg_quality_comparison import verify_saved_identity
    from georg_quality_general_audit import verify_contact_identity
    base, variant = Path(base).resolve(), Path(variant).resolve()
    files = [Path(inherited_forward).resolve(), Path(direct_forward).resolve()]
    rows = [read(p) for p in files]
    facts = read(variant/'contact_variant.local.json')
    if facts['base_directory'] != str(base) or facts['base_plan_sha256'] != sha(base/'plan.local.json'):
        raise ValueError('Direct variant belongs to another base')
    if facts['variant_plan_sha256'] != sha(variant/'plan.local.json'):
        raise ValueError('Direct variant plan changed')
    for source in facts['sources']:
        if sha(source['path']) != source['sha256']:
            raise ValueError('Contact variant dependency changed')
    verified = [verify_saved_identity(row,directory,ROOT/'src/run_georg_quality_forward.m')
                for row,directory in zip(rows,(base,variant))]
    plans = [p[0] for p in verified]
    if Path(plans[1]['parent_directory']).resolve() != base or plans[1]['boundary_parent_map']:
        raise ValueError('Expected a DIRECT variant linked to the same bulk parent')
    for name in FILES:
        if sha(base/name) != sha(variant/name) or sha(base/name) != facts['bulk_file_sha256'][name]:
            raise ValueError('Bulk changed during contact comparison: '+name)
    for row,plan,directory in zip(rows,plans,(base,variant)):
        verify_contact_identity(row,plan,directory)
        if plan['numerical_quadrature_points'] != list(POINTS):
            raise ValueError('Contact comparison uses different integration rules')
        if not row['patch_qc']['passed'] or not row['system_checks_passed']:
            raise ValueError('An inadmissible contact/linear-system result is not a valid comparison')
    for key in ('parameters','pose','record_ids','axes','states','target_Z_ohm','source_rule'):
        if rows[0][key] != rows[1][key]:
            raise ValueError('Fixed contact-comparison input changed: '+key)
    reference_receipt=verify_reference_equivalence(rows, reference_files)
    for key in ('observations_sha256','physical_contacts_sha256','physical_montage_helper_sha256'):
        if rows[0]['identity'][key] != rows[1]['identity'][key]:
            raise ValueError('Contact-comparison source contract changed: '+key)
    if rows[0]['identity']['physical_contacts_sha256'] != facts['physical_contact_inventory_sha256']:
        raise ValueError('Executed physical inventory differs from the variant contract')
    if rows[0]['identity']['physical_montage_helper_sha256'] != sha(ROOT/'src/trkg4_georg_physical_montages.m'):
        raise ValueError('Executed physical grouping helper changed')
    for a,b in zip(rows[0]['patches'],rows[1]['patches'],strict=True):
        for key in ('record_id','xyz','gids','electrical_groups','physical_patches'):
            if a[key] != b[key]:
                raise ValueError('Physical centres or electrical groups changed: '+key)
    result = contact_difference(rows[0]['fine_Z_ohm'],rows[1]['fine_Z_ohm'])
    if len(result['difference_ohm']) != len(rows[0]['record_ids']):
        raise ValueError('Contact differences do not address the saved observations')
    ancestry=verify_reused_ancestry(base,variant,verified[0][1],verified[1][1],hdf5_library)
    contact_passed=result['passed'];quadrature_passed=all(r['quadrature_passed'] for r in rows)
    result.update(schema='georg_fixed_bulk_contact_difference_v1',compared_rule_points=4096,
        record_ids=rows[0]['record_ids'],axes=rows[0]['axes'],states=rows[0]['states'],
        inherited_Z_ohm=rows[0]['fine_Z_ohm'], direct_Z_ohm=rows[1]['fine_Z_ohm'],
        parameters=rows[0]['parameters'],pose=rows[0]['pose'],
        difference_direction='DIRECT minus inherited contacts; same bulk/parameters/pose',
        tolerance_formula='0.02+0.005*abs(Z_DIRECT)', tolerance_is_diagnostic_not_physical_bound=True,
        bulk_identical_verified=True, **ancestry,
        expected_direct_parent_prepared_sha256=rows[0]['identity']['prepared_sha256'],
        all_physical_patch_types_reselected=facts['all_physical_patch_types_reselected'],
        contact_difference_passed=contact_passed,quadrature_checks_passed=quadrature_passed,
        passed=contact_passed and quadrature_passed,
        sources=[dict(path=str(p),sha256=sha(p)) for p in
                 [*files,base/'plan.local.json',variant/'plan.local.json',variant/'contact_variant.local.json',
                  *[v[1] for v in verified],ROOT/'src/trkg4_georg_contact_inheritance.m',
                  ROOT/'src/run_georg_quality_forward.m',ROOT/'src/trkg4_georg_physical_montages.m',
                  ROOT/'src/trkg4_georg_lateral_materials.m',ROOT/'src/trkg4_georg_material_rules.m',
                  ROOT/'src/trkg4_scale_fmdl_to_si.m',ROOT/'tools/georg_quality_comparison.py',
                  ROOT/'tools/georg_quality_general_audit.py',Path(__file__)]],
        quantitative_result_accepted=False,global_mesh_convergence_proven=False,
        final_joint_or_selected_validation_complete=False)
    result['reference_equivalence']=reference_receipt
    result['sources'] += [dict(path=str(Path(p).resolve()),sha256=sha(p)) for p in reference_files]
    return result

