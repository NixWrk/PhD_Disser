"""Prepare a geometry-only shared contract; never run FEM or legacy workers.

Paths in CLI arguments may be absolute; relative paths in input JSON are relative
to --root (the MATLAB project). --source-config must explicitly identify the
current segmentation_directory, e.g. the selected refinement preparation.json.
--local-check files retain their actual outcomes, including negative outcomes.

Selection is a separate operation, requiring a hash-bound guard-fix receipt:
schema=georg_shared_selection_guards_v1; candidate_sha256; boolean_guards_passed
and legacy_routing_blocked (literal true); tests and sources=[{path,sha256}]
(both nonempty; sources identify the tested resolver and routing versions).
The receipt documents externally executed guard tests, not tests run by this
helper. Do not create a receipt until the resolver/routing fixes are tested.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import tempfile


FALSE_FLAGS = ('mesh_convergence_passed', 'integration_comparison_passed',
    'quantitative_inversion_allowed', 'quantitative_result_accepted',
    'global_mesh_convergence_proven', 'independent_anatomical_validation',
    'physiological_validation', 'placement_accuracy_is_known')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def new_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def located(root, path):
    p = Path(path)
    return (p if p.is_absolute() else Path(root)/p).resolve()


def named(root, path):
    p = Path(path).resolve()
    try:
        return p.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(p)  # Local contract only; no medical paths in tracked code.


def prepare(root, bundle, fem, source_config, out, local_checks=()):
    root = Path(root).resolve()
    bundle, fem, source_config, out = [located(root, p) for p in
                                      (bundle, fem, source_config, out)]
    if out.exists():
        raise ValueError('Candidate output already exists')
    files = {}

    def pin(path, expected=None):
        p = located(root, path)
        actual = sha(p)
        if expected is not None and actual != expected:
            raise ValueError('SHA mismatch: '+str(p))
        if p in files and files[p] != actual:
            raise ValueError('Source changed during preparation')
        files[p] = actual
        return actual

    def doc(path):
        pin(path)
        return read(path)

    def sources(rows):
        for row in rows:
            pin(row['path'], row['sha256'])

    manifest = doc(bundle/'geometry_bundle.local.json')
    if manifest['schema'] != 'georg_quality_geometry_bundle_v1':
        raise ValueError('Unsupported geometry bundle')
    if manifest['status'] != 'prepared_not_selected':
        raise ValueError('Expected immutable prepared bundle')
    for key in ('byte_copies_verified', 'raw_normalised_centres_equal',
                'source_contexts_preserved', 'all_24_transfer_records_verified',
                'source_hashes_rechecked'):
        if manifest['checks'].get(key) is not True:
            raise ValueError('Incomplete landmark provenance: '+key)
    sources(manifest['sources'])
    for row in manifest['copies']+manifest['products']:
        pin(bundle/row['path'], row['sha256'])
    roles = manifest['geometry_roles']
    mapped_roles = {}
    for name, role in roles.items():
        role = dict(role)
        if role.get('path'):
            pin(bundle/role['path'], role['sha256'])
            role['path'] = named(root, bundle/role['path'])
        if role.get('passport_path'):
            pin(bundle/role['passport_path'], role['passport_sha256'])
            role['passport_path'] = named(root, bundle/role['passport_path'])
        mapped_roles[name] = role
    boundary = mapped_roles['FEM_boundary_reference']
    if not (boundary.get('supplied') is True and
            boundary.get('passport_surface_bytes_bound') is True):
        raise ValueError('Actual FEM surface bytes and passport are required')
    surface = doc(located(root, boundary['passport_path']))
    if (surface['passed'] is not True or
        surface['candidate_surface_sha256'] != boundary['sha256'] or
        surface['source_surface_sha256'] != roles['body_union_picker_surface']['sha256']):
        raise ValueError('FEM surface provenance/QC mismatch')
    geometry = doc(bundle/'geometry.local.json')
    if (geometry['watertight'] is not True or geometry['positive_volume'] is not True or
        geometry['connected_components'] != 1 or geometry['self_intersecting_faces'] != 0 or
        geometry['surface_sha256'] != roles['body_union_picker_surface']['sha256']):
        raise ValueError('Geometry QC failed')
    if not geometry['joins'] or any(j['cross_section_covered_fraction'] < 1-1e-9
                                   for j in geometry['joins'].values()):
        raise ValueError('Incomplete arm section')
    config = doc(source_config)
    segmentation = located(root, config['segmentation_directory'])
    pin(segmentation/'static_ct.nii.gz', manifest['ct_sha256'])
    contour = doc(bundle/'contour_qc.local.json')
    if contour['ct_sha256'] != manifest['ct_sha256']:
        raise ValueError('CT identity differs')
    plan = doc(fem/'plan.local.json')
    sources(plan['sources'])
    # Bind current masks to the executed material plan, not just to a directory.
    expected = {located(root, r['path']): r['sha256'] for r in plan['sources']}
    for name in ('lung', 'bone', 'heart', 'fat'):
        p = segmentation/(name+'.nii.gz')
        if p not in expected:
            raise ValueError('Selected segmentation is absent from material plan: '+name)
        pin(p, expected[p])
    original_geometry = located(root, plan['geometry_directory'])
    for name in ('body_ct_native.stl', 'body_ct_surface.stl', 'body_embedded_arms_370mm.stl'):
        pin(original_geometry/name, sha(bundle/name))
    qc = doc(fem/'mesh_quality.local.json')
    if qc['passed'] is not True or qc['all_positive'] is not True or plan['shape_QC_passed'] is not True:
        raise ValueError('Mesh QC failed')
    pin(fem/'mesh.mat', qc['mesh_sha256'])
    for key, qkey in [('pose_reference_mesh', 'pose_reference_mesh_sha256'),
                     ('exact_contact_parent_mesh', 'contact_master_mesh_sha256'),
                     ('boundary_parent_map', 'contact_parent_maps_sha256'),
                     ('geometry_boundary_parent_map', 'geometry_parent_maps_sha256')]:
        pin(plan[key], qc[qkey])
    for n in plan['numerical_quadrature_points']:
        p = fem/f'fractions_q{n}.mat'
        if p not in expected:
            raise ValueError('Fraction hash missing from material plan')
        pin(p, expected[p])
    pin(fem/'reference_materials.local.json')
    checks = []
    for file in local_checks:
        p = located(root, file)
        d = doc(p)
        if not d.get('scope') or not d.get('sources'):
            raise ValueError('Local check needs explicit scope and hash-bound sources')
        sources(d['sources'])
        if d.get('model_plan_sha256', files[fem/'plan.local.json']) != files[fem/'plan.local.json']:
            raise ValueError('Local check belongs to another material plan')
        checks.append(dict(path=named(root, p), sha256=files[p], scope=d['scope'],
                           outcome=d.get('all_checks_passed', d.get('passed')),
                           outcome_is_global=False))
        if type(checks[-1]['outcome']) is not bool:
            raise ValueError('Local outcome must be explicitly true or false')
    pointer = root/'output/georg_current_model.local.json'
    previous = sha(pointer)  # Require an existing pointer; never silently replace unknown state.
    result = dict(schema='georg_current_model_v1', status='geometry_reference_checked',
        subject_alias='georg', candidate_only=True, selection_state='prepared_not_selected',
        model_directory=named(root, bundle), fem_directory=named(root, fem),
        segmentation_directory=named(root, segmentation), geometry_roles=mapped_roles,
        local_numerical_checks=checks, geometry_only_selection=True,
        strict=False, previous_pointer_sha256=previous,
        source_config=dict(path=named(root, source_config), sha256=files[source_config]),
        mesh_surface_relation='Recorded nested hierarchy; hashes checked, boundary geometry not recomputed here',
        requires_guard_fixes_before_selection=True,
        legacy_worker_compatible=False, model_scope='Geometry reference; local FEM checks only',
        **{key: False for key in FALSE_FLAGS})
    result['files'] = [dict(path=named(root, p), sha256=h) for p,h in sorted(files.items())]
    # Recheck on read-only inputs before creating the one new contract.
    for p,h in files.items():
        if sha(p) != h:
            raise ValueError('Input changed during preparation: '+str(p))
    new_json(out, result)
    return result


def select(root, candidate, guards):
    """Explicitly select only after an independently executed guard-test receipt."""
    root = Path(root).resolve()
    candidate, guards = located(root, candidate), located(root, guards)
    c, g = read(candidate), read(guards)
    if (c.get('schema') != 'georg_current_model_v1' or
        c['status'] != 'geometry_reference_checked' or c['candidate_only'] is not True or
        c['selection_state'] != 'prepared_not_selected' or c.get('strict') is not False or
        any(c.get(k) is not False for k in FALSE_FLAGS)):
        raise ValueError('Only geometry-only candidates may be selected')
    if (g.get('schema') != 'georg_shared_selection_guards_v1' or
        g.get('candidate_sha256') != sha(candidate) or
        g.get('boolean_guards_passed') is not True or
        g.get('legacy_routing_blocked') is not True or not g.get('tests') or not g.get('sources')):
        raise ValueError('Missing candidate-bound guard-fix receipt')
    for row in c['files']+g['tests']+g['sources']:
        if sha(located(root, row['path'])) != row['sha256']:
            raise ValueError('Pinned dependency changed: '+row['path'])
    pointer = root/'output/georg_current_model.local.json'
    old = pointer.read_bytes()
    if hashlib.sha256(old).hexdigest() != c['previous_pointer_sha256']:
        raise ValueError('Common pointer changed since preparation; prepare again')
    backup = pointer.with_name(pointer.name+'.before-'+c['previous_pointer_sha256']+'.bak')
    if backup.exists():
        if backup.read_bytes() != old:
            raise ValueError('Backup conflict')
    else:
        with backup.open('xb') as f:
            f.write(old)
    selected = dict(c, candidate_only=False, selection_state='selected_geometry_only',
                    selection_guard_receipt=dict(path=named(root, guards), sha256=sha(guards)))
    selected['files'] = c['files']+g['tests']+g['sources']+[
        dict(path=named(root, guards), sha256=sha(guards))]
    fd, tmp = tempfile.mkstemp(prefix='.georg-pointer-', dir=pointer.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(selected, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        if pointer.read_bytes() != old:
            raise ValueError('Concurrent pointer update; selection refused')
        os.replace(tmp, pointer)
    finally:
        if Path(tmp).exists():
            Path(tmp).unlink()
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--select', type=Path, metavar='CANDIDATE')
    p.add_argument('--guards', type=Path)
    for flag in ('bundle', 'fem', 'source-config', 'out'):
        p.add_argument('--'+flag, type=Path)
    p.add_argument('--local-check', action='append', default=[], type=Path)
    a = p.parse_args()
    if a.select:
        if not a.guards:
            p.error('--select requires --guards')
        select(a.root, a.select, a.guards)
    else:
        if not all((a.bundle, a.fem, a.source_config, a.out)):
            p.error('prepare requires --bundle --fem --source-config --out')
        prepare(a.root, a.bundle, a.fem, a.source_config, a.out, a.local_check)


if __name__ == '__main__':
    main()
