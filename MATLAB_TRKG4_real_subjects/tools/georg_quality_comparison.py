"""Compare saved, fixed-pose FEM controls without solving or fitting.

Space: coarse -> refined with exact coefficient/contact inheritance.
Fractions: inherited -> resampled on the same refined mesh.
The total difference is also checked; cancellation cannot hide either effect.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def difference(left, right, relative=.005):
    left, right = np.asarray(left), np.asarray(right)
    delta = np.abs(right-left)
    tolerance = .02+relative*np.abs(right)
    return dict(difference_ohm=delta.tolist(), tolerance_ohm=tolerance.tolist(),
                signed_difference_ohm=(right-left).tolist(),
                failed_values=int(np.count_nonzero(delta>tolerance)),
                maximum_ratio=float(np.max(delta/tolerance)),
                passed=bool(np.all(delta<=tolerance)))


def compare(coarse, inherited, remapped):
    for other in (inherited, remapped):
        for field in ('record_ids', 'axes', 'states', 'parameters', 'pose', 'target_Z_ohm'):
            if coarse[field] != other[field]:
                raise ValueError('Fixed comparison input changed: '+field)
        for field in ('physical_contacts_sha256', 'reference_sha256',
                      'observations_sha256', 'physical_montage_helper_sha256'):
            if coarse['identity'][field] != other['identity'][field]:
                raise ValueError('Physical comparison contract changed: '+field)
        for parent, child in zip(coarse['patches'], other['patches'], strict=True):
            for field in ('record_id', 'xyz', 'gids', 'electrical_groups', 'physical_patches'):
                if parent[field] != child[field]:
                    raise ValueError('Physical patches changed: '+field)
        pa, pb = coarse['patch_qc']['contacts'], other['patch_qc']['contacts']
        if len(pa) != len(pb):
            raise ValueError('Contact count changed')
        for a, b in zip(pa, pb, strict=True):
            if (a['row'],a['contact'],a['group']) != (b['row'],b['contact'],b['group']):
                raise ValueError('Contact identity changed')
            if abs(a['area_mm2']-b['area_mm2']) > 1e-9:
                raise ValueError('Inherited contact area changed')
    spatial = difference(coarse['fine_Z_ohm'], inherited['fine_Z_ohm'])
    fractions = difference(inherited['fine_Z_ohm'], remapped['fine_Z_ohm'])
    total = difference(coarse['fine_Z_ohm'], remapped['fine_Z_ohm'])
    return dict(spatial=spatial, fractions=fractions, total=total,
                separated_comparison_passed=spatial['passed'] and fractions['passed'] and total['passed'],
                global_mesh_convergence_proven=False,
                scope='one conforming local refinement, same polyhedron and inherited finite contacts')


def verify_saved_identity(control, base, worker_file):
    base,worker_file=map(Path,(base,worker_file))
    identity=control['identity'];planfile=base/'plan.local.json'
    if identity['worker_sha256'] != sha(worker_file):
        raise ValueError('Executed forward worker changed')
    if identity['plan_sha256'] != sha(planfile):
        raise ValueError('Executed model plan changed')
    plan=read(planfile)
    for row in plan['sources']:
        if sha(row['path']) != row['sha256']:
            raise ValueError('Prepared model source changed')
    prepared=[path for path in base.glob('prepared_material_*.mat')
              if sha(path)==identity['prepared_sha256']]
    if len(prepared)!=1:raise ValueError('Executed material cache is missing or ambiguous')
    return plan,prepared[0]


def audit(work, coarse_file, inherited_file, remapped_file):
    work=Path(work); files=list(map(Path,[coarse_file,inherited_file,remapped_file]))
    rows=[read(path) for path in files]; result=compare(*rows)
    meshroot=work/'mesh_v9'; model=meshroot/'fine'
    pipe=Path(__file__).resolve().parents[1]
    bases=[meshroot/name for name in ('coarse_operators','fine_inherited','fine')]
    verified=[verify_saved_identity(row,base,pipe/'src/run_georg_quality_forward.m')
              for row,base in zip(rows,bases,strict=True)]
    plans=[x[0] for x in verified]
    if sha(bases[1]/'mesh.mat')!=sha(bases[2]/'mesh.mat'):
        raise ValueError('Fraction comparison uses different meshes')
    if plans[1]['fraction_origin']!='exact parent inheritance' or plans[2]['fraction_origin']!='CT mask integration':
        raise ValueError('Isolated material models have different provenance')
    for plan in plans:
        if sha(plan['pose_reference_mesh'])!=sha(bases[0]/'mesh.mat'):
            raise ValueError('Pose reference mesh changed')
    inheritance=read(meshroot/'fine_inherited/inheritance.local.json')
    if inheritance['parent_mesh_sha256']!=sha(bases[0]/'mesh.mat') or inheritance['fine_mesh_sha256']!=sha(bases[1]/'mesh.mat'):
        raise ValueError('Coefficient inheritance belongs to different meshes')
    if inheritance['parent_maps_sha256']!=sha(meshroot/'fine/parent_maps.mat'):
        raise ValueError('Coefficient inheritance map changed')
    energy=read(meshroot/'fine_inherited/inheritance.local.json')['energy_check']
    qc=[read(meshroot/name/'mesh_quality.local.json') for name in ('coarse','fine')]
    derivative_rows=rows[0]['derivatives']
    if len(derivative_rows)!=2:
        raise ValueError('Both cross derivative controls are required')
    prerequisites=dict(shape_QC_passed=all(x['passed'] for x in qc),
        energy_inheritance_passed=energy['passed'],
        quadrature_passed=all(x['quadrature_passed'] for x in rows),
        derivatives_passed=all(x['passed'] for x in derivative_rows),
        system_checks_passed=all(x['system_checks_passed'] and x['patch_qc']['passed'] for x in rows))
    dependencies=[*files,*(base/'plan.local.json' for base in bases),
        *(x[1] for x in verified),meshroot/'fine_inherited/inheritance.local.json',
        model/'plan.local.json',model/'mesh_quality.local.json',
        work/'geometry_v5/fem_contacts.local.json',Path(__file__),
        *(pipe/p for p in ('src/run_georg_quality_forward.m',
            'src/trkg4_georg_physical_montages.m','src/trkg4_lateral_cem_response.m',
            'src/trkg4_georg_lateral_materials.m','src/trkg4_georg_material_rules.m',
            'src/trkg4_scale_fmdl_to_si.m','src/trkg4_cem_contact_matrix.m',
            'tools/trkg4_local_exact_patches.m','vendor_stl_eidors/electrode_faces_by_area.m'))]
    result.update(schema='georg_quality_separated_numerics_v1',complete=True,
        **prerequisites,all_checks_passed=all(prerequisites.values()) and result['separated_comparison_passed'],
        model_plan_sha256=sha(model/'plan.local.json'),
        prepared_sha256=rows[2]['identity']['prepared_sha256'],
        physical_contacts_sha256=rows[2]['identity']['physical_contacts_sha256'],
        physical_montage_helper_sha256=rows[2]['identity']['physical_montage_helper_sha256'],
        sources=[dict(path=str(path.resolve()),sha256=sha(path)) for path in dependencies],
        quantitative_result_accepted=False,physiological_validation=False,
        record_ids=rows[0]['record_ids'],axes=rows[0]['axes'],states=rows[0]['states'])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True);p.add_argument('--label',required=True)
    a=p.parse_args();work=a.work
    result=audit(work,*[work/(prefix+'_'+a.label+'.local.json') for prefix in
                       ('coarse_forward','fine_inherited','fine_remapped')])
    out=work/('numerical_audit_'+a.label+'.local.json')
    if out.exists():raise ValueError('Use a fresh immutable numerical audit')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('shape_QC_passed','quadrature_passed',
          'derivatives_passed','system_checks_passed','separated_comparison_passed')},ensure_ascii=False))
