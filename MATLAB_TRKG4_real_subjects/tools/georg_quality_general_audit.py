"""Audit isolated space/fraction controls with independent pose/contact masters."""
from pathlib import Path
import json
from georg_quality_comparison import read,sha,compare,verify_saved_identity

PIPE=Path(__file__).resolve().parents[1]


def verify_contact_identity(row,plan,base):
    identity=row['identity']['contact_inheritance'];base=Path(base)
    expected=dict(contact_parent_mesh_sha256=sha(plan['exact_contact_parent_mesh']),
        current_mesh_sha256=sha(base/'mesh.mat'),
        boundary_parent_map_sha256=sha(plan['boundary_parent_map']) if plan['boundary_parent_map'] else '',
        mapping_helper_sha256=sha(PIPE/'src/trkg4_georg_contact_inheritance.m'))
    for field,value in expected.items():
        if identity[field]!=value:raise ValueError('Executed contact mapping changed: '+field)


def compile_audit(base,inherited,remapped,output_files):
    bases=list(map(Path,(base,inherited,remapped)));files=list(map(Path,output_files))
    rows=[read(path) for path in files]
    verified=[verify_saved_identity(row,directory,PIPE/'src/run_georg_quality_forward.m')
              for row,directory in zip(rows,bases,strict=True)]
    plans=[item[0] for item in verified];result=compare(*rows)
    if sha(bases[1]/'mesh.mat')!=sha(bases[2]/'mesh.mat'):
        raise ValueError('Isolated tissue comparison must use one mesh')
    if plans[1]['fraction_origin']!='exact parent inheritance' or plans[2]['fraction_origin']!='CT mask integration':
        raise ValueError('Unexpected tissue fraction provenance')
    inheritance=read(bases[1]/'inheritance.local.json')
    if inheritance['parent_mesh_sha256']!=sha(bases[0]/'mesh.mat') or inheritance['fine_mesh_sha256']!=sha(bases[1]/'mesh.mat'):
        raise ValueError('Material inheritance belongs to another parent/fine mesh')
    if inheritance['parent_maps_sha256']!=sha(bases[2]/'parent_maps.mat'):
        raise ValueError('Material inheritance map changed')
    pose_hashes={sha(plan['pose_reference_mesh']) for plan in plans}
    contact_hashes={sha(plan['exact_contact_parent_mesh']) for plan in plans}
    if len(pose_hashes)!=1 or len(contact_hashes)!=1:
        raise ValueError('Pose/contact masters must each remain fixed')
    for row,plan,base in zip(rows,plans,bases,strict=True):verify_contact_identity(row,plan,base)
    derivatives=rows[0]['derivatives']
    if len(derivatives)!=2:raise ValueError('Both cross derivative controls are required')
    prerequisites=dict(shape_QC_passed=all(read(directory/'mesh_quality.local.json')['passed'] for directory in bases),
        energy_inheritance_passed=inheritance['energy_check']['passed'],
        quadrature_passed=all(row['quadrature_passed'] for row in rows),
        derivatives_passed=all(row['passed'] for row in derivatives),
        system_checks_passed=all(row['system_checks_passed'] and row['patch_qc']['passed'] for row in rows))
    paths=[*files,*(directory/'plan.local.json' for directory in bases),*(item[1] for item in verified),
        bases[1]/'inheritance.local.json',bases[2]/'parent_maps.mat',Path(__file__),
        PIPE/'tools/georg_quality_comparison.py',*(PIPE/path for path in (
            'src/run_georg_quality_forward.m','src/trkg4_georg_contact_inheritance.m',
            'src/trkg4_georg_physical_montages.m','src/trkg4_georg_lateral_materials.m',
            'src/trkg4_georg_material_rules.m','src/trkg4_lateral_cem_response.m',
            'src/trkg4_scale_fmdl_to_si.m','src/trkg4_cem_contact_matrix.m',
            'tools/trkg4_local_exact_patches.m','vendor_stl_eidors/electrode_faces_by_area.m'))]
    result.update(schema='georg_quality_separated_numerics_v1',complete=True,**prerequisites,
        all_checks_passed=all(prerequisites.values()) and result['separated_comparison_passed'],
        model_plan_sha256=sha(bases[2]/'plan.local.json'),prepared_sha256=rows[2]['identity']['prepared_sha256'],
        physical_contacts_sha256=rows[2]['identity']['physical_contacts_sha256'],
        physical_montage_helper_sha256=rows[2]['identity']['physical_montage_helper_sha256'],
        pose_reference_mesh_sha256=next(iter(pose_hashes)),contact_parent_mesh_sha256=next(iter(contact_hashes)),
        sources=[dict(path=str(path.resolve()),sha256=sha(path)) for path in dict.fromkeys(paths)],
        quantitative_result_accepted=False,physiological_validation=False,
        record_ids=rows[0]['record_ids'],axes=rows[0]['axes'],states=rows[0]['states'],
        model_directories=[str(directory.resolve()) for directory in bases])
    return result
