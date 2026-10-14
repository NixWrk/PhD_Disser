"""Publish or verify the shared Georg model after numerical checks.

The pointer and all medical arrays are local, ignored files. Publication here
means selection as a numerical reference, never physiological validation.
"""
from pathlib import Path
import argparse

import numpy as np

from georg_ct_model import PIPE, OUT, SEG
from scorpion_georg_fem_prepare import sha, read, write

POINTER=PIPE/'output/georg_current_model.local.json'


def resolve(pointer=POINTER, require_converged=False):
    c=read(pointer)
    if c['status'] not in ['numerical_reference_checked','geometry_reference_checked']:
        raise ValueError('Georg model has not passed numerical checks')
    required=('mesh_convergence_passed','integration_comparison_passed',
              'quantitative_inversion_allowed')
    if require_converged and (c['status']!='numerical_reference_checked'
            or any(c.get(key) is not True for key in required)):
        raise ValueError('Georg FEM precision is unresolved; quantitative inversion is not permitted')
    for item in c['files']:
        path=PIPE/item['path']
        if not path.is_file() or sha(path)!=item['sha256']:
            raise ValueError('Shared model file changed: '+item['path'])
    return PIPE/c['model_directory'], c


def publish(out, coarse=6, fine=4, select_geometry=False, audit_mesh=None):
    geometry=read(out/'geometry.local.json');contour=read(out/'contour_qc.local.json')
    if not geometry['watertight'] or geometry['self_intersecting_faces']:
        raise ValueError('Exterior QC failed')
    for join in geometry['joins'].values():
        if join['cross_section_covered_fraction']<1-1e-9:raise ValueError('Incomplete cylinder join')
    fine_dir=out/f'fem_h{fine:g}';coarse_dir=out/f'fem_h{coarse:g}'
    mesh_sizes=[coarse,fine]+([] if audit_mesh is None else [audit_mesh])
    for h in mesh_sizes:
        mesher=read(out/f'mesher_h{h:g}.local.json')
        qc=read(out/f'mesh_h{h:g}_qc.local.json')
        if mesher['source_sha256']!=sha(out/'body_embedded_arms_370mm.stl') or mesher['mesh_sha256']!=sha(out/f'mesh_h{h:g}.msh'):
            raise ValueError('Mesh source provenance changed')
        if not qc['accepted'] or not qc['watertight'] or qc['self_intersecting_boundary_faces']:
            raise ValueError('Volumetric mesh exterior failed')
    directories=[coarse_dir,fine_dir]+([] if audit_mesh is None else [out/f'fem_h{audit_mesh:g}'])
    plans=[read(d/'plan.local.json') for d in directories]
    producer_sources=[]
    for plan in plans:
        if not plan['all_tetrahedra_positive'] or not plan['conforming_boundary_verified']:
            raise ValueError('Refined volume mesh failed positivity or boundary checks')
        producer=PIPE/'tools/georg_ct_fem_prepare.py'
        if plan['code_sha256']!=sha(producer):
            producer=out/'code_sources'/(plan['code_sha256']+'.py')
            if not producer.is_file() or sha(producer)!=plan['code_sha256']:
                raise ValueError('Executed mesh/material preparation source is unavailable')
        producer_sources.append(producer)
        for source in plan['sources']:
            if sha(source['path'])!=source['sha256']:raise ValueError('Prepared model source changed')
    A={r['id']:r for r in read(coarse_dir/'forward_results.local.json')}
    B={r['id']:r for r in read(fine_dir/'forward_results.local.json')}
    contacts=read(out/'fem_contacts.local.json')
    expected={r['id'] for r in contacts['scenarios']}
    U=B
    if set(U)!=expected:raise ValueError('Position uncertainty series is incomplete on the selected mesh')
    nominal=[r['id'] for r in contacts['scenarios'] if r['perturbation']=='nominal' or r['family'] in ['TTRKG','reference']]
    if not set(nominal)<=set(B):raise ValueError('Second mesh reference series is incomplete')
    rows=[]
    for name in nominal:
        a,b=A[name],B[name];diff=abs(a['Z_ohm']-b['Z_ohm'])
        tolerance=.02+.005*abs(b['Z_ohm'])
        rows.append(dict(id=name,coarse_ohm=a['Z_ohm'],fine_ohm=b['Z_ohm'],difference_ohm=diff,
            tolerance_ohm=tolerance,passed=diff<=tolerance))
    write(out/'mesh_convergence.local.json',dict(cases=rows,
        tolerance='0.02 Ohm + 0.5 percent of reference impedance',
        checked_mesh_pair_mm=[coarse,fine],is_global_convergence_proof=False))
    mesh_passed=all(r['passed'] for r in rows)
    if not mesh_passed and not select_geometry:
        raise ValueError('Mesh comparison failed; current pointer not changed')
    checked_results=[(coarse_dir,A),(fine_dir,B)]
    if audit_mesh is not None:
        audit_dir=out/f'fem_h{audit_mesh:g}'
        C={r['id']:r for r in read(audit_dir/'forward_results.local.json')}
        if set(C)!=set(nominal):raise ValueError('Additional mesh nominal series is incomplete')
        checked_results.append((audit_dir,C))
    all_rows=[r for _,results in checked_results for r in results.values()]
    for directory,results in checked_results:
        expected_plan=sha(directory/'plan.local.json')
        expected_worker=sha(PIPE/'src/run_georg_ct_model.m')
        for row in results.values():
            if row['identity']['plan_sha256']!=expected_plan or row['identity']['worker_sha256']!=expected_worker:
                raise ValueError('Forward result is from another plan or worker')
            for key,path in [('patch_builder_sha256',PIPE/'vendor_stl_eidors/electrode_faces_by_area.m'),
                    ('cem_builder_sha256',PIPE/'src/trkg4_cem_contact_matrix.m')]:
                if row['identity'][key]!=sha(path):raise ValueError('Contact or CEM builder changed')
            patches=row['contact_qc']
            if not patches or any(p['relative_area_error']>=.12 or p['centroid_offset_mm']>=1 or
                p['radial_extent_mm']>=3.7 for p in patches):
                raise ValueError('Finite contact geometry failed')
    integration_passed=True
    for r in all_rows:
        if r['linear_system_relative_residual']>=1e-7 or r['reciprocity_error_ohm']>=1e-7:
            raise ValueError('CEM numerical verification failed')
        if r['quadrature_difference_ohm'] is not None and r['quadrature_difference_ohm']>.02+.001*abs(r['Z_ohm']):
            integration_passed=False
    if not integration_passed and not select_geometry:raise ValueError('Integration comparison failed')
    for _,results in checked_results:
        test=results['TEPC4_nominal']
        if test['jacobian_relative_error']>=1e-5 or test['eidors_assembly_relative_error']>=1e-12:
            raise ValueError('EIDORS assembly or derivative check failed')
    audit_path=None
    if audit_mesh is not None:
        audit_rows=[]
        for name in nominal:
            a,b=B[name],C[name];diff=abs(a['Z_ohm']-b['Z_ohm'])
            tolerance=.02+.005*abs(b['Z_ohm'])
            audit_rows.append(dict(id=name,coarse_ohm=a['Z_ohm'],fine_ohm=b['Z_ohm'],
                difference_ohm=diff,tolerance_ohm=tolerance,passed=diff<=tolerance))
        audit_path=out/'additional_mesh_comparison.local.json'
        write(audit_path,dict(cases=audit_rows,checked_mesh_pair_mm=[fine,audit_mesh],
            tolerance='0.02 Ohm + 0.5 percent of reference impedance',
            is_global_convergence_proof=False,uncertainty_cases_recomputed=False))
        mesh_passed=mesh_passed and all(r['passed'] for r in audit_rows)
        if not mesh_passed and not select_geometry:
            raise ValueError('Additional mesh comparison failed; current pointer not changed')
    summary=[]
    for n in range(2,8):
        nominal_row=U[f'TEPC{n}_nominal'];base=nominal_row['Z_ohm']
        for amplitude in [1,3,5,10]:
            trials=[r for r in U.values() if r['montage']==str(n) and abs(r['shift_mm'])==amplitude and r['perturbation'].startswith('all contacts')]
            zz=[r['Z_ohm'] for r in trials]
            summary.append(dict(montage=n,amplitude_mm=amplitude,nominal_ohm=base,
                min_ohm=min(zz),max_ohm=max(zz),max_abs_change_ohm=max(abs(z-base) for z in zz),
                relative_max_change_percent=100*max(abs(z-base) for z in zz)/abs(base),
                amplitudes_are_confidence_intervals=False))
    write(out/'position_uncertainty.local.json',summary)
    individual=[]
    for n in range(2,8):
        base=U[f'TEPC{n}_nominal']['Z_ohm']
        trials=[r for r in U.values() if r['montage']==str(n) and r['perturbation'].startswith('one contact')]
        for r in trials:
            source=next(s for s in contacts['scenarios'] if s['id']==r['id'])
            individual.append(dict(montage=n,contact=source['shifted_contact'],shift_mm=r['shift_mm'],
                actual_displacement_mm=source['actual_displacement_mm'],Z_ohm=r['Z_ohm'],
                difference_from_nominal_ohm=r['Z_ohm']-base,is_confidence_interval=False))
    write(out/'individual_position_uncertainty.local.json',individual)
    J=np.array([B[f'TEPC{n}_nominal']['J_logrho_ohm'] for n in range(2,8)])[:,[0,1,4]]
    singular=np.linalg.svd(J,compute_uv=False)
    write(out/'material_sensitivity.local.json',dict(materials=['soft_background','lung','fat'],
        J_logrho_ohm=J.tolist(),singular_values_ohm=singular.tolist(),
        condition_number=float(singular[0]/singular[-1]),is_inverse_estimate=False,
        fixed_materials=['bone','heart'],materials_are_literature_scenario=True))
    files=[out/name for name in ['body_embedded_arms_370mm.stl','body_ct_surface.stl','body_ct_native.stl',
        'body_mask.nii.gz','geometry.local.json','contour_qc.local.json','contacts.local.json',
        'fem_contacts.local.json','centre_transfer.local.json','mesh_convergence.local.json',
        'position_uncertainty.local.json','individual_position_uncertainty.local.json',
        'material_sensitivity.local.json','contour_reproducibility.local.json',
        'contour_atlas_inventory.local.json','contour_visual_review.local.json']]
    files.extend(out/f'mesher_h{h:g}.local.json' for h in mesh_sizes)
    files.extend(out/f'mesh_h{h:g}_qc.local.json' for h in mesh_sizes)
    files.extend(out/f'mesh_h{h:g}.msh' for h in mesh_sizes)
    files.extend(seg for seg in (SEG/(name+'.nii.gz') for name in ['static_ct','heart','lung','bone','fat']))
    for d in directories:
        files.extend(d/name for name in ['plan.local.json','mesh.mat','fractions_q1024.mat','reference_materials.local.json','forward_results.local.json'])
        files.extend(sorted((d/'scenarios').glob('*_patches.local.json')))
        tiny_qc=d/'tiny_tetrahedra_qc.local.json'
        if tiny_qc.exists():files.append(tiny_qc)
    for plan in plans:
        files.extend(Path(source['path']) for source in plan['sources'])
    if audit_path is not None:files.append(audit_path)
    files.extend(PIPE/name for name in ['src/run_georg_ct_model.m','tools/georg_ct_fem_prepare.py',
        'tools/scorpion_georg_fem_prepare.py','tools/refine_surface_contact_mesh.py',
        'tools/georg_ct_model.py','src/trkg4_georg_current_model.m','run_pipeline.m',
        'tools/georg_model_contract.py','tools/scorpion_georg_geometry_revision.py',
        'vendor_stl_eidors/electrode_faces_by_area.m','src/trkg4_cem_contact_matrix.m',
        'configs/tissue_literature_v2.json','configs/georg_mesh_requirements.txt'])
    files.extend(sorted((out/'landmarks').glob('*.json')))
    files.extend(producer_sources)
    history=out/'mesh_h8_h6_negative_comparison.local.json'
    if history.exists():files.append(history)
    status='numerical_reference_checked' if mesh_passed and integration_passed else 'geometry_reference_checked'
    write(POINTER,dict(schema='georg_current_model_v1',status=status,
        subject_alias='georg',model_directory=out.relative_to(PIPE).as_posix(),
        fem_directory=fine_dir.relative_to(PIPE).as_posix(),segmentation_directory=SEG.relative_to(PIPE).as_posix(),
        mesh_comparison_mm=[coarse,fine],uncertainty_mesh_mm=fine,
        additional_mesh_comparison_mm=None if audit_mesh is None else [fine,audit_mesh],
        mesh_convergence_passed=mesh_passed,integration_comparison_passed=integration_passed,
        quantitative_inversion_allowed=status=='numerical_reference_checked',
        model_scope='shared exterior, cylinders, tissue partition and finite-contact forward reference',
        independent_anatomical_validation=False,physiological_validation=False,
        individual_tissue_resistivity_estimated=False,placement_accuracy_is_known=False,
        lateral_global_localization_performed=False,
        files=[dict(path=p.relative_to(PIPE).as_posix(),sha256=sha(p)) for p in dict.fromkeys(files)]))
    resolve();print('Shared Georg model selected; status:',status,'pointer:',POINTER,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--coarse',type=float,default=6);p.add_argument('--fine',type=float,default=4)
    p.add_argument('--audit-mesh',type=float,help='Attach a completed nominal series on an additional mesh')
    p.add_argument('--select-geometry',action='store_true',help='Select checked geometry while retaining failed FEM precision flags')
    p.add_argument('--verify',action='store_true');a=p.parse_args()
    if a.verify:print(resolve()[1]['status'])
    else:publish(a.out,a.coarse,a.fine,a.select_geometry,a.audit_mesh)
