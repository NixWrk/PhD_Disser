"""Immutable geometry/author-landmark preparation; no pointer, picker or FEM run.

Copies a flat geometry directory and both source JSON sets byte-for-byte,
then calls georg_ct_model.transfer ONLY on the fresh copy. Historical
projection products are rejected. Original contexts remain in source copies.
An optional FEM surface/passport is a separate reference, never picker skin.

The result is prepared_not_selected, with all global numerical/acceptance
flags false. A failed preparation leaves an incomplete directory; use a new
output, never overwrite or remove a previous bundle through this tool.
"""
from pathlib import Path
import argparse
import ast
import copy
import hashlib
import importlib.metadata
import json
import re
import shutil

ROOT=Path(__file__).resolve().parents[1]
ROLES=['I_plus','V_plus','V_minus','I_minus']
NAMES=[f'georg_tepc_{n}.json' for n in range(2,8)]
REQUIRED_GEOMETRY=['body_ct_native.stl','body_ct_surface.stl','body_embedded_arms_370mm.stl',
                   'geometry.local.json','contour_qc.local.json']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value,*,fresh=True):
    with Path(path).open('x' if fresh else 'w',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def require_sha(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):
        raise ValueError('Expected an explicit SHA256, not an inferred identity')
    return value


def load_transfer():
    """Execute the unchanged transfer definition without mesher imports.

    The original module imports Gmsh/segmentation packages although transfer
    itself only needs JSON, NumPy, trimesh and file helpers. Bind those minimal
    globals explicitly; do not edit or reimplement the projection function.
    """
    import numpy as np
    import trimesh
    source=ROOT/'tools/georg_ct_model.py'
    tree=ast.parse(source.read_text(encoding='utf-8-sig'),filename=str(source))
    definitions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='transfer']
    if len(definitions)!=1:raise ValueError('Expected exactly one original transfer definition')
    namespace=dict(np=np,trimesh=trimesh,json=json,sha=sha,read=read,write=write,
                   __name__='georg_ct_model_transfer_only',__file__=str(source))
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(source),'exec'),namespace)
    return namespace['transfer']


def coordinates(payload):
    import numpy as np
    if (payload.get('placement_mode')!='explicit_points' or payload.get('units')!='mm' or
            payload.get('electrode_order')!=ROLES):
        raise ValueError('Expected explicit I-U-U-I coordinates in mm')
    if 'ct_transfer_provenance' in payload or 'projection' in payload:
        raise ValueError('Previous projections are not valid author-coordinate sources')
    context=payload['measurement_context']
    if (context.get('physical_coordinate_frame')!='LPS' or context.get('units')!='mm' or
            context.get('subject_pseudonym')!='Georg' or context.get('experiment')!='scorpion_v2_georg' or
            context.get('electrode_configuration_family')!='TEPC'):
        raise ValueError('Expected the original Georg TEPC experiment, LPS/mm')
    points=payload['preliminary_electrode_centers_xyz_mm']
    if set(points)!=set(ROLES):raise ValueError('All four named author centres are required')
    result=np.asarray([points[k] for k in ROLES],float)
    if result.shape!=(4,3) or not np.isfinite(result).all():raise ValueError('Invalid author coordinates')
    return result


def validate_landmark_sources(raw_dir,normalised_dir,ct_sha,*,filename_mapping_confirmed):
    import numpy as np
    if not filename_mapping_confirmed:
        raise ValueError('Filename mapping 2..7 -> 012..017 needs explicit author confirmation')
    rows=[]
    for directory in (raw_dir,normalised_dir):
        if sorted(p.name for p in directory.glob('georg_tepc_*.json'))!=NAMES:
            raise ValueError('Each source directory must contain all six distinct montage JSON files')
    for n,name in zip(range(2,8),NAMES):
        rawfile=raw_dir/name;normalfile=normalised_dir/name
        if rawfile.is_symlink() or normalfile.is_symlink():raise ValueError('Landmark snapshots must be regular files')
        raw=read(rawfile);normal=read(normalfile);a=coordinates(raw);b=coordinates(normal)
        if not np.array_equal(a,b):raise ValueError('Raw and normalised author coordinates differ: '+name)
        rc,nc=raw['measurement_context'],normal['measurement_context']
        if rc['ct_sha256']!=ct_sha or nc['ct_sha256']!=ct_sha:
            raise ValueError('Landmarks and geometry must refer to the same CT')
        require_sha(rc['body_sha256'])
        for key in ('body_sha256','tissue_mask_sha256','experiment_date'):
            if rc.get(key)!=nc.get(key):raise ValueError('Raw/normalised source context changed: '+key)
        if nc['recording']['id']!=f'{n+10:03d}' or nc['recording']['montage_id']!=f'georg_tepc_{n}':
            raise ValueError('Normalised recording does not match the author-confirmed filename')
        provenance=normal['import_provenance'];rawsha=sha(rawfile)
        if (provenance['original_sha256']!=rawsha or provenance['selected_coordinates_changed'] is not False or
                provenance['original_recording']!=rc['recording']):
            raise ValueError('Normalised provenance does not bind the unchanged original: '+name)
        rows.append(dict(filename=name,montage=n,recording=f'{n+10:03d}',raw=raw,normalised=normal,
                         raw_sha256=rawsha,normalised_sha256=sha(normalfile),raw_centres=a.tolist()))
    if len({r['raw']['measurement_context']['body_sha256'] for r in rows})!=1:
        raise ValueError('The six author sources mix different old body surfaces')
    return rows


def prepare(geometry,raw_landmarks,normalised_landmarks,out,*,filename_mapping_confirmed=False,
            fem_surface=None,fem_surface_passport=None):
    import numpy as np
    directories=list(map(lambda p:Path(p).resolve(),(geometry,raw_landmarks,normalised_landmarks)))
    geometry,raw_landmarks,normalised_landmarks=directories
    if Path(out).is_symlink():raise ValueError('Use a fresh separate bundle directory')
    out=Path(out).resolve()
    if out.exists() or any(out.is_relative_to(p) or p.is_relative_to(out) for p in directories):
        raise ValueError('Use a fresh separate bundle directory')
    if not all(p.is_dir() for p in directories):raise ValueError('All input directories must exist')
    geometry_files=sorted(geometry.iterdir())
    if any(p.is_symlink() or not p.is_file() for p in geometry_files):
        raise ValueError('Expected flat geometry files; do not clone prior landmark/product directories')
    if any(p.name in ('centre_transfer.local.json','geometry_bundle.local.json') for p in geometry_files):
        raise ValueError('Geometry input includes previous transfer/bundle products')
    if not all((geometry/name).is_file() for name in REQUIRED_GEOMETRY):
        raise ValueError('Incomplete native CT/skin/union geometry package')
    g=read(geometry/'geometry.local.json');qc=read(geometry/'contour_qc.local.json')
    ct_sha=require_sha(qc['ct_sha256'])
    union_sha=sha(geometry/'body_embedded_arms_370mm.stl')
    if g['surface_sha256']!=union_sha or g['source_sha256']!=sha(geometry/'body_ct_surface.stl'):
        raise ValueError('Geometry passport differs from actual union or CT surface')
    rows=validate_landmark_sources(raw_landmarks,normalised_landmarks,ct_sha,
                                   filename_mapping_confirmed=filename_mapping_confirmed)
    sources={}
    def pin(path,expected=None):
        path=Path(path).resolve()
        if path not in sources:sources[path]=sha(path)
        if expected is not None and sources[path]!=expected:raise ValueError('Input snapshot changed: '+str(path))
        return sources[path]
    for file in geometry_files:pin(file)
    for row in rows:
        pin(raw_landmarks/row['filename'],row['raw_sha256'])
        pin(normalised_landmarks/row['filename'],row['normalised_sha256'])
    fem_surface=Path(fem_surface).resolve() if fem_surface else None
    fem_surface_passport=Path(fem_surface_passport).resolve() if fem_surface_passport else None
    fem_qc=None
    if fem_surface:
        if fem_surface.suffix.lower()!='.stl':raise ValueError('Supply an explicit FEM boundary STL, not a volume mesh')
        pin(fem_surface)
    if fem_surface_passport:
        pin(fem_surface_passport);fem_qc=read(fem_surface_passport)
        if fem_qc['source_surface_sha256']!=union_sha:raise ValueError('FEM passport refers to another native body union')
        if fem_surface and fem_qc['candidate_surface_sha256']!=pin(fem_surface):
            raise ValueError('Supplied surface is not the passport\'s actual FEM boundary')
    transfer_source=ROOT/'tools/georg_ct_model.py'
    for file in (Path(__file__),transfer_source):pin(file)
    transfer=load_transfer()
    out.mkdir(parents=True,exist_ok=False)
    copies=[]
    def clone(source,dest):
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
        if sha(dest)!=pin(source):raise ValueError('Byte-copy verification failed: '+str(source))
        copies.append(dict(source_path=str(source),path=dest.relative_to(out).as_posix(),
                           sha256=pin(source),byte_identical_verified=True))
    for file in geometry_files:clone(file,out/file.name)
    for row in rows:
        name=row['filename']
        clone(raw_landmarks/name,out/'landmark_sources/originals'/name)
        clone(normalised_landmarks/name,out/'landmark_sources/normalised'/name)
    fem_role=dict(supplied=False,used_for_landmark_projection=False,used_as_picker_geometry=False,
                  FEM_was_run=False)
    if fem_surface:
        clone(fem_surface,out/'fem_reference/fem_boundary.stl')
        fem_role.update(supplied=True,path='fem_reference/fem_boundary.stl',sha256=pin(fem_surface))
    if fem_surface_passport:
        clone(fem_surface_passport,out/'fem_reference/surface_passport.local.json')
        fem_role.update(passport_path='fem_reference/surface_passport.local.json',
            passport_sha256=pin(fem_surface_passport),passport_is_reference_only=True,
            passport_surface_bytes_bound=bool(fem_surface),recorded_surface_sha256=fem_qc['candidate_surface_sha256'],
            recorded_passed=fem_qc.get('passed'),anatomical_accuracy_validated=False)
    (out/'landmarks').mkdir()
    transfer(out,out/'landmark_sources/normalised')
    transfer_rows=read(out/'centre_transfer.local.json')
    if len(transfer_rows)!=24:raise ValueError('Transfer did not produce all 24 contact records')
    products=[];landmarks=[]
    for row in rows:
        name=row['filename'];path=out/'landmarks'/name;derived=read(path)
        allowed=copy.deepcopy(row['normalised'])
        allowed['measurement_context']['body_sha256']=union_sha
        allowed['preliminary_electrode_centers_xyz_mm']=derived['preliminary_electrode_centers_xyz_mm']
        allowed['ct_transfer_provenance']=derived['ct_transfer_provenance']
        if derived!=allowed:raise ValueError('Transfer changed additional source context fields')
        projected=np.asarray([derived['preliminary_electrode_centers_xyz_mm'][k] for k in ROLES],float)
        raw=np.asarray(row['raw_centres']);displacement=np.linalg.norm(projected-raw,axis=1)
        old=derived['ct_transfer_provenance']
        if (projected.shape!=(4,3) or not np.isfinite(projected).all() or
                old['source_sha256']!=row['normalised_sha256'] or
                old['raw_centres_lps_mm']!=row['raw_centres'] or
                not np.allclose(displacement,old['displacement_mm'],rtol=1e-12,atol=1e-10)):
            raise ValueError('Transferred coordinates/displacements/provenance disagree')
        per_contact=[r for r in transfer_rows if r['montage']==f"TEPC{row['montage']}"]
        if [r['contact'] for r in per_contact]!=ROLES:raise ValueError('Contact order changed during transfer')
        for k,report in enumerate(per_contact):
            if (report['recording']!=row['recording'] or report['original_lps_mm']!=row['raw_centres'][k] or
                    report['projected_lps_mm']!=projected[k].tolist() or
                    not np.isclose(report['displacement_mm'],displacement[k],rtol=1e-12,atol=1e-10) or
                    report['displacement_is_placement_error'] is not False):
                raise ValueError('Transfer report differs from saved landmarks')
        provenance=dict(source_sha256=row['normalised_sha256'],raw_original_sha256=row['raw_sha256'],
            normalised_sha256=row['normalised_sha256'],
            source_raw_path='landmark_sources/originals/'+name,
            source_normalised_path='landmark_sources/normalised/'+name,
            old_body_sha256=row['raw']['measurement_context']['body_sha256'],new_body_sha256=union_sha,
            ct_sha256=ct_sha,ct_identity_checked_against='recorded contour passport SHA; raw CT bytes not loaded',
            raw_centres_lps_mm=row['raw_centres'],projected_centres_lps_mm=projected.tolist(),
            displacement_mm=displacement.tolist(),method=old['method'],is_new_manual_placement=False,
            projection_is_candidate=True,placement_accuracy_mm=None,displacement_is_placement_error=False,
            filename_mapping_source='direct_author_confirmation',recording=row['recording'],
            raw_source_measurement_context=copy.deepcopy(row['raw']['measurement_context']),
            normalised_source_measurement_context=copy.deepcopy(row['normalised']['measurement_context']),
            transfer_helper_sha256=pin(transfer_source))
        derived['ct_transfer_provenance']=provenance
        derived['measurement_context']['status']='derived_projection_of_author_reconstruction_unvalidated'
        write(path,derived,fresh=False)  # Only the product created by this fresh preparation.
        products.append(dict(path=path.relative_to(out).as_posix(),sha256=sha(path)))
        landmarks.append(dict(filename=name,**provenance))
    products.append(dict(path='centre_transfer.local.json',sha256=sha(out/'centre_transfer.local.json')))
    for source,digest in sources.items():
        if sha(source)!=digest:raise ValueError('Source changed during bundle preparation: '+str(source))
    for record in copies:
        if sha(out/record['path'])!=record['sha256']:raise ValueError('Copied source was modified during transfer')
    roles=dict(native_CT_contour=dict(path='body_ct_native.stl',sha256=pin(geometry/'body_ct_native.stl')),
        corrected_CT_skin=dict(path='body_ct_surface.stl',sha256=pin(geometry/'body_ct_surface.stl')),
        body_union_picker_surface=dict(path='body_embedded_arms_370mm.stl',sha256=union_sha,
            includes_artificial_cylindrical_arms=True),FEM_boundary_reference=fem_role)
    manifest=dict(schema='georg_quality_geometry_bundle_v1',status='prepared_not_selected',
        geometry_roles=roles,ct_sha256=ct_sha,landmarks=landmarks,copies=copies,products=products,
        sources=[dict(path=str(p),sha256=d) for p,d in sources.items()],
        source_geometry_directory=str(geometry),producer_sha256=pin(Path(__file__)),
        runtime={name:importlib.metadata.version(name) for name in ('numpy','trimesh','rtree')},
        transfer_loading='Unchanged original transfer AST; minimal projection globals; mesher/segmentation modules not imported',
        checks=dict(byte_copies_verified=True,raw_normalised_centres_equal=True,source_contexts_preserved=True,
            all_24_transfer_records_verified=True,source_hashes_rechecked=True),
        global_model_pointer_changed=False,common_model_selected=False,legacy_worker_routing_changed=False,
        mesh_convergence_passed=False,integration_comparison_passed=False,quantitative_inversion_allowed=False,
        global_mesh_convergence_proven=False,independent_anatomical_validation=False,
        physiological_validation=False,placement_accuracy_is_known=False,FEM_was_run=False)
    write(out/'geometry_bundle.local.json',manifest)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry',required=True,type=Path)
    p.add_argument('--raw-landmarks',required=True,type=Path)
    p.add_argument('--normalised-landmarks',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--confirmed-filename-mapping',action='store_true')
    p.add_argument('--fem-surface',type=Path,help='Optional actual FEM boundary STL, kept separate')
    p.add_argument('--fem-surface-passport',type=Path,help='Optional reference passport; never replaces picker skin')
    a=p.parse_args()
    result=prepare(a.geometry,a.raw_landmarks,a.normalised_landmarks,a.out,
        filename_mapping_confirmed=a.confirmed_filename_mapping,
        fem_surface=a.fem_surface,fem_surface_passport=a.fem_surface_passport)
    print(json.dumps(dict(output=str(a.out.resolve()),status=result['status'],
                          landmarks=len(result['landmarks']),FEM_was_run=False)))


if __name__=='__main__':main()
