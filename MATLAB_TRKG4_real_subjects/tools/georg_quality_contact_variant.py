"""Make an immutable bulk-identical DIRECT contact variant; never runs FEM.

From MATLAB_TRKG4_real_subjects::

    python -B tools/georg_quality_contact_variant.py --base "<selected fine2>" --out "<fresh direct_contact>"

Copy mesh/fractions/shape-QC/reference materials byte-for-byte. Keep the pose
reference and its geometry map; choose contacts on the copied CURRENT mesh.
parent_directory points to base so the unchanged material helper can reuse
both rules after checking mesh, fraction SHA, units and builder identity.
No prepared cache is copied or relabelled. New contact-patch QC is pending.

Later run BOTH fixed-point forwards with the SAME selected ReferenceFile and
compare_saved_controls(base, variant, inherited_forward, direct_forward).
With no h5py, supply hdf5_library=<installed runtime library path> to that
function, or set HDF5_LIBRARY_PATH. Only MAT identity/scalar metadata are read.
It retains signed differences and diagnostic ratios with qualified sources.
Contact footprints (lateral AND arms) may change; this comparison must not use
the space/fractions comparator's requirement of identical patch areas.
The result is a contact-discretization difference, not a physical error bound
or proof of mesh convergence. Selected-point completion is not inferred here.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import shutil

ROOT = Path(__file__).resolve().parents[1]
POINTS = (1024, 4096)
MATERIALS = ['soft_background', 'lung', 'bone', 'heart', 'fat']
FILES = ('mesh.mat', 'fractions_q1024.mat', 'fractions_q4096.mat',
         'mesh_quality.local.json', 'reference_materials.local.json')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def contract_path(value, base):
    path = Path(value)
    if path.is_absolute():
        if not path.is_file():
            raise ValueError('Contract file is missing: '+str(path))
        return path.resolve()
    choices = {p.resolve() for p in (base/path, ROOT/path, Path.cwd()/path) if p.is_file()}
    if len(choices) != 1:
        raise ValueError('Relative contract path is missing or ambiguous: '+str(path))
    return choices.pop()


def prepare(base, out):
    if Path(out).is_symlink():
        raise ValueError('Use a fresh separate contact-variant directory')
    base, out = Path(base).resolve(), Path(out).resolve()
    if out.exists() or out.is_symlink() or out.is_relative_to(base) or base.is_relative_to(out):
        raise ValueError('Use a fresh separate contact-variant directory')
    planfile = base/'plan.local.json'; plan = read(planfile)
    if (plan['frequency_hz'] != 100000 or plan['material_order'] != MATERIALS or
            plan['numerical_quadrature_points'] != list(POINTS) or
            not plan['constant_physical_contact_layout']):
        raise ValueError('Expected the fixed 100kHz/two-rule/permanent-contact model')
    pinned = {}
    def pin(path, expected=None):
        path = Path(path).resolve()
        if path not in pinned:
            pinned[path] = sha(path)
        digest = pinned[path]
        if expected is not None and digest != expected:
            raise ValueError('Base contract source changed: '+str(path))
        return digest
    for source in plan['sources']:
        pin(contract_path(source['path'], base), source['sha256'])
    bulk = {name:pin(base/name) for name in FILES}
    qc = read(base/'mesh_quality.local.json')
    if not qc['passed'] or qc['mesh_sha256'] != bulk['mesh.mat']:
        raise ValueError('Base shape QC or mesh identity failed')
    refs = read(base/'reference_materials.local.json')
    if refs['materials'] != MATERIALS or refs['frequency_hz'] != plan['frequency_hz']:
        raise ValueError('Reference materials do not match the base plan')
    pose = contract_path(plan['pose_reference_mesh'], base)
    old_master = contract_path(plan['exact_contact_parent_mesh'], base)
    contacts = contract_path(plan['contacts_file'], base)
    for path in (planfile, pose, old_master, contacts, Path(__file__)):
        pin(path)
    old_map = plan.get('boundary_parent_map', '')
    if old_map:
        old_map = contract_path(old_map, base); pin(old_map)
    elif pin(old_master) != bulk['mesh.mat']:
        raise ValueError('Inherited base has no map to its contact master')
    geometry = plan.get('geometry_boundary_parent_map')
    if geometry is None:
        if pin(pose) == pin(old_master):
            geometry = old_map  # Compatible coupled-master base.
        elif pin(pose) == bulk['mesh.mat']:
            geometry = ''
        else:
            raise ValueError('Independent pose reference requires geometry_boundary_parent_map')
    if geometry:
        geometry = contract_path(geometry, base); pin(geometry)
    elif pin(pose) != bulk['mesh.mat']:
        raise ValueError('Different pose mesh requires a geometry map')
    out.mkdir(parents=True, exist_ok=False)
    for name in FILES:
        shutil.copy2(base/name, out/name)
        if pin(out/name) != bulk[name]:
            raise ValueError('Copied bulk file changed: '+name)
    variant = copy.deepcopy(plan)
    variant.update(parent_directory=str(base), exact_contact_parent_mesh=str(out/'mesh.mat'),
        boundary_parent_map='', geometry_boundary_parent_map=str(geometry) if geometry else '',
        pose_reference_mesh=str(pose), contacts_file=str(contacts),
        contact_parent_mesh_sha256=bulk['mesh.mat'], shared_geometry_changed=False,
        geometry_changed=False, contact_discretization_changed=True,
        contact_geometry_rule='DIRECT current-mesh patches; unchanged bulk and original pose construction',
        contact_patch_QC_status='not_evaluated', contact_patch_change_verified=False,
        quantitative_inversion_allowed=False,
        global_mesh_convergence_proven=False, producer_sha256=pin(Path(__file__)),
        sources=[dict(path=str(path),sha256=digest) for path,digest in pinned.items()])
    # Preserve no misleading reference to the old model's prepared artifact.
    for key in ('validation_scope','prepared_file','prepared_sha256','material_cache_file','material_cache_sha256'):
        variant.pop(key, None)
    write_new(out/'plan.local.json', variant)
    facts = dict(schema='georg_direct_contact_variant_v1', base_directory=str(base),
        base_plan_sha256=pin(planfile), variant_plan_sha256=sha(out/'plan.local.json'),
        bulk_file_sha256=bulk, pose_reference_mesh_sha256=pin(pose),
        geometry_boundary_parent_map_sha256=pin(geometry) if geometry else '',
        physical_contact_inventory_sha256=pin(contacts),
        old_contact_master_sha256=pin(old_master), new_contact_master_sha256=bulk['mesh.mat'],
        old_boundary_parent_map_sha256=pin(old_map) if old_map else '', new_boundary_parent_map_sha256='',
        material_cache_policy=dict(parent_directory=str(base), expected_reused_rule_rows=[1,2],
            actual_reuse_verified_here=False, prepared_cache_copied=False,
            runtime_verification='Unchanged material helper checks parent cache identity, units, builder and fraction SHA'),
        sources=[dict(path=str(path),sha256=digest) for path,digest in pinned.items()],
        all_physical_patch_types_reselected=['lateral','permanent physical arms'],
        contact_patch_QC_status='not_evaluated', contact_patch_change_verified=False, FEM_was_run=False,
        quantitative_result_accepted=False, global_mesh_convergence_proven=False)
    write_new(out/'contact_variant.local.json', facts)
    return facts


def contact_difference(inherited, direct):
    """Finite vectors only: positive delta means DIRECT has the larger Z."""
    import numpy as np
    a, b = np.asarray(inherited, float), np.asarray(direct, float)
    if a.ndim != 1 or a.shape != b.shape or not len(a) or not np.all(np.isfinite(a)&np.isfinite(b)):
        raise ValueError('Expected equal nonempty finite one-dimensional Z vectors')
    with np.errstate(over='ignore', invalid='ignore'):
        signed = b-a; tolerance = .02+.005*np.abs(b); ratios = signed/tolerance
    if not np.all(np.isfinite(signed)&np.isfinite(ratios)):
        raise ValueError('Contact difference or ratio overflowed')
    return dict(signed_difference_ohm=signed.tolist(), difference_ohm=np.abs(signed).tolist(),
        tolerance_ohm=tolerance.tolist(), signed_ratio=ratios.tolist(), absolute_ratio=np.abs(ratios).tolist(),
        failed_values=int(np.count_nonzero(np.abs(ratios)>1)), maximum_ratio=float(np.max(np.abs(ratios))),
        passed=bool(np.all(np.abs(ratios)<=1)))


def read_prepared_metadata(path, hdf5_library=None):
    """Read identity and one scalar only; never read mm, parts or FEM fields.

    MAT v5 uses scipy's variable_names filter. MAT v7.3 uses h5py when
    available, otherwise an explicitly supplied HDF5 shared library (or
    HDF5_LIBRARY_PATH). The native reader also has a 100KB metadata limit.
    """
    import numpy as np
    path = Path(path).resolve()
    with path.open('rb') as stream:
        header = stream.read(128)
    if header.startswith(b'MATLAB 5.0'):
        from scipy.io import loadmat
        values = loadmat(path, variable_names=['identity','homogeneous_bulk_relative_difference'],
                         simplify_cells=True)
        def plain(value):
            if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
            if isinstance(value,np.ndarray):return [plain(v) for v in value.ravel()]
            if isinstance(value,(list,tuple)):return [plain(v) for v in value]
            if isinstance(value,np.generic):return value.item()
            return value
        identity = plain(values['identity']); delta = float(values['homogeneous_bulk_relative_difference'])
    else:
        try:
            import h5py
        except ImportError:
            return _read_native_hdf_metadata(path, hdf5_library)
        budget = [0]
        with h5py.File(path, 'r') as handle:
            def decode(obj, depth=0):
                if depth > 8:
                    raise ValueError('MAT metadata nesting exceeds the read limit')
                if isinstance(obj, h5py.Group):
                    if len(obj) > 100:
                        raise ValueError('Too many metadata fields')
                    return {key:decode(value,depth+1) for key,value in obj.items()}
                budget[0] += obj.size*obj.dtype.itemsize
                if budget[0] > 100000:
                    raise ValueError('Metadata-only size guard')
                if obj.attrs.get('MATLAB_empty',0):
                    return []
                data = obj[()]
                if h5py.check_dtype(ref=obj.dtype) is not None:
                    values = [decode(handle[ref],depth+1) for ref in data.ravel()]
                elif obj.attrs.get('MATLAB_class') in (b'char','char'):
                    return np.asarray(data,dtype='<u2').tobytes().decode('utf-16le').rstrip('\0')
                else:
                    values = data.ravel().tolist()
                return values[0] if len(values)==1 else values
            identity = decode(handle['identity'])
            delta = float(decode(handle['homogeneous_bulk_relative_difference']))
    if not isinstance(identity,dict) or len(json.dumps(identity))>100000 or not math.isfinite(delta):
        raise ValueError('Invalid prepared metadata')
    return identity, delta


def _read_native_hdf_metadata(path, library):
    """Bounded, read-only HDF5 C API fallback for the MATLAB desktop runtime."""
    import ctypes as c
    import os
    import struct
    library = library or os.environ.get('HDF5_LIBRARY_PATH')
    if not library:
        raise ValueError('MAT v7.3 needs h5py or an explicit hdf5_library/HDF5_LIBRARY_PATH')
    library = Path(library).resolve()
    dll_dir = os.add_dll_directory(str(library.parent)) if os.name=='nt' else None
    try:
        h = c.CDLL(str(library)); ID=c.c_longlong; SIZE=c.c_size_t
        def api(name,result,args):
            function=getattr(h,name);function.restype=result;function.argtypes=args;return function
        fo=api('H5Fopen',ID,[c.c_char_p,c.c_uint,ID]);fc=api('H5Fclose',c.c_int,[ID])
        go=api('H5Gopen2',ID,[ID,c.c_char_p,ID]);gc=api('H5Gclose',c.c_int,[ID])
        do=api('H5Dopen2',ID,[ID,c.c_char_p,ID]);dc=api('H5Dclose',c.c_int,[ID])
        dt=api('H5Dget_type',ID,[ID]);ds=api('H5Dget_space',ID,[ID])
        tc=api('H5Tget_class',c.c_int,[ID]);ts=api('H5Tget_size',SIZE,[ID])
        to=api('H5Tget_order',c.c_int,[ID]);sign=api('H5Tget_sign',c.c_int,[ID])
        tclose=api('H5Tclose',c.c_int,[ID]);sclose=api('H5Sclose',c.c_int,[ID])
        count=api('H5Sget_simple_extent_npoints',ID,[ID])
        dr=api('H5Dread',c.c_int,[ID,ID,ID,ID,ID,c.c_void_p])
        it=api('H5Iget_type',c.c_int,[ID]);rd=api('H5Rdereference2',ID,[ID,ID,c.c_int,c.c_void_p])
        ln=api('H5Lget_name_by_idx',ID,[ID,c.c_char_p,c.c_int,c.c_int,c.c_ulonglong,c.c_char_p,SIZE,ID])
        ae=api('H5Aexists',c.c_int,[ID,c.c_char_p]);ao=api('H5Aopen',ID,[ID,c.c_char_p,ID])
        at=api('H5Aget_type',ID,[ID]);ar=api('H5Aread',c.c_int,[ID,ID,c.c_void_p])
        ac=api('H5Aclose',c.c_int,[ID]);variable=api('H5Tis_variable_str',c.c_int,[ID])
        api('H5Eset_auto2',c.c_int,[ID,c.c_void_p,c.c_void_p])(0,None,None)
        budget=[0]
        def matlab_char(obj):
            if ae(obj,b'MATLAB_class')<=0:return False
            a=ao(obj,b'MATLAB_class',0);typ=at(a)
            try:
                size=ts(typ)
                if tc(typ)!=3 or variable(typ)!=0 or size>100:raise ValueError('Invalid MATLAB_class attribute')
                buf=c.create_string_buffer(size+1)
                if ar(a,typ,buf)<0:raise ValueError('Cannot read MATLAB_class')
                return buf.value==b'char'
            finally:tclose(typ);ac(a)
        def dataset(obj,depth):
            typ=dt(obj);space=ds(obj)
            try:
                n=count(space);kind=tc(typ);size=ts(typ);budget[0]+=n*size
                if n<0 or budget[0]>100000:raise ValueError('Metadata-only size guard')
                if ae(obj,b'MATLAB_empty')>0:return []
                buf=c.create_string_buffer(max(1,n*size))
                if dr(obj,typ,0,0,0,buf)<0:raise ValueError('Cannot read HDF5 metadata')
                if kind==7:
                    values=[]
                    for k in range(n):
                        target=rd(obj,0,0,c.cast(c.byref(buf,k*size),c.c_void_p))
                        if target<0:raise ValueError('Cannot dereference metadata')
                        try:values.append(object_value(target,depth+1))
                        finally:close_object(target)
                else:
                    order=to(typ)
                    if order not in (0,1):raise ValueError('Unsupported metadata byte order')
                    endian='<' if order==0 else '>'
                    raw=buf.raw[:n*size]
                    if matlab_char(obj):
                        if kind!=0 or size!=2:raise ValueError('Unsupported MATLAB char storage')
                        return raw.decode('utf-16le' if order==0 else 'utf-16be').rstrip('\0')
                    if kind==1 and size in (4,8):fmt={4:'f',8:'d'}[size]
                    elif kind==0 and size in (1,2,4,8):fmt={1:'b',2:'h',4:'i',8:'q'}[size] if sign(typ)==1 else {1:'B',2:'H',4:'I',8:'Q'}[size]
                    else:raise ValueError('Unsupported metadata numeric type')
                    values=list(struct.unpack(endian+str(n)+fmt,raw))
                return values[0] if n==1 else values
            finally:sclose(space);tclose(typ)
        def close_object(obj):
            kind=it(obj)
            if kind==2:gc(obj)
            elif kind==5:dc(obj)
            else:raise ValueError('Metadata reference is not a dataset or group')
        def object_value(obj,depth=0):
            if depth>8:raise ValueError('MAT metadata nesting exceeds the read limit')
            if it(obj)!=2:return dataset(obj,depth)
            result={}
            for i in range(101):
                size=ln(obj,b'.',0,0,i,None,0,0)
                if size<0:return result
                if i==100 or size>1000:raise ValueError('Too many or oversized metadata fields')
                name=c.create_string_buffer(size+1);ln(obj,b'.',0,0,i,name,size+1,0)
                child=go(obj,name.value,0)
                if child<0:child=do(obj,name.value,0)
                if child<0:raise ValueError('Cannot open identity field')
                try:result[name.value.decode('utf-8')]=object_value(child,depth+1)
                finally:close_object(child)
            raise ValueError('Metadata field limit')
        file=fo(str(Path(path).resolve()).encode('utf-8'),0,0)
        if file<0:raise ValueError('Cannot open prepared MAT in read-only mode')
        try:
            group=go(file,b'identity',0)
            if group<0:raise ValueError('No prepared identity group')
            try:identity=object_value(group)
            finally:gc(group)
            d=do(file,b'homogeneous_bulk_relative_difference',0)
            if d<0:raise ValueError('No homogeneous operator check')
            try:delta=float(dataset(d,0))
            finally:dc(d)
        finally:fc(file)
    finally:
        if dll_dir is not None:dll_dir.close()
    if not isinstance(identity,dict) or not math.isfinite(delta):raise ValueError('Invalid prepared metadata')
    return identity,delta


def verify_reused_ancestry(base, variant, parent_cache, direct_cache, hdf5_library=None):
    """Verify exact reuse selected by the executed material helper and metadata.

    This verifies lineage/rule eligibility, not entrywise equality of parts.
    """
    base,variant,parent_cache,direct_cache=map(lambda p:Path(p).resolve(),(base,variant,parent_cache,direct_cache))
    if Path(read(variant/'plan.local.json')['parent_directory']).resolve()!=base:
        raise ValueError('DIRECT material parent_directory changed')
    parent_hash=sha(parent_cache);direct_hash=sha(direct_cache)
    identities=[];deltas=[]
    for directory,cache in ((base,parent_cache),(variant,direct_cache)):
        identity,delta=read_prepared_metadata(cache,hdf5_library)
        if identity['plan_sha256']!=sha(directory/'plan.local.json') or identity['mesh_sha256']!=sha(directory/'mesh.mat'):
            raise ValueError('Prepared identity belongs to another plan or mesh')
        for key,file in (('worker_sha256','trkg4_georg_lateral_materials.m'),('rule_helper_sha256','trkg4_georg_material_rules.m')):
            if identity[key]!=sha(ROOT/'src'/file):raise ValueError('Executed material helper changed: '+key)
        if identity['quadrature_points']!=list(POINTS):raise ValueError('Prepared quadrature points changed')
        rules=identity['material_rules']
        if (rules['length_unit'] not in ('mm','millimeter','millimetre') or rules['scale_nodes_to_si']!=True or
                rules['node_scale_to_si']!=.001 or rules['quadrature_points']!=list(POINTS) or
                rules['scale_helper_sha256']!=sha(ROOT/'src/trkg4_scale_fmdl_to_si.m')):
            raise ValueError('Prepared SI/scale/rule identity failed')
        if rules['fractions_sha256']!=[sha(directory/f'fractions_q{q}.mat') for q in POINTS]:
            raise ValueError('Prepared fractions identity failed')
        if not math.isfinite(delta) or not 0<=delta<1e-12:raise ValueError('Homogeneous operator check failed')
        identities.append(identity);deltas.append(delta)
    if identities[1]['parent_prepared_sha256']!=parent_hash:
        raise ValueError('DIRECT prepared cache has a different actual parent')
    for key in ('mesh_sha256','field_builder_sha256','worker_sha256','rule_helper_sha256','quadrature_points','material_rules'):
        if identities[0][key]!=identities[1][key]:raise ValueError('Bulk cache reuse is ineligible: '+key)
    if sha(parent_cache)!=parent_hash or sha(direct_cache)!=direct_hash:
        raise ValueError('A prepared cache changed during metadata inspection')
    return dict(actual_parent_prepared_ancestry_verified=True,
        units_and_material_cache_identity_verified_here=True,reused_rule_rows=[1,2],
        verification_basis='Actual cache SHA, recorded identities and unchanged executed material/rule helpers',
        matrix_entries_compared=False,parent_prepared_sha256=parent_hash,direct_prepared_sha256=direct_hash,
        homogeneous_bulk_relative_difference=deltas,
        parent_cache_identity=identities[0],direct_cache_identity=identities[1])


def compare_saved_controls(base, variant, inherited_forward, direct_forward, hdf5_library=None):
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
    for key in ('observations_sha256','reference_sha256','physical_contacts_sha256','physical_montage_helper_sha256'):
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
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    a = parser.parse_args();facts = prepare(a.base,a.out)
    print(json.dumps(dict(output=str(a.out.resolve()),bulk_identical=True,
        expected_rule_rows=facts['material_cache_policy']['expected_reused_rule_rows'],
        contact_patch_QC_status='not_evaluated',FEM_was_run=False)))


if __name__ == '__main__':
    main()
