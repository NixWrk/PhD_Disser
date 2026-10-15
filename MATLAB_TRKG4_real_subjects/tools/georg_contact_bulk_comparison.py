"""Compare contact-only variants, including independently assembled bulk caches.

No FEM, cache rewriting or inferred cache ancestry. All ten sparse coefficient
matrices must have identical decoded CSC entries before contact isolation is
accepted. Uses the configured local HDF5 runtime in read-only mode.
"""
from pathlib import Path
import ctypes as c
import hashlib
import os
import numpy as np
from georg_quality_contact_variant import (ROOT, FILES, POINTS, read, sha,
    read_prepared_metadata, contact_difference)
from georg_quality_comparison import verify_saved_identity
from georg_quality_general_audit import verify_contact_identity


def sparse_part_fingerprints(path, library, nodes):
    library=Path(library).resolve()
    dll=os.add_dll_directory(str(library.parent)) if os.name=='nt' else None
    try:
        h=c.CDLL(str(library));ID=c.c_longlong
        def api(name,result,args):
            f=getattr(h,name);f.restype=result;f.argtypes=args;return f
        fo=api('H5Fopen',ID,[c.c_char_p,c.c_uint,ID]);fc=api('H5Fclose',c.c_int,[ID])
        do=api('H5Dopen2',ID,[ID,c.c_char_p,ID]);dc=api('H5Dclose',c.c_int,[ID])
        dt=api('H5Dget_type',ID,[ID]);ds=api('H5Dget_space',ID,[ID])
        tc=api('H5Tget_class',c.c_int,[ID]);ts=api('H5Tget_size',c.c_size_t,[ID])
        order=api('H5Tget_order',c.c_int,[ID]);sign=api('H5Tget_sign',c.c_int,[ID])
        tclose=api('H5Tclose',c.c_int,[ID]);sclose=api('H5Sclose',c.c_int,[ID])
        count=api('H5Sget_simple_extent_npoints',ID,[ID])
        dr=api('H5Dread',c.c_int,[ID,ID,ID,ID,ID,c.c_void_p])
        rd=api('H5Rdereference2',ID,[ID,ID,c.c_int,c.c_void_p]);gc=api('H5Gclose',c.c_int,[ID])
        ao=api('H5Aopen',ID,[ID,c.c_char_p,ID]);at=api('H5Aget_type',ID,[ID])
        ar=api('H5Aread',c.c_int,[ID,ID,c.c_void_p]);ac=api('H5Aclose',c.c_int,[ID])
        def dataset(group,name,kind):
            obj=do(group,name.encode(),0)
            if obj<0:raise ValueError('Missing sparse dataset: '+name)
            typ=dt(obj);space=ds(obj)
            try:
                n=int(count(space));size=int(ts(typ));byteorder=int(order(typ))
                if tc(typ)!=kind or size!=8 or byteorder not in (0,1) or not 0<=n<=2**28:
                    raise ValueError('Unsupported sparse dataset type or extent')
                if kind==0 and sign(typ)!=0:raise ValueError('Sparse indices must be unsigned')
                buf=c.create_string_buffer(max(1,n*size))
                if dr(obj,typ,0,0,0,buf)<0:raise ValueError('Cannot read sparse dataset')
                values=np.frombuffer(buf,dtype=('<' if byteorder==0 else '>')+('f8' if kind==1 else 'u8'),count=n)
                if kind==1 and not np.isfinite(values).all():raise ValueError('Nonfinite coefficient matrix')
                if name=='ir' and n and int(values.max())>=nodes:raise ValueError('Sparse row outside mesh')
                if name=='jc' and (n!=nodes+1 or values[0]!=0 or np.any(values[1:]<values[:-1])):
                    raise ValueError('Invalid sparse column pointers')
                result=dict(count=n,type_class=kind,itemsize=size,byteorder=byteorder,
                    sha256=hashlib.sha256(memoryview(buf).cast('B')[:n*size]).hexdigest())
                if name=='jc':result['nonzeros']=int(values[-1])
                return result
            finally:sclose(space);tclose(typ);dc(obj)
        file=fo(str(Path(path).resolve()).encode('utf-8'),0,0)
        if file<0:raise ValueError('Cannot open actual prepared cache')
        try:
            obj=do(file,b'parts',0)
            if obj<0:raise ValueError('Missing material matrix cell array')
            typ=dt(obj);space=ds(obj)
            try:
                if tc(typ)!=7 or int(count(space))!=10 or ts(typ)!=8:
                    raise ValueError('Expected ten sparse material matrices')
                refs=c.create_string_buffer(80)
                if dr(obj,typ,0,0,0,refs)<0:raise ValueError('Cannot read matrix references')
                answer=[]
                for i in range(10):
                    group=rd(file,0,0,c.byref(refs,8*i))
                    if group<0:raise ValueError('Cannot dereference material matrix')
                    try:
                        attr=ao(group,b'MATLAB_sparse',0)
                        if attr<0:raise ValueError('Expected MATLAB sparse matrix')
                        atype=at(attr)
                        try:
                            if ts(atype)!=8 or tc(atype)!=0 or order(atype) not in (0,1):raise ValueError('Invalid sparse dimension')
                            value=c.create_string_buffer(8)
                            if ar(attr,atype,value)<0:raise ValueError('Cannot read sparse dimension')
                            if int.from_bytes(value.raw,'little' if order(atype)==0 else 'big')!=nodes:
                                raise ValueError('Sparse matrix mesh dimension changed')
                        finally:tclose(atype);ac(attr)
                        row={name:dataset(group,name,kind) for name,kind in [('data',1),('ir',0),('jc',0)]}
                        if row['data']['count']!=row['ir']['count'] or row['jc']['nonzeros']!=row['data']['count']:
                            raise ValueError('Sparse nonzero counts disagree')
                        answer.append(row)
                    finally:gc(group)
                return answer
            finally:sclose(space);tclose(typ);dc(obj)
        finally:fc(file)
    finally:
        if dll is not None:dll.close()


def require_identical_parts(left,right):
    if len(left)!=10 or len(right)!=10 or left!=right:
        raise ValueError('Actual bulk coefficient matrix entries differ')


def compare_saved_controls(base,variant,inherited_forward,direct_forward,hdf5_library=None):
    base,variant=Path(base).resolve(),Path(variant).resolve()
    files=[Path(inherited_forward).resolve(),Path(direct_forward).resolve()]
    rows=[read(p) for p in files]
    verified=[verify_saved_identity(r,p,ROOT/'src/run_georg_quality_forward.m') for r,p in zip(rows,(base,variant))]
    plans=[item[0] for item in verified];caches=[item[1] for item in verified]
    if plans[1]['boundary_parent_map'] or sha(Path(plans[1]['exact_contact_parent_mesh']))!=sha(variant/'mesh.mat'):
        raise ValueError('Variant must select physical contacts directly on the current mesh')
    for name in FILES:
        if sha(base/name)!=sha(variant/name):raise ValueError('Bulk source differs: '+name)
    for row,plan,directory in zip(rows,plans,(base,variant)):
        verify_contact_identity(row,plan,directory)
        if plan['numerical_quadrature_points']!=list(POINTS) or not all(
                row[k] for k in ('system_checks_passed','quadrature_passed')) or not row['patch_qc']['passed']:
            raise ValueError('Incomplete fixed-point numerical control')
    for key in ('parameters','pose','record_ids','axes','states','target_Z_ohm','source_rule'):
        if rows[0][key]!=rows[1][key]:raise ValueError('Fixed input differs: '+key)
    for key in ('observations_sha256','reference_sha256','physical_contacts_sha256','physical_montage_helper_sha256'):
        if rows[0]['identity'][key]!=rows[1]['identity'][key]:raise ValueError('Physical contract differs: '+key)
    if rows[0]['identity']['physical_montage_helper_sha256']!=sha(ROOT/'src/trkg4_georg_physical_montages.m'):
        raise ValueError('Executed physical grouping helper changed')
    for a,b in zip(rows[0]['patches'],rows[1]['patches'],strict=True):
        for key in ('record_id','xyz','gids','electrical_groups','physical_patches'):
            if a[key]!=b[key]:raise ValueError('Physical contact/group differs: '+key)
    identities=[];deltas=[];fingerprints=[];cachehashes=[sha(p) for p in caches]
    for directory,cache,plan in zip((base,variant),caches,plans):
        identity,delta=read_prepared_metadata(cache,hdf5_library)
        if identity['plan_sha256']!=sha(directory/'plan.local.json') or identity['mesh_sha256']!=sha(directory/'mesh.mat'):
            raise ValueError('Actual cache identity differs')
        for key,name in [('worker_sha256','trkg4_georg_lateral_materials.m'),('rule_helper_sha256','trkg4_georg_material_rules.m')]:
            if identity[key]!=sha(ROOT/'src'/name):raise ValueError('Executed coefficient helper changed')
        rules=identity['material_rules']
        if (identity['quadrature_points']!=list(POINTS) or rules['quadrature_points']!=list(POINTS) or
                rules['scale_nodes_to_si']!=1 or rules['length_unit']!='mm' or rules['node_scale_to_si']!=.001 or
                rules['scale_helper_sha256']!=sha(ROOT/'src/trkg4_scale_fmdl_to_si.m') or
                rules['fractions_sha256']!=[sha(directory/f'fractions_q{q}.mat') for q in POINTS] or not 0<=delta<1e-12):
            raise ValueError('Actual coefficient fractions/rules/units failed')
        identities.append(identity);deltas.append(delta)
        fingerprints.append(sparse_part_fingerprints(cache,hdf5_library,read(directory/'mesh_quality.local.json')['nodes']))
    for key in ('mesh_sha256','field_builder_sha256','worker_sha256','rule_helper_sha256','quadrature_points','material_rules'):
        if identities[0][key]!=identities[1][key]:raise ValueError('Coefficient identity differs: '+key)
    require_identical_parts(*fingerprints)
    if [sha(p) for p in caches]!=cachehashes:raise ValueError('Cache changed during comparison')
    result=contact_difference(rows[0]['fine_Z_ohm'],rows[1]['fine_Z_ohm'])
    result.update(schema='georg_fixed_bulk_contact_difference_v1',bulk_identical_verified=True,
        matrix_entries_compared=True,actual_bulk_entries_bitwise_identical=True,
        actual_parent_prepared_ancestry_verified=False,units_and_material_cache_identity_verified_here=True,
        verification_basis='All ten independently assembled CSC coefficient matrices: dimensions, data, row indices and column pointers',
        matrix_fingerprints=fingerprints[0],homogeneous_bulk_relative_difference=deltas,
        contact_difference_passed=result['passed'],quadrature_checks_passed=True,
        all_physical_patch_types_reselected=['lateral','permanent physical arms'],
        quantitative_result_accepted=False,global_mesh_convergence_proven=False,
        sources=[dict(path=str(p),sha256=sha(p)) for p in [*files,*caches,
            base/'plan.local.json',variant/'plan.local.json',Path(__file__),
            ROOT/'tools/georg_quality_contact_variant.py',ROOT/'tools/georg_quality_comparison.py',
            ROOT/'tools/georg_quality_general_audit.py']])
    return result
