"""CT -> TotalSegmentator v2 -> native masks/STL -> common mesh preparation."""
from pathlib import Path
import copy
import subprocess
import sys

import nibabel as nib
import numpy as np
import trimesh
from scipy import ndimage as ndi

import ct_fem_core as core


def resolve(path, base):
    return str((base / path).resolve())


def verify_segmentation(folder, ct, *, roles):
    folder = Path(folder)
    path = folder / 'segmentation.local.json'
    if not path.is_file(): path = folder / 'segmentation.json'
    value = core.read(path)
    if (not value.get('complete') or value.get('backend') != 'TotalSegmentator' or
            not str(value.get('version', '')).startswith('2.') or value['ct_sha256'] != core.sha(ct)):
        raise ValueError('Complete TotalSegmentator v2 passport for this exact CT required')
    files = {str(path.resolve()): core.sha(path)}
    for key in ('model_weights_and_metadata','totalsegmentator_source'):
        for source, digest in value.get(key,{}).items():
            if core.sha(source) != digest: raise ValueError('Changed segmentator runtime/model: '+source)
            files[source]=digest
    for name in roles:
        product = value['products'][name]
        target = folder / product.get('file', name+'.nii.gz')
        if core.sha(target) != product['sha256']:
            raise ValueError('Changed TotalSegmentator aggregate: '+name)
        files[str(target.resolve())] = product['sha256']
        for source in product['source_masks']:
            target = folder / source['file']
            if core.sha(target) != source['sha256']:
                raise ValueError('Changed source class: '+source['file'])
            files[str(target.resolve())] = source['sha256']
    return value, files


def surface_from_body(folder, cfg, out, policy):
    """Native TS body plus explicitly declared TS tissue coverage; no HU fallback."""
    ct = nib.load(cfg['inputs']['ct']); image = nib.load(Path(folder) / 'body.nii.gz')
    if (image.shape != ct.shape or not np.array_equal(image.affine, ct.affine) or
            image.header.get_xyzt_units()[0] != 'mm'):
        raise ValueError('Body mask must retain the native CT grid and mm units')
    data = np.asanyarray(image.dataobj)
    if not np.isin(data, [0,1]).all() or not data.any():
        raise ValueError('Body mask must be nonempty binary')
    native = data > 0
    if type(policy['include_TS_tissues']) is not bool:
        raise ValueError('TS tissue inclusion must be an explicit boolean')
    for name in ['maximum_added_relative','maximum_disconnected_relative','maximum_internal_cavity_relative']:
        if not np.isfinite(policy[name]) or not 0 <= policy[name] <= .001:
            raise ValueError('TS domain discrepancy budgets must be finite and at most 0.001')
    domain = native.copy(); outside = {}; tissues = {}
    for name, path in cfg['inputs']['masks'].items():
        tissue = nib.load(path)
        if tissue.shape != ct.shape or not np.array_equal(tissue.affine,ct.affine):
            raise ValueError('Tissue grid differs from native CT: '+name)
        tissue_data = np.asanyarray(tissue.dataobj)
        if not np.isin(tissue_data,[0,1]).all(): raise ValueError('Non-binary tissue mask: '+name)
        mask = tissue_data > 0; tissues[name] = mask
        outside[name] = int(np.count_nonzero(mask & ~native))
        if policy['include_TS_tissues']: domain |= mask
    change = int(np.count_nonzero(domain != native)); relative = change / int(native.sum())
    if relative > policy['maximum_added_relative']:
        raise ValueError('TS tissue/body disagreement exceeds the explicit domain budget')
    labels,_ = ndi.label(domain)
    body_components = np.unique(labels[native])
    if len(body_components) != 1:
        raise ValueError('Native TotalSegmentator body spans disconnected components')
    retained = labels == body_components[0]; isolated = domain & ~retained
    excluded = {name:int(np.count_nonzero(mask & ~retained)) for name,mask in tissues.items()}
    isolated_count = int(isolated.sum())
    if isolated_count / int(native.sum()) > policy['maximum_disconnected_relative'] or excluded['lung'] or excluded['heart']:
        raise ValueError('Disconnected TS components exceed the declared scope or include heart/lung')
    domain = retained
    if isolated_count:
        image = nib.Nifti1Image(isolated.astype(np.uint8),ct.affine);image.header.set_xyzt_units('mm')
        nib.save(image,out/'isolated_TS_components.nii.gz')
    # Original neural masks remain immutable. Domain-restricted material masks
    # are new products, with explicit exclusion counts pending anatomical review.
    for name,mask in tissues.items():
        image = nib.Nifti1Image((mask & domain).astype(np.uint8),ct.affine);image.header.set_xyzt_units('mm')
        target = out/(name+'_within_domain.nii.gz');nib.save(image,target)
        cfg['inputs']['masks'][name] = str(target)
    domain_image = nib.Nifti1Image(domain.astype(np.uint8),ct.affine);domain_image.header.set_xyzt_units('mm')
    nib.save(domain_image,out/'computational_body_mask.nii.gz')
    vertices,faces,_,_ = core.marching_cubes(np.pad(domain,1).astype(np.uint8),level=.49)
    vertices = nib.affines.apply_affine(ct.affine,vertices-1)*[-1.,-1.,1.]
    interfaces = trimesh.Trimesh(vertices,faces,process=True);interfaces.fix_normals(multibody=True)
    components = sorted(interfaces.split(only_watertight=False),key=lambda part:abs(part.volume),reverse=True)
    if not components or any(not part.is_volume for part in components):
        raise ValueError('Native TS interfaces must be closed positive surfaces')
    mesh = components[0]; cavities = components[1:]
    cavity_relative = sum(abs(part.volume) for part in cavities)/abs(mesh.volume)
    if cavity_relative > policy['maximum_internal_cavity_relative']:
        raise ValueError('Internal TS cavity interfaces exceed the explicit solid-domain budget')
    if any(not mesh.contains(part.vertices).all() for part in cavities):
        raise ValueError('Additional TS surface is not an internal cavity; no component may be silently removed')
    if cavities: interfaces.export(out/'native_TS_all_interfaces.stl')
    qc = core.surface_qc(mesh)
    target = out / 'body_totalsegmentator_LPS.stl'; mesh.export(target)
    exported = trimesh.load(target, force='mesh'); core.surface_qc(exported)
    kernel = core.kernels(); count = cfg['surface_qc']['sample_count']
    distances = dict(native_to_STL=kernel['sampled_distance'](mesh,exported,count),
        STL_to_native=kernel['sampled_distance'](exported,mesh,count))
    if max(x['max_mm'] for x in distances.values()) > .001:
        raise ValueError('Native TS body to STL export fidelity failed')
    core.write_new(out / 'body_STL_QC.local.json',dict(source='TotalSegmentator_v2_body_union',
        classes=['body_trunc','body_extremities'],surface_level=.49,
        HU_threshold_fallback=False,manual_STL_fallback=False,
        morphology_applied=False,source_affine_RAS=ct.affine.tolist(),domain_policy=policy,
        tissue_voxels_outside_raw_TS_body=outside,added_TS_tissue_voxels=change,added_relative=relative,
        isolated_TS_component_voxels=isolated_count,excluded_material_voxels=excluded,
        source_masks_unchanged=True,anatomical_disagreement_pending_review=bool(isolated_count or cavities),
        internal_cavity_components=len(cavities),internal_cavity_relative_volume=cavity_relative,
        internal_cavity_volumes_mm3=[abs(float(part.volume)) for part in cavities],
        STL_domain_assumption='Outer skin solid; internal cavity interfaces retained diagnostically, not separate conductor boundaries',
        geometry_frame='LPS',units='mm',surface_QC=qc,export_distances=distances,
        body_mask_sha256=core.sha(Path(folder)/'body.nii.gz'),STL_sha256=core.sha(target),**core.FLAGS))
    return target


def prepare(config_path, out):
    config_path = Path(config_path).resolve(); out = Path(out).resolve()
    request = core.read(config_path); base = config_path.parent
    if request.get('schema') != 'ct_fem_ct_pipeline_v1':
        raise ValueError('Explicit CT/TotalSegmentator pipeline schema required')
    original_path = Path(resolve(request['geometry_configuration'], base))
    cfg = core.read(original_path)
    cfg['inputs']['ct'] = resolve(cfg['inputs']['ct'], original_path.parent)
    cfg['inputs']['masks'] = {k:resolve(v,original_path.parent) for k,v in cfg['inputs']['masks'].items()}
    ct = cfg['inputs']['ct']; start_sources = {str(config_path):core.sha(config_path),
        str(original_path):core.sha(original_path),ct:core.sha(ct),str(Path(__file__)):core.sha(__file__)}
    runtime = copy.deepcopy(request['segmentation'])
    for key in ('python','weights_directory','config_file'):
        if runtime.get(key): runtime[key] = resolve(runtime[key],base)
    core.positive(runtime['maximum_seconds'],'segmentation.maximum_seconds')
    if not str(runtime['version']).startswith('2.'):
        raise ValueError('This entry point requires TotalSegmentator v2')
    reuse = request.get('reuse_tissue_segmentation')
    if reuse:
        reuse = Path(resolve(reuse,base))
        _, bound = verify_segmentation(reuse,ct,roles=['lung','bone','heart','fat'])
        start_sources.update(bound)
    if out.exists(): raise ValueError('CT pipeline requires a fresh output directory')
    out.mkdir(parents=True)
    try:
        inference_config = out / 'segmentation_input.local.json'
        core.write_new(inference_config,dict(ct=ct,max_ct_voxels=cfg['resources']['max_ct_voxels'],segmentation=runtime))
        worker = Path(__file__).with_name('ct_fem_totalsegmentator.py')
        start_sources[str(worker)] = core.sha(worker)
        cmd = [runtime['python'],'-X','utf8',str(worker),'--config',str(inference_config),
            '--out',str(out/'segmentation'),'--tasks',*(['body'] if reuse else ['body','total','tissue_types'])]
        print('CT preparation: TotalSegmentator v2',flush=True)
        with (out/'segmentation.log').open('w',encoding='utf-8') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=runtime['maximum_seconds'],check=True)
        _, bound = verify_segmentation(out/'segmentation',ct,roles=['body'] if reuse else ['body','lung','bone','heart','fat'])
        start_sources.update(bound)
        tissue_dir = reuse or out/'segmentation'
        cfg['inputs']['masks'] = {name:str(tissue_dir/(name+'.nii.gz')) for name in ['lung','bone','heart','fat']}
        cfg['inputs']['mask_provenance'] = dict(method='automatic',author_or_backend='TotalSegmentator v2',
            passport=str(tissue_dir/('segmentation.json' if reuse else 'segmentation.local.json')),
            reused_precomputed_tissues=bool(reuse),independently_validated=False,
            material_masks='Explicit computational-domain restriction; source masks and exclusion QC retained')
        source = surface_from_body(out/'segmentation',cfg,out,request['body_domain'])
        for name in ['body_STL_QC.local.json','computational_body_mask.nii.gz','isolated_TS_components.nii.gz','native_TS_all_interfaces.stl']:
            file=out/name
            if file.exists():start_sources[str(file)]=core.sha(file)
        if request.get('contact_placement'):
            from ct_fem_contact_placement import apply as place_contacts
            skin = trimesh.load(source,force='mesh')
            placement = place_contacts(cfg,skin,request['contact_placement'])
            core.write_new(out/'contact_placement.local.json',placement)
            helper = Path(__file__).with_name('ct_fem_contact_placement.py')
            start_sources[str(helper)] = core.sha(helper)
        cfg['body'] = dict(strategy='supplied_STL_LPS',source_kind='TotalSegmentator_v2',
            source_stl=str(source),source_stl_sha256=core.sha(source),units='mm',frame='LPS',
            provenance='Native TotalSegmentator v2 body union with declared tissue-coverage policy; marching cubes, no HU/manual fallback',
            maximum_outside_body_voxels=request['maximum_outside_body_voxels'],
            coverage_limits_basis=request['coverage_limits_basis'])
        cfg['surface_qc']['supplied_STL_export_maximum_sampled_distance_mm'] = .001
        cfg.pop('manual_domain_audit',None)
        cfg['segmentation_sources'] = start_sources
        effective = out/'geometry_configuration.local.json'; core.write_new(effective,cfg)
        core.verify_sources(start_sources)
        mesh = core.prepare_mesh_pipeline(effective,out/'geometry')
        core.verify_sources(start_sources)
        result = dict(stage='pipeline',status='prepared_not_validated',
            segmentation_backend='TotalSegmentator_v2',segmentation_complete=True,
            geometry_and_shape_QC_passed=True,mesh_directory=mesh['mesh_directory'],
            sources=start_sources,effective_configuration_sha256=core.sha(effective),**core.FLAGS)
        core.write_new(out/'pipeline.local.json',result)
        return result
    except BaseException as error:
        core.write_new(out/'pipeline_failed.local.json',dict(status='failed',error_type=type(error).__name__,
            error=str(error),**core.FLAGS))
        raise
