"""Selected CT -> TotalSegmentator v2 -> five native anatomical STL surfaces.

This stage preserves all segmented components and cavity interfaces. It does
not add arms, place electrodes, select a FEM domain, or generate a volume mesh.
"""
from pathlib import Path
import argparse
import importlib.metadata
import subprocess
import sys

import nibabel as nib
import numpy as np
from skimage.measure import marching_cubes
import trimesh

import ct_fem_core as core
from ct_fem_ct_pipeline import resolve, verify_segmentation

ROLES = ['body','lung','bone','heart','fat']


def export_mask(mask_path, ct_path, target):
    ct = nib.load(ct_path); image = nib.load(mask_path)
    if (image.shape != ct.shape or not np.array_equal(image.affine,ct.affine) or
            image.header.get_xyzt_units()[0] != 'mm' or ct.header.get_xyzt_units()[0] != 'mm'):
        raise ValueError('STL source must retain the native CT grid in mm')
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3 or not np.isin(data,[0,1]).all() or not data.any():
        raise ValueError('Nonempty finite binary 3D segmentation required')
    vertices,faces,_,_ = marching_cubes(np.pad(data,1).astype(np.uint8),level=.49)
    vertices = nib.affines.apply_affine(image.affine,vertices-1)*[-1.,-1.,1.]
    surface = trimesh.Trimesh(vertices,faces,process=True)
    # Preserve inner cavity orientation; a material may have many disconnected
    # pieces. Do not invert all cavity shells independently into solid objects.
    surface.fix_normals()
    if not surface.is_volume or not np.isfinite(surface.vertices).all():
        raise ValueError('Segmented surface is not a closed consistently oriented positive volume')
    surface.export(target)
    exported = trimesh.load(target,force='mesh',process=False)
    if len(surface.faces) != len(exported.faces): raise ValueError('STL face count changed')
    maximum = 0.
    for start in range(0,len(surface.faces),100000):
        original = surface.vertices[surface.faces[start:start+100000]]
        saved = exported.vertices[exported.faces[start:start+100000]]
        # Same ordered triangles: the maximum vertex displacement also bounds
        # corresponding barycentric points; no surface subsampling is used.
        maximum = max(maximum,float(np.linalg.norm(original-saved,axis=2).max()))
    if maximum > .001: raise ValueError('STL quantisation exceeds 0.001mm')
    return dict(mask_sha256=core.sha(mask_path),STL_sha256=core.sha(target),
        vertices=len(surface.vertices),triangles=len(surface.faces),
        connected_surface_components=len(surface.split(only_watertight=False)),
        watertight=bool(surface.is_watertight),winding_consistent=bool(surface.is_winding_consistent),
        signed_surface_volume_ml=float(surface.volume/1000),mask_voxels=int(np.count_nonzero(data)),
        mask_volume_ml=float(np.count_nonzero(data)*abs(np.linalg.det(image.affine[:3,:3]))/1000),
        maximum_export_vertex_displacement_mm=maximum,ordered_triangles_checked=True,
        surface_level=.49,coordinates='LPS',units='mm',all_components_and_cavity_interfaces_preserved=True)


def prepare(config_path, out):
    config_path=Path(config_path).resolve();out=Path(out).resolve();request=core.read(config_path)
    if request.get('schema')!='ct_fem_ct_pipeline_v1': raise ValueError('Explicit CT/TotalSegmentator configuration required')
    geometry_path=Path(resolve(request['geometry_configuration'],config_path.parent));cfg=core.read(geometry_path)
    ct=resolve(cfg['inputs']['ct'],geometry_path.parent)
    if cfg['inputs'].get('ct_intensity_units')!='HU':raise ValueError('Declare CT intensities as HU')
    runtime=dict(request['segmentation'])
    for key in ['python','weights_directory','config_file']:
        if runtime.get(key):runtime[key]=resolve(runtime[key],config_path.parent)
    core.positive(runtime['maximum_seconds'],'segmentation.maximum_seconds')
    reuse=request.get('reuse_tissue_segmentation')
    sources={str(p):core.sha(p) for p in [config_path,geometry_path,Path(ct),Path(__file__),
        Path(core.__file__),Path(__file__).with_name('ct_fem_ct_pipeline.py'),
        Path(__file__).with_name('ct_fem_totalsegmentator.py')]}
    if reuse:
        reuse=Path(resolve(reuse,config_path.parent))
        _,bound=verify_segmentation(reuse,ct,roles=ROLES[1:]);sources.update(bound)
    if out.exists():raise ValueError('CT-to-STL stage requires a fresh output directory')
    out.mkdir(parents=True)
    try:
        inference=out/'segmentation_input.local.json'
        core.write_new(inference,dict(ct=ct,max_ct_voxels=cfg['resources']['max_ct_voxels'],segmentation=runtime))
        worker=Path(__file__).with_name('ct_fem_totalsegmentator.py')
        cmd=[runtime['python'],'-X','utf8',str(worker),'--config',str(inference),'--out',str(out/'segmentation'),
             '--tasks',*(['body'] if reuse else ['body','total','tissue_types'])]
        with (out/'segmentation.log').open('x',encoding='utf-8') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=runtime['maximum_seconds'],check=True)
        _,bound=verify_segmentation(out/'segmentation',ct,roles=['body'] if reuse else ROLES);sources.update(bound)
        products={}
        for role in ROLES:
            folder=out/'segmentation' if role=='body' or not reuse else reuse
            print('Native TotalSegmentator STL:',role,flush=True)
            products[role]=export_mask(folder/(role+'.nii.gz'),ct,out/(role+'.stl'))
        core.verify_sources(sources)
        versions={name:importlib.metadata.version(name) for name in ['numpy','nibabel','trimesh','scikit-image']}
        value=dict(schema='ct_totalsegmentator_STL_v1',stage='segmented_STL',status='automatic_pending_anatomical_review',
            subject_alias=cfg['subject_alias'],segmentation_backend='TotalSegmentator',
            segmentation_version=runtime['version'],ct_sha256=core.sha(ct),sources=sources,
            products=products,runtime=versions,reused_tissue_masks=bool(reuse),
            morphology_applied=False,domain_clipping_applied=False,synthetic_arms_added=False,
            electrodes_placed=False,volume_mesh_generated=False,**core.FLAGS)
        core.write_new(out/'segmented_STL.local.json',value)
        return value
    except BaseException as error:
        core.write_new(out/'segmentation_STL_failed.local.json',dict(status='failed',error_type=type(error).__name__,
            error=str(error),**core.FLAGS))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();result=prepare(args.config,args.out)
    print(result['status'],flush=True)
