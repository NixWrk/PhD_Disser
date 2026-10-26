"""Author-selected manual outer STL; shares the standard joins and FEM stages."""
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np
import trimesh


def voxel_centres_inside(surface,ct):
    """Rasterise a closed LPS surface at CT centres, preserving the native grid."""
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray,vtkPolyData
    from vtkmodules.vtkImagingStencil import vtkPolyDataToImageStencil,vtkImageStencilToImage
    from vtkmodules.util.numpy_support import numpy_to_vtk,numpy_to_vtkIdTypeArray,vtk_to_numpy
    canonical=nib.as_closest_canonical(ct)
    affine=canonical.affine
    if not np.allclose(affine[:3,:3],np.diag(np.diag(affine[:3,:3])),atol=1e-5):
        raise ValueError('Manual-surface rasterisation requires axis-aligned CT')
    points=vtkPoints();points.SetData(numpy_to_vtk(np.asarray(surface.vertices)*[-1.,-1.,1.],deep=True))
    cells=vtkCellArray()
    faces=np.column_stack([np.full(len(surface.faces),3),surface.faces]).astype(np.int64)
    cells.SetCells(len(faces),numpy_to_vtkIdTypeArray(faces.ravel(),deep=True))
    poly=vtkPolyData();poly.SetPoints(points);poly.SetPolys(cells)
    stencil=vtkPolyDataToImageStencil();stencil.SetInputData(poly)
    stencil.SetOutputOrigin(*affine[:3,3]);stencil.SetOutputSpacing(*np.diag(affine[:3,:3]))
    stencil.SetOutputWholeExtent(0,canonical.shape[0]-1,0,canonical.shape[1]-1,0,canonical.shape[2]-1)
    stencil.SetTolerance(0.);stencil.Update()
    image=vtkImageStencilToImage();image.SetInputConnection(stencil.GetOutputPort())
    image.SetInsideValue(1);image.SetOutsideValue(0);image.SetOutputScalarTypeToUnsignedChar();image.Update()
    mask=vtk_to_numpy(image.GetOutput().GetPointData().GetScalars()).reshape(canonical.shape,order='F')>0
    transform=nib.orientations.ornt_transform(nib.orientations.io_orientation(affine),nib.orientations.io_orientation(ct.affine))
    native=nib.orientations.apply_orientation(mask,transform)
    if native.shape!=ct.shape:raise ValueError('Manual rasterisation changed native CT shape')
    return native


def settings(cfg):
    import ct_fem_core as core
    value=cfg['body']
    if value['strategy']!='author_STL_LPS':raise ValueError('Explicit manual STL strategy required')
    if not value.get('provenance') or not value.get('coverage_limits_basis'):
        raise ValueError('Manual source provenance and outside-domain budget basis required')
    if value.get('units')!='mm' or value.get('frame')!='LPS':raise ValueError('Manual STL must explicitly declare LPS/mm')
    if core.sha(value['source_stl'])!=value['source_stl_sha256']:raise ValueError('Manual STL source changed')
    core.positive(cfg['surface_qc']['manual_STL_export_maximum_sampled_distance_mm'],'manual export bound')
    return value


def build(cfg,out,ct,masks):
    import ct_fem_core as core
    value=settings(cfg);source=Path(value['source_stl'])
    surface=trimesh.load(source,force='mesh');qc=core.surface_qc(surface)
    body=voxel_centres_inside(surface,ct)
    coverage={k:dict(outside_before=int(np.count_nonzero(mask & ~body)),
                    outside_after=int(np.count_nonzero(mask & ~body)),newly_lost=0,
                    total_mask_voxels=int(mask.sum())) for k,mask in masks.items()}
    audit=dict(strategy=value['strategy'],source_stl_sha256=core.sha(source),
        CT_grid_voxelisation_only=True,source_surface_modified=False,
        tissue_coverage=coverage,coverage_limits=value['maximum_outside_body_voxels'],
        coverage_limits_basis=value['coverage_limits_basis'],source_QC=qc,**core.FLAGS)
    core.write_new(out/'manual_surface_audit.local.json',audit)
    if any(c['outside_after']>value['maximum_outside_body_voxels'][k] for k,c in coverage.items()):
        raise ValueError('Tissue outside manual body exceeds declared domain budget; see manual_surface_audit.local.json')
    image=nib.Nifti1Image(body.astype(np.uint8),ct.affine);image.header.set_xyzt_units('mm')
    nib.save(image,out/'body_manual_STL_voxelisation.nii.gz')
    # This compatibility filename carries the unchanged manual source, with
    # its role stated explicitly in QC; no HU morphological closing is done.
    shutil.copyfile(source,out/'skin_regularized.stl')
    exported=trimesh.load(out/'skin_regularized.stl',force='mesh');core.surface_qc(exported)
    if core.sha(out/'skin_regularized.stl')!=value['source_stl_sha256']:
        raise ValueError('Manual source export must preserve exact STL bytes')
    qc.update(tissue_coverage=coverage,changed_voxels=0,changed_relative=0.,added_voxels=0,removed_voxels=0,
        HU_hole_fill_added_voxels=0,voxel_euler=None,body_mask_assumption='Author-selected manual Inobitec STL; voxelisation audits domain coverage only',
        topology_correction_qc=dict(applicable=False,reason='Unchanged author-selected STL; no HU contour correction'),
        export_fidelity_qc=dict(passed=True,identical_STL_bytes=True,maximum_distance_mm=0.),
        manual_source_provenance=value['provenance'],manual_source_stl_sha256=value['source_stl_sha256'])
    return exported,qc
