"""Two-material geometric reduction study, consumed by notebook 20.13.

Geometry is fitted to CT-derived regions only. Electrical observations never
enter the geometry fit. All outputs are exploratory and private.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import minimize_scalar

from compare_curved_two_tissue import analytic_curves, sha

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'output/exploratory/geometry_reduction_20260910'
SOURCE = ROOT / 'output/two_tissue_auditfix_20260908'
MODEL_LABELS = {
    'reference': 'КТ: исходная граница лёгкого',
    'plane': 'Плоская граница в объёме тела',
    'ellipsoid': 'Эллипсоид по объёму и моментам',
    'ellipsoid_depth': 'Эллипсоид с привязкой по глубине',
    'planar_surface_h': 'Аналитика: постоянная глубина',
    'adapted_mean_h': 'Аналитика: средняя глубина по паре',
    'adapted_mean_kernel': 'Аналитика: среднее двухслойное ядро',
    'adapted_finite_interval': 'Аналитика: конечная толщина лёгкого',
}


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def ellipsoid_interval(origin, direction, centre, quadratic):
    x = np.asarray(origin) - centre
    a = direction @ quadratic @ direction
    b = 2 * (x @ quadratic @ direction)
    c = np.einsum('...i,ij,...j->...', x, quadratic, x) - 1
    discriminant = b*b - 4*a*c
    root = np.sqrt(np.maximum(discriminant, 0))
    return np.where(discriminant >= 0, (-b-root)/(2*a), np.nan), np.where(discriminant >= 0, (-b+root)/(2*a), np.nan)


def prepare(out):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    out.mkdir(parents=True, exist_ok=True)
    g = json.loads((SOURCE / 'geometry.json').read_text(encoding='utf-8'))
    c = json.loads((SOURCE / 'comparison_summary.json').read_text(encoding='utf-8'))
    for name, checksum in c['artifact_sha256'].items():
        assert sha(SOURCE / name) == checksum, f'Stale source artifact: {name}'
    sample = loadmat(SOURCE / 'mesh_samples.mat', simplify_cells=True)
    xyz = np.asarray(sample['centres_mm'])
    volume = sample['element_volume_mm3']
    closed = loadmat(SOURCE / 'lung_masks.mat', simplify_cells=True)['closed_lung_mask'].astype(bool)
    centre, inward, axis = [np.array(g[k]) for k in ('centre_mm', 'inward', 'axis')]
    transverse = np.cross(axis, inward)
    frame = np.column_stack([axis, transverse, inward])
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)
    stl = ROOT / 'data/nik/stl/lungs_solid_v3_volume_fill.stl'
    assert sha(stl) == g['lung_stl_sha256']
    reader = vtk.vtkSTLReader(); reader.SetFileName(str(stl)); reader.Update()
    clean = vtk.vtkCleanPolyData(); clean.SetInputData(reader.GetOutput()); clean.Update()
    surface = clean.GetOutput()
    assert vtk.vtkSelectEnclosedPoints.IsSurfaceClosed(surface), 'Lung surface is not closed'
    depth = (xyz-centre) @ inward
    split_depth = float((depth[closed].min()+depth[closed].max())/2)
    selected = closed & (depth <= split_depth)
    assert 0 < selected.sum() < closed.sum()
    other = closed & ~selected
    w = volume[selected]; target_volume = w.sum()
    ell_centre = np.average(xyz[selected], axis=0, weights=w)
    centred = xyz[selected] - ell_centre
    covariance = (centred.T * w) @ centred / target_volume
    eigenvalues, rotation = np.linalg.eigh(covariance)
    radii = np.sqrt(5 * eigenvalues)
    radii *= (target_volume / (4*np.pi*np.prod(radii)/3))**(1/3)
    quadratic = (rotation / radii**2) @ rotation.T
    tree = vtk.vtkOBBTree(); tree.SetDataSet(surface); tree.BuildLocator()

    def crossings(origin):
        hits = vtk.vtkPoints()
        tree.IntersectWithLine(origin-100*inward, origin+600*inward, hits, None)
        if not hits.GetNumberOfPoints(): return []
        depths = (vtk_to_numpy(hits.GetData()) - origin) @ inward
        return sorted(float(x) for x in depths if x > 0)

    centre_hits = crossings(centre)
    assert len(centre_hits) >= 2
    entry = centre_hits[0]
    ell_entry, _ = ellipsoid_interval(centre, inward, ell_centre, quadratic)
    assert np.isfinite(ell_entry)
    shift = float(entry-ell_entry)
    anchored_centre = ell_centre + shift*inward
    np.testing.assert_allclose(ellipsoid_interval(centre,inward,anchored_centre,quadratic)[0],entry,atol=1e-9)

    def occupancy(ec):
        q = xyz-ec
        return np.einsum('ij,jk,ik->i',q,quadratic,q) <= 1

    masks = np.column_stack([closed, (((xyz-centre)@inward)>=entry)|other,
                             occupancy(ell_centre)|other, occupancy(anchored_centre)|other])
    model_names = ['reference','plane','ellipsoid','ellipsoid_depth']
    rows = []
    for s in np.arange(-80,81,10):
        for t in np.arange(-60,61,10):
            origin=centre+s*axis+t*transverse
            hits=crossings(origin)
            e1,x1=ellipsoid_interval(origin,inward,ell_centre,quadratic)
            e2,x2=ellipsoid_interval(origin,inward,anchored_centre,quadratic)
            rows.append([s,t,hits[0] if len(hits)>=2 else np.nan,hits[-1] if len(hits)>=2 else np.nan,float(e1),float(x1),float(e2),float(x2)])
    pd.DataFrame(rows,columns=['s_mm','t_mm','entry_reference_mm','exit_reference_mm','entry_ellipsoid_mm','exit_ellipsoid_mm','entry_anchored_mm','exit_anchored_mm']).to_csv(out/'ray_profiles.csv',index=False)
    # Same grid for all models. Holdouts are solved independently, not spline knots.
    knots=np.geomspace(.5,32,19)
    truth=np.array([2.8,g['rho_lung_ohm_m']/g['rho_soft_ohm_m'],8.6])
    check=np.sqrt(knots[:-1]*knots[1:])[::3]
    ratios=np.unique(np.r_[knots,truth,check,1.])
    savemat(out/'geometry_masks.mat',{'masks':masks,'model_names':np.array(model_names,dtype=object),
        'mesh_sha256':sample['mesh_sha256'],'ratios':ratios,'knots':knots,'truth_ratios':truth,'check_ratios':check})
    geometry={'status':'exploratory_hypothesis_not_validated','model_names':model_names,'reference_geometry':g,
        'mesh_sha256':str(sample['mesh_sha256']),'source_summary_sha256':sha(SOURCE/'comparison_summary.json'),
        'source_fem_contract_sha256':sha(SOURCE/'fem_contract.json'),
        'source_geometry_sha256':sha(SOURCE/'geometry.json'),'source_lung_mask_sha256':sha(SOURCE/'lung_masks.mat'),
        'geometry_source_sha256':sha(__file__),'masks_sha256':sha(out/'geometry_masks.mat'),
        'frame_columns_s_t_d':frame.tolist(),'ellipsoid_centre_mm':ell_centre.tolist(),
        'ellipsoid_radii_mm':radii.tolist(),'ellipsoid_rotation':rotation.tolist(),
        'anchored_centre_mm':anchored_centre.tolist(),'anchoring_shift_mm':shift,'central_entry_depth_mm':entry,
        'geometric_split_depth_mm':split_depth,'selected_lung_volume_litres':float(target_volume/1e6),'model_lung_volumes_litres':dict(zip(model_names,(volume@masks/1e6).tolist())),
        'knots':knots.tolist(),'truth_ratios':truth.tolist(),'check_ratios':check.tolist(),
        'assumptions':['Two real, scalar resistivities; every nonlung tissue is background.',
        'All spatial models share body mesh, electrode nodes and the retained distal portion of the lung mask.',
        'The connected two-lung mask is partitioned at the midrange depth along the assembly inward axis; this is a geometric split, not anatomical segmentation.',
        'The planar model replaces the proximal region by an inward half-space intersected with the body.',
        'Ellipsoid orientation and axis ratios derive from volume-weighted moments; total continuous volume is matched before body clipping.',
        'Depth anchoring translates the ellipsoid only; it does not fit electrical data.',
        'Same-mesh PEM comparisons isolate the declared mask change, but do not establish mesh convergence or anatomical validation.']}
    write_json(out/'geometry.json',geometry)
    print(json.dumps({k:geometry[k] for k in ['model_names','ellipsoid_radii_mm','anchoring_shift_mm','model_lung_volumes_litres']},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare']);parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args();prepare(args.out.resolve())
