"""Mesh and material preparation for the shared Georg CT model (CEM/EIDORS)."""
from pathlib import Path
import argparse
import time
import sys
import decimal

import gmsh
import numpy as np
import trimesh
from scipy.io import savemat
from scipy import ndimage as ndi
import nibabel as nib

from scorpion_georg_fem_prepare import sha, read, write, load_mesh, map_fractions, MATERIALS
import refine_surface_contact_mesh as refine
from georg_ct_model import PIPE, OUT, SEG
from scorpion_georg_geometry_revision import arm_groups, longitudinal_centres

refine.np = np


def project(surface, points):
    return surface.nearest.on_surface(np.asarray(points))[0]


def scenarios(out):
    surface = trimesh.load(out/'body_embedded_arms_370mm.stl', force='mesh')
    rows = []
    keys = ['I_plus', 'V_plus', 'V_minus', 'I_minus']
    for n in range(2, 8):
        data = read(out/'landmarks'/f'georg_tepc_{n}.json')
        xyz = np.array([data['preliminary_electrode_centers_xyz_mm'][k] for k in keys])
        _, _, ff = surface.nearest.on_surface(xyz)
        normals = surface.face_normals[ff]
        axes = []
        for normal in normals:
            a = np.array([0., 0., 1.]); a -= normal*(a@normal)
            a /= np.linalg.norm(a); b = np.cross(normal, a)
            axes.append([a, b])
        axes = np.asarray(axes)
        rows.append(dict(id=f'TEPC{n}_nominal', family='TEPC', montage=n,
            recording=f'{n+10:03d}', perturbation='nominal', shift_mm=0.,
            groups_mm=[[x.tolist()] for x in xyz], drive=[1,4], measure=[2,3]))
        # Scenario amplitudes are not confidence intervals for manual placement.
        for delta in [1., 3., 5., 10.]:
            for axis in [0,1]:
                for sign in [-1,1]:
                    moved = project(surface, xyz+sign*delta*axes[:,axis])
                    rows.append(dict(id=f'TEPC{n}_common_{delta:g}_{axis}_{sign}',
                        family='TEPC', montage=n, recording=f'{n+10:03d}',
                        perturbation='all contacts along their local tangent', tangent_axis=axis,
                        shift_mm=sign*delta, actual_displacement_mm=np.linalg.norm(moved-xyz,axis=1).tolist(),
                        groups_mm=[[x.tolist()] for x in moved], drive=[1,4], measure=[2,3]))
        for contact in range(4):
            for sign in [-1,1]:
                moved = xyz.copy(); moved[contact] = project(surface,
                    [xyz[contact]+sign*5*axes[contact,0]])[0]
                rows.append(dict(id=f'TEPC{n}_individual_{contact}_{sign}', family='TEPC', montage=n,
                    recording=f'{n+10:03d}', perturbation='one contact along local longitudinal tangent',
                    shifted_contact=keys[contact], shift_mm=sign*5.,
                    actual_displacement_mm=np.linalg.norm(moved-xyz,axis=1).tolist(),
                    groups_mm=[[x.tolist()] for x in moved], drive=[1,4], measure=[2,3]))
    geometry = read(out/'geometry.local.json')
    original = read(PIPE/'output/exploratory/scorpion_georg_fem_20261008/arms_370mm.json')['arm_extension']
    for name, counts in [('1x1',(1,1)), ('1x4',(1,4)), ('2x2',(2,2))]:
        groups = arm_groups(surface, original, geometry['joins'], *counts)
        if name=='1x1':
            full=arm_groups(surface,original,geometry['joins'],1,4)
            groups=[g[1:2] if i in (1,2) else g for i,g in enumerate(full)]
        rows.append(dict(id=f'TTRKG_{name}', family='TTRKG', montage=name,
            perturbation='nominal cylinder scenario', shift_mm=0., groups_mm=groups,
            drive=[1,4], measure=[2,3], pose_is_measured=False,
            single_U_top_is_unconfirmed_scenario=name=='1x1'))
    # These are reference montages only. Their centre has NOT been localized.
    for size, xyz in zip([60,80,100,120,140],longitudinal_centres(surface,0.,-210.)):
        groups = arm_groups(surface, original, geometry['joins'],1,4)+[[x] for x in xyz]
        for kind, drive, measure in [('lateral',[5,8],[6,7]),('ttrkg',[1,4],[2,3])]:
            rows.append(dict(id=f'reference_{size}_{kind}', family='reference', montage=size,
                perturbation='illustrative longitudinal centre, not a localization', shift_mm=0.,
                groups_mm=groups, drive=drive, measure=measure, pose_is_measured=False))
    fields=sorted({key for row in rows for key in row})
    rows=[{key:row.get(key) for key in fields} for row in rows]
    for row in rows: row['montage']=str(row['montage'])
    write(out/'fem_contacts.local.json', dict(diameter_mm=5., nominal_area_mm2=float(np.pi*2.5**2),
        actual_contact_diameter_is_known=False, scenarios=rows,
        passive_contacts='all groups in each specified montage; absent physical groups are not invented',
        unrecorded_simultaneous_electrodes_are_unknown=True,
        uncertainty_amplitudes_mm=[1,3,5,10], amplitudes_are_confidence_intervals=False))
    return rows


def mesh(out, size):
    source = out/'body_embedded_arms_370mm.stl'; dest=out/f'mesh_h{size:g}.msh'
    surface=trimesh.load(source,force='mesh');surface_path=source
    ct=nib.load(SEG/'static_ct.nii.gz');body=ndi.binary_erosion(np.asanyarray(nib.load(out/'body_mask.nii.gz').dataobj)>0,iterations=2)
    axes=[np.arange(low+size,high-size/2,size) for low,high in surface.bounds.T]
    xyz=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
    vox=nib.affines.apply_affine(np.linalg.inv(ct.affine),xyz*[-1,-1,1])
    inside=ndi.map_coordinates(body.astype(np.uint8),vox.T,order=0,mode='constant',cval=0)>0
    geometry=read(out/'geometry.local.json')
    for join in geometry['joins'].values():
        lo,hi=sorted([join['x_mm'],join['far_x_mm']]);yz=np.asarray(join['centre_yz_mm'])
        inside|=(xyz[:,0]>lo+2)&(xyz[:,0]<hi-2)&(np.linalg.norm(xyz[:,1:]-yz,axis=1)<join['radius_mm']-2)
    interior=xyz[inside]
    print('Regular interior points',len(interior),'surface faces',len(surface.faces),flush=True)
    start=time.perf_counter(); gmsh.initialize()
    try:
        gmsh.option.setNumber('General.NumThreads',6)
        gmsh.option.setNumber('General.Verbosity',4)
        # Recover the exact polyhedral boundary first. Longest-edge bisection
        # below controls resolution without repeated sliver circumcentre
        # insertion, which failed on this particular imported CT polyhedron.
        gmsh.option.setNumber('Mesh.Algorithm3D',3)
        gmsh.option.setNumber('Mesh.MeshSizeMin',1)
        gmsh.option.setNumber('Mesh.MeshSizeMax',size)
        gmsh.option.setNumber('Mesh.MeshSizeFromPoints',0)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',0)
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary',0)
        gmsh.option.setNumber('Mesh.Optimize',0)
        gmsh.option.setNumber('Mesh.MshFileVersion',2.2)
        gmsh.model.add('Georg_CT_current');print('Importing checked exterior',flush=True)
        gmsh.merge(str(surface_path.resolve()));print('Constructing the volume',flush=True)
        loop=gmsh.model.geo.addSurfaceLoop([tag for _,tag in gmsh.model.getEntities(2)])
        volume=gmsh.model.geo.addVolume([loop])
        seeds=[gmsh.model.geo.addPoint(*point) for point in interior]
        gmsh.model.geo.synchronize();gmsh.model.mesh.embed(0,seeds,3,volume)
        gmsh.model.addPhysicalGroup(3,[volume],1)
        constant=gmsh.model.mesh.field.add('MathEval')
        gmsh.model.mesh.field.setString(constant,'F',str(size))
        gmsh.model.mesh.field.setAsBackgroundMesh(constant)
        print('Generating tetrahedra',flush=True)
        gmsh.model.mesh.generate(3); gmsh.write(str(dest.resolve()))
    finally: gmsh.finalize()
    write(out/f'mesher_h{size:g}.local.json',dict(source_sha256=sha(source),mesh_sha256=sha(dest),
        target_interior_size_mm=size,regular_interior_points=len(interior),
        original_polyhedral_surface_preserved=True,
        seconds=time.perf_counter()-start, algorithm='Gmsh initial constrained Delaunay with regular interior points',
        accepted=False, requires_external_mesh_and_cem_checks=True))


def mesh_tetgen(out,size,deps=None,candidate=None,steiner_limit=500000):
    if deps is not None:sys.path.insert(0,str(Path(deps).resolve()))
    import tetgen
    source=out/'body_embedded_arms_370mm.stl';dest=out/f'mesh_h{size:g}.msh'
    start=time.perf_counter()
    if candidate is not None:
        arrays=np.load(candidate);nodes=arrays['nodes'];elems=arrays['elems']
    else:
        surface=trimesh.load(source,force='mesh');engine=tetgen.TetGen(surface.vertices,surface.faces.astype(np.int32))
        nodes,elems,*_=engine.tetrahedralize(plc=True,quality=True,minratio=2.,mindihedral=0.,
            fixedvolume=True,maxvolume=size**3/6,nobisect=True,steinerleft=steiner_limit,verbose=1)
    if dest.exists():
        history=out/'rejected_gmsh';history.mkdir(exist_ok=True)
        if not (history/dest.name).exists():(history/dest.name).write_bytes(dest.read_bytes())
    with dest.open('w',encoding='ascii',newline='\n') as stream:
        stream.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n'+str(len(nodes))+'\n')
        for n,xyz in enumerate(nodes,1):stream.write(f'{n} {xyz[0]:.17g} {xyz[1]:.17g} {xyz[2]:.17g}\n')
        stream.write('$EndNodes\n$Elements\n'+str(len(elems))+'\n')
        for n,indices in enumerate(elems,1):stream.write(f'{n} 4 2 1 1 '+' '.join(str(int(i)+1) for i in indices)+'\n')
        stream.write('$EndElements\n')
    write(out/f'mesher_h{size:g}.local.json',dict(source_sha256=sha(source),mesh_sha256=sha(dest),
        algorithm='TetGen constrained Delaunay',tetgen_version=tetgen.__version__,
        min_radius_edge_ratio=2.,maximum_volume_mm3=size**3/6,boundary_bisection=False,
        maximum_steiner_points=steiner_limit,
        nominal_resolution_mm=size,nodes=len(nodes),tetrahedra=len(elems),
        seconds=time.perf_counter()-start,candidate_file=None if candidate is None else str(candidate),
        candidate_sha256=None if candidate is None else sha(candidate),
        accepted=False,requires_external_mesh_and_cem_checks=True))
    print('TetGen candidate written',dest,len(nodes),len(elems),flush=True)


def validate_tetrahedra(nodes,tets,volume):
    if not np.isfinite(volume).all() or np.any(volume<=0):
        raise ValueError('Nonpositive or nonfinite refined tetrahedron')
    ids=np.flatnonzero(volume<=1e-12);errors=[]
    # An absolute volume cutoff rejects valid dyadic children of flat boundary
    # cells. Retain them only after an exact-coordinate 80-digit determinant
    # confirms the positive sign and agrees with the double calculation.
    with decimal.localcontext() as context:
        context.prec=80
        for index in ids:
            p=[[decimal.Decimal.from_float(float(x)) for x in row] for row in nodes[tets[index]]]
            a=[[row[k]-p[0][k] for k in range(3)] for row in p[1:]]
            exact=(a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])-a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])+a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]))/6
            if exact<=0:raise ValueError('Exact-coordinate determinant is nonpositive')
            errors.append(abs(float(exact)-volume[index])/float(exact))
    maximum=max(errors,default=0.)
    if maximum>1e-6:raise ValueError('Tiny cell determinant is numerically unresolved')
    return dict(tiny_cells_verified=len(ids),decimal_precision=80,
        maximum_relative_volume_error=maximum,relative_volume_tolerance=1e-6,
        no_elements_removed=True)


def prepare(out, seg, size):
    target=out/f'fem_h{size:g}'; target.mkdir(exist_ok=True)
    rows=scenarios(out)
    nodes,tets,boundary=load_mesh(out/f'mesh_h{size:g}.msh',out/'body_embedded_arms_370mm.stl')
    centres=np.unique(np.vstack([np.vstack(r['groups_mm']) for r in rows]),axis=0)
    parent_volume=float(np.abs(refine.volumes(nodes,tets)).sum())
    nodes,tets,boundary,_,_,history=refine.refine(nodes,tets,boundary,
        np.zeros(len(tets),np.uint8),np.zeros(len(tets),np.uint8),centres,
        radius=5.,target=.7,max_rounds=16)
    v=refine.volumes(nodes,tets); reverse=v<0; tets[reverse]=tets[reverse][:,[1,0,2,3]]
    v=refine.volumes(nodes,tets)
    tiny_qc=validate_tetrahedra(nodes,tets,v)
    if abs(v.sum()/parent_volume-1)>1e-10:raise ValueError('Refinement changed total volume')
    write(target/'tiny_tetrahedra_qc.local.json',tiny_qc)
    refine.verify_boundary(tets,boundary)
    savemat(target/'mesh.mat',dict(nodes_mm=nodes,elems=tets+1,boundary=boundary+1),do_compression=False)
    geo=map_fractions(nodes,tets,seg,target)
    registry=read(PIPE/'configs/tissue_literature_v2.json')
    refs={key:registry['tissues'][name]['gabriel_at_frequency']['100000']['rho_ohm_m'] for key,name in
        [('soft','Muscle'),('lung','Lung (Inflated)'),('bone','Bone (Cortical)'),('heart','Heart Muscle'),('fat','Fat')]}
    write(target/'reference_materials.local.json',dict(rho_ohm_m=[refs[x] for x in ['soft','lung','bone','heart','fat']],
        materials=MATERIALS,frequency_hz=100000,registry_sha256=sha(PIPE/'configs/tissue_literature_v2.json'),
        nominal_rationale='frequency-specific Gabriel references, not fitted or interval midpoints',
        soft_background_as_muscle_is_assumption=True, whole_heart_as_myocardium_is_assumption=True,
        whole_bone_as_cortical_is_assumption=True))
    files=[target/'mesh.mat',target/'fractions_q256.mat',target/'fractions_q1024.mat',
        out/'fem_contacts.local.json',out/'geometry.local.json',out/'contour_qc.local.json',
        out/'body_embedded_arms_370mm.stl',out/'centre_transfer.local.json',target/'reference_materials.local.json',
        *(seg/(name+'.nii.gz') for name in ['static_ct','heart','lung','bone','fat'])]
    write(target/'plan.local.json',dict(schema='georg_shared_ct_fem_v1',subject_alias='georg',
        model_role='common geometry and material partition; electrical values remain scenarios',
        frequency_hz=100000,arm_circumference_mm=370.,material_order=MATERIALS,geometry=geo,
        nodes=len(nodes),tetrahedra=len(tets),volume_ml=float(v.sum()/1000),
        min_tetrahedron_volume_mm3=float(v.min()),all_tetrahedra_positive=True,
        tiny_tetrahedra_qc=tiny_qc,refinement_preserves_total_volume=True,
        conforming_boundary_verified=True,contact_edge_target_mm=.7,refinement_history=history,
        interior_point_spacing_mm=size,
        contact_impedance_ohm_m2=2.5e-7*.05/(np.pi*.01**2/4),contact_impedance_is_measured=False,
        contacts_file=str((out/'fem_contacts.local.json').resolve()),
        sources=[dict(path=str(p.resolve()),sha256=sha(p)) for p in files],
        code_sha256=sha(__file__),independent_physical_validation=False,
        integration_rules=[256,1024],interior_target_mm=size))
    print('Prepared',target,len(nodes),'nodes',len(tets),'tetrahedra',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--segmentation',type=Path,default=SEG)
    p.add_argument('--size',type=float,default=8.)
    p.add_argument('--stage',choices=['mesh','prepare','contacts'],required=True)
    p.add_argument('--mesher',choices=['tetgen','gmsh'],default='tetgen')
    p.add_argument('--mesh-deps',type=Path,default=OUT/'python_mesh_deps')
    p.add_argument('--candidate',type=Path,help='Explicit previously generated TetGen array snapshot')
    p.add_argument('--steiner-limit',type=int,default=500000)
    a=p.parse_args()
    if a.stage=='mesh':
        if a.mesher=='tetgen':mesh_tetgen(a.out,a.size,a.mesh_deps,a.candidate,a.steiner_limit)
        else:mesh(a.out,a.size)
    if a.stage=='prepare':prepare(a.out,a.segmentation,a.size)
    if a.stage=='contacts':scenarios(a.out)


if __name__=='__main__':main()
