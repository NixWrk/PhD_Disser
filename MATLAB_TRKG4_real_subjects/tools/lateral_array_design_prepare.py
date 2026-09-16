"""Prepare finite 5 mm contacts on the fixed C01 geometry for notebook 33.09."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from scipy.io import loadmat,savemat
import refine_surface_contact_mesh as refinement

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'output/exploratory/transverse_consistency_20260914'
DEFAULT_OUT=ROOT/'output/exploratory/lateral_array_design_20260916'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def prepare(out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'plan.json').exists():raise FileExistsError('A frozen design plan already exists.')
    src=loadmat(SOURCE/'geometry_input.mat',simplify_cells=True)
    masks=loadmat(SOURCE/'geometry_masks.mat',simplify_cells=True)
    contract=json.loads((SOURCE/'input_contract.json').read_text(encoding='utf8'))
    names=list(masks['model_names']);assert names[0]=='reference' and names[2]=='variable_transverse'
    assert src['meshsha']==masks['mesh_sha256']==contract['mesh_sha256']
    nodes=np.asarray(src['nodes_mm'],float);tets=np.asarray(src['elems'],np.int32)-1
    boundary=np.asarray(src['boundary_faces'],np.int32)-1
    ref=np.asarray(masks['masks'][:,0],np.int32);selected=np.asarray(masks['masks'][:,2],np.int32)
    e=pd.read_csv(SOURCE/'source_electrodes.csv').sort_values(['L_mm'],kind='stable')
    sizes=np.asarray(sorted(e.L_mm.unique()),int)
    assert sizes.tolist()==[50,60,70,80,90,110,120,130,140]
    assert e.electrode.tolist()==['I+','V+','V-','I-']*len(sizes)
    centres=e[['x_mm','y_mm','z_mm']].to_numpy()
    refinement.np=np
    vol=refinement.volumes(nodes,tets)
    flipped=vol<0
    tets[flipped,1:3]=tets[flipped,1:3][:,::-1]
    assert np.all(refinement.volumes(nodes,tets)>0)
    before=np.array([np.abs(vol).sum(),np.abs(vol)[ref==1].sum(),np.abs(vol)[selected==1].sum()])
    original=nodes.copy()
    nodes,tets,boundary,ref,selected,history=refinement.refine(nodes,tets,boundary,ref,selected,centres,radius=4.,target=.5)
    vol=refinement.volumes(nodes,tets)
    after=np.array([vol.sum(),vol[ref==1].sum(),vol[selected==1].sum()])
    assert np.all(vol>0) and np.array_equal(nodes[:len(original)],original)
    np.testing.assert_allclose(after,before,rtol=1e-10,atol=1e-5)
    refinement.verify_boundary(tets,boundary)
    path=out/'refined_mesh.mat'
    savemat(path,dict(nodes_mm=nodes,elems=tets+1,boundary=boundary+1,reference_mask=ref[:,None],
        variable_transverse_mask=selected[:,None],electrode_centres_mm=centres,sizes_mm=sizes[:,None]),do_compression=False)
    bounds=np.asarray(masks['bounds'],float)
    axes=[np.exp(np.linspace(np.log(lo),np.log(hi),7)) for lo,hi in bounds]
    points=[]
    for i,r1 in enumerate(axes[0]):
        for j,r2 in enumerate(axes[1]):points.append(dict(point_id=f'g{i}_{j}',kind='grid',rho1=float(r1),rho2=float(r2)))
    f=np.array([.125,.375,.625,.875])
    v=[np.exp(np.log(lo)+f*np.log(hi/lo)) for lo,hi in bounds]
    for i,r1 in enumerate(v[0]):
        for j,r2 in enumerate(v[1]):points.append(dict(point_id=f'v{i}_{j}',kind='validation',rho1=float(r1),rho2=float(r2)))
    pd.DataFrame(points).to_csv(out/'points.csv',index=False)
    actual=ROOT/'output/exploratory/lateral_real_application_20260915/observations.csv'
    plan=dict(status='exploratory_hypothesis_not_validated',subject='exp02_nik',candidate='C01',
        task='select independently recorded montage sizes for baseline rho1/rho2 with fixed CT geometry',
        centre_mm=contract['centre_mm'],sizes_mm=sizes.tolist(),bounds_ohm_m=bounds.tolist(),
        grid_rho1=axes[0].tolist(),grid_rho2=axes[1].tolist(),finite_electrodes=True,diameter_mm=5.,radius_mm=2.5,
        nominal_area_mm2=float(np.pi*2.5**2),contact_ohm_m2=contract['contact_ohm_m2'],current_ampere=1.,
        electrode_count_per_measurement=4,simultaneous_all_sizes=False,refined_mesh_sha256=sha(path),
        refinement_source_sha256=sha(Path(refinement.__file__)),preparation_source_sha256=sha(Path(__file__)),
        points_sha256=sha(out/'points.csv'),observations_sha256=sha(actual),observations_source=str(actual.relative_to(ROOT)),
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [SOURCE/'geometry_input.mat',SOURCE/'geometry_masks.mat',SOURCE/'input_contract.json',SOURCE/'source_electrodes.csv',actual]},
        geometry_qc=dict(original_vertices_unchanged=True,boundary_conforming=True,positive_tetrahedra=True,
            volumes_before_mm3=before.tolist(),volumes_after_mm3=after.tolist(),source_reoriented_tetrahedra=int(flipped.sum()),
            nodes=int(len(nodes)),tetrahedra=int(len(tets)),local_edge_mm=.5,radius_of_refinement_mm=4.,rounds=history),
        acceptance=dict(area_relative_error_max=.10,centroid_error_mm_max=.75,radial_extent_mm_max=3.5,
            system_relative_residual_max=1e-7,reciprocity_abs_ohm_max=1e-7),
        limitations=['One anatomy only; exact fixed CT montage is accepted.',
            'h and all geometry fixed; two scalar resistivities only.',
            '5 mm is a new design assumption, not a measured contact diameter of the old experiment.',
            'Recorded-data fits with 5 mm are conditional scenarios, not independent validation.',
            'Contact impedance is a transferred assumption, not newly measured.',
            'Linear sensitivity noise scenarios do not define experimental noise or confidence intervals.',
            'No simultaneous multi-array coupling and no pulse protocol optimization in this study.'])
    (out/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('Prepared',out,'nodes',len(nodes),'tetrahedra',len(tets),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT_OUT);a=p.parse_args();prepare(a.out)
