"""One C01 contract for all baseline geometric reductions; source results stay immutable."""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
import pandas as pd
from scipy.io import loadmat,savemat
from geometry_reduction_study import ellipsoid_interval
from transverse_geometry_study import read_surface
from compare_curved_two_tissue import classify_lung

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'output/exploratory/transverse_consistency_20260914'
HALF=ROOT/'output/exploratory/halfspace_bem_20260914'
OUT=ROOT/'output/exploratory/c01_baseline_models_20260914'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def prepare():
    started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'geometry.json').exists():raise FileExistsError('Frozen C01 geometry exists')
    g=loadmat(SOURCE/'geometry_input.mat',simplify_cells=True)
    contract=json.loads((SOURCE/'input_contract.json').read_text('utf-8'))
    base=json.loads((SOURCE/'geometry.json').read_text('utf-8'))
    xyz=g['centres_mm'];vol=g['volume_mm3'];centre=g['centre'];frame=g['frame'];inward=frame[:,2]
    np.testing.assert_allclose(centre,contract['centre_mm'],atol=1e-12)
    np.testing.assert_allclose(frame,contract['frame_columns_s_t_d'],atol=1e-12)
    m=loadmat(SOURCE/'geometry_masks.mat',simplify_cells=True)
    reference=m['masks'][:,0].astype(bool)
    lung=ROOT/'data/nik/stl/lungs_solid_v3_volume_fill.stl'
    closed,extended=classify_lung(xyz,lung,centre,frame[:,0],inward)
    np.testing.assert_array_equal(closed,reference)
    profile=pd.read_csv(SOURCE/'central_profile.csv');central=profile.loc[profile.s_mm==0].iloc[0]
    crossings=np.array(json.loads(central.lung_crossings_d_mm))
    assert len(crossings)>=2 and len(crossings)%2==0
    entry=float(crossings[0]);depth=(xyz-centre)@inward
    split=(depth[reference].min()+depth[reference].max())/2
    selected=reference&(depth<=split);other=reference&~selected
    w=vol[selected];volume=w.sum();ec=np.average(xyz[selected],weights=w,axis=0)
    delta=xyz[selected]-ec;cov=(delta.T*w)@delta/volume
    val,rotation=np.linalg.eigh(cov);radii=np.sqrt(5*val)
    radii*=(volume/(4*np.pi*np.prod(radii)/3))**(1/3)
    quadratic=(rotation/radii**2)@rotation.T
    ell_entry,_=ellipsoid_interval(centre,inward,ec,quadratic)
    if not np.isfinite(ell_entry):raise ValueError('C01 central ray misses moment ellipsoid; anchoring rule undefined')
    shift=float(entry-ell_entry);anchored=ec+shift*inward
    np.testing.assert_allclose(ellipsoid_interval(centre,inward,anchored,quadratic)[0],entry,atol=1e-9)
    def occupied(c):
        d=xyz-c;return np.einsum('ij,jk,ik->i',d,quadratic,d)<=1
    names=['plane','ellipsoid','ellipsoid_depth','extended_lung']
    masks=np.column_stack([(depth>=entry)|other,occupied(ec)|other,occupied(anchored)|other,extended])
    bounds=np.asarray(base['bounds_ohm_m'])
    oldcases=pd.read_csv(ROOT/'output/exploratory/geometry_validation_20260910/synthetic_cases.csv')
    truth=np.unique(np.r_[np.array(base['truth']),oldcases[['rho1','rho2']].to_numpy()],axis=0)
    knots=np.array(base['knots']);checks=np.array(base['checks'])
    ratios=np.unique(np.r_[knots,checks,1.,4.,4*np.exp(-.001),4*np.exp(.001)])
    savemat(OUT/'geometry_masks.mat',dict(masks=masks,model_names=np.array(names,dtype=object),ratios=ratios,
        knots=knots,check_ratios=checks,truth=truth,bounds=bounds,mesh_sha256=contract['mesh_sha256']))
    pd.DataFrame(truth,columns=['rho1','rho2']).rename_axis('case').reset_index().to_csv(OUT/'synthetic_cases.csv',index=False)
    observations=pd.read_csv(ROOT/'data/nik/experimental/nik_breath_hold_levels.csv')
    obs=observations[observations.is_independent==1].sort_values('L_mm')[['L_mm','Z_inhale_hold_ohm']]
    np.testing.assert_array_equal(obs.L_mm,g['sizes']);obs.to_csv(OUT/'experimental_observations.csv',index=False)
    import vtk
    fn=vtk.vtkImplicitPolyDataDistance();fn.SetInput(read_surface(lung))
    nearest=abs(fn.EvaluateFunction(centre))
    inputs=[SOURCE/n for n in ['geometry_input.mat','geometry.json','geometry_masks.mat','input_contract.json','contacts.csv','pem_library.csv','cem_baseline.csv','central_profile.csv','prepared_fem.mat']]
    inputs += [HALF/n for n in ['experiment_contract.json','library_m3h.csv','library_m4h.csv','electrodes_projected.csv']]
    inputs += [lung,ROOT/'output/exploratory/geometry_validation_20260910/synthetic_cases.csv',ROOT/'data/nik/experimental/nik_breath_hold_levels.csv']
    write(OUT/'geometry.json',dict(source_prepared_sha256=sha(SOURCE/'prepared_fem.mat'),candidate='C01',centre_mm=centre.tolist(),frame_columns_s_t_d=frame.tolist(),
      mesh_sha256=contract['mesh_sha256'],h_centre_mm=float(central.h_mm),skin_centre_d_mm=float(central.skin_d_mm),
      central_entry_d_mm=entry,nearest_lung_distance_mm=float(nearest),model_names=names,
      split_d_mm=float(split),ellipsoid_centre_mm=ec.tolist(),ellipsoid_radii_mm=radii.tolist(),
      ellipsoid_rotation=rotation.tolist(),anchored_centre_mm=anchored.tolist(),anchoring_shift_mm=shift,
      bounds_ohm_m=bounds.tolist(),knots=knots.tolist(),checks=checks.tolist(),truth=truth.tolist(),
      sizes_mm=g['sizes'].tolist(),masks_sha256=sha(OUT/'geometry_masks.mat'),
      input_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in inputs},source_sha256=sha(Path(__file__)),
      preparation_seconds=time.perf_counter()-started,scope='baseline only; known CT h; 65 known-rho states and real inhale observations',
      assumptions=['All nonlung volumes share scalar rho1; lung rho2; no anatomy or montage refit.',
       'Proximal/distal split is a geometric half of depth range, not anatomical left/right segmentation.',
       'Moment ellipsoid plus retained distal CT mask, translated only for depth-anchored version.',
       'Extended lung removes every distal exit along global inward rays inside unchanged body.',
       'New synthetic grid unites previous16 and49 states; point and complete-electrode results remain separate.']))
    pd.DataFrame([dict(model=name,lung_litres=float(vol@mask/1e6),changed_litres=float(vol@(mask!=reference)/1e6)) for name,mask in zip(names,masks.T)]).to_csv(OUT/'geometry_volumes.csv',index=False)
    print(json.dumps(dict(centre=centre.tolist(),new_models=names,states=len(truth),h=float(central.h_mm),nearest=float(nearest),shift=shift,seconds=time.perf_counter()-started)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare']);args=parser.parse_args();prepare()
