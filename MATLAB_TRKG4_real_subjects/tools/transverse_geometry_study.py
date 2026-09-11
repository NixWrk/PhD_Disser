"""Geometry-only preparation for 20.16; full contours, fixed C01 pose.

The local signed-distance representation preserves multiple crossings. The
outer body and contacts are never flattened for the two spatial candidates.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize_scalar
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/exploratory/transverse_geometry_20260911'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write(p,d):
    Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def read_surface(path,centre=None,frame=None):
    reader=vtk.vtkSTLReader();reader.SetFileName(str(path));reader.Update()
    clean=vtk.vtkCleanPolyData();clean.SetInputData(reader.GetOutput());clean.Update()
    poly=vtk.vtkPolyData();poly.DeepCopy(clean.GetOutput())
    assert vtk.vtkSelectEnclosedPoints.IsSurfaceClosed(poly)
    if centre is not None:
        coords=(vtk_to_numpy(poly.GetPoints().GetData()).astype(float)-centre)@frame
        pp=vtk.vtkPoints();pp.SetData(numpy_to_vtk(np.ascontiguousarray(coords),deep=True));poly.SetPoints(pp)
    normals=vtk.vtkPolyDataNormals();normals.SetInputData(poly);normals.ConsistencyOn();normals.AutoOrientNormalsOn();normals.SplittingOff();normals.Update()
    result=vtk.vtkPolyData();result.DeepCopy(normals.GetOutput());return result

def distance_grid(poly,s,t,d):
    fn=vtk.vtkImplicitPolyDataDistance();fn.SetInput(poly)
    sample=vtk.vtkSampleFunction();sample.SetImplicitFunction(fn)
    sample.SetModelBounds(s[0],s[-1],t[0],t[-1],d[0],d[-1])
    sample.SetSampleDimensions(len(s),len(t),len(d));sample.ComputeNormalsOff();sample.Update()
    return vtk_to_numpy(sample.GetOutput().GetPointData().GetScalars()).reshape(len(d),len(t),len(s)).transpose(2,1,0).copy()

def smooth_window(x,inner,outer):
    q=np.clip((np.abs(x)-inner)/(outer-inner),0,1)
    return .5*(1+np.cos(np.pi*q))

def sag(t,kappa):
    # Stable circular sag, zero limit included; curvature is signed 1/R.
    return kappa*np.asarray(t)**2/(1+np.sqrt(np.maximum(0,1-(kappa*np.asarray(t))**2)))

def prepare(out):
    started=time.perf_counter();out.mkdir(parents=True,exist_ok=True)
    if (out/'geometry.json').exists():raise FileExistsError('Frozen geometry exists: use another output directory.')
    g=loadmat(out/'geometry_input.mat',simplify_cells=True)
    contract=json.loads((out/'input_contract.json').read_text(encoding='utf-8'))
    centre=np.asarray(g['centre']);frame=np.asarray(g['frame']);xyz=np.asarray(g['centres_mm']);vol=g['volume_mm3']
    assert np.linalg.det(frame)>0;np.testing.assert_allclose(frame.T@frame,np.eye(3),atol=1e-10)
    source=ROOT/'output/two_tissue_auditfix_20260908'
    original=json.loads((source/'geometry.json').read_text(encoding='utf-8'))
    lungfile=ROOT/'data/nik/stl/lungs_solid_v3_volume_fill.stl'
    bodyfile=ROOT/'data/nik/stl/body_solid_v3_1mm_r10_volume_fill.stl'
    assert sha(lungfile)==original['lung_stl_sha256']
    sample=loadmat(source/'mesh_samples.mat',simplify_cells=True)
    assert str(sample['mesh_sha256'])==contract['mesh_sha256']
    np.testing.assert_allclose(sample['centres_mm'],xyz,atol=1e-9)
    reference=loadmat(source/'lung_masks.mat',simplify_cells=True)['closed_lung_mask'].astype(bool)
    local=(xyz-centre)@frame
    lung=read_surface(lungfile,centre,frame)
    s=np.arange(-116,116.1,2.);t=np.arange(-65,65.1,5.);d=np.arange(-40,240.1,2.)
    cache=out/'distance_grid.npz'
    if cache.exists():
        q=np.load(cache);phi=q['phi'];assert str(q['surface_sha'])==sha(lungfile);np.testing.assert_allclose(q['frame'],frame)
    else:
        print('Sampling CT signed distance',len(s)*len(t)*len(d),flush=True)
        phi=distance_grid(lung,s,t,d)
        np.savez_compressed(cache,s=s,t=t,d=d,phi=phi,frame=frame,surface_sha=sha(lungfile))
    interp=RegularGridInterpolator((s,t,d),phi,bounds_error=False,fill_value=np.nan)
    support=(abs(local[:,0])<115)&(abs(local[:,1])<65)&(local[:,2]>-10)&(local[:,2]<220)
    ids=np.flatnonzero(support);coords=local[ids];ct=interp(coords)
    assert np.isfinite(ct).all()
    sign_mismatch=(ct<0)!=reference[ids]
    if np.mean(sign_mismatch)>.02:raise ValueError('Signed-distance orientation or mapping does not match original occupancy.')
    base=phi[:,len(t)//2,:]
    interp0=RegularGridInterpolator((s,d),base,bounds_error=False,fill_value=np.nan)
    S,T,D=np.meshgrid(s,t,d,indexing='ij')
    # Fixed fitting window independent of the candidate curvature and impedance.
    fit=(abs(phi)<12)&(abs(S)<=95)&(abs(T)<=45)&(D>=0)&(D<=180)
    coords_fit=np.column_stack([S[fit],T[fit],D[fit]]);target=phi[fit]
    def loss(k):
        v=interp0(np.column_stack([coords_fit[:,0],coords_fit[:,2]-sag(coords_fit[:,1],k)]))
        assert np.isfinite(v).all()
        return np.mean((v-target)**2)
    kk=np.linspace(-1/80,1/80,101);ll=np.array([loss(k) for k in kk]);best=int(np.argmin(ll))
    trial=minimize_scalar(loss,bounds=(kk[max(0,best-1)],kk[min(len(kk)-1,best+1)]),method='bounded',options={'xatol':1e-11})
    choices=[(0.,loss(0.)),(float(trial.x),float(trial.fun)),(kk[0],ll[0]),(kk[-1],ll[-1])];kappa,value=min(choices,key=lambda x:x[1])
    # Cubic dependence across t with exact equality to the central field at t=0.
    tt=t/65;X=np.column_stack([tt,tt**2,tt**3]);delta=phi-base[:,None,:]
    coefficients=np.einsum('kt,std->sdk',np.linalg.pinv(X),delta)
    def variable(points):
        values=RegularGridInterpolator((s,d),coefficients,bounds_error=False,fill_value=np.nan)(points[:,[0,2]])
        z=points[:,1]/65
        return interp0(points[:,[0,2]])+np.sum(values*np.column_stack([z,z*z,z*z*z]),axis=1)
    uniform=interp0(np.column_stack([coords[:,0],coords[:,2]-sag(coords[:,1],kappa)]))
    varying=variable(coords)
    if not np.isfinite(uniform).all():raise ValueError('Uniform model query outside geometric grid.')
    w=smooth_window(coords[:,0],95,115)*smooth_window(coords[:,1],45,65)
    w*=smooth_window(coords[:,2]-85,95,135) # d core [-10,180], transition to [-50,220]
    # Reference membership is retained wherever a local approximation is not applied.
    masks=[reference.copy() for _ in range(8)]
    masks[1][ids]=((1-w)*ct+w*uniform)<0
    masks[2][ids]=((1-w)*ct+w*varying)<0
    masks[3][ids]=ct<0 # interpolation-only control, same occupancy source elsewhere
    for k,shift in enumerate([-2.,-1.,1.,2.],start=4):
        q=coords.copy();q[:,2]-=shift
        shifted=interp(q);assert np.isfinite(shifted).all()
        masks[k][ids]=((1-w)*ct+w*shifted)<0
    masks=np.column_stack(masks)
    names=['reference','uniform_transverse','variable_transverse','grid_control','h_minus2','h_minus1','h_plus1','h_plus2']
    # Independent t planes check geometric interpolation and cubic extrapolation-free approximation.
    th=(t[:-1]+t[1:])/2
    hold=distance_grid(lung,s,th,d)
    Sh,Th,Dh=np.meshgrid(s,th,d,indexing='ij');points=np.column_stack([Sh.ravel(),Th.ravel(),Dh.ravel()])
    pred0=interp(points);pred3=interp0(np.column_stack([points[:,0],points[:,2]-sag(points[:,1],kappa)]));pred4=variable(points)
    window=(abs(hold.ravel())<8)&(abs(points[:,0])<=95)&(abs(points[:,1])<=45)&(points[:,2]>=0)&(points[:,2]<=180)
    geometry_qc=[]
    for name,pred in [('grid_control',pred0),('uniform_transverse',pred3),('variable_transverse',pred4)]:
        err=pred[window]-hold.ravel()[window]
        geometry_qc.append(dict(model=name,heldout_near_boundary_rms_mm=float(np.sqrt(np.mean(err**2))),heldout_near_boundary_max_mm=float(np.max(abs(err))),sign_disagreement_fraction=float(np.mean((pred[window]<0)!=(hold.ravel()[window]<0)))))
    pd.DataFrame(geometry_qc).to_csv(out/'geometry_holdout.csv',index=False)
    # Contours in local coordinates, including all crossings rather than first/last pairing.
    body=read_surface(bodyfile,centre,frame)
    def crossings(poly,ss):
        tr=vtk.vtkOBBTree();tr.SetDataSet(poly);tr.BuildLocator()
        result=[]
        for x in ss:
            hits=vtk.vtkPoints();tr.IntersectWithLine([x,0,-150],[x,0,650],hits,None)
            z=sorted(vtk_to_numpy(hits.GetData())[:,2].tolist()) if hits.GetNumberOfPoints() else []
            result.append(np.asarray(z)[np.r_[True,np.diff(z)>1e-4]] if len(z) else np.array([]))
        return result
    ss=np.arange(-80,80.01,1.);skin_hits=crossings(body,ss);lung_hits=crossings(lung,ss)
    rows=[]
    for i,x in enumerate(ss):
        skin=float(min(skin_hits[i],key=abs));deeper=lung_hits[i][lung_hits[i]>skin+.1]
        h=float(deeper[0]-skin) if len(deeper) else float('inf')
        rows.append(dict(s_mm=x,skin_d_mm=skin,h_mm=h,lung_crossings_d_mm=json.dumps(deeper.tolist())))
    profile=pd.DataFrame(rows);profile.to_csv(out/'central_profile.csv',index=False)
    hcentre=float(profile.loc[profile.s_mm==0,'h_mm'].iloc[0]);assert np.isfinite(hcentre) and hcentre>0
    bounds=np.array([[1.3774104683195594,10.],[9.741271489527174,24.074073806584366]])
    rho1=np.array([2.,4.,6.,8.]);rho2=np.array([10.,14.,18.,22.]);truth=np.array(np.meshgrid(rho1,rho2,indexing='ij')).reshape(2,-1).T
    knots=np.geomspace(.90,18,19);checks=np.sqrt(knots[:-1]*knots[1:])[::3]
    ratios=np.unique(np.r_[knots,checks,truth[:,1]/truth[:,0],1.,3.,3.6,4.,4*np.exp(-.001),4*np.exp(.001),12/4.04])
    savemat(out/'geometry_masks.mat',dict(masks=masks,model_names=np.array(names,dtype=object),ratios=ratios,extra_ratios=np.array([1.,3.,3.6,4.]),knots=knots,check_ratios=checks,truth=truth,bounds=bounds,mesh_sha256=contract['mesh_sha256']))
    signed_uniform=interp0(np.column_stack([S.ravel(),(D-sag(T,kappa)).ravel()])).reshape(phi.shape)
    signed_variable=base[:,None,:]+np.einsum('sdk,tk->std',coefficients,X)
    np.savez_compressed(out/'geometry_fields.npz',s=s,t=t,d=d,reference=phi,uniform=signed_uniform,variable=signed_variable,coefficients=coefficients,centre=centre,frame=frame)
    pd.DataFrame([dict(model=name,lung_litres=float(vol@mask/1e6),changed_from_CT_litres=float(vol@(mask!=reference)/1e6),changed_elements=int(np.sum(mask!=reference))) for name,mask in zip(names,masks.T)]).to_csv(out/'geometry_volumes.csv',index=False)
    pd.DataFrame({'kappa_per_mm':kk,'distance_loss_mm2':ll}).to_csv(out/'curvature_fit.csv',index=False)
    write(out/'geometry.json',dict(status='exploratory_hypothesis_not_validated',mesh_sha256=contract['mesh_sha256'],masks_sha256=sha(out/'geometry_masks.mat'),input_contract_sha256=sha(out/'input_contract.json'),source_sha256=sha(__file__),lung_sha256=sha(lungfile),body_sha256=sha(bodyfile),model_names=names,constant_transverse_kappa_per_mm=float(kappa),constant_radius_mm=None if abs(kappa)<1e-10 else float(1/kappa),fit_rms_mm=float(np.sqrt(value)),h_centre_mm=hcentre,profile_no_lung_fraction=float(np.mean(~np.isfinite(profile.h_mm))),bounds_ohm_m=bounds.tolist(),truth=truth.tolist(),knots=knots.tolist(),ratios=ratios.tolist(),checks=checks.tolist(),grid_shape=list(phi.shape),grid_steps_mm=[2,5,2],signed_grid_disagreement_elements=int(sign_mismatch.sum()),signed_grid_disagreement_litres=float(vol[ids]@sign_mismatch/1e6),preparation_seconds=time.perf_counter()-started,assumptions=['Two piecewise constant real scalar resistivities; all nonlung materials are background.','Geometry fitted to CT signed distances, never to electrical observations.','Local approximation in a declared window; original CT remains outside; cosine blending is an additional numerical construction.','Same body, mesh, contacts and central signed-distance slice for the two spatial candidates.','Uniform curvature means a common circular depth translation; variable model is cubic in transverse coordinate with spatially varying coefficients.','Cubic level-set representation preserves possible multiple crossings but is not a minimal-parameter final model.','The original section used in earlier notebooks belongs to another pose; every geometry in this study is regenerated at C01.','Scenario depth displacements are prescribed local translations, not measured respiratory deformation.']))
    print('Geometry prepared',json.dumps(geometry_qc),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);args=p.parse_args();prepare(args.out)
