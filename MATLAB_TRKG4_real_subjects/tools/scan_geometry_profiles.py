"""Read-only CT geometry descriptors for a stopped CEM scan; no new FEM solves."""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
import pandas as pd
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from scan_layered_library import planar

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'output/exploratory/scan_localization_20260910'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

class Surface:
    def __init__(self,path):
        r=vtk.vtkSTLReader();r.SetFileName(str(path));r.Update()
        c=vtk.vtkCleanPolyData();c.SetInputData(r.GetOutput());c.Update();self.poly=c.GetOutput()
        edges=vtk.vtkFeatureEdges();edges.SetInputData(self.poly);edges.BoundaryEdgesOn();edges.NonManifoldEdgesOn();edges.FeatureEdgesOff();edges.ManifoldEdgesOff();edges.Update()
        self.bad_edges=edges.GetOutput().GetNumberOfCells()
        self.tree=vtk.vtkOBBTree();self.tree.SetDataSet(self.poly);self.tree.BuildLocator()
        self.near=vtk.vtkStaticCellLocator();self.near.SetDataSet(self.poly);self.near.BuildLocator()
    def distance(self,p):
        closest=[0.,0.,0.];cell=vtk.reference(0);sub=vtk.reference(0);d=vtk.reference(0.)
        self.near.FindClosestPoint(p,closest,cell,sub,d)
        return np.sqrt(float(d))
    def intervals(self,p,d,limit=600.):
        # Closed surfaces and outside start/end are required for interval lengths.
        if self.bad_edges:return None,'nonclosed_surface'
        points=vtk.vtkPoints();ids=vtk.vtkIdList()
        self.tree.IntersectWithLine(p,p+limit*d,points,ids)
        if points.GetNumberOfPoints()==0:return np.empty((0,2)),'no_intersection'
        ts=np.sort((vtk_to_numpy(points.GetData()).astype(float)-p)@d)
        ts=ts[np.r_[True,np.diff(ts)>1e-4]]
        if len(ts)%2 or ts[0]<=1e-4 or ts[-1]>=limit-1e-4:return None,'unpaired_or_boundary_hits'
        if self.tree.InsideOrOutside(p)!=1:return None,'origin_not_outside'
        pairs=ts.reshape(-1,2)
        for lo,hi in pairs:
            if self.tree.InsideOrOutside(p+(lo+hi)*.5*d)!=-1:return None,'interval_orientation_ambiguous'
        return pairs,'ok'

def describe(lung,bone,p,normal):
    d=-np.asarray(normal,float);d/=np.linalg.norm(d);p=np.asarray(p,float)
    pairs,status=lung.intervals(p,d)
    row=dict(lung_ray_qc=status,lung_entry_mm=np.nan,lung_first_chord_mm=np.nan,
             bone_before_lung_mm=np.nan,bone_ray_qc='lung_unavailable',lung_nearest_surface_mm=lung.distance(p))
    if pairs is not None and len(pairs):
        lo,hi=pairs[0];row.update(lung_entry_mm=lo,lung_first_chord_mm=hi-lo)
        bp,bs=bone.intervals(p,d);row['bone_ray_qc']=bs
        if bp is not None:row['bone_before_lung_mm']=float(np.maximum(0,np.minimum(bp[:,1],lo)-bp[:,0]).sum())
    return row

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['centres','candidates']);parser.add_argument('--out',type=Path,default=DEFAULT);args=parser.parse_args();out=args.out
    sources=json.loads((out/'geometry_sources.json').read_text(encoding='utf-8-sig'))['surfaces']
    for s in sources.values():assert sha(s['path'])==s['sha256']
    lung=Surface(sources['lungs']['path']);bone=Surface(sources['bones']['path'])
    print('Surface QC: lung/bone bad edges',lung.bad_edges,bone.bad_edges,flush=True)
    start=time.perf_counter()
    if args.stage=='centres':
        data=np.load(out/'scan_arrays.npz');centres=pd.read_csv(out/'centres_geometry.csv')
        indices=np.flatnonzero(np.any(data['status']==1,axis=1));records=[]
        for i,k in enumerate(indices):
            c=centres.iloc[k];record=describe(lung,bone,c[['x_mm','y_mm','z_mm']].to_numpy(float),c[['nx','ny','nz']].to_numpy(float))
            record['centre_index']=int(k+1);records.append(record)
            if i%1000==0:print('CT centre rays',i,'/',len(indices),flush=True)
        table=pd.DataFrame(records);table.to_csv(out/'centre_ct_geometry.csv',index=False)
        h=np.full(len(centres),np.nan);h[indices]=table.lung_entry_mm
        z=np.full((len(centres),len(data['sizes'])),np.nan);ok=np.isfinite(h)
        z[ok]=planar(h[ok],data['sizes'],*data['rho'])
        ci=data['flat']//len(data['phi']);fit=np.load(out/'layered_fits.npz')
        rmse=np.sqrt(np.mean((z[ci]-data['Z_fem'])**2,axis=1))
        np.savez_compressed(out/'ct_comparison.npz',normal_depth_by_centre_mm=h,
             rmse_with_ct_normal_depth_ohm=rmse,h_eff_minus_ct_depth_mm=fit['h_mm']-h[ci])
        valid=np.isfinite(rmse);delta=fit['h_mm'][valid]-h[ci[valid]]
        summary=dict(centres_with_evaluated_poses=len(indices),ray_qc_counts=table.lung_ray_qc.value_counts().to_dict(),
            bone_qc_counts=table.bone_ray_qc.value_counts().to_dict(),poses_with_ct_normal_depth=int(valid.sum()),
            median_rmse_with_ct_normal_depth_ohm=float(np.median(rmse[valid])),
            median_rmse_with_fitted_h_same_poses_ohm=float(np.median(fit['rmse_ohm'][valid])),
            median_h_eff_minus_ct_depth_mm=float(np.median(delta[np.isfinite(delta)])))
    else:
        summary={}
        for name in ['candidate_electrodes','candidate_axis_geometry']:
            table=pd.read_csv(out/(name+'.csv'));records=[]
            for _,c in table.iterrows():
                row=describe(lung,bone,c[['x_mm','y_mm','z_mm']].to_numpy(float),c[['nx','ny','nz']].to_numpy(float))
                if name=='candidate_axis_geometry':
                    parallel=describe(lung,bone,c[['x_mm','y_mm','z_mm']].to_numpy(float),c[['centre_nx','centre_ny','centre_nz']].to_numpy(float))
                    row.update({'parallel_'+k:v for k,v in parallel.items()})
                records.append(row)
            joined=pd.concat([table,pd.DataFrame(records)],axis=1);joined.to_csv(out/(name+'_ct.csv'),index=False)
            if name=='candidate_axis_geometry':
                rows=[]
                for key,t in joined.groupby('candidate_id',sort=False):
                    valid=t.lung_ray_qc=='ok';par=t.parallel_lung_ray_qc=='ok'
                    r=dict(candidate_id=key,valid_axis_rays=int(valid.sum()),axis_points=len(t),
                        valid_parallel_rays=int(par.sum()))
                    for field in ['lung_entry_mm','lung_first_chord_mm','bone_before_lung_mm']:
                        v=t.loc[valid,field].dropna();r[field+'_min']=v.min();r[field+'_median']=v.median();r[field+'_max']=v.max()
                    # 1-D directional graph curvature, not a 3-D principal curvature.
                    if par.sum()>=3:
                        s=t.loc[par,'axis_s_mm'].to_numpy();z=t.loc[par,'parallel_lung_entry_mm'].to_numpy()
                        # Include skin displacement to express the lung boundary in a common frame.
                        xyz=t.loc[par,['x_mm','y_mm','z_mm']].to_numpy();normal=t.loc[par,['centre_nx','centre_ny','centre_nz']].to_numpy()[0]
                        lung_depth=z-((xyz-xyz[np.argmin(abs(s))])@normal)
                        coef=np.polyfit(s,lung_depth,2);r['lung_directional_graph_curvature_per_mm']=2*coef[0]/(1+coef[1]**2)**1.5
                        r['quadratic_profile_residual_mm']=float(np.sqrt(np.mean((np.polyval(coef,s)-lung_depth)**2)))
                    rows.append(r)
                pd.DataFrame(rows).to_csv(out/'candidate_ct_summary.csv',index=False)
        summary['candidates']=len(rows)
    summary.update(no_new_FEM=True,seconds=time.perf_counter()-start,lung_bad_edges=lung.bad_edges,bone_bad_edges=bone.bad_edges,
        ray_limit_mm=600,ray_direction='inward local surface normal; separate parallel-centre-normal candidate profiles',
        interpretation='First closed lung interval along a ray; includes nonlung structures before entry. Not a tissue-specific soft-layer thickness.',
        bone_interpretation='Total bone interval length before first lung entry, if closed-surface intersections pass QC.',
        curvature_interpretation='Quadratic 1-D directional graph descriptor across nominal axis offsets; not principal curvature or causal explanation of error.',
        h_comparison='Effective electrical fit and geometric depth are distinct. Four-material CEM / curved finite contacts vs two-layer point-contact model confound attribution.',
        producer_sha256=sha(__file__))
    (out/('ct_'+args.stage+'_summary.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
