"""Half-space BEM libraries and matched-coordinate inverse experiments (20.17)."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from halfspace_bem import InclusionBEM
from transverse_geometry_analysis import Operator,direct,analytic

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'output/exploratory/transverse_consistency_20260914'
OUT=ROOT/'output/exploratory/halfspace_bem_20260914'

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def prepare_inputs():
    from scipy.io import loadmat
    OUT.mkdir(parents=True,exist_ok=True)
    g=loadmat(SOURCE/'geometry_input.mat',simplify_cells=True);c=pd.read_csv(SOURCE/'contacts.csv')
    xyz=g['nodes_mm'][c.point_node.to_numpy(int)-1];local=(xyz-g['centre'])@g['frame']
    c[['source_x_mm','source_y_mm','source_z_mm']]=xyz
    c[['s_mm','t_mm','d_mm_before_flatten']]=local;c['halfspace_d_mm']=0.
    c.to_csv(OUT/'electrodes_projected.csv',index=False)
    inputs=['geometry_input.mat','input_contract.json','geometry.json','geometry_fields.npz','pem_library.csv','cem_baseline.csv','contacts.csv','source_electrodes.csv','inverse_recovery.csv','linked_scenarios.csv']
    write(OUT/'experiment_contract.json',dict(status='exploratory_hypothesis_not_validated',source_directory=SOURCE.relative_to(ROOT).as_posix(),inputs_sha256={p:sha(SOURCE/p) for p in inputs},scope=['M3-H constant transverse section, Fourier contour BEM','M4-H variable transverse section, 3D BEM','CT/FEM inverse recovery','h and electrode-placement perturbations','runtime including FEM-prepared surrogate','pulse Jacobian and nonlinear FEM synthetic controls'],deferred='Multi-subject anatomical site search and final assembly pair selection',primary_electrodes='Point currents and point voltages at actual CT/FEM mesh nodes orthogonally projected onto d=0; s,t unchanged. Geometry flattening is a declared model change.',finite_electrode_check='M4-H one state rho4/16, equal-area flat disks, complete electrode condition; no finite-electrode inverse sweep.',pulse_data='Synthetic tests only; no experimental pulse resistivities inferred.',author_h_scenarios='Between-state range <=2 mm deep breathing and <=1 mm quiet; not plus/minus amplitude about one reference.',threads=6))
    sizes,p=electrodes();df=pd.read_csv(SOURCE/'pem_library.csv')
    a=df[(df.model=='reference') & np.isclose(df.ratio,1)].sort_values('L_mm');f,_=planar(1.,p,.067)
    pd.DataFrame(dict(L_mm=sizes,CT_FEM_homogeneous_unit_rho=a.f.to_numpy(),halfspace_projected_unit_rho=f,relative_error_pct=100*(f/a.f.to_numpy()-1))).to_csv(OUT/'homogeneous_body_control.csv',index=False)


def electrodes():
    df=pd.read_csv(OUT/'electrodes_projected.csv')
    sizes=np.sort(df.L_mm.unique());p=[]
    for L in sizes:
        d=df[df.L_mm==L]
        assert d.electrode.tolist()==['I+','V+','V-','I-']
        p.append(np.c_[d[['s_mm','t_mm']].to_numpy()/1000,np.zeros(4)])
    return sizes,np.array(p)

def planar(q,points,h):
    """Exact two-layer half-space kernel, actual projected point locations."""
    k=(q-1)/(q+1);dk=2*q/(q+1)**2;n=np.arange(1,513)
    f=np.zeros(len(points));df=f.copy()
    for a,b,sign in [(1,0,1),(1,3,-1),(2,0,-1),(2,3,1)]:
        r=np.linalg.norm(points[:,a]-points[:,b],axis=1)
        denom=np.hypot(r[:,None],2*h*n)
        f+=sign*(1/r+2*np.sum(k**n/denom,axis=1))/(2*np.pi)
        df+=sign*2*np.sum(n*k**(n-1)*dk/denom,axis=1)/(2*np.pi)
    return f,df

def build_library(surface_path,name):
    g=json.loads((SOURCE/'geometry.json').read_text(encoding='utf-8'))
    sizes,points=electrodes();data=np.load(surface_path)
    v=data['vertices'];f=data['faces']
    identity=dict(surface_sha256=sha(surface_path),solver_sha256=sha(Path(__file__).with_name('halfspace_bem.py')),electrodes_sha256=sha(OUT/'electrodes_projected.csv'))
    csv=OUT/f'library_{name}.csv';meta=OUT/f'library_{name}.json'
    if csv.exists() and meta.exists():
        old=json.loads(meta.read_text(encoding='utf-8'))
        if old['identity']==identity and old['csv_sha256']==sha(csv):return
        raise RuntimeError(f'Stale {name} library: choose explicit new name')
    start=time.perf_counter();bem=InclusionBEM(v,f,points)
    knots=np.asarray(g['knots']);checks=np.sqrt(knots[:-1]*knots[1:])[::3]
    qs=np.unique(np.r_[knots,checks,1.,3.,4.,3.6,12/4.04])
    rows=[]
    for q in qs:
        r=bem.solve(q)
        for L,z,df,rec in zip(sizes,r['f'],r['df_dlogq'],r['reciprocity_abs']):
            rows.append(dict(model=name,ratio=q,L_mm=L,f=z,df_dlogq=df,reciprocity_abs=rec,relative_residual=r['relative_residual'],elapsed_seconds=r['seconds'],is_knot=bool(np.any(np.isclose(q,knots,atol=1e-12,rtol=0)))))
        print(f'{name} q={q:.5g}: {r["seconds"]:.3f}s',flush=True)
    pd.DataFrame(rows).to_csv(csv,index=False)
    write(meta,dict(identity=identity,csv_sha256=sha(csv),surface_file=Path(surface_path).resolve().relative_to(ROOT).as_posix(),triangles=len(f),vertices=len(v),volume_litres=bem.volume*1000,assembly_seconds=bem.assembly_seconds,total_seconds=time.perf_counter()-start,contrast_count=len(qs),knots=knots.tolist(),checks=checks.tolist(),solid_angle_max_error=bem.row_sum_error,threads=6))

def build_fourier_library(name='m3h',shift_mm=0.,n_segments=360,n_k=64):
    from halfspace_25d import ExtrudedBEM,load_contours
    g=json.loads((SOURCE/'geometry.json').read_text(encoding='utf-8'));sizes,p=electrodes()
    contours=load_contours(OUT/'geometry_contours.npz')
    for c in contours:c[:,1]+=shift_mm/1000
    started=time.perf_counter();bem=ExtrudedBEM(contours,p,n_segments,n_k)
    knots=np.asarray(g['knots']);checks=np.sqrt(knots[:-1]*knots[1:])[::3]
    qs=np.unique(np.r_[knots,checks,1.,3.,4.,3.6,12/4.04]);rows=[]
    for q in qs:
        r=bem.solve(q)
        for L,z,d,rec in zip(sizes,r['f'],r['df_dlogq'],r['reciprocity_abs']):rows.append(dict(model=name,ratio=q,L_mm=L,f=z,df_dlogq=d,reciprocity_abs=rec,relative_residual=r['relative_residual'],elapsed_seconds=r['seconds']))
        print(f'{name} Fourier q={q:.5g}: {r["seconds"]:.3f}s',flush=True)
    csv=OUT/f'library_{name}.csv';pd.DataFrame(rows).to_csv(csv,index=False)
    write(OUT/f'library_{name}.json',dict(solver='Fourier contour BEM',segments=len(bem.segments),k_nodes=n_k,shift_mm=shift_mm,assembly_seconds=bem.assembly_seconds,total_seconds=time.perf_counter()-started,contrast_count=len(qs),knots=knots.tolist(),checks=checks.tolist(),threads=6,csv_sha256=sha(csv),identity={p.name:sha(p) for p in [Path(__file__).with_name('halfspace_25d.py'),OUT/'geometry_contours.npz',OUT/'electrodes_projected.csv']}))


def from_frame(df,bounds,knots=None):
    if knots is not None:
        q=df.ratio.unique();take=q[[np.argmin(abs(q-x)) for x in knots]]
        if max(abs(take-knots))>1e-10:raise ValueError('Missing interpolation knots')
        df=df[df.ratio.isin(take)]
    f=df.pivot(index='ratio',columns='L_mm',values='f').sort_index()
    d=df.pivot(index='ratio',columns='L_mm',values='df_dlogq').sort_index()
    return Operator(f.index.to_numpy(),f.to_numpy(),d.to_numpy(),bounds)

def operators(names):
    g=json.loads((SOURCE/'geometry.json').read_text(encoding='utf-8'));bounds=np.asarray(g['bounds_ohm_m']);ref=pd.read_csv(SOURCE/'pem_library.csv')
    sizes,points=electrodes();ops={};libs={}
    for name in ['reference','uniform_transverse','variable_transverse']:
        d=ref[ref.model==name];ops[name]=from_frame(d,bounds,g['knots']);libs[name]=d
    for name in names:
        d=pd.read_csv(OUT/f'library_{name}.csv');ops[name]=from_frame(d,bounds,g['knots']);libs[name]=d
    profile=pd.read_csv(SOURCE/'central_profile.csv')
    for name in ['planar_projected','planar_nominal']:
        qs=np.geomspace(.9,18,151);ff=[];dd=[]
        for q in qs:
            f,d=planar(q,points,g['h_centre_mm']/1000) if name=='planar_projected' else analytic(q,sizes,profile,g['h_centre_mm'],'planar')
            ff.append(f);dd.append(d)
        ops[name]=Operator(qs,np.array(ff),np.array(dd),bounds)
    return g,ref,sizes,ops,libs

def analyze(names):
    g,ref,sizes,ops,libs=operators(names);bounds=np.array(g['bounds_ohm_m']);rows=[];pred=[];jacrows=[];pert=[];qc=[];times=[]
    for name in names:
        for q in json.loads((OUT/f'library_{name}.json').read_text(encoding='utf-8'))['checks']:
            z,J=direct(libs[name],name,np.array([4.,4*q]));zh,Jh=ops[name].field([4.,4*q])
            qc.append(dict(model=name,ratio=q,max_Z_error_ohm=float(max(abs(zh-z))),max_J_error=float(np.max(abs(Jh-J)))))
    pd.DataFrame(qc).to_csv(OUT/'interpolation_qc.csv',index=False)
    if max(r['max_Z_error_ohm'] for r in qc)>.02:raise ValueError('Refine BEM contrast library')
    for case,rho in enumerate(np.array(g['truth'])):
        y,Jtrue=direct(ref,'reference',rho)
        for name,op in ops.items():
            r,loss=op.fit(y);z,J=op.field(r);_,Jknown=op.field(rho);err=100*(r/rho-1)
            sv=np.linalg.svd(J*r,compute_uv=False)
            rows.append(dict(case=case,model=name,rho1_true=rho[0],rho2_true=rho[1],rho1=r[0],rho2=r[1],error1_pct=err[0],error2_pct=err[1],rmse_ohm=loss,condition_log=sv[0]/sv[-1],on_bound=bool(np.any(np.isclose(r,bounds[:,0],atol=1e-5,rtol=0)|np.isclose(r,bounds[:,1],atol=1e-5,rtol=0)))))
            for L,yy,zz in zip(sizes,y,z):pred.append(dict(case=case,model=name,L_mm=L,observed=yy,predicted=zz))
            for j in [0,1]:jacrows.append(dict(case=case,model=name,tissue=j+1,known_rho_J_relative_error=float(np.linalg.norm(Jknown[:,j]-Jtrue[:,j])/np.linalg.norm(Jtrue[:,j])),estimated_rho_J_relative_error=float(np.linalg.norm(J[:,j]-Jtrue[:,j])/np.linalg.norm(Jtrue[:,j]))))
            for channel in range(len(sizes)):
                for sign in [-1,1]:
                    yp=y.copy();yp[channel]*=1+sign*.01;rr,_=op.fit(yp)
                    pert.append(dict(case=case,model=name,L_mm=sizes[channel],sign=sign,error1_pct=100*(rr[0]/rho[0]-1),error2_pct=100*(rr[1]/rho[1]-1),change1_pct=100*(rr[0]/r[0]-1),change2_pct=100*(rr[1]/r[1]-1)))
    pd.DataFrame(rows).to_csv(OUT/'inverse_recovery.csv',index=False);pd.DataFrame(pred).to_csv(OUT/'inverse_predictions.csv',index=False);pd.DataFrame(jacrows).to_csv(OUT/'jacobian_comparison.csv',index=False);pd.DataFrame(pert).to_csv(OUT/'perturbations.csv',index=False)
    base=np.array([4.,16.]);yA,_=direct(ref,'reference',base)
    scenarios=[('fixed_deep','reference',[4.,12.],0.),('rho1_deep','reference',[4.04,12.],0.),('h_minus2_deep','h_minus2',[4.,12.],-2.),('h_plus2_deep','h_plus2',[4.,12.],2.),('fixed_quiet','reference',[4.,14.4],0.),('h_minus1_quiet','h_minus1',[4.,14.4],-1.),('h_plus1_quiet','h_plus1',[4.,14.4],1.)]
    dynamic=[]
    for scenario,truth_model,rhoB,dh in scenarios:
        rhoB=np.array(rhoB);yB,JB=direct(ref,truth_model,rhoB);dp=rhoB*np.array([.0001,.001]);dy=JB@dp
        for name,op in ops.items():
            rA,_=op.fit(yA);rB,rmse=op.fit(yB);_,J=op.field(rB);pulse=np.linalg.lstsq(J,dy,rcond=None)[0];drop=J[:,1]@dy/(J[:,1]@J[:,1])
            lb=np.log([bounds[0,0],bounds[1,0],bounds[1,0]]);ub=np.log([bounds[0,1],bounds[1,1],bounds[1,1]])
            def resid(x):
                r=np.exp(x);return np.r_[op.field(r[[0,1]])[0]-yA,op.field(r[[0,2]])[0]-yB]
            joint=least_squares(resid,np.clip(np.log([np.sqrt(rA[0]*rB[0]),rA[1],rB[1]]),lb+1e-10,ub-1e-10),bounds=(lb,ub),xtol=1e-10,ftol=1e-10,gtol=1e-10)
            rr=np.exp(joint.x)
            dynamic.append(dict(scenario=scenario,model=name,dh_mm=dh,rho1_B=rB[0],rho2_B=rB[1],rho1_B_true=rhoB[0],rho2_B_true=rhoB[1],delta1=rB[0]-rA[0],delta2=rB[1]-rA[1],delta1_true=rhoB[0]-base[0],delta2_true=rhoB[1]-base[1],delta1_error_ohm_m=(rB[0]-rA[0])-(rhoB[0]-base[0]),delta2_error_pct=100*((rB[1]-rA[1])/(rhoB[1]-base[1])-1),shared_delta2=rr[2]-rr[1],shared_delta2_error_pct=100*((rr[2]-rr[1])/(rhoB[1]-base[1])-1),shared_rmse_ohm=float(np.sqrt(np.mean(joint.fun**2))),pulse1=pulse[0],pulse2=pulse[1],pulse1_true=dp[0],pulse2_true=dp[1],pulse1_error_pct=100*(pulse[0]/dp[0]-1),pulse2_error_pct=100*(pulse[1]/dp[1]-1),pulse2_if_drop_soft=drop,pulse2_if_drop_soft_error_pct=100*(drop/dp[1]-1)))
    pd.DataFrame(dynamic).to_csv(OUT/'linked_scenarios.csv',index=False)
    for name,op in ops.items():
        for task in ['field_and_jacobian','inverse']:
            for repeat in range(7):
                start=time.perf_counter()
                if task=='field_and_jacobian':
                    for k in range(100):op.field([4.,16.])
                    elapsed=(time.perf_counter()-start)/100
                else:op.fit(yA);elapsed=time.perf_counter()-start
                times.append(dict(model=name,task=task,repeat=repeat,seconds=elapsed))
    pd.DataFrame(times).to_csv(OUT/'surrogate_timings.csv',index=False)
    write(OUT/'analysis_summary.json',dict(models=list(ops),inverse_states=16,sizes_mm=sizes.tolist(),h_mm=g['h_centre_mm'],no_pair_selection=True,assumptions=['The nine point montages retain actual CT/FEM s,t coordinates; d is projected to zero in half-space models.','Known CT/FEM resistivities are synthetic reference values within specified working bounds.','Equal-ohm residual weights; one-channel +/-1% is a deterministic stress test, not an empirical noise model.','Boundary hits are reported as model inadequacy diagnostics, never accepted as physiologically accurate estimates.','Pulse truth is a specified local perturbation with both tissue terms; independent nonlinear FEM controls are separate.','No probability distribution or clinical precision is inferred from these calculations.'],input_hashes={str(p.relative_to(ROOT)):sha(p) for p in [SOURCE/'pem_library.csv',OUT/'electrodes_projected.csv',*[OUT/f'library_{n}.csv' for n in names]]}))
    print(pd.DataFrame(rows).groupby('model').agg(median_abs_rho1=('error1_pct',lambda x:np.median(abs(x))),median_abs_rho2=('error2_pct',lambda x:np.median(abs(x))),max_abs_rho2=('error2_pct',lambda x:max(abs(x))),bound_cases=('on_bound','sum')).to_string())

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('library');b.add_argument('surface',type=Path);b.add_argument('name')
    sub.add_parser('fourier')
    sub.add_parser('inputs')
    a=sub.add_parser('analyze');a.add_argument('names',nargs='+')
    args=p.parse_args()
    if args.command=='library':build_library(args.surface,args.name)
    elif args.command=='inputs':prepare_inputs()
    elif args.command=='fourier':build_fourier_library()
    elif args.command=='analyze':analyze(args.names)
