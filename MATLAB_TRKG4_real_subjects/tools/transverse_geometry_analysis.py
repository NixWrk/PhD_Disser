"""Inverse accuracy and linked fixed-geometry/respiratory scenarios for 20.16."""
from pathlib import Path
import argparse,json,time
import numpy as np
import pandas as pd
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import minimize_scalar,least_squares
from transverse_geometry_study import ROOT,OUT,write,sha

LABELS={'reference':'КТ-референс','planar':'1. Постоянная h','averaged':'2. Распределение h',
        'uniform_transverse':'3. Общая поперечная кривизна','variable_transverse':'4. Переменная поперечная форма'}

def analytic(q,sizes,profile,hcentre,mode):
    h=profile.h_mm.to_numpy() if mode=='averaged' else np.array([hcentre])
    s=profile.s_mm.to_numpy();n=np.arange(1,257)[:,None]
    k=(q-1)/(q+1);dk=2*q/(q+1)**2
    powers=k**n;deriv=n*k**(n-1)*dk
    result=[];derivatives=[]
    for L in sizes:
        kernels={};ders={}
        for r in [L/4,3*L/4]:
            denominator=np.hypot(r,2*n*h)
            kernels[r]=1000/r+2000*np.sum(powers/denominator,axis=0)
            ders[r]=2000*np.sum(deriv/denominator,axis=0)
        def pair(a,b,values):
            r=abs(a-b)
            if mode=='planar':return values[r][0]
            lo,hi=sorted([a,b]);points=np.r_[lo,s[(s>lo)&(s<hi)],hi]
            return np.trapezoid(np.interp(points,s,values[r]),points)/r
        a,m,n0,b=-L/2,-L/4,L/4,L/2
        for values,arr in [(kernels,result),(ders,derivatives)]:
            arr.append((pair(a,m,values)-pair(b,m,values)-pair(a,n0,values)+pair(b,n0,values))/(2*np.pi))
    return np.array(result),np.array(derivatives)

class Operator:
    def __init__(self,q,f,df,bounds):
        self.spline=CubicHermiteSpline(np.log(q),f,df,axis=0,extrapolate=False)
        bounds=np.asarray(bounds);self.bounds=bounds;self.grid=np.linspace(np.log(bounds[1,0]/bounds[0,1]),np.log(bounds[1,1]/bounds[0,0]),301)
        self.grid_f=self.spline(self.grid)
    def field(self,rho):
        q=rho[1]/rho[0];f=self.spline(np.log(q));df=self.spline(np.log(q),1)
        z=rho[0]*f
        jac=np.column_stack([f-df,rho[0]*df/rho[1]])
        return z,jac
    def fit(self,y,select=None):
        if select is None:select=np.arange(len(y))
        y=np.asarray(y)[select];ff=self.grid_f[:,select];q=np.exp(self.grid)
        lo=np.maximum(self.bounds[0,0],self.bounds[1,0]/q);hi=np.minimum(self.bounds[0,1],self.bounds[1,1]/q)
        scales=np.clip((ff@y)/np.sum(ff*ff,axis=1),lo,hi)
        loss=np.mean((ff*scales[:,None]-y)**2,axis=1)
        def one(x):
            qq=np.exp(x);f=self.spline(x)[select]
            lower=max(self.bounds[0,0],self.bounds[1,0]/qq);upper=min(self.bounds[0,1],self.bounds[1,1]/qq)
            scale=np.clip(f@y/(f@f),lower,upper)
            return float(np.mean((scale*f-y)**2)),scale
        indices=np.flatnonzero((loss[1:-1]<=loss[:-2])&(loss[1:-1]<=loss[2:]))+1
        options=[(self.grid[0],*one(self.grid[0])),(self.grid[-1],*one(self.grid[-1]))]
        for i in indices:
            r=minimize_scalar(lambda x:one(x)[0],bounds=self.grid[[i-1,i+1]],method='bounded',options={'xatol':1e-11})
            options.append((r.x,*one(r.x)))
        x,loss,scale=min(options,key=lambda x:x[1]);rho=np.array([scale,scale*np.exp(x)])
        assert np.all(rho>=self.bounds[:,0]-1e-7) and np.all(rho<=self.bounds[:,1]+1e-7)
        return rho,np.sqrt(loss)

def load_operators(out):
    g=json.loads((out/'geometry.json').read_text(encoding='utf-8'));df=pd.read_csv(out/'pem_library.csv')
    sizes=np.sort(df.L_mm.unique());profile=pd.read_csv(out/'central_profile.csv');ops={}
    for name in ['reference','uniform_transverse','variable_transverse']:
        d=df[df.model==name];q=d.ratio.unique();take=np.array([q[np.argmin(abs(q-x))] for x in g['knots']]);assert np.max(abs(take-g['knots']))<1e-10
        a=d[d.ratio.isin(take)];f=a.pivot(index='ratio',columns='L_mm',values='f').sort_index();der=a.pivot(index='ratio',columns='L_mm',values='df_dlogq').sort_index()
        ops[name]=Operator(f.index.to_numpy(),f.to_numpy(),der.to_numpy(),g['bounds_ohm_m'])
    rows=[];q=np.geomspace(.9,18.,151)
    for name in ['planar','averaged']:
        f=[];der=[]
        for x in q:
            a,b=analytic(x,sizes,profile,g['h_centre_mm'],name);f.append(a);der.append(b)
            for L,y,z in zip(sizes,a,b):rows.append(dict(model=name,ratio=x,L_mm=L,f=y,df_dlogq=z))
        ops[name]=Operator(q,np.array(f),np.array(der),g['bounds_ohm_m'])
    pd.DataFrame(rows).to_csv(out/'analytic_library.csv',index=False)
    return g,df,sizes,ops

def direct(df,model,rho):
    d=df[df.model==model];q=rho[1]/rho[0];closest=d.ratio.iloc[np.argmin(abs(d.ratio-q))]
    assert abs(q-closest)<1e-9,(model,q,closest)
    d=d[abs(d.ratio-closest)<1e-12].sort_values('L_mm');f=d.f.to_numpy();der=d.df_dlogq.to_numpy()
    return rho[0]*f,np.column_stack([f-der,rho[0]*der/rho[1]])

def analysis(out):
    started=time.perf_counter();g,df,sizes,ops=load_operators(out);bounds=np.array(g['bounds_ohm_m'])
    # Holdout contrast solutions are excluded from the Hermite interpolation knots.
    qc=[]
    for name,op in ops.items():
        for q in g['checks']:
            rho=np.array([4.,4*q])
            if name in ['planar','averaged']:
                f,dd=analytic(q,sizes,pd.read_csv(out/'central_profile.csv'),g['h_centre_mm'],name);z=rho[0]*f;J=np.column_stack([f-dd,dd/q])
            else:z,J=direct(df,name,rho)
            a,b=op.field(rho);qc.append(dict(model=name,ratio=q,max_Z_error_ohm=float(max(abs(a-z))),max_J_error=float(np.max(abs(b-J)))))
    qc=pd.DataFrame(qc);qc.to_csv(out/'interpolation_qc.csv',index=False)
    assert qc.max_Z_error_ohm.max()<.02,'Interpolation must be refined before inverse comparisons'
    homogeneous=df[np.isclose(df.ratio,1,rtol=0,atol=1e-12)].pivot(index='L_mm',columns='model',values='f')
    hom_error=float(np.max(abs(homogeneous.to_numpy()-homogeneous.iloc[:,0].to_numpy()[:,None])))
    assert hom_error<1e-8
    dq=.001;derivative=[]
    for name in ['reference','uniform_transverse','variable_transverse']:
        plus,_=direct(df,name,np.array([4.,16*np.exp(dq)]));minus,_=direct(df,name,np.array([4.,16*np.exp(-dq)]));z,J=direct(df,name,np.array([4.,16.]));fd=(plus-minus)/(2*dq)
        derivative.append(dict(model=name,max_abs_error=float(np.max(abs(fd-J[:,1]*16))),max_relative_error=float(np.max(abs(fd-J[:,1]*16)/np.maximum(abs(fd),1e-10)))))
    pd.DataFrame(derivative).to_csv(out/'derivative_qc.csv',index=False)
    assert max(r['max_relative_error'] for r in derivative)<1e-4
    estimates=[];predictions=[];perturbations=[]
    for ci,rho in enumerate(np.array(g['truth'])):
        y,Jtrue=direct(df,'reference',rho)
        for name,op in ops.items():
            r,loss=op.fit(y);z,J=op.field(r);err=100*(r/rho-1);sing=np.linalg.svd(J*r,compute_uv=False)
            estimates.append(dict(case=ci,model=name,rho1_true=rho[0],rho2_true=rho[1],rho1=r[0],rho2=r[1],error1_pct=err[0],error2_pct=err[1],rmse_ohm=loss,condition_log=sing[0]/sing[-1],on_bound=bool(np.any(np.isclose(r,bounds[:,0],rtol=0,atol=1e-5)|np.isclose(r,bounds[:,1],rtol=0,atol=1e-5)))))
            for L,yy,zz in zip(sizes,y,z):predictions.append(dict(case=ci,model=name,L_mm=L,observed=yy,predicted=zz))
            for channel in range(len(sizes)):
                for sign in [-1,1]:
                    yp=y.copy();yp[channel]*=1+sign*.01;rr,ll=op.fit(yp);ee=100*(rr/rho-1)
                    perturbations.append(dict(case=ci,model=name,L_mm=sizes[channel],sign=sign,error1_pct=ee[0],error2_pct=ee[1],change1_pct=100*(rr[0]/r[0]-1),change2_pct=100*(rr[1]/r[1]-1)))
    pd.DataFrame(estimates).to_csv(out/'inverse_recovery.csv',index=False);pd.DataFrame(predictions).to_csv(out/'inverse_predictions.csv',index=False);pd.DataFrame(perturbations).to_csv(out/'perturbations.csv',index=False)
    base=np.array([4.,16.]);yA,_=direct(df,'reference',base)
    scenarios=[('fixed_deep','reference',np.array([4.,12.]),0.,'deep'),('rho1_deep','reference',np.array([4.04,12.]),0.,'deep'),
       ('h_minus2_deep','h_minus2',np.array([4.,12.]),-2.,'deep'),('h_plus2_deep','h_plus2',np.array([4.,12.]),2.,'deep'),
       ('fixed_quiet','reference',np.array([4.,14.4]),0.,'quiet'),('h_minus1_quiet','h_minus1',np.array([4.,14.4]),-1.,'quiet'),('h_plus1_quiet','h_plus1',np.array([4.,14.4]),1.,'quiet')]
    dynamic=[]
    for sid,true_model,rhoB,dh,mode in scenarios:
        yB,JB=direct(df,true_model,rhoB);pulse_true=rhoB*np.array([.0001,.001]);dy=JB@pulse_true
        for name,op in ops.items():
            rA,_=op.fit(yA);rB,_=op.fit(yB);_,J=op.field(rB);pulse=np.linalg.lstsq(J,dy,rcond=None)[0]
            pulse_only=float(J[:,1]@dy/(J[:,1]@J[:,1]))
            true_delta=rhoB-base;delta=rB-rA
            # Shared rho1 is tested on the same paired states, with equal-ohm residuals.
            lb=np.log([bounds[0,0],bounds[1,0],bounds[1,0]]);ub=np.log([bounds[0,1],bounds[1,1],bounds[1,1]])
            def residual(x):
                r=np.exp(x);return np.r_[op.field(r[[0,1]])[0]-yA,op.field(r[[0,2]])[0]-yB]
            joint=least_squares(residual,np.clip(np.log([np.sqrt(rA[0]*rB[0]),rA[1],rB[1]]),lb+1e-10,ub-1e-10),bounds=(lb,ub),xtol=1e-11,ftol=1e-11,gtol=1e-10)
            rr=np.exp(joint.x);shared_delta=rr[2]-rr[1]
            dynamic.append(dict(scenario=sid,mode=mode,model=name,dh_mm=dh,rho1_B_true=rhoB[0],rho2_B_true=rhoB[1],rho1_B=rB[0],rho2_B=rB[1],delta1_true=true_delta[0],delta2_true=true_delta[1],delta1=delta[0],delta2=delta[1],delta1_error_ohm_m=delta[0]-true_delta[0],delta2_error_pct=100*(delta[1]/true_delta[1]-1),shared_rho1=rr[0],shared_delta2=shared_delta,shared_delta2_error_pct=100*(shared_delta/true_delta[1]-1),shared_fit_rmse=float(np.sqrt(np.mean(joint.fun**2))),pulse1_true=pulse_true[0],pulse2_true=pulse_true[1],pulse1=pulse[0],pulse2=pulse[1],pulse1_error_pct=100*(pulse[0]/pulse_true[0]-1),pulse2_error_pct=100*(pulse[1]/pulse_true[1]-1),pulse2_if_no_soft=pulse_only,pulse2_if_no_soft_error_pct=100*(pulse_only/pulse_true[1]-1)))
    pd.DataFrame(dynamic).to_csv(out/'linked_scenarios.csv',index=False)
    pair_rows=[]
    for name,op in ops.items():
        for i in range(len(sizes)):
            for j in range(i+1,len(sizes)):
                r,loss=op.fit(yA,[i,j]);_,J=op.field(r);sv=np.linalg.svd((J*r)[[i,j]],compute_uv=False)
                pair_rows.append(dict(model=name,L1_mm=sizes[i],L2_mm=sizes[j],rho1=r[0],rho2=r[1],error1_pct=100*(r[0]/4-1),error2_pct=100*(r[1]/16-1),condition_log=sv[0]/sv[-1]))
    pd.DataFrame(pair_rows).to_csv(out/'pair_diagnostics.csv',index=False)
    controls=[]
    for name in ['grid_control','h_minus2','h_minus1','h_plus1','h_plus2']:
        z,J=direct(df,name,base);r,loss=ops['reference'].fit(z)
        controls.append(dict(model=name,max_delta_Z_ohm=float(np.max(abs(z-yA))),rho1=r[0],rho2=r[1],error1_pct=100*(r[0]/4-1),error2_pct=100*(r[1]/16-1)))
    pd.DataFrame(controls).to_csv(out/'geometry_control_recovery.csv',index=False)
    initial=[]
    target_cem=pd.read_csv(out/'cem_baseline.csv')
    target_cem=target_cem[target_cem.model=='reference'].sort_values('L_mm').Z.to_numpy()
    for name,op in ops.items():
        r,_=op.fit(target_cem);initial.append(dict(model=name,rho1=r[0],rho2=r[1]))
    pd.DataFrame(initial).to_csv(out/'cem_initial.csv',index=False)
    write(out/'analysis_summary.json',dict(scope='Synthetic inverse comparison at one frozen C01 pose; field libraries and linked scenarios.',inverse_cases=len(estimates),perturbation_cases=len(perturbations),dynamic_cases=len(dynamic),pair_cases=len(pair_rows),homogeneous_max_error=hom_error,max_interpolation_error_ohm=float(qc.max_Z_error_ohm.max()),analysis_seconds=time.perf_counter()-started,input_sha256={p.name:sha(p) for p in [out/'geometry.json',out/'pem_library.csv']},assumptions=['All primary inverse cases use matched point contacts within the spatial group.','Analytical comparisons additionally flatten body and montage; their total errors do not isolate lung shape alone.','One-channel plus/minus one percent is a deterministic perturbation scenario, not a measured noise distribution.','Respiratory states and small pulse amplitudes are prescribed synthetic scenarios, not observed physiology.','Pulse forward data are local first-order derivatives; base and respiratory inverse biases are propagated to the pulse Jacobian.','Depth offsets of 1 and 2 mm are differences between two states, not plus/minus excursions combined into one cycle.','Zero true change in rho1 is evaluated by absolute error, avoiding undefined relative percentages.']))
    print('Analysis finished',len(estimates),len(dynamic),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();analysis(a.out)
