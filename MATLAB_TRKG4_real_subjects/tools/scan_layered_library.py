"""Certified polyline approximation of the positive-contrast planar 2L curve.

No FEM is run. One h is fitted to a vector of nine signed transfer impedances.
The componentwise chord error bound applies to every held-out-size subset.
"""
from __future__ import annotations
import numpy as np


def planar(h_mm, sizes_mm, rho1, rho2, terms=160):
    h=np.atleast_1d(h_mm).astype(float); sizes=np.asarray(sizes_mm,float)
    if not (rho2>rho1>0) or np.any(h<0) or np.any(np.isnan(h)):
        raise ValueError('This certified fitter requires rho2>rho1>0 and h>=0')
    a=sizes*.001/4; b=3*a; q=(rho2-rho1)/(rho2+rho1)
    out=np.empty((len(h),len(a)))
    for start in range(0,len(h),2048):
        hh=h[start:start+2048]; n=np.arange(1,terms+1)
        t=hh[:,None,None]*(2*n[None,:,None]*.001)
        u=np.hypot(a,t);v=np.hypot(b,t)
        gap=(b*b-a*a)/(u*v*(u+v))
        out[start:start+len(hh)]=rho1/np.pi*(1/a-1/b+2*np.sum(q**n[None,:,None]*gap,axis=1))
    return out


def second_derivative_bound(h_mm, sizes_mm, rho1, rho2, terms=160):
    """Global componentwise |d²Z/dh_mm²| bound for all h>=h_mm.

    For F(c,t)=(c²+t²)^(-1/2), |F_tt(a)-F_tt(b)| is bounded
    both by individual terms and by 6(b²-a²)/(a²+t²)^(5/2).
    The uncomputed positive-q series uses its exact sum of n² q^n.
    """
    a=np.asarray(sizes_mm)*.001/4;b=3*a;q=(rho2-rho1)/(rho2+rho1)
    n=np.arange(1,terms+1,dtype=float);t=2*n[:,None]*h_mm*.001
    aa=a*a+t*t;bb=b*b+t*t
    bound=np.minimum(2/aa**1.5+2/bb**1.5,6*(b*b-a*a)/aa**2.5)
    value=np.sum(q**n[:,None]*(2*n[:,None]*.001)**2*bound,axis=0)
    # Analytic remainder, avoiding cancellation of total minus partial sum.
    m=terms+1
    tail_n2=q**m*(m*m/(1-q)+2*m*q/(1-q)**2+q*(1+q)/(1-q)**3)
    value+=4e-6*tail_n2*(2/a**3+2/b**3)
    return 2*rho1/np.pi*value


def build_library(sizes, rho1, rho2, epsilon_ohm=.0005):
    """A finite polyline plus the exact h=infinity endpoint.

    Any true curve point and its linearly interpolated counterpart differ
    by <=epsilon_ohm in each component. Thus the fitted true RMSE differs
    from the global infimum by <=2*epsilon_ohm (+float rounding).
    h values beyond H share the same asymptote within epsilon_ohm.
    """
    sizes=np.asarray(sizes,float);q=(rho2-rho1)/(rho2+rho1)
    terms=160;lower=8*rho1/(3*np.pi*sizes*.001)
    tail=float(np.max(2*rho1/np.pi*(1/(sizes*.001/4)-1/(sizes*.001*3/4))*q**(terms+1)/(1-q)))
    if tail>=epsilon_ohm/100:raise ValueError('Series tail exceeds the error budget')
    high=200.
    while np.max(planar([high],sizes,rho1,rho2,terms)[0]-lower)+tail>epsilon_ohm:
        high*=2
    stack=[(0.,high)];accepted=[];maximum=0.
    while stack:
        lo,hi=stack.pop();e=float(np.max(second_derivative_bound(lo,sizes,rho1,rho2,terms))*(hi-lo)**2/8+2*tail)
        if e<=epsilon_ohm:
            accepted.append((lo,hi));maximum=max(maximum,e)
        else:
            mid=(lo+hi)/2;stack.extend([(mid,hi),(lo,mid)])
    h=np.array(sorted(set(x for pair in accepted for x in pair)))
    z=planar(h,sizes,rho1,rho2,terms)
    assert np.all(np.diff(z,axis=0)<=1e-9)
    return dict(h=h,z=z,asymptote=lower,epsilon_ohm=epsilon_ohm,
                max_chord_bound_ohm=maximum,tail_start_mm=high,series_terms=terms,
                uniform_series_tail_ohm=tail,rho1=rho1,rho2=rho2,sizes=sizes)


def fit_batch(target, library, *, leave_one_out=True):
    """Global least-squares projection onto every certified line segment.

    Uses float64. Return one shared h, fit RMSE, and held-out predictions.
    For each fold h uses eight sizes and predicts only the ninth omitted size.
    """
    y=np.asarray(target,float);a=library['z'][:-1];v=np.diff(library['z'],axis=0)
    h=library['h'];linf=library['asymptote'];dim=y.shape[1]
    if y.ndim!=2 or dim!=a.shape[1] or not np.isfinite(y).all():raise ValueError('Finite curves required')
    av=np.einsum('ij,ij->i',a,v);vv=np.einsum('ij,ij->i',v,v)
    # Shared dot products reused by nine eight-size fits.
    yy=np.einsum('ij,ij->i',y,y)
    base=yy[:,None]+np.einsum('ij,ij->i',a,a)-2*y@a.T
    dot=y@v.T-av
    answer={};fold_h=[];pred=[]
    for omitted in [-1]+(list(range(dim)) if leave_one_out else []):
        if omitted<0:
            b=base;dv=dot;length=vv;n=dim;end=np.sum((y-linf)**2,axis=1)
        else:
            difference=y[:,omitted,None]-a[:,omitted]
            b=base-difference*difference;dv=dot-difference*v[:,omitted]
            length=vv-v[:,omitted]**2;n=dim-1
            end=np.sum((y-linf)**2,axis=1)-(y[:,omitted]-linf[omitted])**2
        t=np.clip(dv/length,0,1)
        losses=np.maximum(b-2*t*dv+t*t*length,0)
        idx=losses.argmin(axis=1);rows=np.arange(len(y));alpha=t[rows,idx]
        best=losses[rows,idx];infinite=end<best
        selected_h=h[idx]+alpha*(h[idx+1]-h[idx]);selected_h[infinite]=np.inf
        zfit=a[idx]+alpha[:,None]*v[idx];zfit[infinite]=linf
        # Recompute residuals explicitly at selected point to avoid subtraction cancellation.
        mask=np.ones(dim,bool)
        if omitted>=0:mask[omitted]=False
        error=np.sqrt(np.mean((zfit[:,mask]-y[:,mask])**2,axis=1))
        if omitted<0:
            answer.update(h_mm=selected_h,rmse_ohm=error,prediction=zfit,
                          infinite=infinite,at_zero=selected_h==0)
        else:
            fold_h.append(selected_h);pred.append(zfit[:,omitted])
    if leave_one_out:
        ph=np.column_stack(pred);hs=np.column_stack(fold_h)
        answer.update(loo_rmse_ohm=np.sqrt(np.mean((ph-y)**2,axis=1)),
                      loo_predictions=ph,loo_h_mm=hs,
                      loo_h_min_mm=hs.min(axis=1),loo_h_max_mm=hs.max(axis=1))
    return answer
