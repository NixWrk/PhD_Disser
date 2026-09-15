"""C01 analytical reductions and a common bounded inverse test for every model."""
from pathlib import Path
import argparse,json,time
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.special import roots_legendre,j0
from c01_baseline_study import ROOT,SOURCE,HALF,OUT,sha,write
from transverse_geometry_analysis import analytic
from halfspace_study import from_frame,planar,electrodes
from geometry_validation_tests import recover_bounded
from compare_curved_two_tissue import stack_green

LABELS={'reference':'КТ/FEM: исходное лёгкое','planar_nominal':'Плоская аналитическая: номинальные точки',
 'planar_projected':'Плоская аналитическая: проекции контактов','averaged':'Среднее ядро: номинальные точки',
 'uniform_transverse':'Общая поперечная кривизна в КТ-теле','variable_transverse':'Переменная поперечная форма в КТ-теле',
 'm3h':'Постоянное сечение в полупространстве','m4h':'Переменная форма в полупространстве',
 'plane':'Плоская граница внутри КТ-тела','ellipsoid':'Эллипсоид внутри КТ-тела',
 'ellipsoid_depth':'Эллипсоид с привязкой по глубине','extended_lung':'Продолженное лёгкое внутри КТ-тела',
 'planar_nearest':'Плоская аналитическая: кратчайшая глубина','mean_h':'Средняя глубина по паре','averaged_projected':'Среднее ядро: проекции контактов',
 'central_finite':'Три слоя: первый центральный интервал','mean_finite':'Три слоя: средний первый интервал',
 'central_stack':'Все интервалы центрального луча','mean_stack':'Среднее ядро всех интервалов'}
CONTACTS={k:('узлы на КТ' if k in ['reference','uniform_transverse','variable_transverse','plane','ellipsoid','ellipsoid_depth','extended_lung'] else 'номинальные точки' if k in ['planar_nominal','averaged'] else 'проекции узлов') for k in LABELS}

def checked():
    g=json.loads((OUT/'geometry.json').read_text('utf-8'))
    for p,h in g['input_sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('Changed source '+p)
    if sha(OUT/'geometry_masks.mat')!=g['masks_sha256']:raise ValueError('Changed C01 masks')
    return g

def geometry_profiles():
    p=pd.read_csv(SOURCE/'central_profile.csv')
    depths=[np.asarray(json.loads(row.lung_crossings_d_mm))-row.skin_d_mm for row in p.itertuples()]
    assert all(len(d)%2==0 and np.all(np.diff(np.r_[0.,d])>0) for d in depths)
    return p,depths

def stack_values(q,r_mm,depths,nquad=192):
    # Integrate v=lambda*h_first. Each ray has its own scaling and physical interfaces.
    x,w=roots_legendre(nquad);v=12*(x+1);w=12*w
    ans=np.full(len(depths),1000/(2*np.pi*r_mm))
    for nbound in sorted(set(map(len,depths))-{0}):
        ids=[i for i,d in enumerate(depths) if len(d)==nbound]
        ds=np.stack([depths[i] for i in ids])/1000;h0=ds[:,0]
        lam=v[None,:]/h0[:,None];dz=np.diff(np.c_[np.zeros(len(ids)),ds],axis=1)
        z=np.ones_like(lam)
        for j in range(nbound-1,-1,-1):
            rho=1. if j%2==0 else q;th=np.tanh(lam*dz[:,j,None])
            z=rho*(z+rho*th)/(rho+z*th)
        ans[ids]+=np.sum(w[None,:]*(z-1)*j0(lam*r_mm/1000),axis=1)/(2*np.pi*h0)
    return ans

def point_two(q,r,h):
    h=np.asarray(h);n=np.arange(1,513)[:,None];k=(q-1)/(q+1)
    return (1000/r+2000*np.sum(k**n/np.hypot(r,2*n*h),axis=0))/(2*np.pi)

def extra_curves(q,g,profile,depths,points,nquad=192):
    s=profile.s_mm.to_numpy();h=profile.h_mm.to_numpy();ci=int(np.flatnonzero(s==0)[0])
    centre=[depths[ci]];result={k:np.zeros(len(points)) for k in ['planar_nearest','mean_h','averaged_projected','central_finite','mean_finite','central_stack','mean_stack']}
    for k,p in enumerate(points):
        for a,b,sign in [(1,0,1),(1,3,-1),(2,0,-1),(2,3,1)]:
            r=np.linalg.norm(p[a]-p[b])*1000;lo,hi=sorted([p[a,0]*1000,p[b,0]*1000])
            assert s[0]<=lo<hi<=s[-1]
            ss=np.r_[lo,s[(s>lo)&(s<hi)],hi]
            # Segment weights for the interpolated sampled scalar profile.
            def avg(values):return np.trapezoid(np.interp(ss,s,values),ss)/(hi-lo)
            hh=np.interp(ss,s,h);hm=np.inf if np.any(~np.isfinite(hh)) else np.trapezoid(hh,ss)/(hi-lo)
            first=[d[:2] for d in depths];present=np.array([bool(len(d)) for d in first]);fraction=avg(present.astype(float))
            if fraction>0:
                ha=avg(np.array([d[0] if len(d) else 0 for d in first]))/fraction
                ta=avg(np.array([d[1]-d[0] if len(d) else 0 for d in first]))/fraction
                meanfinite=fraction*stack_values(q,r,[np.array([ha,ha+ta])],nquad)[0]+(1-fraction)*1000/(2*np.pi*r)
            else:meanfinite=1000/(2*np.pi*r)
            vals=[point_two(q,r,[g['nearest_lung_distance_mm']])[0],point_two(q,r,[hm])[0],avg(point_two(q,r,h)),
                  stack_values(q,r,[centre[0][:2]],nquad)[0],meanfinite,
                  stack_values(q,r,centre,nquad)[0],avg(stack_values(q,r,depths,nquad))]
            for name,value in zip(result,vals):result[name][k]+=sign*value
    return result

def analytic_library():
    g=checked();profile,depths=geometry_profiles();sizes,points=electrodes();knots=np.geomspace(.9,18.,97)
    rows=[];t0=time.perf_counter()
    for q in knots:
        values=extra_curves(q,g,profile,depths,points)
        for name,values in values.items():
            rows.extend(dict(model=name,ratio=q,L_mm=L,f=f) for L,f in zip(sizes,values))
    df=pd.DataFrame(rows)
    qc=[]
    for q in [1.,4.,17.4]:
        a=extra_curves(q,g,profile,depths,points,192);b=extra_curves(q,g,profile,depths,points,384)
        qc.extend(dict(model=k,ratio=q,max_quad_difference=float(np.max(abs(a[k]-b[k])))) for k in a)
        d=next(d for d in depths if len(d)>=2)
        for r in [35.,105.]:
            err=abs(stack_values(q,r,[d],192)[0]-stack_green(r,d,1.,q))
            qc.append(dict(model='adaptive_stack_control',ratio=q,max_quad_difference=float(err)))
    qc=pd.DataFrame(qc);assert qc.max_quad_difference.max()<1e-6,qc
    df.to_csv(OUT/'analytic_library.csv',index=False)
    qc.to_csv(OUT/'analytic_quadrature_qc.csv',index=False)
    write(OUT/'analytic_identity.json',dict(source_sha256=sha(Path(__file__)),geometry_sha256=sha(OUT/'geometry.json'),
        library_sha256=sha(OUT/'analytic_library.csv'),qc_passed=True,seconds=time.perf_counter()-t0,
        rules=['Projected s,t points for seven newly migrated analytic variants.',
         'Raw CT profile replaces historical polynomial fit; missing lung is preserved.',
         'Arithmetic mean h is infinite when a pair spans a no-lung interval.',
         'Mean finite first interval is conditional on lung presence and weighted by coverage; remaining fraction homogeneous.',
         'Three layers mean soft/lung/soft: two tissue types, global lower layer is a diagnostic idealization.']))
    print('Analytic library',len(df),'rows; seconds',time.perf_counter()-t0,flush=True)

def validated_frame(df,bounds,knots=None):
    expected=np.array(json.loads((OUT/'geometry.json').read_text('utf-8'))['sizes_mm'])
    np.testing.assert_array_equal(np.sort(df.L_mm.unique()),expected)
    assert not df.duplicated(['ratio','L_mm']).any()
    assert np.isfinite(df[['f','df_dlogq']].to_numpy()).all()
    return from_frame(df,bounds,knots)

def load_models():
    g=checked();bounds=g['bounds_ohm_m'];models={};raw=pd.read_csv(SOURCE/'pem_library.csv')
    for name in ['reference','uniform_transverse','variable_transverse']:
        models[name]=validated_frame(raw[raw.model==name],bounds,np.array(g['knots'])).spline
    accept=json.loads((OUT/'fem_acceptance.json').read_text('utf-8'))
    for p,h in accept['outputs_sha256'].items():
        if sha(OUT/p)!=h:raise ValueError('Changed accepted FEM output '+p)
    if accept['geometry_sha256']!=sha(OUT/'geometry.json'):raise ValueError('FEM belongs to another geometry')
    new=pd.read_csv(OUT/'pem_library.csv')
    for name in g['model_names']:models[name]=validated_frame(new[new.model==name],bounds,np.array(g['knots'])).spline
    for name in ['m3h','m4h']:models[name]=validated_frame(pd.read_csv(HALF/f'library_{name}.csv'),bounds,np.array(g['knots'])).spline
    sizes,points=electrodes();np.testing.assert_array_equal(sizes,g['sizes_mm']);profile,_=geometry_profiles()
    for name in ['planar_nominal','planar_projected','averaged']:
        rows=[]
        for q in g['knots']:
            f,d=planar(q,points,g['h_centre_mm']/1000) if name=='planar_projected' else analytic(q,sizes,profile,g['h_centre_mm'],'averaged' if name=='averaged' else 'planar')
            rows.extend(dict(ratio=q,L_mm=L,f=z,df_dlogq=j) for L,z,j in zip(sizes,f,d))
        models[name]=validated_frame(pd.DataFrame(rows),bounds).spline
    ai=json.loads((OUT/'analytic_identity.json').read_text('utf-8'))
    assert ai['qc_passed'] and ai['geometry_sha256']==sha(OUT/'geometry.json') and ai['library_sha256']==sha(OUT/'analytic_library.csv')
    assert ai['source_sha256']==sha(Path(__file__)), 'Rebuild analytic library for changed source'
    raw=pd.read_csv(OUT/'analytic_library.csv')
    for name,group in raw.groupby('model'):
        assert not group.duplicated(['ratio','L_mm']).any()
        t=group.pivot(index='ratio',columns='L_mm',values='f').sort_index();np.testing.assert_array_equal(t.columns,g['sizes_mm']);assert np.isfinite(t.to_numpy()).all()
        models[name]=CubicSpline(np.log(t.index.to_numpy()),t.to_numpy(),axis=0,extrapolate=False)
    assert set(models)==set(LABELS)
    for name,sp in models.items():
        assert np.isfinite(sp(np.log(4.))).all() and sp(np.log(4.)).shape==(9,), name
    return g,{k:models[k] for k in LABELS}

def analyse():
    g,models=load_models();bounds=g['bounds_ohm_m'];sizes=np.array(g['sizes_mm']);cases=pd.read_csv(OUT/'synthetic_cases.csv')
    direct=pd.read_csv(OUT/'reference_holdouts.csv');synth=[];forward=[];interp=[];synthetic_search=[]
    for case in cases.itertuples():
        q=case.rho2/case.rho1;t=direct[np.isclose(direct.ratio,q,rtol=1e-11,atol=1e-12)].sort_values('L_mm')
        np.testing.assert_array_equal(t.L_mm,sizes);y=case.rho1*t.f.to_numpy()
        for name,spline in models.items():
            fit,pred,J=recover_bounded(spline,y,bounds)
            synth.append(dict(case=case.case,model=name,rho1_true=case.rho1,rho2_true=case.rho2,error1_pct=100*(fit['rho1']/case.rho1-1),error2_pct=100*(fit['rho2']/case.rho2-1),**fit))
            fine,_,_=recover_bounded(spline,y,bounds,grid_size=1601)
            synthetic_search.append(dict(case=case.case,model=name,delta_rho1=fine['rho1']-fit['rho1'],delta_rho2=fine['rho2']-fit['rho2'],delta_rmse=fine['train_rmse']-fit['train_rmse']))
            delta=case.rho1*spline(np.log(q))-y
            forward.append(dict(case=case.case,model=name,rmse_ohm=float(np.sqrt(np.mean(delta**2))),relative_error_pct=float(100*np.linalg.norm(delta)/np.linalg.norm(y))))
        interp.append(dict(case=case.case,max_abs_ohm=float(np.max(abs(case.rho1*models['reference'](np.log(q))-y)))))
    pd.DataFrame(synthetic_search).to_csv(OUT/'synthetic_search_qc.csv',index=False)
    pd.DataFrame(synth).to_csv(OUT/'inverse_recovery.csv',index=False);pd.DataFrame(forward).to_csv(OUT/'forward_errors.csv',index=False)
    pd.DataFrame(interp).to_csv(OUT/'reference_interpolation_qc.csv',index=False)
    obs=pd.read_csv(OUT/'experimental_observations.csv');np.testing.assert_array_equal(obs.L_mm,sizes);y=obs.Z_inhale_hold_ohm.to_numpy()
    fits=[];predictions=[];holdouts=[];perturb=[];search=[]
    for name,spline in models.items():
        fit,pred,J=recover_bounded(spline,y,bounds);fits.append(dict(model=name,**fit))
        predictions.extend(dict(model=name,L_mm=L,observed_ohm=yy,predicted_ohm=zz,residual_ohm=zz-yy) for L,yy,zz in zip(sizes,y,pred))
        fine,_,_=recover_bounded(spline,y,bounds,grid_size=1601)
        search.append(dict(model=name,delta_rho1=fine['rho1']-fit['rho1'],delta_rho2=fine['rho2']-fit['rho2'],delta_rmse=fine['train_rmse']-fit['train_rmse']))
        for k,L in enumerate(sizes):
            f,z,_=recover_bounded(spline,y,bounds,indices=np.delete(np.arange(len(y)),k))
            holdouts.append(dict(model=name,excluded_L_mm=L,rho1=f['rho1'],rho2=f['rho2'],prediction_ohm=z[k],observation_ohm=y[k],error_ohm=z[k]-y[k]))
            for sign in [-1,1]:
                yy=y.copy();yy[k]*=1+sign*.01;f,_,_=recover_bounded(spline,yy,bounds)
                perturb.append(dict(model=name,L_mm=L,sign=sign,rho1=f['rho1'],rho2=f['rho2'],delta_rho1=f['rho1']-fit['rho1'],delta_rho2=f['rho2']-fit['rho2']))
    for name,rows in [('experimental_fits',fits),('experimental_predictions',predictions),('conditional_holdouts',holdouts),('experimental_perturbations',perturb),('search_resolution_check',search)]:pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    # Analytical interpolation checked independently at all65 real-valued q; no truth from its spline.
    profile,depths=geometry_profiles();_,points=electrodes();qc=[]
    for q in np.unique(cases.rho2/cases.rho1):
        vals=extra_curves(q,g,profile,depths,points)
        for name,f in vals.items():qc.append(dict(model=name,ratio=q,max_unit_rho_error=float(np.max(abs(models[name](np.log(q))-f)))))
    pd.DataFrame(qc).to_csv(OUT/'analytic_interpolation_qc.csv',index=False)
    timing=[]
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=6):
        for name,spline in models.items():
            for repeat in range(7):
                start=time.perf_counter()
                for _ in range(300):
                    ff=spline(np.log(4.));dd=spline(np.log(4.),1)
                    zz=4*ff;jj=np.column_stack([ff-dd,dd/4])
                timing.append(dict(model=name,task='nine_Z_and_J_cached',repeat=repeat,seconds=(time.perf_counter()-start)/300))
                start=time.perf_counter();recover_bounded(spline,y,bounds)
                timing.append(dict(model=name,task='bounded_inverse_cached',repeat=repeat,seconds=time.perf_counter()-start))
    pd.DataFrame(timing).to_csv(OUT/'library_timings.csv',index=False)
    write(OUT/'analysis_summary.json',dict(candidate='C01',models=len(models),states=len(cases),inverse_count=len(synth),
       max_reference_interpolation_ohm=max(x['max_abs_ohm'] for x in interp),
       max_analytic_interpolation_unit_rho=max(x['max_unit_rho_error'] for x in qc),
       source_sha256=sha(Path(__file__)),status='exploratory_hypothesis_not_validated',
       scope='known geometry; baseline PEM or nominal/projected analytic points; CEM handled separately'))
    print('Completed',len(synth),'synthetic fits and',len(fits),'experimental fits',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['analytic','analyse']);a=p.parse_args()
    analytic_library() if a.stage=='analytic' else analyse()
