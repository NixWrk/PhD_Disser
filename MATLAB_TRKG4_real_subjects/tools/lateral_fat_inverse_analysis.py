"""Validate the fixed-montage inverse experiment; summarize conditional estimates."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lateral_fat_3d import OUT as STUDY_OUT,PIPE,BASE,FAT,sha,read,write
OUT=STUDY_OUT/'capped100'


def analyze():
    fits=[];pred=[];jac=[];profiles=[];profile_solves=0
    for state,count in [('inhale',6),('exhale',5)]:
        e=read(OUT/f'execution_{state}.json');assert e['complete'] and e['fits']==count
        assert e['physiological_bounds_used'] is False
        profile=pd.read_csv(OUT/f'profile_{state}.csv');assert profile.rho2.tolist()==[.1,1,5,10,20,40,60,100]
        assert sha(OUT/f'profile_{state}.csv')==e['profile_sha256'];profiles.append(profile);profile_solves+=e['CEM_solves_profile']
        assert e['identity']['worker_sha256']==sha(PIPE/'src/run_lateral_fat_inverse.m')
        deps={'matrix_sha256':FAT/'refined/matrices_q4096.mat','coarse_matrix_sha256':FAT/'refined/matrices_q1024.mat',
            'base_sha256':BASE/'prepared_fem.mat','forward_plan_sha256':FAT/'refined/plan.json',
            'observations_sha256':PIPE/'output/exploratory/lateral_real_application_20260915/observations.csv',
            'literature_sha256':FAT/'refined/literature_values.csv'}
        for key,path in deps.items():assert e['identity'][key]==sha(path),key
        for prefix,key,dest in [('fits','fit_sha256',fits),('predictions','predictions_sha256',pred),('jacobian','jacobian_sha256',jac)]:
            p=OUT/f'{prefix}_{state}.csv';assert sha(p)==e[key];dest.append(pd.read_csv(p))
    a=pd.concat(fits,ignore_index=True);b=pd.concat(pred,ignore_index=True);j=pd.concat(jac,ignore_index=True)
    assert len(a)==11 and len(b)==99 and len(j)==99 and not a.case_id.duplicated().any()
    obs=pd.read_csv(PIPE/'output/exploratory/lateral_real_application_20260915/observations.csv').sort_values('L_mm')
    diagnostic=[]
    for r in a.itertuples():
        g=b[b.case_id==r.case_id].sort_values('L_mm');q=j[j.case_id==r.case_id].sort_values('L_mm')
        assert g.L_mm.tolist()==obs.L_mm.tolist()
        if not r.synthetic:np.testing.assert_allclose(g.target_Z,obs['Z_'+r.state+'_hold_ohm'],rtol=0,atol=1e-11)
        np.testing.assert_allclose(g.predicted_Z-g.target_Z,g.residual_Z,atol=1e-10)
        assert abs(np.sqrt(np.mean(g.residual_Z**2))-r.rmse_ohm)<1e-9
        err=np.abs(g.predicted_Z-g.coarse_Z)
        assert bool(np.all(err<=.02+.001*np.abs(g.predicted_Z)))==bool(r.quadrature_within_tolerance)
        assert r.exitflag>0
        case=read(OUT/(r.case_id+'.json'));assert case['identity']==e['identity'] and case['physiological_bounds_used'] is False
        np.testing.assert_allclose(case['fitted_rho_all'],[r.rho1,r.rho2,r.rho_bone,r.rho_heart,r.rho3_fat],rtol=1e-12)
        assert abs(min(case['start_objectives'])-9*r.rmse_ohm**2)<1e-7
        limits=np.asarray(case['bounds_ohm_m']);np.testing.assert_allclose(limits,np.tile([.1,100],(r.free_parameters,1)));parameters=np.array([r.rho1,r.rho2,r.rho3_fat])[:r.free_parameters]
        assert np.all(parameters>=limits[:,0]*(1-1e-12)) and np.all(parameters<=limits[:,1]*(1+1e-12))
        boundary=np.any(np.abs(np.log(parameters[:,None])-np.log(limits))<1e-4,axis=1)
        assert boundary.tolist()==[bool(r.bound_rho1),bool(r.bound_rho2),bool(r.bound_rho3)][:r.free_parameters]
        A=q[['dZ_dlnrho1','dZ_dlnrho2','dZ_dlnrho3']].to_numpy()
        fixed=A[:,:2];g2=np.linalg.norm(np.linalg.pinv(fixed)[1]) if np.linalg.matrix_rank(fixed)==2 else np.inf
        g3=np.linalg.norm(np.linalg.pinv(A)[1]) if np.linalg.matrix_rank(A)==3 else np.inf
        diagnostic.append(dict(case_id=r.case_id,rank_three=int(np.linalg.matrix_rank(A)),
            condition_two=float(np.linalg.cond(fixed)),condition_three=float(np.linalg.cond(A)),
            rho2_gain_fixed_fat_per_ohm=float(g2),rho2_gain_free_fat_per_ohm=float(g3),
            rho2_gain_ratio=float(g3/g2) if np.isfinite(g2) else np.nan,singular_min_three_ohm=float(np.linalg.svd(A,compute_uv=False)[-1])))
    synthetic=a[a.synthetic==1].iloc[0]
    lit=pd.read_csv(FAT/'refined/literature_values.csv')
    fat50=float(lit[(lit.tissue=='Fat')&(lit.frequency_hz==50000)].rho_ohm_m.iloc[0])
    synthetic_error=np.array([synthetic.rho1/4-1,synthetic.rho2/12-1,synthetic.rho3_fat/fat50-1])
    assert np.max(np.abs(synthetic_error))<1e-3
    d=read(OUT/'derivative_qc.json');assert d['identity']==e['identity'];assert d['complete'] and max(d['relative_column_errors'])<1e-4
    a['rho1_in_literature']=(a.rho1>=1.3774104683195594)&(a.rho1<=10)
    lower={50000:3.8172830517819216,100000:3.681694081}
    a['rho2_in_literature']=(a.rho2>=a.frequency_hz.map(lower))&(a.rho2<=24.074073806584366)
    profile=pd.concat(profiles,ignore_index=True);assert ((profile.rho1>=.1*(1-1e-10))&(profile.rho1<=100*(1+1e-10))).all()
    profile.to_csv(OUT/'profile.csv',index=False)
    for r in profile.itertuples():
        q=read(OUT/f'profile_{r.state}_rho2_{r.rho2:g}.json')
        assert abs(np.sqrt(np.mean(np.array(q['residual'])**2))-r.rmse_ohm)<1e-8
    a.to_csv(OUT/'fits.csv',index=False);b.to_csv(OUT/'predictions.csv',index=False);pd.DataFrame(diagnostic).to_csv(OUT/'identifiability.csv',index=False)
    fields=['fits_inhale.csv','fits_exhale.csv','predictions_inhale.csv','predictions_exhale.csv','jacobian_inhale.csv','jacobian_exhale.csv',
        'execution_inhale.json','execution_exhale.json','derivative_qc.json','profile_inhale.csv','profile_exhale.csv']
    fields += [str(x)+'.json' for x in a.case_id] + sorted(x.name for x in OUT.glob('profile_*_rho2_*.json'))
    write(OUT/'inverse_qc.json',dict(complete=True,experimental_fits=10,synthetic_fits=1,starts_per_fit=2,
        total_CEM_solves=int(a.CEM_solves.sum()+9*len(a)+9+63+profile_solves),
        quadrature_all_within_tolerance=bool(a.quadrature_within_tolerance.all()),
        quadrature_max_difference_ohm=float(a.quadrature_max_difference_ohm.max()),
        max_two_start_log_difference=float(a.two_start_log_difference.max()),
        max_relative_residual=float(a.max_relative_residual.max()),max_reciprocity_ohm=float(a.max_reciprocity_ohm.max()),
        synthetic_relative_error=synthetic_error.tolist(),synthetic_truth=[4,12,fat50],
        derivative_relative_errors=d['relative_column_errors'],
        fat_interval_kind='fixed plus/minus25 scenarios only; joint free-fat fit uses user-selected 0.1 to 100 ohm m bounds',
        physiological_bounds_used=False,profile_rho2_values=sorted(profile.rho2.unique().tolist()),
        profile_quadrature_all_within_tolerance=bool(profile.quadrature_within_tolerance.all()),
        profile_max_quadrature_difference_ohm=float(profile.quadrature_max_difference_ohm.max()),
        no_new_localization=True,physical_validation=False,global_mesh_convergence=False,
        inputs_sha256={n:sha(OUT/n) for n in fields},
        outputs_sha256={n:sha(OUT/n) for n in ['fits.csv','predictions.csv','identifiability.csv','profile.csv']},source_sha256=sha(__file__)))
    print(a[['case_id','rho1','rho2','rho3_fat','rmse_ohm','bound_rho2','bound_rho3','quadrature_within_tolerance']].to_string(index=False))
    assert a.quadrature_within_tolerance.all(),'Quadrature check at recovered resistivities needs investigation.'


def figure():
    qc=read(OUT/'inverse_qc.json');assert qc['complete']
    for n,h in {**qc['inputs_sha256'],**qc['outputs_sha256']}.items():assert sha(OUT/n)==h,n
    a=pd.read_csv(OUT/'fits.csv');b=pd.read_csv(OUT/'predictions.csv')
    fig=make_subplots(rows=2,cols=2,vertical_spacing=.20,horizontal_spacing=.09,
        subplot_titles=['Вдох: эксперимент и подгонка','Выдох: эксперимент и подгонка',
                        'Вдох: расчёт минус эксперимент','Выдох: расчёт минус эксперимент'])
    scenarios=[(50000,1,2,'50 кГц · ρ₃ по литературе'),(100000,1,2,'100 кГц · ρ₃ по литературе'),
        (50000,.75,2,'50 кГц · ρ₃ ниже на 25%'),(50000,1.25,2,'50 кГц · ρ₃ выше на 25%'),
        (50000,1,3,'50 кГц · подбор ρ₁, ρ₂, ρ₃')]
    for index,(freq,factor,npar,title) in enumerate(scenarios):
        for col,state in enumerate(['inhale','exhale'],1):
            r=a[(a.frequency_hz==freq)&(a.fat_factor==factor)&(a.free_parameters==npar)&(a.state==state)&(a.synthetic==0)].iloc[0]
            g=b[b.case_id==r.case_id].sort_values('L_mm')
            for field,label,color,row in [('target_Z','Эксперимент','#cf3f3f',1),('predicted_Z','Подгонка пяти тканей','#c88705',1),('residual_Z','Остаточная невязка','#326eaa',2)]:
                fig.add_trace(go.Scatter(x=g.L_mm,y=g[field],name=label,legendgroup=label,showlegend=col==1,
                    mode='lines+markers',visible=index==0,line=dict(color=color),
                    customdata=np.tile([r.rho1,r.rho2,r.rho3_fat,r.rmse_ohm],(9,1)),
                    hovertemplate='L=%{x} мм; Z=%{y:.3f} Ом<br>ρ₁=%{customdata[0]:.3f}; ρ₂=%{customdata[1]:.3f}; ρ₃=%{customdata[2]:.3f} Ом·м<br>RMSE=%{customdata[3]:.3f} Ом<extra>%{fullData.name}</extra>'),row=row,col=col)
    for c in [1,2]:
        fig.update_yaxes(title='Z, Ом' if c==1 else None,row=1,col=c)
        fig.update_yaxes(title='ΔZ, Ом' if c==1 else None,zeroline=True,zerolinecolor='#555',row=2,col=c)
        fig.update_xaxes(title='Размер сборки L, мм',row=2,col=c)
    fig.update_layout(template='plotly_white',width=1280,height=850,margin=dict(l=80,r=40,t=130,b=110),
        title=dict(text='Восстановление сопротивлений · '+scenarios[0][3],x=.5,y=.99,yanchor='top'),
        legend=dict(orientation='h',x=.5,xanchor='center',y=-.15),hovermode='x unified',
        updatemenus=[dict(x=0,y=1.12,buttons=[dict(label=s[3],method='update',args=[{'visible':[k//6==i for k in range(30)]},
           {'title.text':'Восстановление сопротивлений · '+s[3]}]) for i,s in enumerate(scenarios)])])
    return fig


def profile_figure():
    a=pd.read_csv(OUT/'profile.csv')
    fig=make_subplots(rows=1,cols=2,subplot_titles=['Вдох: профиль невязки по ρ₂','Выдох: профиль невязки по ρ₂'])
    for c,state in enumerate(['inhale','exhale'],1):
        g=a[a.state==state].sort_values('rho2')
        fig.add_trace(go.Scatter(x=g.rho2,y=g.rmse_ohm,mode='lines+markers',name='Повторная подгонка ρ₁',showlegend=c==1,
            line=dict(color='#256d9f'),customdata=g[['rho1','quadrature_max_difference_ohm']],
            hovertemplate='ρ₂=%{x:.4g} Ом·м<br>RMSE=%{y:.5f} Ом<br>ρ₁=%{customdata[0]:.4f} Ом·м<br>Проверка интегрирования: %{customdata[1]:.4f} Ом<extra></extra>'),row=1,col=c)
        fig.add_vrect(x0=3.8172830517819216,x1=24.074073806584366,fillcolor='#79b678',opacity=.16,line_width=0,row=1,col=c)
        fig.update_xaxes(type='log',title='Заданное ρ₂, Ом·м; логарифмическая шкала',row=1,col=c)
        fig.update_yaxes(title='RMSE, Ом' if c==1 else None,row=1,col=c)
    small={f'yaxis{c if c>1 else ""}.range':[float(a[a.state==state].rmse_ohm.min())-.1,float(a[a.state==state].rmse_ohm.min())+1.5] for c,state in enumerate(['inhale','exhale'],1)}
    fig.update_layout(template='plotly_white',width=1280,height=550,margin=dict(l=70,r=35,t=110,b=100),
        title=dict(text='Поиск до 100 Ом·м · 50 кГц · ρ₃ по литературе',x=.5),
        legend=dict(orientation='h',x=.5,xanchor='center',y=-.25),
        updatemenus=[dict(type='buttons',direction='right',x=.35,y=-.18,buttons=[dict(label='Вся невязка',method='relayout',args=[{'yaxis.autorange':True,'yaxis2.autorange':True}]),dict(label='Минимум крупнее',method='relayout',args=[small])])],
        annotations=list(fig.layout.annotations)+[dict(text='Зелёная полоса — литературный ориентир для оценки результата, без ограничения поиска.',
        x=.5,y=1.13,xref='paper',yref='paper',showarrow=False)])
    return fig


if __name__=='__main__':
    analyze();figure().write_html(OUT/'inverse_comparison.html',include_plotlyjs=True,config={'displaylogo':False})
    profile_figure().write_html(OUT/'rho2_profile.html',include_plotlyjs=True,config={'displaylogo':False})
