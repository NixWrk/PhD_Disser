"""Scientific figures for 20.16; numerical sources are local frozen outputs."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from skimage.measure import marching_cubes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transverse_geometry_study import ROOT,OUT,smooth_window,read_surface,distance_grid
from transverse_geometry_analysis import LABELS

COLORS={'reference':'#1c2630','planar':'#ad6b12','averaged':'#8667a7','uniform_transverse':'#3184b6','variable_transverse':'#c84f40'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,'axes.labelsize':11,'figure.facecolor':'white'})

def blended(q):
    S,T,D=np.meshgrid(q['s'],q['t'],q['d'],indexing='ij')
    w=smooth_window(S,95,115)*smooth_window(T,45,65)*smooth_window(D-85,95,135)
    w=np.where((D>-10)&(D<220),w,0)
    fields={'reference':q['reference']}
    for key,name in [('uniform','uniform_transverse'),('variable','variable_transverse')]:
        fields[name]=np.where(w>0,(1-w)*q['reference']+w*q[key],q['reference'])
    return fields

def mesh_trace(field,s,t,d,color,name,opacity=.45):
    # The finite plot box is not a physical bottom boundary.
    v,f,_,_=marching_cubes(field.astype(np.float32),level=0,spacing=(s[1]-s[0],t[1]-t[0],d[1]-d[0]))
    v+=np.array([s[0],t[0],d[0]])
    keep=np.all((abs(v[f,0])<=95)&(abs(v[f,1])<=45)&(v[f,2]>=-15)&(v[f,2]<=180),axis=1)
    f=f[keep]
    return go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],color=color,opacity=opacity,name=name,showlegend=False,hoverinfo='skip',flatshading=False)

def geometry_figures(out):
    dest=out/'figures';dest.mkdir(exist_ok=True)
    q=np.load(out/'geometry_fields.npz');fields=blended(q);s,t,d=q['s'],q['t'],q['d']
    inp=loadmat(out/'geometry_input.mat',simplify_cells=True);xyz=inp['nodes_mm'];frame=q['frame'];centre=q['centre']
    body_cache=out/'body_distance.npz'
    if body_cache.exists():body=np.load(body_cache)['phi']
    else:
        poly=read_surface(ROOT/'data/nik/stl/body_solid_v3_1mm_r10_volume_fill.stl',centre,frame)
        body=distance_grid(poly,s,t,d);np.savez_compressed(body_cache,phi=body)
    # Actual C01 central and off-axis sections. Independent model differences are visible.
    fig,axes=plt.subplots(1,3,figsize=(15,5.6),sharex=True,sharey=True,layout='constrained')
    for ax,tval in zip(axes,[-40,0,40]):
        ti=np.argmin(abs(t-tval));occupied=np.where(body[:,ti,:].T<=0,(fields['reference'][:,ti,:].T<0).astype(float),np.nan)
        from matplotlib.colors import ListedColormap
        ax.pcolormesh(s,d,occupied,cmap=ListedColormap(['#f6dfbb','#c3e6f4']),vmin=0,vmax=1,shading='auto',rasterized=True)
        ax.contour(s,d,body[:,ti,:].T,levels=[0],colors='#8d652b',linewidths=1.4)
        for name,ls in [('reference','-'),('uniform_transverse','--'),('variable_transverse',':')]:
            ax.contour(s,d,fields[name][:,ti,:].T,levels=[0],colors=COLORS[name],linestyles=ls,linewidths=1.5)
        ax.set(xlim=(-95,95),ylim=(180,-20),xlabel='s, мм',title=f'Поперечное смещение t = {tval} мм');ax.grid(alpha=.15)
    axes[0].set_ylabel('d внутрь от касательной плоскости, мм')
    fig.legend(handles=[Line2D([0],[0],color=COLORS[n],ls=ls,label=LABELS[n]) for n,ls in [('reference','-'),('uniform_transverse','--'),('variable_transverse',':')]],loc='outside lower center',ncol=3)
    fig.savefig(dest/'01_ct_sections.png',dpi=160);plt.close(fig)
    # Thickness used by the analytical candidates, with absent intersections retained.
    p=pd.read_csv(out/'central_profile.csv');geom=json.loads((out/'geometry.json').read_text(encoding='utf-8'))
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained');finite=np.isfinite(p.h_mm)
    ax.plot(p.s_mm,np.where(finite,p.h_mm,np.nan),color='#245f88',label='Расстояние кожа–первое лёгкое по направлению d')
    ax.axhline(geom['h_centre_mm'],color=COLORS['planar'],ls='--',label=f'Постоянная h = {geom["h_centre_mm"]:.1f} мм')
    for x in p.loc[~finite,'s_mm']:ax.axvspan(x-.5,x+.5,color='#df8888',alpha=.3,lw=0)
    ax.set(xlabel='s вдоль сборки, мм',ylabel='h, мм',title='Глубина лёгкого в центральном срезе C01');ax.grid(alpha=.2)
    ax.legend(loc='best');fig.savefig(dest/'02_depth_profile.png',dpi=160);plt.close(fig)
    # Four actual geometry representations; M2 explicitly depicts separate flat kernels.
    fig=make_subplots(rows=2,cols=2,specs=[[{'type':'scene'}]*2]*2,subplot_titles=[LABELS[n] for n in ['planar','averaged','uniform_transverse','variable_transverse']],vertical_spacing=.08)
    contacts=pd.read_csv(out/'contacts.csv') if (out/'contacts.csv').exists() else None
    if contacts is not None:
        e=contacts[contacts.L_mm==140];electrodes=(xyz[e.point_node.to_numpy(dtype=int)-1]-centre)@frame
    else:
        e=pd.read_csv(out/'source_electrodes.csv');e=e[e.L_mm==140];electrodes=(e[['patch_centroid_x_mm','patch_centroid_y_mm','patch_centroid_z_mm']].to_numpy()-centre)@frame
    for index,name in enumerate(['planar','averaged','uniform_transverse','variable_transverse']):
        row,col=index//2+1,index%2+1
        if name in ['planar','averaged']:
            xx=np.array([-95,95]);yy=np.array([-45,45]);fig.add_trace(go.Surface(x=xx,y=yy,z=np.zeros((2,2)),colorscale=[[0,'#d5ba8e'],[1,'#d5ba8e']],showscale=False,opacity=.28,hoverinfo='skip'),row,col)
            if name=='planar':fig.add_trace(go.Surface(x=xx,y=yy,z=np.full((2,2),geom['h_centre_mm']),colorscale=[[0,'#4bb3d1'],[1,'#4bb3d1']],showscale=False,opacity=.7,hoverinfo='skip'),row,col)
            else:
                for x in np.arange(-60,61,20):
                    h=np.interp(x,p.s_mm,p.h_mm)
                    if np.isfinite(h):fig.add_trace(go.Surface(x=np.array([x-6,x+6]),y=yy,z=np.full((2,2),h),colorscale=[[0,'#4bb3d1'],[1,'#4bb3d1']],showscale=False,opacity=.8,hoverinfo='skip'),row,col)
            ep=np.column_stack([[-70,-35,35,70],np.zeros(4),np.zeros(4)])
        else:
            fig.add_trace(mesh_trace(fields[name],s,t,d,'#3babc7','Граница лёгкого',.60),row,col)
            fig.add_trace(mesh_trace(body,s,t,d,'#c6a66d','Кожа',.20),row,col);ep=electrodes
        fig.add_trace(go.Scatter3d(x=ep[:,0],y=ep[:,1],z=ep[:,2],mode='markers+text',text=['I+','V+','V−','I−'],textposition='top center',marker={'size':4,'color':'#202d3a'},textfont={'size':12},showlegend=False,name='Электроды'),row,col)
        fig.update_scenes(xaxis={'title':'s, мм','range':[-95,95]},yaxis={'title':'t, мм','range':[-45,45]},zaxis={'title':'d, мм','range':[180,-20]},aspectmode='data',camera={'eye':{'x':1.5,'y':-2.0,'z':1.1},'projection':{'type':'orthographic'}},row=row,col=col)
    fig.update_layout(height=1000,width=None,margin={'l':5,'r':5,'t':55,'b':10},template='plotly_white')
    fig.write_json(dest/'03_four_models_3d.json')
    fig.write_html(dest/'03_four_models_3d.html',include_plotlyjs=True,full_html=True,config={'responsive':True,'displaylogo':False})
    # Static volume companion: common central profile in two transverse views.
    fig=plt.figure(figsize=(12,5.5),layout='constrained')
    for idx,name in enumerate(['uniform_transverse','variable_transverse']):
        ax=fig.add_subplot(1,2,idx+1,projection='3d');z=fields[name]
        for tv in [-40,-20,0,20,40]:
            ti=np.argmin(abs(t-tv));cs=plt.figure();aa=cs.add_subplot();c=aa.contour(s,d,z[:,ti,:].T,levels=[0]);segments=c.allsegs[0];plt.close(cs)
            for v in segments:
                keep=(abs(v[:,0])<=95)&(v[:,1]>=0)&(v[:,1]<=180)
                vv=v.copy();vv[~keep]=np.nan;ax.plot(vv[:,0],np.full(len(vv),tv),vv[:,1],color=COLORS[name],alpha=.8 if tv==0 else .4,lw=2 if tv==0 else 1)
        ax.scatter(electrodes[:,0],electrodes[:,1],electrodes[:,2],c='#202d3a',s=25)
        ax.set(xlim=(-95,95),ylim=(-45,45),zlim=(180,-20),xlabel='s, мм',ylabel='t, мм',zlabel='d, мм',title=LABELS[name]);ax.view_init(elev=20,azim=-65)
    fig.savefig(dest/'04_transverse_profiles_3d.png',dpi=160);plt.close(fig)

def result_figures(out):
    dest=out/'figures';r=pd.read_csv(out/'inverse_recovery.csv');models=['planar','averaged','uniform_transverse','variable_transverse'];truth1=sorted(r.rho1_true.unique());truth2=sorted(r.rho2_true.unique())
    fig,axes=plt.subplots(2,4,figsize=(15,7),layout='constrained')
    maximum=max(10,float(np.max(abs(r[['error1_pct','error2_pct']].to_numpy()))))
    for k in [0,1]:
        for j,name in enumerate(models):
            a=r[r.model==name].pivot(index='rho2_true',columns='rho1_true',values=f'error{k+1}_pct').sort_index();ax=axes[k,j]
            im=ax.imshow(a,origin='lower',cmap='RdBu_r',vmin=-maximum,vmax=maximum,aspect='auto')
            for ii in range(len(truth2)):
                for jj in range(len(truth1)):ax.text(jj,ii,f'{a.iloc[ii,jj]:.3f}' if abs(a.iloc[ii,jj])<1 else f'{a.iloc[ii,jj]:.1f}',ha='center',va='center',fontsize=9,color='white' if abs(a.iloc[ii,jj])>.65*maximum else 'black')
            ax.set(xticks=range(len(truth1)),xticklabels=truth1,yticks=range(len(truth2)),yticklabels=truth2,xlabel='Заданное ρ₁, Ом·м',title=LABELS[name] if k==0 else '')
            if j==0:ax.set_ylabel(f'Ошибка ρ{k+1}, %\nЗаданное ρ₂, Ом·м')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Знаковая ошибка параметра, %',shrink=.85);fig.savefig(dest/'05_inverse_errors.png',dpi=170);plt.close(fig)
    p=pd.read_csv(out/'inverse_predictions.csv');sample=r.iloc[np.argmin(abs(r.rho1_true-4)+abs(r.rho2_true-18))]['case'];p=p[p.case==sample]
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    ref=p[p.model=='reference'];axes[0].plot(ref.L_mm,ref.observed,'ko',label='Заданные КТ/FEM-импедансы')
    for name in models:
        q=p[p.model==name];axes[0].plot(q.L_mm,q.predicted,'-',color=COLORS[name],label=LABELS[name]);axes[1].plot(q.L_mm,q.predicted-q.observed,'o-',color=COLORS[name],label=LABELS[name])
    for name in models[2:]:
        q=p[p.model==name];axes[2].plot(q.L_mm,q.predicted-q.observed,'o-',color=COLORS[name])
    axes[2].set(ylabel='Модель − КТ/FEM, Ом',title='Остатки пространственных моделей');axes[2].axhline(0,color='gray',lw=.7)
    for ax in axes:ax.set_xlabel('Размер сборки L, мм');ax.grid(alpha=.2)
    axes[0].set(ylabel='Импеданс, Ом',title='Подгонка при заданных ρ₁ = 4 и ρ₂ = 18 Ом·м');axes[1].set(ylabel='Модель − КТ/FEM, Ом',title='Остатки после восстановления сопротивлений');axes[1].axhline(0,color='gray',lw=.7)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3);fig.savefig(dest/'06_impedance_fit.png',dpi=160);plt.close(fig)
    x=pd.read_csv(out/'linked_scenarios.csv');subset=['fixed_deep','h_minus2_deep','h_plus2_deep','rho1_deep','fixed_quiet','h_minus1_quiet','h_plus1_quiet'];labels=['Глубокое: h постоянна','Глубокое: −2 мм','Глубокое: +2 мм','Глубокое: ρ₁ +1%','Обычное: h постоянна','Обычное: −1 мм','Обычное: +1 мм']
    fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True,layout='constrained')
    for i,name in enumerate(models):
        a=x[x.model==name].set_index('scenario').loc[subset];offset=(i-1.5)*.16
        axes[0].bar(np.arange(7)+offset,a.delta2_error_pct,.16,label=LABELS[name],color=COLORS[name]);axes[1].bar(np.arange(7)+offset,a.pulse2_error_pct,.16,color=COLORS[name])
    axes[0].set(ylabel='Ошибка дыхательного Δρ₂, %',title='Последовательное восстановление на заданных сценариях');axes[1].set(ylabel='Ошибка пульсового δρ₂, %')
    for i,name in enumerate(models[2:]):
        a=x[x.model==name].set_index('scenario').loc[subset];axes[2].bar(np.arange(7)+(i-.5)*.2,a.pulse2_error_pct,.2,color=COLORS[name])
    axes[2].set(ylabel='Ошибка пульсового δρ₂, %',title='Пульсовая оценка: пространственные модели в увеличенном масштабе',xticks=range(7),xticklabels=labels)
    for ax in axes:ax.axhline(0,color='gray',lw=.8);ax.grid(axis='y',alpha=.2)
    plt.setp(axes[2].get_xticklabels(),rotation=18,ha='right');axes[0].legend(ncol=2);fig.savefig(dest/'07_linked_errors.png',dpi=160);plt.close(fig)
    q=pd.read_csv(out/'perturbations.csv');fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
    for name in models:
        vals=q[q.model==name].groupby('L_mm').change2_pct.apply(lambda x:np.max(abs(x)))
        ax.plot(vals.index,vals,'o-',color=COLORS[name],label=LABELS[name])
    ax.set(xlabel='Размер единственной возмущённой сборки, мм',ylabel='Максимальное изменение оценки ρ₂, %',title='Одноканальное возмущение импеданса ±1%: 16 исходных состояний');ax.grid(alpha=.2);ax.legend();fig.savefig(dest/'08_perturbations.png',dpi=160);plt.close(fig)
    if (out/'cem_inverse.csv').exists():
        a=pd.read_csv(out/'cem_inverse.csv').sort_values('rmse_ohm').groupby('model',sort=False).first().reindex(models[2:])
        fig,axes=plt.subplots(1,2,figsize=(14,5.8),layout='constrained')
        for k in [0,1]:
            axes[k].bar([LABELS[n].split('. ',1)[1] for n in models[2:]],a[f'error{k+1}_pct'],color=[COLORS[n] for n in models[2:]]);axes[k].set(ylabel=f'Ошибка ρ{k+1}, %',title=f'Конечные электроды: восстановление ρ{k+1}');axes[k].axhline(0,color='gray',lw=.7);axes[k].grid(axis='y',alpha=.2)
            plt.setp(axes[k].get_xticklabels(),rotation=12,ha='right')
        fig.savefig(dest/'09_cem_inverse.png',dpi=160);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);p.add_argument('--geometry-only',action='store_true');a=p.parse_args();geometry_figures(a.out)
    if not a.geometry_only:result_figures(a.out)
