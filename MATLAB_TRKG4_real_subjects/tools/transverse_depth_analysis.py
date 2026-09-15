"""Depth is the shortest Euclidean distance to the closed external skin STL."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import vtk
from vtk.util.numpy_support import vtk_to_numpy,numpy_to_vtk
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transverse_geometry_study import OUT,ROOT,read_surface,write

def run(out=OUT):
    (out/'figures').mkdir(exist_ok=True)
    g=loadmat(out/'geometry_input.mat',simplify_cells=True)
    cache=out/'centroid_skin_depth.npy'
    if cache.exists():depth=np.load(cache)
    else:
        poly=read_surface(ROOT/'data/nik/stl/body_solid_v3_1mm_r10_volume_fill.stl')
        fn=vtk.vtkImplicitPolyDataDistance();fn.SetInput(poly)
        pts=numpy_to_vtk(np.ascontiguousarray(g['centres_mm']),deep=True)
        vals=vtk.vtkDoubleArray();vals.SetNumberOfComponents(1);vals.SetNumberOfTuples(len(g['centres_mm']))
        fn.FunctionValue(pts,vals)
        depth=np.abs(vtk_to_numpy(vals));np.save(cache,depth)
    a=loadmat(out/'depth_sensitivity.mat',simplify_cells=True);s=a['sensitivity'];mask=a['mask'].astype(bool)
    rows=[];curves=[]
    for j,L in enumerate(a['sizes']):
        v=s[:,j];av=np.abs(v);all_abs=av.sum();lung_abs=av[mask].sum();cut=depth>L/3
        rows.append(dict(L_mm=L,threshold_mm=L/3,absolute_beyond_pct=100*av[cut].sum()/all_abs,lung_absolute_beyond_pct=100*av[cut&mask].sum()/lung_abs,lung_signed_log_sensitivity_ohm=v[mask].sum(),soft_signed_log_sensitivity_ohm=v[~mask].sum(),negative_abs_fraction_pct=100*av[v<0].sum()/all_abs))
        order=np.argsort(depth);cum=np.cumsum(av[order])/all_abs
        for x in np.arange(0,101,2):curves.append(dict(L_mm=L,depth_mm=x,cumulative_absolute_pct=100*np.interp(x,depth[order],cum,left=0,right=1)))
    pd.DataFrame(rows).to_csv(out/'depth_sensitivity_summary.csv',index=False)
    c=pd.DataFrame(curves);c.to_csv(out/'depth_sensitivity_curves.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    for L in [50,90,140]:
        q=c[c.L_mm==L];line=axes[0].plot(q.depth_mm,q.cumulative_absolute_pct,label=f'L = {L:g} мм')[0]
        axes[0].axvline(L/3,color=line.get_color(),ls=':',alpha=.6)
    axes[0].set(xlabel='Расстояние до ближайшей поверхности кожи, мм',ylabel='Накопленная абсолютная чувствительность, %',title='Чувствительность распределена по глубине');axes[0].legend();axes[0].grid(alpha=.2)
    r=pd.DataFrame(rows);axes[1].plot(r.L_mm,r.absolute_beyond_pct,'o-',label='Все ткани');axes[1].plot(r.L_mm,r.lung_absolute_beyond_pct,'s-',label='Только лёгкое')
    axes[1].set(xlabel='Размер сборки L, мм',ylabel='Доля глубже L/3, %',title='Чувствительность за условной границей L/3');axes[1].legend();axes[1].grid(alpha=.2)
    fig.savefig(out/'figures/10_depth_sensitivity.png',dpi=160);plt.close(fig)
    write(out/'depth_analysis.json',{'status':'completed','depth_definition':'Absolute signed distance from each element centroid to outer body STL; Euclidean nearest skin, not local d.','weight':'Absolute elementwise dZ/dln(rho_region), including element volume. Normalized separately for all tissue and lung.','cutoff_is_hard_boundary':False,'reference_rho':[4,16],'electrodes':'point'})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT)
    run(parser.parse_args().out)
