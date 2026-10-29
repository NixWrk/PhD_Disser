"""Re-render figures 13/15 from frozen public CSV, using original plotting AST.

Only titles/labels and the existing placement layout are applied. No FEM,
inverse problem, detection, fitting or source-file write is performed.
Run from any directory with Python, numpy, matplotlib and Pillow installed.
"""
from pathlib import Path
import ast
import csv
import hashlib
import io
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')
REV=Path(__file__).resolve().parents[1]
ROOT=REV.parents[3]
BASE=REV.parent/'revision_20261029_author_review'
RESULTS=REV.parent/'results/montage_sensitivity_v1'
SOURCE=ROOT/'MATLAB_TRKG4_real_subjects/tools/heart_montage_figures.py'
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
rel=lambda p: p.relative_to(ROOT).as_posix()
tracked={}

def track(p):
    tracked[p]=sha(p)
    return p

def table(name):
    p=track(RESULTS/'tables'/name)
    expected=json.loads((RESULTS/'manifest.json').read_text(encoding='utf-8'))['files']['tables/'+name]['sha256']
    assert sha(p)==expected, name
    with p.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))

def plotting_ast(which,ns):
    """Select only original plot 1 or 3; never import/execute main()."""
    tree=ast.parse(track(SOURCE).read_text(encoding='utf-8'))
    constants={'SHAPES','SCENARIOS','MONTAGES','COLORS','LABELS','TITLES'}
    nodes=[n for n in tree.body if
           isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in n.targets)
           or isinstance(n,ast.FunctionDef) and n.name in {'choose','one'}]
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    starts=[i for i,n in enumerate(main.body) if isinstance(n,ast.Assign)
            and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='plt.subplots']
    begin=starts[which]
    end=next(i for i in range(begin,len(main.body)) if isinstance(main.body[i],ast.Assign)
             and any(isinstance(t,ast.Name) and t.id=='p' for t in main.body[i].targets))
    selected=nodes+main.body[begin:end]
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),str(SOURCE),'exec'),ns)
    return ns['fig'],ns['axes'],[main.body[begin].lineno,main.body[end-1].end_lineno]

def science(axes):
    def arr(x):
        a=np.asarray(x)
        return [list(a.shape),str(a.dtype),hashlib.sha256(a.tobytes()).hexdigest()]
    return [dict(xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim()),
        lines=[dict(x=arr(l.get_xdata()),y=arr(l.get_ydata()),color=l.get_color(),
                    marker=l.get_marker(),lw=l.get_linewidth(),ls=l.get_linestyle(),
                    ms=l.get_markersize(),mfc=l.get_markerfacecolor(),mec=l.get_markeredgecolor()) for l in ax.lines],
        patches=[dict(vertices=arr(p.get_path().vertices),transform=arr(p.get_patch_transform().get_matrix()),
                      face=list(p.get_facecolor()),edge=list(p.get_edgecolor()),lw=p.get_linewidth()) for p in ax.patches])
        for ax in axes.flat]

def txt(fig,x,y,s,px,ha='center',rotation=0):
    w,h=fig.get_size_inches()*fig.dpi
    fig.text(x/w,1-y/h,s,ha=ha,va='center',rotation=rotation,
             fontproperties=FontProperties(fname='C:/Windows/Fonts/arial.ttf',size=px*72/fig.dpi),color='black')

def legend(fig,x,y,s,c,px):
    w,h=fig.get_size_inches()*fig.dpi
    line=Line2D([x/w,(x+32.5)/w,(x+65)/w],[1-y/h]*3,color=c,lw=5*72/fig.dpi,
                marker='o',markevery=[1],markersize=16*72/fig.dpi,transform=fig.transFigure)
    fig.add_artist(line)
    txt(fig,x+83,y,s,px,ha='left')

def place(fig,axes,w,h,xs,ys,pw,ph):
    fig.set_dpi(180);fig.canvas.draw()
    original_h=fig.bbox.height
    bounds=[ax.get_window_extent().bounds for ax in axes.flat]
    source_x=[171,1227] if w==2160 else [155,1303]
    source_y=[203,703,1204] if w==2160 else [290,773,1257]
    fig.set_layout_engine('none');fig.set_size_inches(w/180,h/180);fig.set_dpi(180)
    for leg in list(fig.legends):leg.remove()
    if fig._suptitle:fig._suptitle.remove();fig._suptitle=None
    for row in range(3):
        for col in range(2):
            ax=axes[row,col]
            ox,oy,ow,oh=bounds[row*2+col]
            top=original_h-oy-oh-source_y[row]+ys[row]
            left=ox-source_x[col]+xs[col]
            ax.set_position([left/w,1-(top+oh)/h,ow/w,oh/h])
            ax.set_title('');ax.set_xlabel('');ax.set_ylabel('')
            ax.tick_params(labelbottom=False,labelleft=False)

def main():
    track(RESULTS/'manifest.json')
    derivatives=table('derivatives.csv');refinement_rows=table('volume_refinement.csv')
    comp=table('finite_compensation_initial.csv');newton_rows=table('finite_compensation_newton.csv')
    states=table('electrical_states.csv')
    bases={float(r['nominal_volume_ml']) for r in states if r['probe_kind']=='base' and r['shape']=='STL'}
    assert len(bases)==1
    vref=bases.pop()
    signed_volume_probes=sorted({float(r['probe_signed_step']) for r in states
                                if r['probe_kind']=='volume' and r['quadrature']=='64'})
    assert signed_volume_probes==[-5.,-2.5,-1.,1.,2.5,5.]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2,'savefig.dpi':180})
    records=[]
    for which in [0,2]:
        ns=dict(np=np,plt=plt,Line2D=Line2D,derivatives=derivatives,
                refinement_rows=refinement_rows,comp=comp,newton_rows=newton_rows,vref=vref)
        fig,axes,lines=plotting_ast(which,ns)
        fig.canvas.draw();before=science(axes)
        cache=track(REV.parent/'revision_20261028/figures/reader_montage'/
                    ('ch07_04_volume_step_quadrature.png' if which==0 else 'ch09_06_finite_compensation.png'))
        buf=io.BytesIO();fig.savefig(buf,dpi=180);buf.seek(0)
        baseline=np.asarray(Image.open(buf).convert('RGB'))
        frozen=np.asarray(Image.open(cache).convert('RGB'))
        fig.set_dpi(180);fig.canvas.draw()
        panel_checks=[]
        for i,ax in enumerate(axes.flat):
            x,y,aw,ah=ax.get_window_extent().bounds
            x0,x1=int(x)+2,int(x+aw)-2
            y0,y1=int(baseline.shape[0]-y-ah)+2,int(baseline.shape[0]-y)-2
            assert np.array_equal(baseline[y0:y1,x0:x1],frozen[y0:y1,x0:x1])
            dx=([171,1227][i%2]-[171,1227][i%2]) if which==0 else ([225,1430][i%2]-[155,1303][i%2])
            dy=([120,530,940][i//2]-[203,703,1204][i//2]) if which==0 else ([200,689,1178][i//2]-[290,773,1257][i//2])
            panel_checks.append(dict(source_box=[x0,y0,x1,y1],target_box=[x0+dx,y0+dy,x1+dx,y1+dy],baseline_pixels_equal=True))
        if which==0:
            w,h=2160,1500;xs=[171,1227];ys=[120,530,940];ph=[365]*3;pw=907
            place(fig,axes,w,h,xs,ys,pw,ph)
            txt(fig,1080,29,'Шаг объёмной производной и квадратура',45)
            yp=[[[269,340,411,482,554],[249,313,376,440,504]],[[743,822,900,979,1058],[767,842,917,991]],[[1210,1274,1339,1404,1469,1534],[1257,1314,1372,1429,1486,1544]]]
            yt=[[['0.195','0.190','0.185','0.180','0.175'],['1.34','1.33','1.32','1.31','1.30']],[['−0.136','−0.138','−0.140','−0.142','−0.144'],['0.795','0.790','0.785','0.780']],[['0.032','0.030','0.028','0.026','0.024','0.022'],['0.930','0.925','0.920','0.915','0.910','0.905']]]
            oldys=[203,703,1204]
            for row,name in enumerate(['STL','G1','L1']):
                for col in range(2):
                    txt(fig,xs[col]+453,ys[row]-30,f'{name} · '+r'$\rho_H = 4$'+' Ом·м' if col==0 else f'{name} · опорные ρ из подгонки STL',38)
                    for pos,label in zip(yp[row][col],yt[row][col]):txt(fig,xs[col]-14,ys[row]+pos-oldys[row],label,36,ha='right')
                    if row==2:
                        for px,label in zip([42,351,865],['1.0','2.5','5.0']):txt(fig,xs[col]+px,ys[row]+388,label,36)
            txt(fig,20,720,'∂Z/∂V, мОм/мл',39,rotation=90)
            txt(fig,1080,1369,r'Объёмная проба $|\Delta V| / V_0$, %',41)
            for x,y,label,c in [(330,1420,'Исходный, 100 мм','#12658c'),(330,1462,'95 мм','#e07a18'),(735,1462,'Поворот на одной руке','#7852a5')]:legend(fig,x,y,label,c,37)
            for x,y,marker,label in [(1257,1420,'D','256 точек; линии — 64'),(1257,1462,'*','1024 точки; только L1')]:
                fig.add_artist(Line2D([x/w],[1-y/h],marker=marker,color='black',markerfacecolor='none' if marker=='D' else 'black',ms=14*72/180,ls='',transform=fig.transFigure))
                txt(fig,x+58,y,label,37,ha='left')
            name='V13_models_step_quadrature_compartment_labels.png';old='V13_models_step_quadrature.png'
        else:
            w,h=2460,1710;xs=[225,1430];ys=[200,689,1178];pw=1011;ph=[376,376,375]
            place(fig,axes,w,h,xs,ys,pw,ph)
            txt(fig,1230,35,r'Конечные FEM-отклики: компенсация объёма изменением $\rho_H$',52)
            txt(fig,1230,96,'Контур — объём; заполнение — линейная компенсация; × — шаг Ньютона',49)
            yp=[[[315,389,463,538,612],[312,367,421,476,531,586,640]],[[794,848,901,954,1008,1061,1115],[812,886,959,1033,1106]],[[1285,1351,1416,1482,1547,1613],[1316,1380,1443,1507,1570]]]
            yt=[[['1.0','0.5','0.0','−0.5','−1.0'],['7.5','5.0','2.5','0.0','−2.5','−5.0','−7.5']],[['0.75','0.50','0.25','0.00','−0.25','−0.50','−0.75'],['4','2','0','−2','−4']],[['0.10','0.05','0.00','−0.05','−0.10','−0.15'],['4','2','0','−2','−4']]]
            oldys=[290,773,1257]
            for row,shape in enumerate(['STL','G1','L1']):
                for col in range(2):
                    txt(fig,xs[col]+505,ys[row]-32,f'{shape} · '+r'$\rho_H = 4$'+' Ом·м' if col==0 else f'{shape} · опорные ρ из подгонки STL',50)
                    for pos,label in zip(yp[row][col],yt[row][col]):txt(fig,xs[col]-15,ys[row]+pos-oldys[row],label,49,ha='right')
                    for off,label in zip([226,785],['V − 1%','V + 1%']):txt(fig,xs[col]+off,ys[row]+ph[row]+30,label,49)
            txt(fig,34,868,'Конечное изменение Z, мОм',49,rotation=90)
            for x,label,c in [(340,'Исходный, 100 мм','#12658c'),(1020,'95 мм','#e07a18'),(1450,'Поворот на одной руке','#7852a5')]:legend(fig,x,1672,label,c,49)
            name='V19_volume_rho_compensation_compartment_labels.png';old='V19_volume_rho_compensation.png'
        fig.canvas.draw();after=science(axes)
        assert before==after,'Scientific artists/limits changed'
        dest=REV/'figures'/name
        fig.savefig(dest,dpi=180,facecolor='white')
        plt.close(fig)
        new=np.asarray(Image.open(dest).convert('RGB'))
        for check in panel_checks:
            x0,y0,x1,y1=check['source_box'];tx0,ty0,tx1,ty1=check['target_box']
            assert np.array_equal(frozen[y0:y1,x0:x1],new[ty0:ty1,tx0:tx1]),'Re-rendered data pixels changed'
            check['new_plot_interior_pixels_equal_frozen']=True
        assert Image.open(dest).size==Image.open(BASE/'figures'/old).size
        records.append(dict(old_asset=rel(BASE/'figures'/old),new_path=rel(dest),sha256=sha(dest),
            dimensions_px=[w,h],source_code=dict(path=rel(SOURCE),sha256=sha(SOURCE),lines=lines),
            source_values_unchanged=True,scientific_artists_and_limits_equal_before_after=True,
            scientific_scene_sha256=hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),
            vref_ml=vref,matplotlib=matplotlib.__version__,raster_panels_copied=False))
        records[-1]['panel_pixel_checks']=panel_checks
        if which==0:
            records[-1]['volume_probe_axis']={
                'label':'Объёмная проба |ΔV| / V₀, %',
                'nominal_signed_percent_in_CSV':signed_volume_probes,
                'unchanged_display_x_formula':'100 * (minus_step + plus_step) / (2 * V0)',
                'definition':'|ΔV| denotes the mean magnitude of the paired actual volume probes; nominal probes are ±1/±2.5/±5%. Actual x values are not rounded or snapped to nominal ticks.'}
    assert all(sha(p)==digest for p,digest in tracked.items()),'Input changed'
    result=dict(figures=records,inputs=[dict(path=rel(p),sha256=digest) for p,digest in tracked.items()],
                all_input_hashes_unchanged=True,public_result_hashes_verified=True)
    (REV/'tools/material_label_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(records,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
