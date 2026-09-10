"""Prepare the scientific reader section of notebook 20.10 from verified results.

Only presentation and deterministic summaries are produced. No notebook cells,
FEM solves, pose searches or resistivity optimizations are executed.
"""
from pathlib import Path
import base64,hashlib,json,re
import numpy as np
import pandas as pd
import nbformat
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'output/exploratory/scan_localization_20260910'
OUT=DATA/'scientific_notebook'
SOURCE=ROOT/'notebooks/TRKG4_saved_scan_scientific.md'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def fmt(x,n=3):return f'{float(x):.{n}f}'.replace('.',',') if np.isfinite(float(x)) else '—'
def table(headers,values):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in values)

def main():
    OUT.mkdir(exist_ok=True)
    input_contract=read(DATA/'input_contract.json');assert sha(DATA/'scan_arrays.npz')==input_contract['array_sha256']
    fit_summary=read(DATA/'fit_summary.json');assert sha(DATA/'layered_fits.npz')==fit_summary['fit_sha256']
    review=DATA/'best_pose_review';cert=read(review/'minimum_certificate.json');prior=read(review/'manifest.json')
    for relative,digest in prior['inputs'].items():assert sha(ROOT/relative)==digest,relative
    d=np.load(DATA/'scan_arrays.npz');f=np.load(DATA/'layered_fits.npz');maps=np.load(DATA/'analysis_maps.npz');ng=np.load(DATA/'neighbourhood_geometry.npz');ct=np.load(DATA/'ct_comparison.npz')
    row=int(np.flatnonzero(d['flat']==cert['best']['flat_index'])[0]);sizes=d['sizes'];observed=d['experiment'];z=d['Z_fem'][row];pred=f['prediction'][row]
    np.testing.assert_allclose(np.sqrt(np.mean((d['Z_fem']-observed)**2,axis=1)),d['rmse_experiment'],rtol=0,atol=1e-11)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,'legend.fontsize':9,'savefig.facecolor':'white'})
    assets={}
    def save(fig,name):
        fig.savefig(OUT/name,dpi=170,bbox_inches='tight');plt.close(fig);assets[name]=sha(OUT/name)
    cmap=LinearSegmentedColormap.from_list('error',['#087443','#62bd60','#e0dc60','#ed8950','#b52136']);cmap.set_bad('#e3e6e9')
    fig,axes=plt.subplots(2,2,figsize=(10.5,9),constrained_layout=True)
    choices=[(maps['experiment'],'а) Согласие FEM с экспериментом'),(ng['robust_experiment_distinct'],'б) Худшая ошибка в окне 125 поз\nс 25 различными центрами'),(maps['two_layer'],'в) Двуслойное приближение FEM'),(maps['joint'],'г) Совместный критерий в одной позе')]
    for ax,(values,title) in zip(axes.ravel(),choices):
        best=np.min(np.where(np.isfinite(values),values,np.inf),axis=1);best[~np.isfinite(best)]=np.nan
        im=ax.imshow(best.reshape(len(d['v']),len(d['u'])),origin='lower',extent=[-80.5,80.5,-120.5,120.5],cmap=cmap,vmin=0,vmax=15,aspect='equal',interpolation='nearest')
        ax.set(title=title,xlabel='u, мм',ylabel='v, мм',xlim=(-80.5,80.5),ylim=(-120.5,120.5))
    fig.colorbar(im,ax=list(axes.ravel()),label='RMSE или максимум RMSE, Ом; насыщение цвета при 15 Ом',shrink=.7)
    save(fig,'fig20_1_maps.png')
    # Actual CEM faces, as checked independently against MATLAB contact areas.
    text=(review/'index.html').read_text(encoding='utf-8');payload=json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>',text,re.S)[1])
    def geometry(key):return np.frombuffer(base64.b64decode(payload[key]['nodes']),dtype='<f4').reshape(-1,3),np.frombuffer(base64.b64decode(payload[key]['faces']),dtype='<u4').reshape(-1,3)
    colours=['#d73545','#e99b21','#1286c6','#7b4cca'];montage=payload['patches'][-1]
    fig=plt.figure(figsize=(12,7))
    for k,key in enumerate(['body','local']):
        ax=fig.add_subplot(1,2,k+1,projection='3d');nodes,faces=geometry(key)
        ax.add_collection3d(Poly3DCollection(nodes[faces],facecolor='#b9c4ce',edgecolor='none',alpha=.12 if k==0 else .32,rasterized=True))
        for i,patch in enumerate(montage):
            points=np.array(patch['nodes']);tri=np.array(patch['faces']);c=np.array(patch['centroid']);normal=np.array(patch['normal'])
            ax.add_collection3d(Poly3DCollection(points[tri],facecolor=colours[i],edgecolor='none',alpha=1))
            ax.scatter(*c,color=colours[i],marker='D',s=15);label=c+normal*7
            ax.text(*label,patch['role'],color=colours[i],fontsize=12,fontweight='bold')
        lo=nodes.min(axis=0);hi=nodes.max(axis=0)
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),xlabel='x, мм',ylabel='y, мм',zlabel='z, мм',title='а) Положение на теле' if k==0 else 'б) Контактные площадки, L = 140 мм')
        ax.set_box_aspect(hi-lo);ax.set_proj_type('ortho');ax.view_init(elev=12,azim=160);ax.tick_params(labelsize=8,pad=1)
        if k==1:
            ax.set_xticks([-195,-175]);ax.set_yticks([10,40,70]);ax.set_zticks([-75,-125,-175,-225])
            ax.set_xlabel('x, мм',labelpad=10);ax.set_ylabel('y, мм',labelpad=8)
        else:
            ax.set_xticks([-200,0,200]);ax.set_yticks([-100,0,100]);ax.set_zticks([-250,-150,-50])
    fig.subplots_adjust(left=0,right=1,bottom=.12,top=.92,wspace=.06);save(fig,'fig20_2_electrodes.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    axes[0].plot(sizes,observed,'ko:',label='Эксперимент');axes[0].plot(sizes,z,'o-',color='#167abd',label='FEM');axes[0].plot(sizes,pred,'s-',color='#208649',label='Два слоя, общее h')
    axes[0].set(title='а) Размерная зависимость',xlabel='L, мм',ylabel='Импеданс, Ом');axes[0].legend()
    axes[1].bar(sizes,z-observed,width=6,color='#3386b7');axes[1].axhline(0,color='#333',lw=.7);axes[1].set(title='б) Остатки FEM − эксперимент',xlabel='L, мм',ylabel='Разность, Ом')
    for ax in axes:ax.grid(alpha=.18)
    save(fig,'fig20_3_curve_residuals.png')
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    local=np.array([[np.nan if x is None else x for x in line] for line in payload['errors'][payload['best_angle_index']]])
    im=axes[0,0].imshow(local,origin='lower',extent=[8.5,17.5,-59.5,-50.5],cmap=cmap,vmin=2.8,vmax=20,interpolation='nearest')
    axes[0,0].plot(13,-55,'D',mfc='none',mec='#bf1631',mew=2,ms=10);axes[0,0].set(title='а) Центры при φ = 14°',xlabel='u, мм',ylabel='v, мм');fig.colorbar(im,ax=axes[0,0],label='RMSE, Ом')
    for ax,key,label,value,letter in zip([axes[0,1],axes[1,0],axes[1,1]],['u','v','phi'],['u, мм','v, мм','φ, °'],[13,-55,14],['б','в','г']):
        s=payload['slices'][key];ax.plot([p['value'] for p in s],[p['rmse_ohm'] for p in s],'o-',color='#2877a6',ms=4)
        ax.plot(value,cert['best']['rmse_ohm'],'D',color='#bf1631',ms=7);ax.set(title=f'{letter}) Срез по {label}',xlabel=label,ylabel='RMSE, Ом');ax.set_ylim(bottom=0);ax.grid(alpha=.2)
    save(fig,'fig20_4_local_minimum.png')
    loo=pd.read_csv(review/'localization_leave_one_size_out.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    axes[0].bar(loo.omitted_L_mm,loo.centre_distance_to_best_mm,width=6,color='#3386b7');axes[0].set(title='а) Изменение выбранного центра',xlabel='Исключённый L, мм',ylabel='Смещение в координатах КТ, мм')
    axes[1].bar(loo.omitted_L_mm,loo.held_out_error_ohm,width=6,color='#a35b4f');axes[1].axhline(0,color='#333',lw=.7);axes[1].set(title='б) Прогноз исключённого размера',xlabel='Исключённый L, мм',ylabel='FEM − эксперимент, Ом')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    save(fig,'fig20_5_localization_holdouts.png')
    ci=d['flat']//len(d['phi']);depth=ct['normal_depth_by_centre_mm'][ci];ok=np.isfinite(depth)&np.isfinite(f['h_mm'])
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    hb=axes[0].hexbin(depth[ok],f['h_mm'][ok],gridsize=50,mincnt=1,bins='log',cmap='viridis');fig.colorbar(hb,ax=axes[0],label='Число параметрических поз')
    limit=max(np.percentile(depth[ok],99.9),np.max(f['h_mm'][ok]));axes[0].plot([0,limit],[0,limit],'k--',lw=1,label='Равенство величин')
    axes[0].set(xlabel='Глубина первого входа в лёгкое, мм',ylabel='Эффективное h, мм',title='а) Различие определений толщины');axes[0].legend()
    bins=np.linspace(0,50,51);axes[1].hist(f['rmse_ohm'][ok],bins=bins,histtype='step',density=False,lw=1.8,label='h подобрано');axes[1].hist(ct['rmse_with_ct_normal_depth_ohm'][ok],bins=bins,histtype='step',lw=1.8,label='h = глубина по КТ')
    axes[1].set(xlabel='RMSE к FEM, Ом',ylabel='Число параметрических поз',title='б) Подбор и подстановка глубины');axes[1].legend();axes[1].grid(alpha=.2)
    save(fig,'fig20_6_thickness.png')
    electrodes=pd.read_csv(review/'best_electrodes_all_sizes.csv');e=electrodes.loc[electrodes.L_mm==140]
    direct=pd.read_csv(review/'six_neighbours.csv');choices=pd.read_csv(DATA/'candidate_requests.csv').set_index('candidate_id')
    substitutions={
        'INPUT_TABLE':table(['L, мм','Эксперимент, Ом','FEM в выбранной позе, Ом','FEM − эксперимент, Ом'],[[int(L),fmt(y),fmt(v),fmt(v-y)] for L,y,v in zip(sizes,observed,z)]),
        'ELECTRODE_TABLE':table(['Электрод','x центра площади, мм','y, мм','z, мм','Площадь, мм²'],[[r.electrode,fmt(r.patch_centroid_x_mm),fmt(r.patch_centroid_y_mm),fmt(r.patch_centroid_z_mm),fmt(r.contact_area_mm2,2)] for r in e.itertuples()]),
        'DIRECT_TABLE':table(['Изменение параметра','RMSE, Ом'],[[next(name+(' + 1' if getattr(r,key)>0 else ' − 1') for name,key in [('u','du'),('v','dv'),('φ','dphi')] if getattr(r,key)),fmt(r.rmse_ohm,6) if r.status==1 else 'Отказ геометрического контроля'] for r in direct.itertuples()]),
        'LOO_TABLE':table(['Исключённый L, мм','u; v, мм','φ, °','Смещение центра, мм','Ошибка исключённого размера, Ом'],[[int(r.omitted_L_mm),f'{int(r.u_mm)}; {int(r.v_mm)}',int(r.phi_deg),fmt(r.centre_distance_to_best_mm,2),fmt(r.held_out_error_ohm,2)] for r in loo.itertuples()]),
        'CHOICE_TABLE':table(['Критерий выбора','u; v, мм','φ, °','FEM ↔ эксперимент, Ом','Два слоя ↔ FEM, Ом','h, мм'],[[label,f'{int(choices.loc[key].u_mm)}; {int(choices.loc[key].v_mm)}',int(choices.loc[key].phi_deg),fmt(choices.loc[key].rmse_experiment_ohm),fmt(choices.loc[key].rmse_2L_FEM_ohm),fmt(choices.loc[key].h_eff_mm)] for key,label in [('C01','Минимум ошибки к эксперименту'),('C03','Минимум ошибки двуслойного приближения'),('C04','Минимум большей из двух ошибок')]]),
        'HISTOGRAM_OMITTED':('Все значения обеих выборок входят в показанный диапазон.' if not np.any(f['rmse_ohm'][ok]>50) and not np.any(ct['rmse_with_ct_normal_depth_ohm'][ok]>50) else 'За правой границей остаются '+str(int(np.sum(f['rmse_ohm'][ok]>50)))+' и '+str(int(np.sum(ct['rmse_with_ct_normal_depth_ohm'][ok]>50)))+' поз соответственно для подбора толщины и подстановки глубины.')
    }
    text=SOURCE.read_text(encoding='utf-8')
    for key,value in substitutions.items():
        marker='{{'+key+'}}';assert text.count(marker)==1,marker;text=text.replace(marker,value)
    assert not re.search(r'\{\{[A-Z_]+\}\}',text)
    fragments=re.split(r'^<!-- cell: ([a-z0-9-]+) -->\s*$',text,flags=re.M);cells=[]
    for k in range(1,len(fragments),2):
        identifier,body=fragments[k:k+2];attachments={}
        for name in re.findall(r'attachment:([^\s)]+)',body):attachments[name]={'image/png':base64.b64encode((OUT/name).read_bytes()).decode()}
        cell=nbformat.v4.new_markdown_cell(body.strip(),id='saved-scan-'+identifier,metadata={'scientific_section':'saved_scan','result_status':'exploratory_hypothesis_not_validated','source_manifest_sha256':sha(review/'manifest.json')})
        if attachments:cell['attachments']=attachments
        cells.append(dict(cell))
    (OUT/'chapter_cells.json').write_text(json.dumps(cells,ensure_ascii=False),encoding='utf-8')
    (OUT/'chapter_for_review.md').write_text('\n\n'.join(c['source'] for c in cells),encoding='utf-8')
    package=dict(title='Результаты локализации по многоразмерному FEM-перебору',source_sha256=sha(SOURCE),producer_sha256=sha(__file__),
        chapter_cells_sha256=sha(OUT/'chapter_cells.json'),figures=assets,inputs={str(p.relative_to(ROOT)):sha(p) for p in [DATA/'scan_arrays.npz',DATA/'layered_fits.npz',DATA/'analysis_maps.npz',DATA/'ct_comparison.npz',DATA/'neighbourhood_geometry.npz',review/'minimum_certificate.json',review/'manifest.json']},
        scientific_status='model_result_not_physically_validated',new_FEM_solutions=0,notebook_cells_executed=False)
    (OUT/'chapter_manifest.json').write_text(json.dumps(package,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Prepared',len(cells),'scientific Markdown cells and',len(assets),'figures:',OUT,flush=True)

if __name__=='__main__':main()
