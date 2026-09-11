"""Visualize committed CEM patches and certify only the sampled, fixed-rho minimum.

No FEM solves, continuous-pose refinement, rho fitting, or statistical noise
assumptions are introduced. All diagnostics use the stopped scan snapshot.
"""
from pathlib import Path
import argparse,base64,hashlib,json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from plotly.offline import get_plotlyjs

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'output/exploratory/scan_localization_20260910'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def rows(t):return json.loads(t.to_json(orient='records',force_ascii=False))
def encoded(x,t='<f4'):return base64.b64encode(np.asarray(x,dtype=t).tobytes()).decode()
def clean(x):return x if np.isfinite(x) else None

def ranking_radius(predictions,observed,best):
    """Exact minimum RMS data perturbation reaching a pairwise SSE bisector.

    ||zq-y-d||²-||zp-y-d||² = gap - 2 d·(zq-zp).
    Cauchy-Schwarz gives ||d|| >= gap/(2||zq-zp||), attained
    by a perturbation parallel to zq-zp. Divide by sqrt(n) for RMS.
    Identical prediction vectors cannot be distinguished by these data.
    """
    sse=np.sum((predictions-observed)**2,axis=1);delta=predictions-predictions[best]
    norm=np.linalg.norm(delta,axis=1);radius=np.full(len(predictions),np.inf);ok=norm>1e-12
    radius[ok]=(sse[ok]-sse[best])/(2*norm[ok]*np.sqrt(len(observed)))
    rival=int(np.argmin(radius));perturbation=(sse[rival]-sse[best])/(2*norm[rival]**2)*delta[rival]
    return rival,float(radius[rival]),perturbation

def build(directory):
    out=directory/'best_pose_review';out.mkdir(exist_ok=True)
    contract=read(directory/'input_contract.json');run=Path(contract['source_run'])
    assert sha(directory/'scan_arrays.npz')==contract['array_sha256']
    assert not read(run/'live/summary.json')['streaming_active']
    manifest=read(run/'run_manifest.json');d=np.load(directory/'scan_arrays.npz');f=np.load(directory/'layered_fits.npz')
    z=d['Z_fem'];y=d['experiment'];errors=np.sqrt(np.mean((z-y)**2,axis=1))
    np.testing.assert_allclose(errors,d['rmse_experiment'],atol=1e-11,rtol=0)
    order=np.argsort(errors,kind='stable');best=int(order[0]);flat=int(d['flat'][best]);A=len(d['phi']);U=len(d['u']);V=len(d['v']);ci,ai=divmod(flat,A);v,u=divmod(ci,U)
    lookup=np.full(d['status'].size,-1,int);lookup[d['flat']]=np.arange(len(z));centres=d['centres']
    def pose(row):
        fid=int(d['flat'][row]);c,a=divmod(fid,A)
        return dict(flat_index=fid,centre_index=c+1,u_mm=float(centres[c,1]),v_mm=float(centres[c,2]),phi_deg=float(d['phi'][a]),
            x_mm=float(centres[c,3]),y_mm=float(centres[c,4]),z_mm=float(centres[c,5]),rmse_ohm=float(errors[row]),
            centre_distance_to_best_mm=float(np.linalg.norm(centres[c,3:6]-centres[ci,3:6])))
    neighbour_tables={};counts={}
    for radius in [1,2]:
        records=[]
        for dv in range(-radius,radius+1):
            for du in range(-radius,radius+1):
                for da in range(-radius,radius+1):
                    vv,uu,aa=v+dv,u+du,ai+da
                    if not (0<=vv<V and 0<=uu<U and 0<=aa<A):continue
                    fid=(vv*U+uu)*A+aa;row=lookup[fid];state=int(d['status'].ravel()[fid]);centre=centres[vv*U+uu]
                    records.append(dict(du=du,dv=dv,dphi=da,flat_index=fid,u_mm=float(centre[1]),v_mm=float(centre[2]),phi_deg=float(d['phi'][aa]),
                        status=state,rmse_ohm=float(errors[row]) if row>=0 else np.nan,
                        x_mm=float(centre[3]),y_mm=float(centre[4]),z_mm=float(centre[5])))
        table=pd.DataFrame(records);table.to_csv(out/f'neighbours_radius_{radius}.csv',index=False);neighbour_tables[radius]=table
        allowed=table.loc[(table.status==1)&(table.flat_index!=flat)]
        counts[radius]=dict(total=len(table),evaluated=int((table.status==1).sum()),invalid=int((table.status==2).sum()),excluded=int((table.status==3).sum()),missing=int((table.status==0).sum()),
            minimum_other_valid_rmse_ohm=float(allowed.rmse_ohm.min()),maximum_valid_rmse_ohm=float(table.rmse_ohm.max()),
            strictly_smaller_than_all_other_valid=bool((allowed.rmse_ohm>errors[best]).all()))
    direct=neighbour_tables[1].loc[neighbour_tables[1][['du','dv','dphi']].abs().sum(axis=1)==1]
    direct.to_csv(out/'six_neighbours.csv',index=False)
    top=pd.DataFrame([pose(int(k)) for k in order[:50]]);top.to_csv(out/'top_50_poses.csv',index=False)
    rival,radius,perturbation=ranking_radius(z,y,best)
    yp=y+1.01*perturbation
    assert np.mean((z[rival]-yp)**2)<np.mean((z[best]-yp)**2)
    assert abs(np.linalg.norm(perturbation)/np.sqrt(len(y))-radius)<1e-12
    pd.DataFrame(dict(L_mm=d['sizes'],observed_ohm=y,constructed_delta_ohm=perturbation,
        constructed_tie_vector_ohm=y+perturbation)).to_csv(out/'ranking_perturbation_scenario.csv',index=False)
    sse=np.sum((z-y)**2,axis=1);loo=[]
    for j,L in enumerate(d['sizes']):
        training=np.sqrt(np.maximum(0,sse-(z[:,j]-y[j])**2)/8);winner=int(np.argmin(training));r=pose(winner)
        r.update(omitted_L_mm=int(L),training_8_size_rmse_ohm=float(training[winner]),held_out_error_ohm=float(z[winner,j]-y[j]),
                 same_pose_as_9_size_fit=winner==best)
        loo.append(r)
    pd.DataFrame(loo).to_csv(out/'localization_leave_one_size_out.csv',index=False)
    source_summary_path=ROOT/'output/nik_trkg4_inverse_direct_search_summary_auditfix_20260908.json'
    source_summary=read(source_summary_path);history_path=ROOT/'output/nik_trkg4_inverse_direct_search_history_auditfix_20260908.csv';history=pd.read_csv(history_path)
    inverse_source=ROOT/'src/run_trkg4_inverse_inhale.m'
    expected=next(x['sha256'] for x in source_summary['identity']['code']['files'] if x['path']=='src/run_trkg4_inverse_inhale.m')
    assert sha(inverse_source)==expected
    rho=dict(current_scan='fixed rho; only u, v and phi were enumerated',rho_ohm_m=d['rho'].tolist(),
        prior_stage='joint derivative-free search over rho_soft, rho_lung, u, v, phi',prior_evaluations=len(history),
        prior_unique_rho_pairs=int(len(history[['rho_soft_ohm_m','rho_lungs_ohm_m']].drop_duplicates())),prior_status=source_summary['status'],
        prior_observed_rho1_range=[float(history.rho_soft_ohm_m.min()),float(history.rho_soft_ohm_m.max())],
        prior_observed_rho2_range=[float(history.rho_lungs_ohm_m.min()),float(history.rho_lungs_ohm_m.max())],
        prior_objective='mean Huber residual + slope discrepancy + geometric depth penalties + lung resistivity prior; not the current plain RMSE',
        source_code_matches_recorded_hash=True,independently_measured_resistivities=False,
        current_pose_is_joint_rho_pose_optimum=False)
    # The publisher preserves boundary-face row order, only renumbers vertices.
    surface_path=run/'live/surface.mat';expected_surface=next(x['sha256'] for x in contract['inputs'] if Path(x['path'])==surface_path)
    assert sha(surface_path)==expected_surface
    surface=loadmat(surface_path);nodes=surface['nodes_mm'];faces=surface['boundary_faces'].astype(np.int64)-1
    block=run/'geometry'/f'block_{int(np.ceil((ci+1)/manifest["physics"]["centre_chunk"])):05d}.mat'
    sidecar=read(str(block)+'.json');assert sha(block)==sidecar['sha256'] and sidecar['fingerprint']==contract['source_fingerprint']
    g=loadmat(block,variable_names=['centre_ids','phi_deg','valid_rows','patch_faces'])
    rr=np.flatnonzero((g['centre_ids'].ravel()==ci+1)&(g['phi_deg'].ravel()==d['phi'][ai]));assert len(rr)==1
    vr=np.flatnonzero(g['valid_rows'].ravel()==rr[0]+1);assert len(vr)==1
    electrode_table=pd.read_csv(directory/'candidate_electrodes.csv');electrode_table=electrode_table.loc[(electrode_table.centre_index==ci+1)&(electrode_table.phi_deg==d['phi'][ai])].copy()
    electrode_table.to_csv(out/'best_electrodes_all_sizes.csv',index=False)
    patches=[];all_faces=[];roles=['I+','V+','V-','I-'];max_area_error=0.;max_centroid_error=0.
    for j,L in enumerate(d['sizes']):
        montage=[]
        for e,role in enumerate(roles):
            ids=g['patch_faces'][vr[0],j,e].astype(np.int64).ravel()-1;tri=faces[ids];xyz=nodes[tri];area=.5*np.linalg.norm(np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]),axis=1);centroid=np.sum(area[:,None]*xyz.mean(axis=1),axis=0)/area.sum()
            reference=electrode_table.loc[(electrode_table.L_mm==L)&(electrode_table.electrode==role)].iloc[0]
            max_area_error=max(max_area_error,abs(area.sum()-reference.contact_area_mm2));max_centroid_error=max(max_centroid_error,np.max(abs(centroid-reference[['patch_centroid_x_mm','patch_centroid_y_mm','patch_centroid_z_mm']].to_numpy(float))))
            vertices,ix=np.unique(tri,return_inverse=True)
            montage.append(dict(role=role,nodes=nodes[vertices].tolist(),faces=ix.reshape(-1,3).tolist(),area_mm2=float(area.sum()),centroid=centroid.tolist(),
                requested=reference[['x_mm','y_mm','z_mm']].to_numpy(float).tolist(),normal=reference[['nx','ny','nz']].to_numpy(float).tolist()))
            all_faces.extend(ids)
        patches.append(montage)
    assert max_area_error<1e-9 and max_centroid_error<1e-9
    patch_vertices=nodes[np.unique(faces[np.unique(all_faces)])];low=patch_vertices.min(axis=0)-20;high=patch_vertices.max(axis=0)+20
    middle=nodes[faces].mean(axis=1);local_faces=faces[np.all((middle>=low)&(middle<=high),axis=1)]
    local_vertices,ix=np.unique(local_faces,return_inverse=True);local_nodes=nodes[local_vertices];local_faces=ix.reshape(-1,3)
    geometry=pd.read_csv(directory/'centres_geometry.csv').iloc[ci]
    normal=geometry[['nx','ny','nz']].to_numpy(float)
    slices={}
    for key in ['u','v','phi']:
        values=d['phi'] if key=='phi' else np.arange(-8,9)+centres[ci,1 if key=='u' else 2]
        rec=[]
        for val in values:
            uu=int(val-d['u'][0]) if key=='u' else u;vv=int(val-d['v'][0]) if key=='v' else v;aa=int(val-d['phi'][0]) if key=='phi' else ai
            fid=(vv*U+uu)*A+aa;row=lookup[fid];rec.append(dict(value=float(val),status=int(d['status'].ravel()[fid]),rmse_ohm=float(errors[row]) if row>=0 else None))
        slices[key]=rec
    local_u=d['u'][u-4:u+5];local_v=d['v'][v-4:v+5];layer=[];states=[]
    for angle in range(A):
        e_layer=[];s_layer=[]
        for vv in range(v-4,v+5):
            er=[];sr=[]
            for uu in range(u-4,u+5):
                fid=(vv*U+uu)*A+angle;row=lookup[fid];er.append(float(errors[row]) if row>=0 else None);sr.append(int(d['status'].ravel()[fid]))
            e_layer.append(er);s_layer.append(sr)
        layer.append(e_layer);states.append(s_layer)
    certificate=dict(status='unique minimum of the evaluated feasible fixed-rho lattice',best=pose(best),runner_up=pose(int(order[1])),
        gap_to_runner_up_ohm=float(errors[order[1]]-errors[best]),evaluated_poses=len(z),ties_within_1e_10_ohm=int(np.sum(abs(errors-errors[best])<=1e-10)),
        neighbourhoods=counts,rho=rho,data_RMS_ranking_radius_ohm=radius,data_boundary_competitor=pose(rival),
        data_radius_scope='Exact deterministic radius to a competing prediction in the saved fixed-rho library, with the FEM values treated as exact; not measured noise, confidence, or physical uncertainty.',
        continuous_local_minimum_proven=False,global_full_grid_minimum_proven=False,measured_installation_uniquely_recovered=False,
        smaller_than_one_grid_step_checked=False,missing_full_grid_poses=int((d['status']==0).sum()),
        localization_LOO_same_pose_count=int(sum(r['same_pose_as_9_size_fit'] for r in loo)),localization_LOO_max_centre_displacement_mm=float(max(r['centre_distance_to_best_mm'] for r in loo)),
        electrode_surface_area_max_error_mm2=max_area_error,electrode_surface_centroid_max_error_mm=max_centroid_error,
        new_FEM_solutions=0,source_fingerprint=contract['source_fingerprint'])
    write(out/'minimum_certificate.json',certificate)
    payload=dict(certificate=certificate,coords=rows(electrode_table),patches=patches,body=dict(nodes=encoded(nodes),faces=encoded(faces,'<u4')),
        local=dict(nodes=encoded(local_nodes),faces=encoded(local_faces,'<u4')),normal=normal.tolist(),centre=centres[ci,3:6].tolist(),
        sizes=d['sizes'].tolist(),experiment=y.tolist(),fem=z[best].tolist(),planar=f['prediction'][best].tolist(),h_eff_mm=float(f['h_mm'][best]),
        phi=d['phi'].tolist(),best_angle_index=ai,slices=slices,local_u=local_u.tolist(),local_v=local_v.tolist(),errors=layer,states=states,
        direct=rows(direct),loo=loo,top=rows(top.head(12)))
    html=TEMPLATE.replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False))
    (out/'index.html').write_text(html,encoding='utf-8')
    inputs=[directory/'scan_arrays.npz',directory/'candidate_electrodes.csv',directory/'centres_geometry.csv',surface_path,block,source_summary_path,history_path,inverse_source]
    write(out/'manifest.json',dict(producer_sha256=sha(__file__),inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        outputs={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'},new_FEM_solutions=0))
    print(json.dumps(certificate,ensure_ascii=False),flush=True)

TEMPLATE=r'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Лучшая поза: электроды и проверка минимума</title><script>__PLOTLY__</script>
<style>body{margin:0;background:#f3f5f8;color:#172b3c;font:16px/1.5 system-ui,sans-serif}main{max-width:1550px;margin:auto;padding:24px}h1{font-size:28px;margin:0}h2{font-size:20px;margin:8px 0}.muted{color:#566777}.card{padding:18px;background:#fff;border:1px solid #d7e0e6;border-radius:10px;margin:16px 0}.views{display:grid;grid-template-columns:1fr 1.6fr;gap:15px}.half{display:grid;grid-template-columns:1fr 1fr;gap:16px}.three{display:grid;grid-template-columns:repeat(3,1fr)}.plot{height:580px;min-width:0}.small{height:360px;min-width:0}.controls{display:flex;flex-wrap:wrap;align-items:center;gap:15px}select,button{font:inherit;padding:7px;background:white;border:1px solid #b4c1cc;border-radius:5px}.badge{padding:7px 12px;background:#e6f1eb;border-radius:6px;display:inline-block;margin:8px 8px 0 0}.notice{border-left:4px solid #ad6a26;padding:12px;background:#fff5e8}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:7px;border-bottom:1px solid #e0e6ea;text-align:right}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}a{color:#156392}#error{color:#b21628;white-space:pre-wrap}code{font-size:14px}@media(max-width:950px){.views,.half,.three{grid-template-columns:1fr}.plot{height:500px}main{padding:12px}}</style></head><body><main>
<h1>Лучшая готовая поза: электроды и проверка минимума</h1><p class="muted">Эксперимент 2 · девять размеров · фиксированные ρ · исходный FEM-перебор остановлен</p>
<div class="badge">u = 13 мм; v = −55 мм; φ = 14°</div><div class="badge">RMSE = 2,881828 Ом</div><div class="badge">Центр: (−187,500; 31,051; −138,051) мм</div>
<p><a href="../localization_and_layers.html">← Общая тепловая карта</a> · <a href="best_electrodes_all_sizes.csv">Координаты всех электродов</a> · <a href="minimum_certificate.json">Численный протокол проверки</a></p>
<section class="card" id="electrode-visual"><div class="controls"><b>Размер сборки</b><select id="size"></select><span>мм</span><button id="reset">Вернуть ракурс</button><span id="busy">Загрузка…</span></div>
<div class="views"><div id="body" class="plot"></div><div id="closeup" class="plot"></div></div>
<p><b style="color:#d73545">I+ — подача тока</b> · <b style="color:#e99b21">V+ — измерение</b> · <b style="color:#1286c6">V− — измерение</b> · <b style="color:#7b4cca">I− — отвод тока</b></p>
<p class="muted">Цветные участки — фактические треугольники CEM-контактов из сохранённого расчёта. Ромбы отмечают центр площади контакта, кольца — заданный центр после проекции на поверхность. Пунктир показывает порядок электродов и не изображает путь тока. Размеры переключаются: это последовательные установки, а не 36 одновременно наклеенных электродов. Координаты относятся к системе исходной КТ/FEM.</p><div id="coordinates" class="scroll"></div></section>
<section class="card"><h2>Что доказано, а что остаётся открытым</h2><p><b>Подтверждено вычислением:</b> это единственный минимум среди всех 475 090 рассчитанных допустимых поз при фиксированных ρ. Проверены и ближайшие соседи. Следующая поза — тот же центр, φ = 13° — имеет RMSE 2,931432 Ом; разница всего 0,049604 Ом.</p>
<p id="neighbour-count"></p><div class="notice"><b>Фактическое положение электродов однозначно не восстановлено.</b> Минимум на дискретной рассчитанной сетке не доказывает минимум непрерывной задачи или истинность геометрии эксперимента. Два из шести прямых соседей не проходят геометрический QC; для них нельзя приписывать большую измеренную ошибку. Шаг менее 1 мм / 1° не исследован, ρ в этой карте не менялись, часть общей сетки не рассчитана.</div>
<div class="half"><div><h2>Соседние центры</h2><label>Угол среза <select id="angle"></select></label><div id="neighbour-map" class="small"></div></div><div><h2>Шесть прямых соседей</h2><div id="neighbours" class="scroll"></div><p class="muted">Серые ячейки: отказ геометрии. Пустые значения не заменены нулевой ошибкой. Красный ромб указывает выбранный центр. Матрица показывает u = 9…17 и v = −59…−51 мм.</p></div></div>
<div class="three"><div id="slice-u" class="small"></div><div id="slice-v" class="small"></div><div id="slice-phi" class="small"></div></div>
<p><a href="neighbours_radius_1.csv">Все 27 поз ближнего окна</a> · <a href="neighbours_radius_2.csv">Все 125 поз расширенного окна</a> · <a href="top_50_poses.csv">50 лучших готовых поз</a></p></section>
<section class="card"><h2>Проверка обоснованности локализации</h2><p><b>Чувствительность к данным.</b> В пределах готовой библиотеки существует изменение девяти экспериментальных значений с RMS всего <b>0,077442 Ом</b>, при котором позы 14° и 13° становятся равноценными. Это точно вычисленная граница устойчивости выбора по RMSE для фиксированной библиотеки; не оценка реального шума и не доверительный интервал. Удельные сопротивления и FEM-предсказания здесь считаются неизменными.</p>
<details><summary>Как получена граница</summary><p>Пусть z₀ — FEM-вектор победителя, zⱼ — конкурента, y — эксперимент. Равенство квадратов ошибок после изменения y на δ требует: 2δ·(zⱼ−z₀) = ‖zⱼ−y‖²−‖z₀−y‖². Минимальное RMS такого δ равно разности квадратов ошибок, делённой на 2√9‖zⱼ−z₀‖. Минимум проверен по всем другим готовым предсказаниям. Направление δ параллельно zⱼ−z₀ достигает этой границы. Изменение в 1,01 раза больше действительно меняет порядок двух поз.</p><a href="ranking_perturbation_scenario.csv">Численный сценарий изменения входного вектора</a></details>
<p><b>Исключение одного размера.</b> Место заново выбиралось среди всей сохранённой библиотеки по восьми размерам. Исходная поза остаётся победителем только в 3 из 9 проверок. Без размера 50 мм центр сдвигается на 33,0 мм, а ошибка прогноза исключённого размера достигает +40,34 Ом. Это ограничение внутренней устойчивости локализации, а не новый эксперимент.</p><div id="loo" class="scroll"></div><p><a href="localization_leave_one_size_out.csv">Таблица проверки локализации без одного размера</a></p>
<div class="half"><div id="curve" class="small"></div><div id="residual" class="small"></div></div></section>
<section class="card"><h2>Перебирались ли ρ?</h2><p><b>В этой карте — нет.</b> ρ₁ = 5,5521664546649 Ом·м и ρ₂ = 30,7069468090815 Ом·м фиксированы во всех позах. Перебирались u, v и φ. Общее h подбиралось затем в аналитической двуслойной модели и не изменяло FEM-расчёты.</p>
<p>Эти ρ получены в более раннем совместном поиске по пяти параметрам: ρ₁, ρ₂, u, v, φ. Он выполнил 100 оценок и завершился по лимиту вычислений (<code>budget_reached</code>), а не с доказательством сходимости. Его критерий включал Huber-ошибку, наклон размерной кривой, геометрические штрафы по глубине и предварительное предпочтение ρ лёгкого. Текущая карта ранжирует обычную RMSE. Следовательно, найденная здесь поза не объявляется совместным оптимумом по положению и сопротивлениям.</p>
<p><b>Для более сильного обоснования</b> нужны совместная проверка ρ и позы, проверяемая геометрия контактов при малых физических смещениях и независимые сведения об установке. На нынешних данных корректный результат — условный численный кандидат.</p></section><div id="error"></div></main>
<script id="data" type="application/json">__DATA__</script><script>
'use strict';const D=JSON.parse(document.getElementById('data').textContent),q=id=>document.getElementById(id),colors=['#d73545','#e99b21','#1286c6','#7b4cca'],roles=['I+','V+','V−','I−'];
function dec(s,T=Float32Array){return new T(Uint8Array.from(atob(s),c=>c.charCodeAt(0)).buffer)}const axis=(a,k)=>Array.from({length:a.length/3},(_,i)=>a[3*i+k]);
function mesh(data,name,opacity){const n=dec(data.nodes),f=dec(data.faces,Uint32Array);return {type:'mesh3d',x:axis(n,0),y:axis(n,1),z:axis(n,2),i:axis(f,0),j:axis(f,1),k:axis(f,2),color:'#bfc8cf',opacity,flatshading:false,hoverinfo:'skip',showlegend:false,name};}
const body=mesh(D.body,'Тело',.24),local=mesh(D.local,'Поверхность',.58);let selected=8;const num=(x,n=3)=>x===null?'—':x.toFixed(n);const config={responsive:true,displaylogo:false,scrollZoom:true};
function scene(detail){const r={xaxis:{title:{text:'x, мм'}},yaxis:{title:{text:'y, мм'}},zaxis:{title:{text:'z, мм'}},aspectmode:'data',camera:{eye:detail?{x:-2.2,y:.65,z:.15}:{x:-1.7,y:.55,z:.4},up:{x:0,y:0,z:1},projection:{type:'orthographic'}}};return r}
function contacts(size){const montage=D.patches[size],traces=[];
 montage.forEach((p,i)=>{traces.push({type:'mesh3d',x:p.nodes.map(v=>v[0]),y:p.nodes.map(v=>v[1]),z:p.nodes.map(v=>v[2]),i:p.faces.map(v=>v[0]),j:p.faces.map(v=>v[1]),k:p.faces.map(v=>v[2]),color:colors[i],opacity:1,flatshading:true,showlegend:false,name:roles[i],hovertemplate:roles[i]+' · площадь '+num(p.area_mm2,2)+' мм²<extra></extra>'});
  const label=p.centroid.map((x,k)=>x+8*p.normal[k]);traces.push({type:'scatter3d',mode:'lines+text',x:[p.centroid[0],label[0]],y:[p.centroid[1],label[1]],z:[p.centroid[2],label[2]],text:['',roles[i]],textfont:{color:colors[i],size:19},line:{color:colors[i],width:2},hoverinfo:'skip',showlegend:false});
  traces.push({type:'scatter3d',mode:'markers',x:[p.centroid[0],p.requested[0]],y:[p.centroid[1],p.requested[1]],z:[p.centroid[2],p.requested[2]],text:['Центр площади '+roles[i],'Проецируемый центр '+roles[i]],marker:{size:[5,5],symbol:['diamond','circle-open'],color:colors[i],line:{color:'#fff',width:1}},hovertemplate:'%{text}<br>(%{x:.3f}, %{y:.3f}, %{z:.3f}) мм<extra></extra>',showlegend:false});});
 traces.push({type:'scatter3d',mode:'lines',x:montage.map(p=>p.requested[0]),y:montage.map(p=>p.requested[1]),z:montage.map(p=>p.requested[2]),line:{color:'#364856',width:3,dash:'dot'},hoverinfo:'skip',showlegend:false});return traces;}
async function showSize(){q('busy').textContent='Обновление…';selected=Number(q('size').value);const traces=contacts(selected);await Promise.all([Plotly.react('body',[body,...traces],{title:{text:'Расположение на теле',font:{size:18}},margin:{l:0,r:0,t:45,b:0},scene:scene(false),uirevision:'body'},config),Plotly.react('closeup',[local,...traces],{title:{text:'Фактические CEM-площадки · L = '+D.sizes[selected]+' мм',font:{size:18}},margin:{l:0,r:0,t:45,b:0},scene:scene(true),uirevision:'closeup'},config)]);
 q('coordinates').innerHTML='<table><tr><th>Электрод</th><th>x центра площади, мм</th><th>y, мм</th><th>z, мм</th><th>Площадь, мм²</th></tr>'+D.patches[selected].map((p,i)=>'<tr><td style="color:'+colors[i]+'">'+roles[i]+'</td>'+p.centroid.map(x=>'<td>'+num(x)+'</td>').join('')+'<td>'+num(p.area_mm2,2)+'</td></tr>').join('')+'</table>';q('busy').textContent='';}
async function neighbourhood(){const a=Number(q('angle').value),bg=D.states[a].map(row=>row.map(s=>s===1?null:s));await Plotly.react('neighbour-map',[{type:'heatmap',x:D.local_u,y:D.local_v,z:bg,zmin:0,zmax:3,colorscale:[[0,'#e5e8eb'],[.5,'#747e87'],[1,'#a48bbb']],showscale:false,hovertemplate:'u=%{x}, v=%{y}<br>Нет допустимого FEM-значения<extra></extra>'},{type:'heatmap',x:D.local_u,y:D.local_v,z:D.errors[a],zmin:2.8,zmax:20,colorscale:[[0,'#087443'],[.2,'#85c766'],[.5,'#e8cd64'],[1,'#ba2439']],colorbar:{title:{text:'RMSE, Ом'},thickness:12},hoverongaps:false,hovertemplate:'u=%{x}, v=%{y}<br>RMSE=%{z:.6f} Ом<extra></extra>',connectgaps:false,zsmooth:false},{type:'scatter',x:[13],y:[-55],mode:'markers',marker:{symbol:'diamond-open',size:17,color:'#c9123f',line:{width:3}},hoverinfo:'skip',showlegend:false}],{margin:{l:55,r:85,t:20,b:45},xaxis:{title:{text:'u, мм'},range:[8.5,17.5],constrain:'domain'},yaxis:{title:{text:'v, мм'},range:[-59.5,-50.5],scaleanchor:'x',constrain:'domain'}},config)}
async function start(){D.sizes.forEach((s,i)=>q('size').add(new Option(s,i)));q('size').value='8';D.phi.forEach((a,i)=>q('angle').add(new Option(a+'°',i)));q('angle').value=D.best_angle_index;
 const n=D.certificate.neighbourhoods['1'];q('neighbour-count').textContent='В окне 3 × 3 × 3: '+n.evaluated+' допустимых поз, '+n.invalid+' отказов QC, '+n.missing+' отсутствующих. Все остальные допустимые позы имеют большую RMSE.';
 q('neighbours').innerHTML='<table><tr><th>Изменение</th><th>RMSE, Ом</th></tr>'+D.direct.map(r=>'<tr><td>'+(['du','dv','dphi'].map((k,i)=>r[k]?['u','v','φ'][i]+' '+(r[k]>0?'+':'')+r[k]:'').filter(Boolean).join(''))+'</td><td>'+(r.rmse_ohm===null?'Отказ QC':num(r.rmse_ohm,6))+'</td></tr>').join('')+'</table>';
 q('loo').innerHTML='<table><tr><th>Исключённый L, мм</th><th>u; v; φ</th><th>Смещение центра, мм</th><th>RMSE 8 размеров, Ом</th><th>Ошибка исключённого размера, Ом</th></tr>'+D.loo.map(r=>'<tr><td>'+r.omitted_L_mm+'</td><td>'+r.u_mm+'; '+r.v_mm+'; '+r.phi_deg+'</td><td>'+num(r.centre_distance_to_best_mm,2)+'</td><td>'+num(r.training_8_size_rmse_ohm,3)+'</td><td>'+num(r.held_out_error_ohm,2)+'</td></tr>').join('')+'</table>';
 await showSize();await neighbourhood();
 for(const [key,name,best] of [['u','u, мм',13],['v','v, мм',-55],['phi','φ, °',14]]){const s=D.slices[key];await Plotly.newPlot('slice-'+key,[{x:s.map(p=>p.value),y:s.map(p=>p.rmse_ohm),type:'scatter',mode:'lines+markers',connectgaps:false,line:{color:'#2877a6'},name:'Готовые FEM'},{x:[best],y:[D.certificate.best.rmse_ohm],mode:'markers',marker:{symbol:'diamond',size:11,color:'#b71937'},name:'Выбрано'}],{title:{text:'Срез по '+name},margin:{l:50,r:10,t:45,b:50},xaxis:{title:{text:name}},yaxis:{title:{text:'RMSE, Ом'},rangemode:'tozero'},showlegend:false},config)}
 await Plotly.newPlot('curve',[{x:D.sizes,y:D.experiment,name:'Эксперимент',mode:'lines+markers',line:{color:'#172b3c',dash:'dot'}},{x:D.sizes,y:D.fem,name:'FEM',mode:'lines+markers',line:{color:'#167abd'}},{x:D.sizes,y:D.planar,name:'2 слоя, h = '+num(D.h_eff_mm,2)+' мм',mode:'lines+markers',line:{color:'#208649'}}],{title:{text:'Девять размеров в выбранной позе'},margin:{l:55,r:10,t:45,b:85},xaxis:{title:{text:'Размер L, мм'}},yaxis:{title:{text:'Импеданс, Ом'}},legend:{orientation:'h',y:-.28}},config);
 await Plotly.newPlot('residual',[{x:D.sizes,y:D.fem.map((x,i)=>x-D.experiment[i]),type:'bar',marker:{color:'#3386b7'}}],{title:{text:'Остатки FEM − эксперимент'},margin:{l:55,r:10,t:45,b:65},xaxis:{title:{text:'Размер L, мм'}},yaxis:{title:{text:'Остаток, Ом'},zeroline:true}},config);
 q('size').addEventListener('change',()=>showSize().catch(error));q('angle').addEventListener('change',()=>neighbourhood().catch(error));q('reset').addEventListener('click',()=>Promise.all([Plotly.relayout('body',{'scene.camera':scene(false).camera}),Plotly.relayout('closeup',{'scene.camera':scene(true).camera})]));window.bestPoseReady=true;}
function error(e){q('error').textContent=String(e.stack||e);q('busy').textContent='Ошибка';}start().catch(error);
</script></body></html>'''

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,default=DEFAULT);args=p.parse_args();build(args.directory)
