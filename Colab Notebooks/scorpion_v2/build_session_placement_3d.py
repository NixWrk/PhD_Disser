"""Local CT display meshes and an offline 3D review; never an FEM geometry."""
from pathlib import Path
import json

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from skimage.measure import marching_cubes
import trimesh

from audit_scorpion_session import read, sha, write
from session_20261022 import EXPERIMENT, SUBJECT_NAMES

STEM='20.41_3D_боковая_установка_Ника_и_Ярослава_Scorpion'
PREP=EXPERIMENT/'derived/exploratory/ct_preparation_3340'
OUT=PREP/'viewer_3d'
Y_OFFSETS=list(range(-20,11,5))
DEFAULT_Y_OFFSET=-15  # Working scenario after the author's further negative-Y correction.
LAYERS={'body':('Грудная клетка','#d0c9c1',.22,45000),
        'lung':('Лёгкие','#419aba',.45,35000),
        'heart':('Сердце','#b91c1c',.65,18000),
        'bone':('Кости','#b6a57f',.6,45000),
        'fat':('Жир','#eab308',.2,35000)}


def reduce_mesh(mesh,limit):
    if len(mesh.faces)>limit:
        mesh=mesh.simplify_quadric_decimation(face_count=limit)
    if not np.isfinite(mesh.vertices).all():raise ValueError('Nonfinite display mesh')
    return mesh


def mask_mesh(path,limit):
    image=nib.load(path);mask=np.asarray(image.dataobj)>0
    # Padding closes display cuts at the field-of-view ends. This operation
    # and the mesh simplification never feed the quantitative preparation.
    v,f,_,_=marching_cubes(np.pad(mask,1).astype(np.uint8),level=.5,step_size=2)
    v=nib.affines.apply_affine(image.affine,v-1);v[:,:2]*=-1
    return reduce_mesh(trimesh.Trimesh(v,f,process=False),limit)


def shifted_skin_scenarios(candidate,body_image,lung_image):
    """Reproject each AP plane onto the own native skin, without changing masks."""
    body_image=nib.as_closest_canonical(body_image)
    lung_image=nib.as_closest_canonical(lung_image)
    affine=body_image.affine
    if (body_image.shape!=lung_image.shape or not np.allclose(affine,lung_image.affine)
            or not np.allclose(affine[:3,:3],np.diag(np.diag(affine[:3,:3])),atol=1e-5)
            or np.any(np.diag(affine[:3,:3])<=0)):
        raise ValueError('Review requires matching axis-aligned own RAS masks')
    body=np.asarray(body_image.dataobj)>0;lung=np.asarray(lung_image.dataobj)>0
    x,y,z=[affine[a,3]+np.arange(body.shape[a])*affine[a,a] for a in range(3)]
    reference=np.asarray(candidate['center']['lps_mm'])
    k0=int(np.argmin(abs(z-reference[2])))
    scenarios={}
    for shift in Y_OFFSETS:
        requested_y=reference[1]+shift
        j=int(np.argmin(abs(y+requested_y)))  # LPS Y = -RAS Y.
        if -requested_y<y[0] or -requested_y>y[-1]:
            raise ValueError('Requested review plane outside CT')
        points={}
        for k in np.flatnonzero(abs(z-z[k0])<=100):
            occupied=np.flatnonzero(body[:,j,k])
            extent=np.flatnonzero(np.any(body[:,:,k],axis=1))
            if not len(occupied) or not len(extent) or occupied[-1]==len(x)-1:continue
            xmid=(x[extent[0]]+x[extent[-1]])/2
            if not np.any(lung[:,j,k] & (x>xmid)):continue
            # Same binary isolevel 0.49 as the source candidate surface.
            points[k]=[-float(x[occupied[-1]]+.51*affine[0,0]),-float(y[j]),float(z[k])]
        if k0 not in points:raise ValueError('No right lung-supported centre at requested Y')
        first=last=k0
        while first-1 in points:first-=1
        while last+1 in points:last+=1
        trace=np.asarray([points[k] for k in range(first,last+1)])
        arc=np.r_[0.,np.linalg.norm(np.diff(trace,axis=0),axis=1).cumsum()]
        arc-=arc[k0-first]
        if arc[0]>-70 or arc[-1]<70:
            raise ValueError('Review montage exceeds contiguous own skin trace')
        contacts={}
        for length in (140,100,60):
            offsets=np.array([-length/2,-length/4,length/4,length/2])
            contacts[length]=np.column_stack([np.interp(offsets,arc,trace[:,a]) for a in range(3)]).tolist()
        scenarios[shift]=dict(requested_delta_y_mm=shift,requested_y_lps_mm=float(requested_y),
            actual_delta_y_mm=float(points[k0][1]-reference[1]),center_lps_mm=points[k0],
            longitudinal_trace_lps_mm=trace.tolist(),longitudinal_trace_arc_offsets_mm=arc.tolist(),
            contacts_lps_mm=contacts)
    if not np.allclose(scenarios[0]['center_lps_mm'],reference):
        raise ValueError('Zero shift does not reproduce source candidate')
    for length,pts in scenarios[0]['contacts_lps_mm'].items():
        original=np.asarray(candidate['longitudinal_trace_lps_mm'])
        arc=np.asarray(candidate['longitudinal_trace_arc_offsets_mm'])
        offsets=np.array([-length/2,-length/4,length/4,length/2])
        expected=np.column_stack([np.interp(offsets,arc,original[:,a]) for a in range(3)])
        if not np.allclose(pts,expected):raise ValueError('Zero shift changed source contacts')
    return scenarios


def build():
    OUT.mkdir(parents=True,exist_ok=True)
    summary=read(PREP/'surface_and_candidate_summary.local.json')
    fig=go.Figure();metadata=[];bounds=[];manifest=[];review={}
    for subject in ['nik','yaroslav']:
        base=PREP/subject/'body_surface_investigation/diagnostics'
        diagnostic=read(base/'body_surface_diagnostic.local.json')
        cfile=Path(summary['subjects'][subject]['candidate_json']['path']);candidate=read(cfile)
        if sha(cfile)!=summary['subjects'][subject]['candidate_json']['sha256']:
            raise ValueError('Candidate geometry changed')
        body_source=Path(candidate['source_body']['path'])
        lung_source=Path(diagnostic['inputs']['lung']['path'])
        if sha(body_source)!=candidate['source_body']['sha256'] or sha(lung_source)!=diagnostic['inputs']['lung']['sha256']:
            raise ValueError('Source review masks changed')
        scenarios=shifted_skin_scenarios(candidate,nib.load(body_source),nib.load(lung_source))
        review[subject]=dict(candidate_source=dict(path=str(cfile),sha256=sha(cfile)),
            body_source=candidate['source_body'],lung_source=diagnostic['inputs']['lung'],scenarios=scenarios)
        for kind,(label,color,opacity,limit) in LAYERS.items():
            source=(base/'diagnostic_native_body.stl' if kind=='body' else
                    Path(diagnostic['inputs'][kind]['path']))
            if kind!='body' and sha(source)!=diagnostic['inputs'][kind]['sha256']:
                raise ValueError('Tissue mask changed')
            mesh=(reduce_mesh(trimesh.load(source,force='mesh'),limit) if kind=='body' else mask_mesh(source,limit))
            vertices=np.asarray(mesh.vertices,dtype=np.float32);faces=np.asarray(mesh.faces,dtype=np.int32)
            bounds.append(mesh.bounds)
            np.savez_compressed(OUT/(subject+'_'+kind+'.local.npz'),vertices=vertices,faces=faces)
            fig.add_trace(go.Mesh3d(x=vertices[:,0],y=vertices[:,1],z=vertices[:,2],
                i=faces[:,0],j=faces[:,1],k=faces[:,2],name=label,color=color,opacity=opacity,
                flatshading=False,lighting=dict(ambient=.5,diffuse=.75,specular=.12),
                visible=subject=='nik' and kind not in ('bone','fat'),showlegend=False,
                hovertemplate=label+'<br>X=%{x:.1f}; Y=%{y:.1f}; Z=%{z:.1f} мм<extra></extra>'))
            metadata.append(dict(subject=subject,kind=kind))
            manifest.append(dict(subject=subject,kind=kind,source_path=str(source),source_sha256=sha(source),
                                 display_vertices=len(vertices),display_faces=len(faces),used_for_FEM=False))
        selected=scenarios[DEFAULT_Y_OFFSET]
        region=np.array([p['lps_mm'] for p in candidate['region_candidates']]);centre=np.array(selected['center_lps_mm'])
        for kind,pts,color,size,label in [('region',region,'#0d9488',4,'Варианты центра'),
                ('reference',np.array(candidate['center']['lps_mm'])[None,:],'#64748b',6,'Исходный центр'),
                ('centre',centre[None,:],'#f59e0b',8,'Центр-кандидат')]:
            fig.add_trace(go.Scatter3d(x=pts[:,0],y=pts[:,1],z=pts[:,2],mode='markers',name=label,
                marker=dict(color=color,size=size,line=dict(color='#172033',width=1)),showlegend=False,
                visible=subject=='nik',hovertemplate=label+'<br>X=%{x:.1f}; Y=%{y:.1f}; Z=%{z:.1f} мм<extra></extra>'))
            metadata.append(dict(subject=subject,kind=kind))
        for length in [140,100,60]:
            pts=np.array(selected['contacts_lps_mm'][length])
            fig.add_trace(go.Scatter3d(x=pts[:,0],y=pts[:,1],z=pts[:,2],mode='lines+markers+text',
                text=['A','M','N','B'],customdata=['токовый','измерительный','измерительный','токовый'],textposition='top center',
                textfont=dict(size=14,color='#111827'),marker=dict(size=7,color=['#dc2626','#1d4ed8','#1d4ed8','#dc2626'],line=dict(color='white',width=1)),
                line=dict(color='#475569',width=4),name=f'Пример {length} мм',showlegend=False,
                visible=subject=='nik' and length==140,
                hovertemplate='%{text}: %{customdata}<br>X=%{x:.1f}; Y=%{y:.1f}; Z=%{z:.1f} мм<extra></extra>'))
            metadata.append(dict(subject=subject,kind='contacts',length=length))
    low=np.min([b[0] for b in bounds],axis=0);high=np.max([b[1] for b in bounds],axis=0)
    centre=(low+high)/2;span=float(np.max(high-low))*1.05
    ranges=[[float(x-span/2),float(x+span/2)] for x in centre]
    fig.update_layout(template='plotly_white',margin=dict(l=0,r=0,t=0,b=0),height=740,
        uirevision='preserve-scorpion-review-camera',
        scene=dict(uirevision='preserve-scorpion-review-camera',aspectmode='data',
            xaxis=dict(title='X LPS, мм · влево',range=ranges[0]),
            yaxis=dict(title='Y LPS, мм · назад',range=ranges[1]),
            zaxis=dict(title='Z LPS, мм · вверх',range=ranges[2]),
            camera=dict(eye=dict(x=-2.1,y=-1.6,z=.65),up=dict(x=0,y=0,z=1))))
    plot=fig.to_html(full_html=False,include_plotlyjs=False,div_id='ct-montage-3d',
                     config=dict(responsive=True,displaylogo=False,scrollZoom=True))
    page='''<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>3D: боковая установка Ника и Ярослава</title><style>
body{margin:0;font:17px/1.6 system-ui,sans-serif;color:#172033;background:#f8fafc}
main{max-width:1450px;margin:auto;padding:22px}h1{font-size:27px;margin:0 0 12px}
.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin:14px 0}
button,select{font:inherit;padding:6px 12px;background:white;border:1px solid #94a3b8;border-radius:5px;cursor:pointer}
.hint{background:#fffbeb;border-left:4px solid #d97706;padding:9px 15px}
#status{font-weight:650}.layers{display:flex;gap:16px;flex-wrap:wrap;margin:8px 0}
#ct-montage-3d{border:1px solid #cbd5e1;background:white;max-width:100%}label{cursor:pointer}
@media(max-width:600px){main{padding:12px}h1{font-size:23px}}</style>
<script>'''+get_plotlyjs()+'''</script><main>
<h1>3D: боковые сборки Ника и Ярослава</h1>
<p>Поверните модель мышью, увеличьте колесом; смещение — с зажатой правой кнопкой.
Выбор участника, размера, смещения и видимости тканей сохраняет текущий ракурс.</p>
<div class="controls"><label>Участник: <select id="subject"><option value="nik">Ник</option><option value="yaroslav">Ярослав</option></select></label>
<label>Пример сборки: <select id="size"><option>140</option><option>100</option><option>60</option></select> мм</label>
<button id="right">Вид справа</button><button id="front">Вид спереди</button><button id="oblique">Объёмный вид</button></div>
<div class="controls"><label>Смещение Y: <select id="y-shift">'''+''.join(
    f'<option value="{v}"'+(' selected' if v==DEFAULT_Y_OFFSET else '')+f'>{v:+d} мм</option>' for v in Y_OFFSETS)+'''</select></label>
<button id="zero-shift">Исходное положение</button>
<span>Минус Y — вперёд, плюс Y — назад. Выбор сохраняется отдельно для каждого участника.</span></div>
<div class="layers"><label><input type="checkbox" data-layer="body" checked> Грудная клетка</label>
<label><input type="checkbox" data-layer="lung" checked> Лёгкие</label>
<label><input type="checkbox" data-layer="heart" checked> Сердце</label>
<label><input type="checkbox" data-layer="bone"> Кости</label>
<label><input type="checkbox" data-layer="fat"> Жир</label>
<label><input type="checkbox" data-layer="region" checked> Исходная область вариантов</label></div>
<p id="status" aria-live="polite"></p>'''+plot+'''
<p>Жёлтый маркер — смещённый центр-кандидат; серый — исходный центр;
зелёные маркеры — исходная область вариантов.
A и B — центры токовых контактов, M и N — центры измерительных.
Размер примера отложен вдоль кожной кривой. Маркеры показывают центры,
их размер на экране не задаёт физическую площадь контакта.</p>
<p>Для расчётного сценария выбрано смещение −15 мм у обоих участников.
24.10.2026 автор предложил сдвинуть сборку ещё к отрицательному Y и запустить расчёты.
Величина −15 мм и одинаковое смещение у обоих — рабочие допущения;
численного измерения положения это уточнение не задаёт.
При смене Y заново выбрана кожная кривая по индивидуальной КТ; X следует наружному контуру,
высота центра Z сохраняется. Положение Y округляется до ближайшей плоскости вокселей.
Фактические координаты центра показаны в строке над сценой.</p>
<p class="hint">Это индивидуальная КТ-геометрия для проверки положения, а не принятая
расчётная FEM-модель. Наружный контур не прошёл все стандартные проверки.
Поверхности упрощены только для просмотра; маски автоматические и требуют экспертной оценки.
Исходный центр выбран по уровню маски сердца и середине переднезадней ширины тела,
без подгонки импеданса. Фактическое положение боковой сборки ещё нужно уточнить.</p>
</main><script>
const metadata='''+json.dumps(metadata,ensure_ascii=False)+''';
const review='''+json.dumps({s:d['scenarios'] for s,d in review.items()},ensure_ascii=False)+''';
const shifts={nik:'''+str(DEFAULT_Y_OFFSET)+''',yaroslav:'''+str(DEFAULT_Y_OFFSET)+'''};
const graph=document.getElementById('ct-montage-3d');let savedCamera=null,redraw=Promise.resolve();
graph.on('plotly_relayout',event=>{if(event['scene.camera'])savedCamera=JSON.parse(JSON.stringify(event['scene.camera']));});
function update(){const subject=document.getElementById('subject').value,size=Number(document.getElementById('size').value);
const shift=shifts[subject],scenario=review[subject][shift];document.getElementById('y-shift').value=String(shift);
const layers={};document.querySelectorAll('[data-layer]').forEach(e=>layers[e.dataset.layer]=e.checked);
const visible=metadata.map(m=>m.subject===subject&&(m.kind==='contacts'?m.length===size:((m.kind==='centre'||m.kind==='reference')||layers[m.kind]===true)));
const indices=[],x=[],y=[],z=[];metadata.forEach((m,index)=>{const p=m.subject===subject&&m.kind==='contacts'?scenario.contacts_lps_mm[m.length]:
    (m.subject===subject&&m.kind==='centre'?[scenario.center_lps_mm]:null);
    if(p){indices.push(index);x.push(p.map(v=>v[0]));y.push(p.map(v=>v[1]));z.push(p.map(v=>v[2]));}});
redraw=redraw.then(()=>{const camera=savedCamera?JSON.parse(JSON.stringify(savedCamera)):null;
    return Plotly.restyle(graph,{x:x,y:y,z:z},indices).then(()=>Plotly.restyle(graph,{visible:visible}))
        .then(()=>camera?Plotly.relayout(graph,{'scene.camera':camera}):null);});
document.getElementById('status').textContent=(subject==='nik'?'Ник':'Ярослав')+' · пример '+size+' мм · ΔY = '+shift+
    ' мм · центр X / Y / Z = '+scenario.center_lps_mm.map(v=>v.toFixed(1)).join(' / ')+' мм (LPS)';}
document.getElementById('subject').addEventListener('change',update);
document.getElementById('size').addEventListener('change',update);
document.getElementById('y-shift').addEventListener('change',event=>{shifts[document.getElementById('subject').value]=Number(event.target.value);update();});
document.getElementById('zero-shift').addEventListener('click',()=>{shifts[document.getElementById('subject').value]=0;update();});
document.querySelectorAll('[data-layer]').forEach(e=>e.addEventListener('change',update));
for(const [id,eye] of Object.entries({right:{x:-2.4,y:0,z:.2},front:{x:0,y:-2.4,z:.2},oblique:{x:-2.1,y:-1.6,z:.65}})){
document.getElementById(id).addEventListener('click',()=>Plotly.relayout(graph,{'scene.camera':{eye:eye,up:{x:0,y:0,z:1},center:{x:0,y:0,z:0}}}));}
update();
</script></html>'''
    output=Path(__file__).resolve().parent/(STEM+'.html');output.write_text(page,encoding='utf8')
    write(OUT/'y_review_scenarios.local.json',dict(producer_sha256=sha(__file__),subjects=review,
        coordinate_system='LPS_mm',author_confirmed_direction='negative_Y',
        default_delta_y_mm=DEFAULT_Y_OFFSET,magnitude_confirmed=False,subject_scope_confirmed=False,
        author_instruction_date='2026-10-24',author_requested_calculations=True,
        review_only=True,geometry_accepted_for_FEM=False))
    write(OUT/'manifest.local.json',dict(producer_sha256=sha(__file__),sources=manifest,
        y_review_sha256=sha(OUT/'y_review_scenarios.local.json'),
        display_only=True,geometry_accepted_for_FEM=False,output_sha256=sha(output)))
    print(output.name,round(output.stat().st_size/2**20,2),'MiB')
    return output


if __name__=='__main__':build()
