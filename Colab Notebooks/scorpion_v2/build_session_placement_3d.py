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


def build():
    OUT.mkdir(parents=True,exist_ok=True)
    summary=read(PREP/'surface_and_candidate_summary.local.json')
    fig=go.Figure();metadata=[];bounds=[];manifest=[]
    for subject in ['nik','yaroslav']:
        base=PREP/subject/'body_surface_investigation/diagnostics'
        diagnostic=read(base/'body_surface_diagnostic.local.json')
        cfile=Path(summary['subjects'][subject]['candidate_json']['path']);candidate=read(cfile)
        if sha(cfile)!=summary['subjects'][subject]['candidate_json']['sha256']:
            raise ValueError('Candidate geometry changed')
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
        trace=np.array(candidate['longitudinal_trace_lps_mm']);arc=np.array(candidate['longitudinal_trace_arc_offsets_mm'])
        region=np.array([p['lps_mm'] for p in candidate['region_candidates']]);centre=np.array(candidate['center']['lps_mm'])
        for kind,pts,color,size,label in [('region',region,'#0d9488',4,'Варианты центра'),
                ('centre',centre[None,:],'#f59e0b',8,'Центр-кандидат')]:
            fig.add_trace(go.Scatter3d(x=pts[:,0],y=pts[:,1],z=pts[:,2],mode='markers',name=label,
                marker=dict(color=color,size=size,line=dict(color='#172033',width=1)),showlegend=False,
                visible=subject=='nik',hovertemplate=label+'<br>X=%{x:.1f}; Y=%{y:.1f}; Z=%{z:.1f} мм<extra></extra>'))
            metadata.append(dict(subject=subject,kind=kind))
        for length in [140,100,60]:
            offsets=np.array([-length/2,-length/4,length/4,length/2])
            if offsets.min()<arc.min() or offsets.max()>arc.max():raise ValueError('Contact beyond own skin trace')
            pts=np.column_stack([np.interp(offsets,arc,trace[:,axis]) for axis in range(3)])
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
Выбор участника, размера и видимости тканей сохраняет текущий ракурс.</p>
<div class="controls"><label>Участник: <select id="subject"><option value="nik">Ник</option><option value="yaroslav">Ярослав</option></select></label>
<label>Пример сборки: <select id="size"><option>140</option><option>100</option><option>60</option></select> мм</label>
<button id="right">Вид справа</button><button id="front">Вид спереди</button><button id="oblique">Объёмный вид</button></div>
<div class="layers"><label><input type="checkbox" data-layer="body" checked> Грудная клетка</label>
<label><input type="checkbox" data-layer="lung" checked> Лёгкие</label>
<label><input type="checkbox" data-layer="heart" checked> Сердце</label>
<label><input type="checkbox" data-layer="bone"> Кости</label>
<label><input type="checkbox" data-layer="fat"> Жир</label>
<label><input type="checkbox" data-layer="region" checked> Область вариантов</label></div>
<p id="status">Ник · пример 140 мм</p>'''+plot+'''
<p>Жёлтый маркер — центр-кандидат; зелёные маркеры — варианты его смещения.
A и B — центры токовых контактов, M и N — центры измерительных.
Размер примера отложен вдоль кожной кривой. Маркеры показывают центры,
их размер на экране не задаёт физическую площадь контакта.</p>
<p class="hint">Это индивидуальная КТ-геометрия для проверки положения, а не принятая
расчётная FEM-модель. Наружный контур не прошёл все стандартные проверки.
Поверхности упрощены только для просмотра; маски автоматические и требуют экспертной оценки.
Центр выбран по уровню маски сердца и середине переднезадней ширины тела,
без подгонки импеданса. Фактическое положение боковой сборки ещё нужно уточнить.</p>
</main><script>
const metadata='''+json.dumps(metadata,ensure_ascii=False)+''';
const graph=document.getElementById('ct-montage-3d');let savedCamera=null;
graph.on('plotly_relayout',event=>{if(event['scene.camera'])savedCamera=JSON.parse(JSON.stringify(event['scene.camera']));});
function update(){const subject=document.getElementById('subject').value,size=Number(document.getElementById('size').value);
const layers={};document.querySelectorAll('[data-layer]').forEach(e=>layers[e.dataset.layer]=e.checked);
const visible=metadata.map(m=>m.subject===subject&&(m.kind==='contacts'?m.length===size:(m.kind==='centre'||layers[m.kind]===true)));
const camera=savedCamera?JSON.parse(JSON.stringify(savedCamera)):null;
Plotly.restyle(graph,{visible:visible}).then(()=>camera?Plotly.relayout(graph,{'scene.camera':camera}):null);
document.getElementById('status').textContent=(subject==='nik'?'Ник':'Ярослав')+' · пример '+size+' мм';}
document.getElementById('subject').addEventListener('change',update);
document.getElementById('size').addEventListener('change',update);
document.querySelectorAll('[data-layer]').forEach(e=>e.addEventListener('change',update));
for(const [id,eye] of Object.entries({right:{x:-2.4,y:0,z:.2},front:{x:0,y:-2.4,z:.2},oblique:{x:-2.1,y:-1.6,z:.65}})){
document.getElementById(id).addEventListener('click',()=>Plotly.relayout(graph,{'scene.camera':{eye:eye,up:{x:0,y:0,z:1},center:{x:0,y:0,z:0}}}));}
</script></html>'''
    output=Path(__file__).resolve().parent/(STEM+'.html');output.write_text(page,encoding='utf8')
    write(OUT/'manifest.local.json',dict(producer_sha256=sha(__file__),sources=manifest,
        display_only=True,geometry_accepted_for_FEM=False,output_sha256=sha(output)))
    print(output.name,round(output.stat().st_size/2**20,2),'MiB')
    return output


if __name__=='__main__':build()
