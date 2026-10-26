"""Local, interactive views of the actual subject geometry and planned contacts.

The report embeds a sibling local HTML in an iframe. Individual mesh arrays
remain outside Git; the tracked notebook and reader HTML contain no 3D arrays.
"""
from pathlib import Path
import html
import json

import numpy as np
import plotly.graph_objects as go
import trimesh

from audit_scorpion_session import read, sha
from build_session_placement_3d import mask_mesh, reduce_mesh
import subject_report_figures as R

FOLDER=Path(__file__).resolve().parent
FILENAMES={'nik':'33.41_3D_модель_и_электроды_Ника.local.html',
           'yaroslav':'33.42_3D_модель_и_электроды_Ярослава.local.html'}
LAYERS={'body':('Наружная поверхность и руки','#c8bfb4',.24,35000),
        'lung':('Лёгкие','#38a5cb',.45,28000),
        'heart':('Целое сердце','#b91c1c',.75,16000),
        'bone':('Кости','#b7a582',.65,38000),
        'fat':('Жир','#eab308',.2,28000)}
CASES=[(140,'2x2'),(100,'2x2'),(60,'2x2'),(60,'1x1')]


def contact_label(identifier):
    if identifier.startswith('arm'):
        side,role,level=identifier.split('_')
        return f'{role} · '+('правая' if side=='arm0' else 'левая')+' · '+('верх' if level=='top' else 'низ')
    size,role=identifier.split('_')
    return role+' · '+('токовый' if role in ('A','B') else 'измерительный')


def contact_disk(point,normal,radius):
    normal=np.array(normal,dtype=float,copy=True);normal/=np.linalg.norm(normal)
    axis=np.eye(3)[np.argmin(abs(normal))]
    u=np.cross(normal,axis);u/=np.linalg.norm(u);v=np.cross(normal,u)
    angle=np.arange(32)*2*np.pi/32
    ring=point+.15*normal+radius*(np.cos(angle)[:,None]*u+np.sin(angle)[:,None]*v)
    vertices=np.vstack([point+.15*normal,ring])
    faces=np.column_stack([np.zeros(32,dtype=int),np.arange(1,33),np.r_[np.arange(2,33),1]])
    return vertices,faces


def scene(subject):
    cfg,plan,q=R.geometry(subject);root=R.model_root(subject)
    passport=read(root/'surface/preparation.local.json')
    paths=[root/'surface/exterior.stl',root/'surface/contacts.local.json']
    for path in paths:
        if sha(path)!=passport['products'][path.name]:raise ValueError('Prepared geometry product changed: '+path.name)
    contacts=read(paths[1])['physical_contacts']
    exterior=trimesh.load(paths[0],force='mesh')
    figure=go.Figure();metadata=[];sources=[]
    for kind,(label,color,opacity,limit) in LAYERS.items():
        source=paths[0] if kind=='body' else Path(cfg['inputs']['masks'][kind])
        digest=sha(source)
        if kind!='body' and digest!=passport['sources'][str(source.resolve())]:
            raise ValueError('Own tissue mask changed: '+kind)
        mesh=reduce_mesh(exterior.copy(),limit) if kind=='body' else mask_mesh(source,limit)
        vertices=np.asarray(mesh.vertices,dtype=np.float32);faces=np.asarray(mesh.faces,dtype=np.int32)
        figure.add_trace(go.Mesh3d(x=vertices[:,0],y=vertices[:,1],z=vertices[:,2],
            i=faces[:,0],j=faces[:,1],k=faces[:,2],name=label,color=color,opacity=opacity,
            visible=kind not in ('bone','fat'),showlegend=False,
            lighting=dict(ambient=.6,diffuse=.7,specular=.1),hoverinfo='skip'))
        metadata.append(dict(kind='layer',layer=kind))
        sources.append(dict(role=kind,path=str(source),sha256=digest,display_faces=len(faces)))
    for contact in contacts:
        identifier=contact['id'];point=np.asarray(contact['projected_lps_mm']);is_arm=identifier.startswith('arm')
        current=('_I_' in identifier) if is_arm else identifier.endswith(('_A','_B'))
        color='#dc2626' if current else '#2563eb';label=contact_label(identifier)
        p,f=contact_disk(point,exterior.face_normals[contact['surface_face']],contact['nominal_disk_radius_mm'])
        data=dict(id=identifier,kind='disk',montage='arm' if is_arm else 'side',
                  size=None if is_arm else int(identifier.split('_')[0][4:]),bottom=identifier.endswith('_bottom'))
        active=is_arm or data['size']==140
        figure.add_trace(go.Mesh3d(x=p[:,0],y=p[:,1],z=p[:,2],i=f[:,0],j=f[:,1],k=f[:,2],
            color=color,visible=active,showlegend=False,hoverinfo='skip'))
        metadata.append(data)
        figure.add_trace(go.Scatter3d(x=[point[0]],y=[point[1]],z=[point[2]],
            mode='markers+text',text=[label],name=label,visible=active,showlegend=False,
            textposition='top center',textfont=dict(size=11,color='#172033'),
            marker=dict(size=5,color=color,line=dict(color='white',width=1)),
            customdata=[[identifier,contact['displacement_mm']]],
            hovertemplate=label+'<br>X / Y / Z = %{x:.2f} / %{y:.2f} / %{z:.2f} мм (LPS)'
                '<br>Номинальный диаметр: 5 мм<br>Проекция: %{customdata[1]:.6f} мм<extra></extra>'))
        metadata.append(dict(data,kind='centre',color=color))
    for length,variant in CASES:
        # One lateral path per alternative size, not three simultaneous arrays.
        if variant!='2x2':continue
        points=np.array([next(c['projected_lps_mm'] for c in contacts if c['id']==f'side{length}_{role}') for role in ('A','M','N','B')])
        figure.add_trace(go.Scatter3d(x=points[:,0],y=points[:,1],z=points[:,2],mode='lines',
            line=dict(color='#475569',width=3),hoverinfo='skip',visible=length==140,showlegend=False))
        metadata.append(dict(kind='path',montage='side',size=length,bottom=False))
    bounds=exterior.bounds;pad=np.ptp(bounds,axis=0)*.06
    view='subject-montage-'+subject
    figure.update_layout(template='plotly_white',height=650,margin=dict(l=0,r=0,t=0,b=0),
        uirevision=view,scene=dict(uirevision=view,aspectmode='data',
            xaxis=dict(title='X LPS, мм · влево',range=[bounds[0,0]-pad[0],bounds[1,0]+pad[0]]),
            yaxis=dict(title='Y LPS, мм · назад',range=[bounds[0,1]-pad[1],bounds[1,1]+pad[1]]),
            zaxis=dict(title='Z LPS, мм · вверх',range=[bounds[0,2]-pad[2],bounds[1,2]+pad[2]]),
            camera=dict(eye=dict(x=-1.75,y=-1.6,z=.85),up=dict(x=0,y=0,z=1))))
    return figure,metadata,dict(subject=subject,sources=sources,
        exterior_sha256=sha(paths[0]),contacts_sha256=sha(paths[1]),
        display_only=True,coordinates='LPS_mm',uses_prepared_projected_contact_centres=True,
        electrode_diameter_mm=5,circumference_mm=370,delta_Y_scenario_mm=-15,
        independent_anatomical_validation=False,interactive_camera_verified=False)


def build(subject):
    fig,metadata,passport=scene(subject)
    name=R.F.SUBJECT_NAMES[subject]
    plot=fig.to_html(full_html=False,include_plotlyjs=True,div_id='subject-geometry',
        config=dict(responsive=True,displaylogo=False,scrollZoom=True))
    options=''.join(f'<option value="{size}_{variant}">Боковая {size} мм + ТТРКГ {variant.replace("x","×")}</option>' for size,variant in CASES)
    layers=''.join(f'<label><input type="checkbox" data-layer="{key}"'+(' checked' if key not in ('bone','fat') else '')+f'>{label}</label>' for key,(label,*_) in LAYERS.items())
    template=Path(__file__).with_name('subject_geometry_3d.html').read_text(encoding='utf8')
    page=template.replace('__NAME__',name).replace('__OPTIONS__',options).replace('__LAYERS__',layers)
    page=page.replace('__PLOT__',plot).replace('__METADATA__',json.dumps(metadata,ensure_ascii=False))
    path=FOLDER/FILENAMES[subject]
    path.write_text('\n'.join(line.rstrip() for line in page.splitlines())+'\n',encoding='utf8',newline='\n')
    passport.update(html_sha256=sha(path),producer_sha256=sha(__file__),
        template_sha256=sha(Path(__file__).with_name('subject_geometry_3d.html')))
    out=R.model_root(subject)/'reader_figures/interactive_3d';out.mkdir(parents=True,exist_ok=True)
    (out/'display_manifest.local.json').write_text(json.dumps(passport,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return path


def widget(subject):
    path=build(subject);filename=html.escape(path.name,quote=True)
    return ('<div class="subject-geometry-view" style="margin:16px 0">'
        f'<p><a href="{filename}" target="_blank" rel="noopener">Открыть 3D-модель и электроды в отдельном окне</a></p>'
        f'<iframe title="{R.F.SUBJECT_NAMES[subject]}: 3D-модель и расположение электродов" '
        f'src="{filename}" width="100%" height="1200" loading="eager" '
        'style="border:1px solid #cbd5e1;border-radius:6px;background:white"></iframe>'
        '<p style="font-size:13px;color:#475569">Индивидуальная 3D-сцена находится рядом с HTML-отчётом '
        'в локальном файле; при переносе отчёта сохраните оба файла вместе.</p></div>')


if __name__=='__main__':
    for subject in ('nik','yaroslav'):print(build(subject))
