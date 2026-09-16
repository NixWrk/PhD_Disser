"""Full CT tissue surfaces and exact saved contact triangles; display only."""
from pathlib import Path
import json
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.io import loadmat
from skimage.measure import marching_cubes
import trimesh
import plotly.graph_objects as go
from lateral_fat_prepare import PIPE, BASE, OUT as FAT, sha, read, write

OUT=PIPE/'output/exploratory/lateral_fat_inverse_20260916'


def meshtrace(v,f,name,color,opacity,**kw):
    return go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],
        name=name,color=color,opacity=opacity,showlegend=True,flatshading=False,
        lighting=dict(ambient=.65,diffuse=.75,specular=.15,roughness=.8),
        hovertemplate=name+'<extra></extra>',**kw)


def simplify(v,f,limit):
    obj=trimesh.Trimesh(v,f,process=False)
    if len(f)>limit:obj=obj.simplify_quadric_decimation(face_count=limit)
    return np.asarray(obj.vertices),np.asarray(obj.faces)


def build():
    OUT.mkdir(exist_ok=True);meta=read(FAT/'segmentation/segmentation.json');contacts=read(OUT/'contacts.local.json')
    assert contacts['prepared_sha256']==sha(BASE/'prepared_fem.mat')
    assert contacts['mesh_sha256']==sha(BASE/'refined_mesh.mat')
    M=loadmat(BASE/'refined_mesh.mat',simplify_cells=True)
    nodes=np.asarray(M['nodes_mm']);faces=np.asarray(M['boundary'],int)-1
    v,f=simplify(nodes,faces,28000)
    fig=go.Figure([meshtrace(v,f,'Наружная поверхность / мягкотканный фон','#c1b8ab',.08)])
    grid=None;affine=None;counts={}
    for name,label in [('fat',4),('heart',3),('bone',2),('lung',1)]:
        p=FAT/'segmentation'/(name+'.nii.gz');assert sha(p)==meta['products'][name]['sha256']
        im=nib.load(p);mask=np.asanyarray(im.dataobj)>0
        if grid is None:grid=np.zeros(im.shape,np.uint8);affine=im.affine
        np.testing.assert_allclose(im.affine,affine,rtol=0,atol=1e-5)
        grid[mask]=label
    # Extract the original voxel-mask surfaces. Decimation changes display only.
    for name,label,title,color,opacity,limit in [('lung',1,'Лёгкие','#238bdf',.80,35000),
         ('bone',2,'Кости','#e0dbc5',.40,60000),('heart',3,'Сердце','#c43849',.85,18000),
         ('fat',4,'Жир: подкожный и внутренний','#eeaa25',.06,65000)]:
        field=np.pad((grid==label).astype(np.uint8),1)
        verts,tri,_,_=marching_cubes(field,.5,step_size=1,allow_degenerate=False)
        verts=nib.affines.apply_affine(np.diag([-1.,-1.,1.,1.])@affine,verts-1)
        before=len(tri);verts,tri=simplify(verts,tri,limit)
        fig.add_trace(meshtrace(verts,tri,title,color,opacity))
        counts[name]=dict(original_surface_faces=before,display_faces=len(tri),priority_voxels=int(np.count_nonzero(grid==label)))
    sizes=np.asarray(M['sizes_mm'],int).ravel();names=['I+','V+','V−','I−'];colors=['#db2727','#32a64a','#ad3cb0','#db2727']
    qc=pd.read_csv(BASE/'contact_qc.csv');areas=[]
    for length in sizes:
        group=[x for x in contacts['contacts'] if x['L_mm']==length];assert len(group)==4
        centers=[]
        for item in group:
            v=np.asarray(item['nodes_mm']);f=np.asarray(item['triangles_zero_based'],int);e=item['electrode']-1
            area=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)/2
            center=np.average(v[f].mean(1),axis=0,weights=area);centers.append(center)
            expected=qc[(qc.L_mm==length)&(qc.electrode==e+1)].actual_area_mm2.iloc[0]
            assert abs(area.sum()-expected)<1e-8;areas.append(float(area.sum()))
            fig.add_trace(meshtrace(v,f,names[e],colors[e],1,visible=length==140,legendgroup=names[e]))
        c=np.asarray(centers)
        fig.add_trace(go.Scatter3d(x=c[:,0],y=c[:,1],z=c[:,2],mode='lines+markers+text',
            text=names,textposition='top center',name=f'Сборка {length} мм',showlegend=False,
            line=dict(color='#202020',width=5),marker=dict(color=colors,size=4),visible=length==140,
            hovertemplate='%{text}<br>X=%{x:.1f}; Y=%{y:.1f}; Z=%{z:.1f} мм<extra></extra>'))
    n=len(fig.data)
    profiles=[('Все ткани',[.08,.80,.40,.85,.06]),('Органы без жира',[.06,.70,.85,1,0]),
        ('Жир крупно',[.05,.08,.12,.15,.85]),('Наружная поверхность',[.80,.1,.1,.1,0])]
    fig.update_layout(template='plotly_white',width=1280,height=880,margin=dict(l=5,r=5,t=125,b=15),
        title=dict(text='Полная сегментация КТ и конечные электроды · сборка 140 мм',x=.5,y=.99,yanchor='top'),
        scene=dict(aspectmode='data',xaxis_title='X, мм',yaxis_title='Y, мм',zaxis_title='Z, мм',
            camera=dict(eye=dict(x=-1.35,y=-1.2,z=.45),up=dict(x=0,y=0,z=1))),
        legend=dict(x=0,y=.93,bgcolor='rgba(255,255,255,.78)'),uirevision='fixed_C01',
        updatemenus=[dict(x=.02,y=1.08,buttons=[dict(label=label,method='restyle',args=[{'opacity':op},list(range(5))]) for label,op in profiles]),
         dict(x=.65,y=1.08,active=8,buttons=[dict(label=f'{length} мм',method='update',args=[{'visible':[True]*5+[k//5==i for k in range(n-5)]},
          {'title.text':f'Полная сегментация КТ и конечные электроды · сборка {length} мм'}]) for i,length in enumerate(sizes)])],
        annotations=[dict(text='Прозрачность и размер сборки переключаются независимо; ткани можно скрывать в легенде.',
         x=.5,y=1.03,xref='paper',yref='paper',showarrow=False,font=dict(size=13))])
    fig.write_html(OUT/'segmentation_3d.html',include_plotlyjs=True,config={'displaylogo':False,'scrollZoom':True})
    fig.write_json(OUT/'segmentation_3d.local.json')
    write(OUT/'segmentation_3d_qc.json',dict(complete=True,source_sha256=sha(__file__),segmentation_sha256=sha(FAT/'segmentation/segmentation.json'),
        mesh_sha256=sha(BASE/'refined_mesh.mat'),contacts_sha256=sha(OUT/'contacts.local.json'),counts=counts,
        coordinate_system='FEM DICOM LPS mm; standard RAS axis flip only',same_priority_as_FEM=True,
        all_36_contact_areas_match=True,area_range_mm2=[min(areas),max(areas)],
        surface_simplification_display_only=True,background_shown_as_external_envelope=True,
        CT_organs_displayed_across_full_acquired_FOV_not_clipped_to_FEM=True,
        output_sha256=sha(OUT/'segmentation_3d.html')))
    print('Built 3D: four segmented organ groups, body envelope, all 36 exact contact patches.',flush=True)


def figure():
    qc=read(OUT/'segmentation_3d_qc.json');assert qc['complete'] and qc['output_sha256']==sha(OUT/'segmentation_3d.html')
    html=(OUT/'segmentation_3d.html').read_text(encoding='utf-8')
    text=html[html.rfind('Plotly.newPlot(')+len('Plotly.newPlot('):].lstrip();dec=json.JSONDecoder()
    _,end=dec.raw_decode(text);text=text[end:].lstrip()[1:].lstrip()
    data,end=dec.raw_decode(text);text=text[end:].lstrip()[1:].lstrip()
    layout,_=dec.raw_decode(text)
    return go.Figure(data=data,layout=layout)


if __name__=='__main__':build()
