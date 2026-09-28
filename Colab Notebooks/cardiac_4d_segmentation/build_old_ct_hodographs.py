"""Geometric CT hodographs from sealed TotalSegmentator 4D masks; no DICOM copied."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
CHAMBERS=('heart_ventricle_left','heart_ventricle_right','heart_atrium_left','heart_atrium_right')

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def write_json(p,value):
    with Path(p).open('w',encoding='utf-8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')

def stats(mask,affine):
    count=int(np.count_nonzero(mask))
    if not count:raise ValueError('Empty mask')
    index=np.argwhere(mask).mean(axis=0)
    center=(affine@np.r_[index,1.])[:3]
    volume=count*abs(float(np.linalg.det(affine[:3,:3])))/1000
    faces=[[bool(np.any(np.take(mask,i,axis=axis))) for i in [0,-1]] for axis in range(3)]
    return {'center_ras_mm':center.tolist(),'volume_ml':volume,'boundary_faces':faces}

def compute(sequence, masks, output):
    sequence=Path(sequence);masks=Path(masks);output=Path(output)
    if output.exists():raise FileExistsError(output)
    manifest=load(sequence/'conversion_manifest.json')
    status_path=masks/'status/run_status.json'
    if not status_path.is_file():status_path=masks/'run_status.json'
    state=load(status_path)
    phases=sorted(manifest['phases'],key=lambda p:p['phase_percent'])
    assert len(phases)==20 and state['status']=='completed'
    assert len(state['completed_phase_ids'])==20
    if [p['phase_percent'] for p in phases]!=list(range(0,100,5)):raise ValueError('Unexpected phase ordering')
    rows=[]
    for phase in phases:
        pid=phase['phase_id'];ct_path=sequence/phase['nifti'];ct=nib.load(str(ct_path))
        if digest(ct_path)!=phase['sha256'] or not np.allclose(ct.affine,phase['affine_ras_mm'],atol=1e-5):raise ValueError('CT provenance or grid changed')
        paths={'whole_heart':masks/'automatic_masks_v2'/pid/'total_heart/heart.nii.gz'}
        paths.update({ch:masks/'automatic_masks_v2'/pid/'heartchambers_highres'/f'{ch}.nii.gz' for ch in CHAMBERS})
        arrays={}
        for key,p in paths.items():
            relative=p.relative_to(masks).as_posix()
            recorded={key.replace('\\','/'):value for key,value in state['phase_hashes'][pid].items()}
            if digest(p)!=recorded[relative]:raise ValueError('Mask hash changed: '+relative)
            img=nib.load(str(p))
            if img.shape!=ct.shape or not np.allclose(img.affine,ct.affine,atol=1e-4):raise ValueError('Mask grid mismatch: '+relative)
            arrays[key]=np.asanyarray(img.dataobj)>0
        blood=np.logical_or.reduce([arrays[ch] for ch in CHAMBERS])
        blood_overlap=int(sum(np.count_nonzero(arrays[a]&arrays[b]) for i,a in enumerate(CHAMBERS) for b in CHAMBERS[i+1:]))
        row={'phase_id':pid,'phase_percent':phase['phase_percent'],'whole_heart':stats(arrays['whole_heart'],ct.affine),'blood_union':stats(blood,ct.affine),
             'blood_outside_whole_heart_voxels':int(np.count_nonzero(blood&~arrays['whole_heart'])),'chamber_overlap_pair_voxels':blood_overlap}
        rows.append(row)
    for target in ('whole_heart','blood_union'):
        c0=np.array(rows[0][target]['center_ras_mm'])
        for row in rows:
            row[target]['displacement_ras_mm']=(np.array(row[target]['center_ras_mm'])-c0).tolist()
    result={'schema_version':1,'case_id':manifest['case_id'],'sequence_id':manifest['sequence_id'],'phase_count':len(rows),
            'mask_method':'TotalSegmentator v2 total/heart and union of four heartchambers_highres chambers without great vessels',
            'coordinate_system':'NIfTI RAS+ mm: x right, y anterior, z superior',
            'rfbr_frontal_mapping':'(delta_x_RAS, delta_z_RAS); plane directions only, no subject registration',
            'reference_status':'automatic exploratory masks; independent centre displacement accuracy not established',
            'source_conversion_sha256':digest(sequence/'conversion_manifest.json'),'source_segmentation_status_sha256':digest(status_path),
            'rows':rows}
    output.mkdir(parents=True)
    write_json(output/'hodographs.json',result)
    draw(result,output)
    return result

def draw(d,out):
    rows=d['rows'];phase=np.array([r['phase_percent'] for r in rows]);cmap=plt.get_cmap('viridis');norm=plt.Normalize(0,95)
    whole=np.array([r['whole_heart']['displacement_ras_mm'] for r in rows]);blood=np.array([r['blood_union']['displacement_ras_mm'] for r in rows])
    fig,axes=plt.subplots(2,2,figsize=(11,9),constrained_layout=True)
    specs=[('Целое сердце · фронтальная плоскость',whole,0,2,'Вправо, мм','К голове, мм'),
           ('Кровь четырёх камер · фронтальная плоскость',blood,0,2,'Вправо, мм','К голове, мм'),
           ('Кровь четырёх камер · поперечная плоскость',blood,0,1,'Вправо, мм','Кпереди, мм'),
           ('Кровь четырёх камер · сагиттальная плоскость',blood,1,2,'Кпереди, мм','К голове, мм')]
    for ax,(title,v,i,j,xlab,ylab) in zip(axes.flat,specs):
        ax.plot(v[:,i],v[:,j],color='#777777',lw=1.2,zorder=1)
        ax.scatter(v[:,i],v[:,j],c=phase,cmap=cmap,norm=norm,s=50,edgecolor='white',lw=.5,zorder=2)
        ax.scatter(v[0,i],v[0,j],marker='s',s=100,c='black',zorder=3)
        ax.axhline(0,color='#cccccc',lw=.7);ax.axvline(0,color='#cccccc',lw=.7)
        ax.set(title=title,xlabel=xlab,ylabel=ylab);ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.18)
        for k in [5,10,15]:ax.annotate(f'{int(phase[k])}%',(v[k,i],v[k,j]),xytext=(4,4),textcoords='offset points',fontsize=8)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=axes.ravel().tolist(),label='Фаза реконструкции, %')
    fig.savefig(out/'hodographs.png',dpi=170,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax,j,label in zip(axes.flat[:3],range(3),['Вправо','Кпереди','К голове']):
        ax.plot(phase,whole[:,j],'-o',ms=4,label='Целое сердце');ax.plot(phase,blood[:,j],'-o',ms=4,label='Кровь четырёх камер')
        ax.set(xlabel='Фаза реконструкции, %',ylabel=f'Δ{label.lower()}, мм',title=f'Смещение центра: {label.lower()}');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    for target,label in [('whole_heart','Целое сердце'),('blood_union','Кровь четырёх камер')]:
        axes.flat[3].plot(phase,[r[target]['volume_ml'] for r in rows],'-o',ms=4,label=label)
    axes.flat[3].set(xlabel='Фаза реконструкции, %',ylabel='Объём маски, мл',title='Объёмы двух объектов');axes.flat[3].grid(alpha=.2)
    fig.savefig(out/'components.png',dpi=170,bbox_inches='tight');plt.close(fig)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sequence',required=True,type=Path);ap.add_argument('--masks',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    a=ap.parse_args();d=compute(a.sequence,a.masks,a.output)
    print('completed',d['case_id'],d['sequence_id'],d['phase_count'])
if __name__=='__main__':main()
