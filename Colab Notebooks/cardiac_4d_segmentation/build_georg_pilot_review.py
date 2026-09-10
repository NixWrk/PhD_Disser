"""Review the Georg pilot: local CT, original/proposed contours, and volume changes."""
from __future__ import annotations
import argparse,csv,hashlib,html,json
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy import ndimage
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CHAMBERS={'heart_ventricle_left':'ЛЖ','heart_ventricle_right':'ПЖ'}


def canonical(path):return nib.as_closest_canonical(nib.load(str(path)))


def contour(mask,color):
    edge=mask & ~ndimage.binary_erosion(mask);out=np.zeros((*mask.shape,4),dtype=np.uint8);out[edge]=[*color,255];return out


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def build(derived_root,pilot_root,output_dir):
    pilot_root=Path(pilot_root);output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((pilot_root/'pilot_manifest.json').read_text(encoding='utf-8'))
    rows=[];links=[]
    for phase in manifest['phases_processed']:
        pid=phase['phase_id'];ctimg=canonical(Path(derived_root)/'georg/phases'/f'{pid}.nii.gz');ct=np.asarray(ctimg.dataobj,dtype=np.float32)
        spacing=np.linalg.norm(ctimg.affine[:3,:3],axis=0)
        for candidate in phase['candidates']:
            ch=candidate['chamber'];target=output_dir/f'{pid}_{ch}';target.mkdir(exist_ok=True)
            rawimg=canonical(Path(derived_root)/'georg/automatic_masks_v2'/pid/'heartchambers_highres'/f'{ch}.nii.gz')
            propimg=canonical(pilot_root/candidate['candidate_file'])
            if rawimg.shape!=ctimg.shape or not np.allclose(rawimg.affine,ctimg.affine) or not np.allclose(propimg.affine,ctimg.affine):raise ValueError('Review grid mismatch')
            raw=np.asarray(rawimg.dataobj)>0;prop=np.asarray(propimg.dataobj)>0;union=raw|prop;changed=raw^prop
            xyz=np.column_stack(np.nonzero(union));lo=np.maximum(xyz.min(0)-15,0);hi=np.minimum(xyz.max(0)+16,ct.shape)
            centre=np.rint(np.column_stack(np.nonzero(raw)).mean(0)).astype(int)
            gray=np.uint8(np.clip((ct+25)/250,0,1)*255)
            # Native canonical sections; no image denoising or mask smoothing.
            def section(arr,axis,index):
                sl=[slice(int(lo[d]),int(hi[d])) for d in range(3)];sl[axis]=int(index)
                return arr[tuple(sl)].T[::-1,::-1]
            def draw_panel(ax,axis,index,mode):
                rest=[d for d in range(3) if d!=axis];aspect=spacing[rest[1]]/spacing[rest[0]]
                ax.imshow(section(gray,axis,index),cmap='gray',vmin=0,vmax=255,aspect=aspect)
                if mode in ('raw','both'):ax.imshow(contour(section(raw,axis,index),(255,178,45)),aspect=aspect,interpolation='nearest')
                if mode in ('proposal','both'):ax.imshow(contour(section(prop,axis,index),(40,205,250)),aspect=aspect,interpolation='nearest')
                ax.set_title(f'{["Sagittal","Coronal","Axial"][axis]} {index} | {mode}',fontsize=9);ax.axis('off')
            for name,indices in [('centres',centre),('largest_changes',np.array([np.argmax(changed.sum(axis=tuple(d for d in range(3) if d!=a))) for a in range(3)]))]:
                if name=='largest_changes' and not changed.any():continue
                fig,axes=plt.subplots(3,3,figsize=(12,11))
                for r,axis in enumerate((2,1,0)):
                    for c,mode in enumerate(('ct','raw','proposal')):draw_panel(axes[r,c],axis,int(indices[axis]),mode)
                fig.suptitle(f'Georg {pid} {CHAMBERS[ch]} | L100/W250 HU | raw orange; proposal cyan | NOT ACCEPTED')
                fig.tight_layout();fig.savefig(target/f'{name}.png',dpi=135);plt.close(fig)
            # Systematic axial sample for visual review; full-resolution slice viewer below.
            zvalues=np.unique(np.rint(np.linspace(lo[2],hi[2]-1,12)).astype(int))
            fig,axes=plt.subplots(3,4,figsize=(14,12))
            for ax,z in zip(axes.flat,zvalues):draw_panel(ax,2,z,'both')
            fig.suptitle(f'Georg {pid} {CHAMBERS[ch]} | axial sweep | orange raw / cyan proposal | L100 W250 HU')
            fig.tight_layout();fig.savefig(target/'axial_sweep.png',dpi=130);plt.close(fig)
            for z in range(int(lo[2]),int(hi[2])):
                Image.fromarray(section(gray,2,z)).save(target/f'ct_{z:03d}.png')
                Image.fromarray(contour(section(raw,2,z),(255,178,45))).save(target/f'raw_{z:03d}.png')
                Image.fromarray(contour(section(prop,2,z),(40,205,250))).save(target/f'proposal_{z:03d}.png')
            meta={'subject':'georg','phase_id':pid,'chamber':ch,'review_status':'pending_visual_review_not_anatomical_acceptance','grid':'canonical_RAS','bounds':[lo.tolist(),hi.tolist()],
                  'ct_sha256':phase['ct_sha256'],'raw_sha256':candidate['raw_sha256'],'proposal_sha256':candidate['candidate_sha256'],'source_manifest_sha256':digest(pilot_root/'pilot_manifest.json'),'window_hu':{'level':100,'width':250},'systematic_axial_sample_indices':zvalues.tolist(),
                  'png_sha256':{p.name:digest(p) for p in target.glob('*.png') if p.name in ('centres.png','largest_changes.png','axial_sweep.png')}}
            (target/'review.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
            page='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Пилот Георга: сравнение масок</title><style>body{max-width:1100px;margin:auto;padding:24px;font:16px system-ui;line-height:1.5;background:#f4f6f9}label{margin-right:20px}.stack{position:relative;max-width:750px;margin:20px auto}.stack img{display:block;width:100%}.overlay{position:absolute;top:0;left:0}input[type=range]{width:65%}a{color:#1653a1}</style>
<p><a href="../index.html">Все камеры и фазы</a></p><h1>__TITLE__</h1><p>Исходный контур — оранжевый. Предложение nnInteractive — голубое. Предложение не принято как исправленная анатомия. Окно 100/250 HU; изображения не сглажены.</p>
<label><input id="rawShow" type="checkbox" checked> Исходная маска</label><label><input id="propShow" type="checkbox" checked> Предложение</label><p><input id="slice" type="range" min="__MIN__" max="__MAX__" value="__MID__"> <span id="z"></span></p>
<div class="stack"><img id="ct" alt="КТ"><img class="overlay" id="raw" alt="Исходная маска"><img class="overlay" id="prop" alt="Предложение"></div><p>Ориентация: R слева, L справа; A сверху, P снизу. Индекс относится к сетке RAS. Изменения границ с предсердиями и сосудами требуют отдельного принятия клапанных плоскостей.</p>
<p><a href="centres.png">Ортогональные сечения</a> · <a href="largest_changes.png">Сечения наибольших изменений</a> · <a href="axial_sweep.png">Обзор аксиальных срезов</a> · <a href="review.json">Происхождение</a></p>
<script>const s=document.getElementById('slice');function update(){let n=String(s.value).padStart(3,'0');document.getElementById('ct').src='ct_'+n+'.png';document.getElementById('raw').src='raw_'+n+'.png';document.getElementById('prop').src='proposal_'+n+'.png';document.getElementById('raw').style.visibility=document.getElementById('rawShow').checked?'visible':'hidden';document.getElementById('prop').style.visibility=document.getElementById('propShow').checked?'visible':'hidden';document.getElementById('z').textContent='Срез '+s.value;}s.oninput=update;document.getElementById('rawShow').onchange=update;document.getElementById('propShow').onchange=update;document.addEventListener('keydown',e=>{if(['ArrowLeft','ArrowRight'].includes(e.key)){e.preventDefault();s.value=Number(s.value)+(e.key==='ArrowRight'?1:-1);update();}});update();</script></html>'''
            for key,value in {'TITLE':f'Георг · {pid} · {CHAMBERS[ch]}','MIN':lo[2],'MAX':hi[2]-1,'MID':centre[2]}.items():page=page.replace('__'+key+'__',str(value))
            if not changed.any():page=page.replace('<a href="largest_changes.png">Сечения наибольших изменений</a> · ','')
            (target/'index.html').write_text(page,encoding='utf-8')
            rows.append({**candidate['metrics'],'review_page':target.name+'/index.html'})
            links.append(f'<li><a href="{target.name}/index.html">{pid} · {CHAMBERS[ch]}</a></li>')
            print('Rendered review',pid,ch,flush=True)
    columns=[('phase_id','Фаза'),('chamber','Камера'),('raw_ml','Исходный объём, мл'),('candidate_ml','Предложение, мл'),('delta_ml','Изменение, мл'),('added_ml','Добавлено, мл'),('removed_ml','Удалено, мл'),('components_26','Компоненты')]
    def cell(v):return f'{v:.2f}'.replace('.',',') if isinstance(v,float) else html.escape(str(v))
    table='<table><tr>'+''.join('<th>'+label+'</th>' for _,label in columns)+'</tr>'+''.join('<tr>'+''.join('<td>'+cell(CHAMBERS[r[k]] if k=='chamber' else r[k])+'</td>' for k,_ in columns)+'</tr>' for r in rows)+'</table>'
    body='<!doctype html><html lang="ru"><meta charset="utf-8"><title>Пилот масок желудочков Георга</title><style>body{max-width:1300px;margin:30px auto;font:16px system-ui;line-height:1.5}td,th{border:1px solid #bbb;padding:8px}table{border-collapse:collapse}a{color:#1653a1}</style><h1>Георг: проверка масок желудочков</h1><p>Четыре фазы, восемь отдельных запусков nnInteractive с исходной маской TotalSegmentator. Независимый ручной референс отсутствует; изменения объёма не являются измерением точности. Исходные маски и расчёты 21.03 не заменены.</p>'+table+'<h2>КТ и контуры до/после</h2><ul>'+''.join(links)+'</ul><p>Оранжевый контур — исходный, голубой — предложение. Выбор кадров выполнен по прежним экстремумам масок. Короткий и длинный R–R не объединяются; механические ED/ES пока не приняты.</p></html>'
    (output_dir/'index.html').write_text(body,encoding='utf-8')
    (output_dir/'review_summary.json').write_text(json.dumps({'rows':rows,'source_manifest_sha256':digest(pilot_root/'pilot_manifest.json'),'status':'proposals_pending_review'},ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--derived-root',type=Path,required=True);p.add_argument('--pilot-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();build(a.derived_root,a.pilot_root,a.output_dir)
