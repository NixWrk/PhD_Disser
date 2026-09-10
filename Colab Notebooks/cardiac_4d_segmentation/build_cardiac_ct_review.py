"""Local CT previews at observed extrema. Never writes or edits anatomical masks."""
import html
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from cardiac_function_analysis import CHAMBERS

COLORS = [(25,105,240),(240,65,55),(40,200,235),(255,175,35)]
PAGE = '''<!doctype html><html lang="ru"><meta charset="utf-8"><title>КТ и маски</title>
<style>body{max-width:950px;margin:auto;padding:22px;font:16px system-ui;background:#f4f6f9;color:#172238}label{margin-right:18px}.stack{position:relative;max-width:700px;margin:20px auto;background:#000}.stack img{display:block;width:100%}#mask{position:absolute;top:0;left:0}input[type=range]{width:50%}a{color:#1653a1}</style>
<p><a href="../index.html">Все выбранные кадры</a></p><h1>__TITLE__</h1>
<p>КТ и контуры исходных автоматических масок. ЛЖ — синий, ПЖ — красный, ЛП — голубой, ПП — оранжевый. Маски не исправлены и не приняты вручную.</p>
<p>Справа на экране — левая сторона пациента (L); слева — R; сверху — передняя сторона (A), снизу — задняя (P). Изображение обрезано вокруг масок с запасом 12 вокселей.</p>
<label>Окно <select id="window"><option value="soft">60 / 160 HU</option><option value="wide">100 / 700 HU</option></select></label><label><input id="show" type="checkbox" checked> Контуры</label><label>Непрозрачность <input id="alpha" type="range" min="0" max="100" value="100"></label>
<p><input id="slice" type="range" min="__MIN__" max="__MAX__" value="__MID__"> <span id="position"></span></p>
<div class="stack"><img id="ct" alt="Срез КТ"><img id="mask" alt="Контуры камер"></div>
<p>Окно задано как уровень / ширина в HU; доступны узкий и широкий варианты. Стрелки ← → переключают срезы. Номер относится к сетке RAS после перестановки и отражения осей; геометрия сохраняется. Для принятия границ нужна проверка всех фаз и клапанных плоскостей в редакторе многоплоскостных реконструкций.</p>
<p><a href="overview.png">Три ортогональных сечения</a> · <a href="review_manifest.json">Происхождение и настройки</a></p>
<script>const slider=document.getElementById('slice'),mask=document.getElementById('mask'),ct=document.getElementById('ct');function update(){let z=Number(slider.value),n=String(z).padStart(3,'0');ct.src=(document.getElementById('window').value==='soft'?'ct_soft_':'ct_')+n+'.png';mask.src='mask_'+n+'.png';mask.style.opacity=document.getElementById('show').checked?document.getElementById('alpha').value/100:0;document.getElementById('position').textContent='Срез '+z+'; S = '+(__Z0__+z*__DZ__).toFixed(2)+' мм';}document.getElementById('window').onchange=update;slider.oninput=update;document.getElementById('show').onchange=update;document.getElementById('alpha').oninput=update;document.addEventListener('keydown',e=>{if(['ArrowLeft','ArrowRight'].includes(e.key)){e.preventDefault();slider.value=Number(slider.value)+(e.key==='ArrowRight'?1:-1);update();}});update();</script></html>'''


def outline(labels):
    out=np.zeros((*labels.shape,4),dtype=np.uint8)
    for code,color in enumerate(COLORS,1):
        mask=labels==code;edge=mask & ~binary_erosion(mask)
        out[edge]=[*color,255]
    return out


def build_review(derived_root, output_dir, results):
    from build_cardiac_function_report import sha256
    root=Path(output_dir)/'function_analysis/ct_review';root.mkdir(parents=True,exist_ok=True)
    selected={};pages=[]
    for row in results:
        if row.get('scenario')=='raw' and row['estimator']=='observed_global_range':
            for role in ('ed','es'):
                selected.setdefault((row['subject'],row[role+'_phase_id']),[]).append(f"{CHAMBERS[row['chamber']]}: {'максимум' if role=='ed' else 'минимум'}")
    for (subject,pid),roles in sorted(selected.items()):
        target=root/f'{subject}_{pid}';target.mkdir(exist_ok=True)
        ct_path=Path(derived_root)/subject/'phases'/f'{pid}.nii.gz'
        masks={ch:Path(derived_root)/subject/'automatic_masks_v2'/pid/'heartchambers_highres'/f'{ch}.nii.gz' for ch in CHAMBERS}
        fingerprint={'builder_sha256':sha256(__file__),'ct_sha256':sha256(ct_path),'mask_sha256':{ch:sha256(p) for ch,p in masks.items()}}
        meta_path=target/'review_manifest.json';cached=json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
        if cached.get('inputs')==fingerprint and (target/'index.html').exists():
            pages.append((subject,pid,roles));continue
        original=nib.load(str(ct_path));ct=nib.as_closest_canonical(original)
        if not np.allclose(ct.affine[:3,:3],np.diag(np.diag(ct.affine[:3,:3])),atol=1e-5):raise ValueError('Oblique CT requires a resliced viewer')
        data=np.asarray(ct.dataobj,dtype=np.float32);labels=np.zeros(ct.shape,dtype=np.uint8)
        for code,(ch,path) in enumerate(masks.items(),1):
            image=nib.as_closest_canonical(nib.load(str(path)))
            if image.shape!=ct.shape or not np.allclose(image.affine,ct.affine,atol=1e-5):raise ValueError('CT/mask grid mismatch')
            mask=np.asarray(image.dataobj)>0
            if np.any(mask & (labels>0)):raise ValueError('Overlapping masks require an explicit overlap display')
            labels[mask]=code
        coords=np.column_stack(np.nonzero(labels));lo=np.maximum(coords.min(0)-12,0);hi=np.minimum(coords.max(0)+13,ct.shape)
        slices=list(range(int(lo[2]),int(hi[2])));gray=np.uint8(np.clip((data+250)/700,0,1)*255)
        soft=np.uint8(np.clip((data+20)/160,0,1)*255)
        spacing=np.diag(ct.affine[:3,:3]);width=int(hi[0]-lo[0]);height=int(round((hi[1]-lo[1])*spacing[1]/spacing[0]))
        xy=(slice(lo[0],hi[0]),slice(lo[1],hi[1]))
        for z in slices:
            sl=gray[xy+(z,)].T[::-1,::-1];lab=labels[xy+(z,)].T[::-1,::-1]
            Image.fromarray(sl).resize((width,height),Image.Resampling.BILINEAR).save(target/f'ct_{z:03d}.png')
            Image.fromarray(soft[xy+(z,)].T[::-1,::-1]).resize((width,height),Image.Resampling.BILINEAR).save(target/f'ct_soft_{z:03d}.png')
            Image.fromarray(outline(lab)).resize((width,height),Image.Resampling.NEAREST).save(target/f'mask_{z:03d}.png')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        centre=np.rint(coords.mean(0)).astype(int)
        fig,axes=plt.subplots(3,2,figsize=(10,13),facecolor='white')
        for row,(axis,title) in enumerate([(2,'Axial'),(1,'Coronal'),(0,'Sagittal')]):
            idx=int(centre[axis]);sl=[slice(int(lo[i]),int(hi[i])) for i in range(3)];sl[axis]=idx
            g=soft[tuple(sl)].T[::-1,::-1];lab=labels[tuple(sl)].T[::-1,::-1]
            remaining=[i for i in range(3) if i!=axis];aspect=spacing[remaining[1]]/spacing[remaining[0]]
            for col in range(2):
                axes[row,col].imshow(g,cmap='gray',vmin=0,vmax=255,aspect=aspect)
                if col:axes[row,col].imshow(outline(lab),aspect=aspect,interpolation='nearest')
                axes[row,col].set_title(f'{title}, RAS index {idx}'+(' + raw mask' if col else ''));axes[row,col].axis('off')
        fig.suptitle(f'{subject.upper()} {pid} | L60 / W160 HU | LV blue, RV red, LA cyan, RA orange',fontsize=12)
        fig.tight_layout();fig.savefig(target/'overview.png',dpi=135);plt.close(fig)
        meta={'status':'preview_generated_not_manually_accepted','subject':subject,'phase_id':pid,'selection':roles,'inputs':fingerprint,
              'display_windows_hu':[{'level':60,'width':160},{'level':100,'width':700}],'source_axis_codes':nib.aff2axcodes(original.affine),'display_grid':'closest_canonical_RAS',
              'crop_index_bounds':[lo.tolist(),hi.tolist()],'axial_indices':slices,'affine':ct.affine.tolist(),'mask_edits':0,'manual_validation':False}
        meta_path.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
        content=PAGE
        for key,value in {'TITLE':html.escape(f'{subject.upper()} · {pid} · '+', '.join(roles)),'MIN':slices[0],'MAX':slices[-1],'MID':int(centre[2]),'Z0':ct.affine[2,3],'DZ':spacing[2]}.items():content=content.replace('__'+key+'__',str(value))
        (target/'index.html').write_text(content,encoding='utf-8');pages.append((subject,pid,roles))
        print(f'CT preview: {subject}/{pid}, {len(slices)} axial slices',flush=True)
    links=''.join(f'<li><a href="{s}_{p}/index.html">{s.upper()} · {p}</a> — {html.escape(", ".join(r))}</li>' for s,p,r in pages)
    (root/'index.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><title>КТ: выбранные кадры</title><style>body{max-width:950px;margin:30px auto;font:17px system-ui;line-height:1.6}li{margin:12px 0}</style><h1>КТ в кадрах наблюдаемых экстремумов</h1><p>Исходные автоматические маски четырёх камер. Максимальный и минимальный объёмы не подтверждают механические ED/ES. Ручная разметка не выполнена.</p><ul>'+links+'</ul></html>',encoding='utf-8')
    return len(pages)
