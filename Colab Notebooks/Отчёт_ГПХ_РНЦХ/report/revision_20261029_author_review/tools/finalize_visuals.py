"""Readable placement copies from frozen figures; no scientific recomputation.

Run after build_visuals.py, prepare_photos.py and render_visuals.cjs.
Photograph bytes and the author's report_overrides.json are never written.
Raster plot interiors are pasted without resampling and checked pixel for pixel.
"""
from pathlib import Path
import hashlib, json, re, shutil, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding='utf-8')
REV = Path(__file__).resolve().parents[1]
ROOT = REV.parents[3]
OUT = REV/'figures'
CACHE = REV.parent/'revision_20261028/figures'
MP = OUT/'manifest.json'
FONT = 'C:/Windows/Fonts/arial.ttf'
sha = lambda b: hashlib.sha256(b).hexdigest()
rel = lambda p: p.relative_to(ROOT).as_posix()
m = json.loads(MP.read_text(encoding='utf-8'))
overrides = OUT/'report_overrides.json'
override_bytes = overrides.read_bytes() if overrides.exists() else None

def text(im, xy, value, size=36, anchor='mm', fill='black'):
    ImageDraw.Draw(im).text(xy, value, font=ImageFont.truetype(FONT,size),
                            fill=fill, anchor=anchor)

def vertical(im, xy, value, size=36):
    font=ImageFont.truetype(FONT,size)
    box=font.getbbox(value)
    strip=Image.new('RGBA',(box[2]+10,box[3]-box[1]+10))
    ImageDraw.Draw(strip).text((5,5-box[1]),value,font=font,fill='black')
    strip=strip.rotate(90,expand=True)
    im.paste(strip,(round(xy[0]-strip.width/2),round(xy[1]-strip.height/2)),strip)

def preserved(src, dst, source_box, target_xy):
    tile=src.crop(source_box)
    dst.paste(tile,target_xy)
    target_box=(*target_xy,target_xy[0]+tile.width,target_xy[1]+tile.height)
    assert np.array_equal(np.asarray(tile),np.asarray(dst.crop(target_box)))
    return dict(source_xyxy=source_box,target_xyxy=target_box,
                rgb_sha256=sha(tile.tobytes()),pixels_exactly_equal=True)

def metadata(p):
    with Image.open(p) as im:w,h=im.size
    return dict(path=rel(p),width=w,height=h,unit='px',sha256=sha(p.read_bytes()))

def update_path(target_path, **fields):
    for e in [*m['figures'],*m.get('photos',[]),*m.get('html_exports',[])]:
        for item in [e,*e.get('assets',[])]:
            if item.get('path')==rel(target_path):item.update(fields)

# CT: only labels in white margins change. The four complete image panels,
# orientation labels and every tissue/heart mask pixel remain exactly equal.
ctsrc=REV.parent/'revision_20261029_contract/figures/html_exports/01_n_ct_tissue_masks.png'
src=Image.open(ctsrc).convert('RGB'); im=src.copy(); d=ImageDraw.Draw(im)
rects=[(140,128,829,813),(1075,128,1764,813),
       (140,937,829,1622),(1075,937,1764,1622)]
for box in [(0,0,1864,127),(0,128,135,813),(932,128,1070,813),
            (0,822,1864,937),(0,937,135,1622),(932,937,1070,1622),
            (0,1631,1864,1707)]:
    d.rectangle((box[0],box[1],box[2]-1,box[3]-1),fill='white')
text(im,(932,24),'КТ участника У1, аксиальный срез z = −154,86 мм',38)
text(im,(932,67),'Ближайший срез к центроиду целого сердца (z = −154,99 мм); RAS',33)
for x,title in [(484,'КТ: анатомическое сечение'),(1419,'Лёгкие')]:text(im,(x,109),title,34)
for x,title in [(484,'Жир'),(1419,'Кости')]:text(im,(x,916),title,34)
xs=[[153,240,327,414,502,589,676,763],
    [1088,1175,1262,1349,1436,1523,1611,1698]]
ys=[[169,256,344,431,518,605,692,779],
    [978,1065,1153,1240,1327,1414,1501,1588]]
for row in range(2):
    for col in range(2):
        for x,label in zip(xs[col],['200','150','100','50','0','−50','−100','−150']):
            text(im,(x,844+809*row),label,30)
        for y,label in zip(ys[row],['150','100','50','0','−50','−100','−150','−200']):
            text(im,(130+935*col,y),label,30,'rm')
        text(im,(484+935*col,874+809*row),'X RAS, мм (+ вправо пациента)',31)
        vertical(im,(31+935*col,470+809*row),'Y RAS, мм (+ кпереди)',31)
for box in rects:assert np.array_equal(np.asarray(src.crop(box)),np.asarray(im.crop(box)))
cttrans=dict(method='larger labels in white margins only',source_path=rel(ctsrc),
    source_sha256=sha(ctsrc.read_bytes()),scientific_panels_pixel_identical=True,
    preserved_panel_rectangles_xyxy=rects,font_px=dict(ticks=30,axes=31,panel_titles=34))
for p in [OUT/'V03_ct_segmentation.png',OUT/'html_exports/01_n_ct_tissue_masks.png']:
    im.save(p,dpi=(300,300));update_path(p,**metadata(p),transformation=cttrans)

# Six numerical-accuracy panels: keep all plot interiors, shorten only repeated
# axis labels, share the horizontal scale and enlarge typography.
step_src=CACHE/'reader_montage/ch07_04_volume_step_quadrature.png'
src=Image.open(step_src).convert('RGB');im=Image.new('RGB',(2160,1500),'white')
text(im,(1080,29),'Шаг объёмной производной и квадратура',45)
proof=[]
top=[203,703,1204];newtop=[120,530,940]
tick_pos=[[[269,340,411,482,554],[249,313,376,440,504]],
          [[743,822,900,979,1058],[767,842,917,991]],
          [[1210,1274,1339,1404,1469,1534],[1257,1314,1372,1429,1486,1544]]]
tick_text=[[
    ['0.195','0.190','0.185','0.180','0.175'],['1.34','1.33','1.32','1.31','1.30']],
    [['−0.136','−0.138','−0.140','−0.142','−0.144'],['0.795','0.790','0.785','0.780']],
    [['0.032','0.030','0.028','0.026','0.024','0.022'],['0.930','0.925','0.920','0.915','0.910','0.905']]]
for row,name in enumerate(['STL','G1','L1']):
    for col in range(2):
        x0=171+1056*col;y0=top[row];nt=newtop[row]
        box=(x0-8,y0,x0+907,y0+365)
        proof.append(preserved(src,im,box,(x0-8,nt)))
        title=f'{name} · ρ сердца = 4 Ом·м' if col==0 else f'{name} · условные ρ из подгонки STL'
        text(im,(x0+453,nt-30),title,38)
        for yp,label in zip(tick_pos[row][col],tick_text[row][col]):
            text(im,(x0-14,nt+yp-y0),label,36,'rm')
        if row==2:
            for xp,label in zip([213+1056*col,522+1056*col,1036+1056*col],['1.0','2.5','5.0']):
                text(im,(xp,nt+388),label,36)
vertical(im,(20,720),'∂Z/∂V, мОм/мл',39)
text(im,(1080,1369),'Полушаг интегрированного объёма, % от V0',41)
# These are the original legend symbols, reused without resampling.
for sb,xy,label in [
    ((675,1676,736,1710),(330,1403),'Исходный, 100 мм'),
    ((675,1713,736,1747),(330,1445),'95 мм'),
    ((675,1750,736,1784),(735,1445),'Поворот на одной руке'),
    ((1126,1676,1160,1710),(1240,1403),'256 точек; линии — 64'),
    ((1126,1713,1160,1747),(1240,1445),'1024 точки; только L1')]:
    im.paste(src.crop(sb),xy);text(im,(xy[0]+75,xy[1]+17),label,37,'lm')
p=OUT/'V13_models_step_quadrature.png';im.save(p,dpi=(300,300))
update_path(p,**metadata(p),transformation=dict(method='reframed frozen panels; larger shared labels',
    source_path=rel(step_src),source_sha256=sha(step_src.read_bytes()),
    scientific_panels=proof,resampling=False,numerical_data_recomputed=False),
    caption='Объёмная чувствительность STL, G1 и L1 при изменении объёмного шага и числа квадратурных точек на одной сетке. Столбцы соответствуют двум материальным опорам; цвет — трём альтернативным монтажам. Горизонтальная шкала общая для всех панелей; обозначение q относится к числу квадратурных точек.')

# Electrical factors: retain the two nonredundant panels and every plotted pixel.
# The redundant RMSE bar chart remains in the complete supplementary original.
elec_src=CACHE/'reader/07_02_electrical_approximation.png'
src=Image.open(elec_src).convert('RGB');im=Image.new('RGB',(1850,1120),'white');proof=[]
text(im,(925,34),'Условные FEM-расчёты на КТ участника У1',46)
for sx,tx in [(146,185),(1829,1110)]:
    proof.append(preserved(src,im,(sx-8,119,sx+697,727),(tx-8,160)))
text(im,(533,111),'Одинаковое ΔV; ρ сердца = 4 Ом·м',40)
text(im,(1458,111),'Три сценария; одна STL и монтаж',40)
for y,label in zip([147,214,282,349,417,485,552,620,688],['10.0','7.5','5.0','2.5','0.0','−2.5','−5.0','−7.5','−10.0']):
    text(im,(169,y+41),label,38,'rm')
for y,label in zip([143,222,301,380,459,538,617,696],['1.25','1.00','0.75','0.50','0.25','0.00','−0.25','−0.50']):
    text(im,(1094,y+41),label,38,'rm')
for x,label in zip([167,276,385,494,604,713,822],['−30','−20','−10','0','10','20','30']):text(im,(x+39,797),label,38)
for x,label in zip([1974,2128,2281,2435],['5','10','15','20']):text(im,(x-719,797),label,38)
vertical(im,(30,463),'Изменение ТТРКГ, мОм',40)
vertical(im,(930,463),'dZ/dV, мОм/мл',40)
text(im,(533,855),'Изменение заданного объёма, мл',40)
text(im,(1458,855),'Заданное ρ целого сердца, Ом·м',40)
draw=ImageDraw.Draw(im)
colors=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2']
for i,(name,color) in enumerate(zip(['STL','G1','G2','G3a','G3b','L1','L2'],colors)):
    x=90+(i%4)*440;y=955+(i//4)*74
    draw.line((x,y,x+64,y),fill=color,width=5);draw.ellipse((x+25,y-8,x+41,y+8),fill=color)
    text(im,(x+80,y),name,42,'lm')
p=OUT/'V17_models_electrical_factors.png';im.save(p,dpi=(300,300))
update_path(p,**metadata(p),caption='Приращения ТТРКГ при одинаковых объёмных пробах разных форм сердца и отдельные сценарии эффективного сердечного сопротивления. Слева — объёмные пробы при одинаковом ρ; справа — сохранённые значения объёмной чувствительности для трёх сценариев ρ одной исходной STL и одного монтажа. Точки сценариев не задают непрерывного физиологического закона.',
    transformation=dict(method='two nonredundant frozen panels with larger margin labels',
    source_path=rel(elec_src),source_sha256=sha(elec_src.read_bytes()),
    scientific_panels=proof,resampling=False,numerical_data_recomputed=False,
    omitted_panel='RMSE bar chart; retained in complete cached original'))

# Remove empty 3-D layout space while retaining complete rendered anatomy and
# coordinate axes, without resampling the scene or changing geometry.
for filename,box,xy,title,legend in [
    ('V04_anatomical_surfaces.png',(1030,390,1990,1300),(600,170),
     'Сегментированные поверхности: лёгкие, сердце и кости',
     [('Лёгкие','#71cbdc'),('Сердце','#c23f60'),('Кости','#ded4bd')]),
    ('V04_fem_surface_contacts.png',(900,460,2090,1270),(305,150),
     'Расчётная поверхность У2: боковая сборка и ТТРКГ',
     [('Боковая сборка 140 мм','#ef553b'),('1×4: токовые','#b92f34'),('1×4: измерительные','#3085ab')])]:
    p=OUT/filename;scene=OUT/(p.stem+'_scene.png')
    with Image.open(p) as check:
        if check.size==(3060,1700):shutil.copyfile(p,scene)
    src=Image.open(scene).convert('RGB');im=Image.new('RGB',(1800,1150),'white')
    proof=preserved(src,im,box,xy)
    text(im,(900,48),title,40)
    draw=ImageDraw.Draw(im)
    for i,(label,color) in enumerate(legend):
        if filename.startswith('V04_anatomical'):
            lx,ly=55,490+i*88
        else:lx,ly=[(80,1017),(80,1091),(805,1091)][i]
        draw.rectangle((lx,ly-12,lx+24,ly+12),fill=color)
        text(im,(lx+45,ly),label,38,'lm')
    im.save(p,dpi=(300,300))
    # Preserve the frozen Plotly provenance alongside this placement operation.
    for entry in m['figures']:
        for item in [entry,*entry.get('assets',[])]:
            if item.get('path')==rel(p):
                prior=item.setdefault('transformation',{})
                prior['placement_reframing']={'scene_path':rel(scene),'scene_sha256':sha(scene.read_bytes()),
                    'preserved_scene':proof,'resampling':False,'legend_labels_redrawn':True}
    update_path(p,**metadata(p))

# A readable ventricular counterexample at <=11.5 cm height. Keep the full four
# chamber image as a supplementary asset; no point, axis or legend is cut above.
e=next(e for e in m['figures'] if e['id']=='V22')
full=OUT/'V22_wrong_extrema_counterexample.png'
p=OUT/'V22_ventricular_phase_reversal.png'
src=Image.open(full).convert('RGB');cropped=src.crop((0,0,src.width,1100));cropped.save(p,dpi=(300,300))
oldpath=e['path'];newpath=rel(p)
e['assets']=[a for a in e.get('assets',[]) if a.get('alias')!='cardiac_phase_reversal']
e['assets'].append(dict(alias='cardiac_phase_reversal',**metadata(p),
    source={'cached_png':oldpath,'sha256':sha(full.read_bytes()),'crop_xyxy':[0,0,src.width,1100]},
    caption='Объёмы левого и правого желудочков У1 на отложенных фазах и контрпример близкого размаха при ошибочной фазовой последовательности. Известная исходная фаза составляет 14% R–R; временной зазор — 140 мс. Отрезки соединяют прогнозы исключённых блоков с разными калибровками и не задают траекторию одной фиксированной модели.',
    restrictions=['Показаны только желудочковые панели; полное представление четырёх камер сохранено отдельно.'],
    transformation={'method':'exact crop of top two complete panels','scientific_pixels_unchanged':True}))
e['alias_paths']['cardiac_phase_reversal']=newpath

# Retain the builder-compatible assets/alias_paths schema, remove stale repeated
# asset records from iterative rendering, refresh every effective file hash.
for e in m['figures']:
    e['aliases']=list(dict.fromkeys(e.get('aliases',[])))
    unique={}
    for a in e.get('assets',[]):
        if a.get('alias')=='ct_surface_control' and not a.get('path','').endswith('V04_geometry_preparation.png'):continue
        unique[a['alias']]=a
    e['assets']=list(unique.values())
    for item in [e,*e['assets']]:
        if not item.get('path'):continue
        p=ROOT/item['path'];item.update(metadata(p))
        width=16.5 if e['id']=='V03' else min(16.5,11.5*item['width']/item['height'])
        item.update(width_cm=width,height_cm=width*item['height']/item['width'])
        item['status']='prepared'
    if e['id']=='V03':e['height_exception_cm']=18

aliases={}
for e in [*m['figures'],*m.get('photos',[])]:
    for alias in e.get('aliases',[]):aliases[alias]=e.get('alias_paths',{}).get(alias,e['path'])
    for a in e.get('assets',[]):aliases[a['alias']]=a['path']
main=[]
for p in sorted((REV/'drafts').glob('[0-9][0-9]_*.md')):
    for alias in re.findall(r'\[\[FIG:([^\]]+)\]\]',p.read_text(encoding='utf-8')):main.append(alias)
assert len(main)==len(set(main))==19
for alias in main:assert alias in aliases and (ROOT/aliases[alias]).exists(),alias
assert aliases['ct_model_preparation']!=aliases['models_ct_fem_preparation']
assert aliases['models_anatomy_contacts'] not in [aliases['ct_model_preparation'],aliases['models_ct_fem_preparation']]
assert aliases['processing_cycles'].endswith('V07_cycle_processing.png')
for e in m.get('photos',[]):
    assert sha((ROOT/e['path']).read_bytes())==e['source']['sha256']
    assert e['byte_identical_to_source'] and e['pixels_unchanged']
if overrides.exists():assert overrides.read_bytes()==override_bytes
m['main_aliases']=main
m['optional_main_aliases']=[]
m['main_figure_count']=19
m['V06_placement']='Retained as a checked saved HTML export; excluded from main text by the author to keep 19 figures.'
m['author_confirmations_sha256']=sha((REV/'AUTHOR_CONFIRMATIONS.md').read_bytes())
m['alias_policy']='Use alias_paths and assets for semantic markers; paths are relative to repository root. Author report_overrides.json is loaded last and is never modified by these tools.'
m['readability_changes']={
    'V03':'larger labels; four scientific image panels pixel-identical; title says КТ участника У1',
    'models_electrical_factors':'two nonredundant complete panels; larger labels; original complete figure retained',
    'models_step_quadrature':'all six numerical panels preserved without resampling; shared horizontal labels',
    'cardiac_phase_reversal':'two complete ventricular panels for main text; complete four-chamber image retained'}
m['assembly_review']={
    'pdf':rel(REV/'render/report.pdf'),
    'pdf_sha256':sha((REV/'render/report.pdf').read_bytes()),
    'reviewed_figure_pages':[21,24,25,27,28,31,36,38,45,52,53,55,57,59,63,64,67,74],
    'reviewed_figures':19,
    'caption_content_mismatches_found':0,
    'photos':'P01/P03/P07: face excluded; target contacts and belt visible; preparation captions match.',
    'processing_cycles':'Ordinary TT 2x2 and lateral 140 mm, U1 record 001; five QRS candidates delimit four cycles.',
    'rebuild_required_for_updated_assets':['ct_segmentation','ct_model_preparation','models_ct_fem_preparation','models_step_quadrature','models_electrical_factors','cardiac_phase_reversal'],
    'remaining_readability_notes':['Small original type remains in models_geometry_comparison, models_finite_compensation and tissue_pulse_projection. No clipped curves or mismatched captions found in the inspected intermediate PDF.'],
    'scope':'Only figure/caption pages inspected; this is not a complete textual or final-pagination review.'}
MP.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Prepared readable placement copies. All 19 main markers resolve; 3 CT images distinct; 7 photos byte-identical; overrides unchanged.')
