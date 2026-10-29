"""Enlarge type for three frozen scientific figures, without new calculations.

All 4/6/4 complete scientific plot interiors are copied from verified cached
PNGs at their native pixel size. Only the surrounding layout and labels change.
Original inset legends are retained as part of those interiors; a readable
shared legend repeats their meanings outside the data. Pixel identity is checked
after every label is drawn. No Word, draft, audit or photo file is written.
"""
from pathlib import Path
import hashlib, json, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding='utf-8')
REV=Path(__file__).resolve().parents[1]
ROOT=REV.parents[3]
OUT=REV/'figures'
CACHE=REV.parent/'revision_20261028/figures'
FONT='C:/Windows/Fonts/arial.ttf'
MP=OUT/'manifest.json'
sha=lambda b:hashlib.sha256(b).hexdigest()
rel=lambda p:p.relative_to(ROOT).as_posix()
WORD_WIDTH_CM=16.5
POINTS_ACROSS=WORD_WIDTH_CM/2.54*72
manifest=json.loads(MP.read_text(encoding='utf-8'))
protected={p:sha(p.read_bytes()) for p in [OUT/'report_overrides.json',
    *sorted((OUT/'photos').glob('photo*.png'))] if p.exists()}

def text(im,xy,value,size,anchor='mm',fill='black'):
    ImageDraw.Draw(im).text(xy,value,font=ImageFont.truetype(FONT,size),anchor=anchor,fill=fill)

def vertical(im,xy,value,size):
    font=ImageFont.truetype(FONT,size);b=font.getbbox(value)
    strip=Image.new('RGBA',(b[2]+12,b[3]-b[1]+12))
    ImageDraw.Draw(strip).text((6,6-b[1]),value,font=font,fill='black')
    strip=strip.rotate(90,expand=True)
    im.paste(strip,(round(xy[0]-strip.width/2),round(xy[1]-strip.height/2)),strip)

def panel(src,im,box,xy,proof):
    tile=src.crop(box);im.paste(tile,xy)
    proof.append(dict(source_xyxy=list(box),target_xyxy=[*xy,xy[0]+tile.width,xy[1]+tile.height],
                      rgb_sha256=sha(tile.tobytes())))

def legend(im,x,y,label,color,size,span=65):
    d=ImageDraw.Draw(im);d.line((x,y,x+span,y),fill=color,width=5)
    d.ellipse((x+span//2-8,y-8,x+span//2+8,y+8),fill=color)
    text(im,(x+span+18,y),label,size,'lm')

def commit(id,im,src_path,proof,fonts,notes):
    src=Image.open(src_path).convert('RGB')
    # Verify after typography and legends, so accidental covering is detected.
    for p in proof:
        expected=src.crop(p['source_xyxy']);actual=im.crop(p['target_xyxy'])
        assert expected.size==actual.size
        assert np.array_equal(np.asarray(expected),np.asarray(actual)),p
        p['pixels_exactly_equal_after_all_labels']=True
    e=next(f for f in manifest['figures'] if f['id']==id)
    assert sha(src_path.read_bytes())==e['source']['sha256']
    dst=ROOT/e['path'];temporary=dst.with_name(dst.stem+'.writing.png')
    im.save(temporary,dpi=(300,300));os.replace(temporary,dst)
    width,height=im.size
    height_cm=WORD_WIDTH_CM*height/width
    assert height_cm<=11.5
    font_points={k:round(v*POINTS_ACROSS/width,2) for k,v in fonts.items()}
    assert min(font_points.values())>=9.0
    e.update(width=width,height=height,width_cm=WORD_WIDTH_CM,height_cm=height_cm,
             sha256=sha(dst.read_bytes()),status='prepared')
    e['transformation']={
        'method':'reframe complete frozen plot panels; readable shared labels and legend',
        'source_path':rel(src_path),'source_sha256':sha(src_path.read_bytes()),
        'scientific_panels':proof,'scientific_panel_count':len(proof),
        'resampling':False,'new_scientific_calculations':False,
        'all_visible_source_values_preserved':True,'font_px':fonts,
        'font_pt_at_16_5_cm':font_points,'layout_notes':notes}
    e['visual_review']={'status':'pending_final_view','target_width_cm':16.5,
        'target_height_cm':round(height_cm,3),'scientific_panels_pixel_identity':'passed'}
    print(json.dumps({'id':id,'path':dst.name,'size_px':im.size,
        'height_cm':round(height_cm,3),'font_pt':font_points,'panels':len(proof)},ensure_ascii=False))

# Figure 12: four complete panels, shared phase scale and readable method key.
src_path=CACHE/'27676a97e7a0.png';src=Image.open(src_path).convert('RGB')
im=Image.new('RGB',(2160,1460),'white');proof=[]
text(im,(1080,29),'У1 · первый R–R · ×: сходимость не подтверждена',46)
source_x=[105,1080];source_y=[71,654];target_x=[200,1250];target_y=[125,745]
titles=[['Абсолютный объём','Изменение объёма от опоры R–R'],
        ['Ошибка перемещения от опоры','Перекрытие с маской']]
ylabels=[['Объём, мл','Изменение, мл'],['Ошибка вектора, мм','Dice']]
yp=[[[92,159,225,291,357,423,489],[121,181,241,301,360,420,480,540]],
    [[720,807,894,981,1068],[676,746,816,886,955,1025,1095]]]
ytext=[[['900','850','800','750','700','650','600'],['40','20','0','−20','−40','−60','−80','−100']],
       [['0.04','0.02','0.00','−0.02','−0.04'],['1.00','0.95','0.90','0.85','0.80','0.75','0.70']]]
for row in range(2):
    for col in range(2):
        sx,sy=source_x[col],source_y[row];tx,ty=target_x[col],target_y[row]
        height=[481,480][row]
        panel(src,im,(sx-8,sy,sx+865,sy+height+8),(tx-8,ty),proof)
        text(im,(tx+432,ty-33),titles[row][col],44)
        vertical(im,(tx-150,ty+height/2),ylabels[row][col],44)
        for y,label in zip(yp[row][col],ytext[row][col]):text(im,(tx-15,ty+y-sy),label,44,'rm')
        for x,label in zip([206,306,407,507,608,708,809,909],['20','30','40','50','60','70','80','90']):
            text(im,(tx+x-105,ty+height+29),label,44)
text(im,(1080,1318),'Фаза внутри данного R–R, %',44)
for i,(name,color) in enumerate(zip(['T0','I1','I2','I3','G1','G2','G3a'],
    ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2'])):
    legend(im,220+240*i,1401,name,color,44)
commit('V13',im,src_path,proof,dict(ticks=44,axes=44,panel_titles=44,shared_legend=44,title=46),
    ['All four panels and all seven methods retained.',
     'Original inset key retained unchanged; large external key repeats the same seven methods.',
     'One shared phase-axis label replaces four repeated labels.'])

# Figure 15: six complete bar-chart panels; common ylabel and montage key.
src_path=CACHE/'reader_montage/ch09_06_finite_compensation.png';src=Image.open(src_path).convert('RGB')
im=Image.new('RGB',(2460,1710),'white');proof=[]
text(im,(1230,35),'Конечные FEM-отклики: компенсация объёма изменением ρ сердца',52)
text(im,(1230,96),'Контур — объём; заполнение — линейная компенсация; × — шаг Ньютона',49)
source_x=[155,1303];source_y=[290,773,1257];target_x=[225,1430];target_y=[200,689,1178]
yp=[[[315,389,463,538,612],[312,367,421,476,531,586,640]],
    [[794,848,901,954,1008,1061,1115],[812,886,959,1033,1106]],
    [[1285,1351,1416,1482,1547,1613],[1316,1380,1443,1507,1570]]]
ytext=[[['1.0','0.5','0.0','−0.5','−1.0'],['7.5','5.0','2.5','0.0','−2.5','−5.0','−7.5']],
       [['0.75','0.50','0.25','0.00','−0.25','−0.50','−0.75'],['4','2','0','−2','−4']],
       [['0.10','0.05','0.00','−0.05','−0.10','−0.15'],['4','2','0','−2','−4']]]
for row,name in enumerate(['STL','G1','L1']):
    for col in range(2):
        sx,sy=source_x[col],source_y[row];tx,ty=target_x[col],target_y[row]
        height=[376,376,375][row]
        panel(src,im,(sx-8,sy,sx+1011,sy+height+8),(tx-8,ty),proof)
        title=f'{name} · ρ сердца = 4 Ом·м' if col==0 else f'{name} · условные ρ из подгонки STL'
        text(im,(tx+505,ty-32),title,50)
        for y,label in zip(yp[row][col],ytext[row][col]):text(im,(tx-15,ty+y-sy),label,49,'rm')
        for offset,label in zip([226,785],['V − 1%','V + 1%']):text(im,(tx+offset,ty+height+30),label,49)
vertical(im,(34,868),'Конечное изменение Z, мОм',49)
for x,label,color in [(340,'Исходный, 100 мм','#12658c'),(1020,'95 мм','#e07a18'),
                      (1450,'Поворот на одной руке','#7852a5')]:
    legend(im,x,1672,label,color,49)
commit('V19',im,src_path,proof,dict(ticks=49,axes=49,panel_titles=50,shared_legend=49,title=52),
    ['All six panels, both material scenarios, all three montages and both volume signs retained.',
     'Outlines, filled bars and Newton markers unchanged; no bar value reconstructed from pixels.',
     'Common ylabel replaces six repeated labels.'])

# Figure 17: preserve all four complete time windows at native size. Window IDs
# move to the left margin so repeated long headers no longer consume height.
src_path=CACHE/'ee60fc3b2861.png';src=Image.open(src_path).convert('RGB')
im=Image.new('RGB',(2150,1495),'white');proof=[]
for x,label,color in [(340,'Мягкий фон','#1f77b4'),(1005,'Лёгкие','#ff7f0e'),(1440,'Сердечная область','#2ca02c')]:
    legend(im,x,27,label,color,43)
source_y=[108,484,859,1235];target_y=[65,403,741,1079]
yp=[[134,191,247,303,360],[532,622,712],[913,1009,1105],[1242,1308,1375,1441,1508]]
ytext=[['100','50','0','−50','−100'],['50','0','−50'],['50','0','−50'],['100','50','0','−50','−100']]
xticks=[[194,399,604,808,1013,1218,1422,1627,1831],
        [194,399,604,808,1013,1218,1422,1627,1831],
        [194,467,740,1013,1286,1559,1831],
        [194,428,662,896,1130,1364,1598,1831]]
xtext=[list(map(str,range(36,45))),list(map(str,range(51,60))),
       list(map(str,range(19,26))),list(map(str,range(30,38)))]
for row,code in enumerate(['02-A','02-B','04-A','04-B']):
    sy=source_y[row];ty=target_y[row];height=[296,295,295,295][row]
    panel(src,im,(103,sy,1915,sy+height+8),(292,ty),proof)
    text(im,(108,ty+height/2),code,43)
    for y,label in zip(yp[row],ytext[row]):text(im,(284,ty+y-sy),label,43,'rm')
    for x,label in zip(xticks[row],xtext[row]):text(im,(x+189,ty+height+24),label,43)
vertical(im,(30,743),'Расчётный вклад, мОм',43)
text(im,(1197,1460),'Время внутри записи, с',43)
commit('V15',im,src_path,proof,dict(ticks=43,axes=43,window_codes=43,shared_legend=43),
    ['All four complete time windows and all three tissue traces retained.',
     'Original first-panel inset legend retained unchanged; large shared key added outside data.',
     'Candidate-window IDs moved to left margin; time limits remain on every panel.'])

# Update metadata only for these three figures; never overwrite a chapter,
# rendered report, photograph or the author-owned overrides.
changes=manifest.setdefault('readability_changes',{})
for alias in ['models_geometry_comparison','models_finite_compensation','tissue_pulse_projection']:
    changes[alias]='All complete source plot interiors retained at native pixels; shared labels and legend are 9–10 pt at 16.5 cm. New numerical analysis not performed.'
manifest['targeted_three_font_review']={
    'figures':['models_geometry_comparison','models_finite_compensation','tissue_pulse_projection'],
    'main_figure_count':19,'word_width_cm':16.5,'max_height_cm':11.5,
    'frozen_source_panels':[4,6,4],'no_source_values_removed':True,
    'status':'awaiting_final_image_view','source_numeric_tables_available':False,
    'method':'native raster panel reuse, readable external typography; no curve digitisation'}
review=manifest.get('assembly_review',{})
review['remaining_readability_notes']=[]
review['rebuild_required_for_updated_assets']=list(dict.fromkeys([
    *review.get('rebuild_required_for_updated_assets',[]),
    'models_geometry_comparison','models_finite_compensation','tissue_pulse_projection']))
review['older_pdf_review_note']='The earlier page list and PDF hash refer to the intermediate PDF inspected before this targeted typography change.'
for p,digest in protected.items():assert sha(p.read_bytes())==digest
temporary=MP.with_name('manifest.writing.json')
temporary.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
os.replace(temporary,MP)
