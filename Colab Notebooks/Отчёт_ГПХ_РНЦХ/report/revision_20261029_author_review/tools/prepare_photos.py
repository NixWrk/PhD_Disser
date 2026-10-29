"""Byte-identical local copies of the seven author-supplied photographs.
Cropping is metadata for Word a:srcRect; no cropped or retouched photo is saved.
"""
from pathlib import Path
import hashlib,json,shutil
from PIL import Image
REV=Path(__file__).resolve().parents[1];ROOT=REV.parents[3];OUT=REV/'figures';D=OUT/'photos';D.mkdir(exist_ok=True)
names=[
'6364ac9f-8b23-471a-a2bb-47d6cb6dbb91','ca31a60d-3fd6-4d89-9319-a149cab45dab',
'35279ec2-6975-4d2b-8290-16f39a12484e','c4918d42-6b0b-4b08-adf4-6f28ea26e10c',
'fe285acd-dca7-4628-9528-7a94f48539be','0a6c0189-6e04-4eab-b2e1-9878dab2e3b8',
'8b2568a5-6326-4aeb-853a-5b59369888f2']
aliases=['photo_belt','photo_lateral_wide','photo_lateral_detail','photo_preparation_contacts','photo_small_array','photo_lateral_arm','photo_tt_wide']
captions=[
'Пояс ЭИТ вокруг грудной клетки на этапе подготовки к исследованию. Фотография показывает установленный пояс; её положение не задаёт плоскость пояса в координатах индивидуальной КТ.',
'Общий вид бокового монтажа и электродов на руках на этапе подготовки к исследованию.',
'Боковая электродная сборка на этапе подготовки к исследованию. Показаны установленные контакты и их кабельные соединения.',
'Размещение электродов на грудной клетке и руках на этапе подготовки. На снимке часть площадок ещё не подключена; наличие электрода не означает, что соответствующее отведение уже регистрировалось.',
'Компактная боковая сборка на этапе подготовки к исследованию. Фотография показывает четыре подключённых контакта; её размер не определяется по изображению без отдельного масштаба.',
'Боковые контакты и электроды на руке на этапе подготовки к исследованию.',
'Общий вид подготовки монтажа трансторакальных измерений. Положение рук на подготовительном снимке отличается от принятой позы регистрации: лёжа на спине, руки вытянуты за голову и не сцеплены.'
]
# Fractions cropped from left/top/right/bottom, not retained rectangles.
crops=[
dict(left=.04,top=.16,right=.10,bottom=.28),
None,
dict(left=.30,top=.07,right=.03,bottom=.06),
dict(left=.015,top=.19,right=.10,bottom=.29),
dict(left=.08,top=.09,right=.08,bottom=.06),
dict(left=.01,top=.055,right=.45,bottom=.20),
dict(left=.12,top=.38,right=.16,bottom=.20)]
records=[]
for i,(name,alias,cap,crop) in enumerate(zip(names,aliases,captions,crops),1):
 src=Path('C:/Users/9838~1/AppData/Local/Temp')/f'codex-clipboard-{name}.png';dst=D/f'photo{i:02}.png'
 b=src.read_bytes();shutil.copyfile(src,dst);h=hashlib.sha256(b).hexdigest()
 assert dst.read_bytes()==b
 with Image.open(dst) as im:w,ht=im.size;pixel=hashlib.sha256(im.convert('RGBA').tobytes()).hexdigest()
 e=dict(id=f'P{i:02}',path=dst.relative_to(ROOT).as_posix(),aliases=[alias],alias_paths={alias:dst.relative_to(ROOT).as_posix()},
  caption=cap,purpose='Документировать реальную установку на этапе подготовки.',suggested_chapter=2,width=w,height=ht,unit='px',
  source={'type':'author_supplied_photograph','original_path':str(src),'sha256':h},sha256=h,rgba_pixel_sha256=pixel,
  byte_identical_to_source=True,pixels_unchanged=True,participant_code=None,record_number=None,
  crop_fraction=crop,crop_fraction_semantics='fractions removed from original image; Word a:srcRect values = round(fraction*100000)',
  word_srcRect={k:round(v*100000) for k,v in crop.items()} if crop else None,
  restrictions=['Фотография относится к подготовке; во время регистрации руки поднимали за голову.','Принадлежность кодам У1/У2/У3 по внешности не устанавливалась.','Исходная фотография содержит персональные визуальные признаки; перед публичным использованием применять проверенное кадрирование в документе.'],
  visual_review={'status':'viewed_individually','contacts_visible':True,'source_pixels_modified':False},
  crop_preserves='целевые контакты и площадки, показанные в выбранной области')
 if i==2:
  e['crop_note']='Одна прямоугольная обрезка не скрывает голову, сохраняя оба верхних контакта на руках. Предпочесть P03 для боковой детали, P07 для общего вида; этот оригинал оставить резервом.'
  e['recommended_main_text']=False
 else:e['recommended_main_text']=i in [1,3,7]
 records.append(e)
m=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'));m['photos']=records
m['photo_compositions']=[
 dict(id='photo_group_belt',panels=['P01'],caption=records[0]['caption'],suggested_chapter=2),
 dict(id='photo_group_lateral',panels=['P03','P05'],caption='Боковые электродные сборки на этапе подготовки: общий вид установленных контактов и компактная сборка. Масштабы фотографий различаются; принадлежность одному участнику не устанавливается.',suggested_chapter=2),
 dict(id='photo_group_tt',panels=['P06','P07'],caption='Подготовка бокового и трансторакального монтажа: контакты на руке и общий вид установки. Изображения относятся к подготовке, а не к позе регистрации.',suggested_chapter=2)]
m['photo_policy']='Keep original PNG bytes; crop in Word only. P04 is preparation with partly disconnected pads. Do not infer subject identity.'
(OUT/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Copied and checked seven photographs; no pixel edits.')
