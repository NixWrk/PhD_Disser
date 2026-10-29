"""Two full-size rendered pages per inspection sheet, with no page resampling."""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw
p=argparse.ArgumentParser()
p.add_argument('pages',type=Path)
p.add_argument('--first',type=int,required=True)
p.add_argument('--last',type=int,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
for start in range(a.first,a.last+1,2):
    pages=[(n,Image.open(a.pages/f'page-{n:03d}.png').convert('RGB')) for n in range(start,min(start+2,a.last+1))]
    sheet=Image.new('RGB',(sum(im.width for _,im in pages)+20*(len(pages)+1),max(im.height for _,im in pages)+60),'#dddddd')
    x=20;draw=ImageDraw.Draw(sheet)
    for n,im in pages:
        draw.text((x,12),f'PDF PAGE {n}',fill='black');sheet.paste(im,(x,40));x+=im.width+20
    sheet.save(a.output/f'pages-{start:03d}-{pages[-1][0]:03d}.png')
print(a.output)
