"""Carry visual acceptance forward only for pixel-identical page bodies."""
from pathlib import Path
from PIL import Image
import hashlib,json,argparse
ap=argparse.ArgumentParser()
ap.add_argument('old',type=Path);ap.add_argument('new',type=Path)
ap.add_argument('--output',required=True,type=Path)
a=ap.parse_args()
def fingerprint(p):
 with Image.open(p) as im:
  # Rendering scale is identical (1.5); bottom 100 px contains the page footer.
  body=im.convert('RGB').crop((0,0,im.width,im.height-100))
  return hashlib.sha256(body.tobytes()).hexdigest()
def number(p):return int(p.stem.split('-')[1])
old={fingerprint(p):number(p) for p in sorted(a.old.glob('page-*.png'))}
matches=[];changed=[]
for p in sorted(a.new.glob('page-*.png')):
 h=fingerprint(p)
 if h in old:matches.append({'new_page':number(p),'reviewed_old_page':old[h],'body_sha256':h})
 else:changed.append(number(p))
out={'comparison':'Exact RGB pixel equality of full page body; only bottom 100px footer excluded. Changed bodies require visual review.','old_pdf_sha256':hashlib.sha256((a.old/'report.pdf').read_bytes()).hexdigest(),'new_pdf_sha256':hashlib.sha256((a.new/'report.pdf').read_bytes()).hexdigest(),'body_matches':matches,'pages_requiring_visual_review':changed}
a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'identical_bodies':len(matches),'requiring_visual_review':changed},ensure_ascii=False))
