"""Flag text outside printable margins; complements, never replaces visual QA."""
import argparse
import json
from pathlib import Path
import pypdfium2 as pdfium

p = argparse.ArgumentParser()
p.add_argument('pdf', type=Path)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
doc = pdfium.PdfDocument(args.pdf)
rows = []
for number, page in enumerate(doc, 1):
    text = page.get_textpage()
    width, height = page.get_size()
    bad = []
    for i in range(text.count_chars()):
        c = text.get_text_range(i, 1)
        if not c.strip():
            continue
        x0, y0, x1, y1 = text.get_charbox(i)
        if y1 > 40 and (x0 < 80 or x1 > width - 38):
            bad.append({'index': i, 'char': c, 'box': [round(v, 2) for v in (x0,y0,x1,y1)]})
    value = text.get_text_range()
    rows.append({'page': number, 'text_chars': len(value), 'start': value[:160],
                 'overflow': bad, 'sparse': len(value.strip()) < 70})
    text.close(); page.close()
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps({'pages': rows, 'scope': 'PDF text geometry; images and scientific correctness require visual review'}, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'pages': len(rows), 'overflow_pages': [r['page'] for r in rows if r['overflow']],
                  'sparse_pages': [r['page'] for r in rows if r['sparse']]}, ensure_ascii=False))
