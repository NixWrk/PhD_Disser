"""Rasterize a Word-exported PDF and retain page text for layout review."""
import argparse
import json
from pathlib import Path

import pypdfium2 as pdfium

ap = argparse.ArgumentParser()
ap.add_argument('pdf', type=Path)
ap.add_argument('--scale', type=float, default=1.5)
args = ap.parse_args()
out = args.pdf.parent
doc = pdfium.PdfDocument(str(args.pdf))
pages = []
for i, page in enumerate(doc):
    image = page.render(scale=args.scale).to_pil()
    image.save(out / f'page-{i+1:03d}.png')
    textpage = page.get_textpage()
    text = textpage.get_text_range()
    (out / f'page-{i+1:03d}.txt').write_text(text, encoding='utf-8')
    pages.append({'page': i+1, 'characters': len(text),
                  'size_pt': list(page.get_size()), 'start': text[:160]})
(out / 'pages.json').write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'pages': len(pages), 'images': str(out)}, ensure_ascii=False))
