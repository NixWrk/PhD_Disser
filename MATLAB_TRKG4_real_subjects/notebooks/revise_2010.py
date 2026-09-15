"""Integrate the scientific figure revision of notebook 20.10.

Works on the canonical saved notebook without rerunning electrical solves.
The same apply() is called after the report generator has assembled its cells.
"""
from pathlib import Path
import argparse, hashlib, importlib, json
import nbformat

MODULES=('revise_2010_cleanup','revise_2010_geometry','revise_2010_pem','revise_2010_prose')
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'MATLAB_TRKG4_real_subjects/output/exploratory/notebook_2010_revision_20260914'

def apply(nb):
    nb=nbformat.reads(json.dumps(nb,ensure_ascii=False),as_version=4)
    for name in MODULES:
        module=importlib.import_module(name)
        result=module.apply(nb)
        if result is not None:nb=result
        nb=nbformat.reads(json.dumps(nb,ensure_ascii=False),as_version=4)
    nbformat.validate(nb)
    return nb

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--notebook',type=Path)
    p.add_argument('--export-html',action='store_true')
    args=p.parse_args()
    target=args.notebook or next((ROOT/'Colab Notebooks').glob('20.10*.ipynb'))
    nb=apply(nbformat.read(target,as_version=4))
    for cell in nb.cells:
        if cell.cell_type=='code':compile(cell.source,'notebook code','exec')
    first=nbformat.writes(nb)
    assert nbformat.writes(apply(nb))==first,'Revision must be idempotent'
    numerical=json.loads((OUT/'numerical_sources_before.json').read_text(encoding='utf8'))
    for path,digest in numerical.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    target.write_text(first,encoding='utf8')
    if args.export_html:
        from nbconvert import HTMLExporter
        exporter=HTMLExporter()
        exporter.exclude_input=True
        exporter.exclude_input_prompt=True
        exporter.exclude_output_prompt=True
        html,_=exporter.from_notebook_node(nb)
        target.with_suffix('.html').write_text(html,encoding='utf8')
    print(target)

if __name__=='__main__':main()
