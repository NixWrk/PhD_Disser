"""Build and execute the geometry-results notebook, then export HTML without code."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(__file__).with_name('heart_geometry_results_ru.md')
NOTEBOOK=ROOT/'40.16_Геометрическое_сравнение_моделей_сердца.ipynb'
CODE={
'setup': '''from pathlib import Path
import sys, json, hashlib
from IPython.display import display, Markdown, Image
root = Path.cwd()
if root.name != 'Colab Notebooks':
    root = root / 'Colab Notebooks'
sys.path.insert(0, str(root / 'cardiac_4d_segmentation'))
import heart_geometry_report as report
import matplotlib.pyplot as plt
work = root / 'cardiac_4d_segmentation/results/reduced_heart_geometry_20260914/exploratory'
records = report.load_results(work / 'comparison_02')
temporal = json.loads((work / 'temporal_01/results.json').read_text(encoding='utf-8'))
affine = [json.loads(line) for line in (work / 'affine_01/affine_comparison.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(affine) == 234
assert len(records) == 234
assert len(temporal['predictions']) == 480
figures = work / 'reader_figures'
figures.mkdir(exist_ok=True)
ct_directory = work / 'ct_overlays_01'
ct_manifest = json.loads((ct_directory / 'manifest.json').read_text(encoding='utf-8'))
assert ct_manifest['baseline_sha256'] == hashlib.sha256((work / 'comparison_02/geometry_comparison.jsonl').read_bytes()).hexdigest()
def show_ct_example(subject):
    example = next(x for x in ct_manifest['examples'] if x['subject'] == subject)
    path = ct_directory / example['figure']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == example['figure_sha256']
    display(Image(filename=str(path)))
def show_figure(figure, name):
    figure.savefig(figures / name, dpi=170, bbox_inches='tight', facecolor='white')
    display(Image(filename=str(figures / name)))
    plt.close(figure)
''',
'ct_adam':"show_ct_example('adam')",
'ct_nix':"show_ct_example('nix')",
'ct_georg':"show_ct_example('georg')",
'geometry_table':"display(Markdown(report.geometry_table(records)))",
'overlap':"show_figure(report.overlap_figure(records), '01_overlap.png')",
'paired':"show_figure(report.paired_figure(records), '02_paired.png')",
'affine_table':"display(Markdown(report.affine_table(records, affine)))",
'affine':"show_figure(report.affine_figure(records, affine), '03_affine.png')",
'overlay':"show_figure(report.overlay_schematic(), '07_overlay.png')",
'boundary':"display(Markdown(report.qc_table(records)))",
'discretization':"show_figure(report.discretization_figure(records), '04_discretization.png')",
'coverage':"show_figure(report.coverage_figure(records), '05_coverage.png')",
 'temporal_table':"display(Markdown(report.temporal_table(temporal)))",
 'temporal':"show_figure(report.temporal_figure(temporal), '06_temporal.png')"
}


def build():
    text=SOURCE.read_text(encoding='utf-8-sig')
    parts=re.split(r'<!-- CODE: (\w+) -->',text)
    cells=[]
    for i,part in enumerate(parts):
        if i%2:
            cells.append(nbformat.v4.new_code_cell(CODE[part],id=f'code-{part}'))
        elif part.strip():
            cells.append(nbformat.v4.new_markdown_cell(part.strip(),id=f'text-{i//2}'))
    nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'evidence_status':'exploratory_hypothesis_not_validated',
        'source_text_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest()})
    NotebookClient(nb,timeout=180,resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(nb,NOTEBOOK)
    exporter=HTMLExporter()
    exporter.exclude_input=True;exporter.exclude_input_prompt=True;exporter.exclude_output_prompt=True
    body,_=exporter.from_notebook_node(nb)
    NOTEBOOK.with_suffix('.html').write_text(body,encoding='utf-8')
    receipt={'notebook':NOTEBOOK.name,'cells':len(nb.cells),
        'executed_code_cells':sum(c.cell_type=='code' for c in nb.cells),
        'output_errors':[o for c in nb.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='error'],
        'source_text_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'notebook_sha256':hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
        'html_sha256':hashlib.sha256(NOTEBOOK.with_suffix('.html').read_bytes()).hexdigest(),
        'evidence_status':'exploratory_hypothesis_not_validated'}
    out=ROOT/'cardiac_4d_segmentation/results/reduced_heart_geometry_20260914/exploratory/reader_build.json'
    out.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':
    build()
