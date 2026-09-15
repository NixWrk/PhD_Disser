"""Build and execute the geometry-results notebook, then export HTML without code."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(__file__).with_name('heart_geometry_results_ru.md')
REGISTRY=ROOT/'40.15_Реестр_методов_аппроксимации_сердца.md'
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
construction_directory = work / 'construction_criteria_01'
construction = json.loads((construction_directory / 'results.json').read_text(encoding='utf-8'))
assert construction['baseline_sha256'] == ct_manifest['baseline_sha256']
assert len(construction['examples']) == 3

def show_construction(subject):
    example = next(x for x in construction['examples'] if x['subject'] == subject)
    path = construction_directory / example['figure']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == example['figure_sha256']
    display(Image(filename=str(path)))

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
'construction_table':"display(Markdown(report.construction_table(construction)))",
'construction_adam':"show_construction('adam')",
'construction_nix':"show_construction('nix')",
'construction_georg':"show_construction('georg')",
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


def build(*, presentation_only=False):
    text=SOURCE.read_text(encoding='utf-8-sig')
    registry=REGISTRY.read_text(encoding='utf-8')
    # Embed the authoritative list as Markdown; do not maintain a second copy.
    registry=re.sub(r'^## (\d+)\.', r'### 2.1.\1.', registry, flags=re.MULTILINE)
    registry=registry.replace('## Как читать реестр', '### Как читать реестр', 1)
    registry=registry.replace('# Общий реестр методов аппроксимации сердца',
                              '## 2.1. Общий реестр методов аппроксимации сердца', 1)
    marker='<!-- INCLUDE: heart_approximation_registry -->'
    if text.count(marker) != 1:
        raise ValueError('Expected exactly one method-registry inclusion')
    text=text.replace(marker,registry.strip())
    parts=re.split(r'<!-- CODE: (\w+) -->',text)
    cells=[]
    for i,part in enumerate(parts):
        if i%2:
            cells.append(nbformat.v4.new_code_cell(CODE[part],id=f'code-{part}'))
        elif part.strip():
            cells.append(nbformat.v4.new_markdown_cell(part.strip(),id=f'text-{i//2}'))
    nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'evidence_status':'exploratory_hypothesis_not_validated',
        'source_text_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'method_registry_sha256':hashlib.sha256(REGISTRY.read_bytes()).hexdigest()})
    if presentation_only:
        previous=nbformat.read(NOTEBOOK,as_version=4)
        saved={cell.id:cell for cell in previous.cells if cell.cell_type=='code'}
        fresh={cell.id:cell for cell in cells if cell.cell_type=='code'}
        if saved.keys() != fresh.keys() or any(saved[key].source != fresh[key].source for key in fresh):
            raise ValueError('Presentation-only mode requires unchanged code cells')
        updated_metadata=deepcopy(previous.metadata)
        updated_metadata.update(nb.metadata)
        nb.metadata=updated_metadata
        nb.cells=[deepcopy(saved[cell.id]) if cell.cell_type=='code' else cell for cell in cells]
    else:
        NotebookClient(nb,timeout=180,resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(nb,NOTEBOOK)
    exporter=HTMLExporter()
    exporter.exclude_input=True;exporter.exclude_input_prompt=True;exporter.exclude_output_prompt=True
    body,_=exporter.from_notebook_node(nb)
    NOTEBOOK.with_suffix('.html').write_text(body,encoding='utf-8')
    receipt={'notebook':NOTEBOOK.name,'cells':len(nb.cells),
        'executed_code_cells':0 if presentation_only else sum(c.cell_type=='code' for c in nb.cells),
        'build_mode':'preserve_saved_outputs' if presentation_only else 'execute',
        'code_cells':sum(c.cell_type=='code' for c in nb.cells),
        'output_errors':[o for c in nb.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='error'],
        'source_text_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'method_registry_sha256':hashlib.sha256(REGISTRY.read_bytes()).hexdigest(),
        'notebook_sha256':hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
        'html_sha256':hashlib.sha256(NOTEBOOK.with_suffix('.html').read_bytes()).hexdigest(),
        'evidence_status':'exploratory_hypothesis_not_validated'}
    out=ROOT/'cardiac_4d_segmentation/results/reduced_heart_geometry_20260914/exploratory/reader_build.json'
    out.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--presentation-only',action='store_true',
                        help='Update prose and HTML while preserving unchanged code cells and outputs')
    args=parser.parse_args()
    build(presentation_only=args.presentation_only)
