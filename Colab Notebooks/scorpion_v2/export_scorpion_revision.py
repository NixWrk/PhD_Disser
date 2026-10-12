"""Export verified, computed revision; Jupyter requires local kernel sockets."""
from build_georg_fem_reports import export
import build_scorpion_experiment_viewer as viewer
import lateral_pilot_revision as pilot
import build_lateral_solution_surfaces_report as surfaces
import build_georg_paired_preflight_report as paired
from build_scorpion_supervisor import build


def supplement_historical_pilot():
    import nbformat
    from nbconvert import HTMLExporter
    path=viewer.FOLDER/'33.20_Первый_проход_боковых_сборок_Scorpion_v2.ipynb'
    notebook=nbformat.read(path,4)
    label='## Пояснение расчёта и уточнение протокола 12.10.2026'
    notebook.cells=[c for c in notebook.cells if not (c.cell_type=='markdown' and c.source.startswith(label))]
    text=(viewer.FOLDER/'33.20_Пояснение_расчёта_и_изменений_12_10_2026.md').read_text(encoding='utf-8')
    text=label+'\n\n'+text.split('\n',1)[1]
    notebook.cells.insert(1,nbformat.v4.new_markdown_cell(text))
    nbformat.write(notebook,path)
    exporter=HTMLExporter()
    exporter.exclude_input=exporter.exclude_input_prompt=exporter.exclude_output_prompt=True
    page,_=exporter.from_notebook_node(notebook)
    path.with_suffix('.html').write_text(page,encoding='utf-8')
    path.with_suffix('.md').write_text('\n\n'.join(c.source for c in notebook.cells if c.cell_type=='markdown')+'\n',encoding='utf-8')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--viewer-only',action='store_true')
    args=parser.parse_args()
    viewer.run()
    export(viewer.STEM,viewer.cells(),viewer.OUT)
    print('10.00 exported',flush=True)
    build()
    if args.viewer_only:raise SystemExit(0)
    export(pilot.STEM,pilot.cells(),pilot.OUT)
    print('33.24 exported',flush=True)
    export(surfaces.STEM,surfaces.cells(),surfaces.OUT)
    print('33.21 labels updated; numerical outputs unchanged',flush=True)
    export(paired.STEM,paired.cells(),paired.OUT)
    print('33.25 exported',flush=True)
    supplement_historical_pilot()
