"""Execute only reader cells; source FEM/geometry results are not recomputed."""
from pathlib import Path
import json,re,hashlib
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(__file__).with_name('heart_method_comparison_ru.md')
NOTEBOOK=ROOT/'40.17_Сравнение_форм_сердца_и_электрического_отклика.ipynb'
CODE={'core':'report.core()', 'spheres':"report.method_tradeoff('G')",'ellipsoids':"report.method_tradeoff('L')",
      'physical_settings':'report.physical_settings()','electrodes':'report.electrodes()',
      'electrical':'report.electrical()','material_resolution':'report.material_resolution()','table':'report.table()','fractional_resolution':'report.fractional_resolution()',
      'libraries':'report.libraries()','phase_stability':'report.phase_stability()','volume_sensitivity':'report.volume_sensitivity()',
      'ellipsoid_surface_validation':'report.ellipsoid_surface_validation()',
      'individual_volume_sensitivity':'report.individual_volume_sensitivity()',
      'individual_quadrature_refinement':'report.individual_quadrature_refinement()',
      'heart_directional_response':'report.heart_directional_response()',
      'heart_spatial_baseline':'report.heart_spatial_baseline()'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    source=SOURCE.read_text(encoding='utf-8-sig')
    nb=nbformat.v4.new_notebook()
    nb.metadata.update(kernelspec={'display_name':'Python 3','language':'python','name':'python3'},
        language_info={'name':'python'},scientific_status='exploratory_not_validated',source_sha256=sha(SOURCE))
    nb.cells=[nbformat.v4.new_code_cell("from pathlib import Path\nimport sys\nroot=Path.cwd()\nif root.name != 'Colab Notebooks': root=root/'Colab Notebooks'\nsys.path.insert(0,str(root/'cardiac_4d_segmentation'))\nimport heart_method_comparison_report as report")]
    for part in re.split(r'(<!-- CODE: [a-z_]+ -->)',source):
        match=re.fullmatch(r'<!-- CODE: ([a-z_]+) -->',part)
        if match:nb.cells.append(nbformat.v4.new_code_cell(CODE[match[1]]))
        elif part.strip():nb.cells.append(nbformat.v4.new_markdown_cell(part.strip()))
    nb.cells.append(nbformat.v4.new_code_cell('report.write_input_provenance()'))
    NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.validate(nb);nbformat.write(nb,NOTEBOOK)
    exporter=HTMLExporter();exporter.exclude_input=True;exporter.exclude_input_prompt=True;exporter.exclude_output_prompt=True
    html,_=exporter.from_notebook_node(nb);NOTEBOOK.with_suffix('.html').write_text(html,encoding='utf-8')
    work=ROOT/'cardiac_4d_segmentation/results/reduced_heart_geometry_20260914/exploratory/method_reader_01'
    receipt={'notebook_sha256':sha(NOTEBOOK),'html_sha256':sha(NOTEBOOK.with_suffix('.html')),
        'source_sha256':sha(SOURCE),'builder_sha256':sha(__file__),'figure_module_sha256':sha(Path(__file__).with_name('heart_method_comparison_report.py')),
        'cell_count':len(nb.cells),'code_cells':sum(c.cell_type=='code' for c in nb.cells),
        'image_outputs':sum('image/png' in o.get('data',{}) for c in nb.cells for o in c.get('outputs',[])),
        'FEM_reexecuted':False,'geometry_reexecuted':False,'html_inputs_excluded':True,
        'input_provenance_sha256':sha(work/'input_provenance.json')}
    (work/'build.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt))
if __name__=='__main__':main()
