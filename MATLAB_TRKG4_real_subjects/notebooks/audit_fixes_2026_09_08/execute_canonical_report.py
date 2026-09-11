"""Build, execute in a fresh kernel, and export the accepted report locally."""
import argparse, hashlib, json, os, runpy, shutil, sys, time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager

here=Path(__file__).resolve().parent
root=here.parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--tag',default='auditfix_20260908')
args=parser.parse_args()
os.chdir(root)
os.environ.pop('KALMYKOV_DICOM_DIR',None)
os.environ.pop('MPLBACKEND',None)
os.environ['TRKG4_RESULT_TAG']=args.tag
notebook_path=root/'notebooks/TRKG4_inverse_inhale_results.ipynb'
html_path=notebook_path.with_suffix('.html')
backup=here/'before'
backup.mkdir(exist_ok=True)
for path in (notebook_path,html_path):
    if path.is_file() and not (backup/path.name).exists():
        shutil.copy2(path,backup/path.name)
os.environ['TRKG4_REPORT_TARGET']=str(notebook_path)
generator=root/'notebooks/build_trkg4_inverse_report.py'
generator_sha=hashlib.sha256(generator.read_bytes()).hexdigest()
runpy.run_path(str(generator))
notebook=nbformat.read(notebook_path,as_version=4)
manager=KernelManager(kernel_name='python3')
manager.kernel_spec.argv[0]=sys.executable
started=time.monotonic()
def report_cell(cell,cell_index,**kwargs):
    if cell.cell_type=='code':
        print(json.dumps({'cell':cell_index,'event':'start','elapsed_s':round(time.monotonic()-started,1)}),flush=True)
client=NotebookClient(notebook,km=manager,timeout=600,allow_errors=False,
    resources={'metadata':{'path':str(root)}},on_cell_start=report_cell)
try:
    client.execute()
finally:
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)
assert not any(output.output_type=='error' for cell in notebook.cells if cell.cell_type=='code' for output in cell.outputs)
nbformat.write(notebook,notebook_path)
exporter=HTMLExporter()
html,_=exporter.from_notebook_node(notebook)
html_path.write_text(html,encoding='utf-8')
result={'tag':args.tag,'code_cells':sum(cell.cell_type=='code' for cell in notebook.cells),
    'errors':0,'generator_sha256':generator_sha,'notebook_sha256':hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
    'html_sha256':hashlib.sha256(html_path.read_bytes()).hexdigest(),'dicom_available':False,'notebook':str(notebook_path),'html':str(html_path),
    'elapsed_s':time.monotonic()-started,'scope':'fresh Python kernel and HTML export; no MATLAB solve'}
(here/'canonical_report_execution.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False),flush=True)
