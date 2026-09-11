"""Execute generated report cells in isolation, optionally hiding historical results."""
import argparse, ast, contextlib, hashlib, io, json, os, runpy, time, traceback
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--without-history',action='store_true');p.add_argument('--figures',action='store_true');args=p.parse_args()
os.chdir(ROOT);os.environ['MPLBACKEND']='Agg';os.environ.pop('KALMYKOV_DICOM_DIR',None)
os.environ['TRKG4_RESULT_TAG']=args.tag
suffix=args.tag+('_minimal' if args.without_history else '')
generated=HERE/f'report_{suffix}.ipynb';os.environ['TRKG4_REPORT_TARGET']=str(generated)
runpy.run_path(str(ROOT/'notebooks/build_trkg4_inverse_report.py'))
nb=json.loads(generated.read_text(encoding='utf-8'))
result={'tag':args.tag,'historical_files_hidden':args.without_history,'scope':'all code cells; no DICOM; no FEM solve','cells':[]}
ns={'__name__':'__report_verification__'}
original_is_file=Path.is_file;original_read_text=Path.read_text;original_read_csv=pd.read_csv
out=(ROOT/'output').resolve()
def hidden(value):
    if not args.without_history or not isinstance(value,(str,os.PathLike)):return False
    path=Path(value).resolve()
    if not path.is_relative_to(out):return False
    name=path.name
    return not (name.endswith('_'+args.tag+'.csv') or name.endswith('_'+args.tag+'.json') or name.endswith('_'+args.tag+'.mat') or name=='nik_trkg4_right_rib_050mm_full_v5_1mm_local2mm.mat')
def is_file(path):return False if hidden(path) else original_is_file(path)
def read_text(path,*a,**kw):
    if hidden(path):raise FileNotFoundError(f'Hidden historical fixture: {path}')
    return original_read_text(path,*a,**kw)
def read_csv(path,*a,**kw):
    if hidden(path):raise FileNotFoundError(f'Hidden historical fixture: {path}')
    return original_read_csv(path,*a,**kw)
with patch.object(Path,'is_file',is_file),patch.object(Path,'read_text',read_text),patch.object(pd,'read_csv',read_csv):
    for i,cell in enumerate(nb['cells']):
        if cell['cell_type']!='code':continue
        source=''.join(cell['source']);start=time.perf_counter()
        record={'cell':i,'status':'passed'}
        try:
            ast.parse(source)
            with contextlib.redirect_stdout(io.StringIO()):exec(compile(source,f'cell_{i}','exec'),ns)
            if 'display' in ns:ns['display']=lambda *a,**kw:None
            if 'show_plotly' in ns:ns['show_plotly']=lambda *a,**kw:None
            if 'plt' in ns:
                if args.figures:
                    for number in ns['plt'].get_fignums():
                        ns['plt'].figure(number).savefig(HERE/f'{suffix}_cell_{i}_figure_{number}.png',dpi=110,bbox_inches='tight')
                ns['plt'].close('all')
        except Exception:
            record['status']='failed';record['traceback']=traceback.format_exc()
        record['seconds']=time.perf_counter()-start;result['cells'].append(record);print(json.dumps(record,ensure_ascii=False),flush=True)
        if record['status']=='failed':break
if all(x['status']=='passed' for x in result['cells']) and 'z_two_layer' in ns:
    z=ns['z_two_layer'];result['homogeneous_relative_error']=max(abs(z(L,5,5,20)/(8*5/(3*np.pi*L/1000))-1) for L in [50,90,140])
result['generator_sha256']=hashlib.sha256((ROOT/'notebooks/build_trkg4_inverse_report.py').read_bytes()).hexdigest()
(HERE/f'report_checks_{suffix}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
if any(x['status']=='failed' for x in result['cells']):raise SystemExit(1)
print('PASSED',len(result['cells']),'code cells')
