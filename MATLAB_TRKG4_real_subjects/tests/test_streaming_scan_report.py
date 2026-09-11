import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('scan_report', ROOT/'notebooks/build_trkg4_full_scan_report.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)

def snapshot(path, rows):
    centres = pd.DataFrame(dict(centre_index=[1,2],u_mm=[0,1],v_mm=[0,0],
        x_mm=[0,1],y_mm=[0,0],z_mm=[0,0],h_vertex_mm=[15,15]))
    centres.to_csv(path/'centres.csv',index=False)
    sizes=[50,60,70,80,90,110,120,130,140]
    data=[]
    for ci,phi,state in rows:
        z=np.arange(9,dtype=float)+ci if state=='evaluated' else np.full(9,np.nan)
        d=dict(centre_index=ci,phi_deg=phi,status=state,rmse_ohm=float(ci) if state=='evaluated' else np.nan,
               mae_ohm=float(ci) if state=='evaluated' else np.nan)
        d.update({f'Z_L{L:03d}_ohm':v for L,v in zip(sizes,z)})
        data.append(d)
    pd.DataFrame(data).to_csv(path/'poses.csv',index=False)
    savemat(path/'surface.mat',dict(nodes_mm=[[0,0,0],[1,0,0],[0,1,0]],boundary_faces=[[1,2,3]]))
    best=min((d for d in data if d['status']=='evaluated'),key=lambda d:d['rmse_ohm'],default=None)
    summary=dict(grid=dict(u=[0,1],v=[0],phi=[-1,0,1]),rho1=5.,rho2=30.,
        experiment=dict(L_mm=sizes,Z_ohm=list(range(9))),complete=False,bestrecord=best,
        counts={k:sum(d['status']==k for d in data) for k in ['evaluated','invalid_geometry','outside_search_region']})
    (path/'summary.json').write_text(json.dumps(summary),encoding='utf-8')
    (path/'input.json').write_text('{}',encoding='utf-8')
    manifests(path)

def manifests(path):
    def record(name):return dict(name=name,path=name,sha256=hashlib.sha256((path/name).read_bytes()).hexdigest())
    (path/'manifest.json').write_text(json.dumps(dict(inputs=[record('input.json')],outputs=[record(n) for n in
        ['centres.csv','poses.csv','summary.json','surface.mat']])),encoding='utf-8')

def test_partial_distinguishes_excluded_failed_and_missing(tmp_path):
    snapshot(tmp_path,[(1,-1,'evaluated'),(1,0,'outside_search_region'),(1,1,'invalid_geometry')])
    s=report.load_scan(tmp_path,allow_partial=True,enforce_production_grid=False)
    assert s['status'].tolist()==[[1,3,2],[0,0,0]]
    assert s['counts']['missing']==3
    assert s['counts']['outside_search_region']==1
    assert report.best_layer(s)[2].tolist()==[1,0]
    report.build_report(s,tmp_path/'report')
    text=(tmp_path/'report/full_scan.html').read_text(encoding='utf-8')
    assert '9874b8' in text and 'Вне области' in text

def test_all_excluded_centre_has_own_display_status(tmp_path):
    snapshot(tmp_path,[(1,p,'outside_search_region') for p in [-1,0,1]])
    s=report.load_scan(tmp_path,allow_partial=True,enforce_production_grid=False)
    assert report.best_layer(s)[2].tolist()==[3,0]

def test_partial_cannot_be_claimed_complete(tmp_path):
    snapshot(tmp_path,[(1,-1,'evaluated')])
    q=json.loads((tmp_path/'summary.json').read_text());q['complete']=True
    (tmp_path/'summary.json').write_text(json.dumps(q));manifests(tmp_path)
    with pytest.raises(ValueError,match='False complete'):
        report.load_scan(tmp_path,allow_partial=True,enforce_production_grid=False)

def test_exclusions_cannot_carry_fabricated_error_scores(tmp_path):
    snapshot(tmp_path,[(1,-1,'outside_search_region')])
    q=pd.read_csv(tmp_path/'poses.csv');q['rmse_ohm']=0;q.to_csv(tmp_path/'poses.csv',index=False);manifests(tmp_path)
    with pytest.raises(ValueError,match='must not contain'):
        report.load_scan(tmp_path,allow_partial=True,enforce_production_grid=False)
