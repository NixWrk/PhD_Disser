"""Reproducible finite-contact and warm-solver timing checks for 20.17."""
import argparse,time,platform,json
import numpy as np
import pandas as pd
from halfspace_study import OUT,SOURCE,electrodes,write,sha
from halfspace_bem import InclusionBEM
from halfspace_cem import finite_contacts
from halfspace_25d import ExtrudedBEM,load_contours

def cem():
    d=pd.read_csv(SOURCE/'contacts.csv');g=json.loads((SOURCE/'input_contract.json').read_text(encoding='utf-8'));sizes,_=electrodes()
    centres=(d[['x_mm','y_mm','z_mm']].to_numpy()-np.array(g['centre_mm']))@np.array(g['frame_columns_s_t_d'])/1000;centres[:,2]=0
    centres=centres.reshape(-1,4,3);areas=d.area_mm2.to_numpy().reshape(-1,4)/1e6
    z=np.load(OUT/'m4h_field025_surface.npz');bem=InclusionBEM(z['vertices'],z['faces']);rows=[]
    for rings in [1,2,3]:
        r,t=finite_contacts(bem,centres,areas,rings=rings)
        print('rings',rings,'seconds',t,flush=True)
        for L,x in zip(sizes,r):rows.append(dict(model='m4h',L_mm=L,rho1=4.,rho2=16.,rings=rings,elapsed_seconds=t,**x))
    pd.DataFrame(rows).to_csv(OUT/'cem_baseline_check.csv',index=False)
    write(OUT/'cem_baseline_identity.json',dict(tissue_state=[4.,16.],primary_inversion_uses='PEM, not these finite-contact results',contact_geometry='Disks on d=0; saved CT/CEM centroids projected; each area conserved, patch shape is changed',contact_ohm_m2=g['contact_ohm_m2'],source='transverse_consistency_20260914/contacts.csv',surface='m4h_field025_surface.npz',solver='constant-panel CEM with inclusion BEM; independent refinement of electrode panels',hashes={p.name:sha(p) for p in [__import__('pathlib').Path(__file__).with_name('halfspace_cem.py'),OUT/'m4h_field025_surface.npz',SOURCE/'contacts.csv']}))

def benchmark():
    sizes,p=electrodes();rows=[]
    for name in ['m3h','m4h']:
        start=time.perf_counter()
        if name=='m3h':b=ExtrudedBEM(load_contours(OUT/'geometry_contours.npz'),p,360,64)
        else:
            s=np.load(OUT/'m4h_field025_surface.npz');b=InclusionBEM(s['vertices'],s['faces'],p)
        preparation=time.perf_counter()-start
        b.solve(4.) # one warm-up excluded, no cached LU
        for repeat in range(5):
            r=b.solve(4.)
            rows.append(dict(model=name,repeat=repeat,seconds=r['seconds'],matrix_preparation_seconds=preparation,scope='9 point-montage Z and both resistivity derivatives, fresh LU per call',threads=6,reciprocity_max_ohm_at_rho4=float(max(r['reciprocity_abs'])*4)))
    pd.DataFrame(rows).to_csv(OUT/'direct_timings.csv',index=False)
    write(OUT/'timing_environment.json',dict(python=platform.python_version(),numpy=np.__version__,system=platform.platform(),processor=platform.processor(),BLAS_threads=6,replicates=5,LU_cached=False,geometry_cached=True,CT_FEM_prepared_matrix_build_time='not measured in this stage; load and assembly of existing blocks recorded separately by MATLAB',warning='Do not infer a clinical speed/accuracy advantage from wall time alone.'))
    print(pd.DataFrame(rows).groupby('model').seconds.agg(['median','min','max']).to_string())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['cem','benchmark']);a=p.parse_args()
    if a.stage=='cem':cem()
    else:benchmark()
