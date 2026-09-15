"""Replay the figure-2 discretization checks without replacing original results."""
import argparse,json,time
import numpy as np
import pandas as pd
from halfspace_study import OUT,electrodes
from halfspace_25d import ExtrudedBEM,load_contours
from halfspace_bem import InclusionBEM

def compare_and_save(name,frame,keys):
    old=pd.read_csv(OUT/(name+'.csv'))
    cols=['f','df_dlogq']
    a=frame.set_index(keys)[cols].sort_index()
    b=old.set_index(keys)[cols].sort_index()
    if not a.index.equals(b.index):raise AssertionError(f'{name}: different cases')
    if not np.allclose(a,b,rtol=2e-8,atol=2e-10):raise AssertionError(f'{name}: numerical mismatch')
    frame.to_csv(OUT/(name+'_recheck.csv'),index=False)
    return {'file':name,'rows':len(frame),'max_abs_f_difference':float(abs(a.f-b.f).max()),
            'max_abs_derivative_difference':float(abs(a.df_dlogq-b.df_dlogq).max())}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['fourier','surface','all'],default='all',nargs='?')
    args=ap.parse_args();sizes,points=electrodes();report=[]
    if args.mode in ['fourier','all']:
        contours=load_contours(OUT/'geometry_contours.npz')
        records=[];controls=[];base={}
        for n,nk in [(120,32),(240,48),(360,64),(360,32),(480,64)]:
            bem=ExtrudedBEM(contours,points,n_segments=n,n_k=nk)
            print(f'Fourier {n}/{nk}',flush=True)
            qs=[1,4,12] if (n,nk) in [(120,32),(240,48),(360,64)] else [4,12]
            for q in qs:
                r=bem.solve(q)
                for i,L in enumerate(sizes):
                    row=dict(segments=n,k_nodes=nk,ratio=q,L_mm=L,f=r['f'][i],
                             df_dlogq=r['df_dlogq'][i],reciprocity_abs=r['reciprocity_abs'][i],
                             assembly_seconds=bem.assembly_seconds,solve_seconds=r['seconds'])
                    if (n,nk)==(360,64):base[(q,L)]=row
                    if (n,nk) in [(120,32),(240,48),(360,64)]:records.append(row)
                    else:
                        row['control']='k_quadrature_at_fixed_segments' if n==360 else 'segment_refinement_at_fixed_k'
                        row['segments_requested']=n
                        controls.append(row)
            del bem
        report.append(compare_and_save('fourier_convergence',pd.DataFrame(records),['segments','k_nodes','ratio','L_mm']))
        report.append(compare_and_save('fourier_independent_qc',pd.DataFrame(controls),['control','segments_requested','k_nodes','ratio','L_mm']))
    if args.mode in ['surface','all']:
        records=[]
        for tol in [.25,.15]:
            z=np.load(OUT/f'm4h_field{int(tol*100):03d}_surface.npz')
            start=time.perf_counter();bem=InclusionBEM(z['vertices'],z['faces'],points);prep=time.perf_counter()-start
            print(f'Surface {tol}: {len(z["faces"])} triangles',flush=True)
            for q in [4,12]:
                r=bem.solve(q)
                for i,L in enumerate(sizes):
                    records.append(dict(tolerance=tol,triangles=len(z['faces']),ratio=q,L_mm=L,
                        f=r['f'][i],df_dlogq=r['df_dlogq'][i],reciprocity_abs=r['reciprocity_abs'][i],
                        assembly_seconds=prep,solve_seconds=r['seconds']))
            del bem
        report.append(compare_and_save('field_resolution_convergence',pd.DataFrame(records),['tolerance','ratio','L_mm']))
    (OUT/f'convergence_reproduction_{args.mode}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
