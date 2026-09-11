"""Solve the full affine EIDORS/CEM systems exported by run_pipeline.

Uses double precision SPD PARDISO; every conductivity state is factorized.
No interpolation or reduced mesh. Paths are relative to this model project.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write_json(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'output/exploratory/arm_sigma_20260908')
    parser.add_argument('--montages',default=','.join(map(str,range(1,17))))
    parser.add_argument('--cases',default=','.join(map(str,range(1,28))))
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args();out=args.output.resolve()
    montages={int(x) for x in args.montages.split(',')};cases={int(x) for x in args.cases.split(',')}
    assert montages<=set(range(1,17)) and cases<=set(range(1,28))
    for variable in ['MKL_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']:
        os.environ[variable]=str(args.threads)
    deps=out/'python_solver_deps'
    if deps.is_dir():sys.path.insert(0,str(deps))
    handles=[]
    if os.name=='nt' and not os.environ.get('PYPARDISO_MKL_RT'):
        libraries=sorted(deps.rglob('mkl_rt*.dll'))
        if libraries:
            os.environ['PYPARDISO_MKL_RT']=str(libraries[0])
            handles.append(os.add_dll_directory(str(libraries[0].parent)))
    import numpy as np
    import pandas as pd
    import h5py
    import scipy
    import scipy.io as sio
    import scipy.sparse as sp
    from pypardiso import PyPardisoSolver

    identity=json.loads((out/'identity.json').read_text(encoding='utf-8-sig'))
    for source in identity['sources']:
        assert digest(ROOT/source['path'])==source['sha256'],f"Changed source: {source['path']}"
    if 'prepared_fem_sha256' in identity:
        assert digest(out/'prepared_fem.mat')==identity['prepared_fem_sha256']
    study=identity['study'];baseline=np.asarray(study['baseline']);design=np.asarray(study['parameter_matrix'])
    solver_probe=PyPardisoSolver(mtype=2)
    runtime=Path(solver_probe.libmkl._name)
    provenance={'matlab_identity_sha256':digest(out/'identity.json'),'solver':'Intel oneMKL PARDISO SPD',
        'precision':'float64','threads':args.threads,'python':platform.python_version(),
        'numpy':np.__version__,'scipy':scipy.__version__,'h5py':h5py.__version__,
        'pypardiso':importlib.metadata.version('pypardiso'),
        'mkl_library_name':runtime.name,'mkl_library_sha256':digest(runtime),
        'source_sha256':digest(__file__)}
    # Thread count affects timing, not the declared physical inputs.
    provenance_file=out/'python_identity.json'
    if provenance_file.exists():
        old=json.loads(provenance_file.read_text(encoding='utf-8'))
        assert {k:v for k,v in old.items() if k!='threads'}=={k:v for k,v in provenance.items() if k!='threads'}
    else:write_json(provenance_file,provenance)
    del solver_probe
    with h5py.File(out/'prepared_fem.mat') as f:
        blocks=[]
        for ref in f['blocks'][:].ravel():
            g=f[ref];n=int(g.attrs['MATLAB_sparse'])
            blocks.append(sp.csc_matrix((g['data'][:],g['ir'][:],g['jc'][:]),shape=(n,n)))
        bone=float(f['bone_sigma'][0,0]);ground=int(f['fmdl_mm/gnd_node'][0,0])-1
    free=np.delete(np.arange(n),ground)
    blocks=[b[free,:][:,free].tocsr() for b in blocks]
    soft=blocks[0]+blocks[1];heart=blocks[2];lung=blocks[3];fixed=bone*blocks[4]
    q=np.zeros((n,2));q[[-4,-1],0]=[1,-1];q[[-3,-2],1]=[1,-1];q=q[free]
    print(f'Loaded full FEM: {len(free)} free unknowns.',flush=True)

    for contact_file in sorted(out.glob('in*_contact.mat')):
        data=sio.loadmat(contact_file,squeeze_me=True)
        index=int(data['montage_index'])
        if index not in montages:continue
        montage=contact_file.name.removesuffix('_contact.mat');inner=int(data['inner_mm']);outer=int(data['outer_mm'])
        kind=str(data['electrode_kind']);assert int(data['ground'])-1==ground
        matrix_error=float(data['matrix_error']);assert matrix_error<1e-12
        C=data['C'][free,:][:,free].tocsr()
        result_file=out/f'{montage}_results.csv';fd_file=out/f'{montage}_derivative_qc.csv'
        rows=pd.read_csv(result_file).to_dict('records') if result_file.exists() else []
        done={int(r['case_index']) for r in rows}
        print(f'Montage {index}: {montage}',flush=True)
        solver=PyPardisoSolver(mtype=2)
        # Explicit nested dissection, double precision, iterative refinement.
        for iparm,value in {1:1,2:3,8:3,10:13,18:-1,19:-1}.items():solver.set_iparm(iparm,value)

        def solve(sig,rhs=q):
            K=(sig[0]*soft+sig[1]*heart+sig[2]*lung+fixed+C).tocsr()
            scale=1/np.sqrt(K.diagonal());D=sp.diags(scale)
            A=sp.triu(D@K@D,format='csr');A.sort_indices()
            x=solver.solve(A,np.asfortranarray(scale[:,None]*rhs))
            v=scale[:,None]*x
            residual=np.linalg.norm(K@v-rhs)/np.linalg.norm(rhs)
            assert np.isfinite(v).all() and residual<1e-7,f'Full FEM residual {residual:g}'
            return v,residual

        def derivatives(v):
            return np.array([-v[:,1]@(B@v[:,0]) for B in [soft,heart,lung]])

        for ci in sorted(cases-done):
            started=time.perf_counter();sig=design[ci-1];v,residual=solve(sig)
            z=float(q[:,1]@v[:,0]);zr=float(q[:,0]@v[:,1]);reciprocity=abs(z-zr)/max(abs(z),abs(zr))
            assert reciprocity<1e-8
            d=derivatives(v);elasticity=d*sig/z
            row=dict(montage_index=index,montage=montage,electrode_kind=kind,inner_mm=inner,outer_mm=outer,case_index=ci,
                sigma_soft=sig[0],sigma_heart=sig[1],sigma_lung=sig[2],Z_direct_ohm=z,Z_reciprocal_ohm=zr,
                reciprocity_relative_error=reciprocity,relative_residual=residual,matrix_relative_error=matrix_error,
                dZ_dsigma_soft=d[0],dZ_dsigma_heart=d[1],dZ_dsigma_lung=d[2],
                elasticity_soft=elasticity[0],elasticity_heart=elasticity[1],elasticity_lung=elasticity[2],
                dZ_dsigma_soft_distal=-v[:,1]@(blocks[1]@v[:,0]),dZ_dsigma_soft_torso=-v[:,1]@(blocks[0]@v[:,0]),
                solver_warning='',solver_warning_message='',elapsed_seconds=time.perf_counter()-started)
            rows.append(row)
            temp=result_file.with_suffix('.tmp');pd.DataFrame(rows).to_csv(temp,index=False);temp.replace(result_file)
            print(f'{montage} case {ci}/27: Z={z:.9g}, Eheart={elasticity[1]:.5g}, residual={residual:.2g}, {row["elapsed_seconds"]:.1f}s',flush=True)
        if inner==20 and outer==80 and not fd_file.exists():
            v,_=solve(baseline);adjoint=derivatives(v);fdrows=[]
            for t in range(3):
                for step in study['finite_difference_relative_steps']:
                    h=baseline[t]*step;zfd=[]
                    for sign in [-1,1]:
                        sig=baseline.copy();sig[t]+=sign*h;w,_=solve(sig,q[:,:1]);zfd.append(q[:,1]@w[:,0])
                    fd=(zfd[1]-zfd[0])/(2*h);error=abs(fd-adjoint[t])/max(abs(adjoint[t]),1e-10)
                    assert error<.002,f'Derivative check failed: {error:g}'
                    fdrows.append(dict(tissue_index=t+1,relative_step=step,adjoint=adjoint[t],finite_difference=fd,relative_error=error))
                    print(f'Derivative tissue {t+1}, step {step}: relative error {error:.3g}',flush=True)
            pd.DataFrame(fdrows).to_csv(fd_file,index=False)
        solver.free_memory(everything=True)
        del solver,C
    result_files=list(out.glob('in*_results.csv'))
    full=pd.concat([pd.read_csv(p) for p in result_files],ignore_index=True) if result_files else pd.DataFrame()
    complete=len(full)==432 and full.montage.nunique()==16 and len(list(out.glob('*_derivative_qc.csv')))==4
    write_json(out/'completion.json',{'status':'matrix_completed_numerical_checks_passed' if complete else 'partial_batch',
        'states':len(full),'montages':int(full.montage.nunique()) if len(full) else 0,
        'research_status':'exploratory_hypothesis_not_validated'})

if __name__=='__main__':main()
