from pathlib import Path
import os,sys,time
out=Path('output/exploratory/arm_sigma_20260908');sys.path.insert(0,str((out/'python_solver_deps').resolve()))
os.environ['MKL_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
lib=next((out/'python_solver_deps').rglob('mkl_rt*.dll'));os.environ['PYPARDISO_MKL_RT']=str(lib.resolve());handle=os.add_dll_directory(str(lib.parent.resolve()))
import numpy as np, scipy.sparse as sp, scipy.io as sio,h5py
from pypardiso import PyPardisoSolver
with h5py.File(out/'prepared_fem.mat') as f:
 blocks=[]
 for ref in f['blocks'][:].ravel():
  g=f[ref];n=int(g.attrs['MATLAB_sparse']);blocks.append(sp.csc_matrix((g['data'][:],g['ir'][:],g['jc'][:]),shape=(n,n)))
 ground=int(f['fmdl_mm/gnd_node'][0,0])-1;bone=float(f['bone_sigma'][0,0])
C=sio.loadmat(out/'in20_out80_outer_planes_inner_rings_contact.mat')['C'];free=np.delete(np.arange(n),ground)
K=(.211498103445482*(blocks[0]+blocks[1])+.195435241710933*blocks[2]+.05744914314603*blocks[3]+bone*blocks[4]+C)[free,:][:,free].tocsr()
q=np.zeros((n,2));q[[-4,-1],0]=[1,-1];q[[-3,-2],1]=[1,-1];b=q[free]
d=1/np.sqrt(K.diagonal());D=sp.diags(d);A=sp.triu(D@K@D,format='csr');A.sort_indices();rhs=np.asfortranarray(d[:,None]*b)
solver=PyPardisoSolver(mtype=2)
for k,v in {1:1,2:3,8:3,10:13,18:-1,19:-1}.items():solver.set_iparm(k,v)
solver._check_A(A);rhs=solver._check_b(A,rhs)
for phase in [11,22,33,23]:
 solver.set_phase(phase);t=time.perf_counter();x=solver._call_pardiso(A,rhs);print('PHASE',phase,'SECONDS',time.perf_counter()-t,flush=True)
 if phase in [33,23]:
  v=d[:,None]*x;print('Z',b[:,1]@v[:,0],'residual',np.linalg.norm(K@v-b)/np.linalg.norm(b),flush=True)
solver.free_memory(everything=True)
