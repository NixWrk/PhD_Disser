from pathlib import Path
import os,sys,time,json
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
out=Path('output/exploratory/arm_sigma_20260908')
sys.path.insert(0,str((out/'python_solver_deps').resolve()))
import h5py,numpy as np,scipy.sparse as sp,scipy.io as sio
from scipy.sparse.linalg import cg
import pyamg
with h5py.File(out/'prepared_fem.mat') as f:
 blocks=[]
 for ref in f['blocks'][:].ravel():
  g=f[ref]; n=int(g.attrs['MATLAB_sparse'])
  blocks.append(sp.csc_matrix((g['data'][:],g['ir'][:],g['jc'][:]),shape=(n,n)))
 bone=float(f['bone_sigma'][0,0])
contact=sio.loadmat(out/'solver_contact_benchmark.mat'); C=contact['C']; ground=int(contact['ground'][0,0])-1
identity=json.loads((out/'preparation_identity.json').read_text(encoding='utf-8-sig'));sig=identity['study']['baseline']
K=sig[0]*(blocks[0]+blocks[1])+sig[1]*blocks[2]+sig[2]*blocks[3]+bone*blocks[4]+C
free=np.delete(np.arange(n),ground);K=K[free,:][:,free].tocsr(); q=np.zeros((n,2));q[[-4,-1],0]=[1,-1];q[[-3,-2],1]=[1,-1];b=q[free]
d=1/np.sqrt(K.diagonal());D=sp.diags(d);A=(D@K@D).tocsr();A.sort_indices()
t=time.perf_counter();ml=pyamg.smoothed_aggregation_solver(A,B=(1/d)[:,None],symmetry='symmetric',max_coarse=200)
print('AMG setup seconds',time.perf_counter()-t,flush=True);print(ml,flush=True)
M=ml.aspreconditioner();v=[];rows=[]
for j in range(2):
 history=[];t=time.perf_counter()
 def callback(x):
  history.append(1)
  if len(history)%100==0: print('iteration',len(history),flush=True)
 x,flag=cg(A,d*b[:,j],rtol=1e-11,atol=0,maxiter=2000,M=M,callback=callback)
 v.append(d*x);res=np.linalg.norm(K@v[-1]-b[:,j])/np.linalg.norm(b[:,j]);print('SOLVE',j,flag,len(history),res,time.perf_counter()-t,flush=True)
 rows.append(dict(rhs=j,flag=flag,iterations=len(history),residual=res,seconds=time.perf_counter()-t))
v=np.array(v).T;z=b[:,1]@v[:,0];zr=b[:,0]@v[:,1]; print('Z',z,zr,flush=True)
np.savez(out/'amg_benchmark.npz',v=v,free=free,z=z,zr=zr)
(out/'amg_benchmark.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
