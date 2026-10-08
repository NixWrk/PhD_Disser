"""Independent 4096-point check of frozen 1024-point material fractions."""
from pathlib import Path
import sys
import nibabel as nib
import numpy as np
from scipy.io import loadmat,savemat
from scipy.stats import qmc
from threadpoolctl import threadpool_limits
import scorpion_georg_fem_prepare as common


def run(out=common.OUT):
    out=Path(out);plan=common.read(out/'plan.local.json')
    for entry in plan['sources']:
        if common.sha(entry['path'])!=entry['sha256']:raise ValueError('Frozen FEM input changed')
    m=loadmat(out/'mesh.mat');nodes=m['nodes_mm'];tets=m['elems'].astype(int)-1
    original=loadmat(out/'fractions_q1024.mat')['fractions']
    seg=out/'segmentation';ct=nib.load(seg/'static_ct.nii.gz');labels=np.zeros(ct.shape,dtype=np.uint8)
    for name,label in [('fat',4),('heart',3),('bone',2),('lung',1)]:
        mask=nib.load(seg/(name+'.nii.gz'));np.testing.assert_allclose(mask.affine,ct.affine,atol=1e-5,rtol=0)
        labels[np.asanyarray(mask.dataobj)>0]=label
    vox=nib.affines.apply_affine(np.linalg.inv(ct.affine)@np.diag([-1.,-1.,1.,1.]),nodes)
    cut=np.sort(qmc.Sobol(3,scramble=True,seed=20261008).random_base2(12),axis=1)
    bary=np.diff(np.c_[np.zeros(4096),cut,np.ones(4096)],axis=1)
    fractions=np.empty_like(original)
    with threadpool_limits(3):
        for start in range(0,len(tets),256):
            stop=min(start+256,len(tets));xyz=np.einsum('qj,bjk->bqk',bary,vox[tets[start:stop]],optimize=True)
            ijk=np.floor(xyz+.5).astype(int);valid=np.all((ijk>=0)&(ijk<np.asarray(labels.shape)),axis=2)
            sampled=np.zeros(valid.shape,dtype=np.uint8);index=ijk[valid]
            sampled[valid]=labels[index[:,0],index[:,1],index[:,2]]
            for j in range(5):
                if not np.array_equal(np.mean(sampled[:,:1024]==j,axis=1),original[start:stop,j]):
                    raise ValueError('Independent implementation did not reproduce frozen fractions')
                fractions[start:stop,j]=np.mean(sampled==j,axis=1)
            if start%102400==0:print('4096-point material check',stop,'/',len(tets),flush=True)
    assert np.array_equal(fractions.sum(1),np.ones(len(tets)))
    target=out/'fractions_q4096.mat';savemat(target,dict(fractions=fractions),do_compression=False)
    common.write(out/'fraction_refinement.local.json',dict(plan_sha256=common.sha(out/'plan.local.json'),
        original_rule=1024,refined_rule=4096,original_fractions_reproduced_exactly=True,
        refined_fractions_sha256=common.sha(target),code_sha256=common.sha(__file__),
        ct_sha256=common.sha(seg/'static_ct.nii.gz'),global_spatial_convergence=False))


if __name__=='__main__':run(Path(sys.argv[1]) if len(sys.argv)>1 else common.OUT)
