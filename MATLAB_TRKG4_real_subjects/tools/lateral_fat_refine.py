"""Increase mask integration after the first 256/1024 comparison failed QC."""
from pathlib import Path
import shutil
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.io import loadmat,savemat
from scipy.stats import qmc
from lateral_fat_prepare import OUT,BASE,PIPE,sample_labels,sha,read,write,source,volumes


def run():
    target=OUT/'refined';target.mkdir(exist_ok=True)
    assert not (target/'plan.json').exists(),'Frozen refined plan exists'
    parent=read(OUT/'plan.json');seg=OUT/'segmentation';ct=nib.load(seg/'static_ct.nii.gz')
    grid=np.zeros(ct.shape,np.uint8)
    for name,label in [('fat',4),('heart',3),('bone',2),('lung',1)]:
        image=nib.load(seg/(name+'.nii.gz'));grid[np.asanyarray(image.dataobj)>0]=label
    m=loadmat(BASE/'refined_mesh.mat',simplify_cells=True);t=np.asarray(m['elems'],int)-1
    transform=np.linalg.inv(ct.affine)@np.diag([-1.,-1.,1.,1.])
    tet=nib.affines.apply_affine(transform,np.asarray(m['nodes_mm']))[t]
    u=np.sort(qmc.Sobol(3,scramble=True,seed=20260916).random_base2(12),axis=1)
    b=np.diff(np.c_[np.zeros(4096),u,np.ones(4096)],axis=1)
    fractions=np.zeros((len(t),5));check1024=np.zeros((len(t),5))
    for start in range(0,len(t),512):
        end=min(start+512,len(t));points=np.einsum('qj,bjk->bqk',b,tet[start:end],optimize=True)
        labels,_=sample_labels(grid,points)
        for j in range(5):
            fractions[start:end,j]=np.mean(labels==j,axis=1)
            check1024[start:end,j]=np.mean(labels[:,:1024]==j,axis=1)
        if start%102400==0:print('4096-point mapping:',end,'/',len(t),flush=True)
    old=loadmat(OUT/'fractions_q1024.mat',simplify_cells=True)['fractions']
    assert np.array_equal(check1024,old),'Nested quadrature changed the earlier points'
    assert np.all(fractions>=0) and np.array_equal(fractions.sum(1),np.ones(len(t)))
    savemat(target/'fractions_q4096.mat',{'fractions':fractions},do_compression=False)
    shutil.copy2(OUT/'fractions_q1024.mat',target/'fractions_q1024.mat')
    for name in ['literature_values.csv','mask_overlap.csv','mask_comparison.csv','geometry_qc.json']:
        shutil.copy2(OUT/name,target/name)
    v=volumes(m['nodes_mm'],t);inventory=pd.read_csv(OUT/'material_inventory.csv').query('quadrature_points==1024')
    added=pd.DataFrame([dict(quadrature_points=4096,tissue=name,volume_ml=float(v@fractions[:,j]/1000),nonzero_elements=int(np.count_nonzero(fractions[:,j]))) for j,name in enumerate(['soft_background','lung','bone','heart','fat'])])
    pd.concat([inventory,added],ignore_index=True).to_csv(target/'material_inventory.csv',index=False)
    q=pd.read_csv(OUT/'requests.csv');q.quadrature_points=q.quadrature_points.map({256:1024,1024:4096})
    q['request_id']=[f'q{x.quadrature_points}_{x.frequency_hz}_{x.state}_{x.model}' if x.kind=='literature' else x.request_id for x in q.itertuples()]
    q.to_csv(target/'requests.csv',index=False)
    parent['sources'] += [source(OUT/'plan.json'),source(OUT/'responses.csv'),source(OUT/'execution.json'),source(OUT/'initial_worker.m')]
    parent['requests_sha256']=sha(target/'requests.csv')
    parent['fractions_sha256']={n:sha(target/n) for n in ['fractions_q1024.mat','fractions_q4096.mat']}
    parent['quadrature'].update(points=[1024,4096],main_points=4096)
    parent['initial_preparation_script_sha256']=parent['preparation_script_sha256']
    parent['preparation_script_sha256']=sha(__file__)
    parent['refinement_reason']='initial 256/1024 comparison exceeded 0.02 ohm + 0.001 abs(Z); max 0.2296144 ohm'
    parent['exact_nested_1024_fraction_reproduction']=True
    write(target/'plan.json',parent);print(added.to_string(index=False),flush=True)


if __name__=='__main__':run()
