"""Map full-static TotalSegmentator masks into the unchanged lateral FEM mesh.

P1 element gradients are constant. Material fractions therefore integrate the
piecewise scalar conductivity in the element stiffness, without inventing a
new mesh or an effective resistivity mixing law.
"""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.io import loadmat, savemat
from scipy.stats import qmc
from lateral_literature_forward import PIPE, BASE, LIT, sha, read, write, source
import refine_surface_contact_mesh as refine
refine.np = np
volumes = refine.volumes

OUT=PIPE/'output/exploratory/lateral_fat_20260916'
PREVIOUS=PIPE/'output/exploratory/lateral_tissue_ablation_20260916'
GEO=PIPE/'output/exploratory/transverse_consistency_20260914'


def sample_labels(grid, points):
    ijk=np.floor(points+.5).astype(np.int32)
    valid=np.all((ijk>=0)&(ijk<np.asarray(grid.shape)),axis=-1)
    result=np.zeros(valid.shape,dtype=np.uint8)
    ind=ijk[valid];result[valid]=grid[ind[:,0],ind[:,1],ind[:,2]]
    return result,valid


def prepare(out):
    out=Path(out);assert not (out/'plan.json').exists(),'Use a fresh output directory'
    seg=out/'segmentation';meta=read(seg/'segmentation.json');conv=read(seg/'conversion.json')
    assert meta['complete'] and meta['version'].startswith('2.')
    assert sha(seg/'static_ct.nii.gz')==meta['ct_sha256']==conv['ct_sha256']
    ct=nib.load(seg/'static_ct.nii.gz');grid=np.zeros(ct.shape,dtype=np.uint8)
    overlaps=[];paths=[]
    # Assign lower priority first: preserve lung > bone > heart > fat > soft.
    for name,label in [('fat',4),('heart',3),('bone',2),('lung',1)]:
        path=seg/(name+'.nii.gz');assert sha(path)==meta['products'][name]['sha256']
        im=nib.load(path);assert im.shape==ct.shape
        np.testing.assert_allclose(im.affine,ct.affine,rtol=0,atol=1e-5)
        mask=np.asanyarray(im.dataobj)>0
        overlaps.append(dict(tissue=name,voxels=int(mask.sum()),overlap_lower_priority_voxels=int(np.count_nonzero(mask&(grid>0)))))
        grid[mask]=label;paths.append(path)
    pd.DataFrame(overlaps).to_csv(out/'mask_overlap.csv',index=False)
    M=loadmat(BASE/'refined_mesh.mat',simplify_cells=True)
    nodes=np.asarray(M['nodes_mm']);tets=np.asarray(M['elems'],int)-1
    # FEM is in DICOM LPS mm; conversion used only the standard LPS/RAS axis flip.
    transform=np.linalg.inv(ct.affine)@np.diag([-1.,-1.,1.,1.])
    vox=nib.affines.apply_affine(transform,nodes)
    tetvox=vox[tets];v=volumes(nodes,tets);assert np.all(v>0)
    centre=tetvox.mean(axis=1);_,inside=sample_labels(grid,centre)
    ev=nib.affines.apply_affine(transform,np.asarray(M['electrode_centres_mm']))
    _,electrode_inside=sample_labels(grid,ev);assert electrode_inside.all(),'CT does not cover every electrode'
    # Nested, reproducible uniform tetrahedral quadrature from sorted Sobol coordinates.
    points=qmc.Sobol(3,scramble=True,seed=20260916).random_base2(10)
    cut=np.sort(points,axis=1);bary=np.diff(np.column_stack([np.zeros(1024),cut,np.ones(1024)]),axis=1)
    assert np.all(bary>=0) and np.allclose(bary.sum(axis=1),1)
    fractions={q:np.zeros((len(tets),5),dtype=np.float64) for q in [256,1024]}
    for start in range(0,len(tets),2048):
        stop=min(start+2048,len(tets))
        xyz=np.einsum('qj,bjk->bqk',bary,tetvox[start:stop],optimize=True)
        labels,_=sample_labels(grid,xyz)
        for q in fractions:
            for label in range(5):fractions[q][start:stop,label]=np.mean(labels[:,:q]==label,axis=1)
        if start%102400==0:print('Mapped elements',stop,'/',len(tets),flush=True)
    inventory=[]
    for q,f in fractions.items():
        assert np.all(f>=0) and np.array_equal(f.sum(axis=1),np.ones(len(f)))
        savemat(out/f'fractions_q{q}.mat',{'fractions':f},do_compression=False)
        for j,name in enumerate(['soft_background','lung','bone','heart','fat']):
            inventory.append(dict(quadrature_points=q,tissue=name,volume_ml=float(v@f[:,j]/1000),nonzero_elements=int(np.count_nonzero(f[:,j]))))
    pd.DataFrame(inventory).to_csv(out/'material_inventory.csv',index=False)
    selection=read(LIT/'literature_selection.json');rawname,rawhash=next(iter(selection['source_files'].items()))
    raw=PIPE/rawname.replace('\\','/');assert sha(raw)==rawhash
    required={'Muscle','Lung (Inflated)','Lung (Deflated)','Bone (Cortical)','Heart Muscle','Fat'};values=[]
    for line,row in enumerate(csv.reader(raw.read_text(encoding='cp1252').splitlines(),delimiter='\t'),1):
        if len(row)<48 or row[1] not in required:continue
        for f in [50000,100000]:
            d=np.array([float(row[i]) for i in [28,31,35,38]])
            t=np.array([float(row[i]) for i in [29,32,36,39]])*[1e-12,1e-9,1e-6,1e-3]
            a=np.array([float(row[i]) for i in [30,33,37,40]])
            w=2*np.pi*f;sigma=float(row[34])-w*8.8542e-12*np.sum(d/(1+(1j*w*t)**(1-a))).imag
            values.append(dict(tissue=row[1],frequency_hz=f,conductivity_S_m=sigma,rho_ohm_m=1/sigma,source_line=line))
    lit=pd.DataFrame(values);assert len(lit)==12;lit.to_csv(out/'literature_values.csv',index=False)
    requests=[]
    for q in [256,1024]:
        for f in [50000,100000]:
            a=lit[lit.frequency_hz==f].set_index('tissue').rho_ohm_m
            for state,tissue in [('inhale','Lung (Inflated)'),('exhale','Lung (Deflated)')]:
                for model in ['ts_without_fat','ts_with_fat']:
                    requests.append(dict(request_id=f'q{q}_{f}_{state}_{model}',kind='literature',quadrature_points=q,
                        frequency_hz=f,state=state,model=model,rho1=a['Muscle'],rho2=a[tissue],rho_bone=a['Bone (Cortical)'],
                        rho_heart=a['Heart Muscle'],rho_fat=a['Fat']))
    for model in ['ts_without_fat','ts_with_fat','homogeneous']:
        requests.append(dict(request_id='equal_'+model,kind='equal_material_limit',quadrature_points=1024,
            frequency_hz=50000,state='control',model=model,rho1=4.,rho2=4.,rho_bone=4.,rho_heart=4.,rho_fat=4.))
    # Same geometry, nontrivial organ contrast: rho_fat = rho_soft must erase only fat.
    for model in ['ts_without_fat','ts_with_fat']:
        requests.append(dict(request_id='fat_equal_'+model,kind='fat_equal_limit',quadrature_points=1024,
            frequency_hz=50000,state='control',model=model,rho1=3.,rho2=10.,rho_bone=48.,rho_heart=5.,rho_fat=3.))
    pd.DataFrame(requests).to_csv(out/'requests.csv',index=False)
    bp=read(BASE/'plan.json');audit=PIPE/'output/exploratory/nix_cardiac_cycle_20260916/coordinate_audit.json'
    assert conv['frame_sha256'] in read(audit)['dicom_series']['301']['frame_hashes']
    inputs=[BASE/'plan.json',BASE/'prepared_fem.mat',BASE/'refined_mesh.mat',BASE/'contact_qc.csv',
        GEO/'input_contract.json',PREVIOUS/'comparison.csv',PREVIOUS/'qc.json',audit,raw,
        PIPE/bp['observations_source'].replace('\\','/'),seg/'conversion.json',seg/'segmentation.json',*paths]
    write(out/'plan.json',dict(complete_preparation=True,status='exploratory_not_validated',candidate='C01',
        purpose='forward comparison only; full static TotalSegmentator v2 organs with and without fat',
        sizes_mm=bp['sizes_mm'],diameter_mm=5,contact_ohm_m2=bp['contact_ohm_m2'],contacts_per_measurement=4,
        sources=[source(p) for p in inputs],requests_sha256=sha(out/'requests.csv'),
        fractions_sha256={f'fractions_q{q}.mat':sha(out/f'fractions_q{q}.mat') for q in fractions},
        preparation_script_sha256=sha(__file__),ras_to_fem_mm=np.diag([-1.,-1.,1.,1.]).tolist(),
        segmentation_version=meta['version'],priority=['lung','bone','heart','fat','soft_background'],
        quadrature=dict(points=[256,1024],main_points=1024,method='nested scrambled Sobol order-statistic tetrahedral coordinates',seed=20260916,
            rule='element sigma is integral mean using material fractions; P1 gradients are constant'),
        geometry_checks=dict(unchanged_nodes_elements_boundary_and_contacts=True,all_electrodes_in_ct=True,
            fem_volume_ml=float(v.sum()/1000),element_centroid_outside_ct_volume_ml=float(v[~inside].sum()/1000),
            ct_outside_rule='retain soft background in extrapolated body/arm regions outside acquired CT; quantify separately'),
        literature=dict(doi=selection['doi'],fat='Fat; same Gabriel coefficients as SAT (Subcutaneous Fat) in this version'),
        limitations=['Automatic masks are research inputs, not a manually validated anatomy.',
            'Whole heart = Heart Muscle; whole bone = cortical bone; remainder = Muscle.',
            'Subcutaneous and torso fat share the Fat conductivity; no HU-based conductivity fitting.',
            'Inspiratory CT fixed for both electrical states; deflated not established as volunteer exhalation.',
            'Real scalar conductivity; fixed contact impedance; no inverse fit or new localization.',
            'Quadrature agreement does not establish global mesh convergence.']))
    # One diagnostic plane is exactly the accepted electrode s-d plane, t=0.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from scipy.ndimage import map_coordinates
    c=read(GEO/'input_contract.json');s=np.linspace(-100,100,650);d=np.linspace(-10,135,470)
    ss,dd=np.meshgrid(s,d);xyz=np.asarray(c['centre_mm'])+ss[...,None]*np.asarray(c['axis'])+dd[...,None]*np.asarray(c['inward'])
    vi=nib.affines.apply_affine(transform,xyz);lab,valid=sample_labels(grid,vi)
    hu=map_coordinates(np.asanyarray(ct.dataobj).astype(float),vi.reshape(-1,3).T,order=1,mode='constant',cval=-1000).reshape(ss.shape)
    fig,ax=plt.subplots(figsize=(10,6));ax.imshow(hu,extent=[s.min(),s.max(),d.max(),d.min()],cmap='gray',vmin=-180,vmax=250)
    overlay=np.ma.masked_where((lab==0)|~valid,lab)
    ax.imshow(overlay,extent=[s.min(),s.max(),d.max(),d.min()],cmap=ListedColormap(['#999','#33aaff','#eeeeaa','#e45555','#f2aa18']),vmin=0,vmax=4,alpha=.55,interpolation='nearest')
    ax.set(xlabel='s вдоль сборки, мм',ylabel='d внутрь от центра монтажа, мм',title='Полная статическая КТ: плоскость электродной сборки, t = 0')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=color,label=label) for color,label in [('#33aaff','Лёгкое'),('#eeeeaa','Кость'),('#e45555','Сердце'),('#f2aa18','Жир')]],loc='lower left')
    fig.tight_layout();fig.savefig(out/'segmentation_plane.png',dpi=150);plt.close(fig)
    print(pd.DataFrame(inventory).to_string(index=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=OUT)
    prepare(ap.parse_args().out)
