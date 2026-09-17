"""Add static CT fat to the sealed cardiac-cycle geometry, using joint sampling.

Original lung/bone labels remain fixed. Fat partitions only the soft background;
the observed heart overrides it pointwise. No product of independent fractions.
"""
import argparse
import itertools
import time
from pathlib import Path
import heart_replacement_electrical_pilot as core


def voxel_box_classes(vertices, inverse, mask, integral):
    import numpy as np
    ijk = vertices @ inverse[:3, :3].T + inverse[:3, 3]
    low = np.floor(ijk.min(axis=1)+.5).astype(np.int64)
    high = np.floor(ijk.max(axis=1)+.5).astype(np.int64)+1
    shape = np.asarray(mask.shape)
    wholly_inside = ((low >= 0) & (high <= shape)).all(axis=1)
    lo = np.clip(low, 0, shape); hi = np.clip(high, 0, shape)
    count = np.zeros(len(vertices), dtype=np.int64)
    for bits in itertools.product((0, 1), repeat=3):
        q = np.column_stack([hi[:, k] if bits[k] else lo[:, k] for k in range(3)])
        count += (-1)**(3-sum(bits))*integral[q[:, 0], q[:, 1], q[:, 2]].astype(np.int64)
    volume = np.prod(hi-lo, axis=1)
    return count == 0, wholly_inside & (volume > 0) & (count == volume)


def prepare(config):
    cfg = core.read(config); runner = core.configure(Path(cfg['deps']), 1)
    import numpy as np
    import nibabel as nib
    from heart_fractional_materials import uniform_tet_barycentric
    from prepare_cardiac_cycle_materials import sample_mask
    out = Path(cfg['output']); out.mkdir(parents=True, exist_ok=True)
    core.require(not (out/'geometry.json').exists(), 'Use a new output directory')
    old = Path(cfg['old_run']); seg = Path(cfg['fat_segmentation'])
    geometry = core.read(old/'materials/geometry.json')
    source = Path(geometry['source_prepared'])
    core.require(core.sha(source) == geometry['source_prepared_sha256'], 'Mesh changed')
    meta = core.read(seg/'segmentation.json'); conv = core.read(seg/'conversion.json')
    audit = core.read(old/'coordinate_audit.json')
    core.require(meta['complete'] and meta['version'].startswith('2.'), 'Incomplete segmentation')
    core.require(conv['frame_sha256'] in audit['dicom_series']['301']['frame_hashes'], 'Static CT frame mismatch')
    core.require(core.sha(seg/'fat.nii.gz') == meta['products']['fat']['sha256'], 'Fat mask changed')
    core.require(core.sha(seg/'static_ct.nii.gz') == meta['ct_sha256'] == conv['ct_sha256'], 'Static CT changed')
    image = nib.load(seg/'fat.nii.gz'); ct = nib.load(seg/'static_ct.nii.gz')
    np.testing.assert_allclose(image.affine, ct.affine, atol=1e-5, rtol=0)
    core.require(image.shape == ct.shape, 'CT/fat grid mismatch')
    mask = np.asanyarray(image.dataobj)>0
    transform = np.asarray(geometry['ras_mm_to_fem_mm'])
    np.testing.assert_allclose(transform, conv['ras_to_fem_mm'], atol=1e-12, rtol=0)
    affine = transform @ image.affine; affine[:3] *= .001
    inverse = np.linalg.inv(affine)
    # Integral image gives a conservative voxel-cell box test for each tetrahedron.
    integral = np.pad(mask.astype(np.uint32), ((1,0),)*3)
    for axis in range(3): np.cumsum(integral, axis=axis, dtype=np.uint32, out=integral)
    with runner.h5py.File(source) as f:
        nodes = np.asarray(f['fmdl_m/nodes']).T
        elements = np.asarray(f['fmdl_m/elems']).T.astype(np.int64)-1
        original = np.asarray(f['tissue_id']).ravel().astype(np.uint8)
    n = len(elements); background = original.copy(); background[background == 3] = 1
    old_background = runner.sio.loadmat(old/'materials/background.mat')['background'].ravel()
    core.require(np.array_equal(background, old_background), 'Background changed')
    levels = [4096,16384]; bary = uniform_tet_barycentric(levels[-1],seed=20260916)
    fat = np.zeros((2,n),dtype=np.float32); volume = np.empty(n); partials=[]
    outside_ct_volume = 0.; started = time.time()
    for start in range(0,n,50000):
        ids = np.arange(start,min(start+50000,n)); vertices=nodes[elements[ids]]
        volume[ids] = np.abs(np.linalg.det(vertices[:,1:]-vertices[:,:1]))/6*1e6
        centre = vertices.mean(axis=1) @ inverse[:3,:3].T + inverse[:3,3]
        outside = ((centre<-.5)|(centre>np.asarray(mask.shape)-.5)).any(axis=1)
        outside_ct_volume += float(volume[ids[outside]].sum())
        eligible = background[ids]==1
        ids = ids[eligible]; vertices=vertices[eligible]
        empty,full=voxel_box_classes(vertices,inverse,mask,integral)
        fat[:,ids[full]]=1
        partials.extend(ids[~(empty|full)].tolist())
    del integral
    partials=np.asarray(partials,dtype=np.int64)
    print('Partial fat tetrahedra:',len(partials),'of',n,flush=True)
    for start in range(0,len(partials),16):
        ids=partials[start:start+16]
        points=np.einsum('pv,tvc->tpc',bary,nodes[elements[ids]],optimize=True)
        membership=sample_mask(points,mask,inverse)
        for k,q in enumerate(levels):fat[k,ids]=membership[:,:q].mean(axis=1)
        if start%16000==0:print('Fat integration',start,'/',len(partials),flush=True)
    np.savez_compressed(out/'fat_background_quadrature.npz',fractions=fat,levels=levels,element_volume_ml=volume)
    runner.sio.savemat(out/'background.mat',{'background':background[:,None]},do_compression=True)
    states=[]; source_masks={Path(v['path']).parent.name:Path(v['path']) for v in geometry['inputs'] if Path(v['path']).name=='whole_heart.nii.gz'}
    inputs=[Path(config),source,old/'materials/geometry.json',old/'materials/background.mat',old/'coordinate_audit.json',seg/'fat.nii.gz',seg/'static_ct.nii.gz',seg/'segmentation.json',seg/'conversion.json']
    for state in geometry['states']:
        sid=state['id']; path=source_masks[sid]
        core.require(core.sha(path)==state['mask_sha256'],'Phase mask changed')
        im=nib.load(path); hm=np.asanyarray(im.dataobj)>0
        ha=transform @ im.affine; ha[:3]*=.001; hi=np.linalg.inv(ha)
        core.require(core.sha(old/'materials'/state['fraction_mat'])==state['fraction_mat_sha256'],'Old heart fraction changed')
        h=runner.sio.loadmat(old/'materials'/state['fraction_mat'])['heart_fraction'].ravel()
        qfile=old/'materials'/f'{sid}_quadrature.npz'
        with np.load(qfile) as qdata:
            ids=qdata['element_indices_zero_based']; hsmall=qdata['fractions']
        ids=ids[(background[ids]==1)&(fat[-1,ids]>0)&(h[ids]>0)]
        available=fat.copy()
        for start in range(0,len(ids),16):
            selected=ids[start:start+16]
            pts=np.einsum('pv,tvc->tpc',bary,nodes[elements[selected]],optimize=True)
            heart=sample_mask(pts,hm,hi); fm=sample_mask(pts,mask,inverse)
            core.require(np.array_equal(heart.mean(axis=1),h[selected]),'Phase quadrature differs')
            for k,q in enumerate(levels):available[k,selected]-=(heart[:,:q]&fm[:,:q]).mean(axis=1)
        core.require(np.all(available>=0) and np.all(available[-1]<=1-h),'Joint partition invalid')
        mat=out/f'{sid}_fractions.mat'
        runner.sio.savemat(mat,{'heart_fraction':h[:,None],'fat_fraction':available[-1,:,None]},do_compression=True)
        row=dict(state, fraction_mat=mat.name,fraction_mat_sha256=core.sha(mat),
                 fat_volume_ml=float(available[-1]@volume),
                 fat_quadrature_volumes_ml=(available@volume).tolist(),
                 fat_quadrature_weighted_l1_ml=float(abs(available[-1]-available[0])@volume))
        states.append(row);inputs.extend([path,old/'materials'/state['fraction_mat'],qfile])
        print(sid,'fat ml',row['fat_volume_ml'],flush=True)
    code=[Path(__file__),Path(core.__file__),Path(__file__).with_name('prepare_cardiac_cycle_materials.py'),Path(__file__).with_name('heart_fractional_materials.py')]
    result=dict(geometry, schema='cardiac_cycle_with_static_fat_v1',states=states,
        background_mat_sha256=core.sha(out/'background.mat'),
        inputs=[{'path':str(p.resolve()),'sha256':core.sha(p)} for p in inputs],
        implementation=[{'path':str(p.resolve()),'sha256':core.sha(p)} for p in code],
        material_order=['muscle_background','lung','heart_region','bone','fat'],
        fat_source=dict(version=meta['version'],classes=meta['products']['fat']['classes'],mask_sha256=meta['products']['fat']['sha256'],
                        ct_sha256=meta['ct_sha256'],segmented_volume_ml=meta['products']['fat']['volume_ml']),
        fat_background_volume_ml=float(fat[-1]@volume),fat_quadrature_levels=levels,
        fat_background_quadrature_volumes_ml=(fat@volume).tolist(),
        fat_background_quadrature_weighted_l1_ml=float(abs(fat[-1]-fat[0])@volume),
        fem_volume_ml=float(volume.sum()),element_centroid_outside_static_ct_volume_ml=outside_ct_volume,
        background_rule='Preserve original lung/bone. Original heart becomes soft. CT fat splits only soft background. Current observed heart overrides all background pointwise.',
        outside_ct_rule='No fat extrapolation into artificial arms or unobserved anatomy; preserve previous background.',
        integration_seconds=time.time()-started)
    core.write(out/'geometry.json',result)
    print('Preparation complete',time.time()-started,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True,type=Path)
    prepare(parser.parse_args().config)
