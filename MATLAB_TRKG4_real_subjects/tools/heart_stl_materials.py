"""Joint CT/STL material quadrature for existing-mesh heart comparisons.

Preparation only; solves go through run_pipeline('heart_stl_electrical_check').
Local JSON supplies every patient path. No raw data are copied into the code.
Outside the changed heart neighbourhood the existing centroid baseline is kept;
this is explicitly a local quadrature check, not global mesh convergence.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh
from scipy.io import loadmat, savemat
from scipy.stats import qmc

from heart_stl_approximation import contains_points, load_nominal
from heart_stl_tetrahedra import classify_tetrahedra
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from ct_fem_core import lps_to_voxel, SEED


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False)+'\n', encoding='utf-8')


def scaled(candidate, factor):
    c = copy.deepcopy(candidate)
    scale = float(factor)**(1/3)
    if c['sampling_family'] == 'stl':
        c['scale'] *= scale
    else:
        c['semi_axes_mm'] = (np.array(c['semi_axes_mm'])*scale).tolist()
    return c


def bounds(candidate, original_bounds):
    centre = np.asarray(candidate['center_mm'])
    if candidate['sampling_family'] == 'stl':
        return centre + (original_bounds-np.array(candidate['reference_centroid_mm']))*candidate['scale']
    extents = np.sqrt(np.sum((np.asarray(candidate['axes_matrix'])*
                             np.array(candidate['semi_axes_mm']))**2, axis=1))
    return np.array([centre-extents, centre+extents])


def main(config_file):
    cfg = json.loads(Path(config_file).read_text(encoding='utf-8-sig'))
    out = Path(cfg['output_dir']); out.mkdir(parents=True, exist_ok=True)
    assert not (out/'materials_plan.local.json').exists(), 'Use a fresh output'
    candidates = load_nominal(cfg['candidates_file'])
    mesh = loadmat(cfg['mesh_path'])
    nodes = mesh['nodes_mm']; elems = mesh['elems'].astype(np.int64)-1
    original = loadmat(cfg['baseline_materials_file'])
    prep=json.loads(Path(cfg['baseline_preparation']).read_text(encoding='utf-8-sig'))
    assert prep['prepared_materials_sha256']==sha(cfg['baseline_materials_file'])
    mesh_source=[v for k,v in prep['sources'].items()
                 if Path(k).resolve()==Path(cfg['mesh_path']).resolve()]
    assert len(mesh_source)==1 and mesh_source[0]==sha(cfg['mesh_path'])
    labels = original['tissue_id'].ravel().astype(np.uint8)
    volumes = original['volume_mm3'].ravel()/1000
    assert len(labels)==len(elems) and np.all(volumes>0) and set(labels)=={1,2,3,4,5}
    raw = trimesh.load_mesh(cfg['stl_path'], process=False)
    assert sha(cfg['stl_path']) == cfg['stl_sha256']
    stl_bounds = raw.bounds
    baseline = labels.copy(); baseline[baseline==4] = 1
    variants = []
    reference_volume_mm3=float(candidates['STL']['volume_mm3'])
    for job in cfg['variants']:
        # Identical absolute volume increments across unequal nominal fits.
        job=dict(job)
        job['actual_factor']=1+(job['factor']-1)*reference_volume_mm3/float(candidates[job['method']]['volume_mm3'])
        assert job['actual_factor']>0
        c = scaled(candidates[job['method']], job['actual_factor'])
        variants.append((job, c))
    selection=variants
    if cfg.get('selection_variants'):
        selection=[]
        for j in cfg['selection_variants']:
            factor=1+(j['factor']-1)*reference_volume_mm3/float(candidates[j['method']]['volume_mm3'])
            selection.append((j,scaled(candidates[j['method']],factor)))
    boxes = np.array([bounds(c, stl_bounds) for _, c in selection])
    lo, hi = boxes[:,0].min(0), boxes[:,1].max(0)
    for _,c in variants:
        box=bounds(c,stl_bounds)
        assert np.all(box[0]>=lo-1e-8) and np.all(box[1]<=hi+1e-8), 'Candidate outside integration neighbourhood'
    selected = []
    for start in range(0,len(elems),65536):
        v = nodes[elems[start:start+65536]]
        keep = np.all(v.max(1)>=lo,axis=1)&np.all(v.min(1)<=hi,axis=1)
        selected.append(start+np.flatnonzero(keep))
    ids = np.concatenate(selected)
    assert len(ids)>0
    background_box=cfg.get('background_quadrature_box_mm')
    if background_box is not None:
        background_box=np.asarray(background_box,dtype=float)
        assert background_box.shape==(2,3) and np.all(np.isfinite(background_box))
        assert np.all(background_box[1]>background_box[0])
    # A conservative AABB selection is used, never a centroid-only boundary filter.
    masks = {}
    sources = {str(Path(config_file)):sha(config_file)}
    for key in ['mesh_path','stl_path','baseline_materials_file','baseline_preparation','candidates_file']:
        sources[cfg[key]]=sha(cfg[key])
    affine = None
    for name in ['fat','bone','lung']:
        path=cfg['mask_paths'][name]; im=nib.load(path)
        if affine is not None:
            assert np.array_equal(affine, im.affine) and shape==im.shape
        affine=im.affine; shape=im.shape
        masks[name]=np.asarray(im.dataobj)>0; sources[path]=sha(path)
    levels=sorted(set(v['q'] for v,_ in variants))
    assert all(q>0 and q&(q-1)==0 for q in levels)
    cuts=np.sort(qmc.Sobol(3,scramble=True,seed=SEED).random_base2(int(np.log2(max(levels)))),axis=1)
    bary=np.diff(np.c_[np.zeros(len(cuts)),cuts,np.ones(len(cuts))],axis=1)
    plan={'schema':'heart_stl_materials_v1','mesh_sha256':sha(cfg['mesh_path']),
          'stl_sha256':cfg['stl_sha256'],'sources':sources,'implementation_sha256':sha(__file__),
          'geometry_implementation_sha256':sha(Path(__file__).with_name('heart_stl_approximation.py')),
          'shortcut_implementation_sha256':sha(Path(__file__).with_name('heart_stl_tetrahedra.py')),
          'names':['soft_background','lung','bone','heart','fat'],
          'heart_priority':'heart replaces background lung/bone/fat where its surface overlaps; removed original heart becomes soft',
          'outside_changed_neighbourhood':'previous five-tissue centroid assignment',
          'global_material_quadrature_converged':False,'selected_tetrahedra':int(len(ids)),
          'quadrature_rule':'nested sorted-Sobol tetrahedral uniform points', 'seed':SEED,
          'probe_volume_basis':'same absolute increment for all shapes, fraction of original STL volume',
          'reference_STL_volume_ml':reference_volume_mm3/1000,
          'variants':[]}
    if background_box is not None:
        plan['background_quadrature_box_mm']=background_box.tolist()
        plan['outside_background_quadrature_box']='frozen baseline centroid material, even inside new heart probe envelope'
    # Per-level point/background blocks are shared by candidates; geometry and
    # tissue fractions are evaluated jointly at exactly the same points.
    for q in levels:
        active=[(v,c) for v,c in variants if v['q']==q]
        arrays={v['id']:np.zeros((len(ids),5),np.float32) for v,_ in active}
        certified={}
        for job,c in active:
            if c['sampling_family']=='stl':
                certified[job['id']]=classify_tetrahedra(c,nodes[elems[ids]])
                print(f'STL_CERTIFIED {job["id"]} boundary_tets={np.count_nonzero(certified[job["id"]]==0)}',flush=True)
        started=time.monotonic()
        for start in range(0,len(ids),1024):
            chunk=ids[start:start+1024]
            points=np.einsum('qi,eij->eqj',bary[:q],nodes[elems[chunk]]).reshape(-1,3)
            ijk=np.floor(lps_to_voxel(points,affine)+.5).astype(np.int32)
            valid=np.all((ijk>=0)&(ijk<np.array(shape)),axis=1)
            bg=np.ones(len(points),np.uint8)
            for name,label in [('fat',5),('bone',3),('lung',2)]:
                hit=np.zeros(len(points),bool)
                hit[valid]=masks[name][tuple(ijk[valid].T)]
                bg[hit]=label
            if background_box is not None:
                vertices=nodes[elems[chunk]]
                in_old_box=np.all(vertices.max(1)>=background_box[0],axis=1)&np.all(vertices.min(1)<=background_box[1],axis=1)
                frozen=np.repeat(~in_old_box,q)
                bg[frozen]=np.repeat(baseline[chunk],q)[frozen]
            for job,candidate in active:
                lab=bg.copy()
                if job['id'] in certified:
                    state=certified[job['id']][start:start+len(chunk)]
                    uncertain=np.repeat(state==0,q)
                    inside=np.repeat(state==1,q)
                    inside[uncertain]=contains_points(candidate,points[uncertain])
                else:
                    inside=contains_points(candidate,points)
                lab[inside]=4
                lab=lab.reshape(len(chunk),q)
                arrays[job['id']][start:start+len(chunk)]=np.stack([(lab==k).mean(1) for k in range(1,6)],axis=1)
            if start%16384==0:
                print(f'QUADRATURE q={q} tetrahedra={start}/{len(ids)} seconds={time.monotonic()-started:.1f}',flush=True)
        for job,candidate in active:
            fractions=np.eye(5,dtype=np.float32)[baseline-1]
            fractions[ids]=arrays[job['id']]
            assert np.all(fractions>=0) and np.all(fractions.sum(1)==1)
            volume_ml=(volumes@fractions).tolist()
            target=float(candidates[job['method']]['volume_mm3'])/1000*job['actual_factor']
            file=out/(job['id']+'.mat')
            assert not file.exists()
            savemat(file,{'fractions':fractions},do_compression=True)
            plan['variants'].append({'id':job['id'],'method':job['method'],
                'volume_factor':job['actual_factor'],'reference_volume_step_fraction':job['factor']-1,
                'quadrature':q,'volume_ml':volume_ml,
                'target_volume_ml':target,'material_file':file.name,'material_sha256':sha(file),
                'heart_volume_error_pct':100*(volume_ml[3]/target-1)})
            write(out/'preparation_progress.local.json',plan)
            print(f'MATERIAL_READY {job["id"]} heart_ml={volume_ml[3]:.6f}',flush=True)
            del fractions
        del arrays
    write(out/'materials_plan.local.json',plan)
    print(json.dumps({'status':'prepared','variants':len(plan['variants']),
                      'selected_tetrahedra':len(ids)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('config'); main(p.parse_args().config)
