"""Local TotalSegmentator v2 CT masks; no manual/HU fallback or FEM solve."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def infer(config_path, out, tasks):
    cfg = read(config_path)
    runner_digest = sha(__file__)
    runtime = cfg['segmentation']
    # Environment is set before importing torch/nnUNet. All temporary data and
    # the mutable TotalSegmentator configuration belong to the fresh local run.
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    temporary = out / 'temporary'; temporary.mkdir()
    os.environ.update(CUDA_VISIBLE_DEVICES=str(runtime['gpu']),
        TOTALSEG_HOME_DIR=str(out / 'runtime_config'),
        TOTALSEG_WEIGHTS_PATH=str(Path(runtime['weights_directory']).resolve()),
        TEMP=str(temporary), TMP=str(temporary))
    import tempfile
    tempfile.tempdir = str(temporary)
    home = out / 'runtime_config'; home.mkdir()
    private = read(runtime['config_file']) if runtime.get('config_file') else {}
    private.update(send_usage_stats=False, prediction_counter=0)
    write_new(home / 'config.json', private)
    import numpy as np
    import nibabel as nib
    import torch
    from totalsegmentator.python_api import totalsegmentator
    from totalsegmentator.map_to_binary import class_map
    import totalsegmentator as package_module
    version = importlib.metadata.version('TotalSegmentator')
    if version != runtime['version'] or not version.startswith('2.'):
        raise ValueError('The declared TotalSegmentator v2 version is required')
    if not torch.cuda.is_available():
        raise ValueError('Configured GPU is unavailable; no silent CPU fallback')
    ct = Path(cfg['ct']).resolve(); digest = sha(ct); reference = nib.load(ct)
    if reference.header.get_xyzt_units()[0] != 'mm' or len(reference.shape) != 3:
        raise ValueError('Selected static CT must have a 3D native grid in mm')
    if np.prod(reference.shape) > cfg['max_ct_voxels']:
        raise ValueError('CT exceeds the declared voxel limit')
    lungs = [n for n in class_map['total'].values() if n.startswith('lung_') and 'lobe' in n]
    exact = {'sacrum','humerus_left','humerus_right','scapula_left','scapula_right',
             'clavicula_left','clavicula_right','femur_left','femur_right','hip_left','hip_right','skull','sternum'}
    bones = [n for n in class_map['total'].values() if n.startswith(('rib_','vertebrae_')) or n in exact]
    groups = dict(body=['body_trunc','body_extremities'],lung=lungs,bone=bones,
                  heart=['heart'],fat=['subcutaneous_fat','torso_fat'])
    requested = dict(body=groups['body'], total=lungs+bones+['heart'],
                     tissue_types=groups['fat']+['skeletal_muscle'])
    dataset_ids = set()
    for task in tasks:
        dataset_ids.update(dict(body=[299],total=[291,292,293,295,298],tissue_types=[481])[task])
    weights = Path(runtime['weights_directory']); model_files = {}
    for identifier in sorted(dataset_ids):
        found = []
        for directory in weights.glob('Dataset'+str(identifier)+'_*'):
            found.extend(p for p in directory.rglob('*') if p.is_file() and p.suffix in {'.pth','.json','.pkl'})
        if not any(p.suffix == '.pth' for p in found):
            raise ValueError('Required local weights are missing; no automatic download: '+str(identifier))
        model_files.update({str(p.resolve()):sha(p) for p in found})
    package = Path(package_module.__file__).parent
    code_files = {str(p.resolve()):sha(p) for p in package.glob('*.py')}
    started = time.monotonic(); runs = []
    for task in tasks:
        folder = out / ('raw_'+task)
        options = dict(task=task,device='gpu',nr_thr_resamp=1,nr_thr_saving=1,
            fast=False,quiet=False,no_derived_masks=True)
        if task == 'total': options['roi_subset'] = requested[task]
        print('TotalSegmentator v2:', task, flush=True)
        totalsegmentator(ct, folder, **options)
        for label in requested[task]:
            image = nib.load(folder / (label+'.nii.gz'))
            if image.shape != reference.shape or not np.array_equal(image.affine, reference.affine):
                raise ValueError('Segmentator output grid differs from native CT: '+label)
        runs.append(dict(task=task,options=options,classes=requested[task]))
    inventory = {}
    for name, labels in groups.items():
        task = 'body' if name == 'body' else 'tissue_types' if name == 'fat' else 'total'
        if task not in tasks: continue
        mask = np.zeros(reference.shape, dtype=bool); sources = []
        for label in labels:
            path = out / ('raw_'+task) / (label+'.nii.gz')
            data = np.asanyarray(nib.load(path).dataobj)
            if not np.isin(data, [0,1]).all(): raise ValueError('Non-binary source: '+label)
            mask |= data > 0
            sources.append(dict(file=path.relative_to(out).as_posix(),sha256=sha(path)))
        if not mask.any(): raise ValueError('Empty aggregate: '+name)
        image = nib.Nifti1Image(mask.astype(np.uint8),reference.affine); image.header.set_xyzt_units('mm')
        target = out / (name+'.nii.gz'); nib.save(image,target)
        inventory[name] = dict(file=target.name,sha256=sha(target),classes=labels,
            source_masks=sources,voxels=int(mask.sum()))
    if sha(ct) != digest: raise ValueError('CT changed during segmentation')
    if sha(__file__) != runner_digest: raise ValueError('Segmentator runner changed during inference')
    for path, checksum in {**model_files,**code_files}.items():
        if sha(path) != checksum: raise ValueError('Segmentator model/runtime changed during inference')
    write_new(out / 'segmentation.local.json',dict(schema='ct_fem_totalsegmentator_masks_v1',
        complete=True,backend='TotalSegmentator',version=version,ct_sha256=digest,
        runner_sha256=runner_digest,products=inventory,runs=runs,
        model_weights_and_metadata=model_files,totalsegmentator_source=code_files,
        elapsed_seconds=time.monotonic()-started,torch_version=torch.__version__,
        automatic_masks_pending_anatomical_review=True,FEM_was_run=False,
        independent_anatomical_validation=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True); parser.add_argument('--out',required=True)
    parser.add_argument('--tasks',nargs='+',choices=['body','total','tissue_types'],default=['body','total','tissue_types'])
    args = parser.parse_args()
    infer(args.config,args.out,args.tasks)
