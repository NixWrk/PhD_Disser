"""Georg TEPC placement using the same surface picker as Nick; no FEM solve."""
from pathlib import Path
import argparse
import hashlib
import json

import nibabel as nib
import numpy as np
import trimesh
from skimage.measure import marching_cubes

from build_surface_landmark_picker import build_picker_html

ROOT = Path(__file__).resolve().parents[2]
GEOMETRY = ROOT/'MATLAB_TRKG4_real_subjects/output/exploratory/scorpion_georg_geometry_20261009'
SEGMENTATION = ROOT/'MATLAB_TRKG4_real_subjects/output/exploratory/scorpion_georg_fem_20261008/segmentation'
OUT = ROOT/'MATLAB_TRKG4_real_subjects/output/exploratory/georg_tepc_manual_20261009'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def mask_surface(path, target):
    image = nib.load(path)
    values = np.asanyarray(image.dataobj) > 0
    if not values.any():
        raise ValueError(f'Empty anatomical mask: {path.name}')
    vertices, faces, _, _ = marching_cubes(values.astype(np.uint8), level=.5)
    ras = nib.affines.apply_affine(image.affine, vertices)
    lps = ras * [-1., -1., 1.]
    trimesh.Trimesh(lps, faces, process=False).export(target)


def build(geometry=GEOMETRY, segmentation=SEGMENTATION, out=OUT, target_faces=20000):
    geometry, segmentation, out = map(Path, (geometry, segmentation, out))
    g = json.loads((geometry/'geometry.local.json').read_text(encoding='utf-8-sig'))
    body = geometry/'body_embedded_arms_370mm.stl'
    if sha(body) != g['surface_sha256']:
        raise ValueError('Corrected body surface hash mismatch')
    out.mkdir(parents=True, exist_ok=True)
    inputs = {'body': sha(body)}
    references = {}
    ct = nib.load(segmentation/'static_ct.nii.gz')
    for name in ['heart', 'lung', 'bone', 'fat']:
        source = segmentation/(name+'.nii.gz')
        image = nib.load(source)
        if image.shape != ct.shape or not np.allclose(image.affine, ct.affine, atol=1e-5, rtol=0):
            raise ValueError('Mask and full CT coordinate frames differ')
        references[name] = out/(name+'_lps.stl')
        mask_surface(source, references[name])
        inputs[name] = sha(source)
    context = dict(title='Георгий: размещение электродов ТЕПС на КТ',
        description='ТЕПС 2–7 связаны с записями 012–017 Scorpion v2 от 05.10.2026. '
            'КТ получена на глубоком вдохе в другой сессии. Сердце, лёгкие, кости и жир — '
            'автоматические маски TotalSegmentator; независимая точность не установлена. '
            'Руки — цилиндры обхватом 37 см, включая искусственное продолжение за пределы поля КТ. '
            'Точки Ника и иллюстративные боковые контакты сюда не перенесены.',
        subject_pseudonym='Georg', experiment='scorpion_v2_georg', experiment_date='2026-10-05',
        physical_coordinate_frame='LPS', units='mm', body_sha256=inputs['body'],
        electrode_configuration_family='TEPC',
        ct_sha256=sha(segmentation/'static_ct.nii.gz'), tissue_mask_sha256={k:v for k,v in inputs.items() if k!='body'},
        source_is_corrected_body_with_cylindrical_arms=True,
        electrode_contact_area_mm2=None, contact_impedance_ohm_m2=None,
        recordings=[dict(id=f'{n+10:03d}', label=f'ТЕПС {n} · запись {n+10:03d}',
            montage_id=f'georg_tepc_{n}') for n in range(2, 8)])
    report = build_picker_html(body, out/'georg_tepc_picker.html', target_faces,
        bones_stl=references['bone'], lungs_stl=references['lung'], heart_stl=references['heart'],
        fat_stl=references['fat'], reference_target_faces=10000, export_context=context)
    report.update(input_sha256=inputs, ct_sha256=context['ct_sha256'],
        builder_sha256=sha(__file__), picker_sha256=sha(Path(__file__).with_name('build_surface_landmark_picker.py')),
        geometry_is_individually_localized=False, electrode_positions_exported=False,
        html_sha256=sha(out/'georg_tepc_picker.html'))
    (out/'picker_provenance.local.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(out/'georg_tepc_picker.html', flush=True)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry', type=Path, default=GEOMETRY)
    p.add_argument('--segmentation', type=Path, default=SEGMENTATION)
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--target-faces', type=int, default=20000)
    args = p.parse_args()
    build(args.geometry, args.segmentation, args.out, args.target_faces)
