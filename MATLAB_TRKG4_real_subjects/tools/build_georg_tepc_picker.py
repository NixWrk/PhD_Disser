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
READER_HTML = ROOT/'Colab Notebooks/scorpion_v2/20.21_Размещение_ТЕПС_Георгия_Scorpion_v2.html'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def mask_surface(path, target, step_size=1):
    image = nib.load(path)
    values = np.asanyarray(image.dataobj) > 0
    if not values.any():
        raise ValueError(f'Empty anatomical mask: {path.name}')
    vertices, faces, _, _ = marching_cubes(values.astype(np.uint8), level=.5, step_size=step_size)
    ras = nib.affines.apply_affine(image.affine, vertices)
    lps = ras * [-1., -1., 1.]
    trimesh.Trimesh(lps, faces, process=False).export(target)


def build(geometry=None, segmentation=None, out=OUT, target_faces=None, landmarks=None, reader_output=None):
    if geometry is None or segmentation is None:
        from georg_model_contract import resolve
        current,contract=resolve()
        if geometry is None:geometry=current
        if segmentation is None:
            segmentation=ROOT/'MATLAB_TRKG4_real_subjects'/contract['segmentation_directory']
        if landmarks is None and Path(geometry).resolve()==current.resolve():
            landmarks=current/'landmarks'
    geometry, segmentation, out = map(Path, (geometry, segmentation, out))
    g = json.loads((geometry/'geometry.local.json').read_text(encoding='utf-8-sig'))
    body = geometry/'body_embedded_arms_370mm.stl'
    if sha(body) != g['surface_sha256']:
        raise ValueError('Corrected body surface hash mismatch')
    body_faces = len(trimesh.load(body, force='mesh').faces)
    if target_faces is None:
        target_faces = body_faces
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
        mask_surface(source, references[name], step_size=3 if name == 'fat' else 1)
        inputs[name] = sha(source)
    context = dict(title='Георгий: размещение электродов ТЕПС на КТ',
        description='ТЕПС 2–7 связаны с записями 012–017 Scorpion v2 от 05.10.2026. '
            'КТ получена на глубоком вдохе в другой сессии. Сердце, лёгкие, кости и жир — '
            'автоматические маски TotalSegmentator; независимая точность не установлена. '
            'Жир отображается с шагом три вокселя без последующего схлопывания треугольников. '
            'Тело показано без дополнительного уменьшения числа треугольников. '
            'Наружный контур заново выделен по полному КТ-тому с ограниченной очисткой внешних '
            'фрагментов. Независимая анатомическая точность автоматического контура не установлена. '
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
    context['body_display_decimated'] = target_faces < body_faces
    if context['body_display_decimated']:
        context['description'] = context['description'].replace(
            'Тело показано без дополнительного уменьшения числа треугольников.',
            'В этой версии число треугольников тела дополнительно уменьшено для предварительного просмотра.')
    manual_hashes = {}
    if landmarks is not None:
        context['manual_reconstructions'] = {}
        for n in range(2,8):
            path = Path(landmarks)/f'georg_tepc_{n}.json'
            data = json.loads(path.read_text(encoding='utf-8-sig'))
            m = data['measurement_context']
            if (m['body_sha256'] != inputs['body'] or m['ct_sha256'] != context['ct_sha256']
                    or m['tissue_mask_sha256'] != context['tissue_mask_sha256']
                    or m['recording']['id'] != f'{n+10:03d}'
                    or m['physical_coordinate_frame'] != 'LPS'):
                raise ValueError('Manual reconstruction geometry or recording mismatch')
            transfer = data.get('ct_transfer_provenance')
            note = m['reconstruction_notes']
            if transfer:
                note += ' Показаны проекции исходных авторских центров на новый КТ-контур; исходные координаты сохранены отдельно.'
            context['manual_reconstructions'][m['recording']['id']] = dict(
                centres=data['preliminary_electrode_centers_xyz_mm'],
                notes=note, status=m['status'],
                filename=path.name, source_sha256=data['import_provenance']['original_sha256'])
            manual_hashes[path.name] = sha(path)
    report = build_picker_html(body, out/'georg_tepc_picker.html', target_faces,
        bones_stl=references['bone'], lungs_stl=references['lung'], heart_stl=references['heart'],
        fat_stl=references['fat'], reference_target_faces=10000,
        fat_target_faces=len(trimesh.load(references['fat'], force='mesh').faces), export_context=context)
    if report['layers']['fat']['decimation']['status'] != 'not_needed':
        raise ValueError('Fat display must not undergo triangle-collapse decimation')
    # A local preview must not silently replace the canonical reader.
    if reader_output is None and out.resolve() == OUT.resolve():
        reader_output = READER_HTML
    if reader_output is not None:
        Path(reader_output).write_bytes((out/'georg_tepc_picker.html').read_bytes())
    report.update(input_sha256=inputs, ct_sha256=context['ct_sha256'],
        builder_sha256=sha(__file__), picker_sha256=sha(Path(__file__).with_name('build_surface_landmark_picker.py')),
        geometry_is_individually_localized=False, electrode_positions_exported=bool(manual_hashes),
        manual_reconstruction_sha256=manual_hashes, anatomical_placement_accuracy_validated=False,
        html_sha256=sha(out/'georg_tepc_picker.html'))
    (out/'picker_provenance.local.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(reader_output or out/'georg_tepc_picker.html', flush=True)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry', type=Path, default=None, help='Default: checked current Georg model contract')
    p.add_argument('--segmentation', type=Path, default=None, help='Default: masks from the current Georg model')
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--target-faces', type=int, default=None,
        help='Optional preview reduction; default retains every body triangle')
    p.add_argument('--landmarks', type=Path, default=None, help='Six normalised author reconstructions')
    p.add_argument('--reader-output', type=Path, default=None,
        help='Explicit reader copy; custom --out otherwise writes only its local HTML')
    args = p.parse_args()
    build(args.geometry, args.segmentation, args.out, args.target_faces, args.landmarks, args.reader_output)
