"""Import author-confirmed TEPC filenames without altering selected coordinates."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import re

from build_surface_landmark_picker import validate_landmark_payload

PIPE = Path(__file__).resolve().parents[1]
GEOMETRY = PIPE/'output/exploratory/scorpion_georg_geometry_20261009'
SEGMENTATION = PIPE/'output/exploratory/scorpion_georg_fem_20261008/segmentation'
OUT = PIPE/'output/exploratory/georg_tepc_manual_20261011'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def normalise(payload, filename, expected_hashes, *, filename_mapping_confirmed=False):
    if not filename_mapping_confirmed:
        raise ValueError('Filename-to-recording mapping requires author confirmation')
    match = re.fullmatch(r'georg_tepc_([2-7])\.json', filename)
    if not match:
        raise ValueError('Expected georg_tepc_2.json through georg_tepc_7.json')
    valid, errors = validate_landmark_payload(payload)
    if not valid:
        raise ValueError('; '.join(errors))
    context = payload.get('measurement_context', {})
    if (context.get('subject_pseudonym') != 'Georg' or context.get('experiment') != 'scorpion_v2_georg'
            or context.get('physical_coordinate_frame') != 'LPS' or context.get('units') != 'mm'
            or context.get('electrode_configuration_family') != 'TEPC'):
        raise ValueError('Georg TEPC experiment and physical LPS frame must be explicit')
    for key, expected in expected_hashes.items():
        if context.get(key) != expected:
            raise ValueError(f'Input geometry or mask hash mismatch: {key}')
    n = int(match[1])
    result = copy.deepcopy(payload)
    result['measurement_context']['recording'] = dict(id=f'{n+10:03d}',
        label=f'ТЕПС {n} · запись {n+10:03d}', montage_id=f'georg_tepc_{n}')
    result['measurement_context']['reconstruction_notes'] = (
        context.get('reconstruction_notes', '') + '\n'
        'Ручная реконструкция автора от 11.10.2026. Автор сообщил, что дефекты поверхности '
        'затрудняли выбор центров; погрешность установки пока не определена. '
        'Связь с записью подтверждена автором по имени файла.').strip()
    result['measurement_context']['status'] = 'manual_reconstruction_surface_uncertainty_unquantified'
    result['import_provenance'] = dict(original_recording=copy.deepcopy(context.get('recording')),
        filename=filename, mapping_source='direct_author_confirmation_2026-10-11',
        selected_coordinates_changed=False, anatomical_accuracy_validated=False,
        signs_and_bei_channel_confirmed=False)
    return result


def import_files(files, out=OUT, *, filename_mapping_confirmed=False):
    import numpy as np
    import trimesh
    surface_path = GEOMETRY/'body_embedded_arms_370mm.stl'
    expected = dict(body_sha256=sha(surface_path), ct_sha256=sha(SEGMENTATION/'static_ct.nii.gz'),
        tissue_mask_sha256={k:sha(SEGMENTATION/(k+'.nii.gz')) for k in ['heart', 'lung', 'bone', 'fat']})
    files = list(map(Path, files))
    if sorted(p.name for p in files) != [f'georg_tepc_{n}.json' for n in range(2,8)]:
        raise ValueError('Import requires all six distinct montage files')
    bundles = []
    surface = trimesh.load(surface_path, force='mesh')
    for path in sorted(files):
        raw = path.read_bytes()
        payload = json.loads(raw.decode('utf-8-sig'))
        normalised = normalise(payload, path.name, expected,
            filename_mapping_confirmed=filename_mapping_confirmed)
        original_hash = hashlib.sha256(raw).hexdigest()
        normalised['import_provenance']['original_sha256'] = original_hash
        points = np.array([payload['preliminary_electrode_centers_xyz_mm'][k] for k in payload['electrode_order']])
        projected, distance, _ = trimesh.proximity.closest_point(surface, points)
        projection = dict(surface_sha256=expected['body_sha256'],
            candidate_centres_xyz_mm=dict(zip(payload['electrode_order'], projected.tolist())),
            displacement_mm=dict(zip(payload['electrode_order'], distance.tolist())),
            projection_is_candidate=True, selected_coordinates_replaced=False,
            anatomical_placement_error_estimated=False)
        bundles.append((path.name, raw, normalised, projection))
    out = Path(out)
    (out/'originals').mkdir(parents=True, exist_ok=True)
    (out/'normalised').mkdir(exist_ok=True)
    def write(path, value):
        path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    report = dict(status='manual_reconstruction_surface_uncertainty_unquantified',
        expected_hashes=expected, importer_sha256=sha(__file__), recordings=[])
    for filename, raw, _, _ in bundles:
        original_path = out/'originals'/filename
        if original_path.exists() and original_path.read_bytes() != raw:
            raise ValueError('An existing input snapshot differs; use a new output directory')
    for filename, raw, payload, projection in bundles:
        original_path = out/'originals'/filename
        original_path.write_bytes(raw)
        write(out/'normalised'/filename, payload)
        write(out/(Path(filename).stem+'_projection.local.json'), projection)
        report['recordings'].append(dict(filename=filename, source_sha256=payload['import_provenance']['original_sha256'],
            recording=payload['measurement_context']['recording'],
            selected_centres_xyz_mm=payload['preliminary_electrode_centers_xyz_mm'],
            projection=projection))
    write(out/'import_manifest.local.json', report)
    print('Imported six montages; original coordinates retained; no FEM solve performed', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--confirmed-filename-mapping', action='store_true')
    args = parser.parse_args()
    import_files(args.files, args.out, filename_mapping_confirmed=args.confirmed_filename_mapping)
