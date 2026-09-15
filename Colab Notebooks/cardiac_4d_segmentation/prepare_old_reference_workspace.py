"""Prepare a local blinded OLD reference-annotation workspace from converted CT.

No TotalSegmentator masks are read. The HU guide is deliberately unlabelled:
it includes contrast, bone and other bright structures and is not a blood mask.
Run from a local JSON config with conversion_manifest, output_dir and phase_ids.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import nibabel as nib

CHAMBERS = {'LV': 'Left ventricle blood', 'RV': 'Right ventricle blood',
            'LA': 'Left atrium blood', 'RA': 'Right atrium blood'}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def binary_nifti(mask, reference, destination):
    image = nib.Nifti1Image(np.asarray(mask, dtype=np.uint8), reference.affine)
    image.header.set_xyzt_units('mm')
    image.set_qform(reference.affine, code=1)
    image.set_sform(reference.affine, code=1)
    nib.save(image, str(destination))


def checked_ct(path, entry):
    if sha256(path) != entry['sha256']:
        raise ValueError('CT hash differs from conversion manifest')
    image = nib.load(str(path))
    if image.shape != tuple(entry['shape']):
        raise ValueError('CT shape differs from manifest')
    if image.header.get_xyzt_units()[0] != 'mm':
        raise ValueError('Explicit spatial units mm required')
    if not np.allclose(image.affine, entry['affine_ras_mm'], rtol=1e-6, atol=1e-5):
        raise ValueError('CT affine differs from conversion provenance')
    if entry.get('intensity_units') != 'HU':
        raise ValueError('HU intensity provenance required')
    return image


def prepare(config_path):
    cfg = json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
    manifest_path = Path(cfg['conversion_manifest']).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    selected = cfg['phase_ids']
    if len(selected) != len(set(selected)) or not selected:
        raise ValueError('Unique nonempty phase selection required')
    by_id = {entry['phase_id']: entry for entry in manifest['phases']}
    if not set(selected) <= set(by_id):
        raise ValueError('Unknown selected phase')
    output = Path(cfg['output_dir']).resolve()
    if output.exists():
        raise FileExistsError('Output directory must be new; existing annotations are never overwritten')
    output.mkdir(parents=True)
    threshold = float(cfg.get('guide_lower_hu', 250))
    if not np.isfinite(threshold):
        raise ValueError('Invalid HU guide threshold')
    package = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'case_id': manifest['case_id'], 'sequence_id': manifest['sequence_id'],
        'conversion_manifest_sha256': sha256(manifest_path), 'preparer_sha256': sha256(__file__),
        'selection_rule': cfg['selection_rule'], 'phase_ids': selected,
        'reference_available': False, 'anatomical_coverage_verified': False,
        'automatic_comparator_masks_loaded': False,
        'guide': {'method': 'CT_HU_lower_threshold_only', 'lower_hu': threshold,
            'anatomical_label': None, 'includes_bone_and_vessels': True,
            'purpose': 'editable_visual_aid_not_reference', 'threshold_is_physiological_bound': False},
        'annotation_policy': {'papillary_muscles': 'unresolved_before_review',
            'trabeculae': 'unresolved_before_review', 'valve_planes': 'unresolved_before_review',
            'vessel_cutoff': 'unresolved_before_review', 'appendages': 'unresolved_before_review'},
        'phases': [], 'status': 'preparing'}
    write_json(output / 'workspace.local.json', package)
    for phase in selected:
        entry = by_id[phase]
        ct_path = (manifest_path.parent / entry['nifti']).resolve()
        ct = checked_ct(ct_path, entry)
        data = np.asanyarray(ct.dataobj)
        if not np.isfinite(data).all():
            raise ValueError('Nonfinite CT intensity')
        folder = output / phase
        folder.mkdir()
        guide_path = folder / 'HU_guide_not_blood_mask.nii.gz'
        guide = data >= threshold
        binary_nifti(guide, ct, guide_path)
        voxel_mm3 = abs(float(np.linalg.det(ct.affine[:3, :3])))
        phase_info = {'phase_id': phase, 'phase_percent': entry['phase_percent'],
            'ct_path': str(ct_path), 'ct_sha256': entry['sha256'],
            'shape': list(ct.shape), 'affine_ras_mm': ct.affine.tolist(),
            'voxel_spacing_mm': list(ct.header.get_zooms()[:3]),
            'slice_thickness_mm': entry['slice_thickness_mm'],
            'guide_path': str(guide_path), 'guide_sha256': sha256(guide_path),
            'guide_voxel_count': int(np.count_nonzero(guide)),
            'guide_volume_ml_not_blood_volume': float(np.count_nonzero(guide)*voxel_mm3/1000),
            'guide_boundary_faces': [[bool(np.any(np.take(guide, side, axis=axis))) for side in (0, -1)] for axis in range(3)],
            'hu_quantiles': {str(q): float(np.percentile(data, q)) for q in (0, 1, 50, 95, 99, 100)},
            'chambers': {key: {'name': name, 'status': 'not_annotated', 'coverage': 'unreviewed',
                'reference_path': None, 'accepted': False} for key, name in CHAMBERS.items()},
            'project_path': str(folder/'annotation.mrml'), 'review_status': 'pending_anatomical_review'}
        # np.float32 zooms are converted explicitly for portable JSON.
        phase_info['voxel_spacing_mm'] = [float(x) for x in phase_info['voxel_spacing_mm']]
        package['phases'].append(phase_info)
        write_json(output/'workspace.local.json', package)
        print(f'Prepared {phase}: independent HU guide; chamber annotations absent', flush=True)
    package['status'] = 'prepared_pending_interactive_annotation'
    write_json(output/'workspace.local.json', package)
    fields = ['phase_id', 'phase_percent', 'chamber', 'coverage', 'apical_boundary', 'basal_valve_plane',
              'vessels_excluded', 'trabeculae_rule', 'papillary_rule', 'author', 'notes', 'accepted']
    with (output/'annotation_decisions.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for phase in package['phases']:
            for chamber in CHAMBERS:
                writer.writerow({'phase_id': phase['phase_id'], 'phase_percent': phase['phase_percent'],
                    'chamber': chamber, 'coverage': 'unreviewed', 'accepted': False})
    return package


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.config)
