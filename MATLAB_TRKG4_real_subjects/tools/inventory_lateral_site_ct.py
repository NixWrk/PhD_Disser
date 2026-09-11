"""Read ZIP directories only; do not extract, move, or validate CT voxels."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile


def inventory(root):
    rows = []
    for dataset in sorted(root.iterdir()):
        if not dataset.is_dir():
            continue
        archives = []
        for path in sorted(dataset.glob('*.zip')):
            with zipfile.ZipFile(path) as archive:
                infos = [x for x in archive.infolist() if not x.is_dir()
                         and '__MACOSX' not in x.filename
                         and not Path(x.filename).name.startswith('._')]
                names = [x.filename for x in infos]
                ct = [n for n in names if n.endswith('/ct.nii.gz') or
                      (any('/' + folder + '/' in '/' + n for folder in
                           ('imagesTr', 'imagesTs', 'images')) and n.endswith('.nii.gz'))]
                raw = [n for n in names if n.lower().endswith('.img')]
                volume = [n for n in names if n.lower().endswith(('.nii.gz', '.nii', '.img', '.mha', '.mhd', '.dcm', '.raw'))]
                labels = ['heart', 'liver', 'lung_upper_lobe_right', 'lung_middle_lobe_right',
                          'lung_lower_lobe_right', 'lung_upper_lobe_left', 'lung_lower_lobe_left']
                label_files = {label: sum(n.endswith('/segmentations/' + label + '.nii.gz') for n in names)
                               for label in labels}
                digest_input = '\n'.join(f'{x.filename}\t{x.file_size}\t{x.CRC}' for x in infos)
                archives.append({'archive': path.name, 'archive_bytes': path.stat().st_size,
                                 'directory_sha256': hashlib.sha256(digest_input.encode()).hexdigest(),
                                 'directory_file_count': len(infos), 'image_nifti_count': len(ct),
                                 'raw_img_count': len(raw), 'volume_like_entries': len(volume),
                                 'anatomical_label_file_counts': label_files,
                                 'validation': 'names_sizes_crc_metadata_only_no_voxel_or_crc_payload_validation'})
        rows.append({'dataset': dataset.name, 'archives': archives,
                     'manifest_only_count': len(list(dataset.glob('*.tcia')))})
    return {'schema': 'lateral_site_archive_inventory_v1', 'date': '2026-09-10',
            'scope': 'archive_directory_inventory_not_anatomy_QC_or_subject_deduplication',
            'root_config': 'external_root_argument_not_embedded_in_report', 'datasets': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--external-root', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    data = inventory(args.external_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'datasets': len(data['datasets']), 'archives': sum(len(d['archives']) for d in data['datasets'])}))
