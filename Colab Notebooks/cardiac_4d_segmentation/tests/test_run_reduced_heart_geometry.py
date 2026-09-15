"""End-to-end geometry smoke test using only small synthetic NIfTI masks."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_reduced_heart_geometry import run, TARGETS, mask_path
from run_affine_heart_geometry import run as run_affine


class ReducedGeometryRunTest(unittest.TestCase):
    def test_separate_cycles_and_crop_coordinates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / 'data'
            affine = np.diag([1.5, 2., 2.5, 1.])
            affine[:3, 3] = [-80., 40., 10.]
            rows = []
            for subject in ['synthetic_a', 'synthetic_b']:
                phases = []
                for p, (cycle, t) in enumerate([(1, 900), (0, 200), (1, 800), (0, 100)]):
                    phase = f'phase_{p:02d}'
                    mask = np.zeros((16, 16, 16), dtype=np.uint8)
                    offset = t // 100 % 2
                    mask[4+offset:8+offset, 5:10, 6:9] = 1
                    image = nib.Nifti1Image(mask, affine)
                    image.header.set_xyzt_units('mm')
                    phase_dir = data / subject / 'phases'
                    phase_dir.mkdir(parents=True, exist_ok=True)
                    nib.save(image, phase_dir / (phase + '.nii.gz'))
                    phases.append({'phase_id': phase, 'nifti': phase + '.nii.gz', 'shape': list(mask.shape),
                                   'voxel_spacing_mm': [1.5, 2., 2.5], 'affine_coordinate_system': 'NIfTI RAS+'})
                    row = {'subject': subject, 'phase_id': phase, 'cycle_index': cycle, 'rr_percent': p*25,
                           'phase_percent_within_cycle': p*25, 'rr_interval_ms': 600,
                           'derived_time_from_R0_ms': t, 'temporal_resolution_ms': 100}
                    for region in TARGETS:
                        path = mask_path(data, subject, phase, region)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        nib.save(image, path)
                        row[region + '_ml'] = mask.sum() * abs(np.linalg.det(affine[:3, :3])) / 1000
                    row['heart_minus_blood_ml'] = 0.
                    row['blood_outside_whole_heart_ml'] = 0.
                    rows.append(row)
                (phase_dir / 'conversion_manifest.json').write_text(json.dumps({'phases': phases}), encoding='utf-8')
            table = root / 'time.csv'
            with table.open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            output = root / 'exploratory' / 'comparison'
            config = root / 'config.json'
            config.write_text(json.dumps({'derived_root': str(data), 'subjects': ['synthetic_a', 'synthetic_b'],
                'time_table': str(table), 'output_dir': str(output), 'mask_definitions': {},
                'reference_rule': 'earliest_within_cycle'}), encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                summary = run(config)
            self.assertEqual(summary['phase_region_pairs'], 48)
            self.assertEqual(summary['reference_pair_count'], 24)
            self.assertEqual(summary['nonreference_pair_count'], 24)
            self.assertEqual(summary['boundary_pair_count'], 0)
            records = [json.loads(line) for line in (output / 'geometry_comparison.jsonl').read_text().splitlines()]
            for record in records:
                self.assertEqual(record['reference_phase_id'], 'phase_03' if record['cycle_index'] == 0 else 'phase_02')
                self.assertAlmostEqual(record['comparisons']['individual_isotropic']['overlap']['dice_voxel_center_approx'], 1.)
                offset = int(record['derived_time_from_R0_ms']) // 100 % 2
                expected = affine @ [5.5+offset, 7., 7., 1.]
                np.testing.assert_allclose(record['moments']['centroid_mm'], expected[:3])
            self.assertEqual(json.loads((output / 'run_status.json').read_text())['status'], 'completed')
            affine_output = root / 'exploratory' / 'affine'
            with contextlib.redirect_stdout(io.StringIO()):
                run_affine(config, output, affine_output)
            affine_records = [json.loads(line) for line in (affine_output / 'affine_comparison.jsonl').read_text().splitlines()]
            self.assertEqual(len(affine_records), 48)
            for record in affine_records:
                for comparison in record['comparisons'].values():
                    self.assertAlmostEqual(comparison['overlap']['dice_voxel_center_approx'], 1.)
            self.assertEqual(json.loads((affine_output / 'run_status.json').read_text())['status'], 'completed')



if __name__ == '__main__':
    unittest.main()
