"""Completed checkpoint recovery from tiny synthetic JSON, without any solve."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from georg_convergence_selected_checks import completed_choice


OBSERVATIONS_SHA = hashlib.sha256(b'independent synthetic observations').hexdigest()
# Explicit, heterogeneous 24-level target; expected MSEs below are known from
# residuals +/-1 and +/-2, not calculated by the implementation under test.
TARGET = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
          34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56]
PREDICTED_MSE_1 = [11, 11, 15, 15, 19, 19, 23, 23, 27, 27, 31, 31,
                   35, 35, 39, 39, 43, 43, 47, 47, 51, 51, 55, 55]
PREDICTED_MSE_4 = [12, 10, 16, 14, 20, 18, 24, 22, 28, 26, 32, 30,
                   36, 34, 40, 38, 44, 42, 48, 46, 52, 50, 56, 54]


def checkpoint(predicted, objective):
    return dict(status='numerical_candidate_not_validated',
                predicted_Z_ohm=list(predicted), target_Z_ohm=list(TARGET),
                parameters=[0.0, 1.0, 2.0, 0.1, 1.1, 2.1, -10.0, -5.0, -2.5],
                pose={'centre_lps_mm': [1.0, 2.0, 3.0]}, mean_square_ohm2=objective,
                identity={'observations_sha256': OBSERVATIONS_SHA})


class CompletedChoiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, index, row):
        path = self.root / f'evaluation_{index:04d}.local.json'
        path.write_text(json.dumps(row), encoding='utf-8')
        return path

    def test_known_24_level_objectives_choose_0041_and_preserve_inventory(self):
        self.write(40, checkpoint(PREDICTED_MSE_4, 4.0))
        best = checkpoint(PREDICTED_MSE_1, 1.0)
        selected = self.write(41, best)
        # Deliberately no usable vectors/identity: status must be checked first.
        self.write(42, dict(status='forward_calculation_running', mean_square_ohm2=0.0))
        self.write(43, dict(status='invalid_geometry', mean_square_ohm2=-100.0))
        before = {path.name: path.read_bytes() for path in self.root.iterdir()}
        result = completed_choice(self.root, OBSERVATIONS_SHA)
        self.assertEqual(result['best'], best)
        self.assertEqual(result['best']['mean_square_ohm2'], 1.0)
        self.assertEqual(result['source_checkpoint'], str(selected.resolve()))
        self.assertEqual(result['source_checkpoint_sha256'],
                         hashlib.sha256(selected.read_bytes()).hexdigest())
        self.assertEqual(result['completed_numerical_evaluations'], 2)
        inventory = result['checkpoint_inventory']
        self.assertEqual([row['status'] for row in inventory],
                         ['numerical_candidate_not_validated']*2 +
                         ['forward_calculation_running', 'invalid_geometry'])
        for row in inventory:
            self.assertEqual(row['sha256'], hashlib.sha256(Path(row['path']).read_bytes()).hexdigest())
        self.assertTrue(result['search_interrupted'])
        for flag in ('resolution_reached', 'quantitative_result_accepted', 'physical_validation'):
            self.assertFalse(result[flag])
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.root.iterdir()})

    def test_objective_four_is_accepted_without_a_better_checkpoint(self):
        path = self.write(1, checkpoint(PREDICTED_MSE_4, 4.0))
        result = completed_choice(self.root, OBSERVATIONS_SHA)
        self.assertEqual(result['best']['mean_square_ohm2'], 4.0)
        self.assertEqual(result['source_checkpoint'], str(path.resolve()))

    def test_observation_mismatch_rejects_even_a_worse_completed_candidate(self):
        self.write(41, checkpoint(PREDICTED_MSE_1, 1.0))
        wrong = checkpoint(PREDICTED_MSE_4, 4.0)
        wrong['identity']['observations_sha256'] = hashlib.sha256(b'other observations').hexdigest()
        self.write(40, wrong)
        with self.assertRaisesRegex(ValueError, 'observation identity'):
            completed_choice(self.root, OBSERVATIONS_SHA)

    def test_wrong_objective_rejected_instead_of_recomputed_or_selected(self):
        self.write(41, checkpoint(PREDICTED_MSE_1, 1.0))
        for objective in (0.0, 2.0, 96.0):  # zero, RMSE, SSE; actual MSE is 4
            with self.subTest(objective=objective):
                self.write(40, checkpoint(PREDICTED_MSE_4, objective))
                with self.assertRaisesRegex(ValueError, 'original unweighted criterion'):
                    completed_choice(self.root, OBSERVATIONS_SHA)

    def test_no_complete_checkpoint_rejected(self):
        with self.assertRaisesRegex(ValueError, 'No complete'):
            completed_choice(self.root, OBSERVATIONS_SHA)
        self.write(1, dict(status='forward_calculation_running'))
        self.write(2, dict(status='invalid_geometry'))
        with self.assertRaisesRegex(ValueError, 'No complete'):
            completed_choice(self.root, OBSERVATIONS_SHA)

    def test_completed_vectors_require_exactly_24_finite_levels(self):
        base = checkpoint(PREDICTED_MSE_1, 1.0)
        for field in ('predicted_Z_ohm', 'target_Z_ohm'):
            for values in (base[field][:-1], [[value] for value in base[field]],
                           [float('nan')] + base[field][1:]):
                with self.subTest(field=field, values=values):
                    row = copy.deepcopy(base)
                    row[field] = values
                    self.write(1, row)
                    with self.assertRaisesRegex(ValueError, 'invalid levels'):
                        completed_choice(self.root, OBSERVATIONS_SHA)

    def test_nonfinite_parameters_rejected(self):
        row = checkpoint(PREDICTED_MSE_1, 1.0)
        row['parameters'][7] = float('inf')
        self.write(1, row)
        with self.assertRaisesRegex(ValueError, 'invalid levels'):
            completed_choice(self.root, OBSERVATIONS_SHA)


if __name__ == '__main__':
    unittest.main()
