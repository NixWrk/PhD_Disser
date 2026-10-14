from pathlib import Path
import json
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'MATLAB_TRKG4_real_subjects/tools'))
from georg_author_search_pipeline import Pipeline, BudgetExpired, PIPE


class DeadlineChecks(unittest.TestCase):
    def configuration(self, job, deadline):
        config = dict(job_directory=str(job), output_directory=str(job / 'results'),
                      search_model=str(job / 'model'), deadline_unix=deadline,
                      search_deadline_unix=deadline, code=[], matlab_executable='unused',
                      numerical_audit='unused', centre_file='unused', max_evaluations=36)
        file = job / 'configuration.local.json'
        file.write_text(json.dumps(config), encoding='utf-8')
        return file

    def test_expired_budget_does_not_launch_child(self):
        with tempfile.TemporaryDirectory(dir=PIPE / 'output') as name:
            job = Path(name)
            p = Pipeline(self.configuration(job, time.time()-1))
            with self.assertRaises(BudgetExpired):
                p.process('expired', [sys.executable, '-c', 'raise RuntimeError()'])
            self.assertIsNone(p.child)

    def test_running_child_is_stopped_and_journaled(self):
        with tempfile.TemporaryDirectory(dir=PIPE / 'output') as name:
            job = Path(name)
            p = Pipeline(self.configuration(job, time.time()+.6))
            start = time.monotonic()
            with self.assertRaises(BudgetExpired):
                p.process('short_budget', [sys.executable, '-c', 'import time;time.sleep(30)'])
            self.assertLess(time.monotonic()-start, 8)
            state = json.loads((job / 'state.local.json').read_text())
            self.assertEqual(state['stages'][0]['status'], 'deadline_stop')
            self.assertIsNone(state['child_pid'])

    def test_interrupted_search_uses_completed_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=PIPE / 'output') as name:
            job = Path(name)
            p = Pipeline(self.configuration(job, time.time()+30))
            run = job / 'results/joint_own_drive_author_local30_v1'
            run.mkdir(parents=True)
            row = dict(status='numerical_candidate_not_validated', parameters=[0.]*9,
                       mean_square_ohm2=4., target_Z_ohm=[0.]*24)
            (run / 'evaluation_0001.local.json').write_text(json.dumps(row))
            (run / 'evaluation_0002.local.json').write_text(json.dumps(dict(
                status='forward_calculation_running',mean_square_ohm2=[])))
            with patch.object(p, 'matlab', side_effect=BudgetExpired('search deadline')):
                file, result = p.search(30)
            self.assertTrue(file.is_file())
            self.assertTrue(result['search_stopped_for_time'])
            self.assertEqual(result['best_mean_square_ohm2'],4.)
            self.assertFalse(result['quantitative_result_accepted'])


if __name__ == '__main__':
    unittest.main()
