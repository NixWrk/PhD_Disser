"""Reproduce two driver defects using mocked MATLAB and temporary files only."""
from pathlib import Path
import argparse
import contextlib
import importlib.util
import io
import json
import tempfile
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('audit_driver', ROOT/'tools/run_restartable_inverse_search.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
result = {}
with tempfile.TemporaryDirectory(prefix='driver_probe_', dir=HERE) as folder:
    driver.OUT = Path(folder)
    point = [5., 20., 0., 0., 0.]
    with patch.object(driver.subprocess, 'run', return_value=argparse.Namespace(returncode=1)), contextlib.redirect_stdout(io.StringIO()):
        received = driver.evaluate_points([point], tag='synthetic_failure', mode='disjoint',
                                         data_selection='manifest', first_evaluation=1)
    result['arbitrary_matlab_failure_returned_as_objective'] = received
    records = [dict(evaluation=i+1, **dict(zip(driver.PARAMETER_NAMES, point)),
                    objective=1.+i, fem_slope_ohm_per_mm=-.5, h_centre_mm=15.) for i in range(5)]
    tag = 'same_tag'
    driver.write_history(driver.OUT/f'nik_trkg4_inverse_direct_search_history_{tag}.csv', records)
    summary = driver.OUT/f'nik_trkg4_inverse_direct_search_summary_{tag}.json'
    summary.write_text(json.dumps(dict(tag=tag, patch_builder_mode='legacy_fast',
                                       data_selection='legacy_100', final_step=[1]*5)))
    arguments = argparse.Namespace(mode='disjoint', tag=tag, budget=5,
                                   data_selection='manifest', no_resume=False)
    with patch.object(driver, 'parse_args', return_value=arguments), \
         patch.object(driver, 'initial_points', return_value=[]), \
         patch.object(driver, 'finalise_result') as finalise, \
         contextlib.redirect_stdout(io.StringIO()):
        driver.main()
    result['incompatible_history_accepted_and_finalised'] = finalise.call_count == 1
    result['new_summary_after_wrong_history_resume'] = json.loads(summary.read_text())
(HERE/'driver_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
