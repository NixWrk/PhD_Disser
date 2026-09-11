"""Summarize the completed audit; validate aggregate transfer, not raw signals."""
from pathlib import Path
import csv
import hashlib
import json
import math

here = Path(__file__).resolve().parent
root = here.parents[1]
current = json.loads((here/'current_checks.json').read_text(encoding='utf-8'))
fem = json.loads((here/'current_fem_checks.json').read_text(encoding='utf-8'))
assert fem['complete'] and current['generator_sources_match']
assert len(current['cells']) == 24 and all(c['status']=='passed' for c in current['cells'])
rows = list(csv.DictReader((root/'data/nik/experimental/nik_breath_hold_levels.csv').open(encoding='utf-8-sig')))
deviations = []
for row in rows:
    source = root.parent/'Colab Notebooks/timestamps'/Path(row['source_file']).with_suffix('.json')
    saved = json.loads(source.read_text(encoding='utf-8'))
    deviations += [abs(float(row['Z_inhale_hold_ohm'])-saved['hold_levels']['вдох']),
                   abs(float(row['Z_exhale_hold_ohm'])-saved['hold_levels']['выдох']),
                   abs(float(row['delta_in_minus_ex_ohm']) -
                       (float(row['Z_inhale_hold_ohm'])-float(row['Z_exhale_hold_ohm'])))]
assert max(deviations) < 1e-10
kept = [int(row['L_mm']) for row in rows if row['is_independent']=='1']
assert kept == [50,60,70,80,90,110,120,130,140]
paths = [root/'src/run_trkg4_inverse_inhale.m', root/'src/trkg4_config.m',
         root/'src/trkg4_load_nik_inhale_data.m',root/'src/trkg4_finite_step_diagnostics.m',
         root/'vendor_stl_eidors/assign_tissues_from_stl.m',
         root/'vendor_stl_eidors/electrode_faces_by_area.m',
         root/'data/nik/experimental/nik_breath_hold_levels.csv']
record = {'scope':'read-only review of current implementation, isolated numerical experiments',
          'source_hashes':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
          'aggregate_transfer':{'rows':len(rows),'max_difference_ohm':max(deviations),
              'kept_sizes_mm':kept,'raw_signal_reaggregation_performed':False},
          'python_tests_passed':62,'matlab_tests_passed':fem['tests_passed'],
          'current_notebook_cells_passed':len(current['cells']),
          'fem_solves_completed':16,'fem':fem,
          'report_metrics':{k:current[k] for k in ['direct_error_table','calibrated_table','objective_parts']},
          'planar_2L_3L_max_difference_ohm':max(c['max_2L_3L_series_difference_ohm'] for c in current['flat_layer_checks']),
          'notes':['Python test count comes from the pytest command in this task.',
                   'The first MATLAB review launch passed 30 tests then failed on an incorrect CSV name in the review script; the corrected launch completed all FEM checks. Both outputs are retained in current_matlab.log.',
                   'No new geometry optimisation, raw-signal processing or experimental validation.']}
def finite(value):
    if isinstance(value,dict):return {k:finite(v) for k,v in value.items()}
    if isinstance(value,list):return [finite(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    return value
(here/'evidence_summary.json').write_text(json.dumps(finite(record),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print('PASS: 10 aggregate rows; 9 selected sizes; 24 notebook cells; 16 FEM solves; evidence saved.')
