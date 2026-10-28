"""Summarize completed Nik M1--M9 fits without refitting or touching masks."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from heart_library_fits import sha, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def table(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    library = Path(args.library)
    target = library/'comparison'
    if target.exists():
        raise FileExistsError(target)
    baseline = [json.loads(line) for line in Path(args.baseline).read_text(encoding='utf-8').splitlines() if line.strip()]
    refs = {f"{r['subject']}_{r['phase_id']}_{r['region']}":r for r in baseline}
    variants, evidence = {}, {}
    for count in (8192,32768):
        folder = library/f'fits_{count}'
        completion = read(folder/'scoring/completion.json')
        for name, expected in completion['artifacts_sha256'].items():
            if sha(folder/'scoring'/name) != expected:
                raise ValueError('Scored output changed: '+name)
        results = read(folder/'scoring/results.json')
        if results['protocol']['baseline_sha256'] != sha(args.baseline):
            raise ValueError('Baseline changed')
        variants[count] = {(r['case_id'],r['method_id']):r for r in results['rows']}
        if len(variants[count]) != 108:
            raise ValueError('Expected 108 unique scored fits')
        evidence[str(count)] = {'scoring_results_sha256':sha(folder/'scoring/results.json'),
            'completion_sha256':sha(folder/'scoring/completion.json'),
            'matlab_results_sha256':sha(folder/'results.json')}
    differences, dynamics, parameter_rows = [], [], []
    for count, fits in variants.items():
        for r in fits.values():
            p = r['parameters']
            line = {'sample_count':count, 'case_id':r['case_id'], 'method_id':r['method_id'],
                'method_name':r['method_name'], 'status':r['status'],
                'settings_json':json.dumps(r['settings'],sort_keys=True),
                'target_mask_sha256':r['target_mask_sha256']}
            if p:
                line['volume_ml'] = p['volume_ml']
                line.update({f'center_{axis}_mm':float(v) for axis,v in zip('xyz',p['center_mm'])})
                line.update({f'semiaxis_{axis}_mm':float(v) for axis,v in zip('abc',p['semi_axes_mm'])})
                line.update({f'R{i+1}{j+1}':float(p['axes_matrix'][i][j]) for i in range(3) for j in range(3)})
            parameter_rows.append(line)
    for key, low in variants[8192].items():
        high = variants[32768][key]
        row = dict(case_id=key[0], method_id=key[1], comparable=bool(low['metrics'] and high['metrics']))
        if row['comparable']:
            a,b = low['parameters'],high['parameters']
            ra,rb = np.asarray(a['axes_matrix']),np.asarray(b['axes_matrix'])
            sa = ra@np.diag(np.square(a['semi_axes_mm']))@ra.T
            sb = rb@np.diag(np.square(b['semi_axes_mm']))@rb.T
            row.update(center_difference_mm=float(np.linalg.norm(np.array(b['center_mm'])-a['center_mm'])),
                volume_difference_ml=float(b['volume_ml']-a['volume_ml']),
                volume_difference_percent_of_high=100*(b['volume_ml']-a['volume_ml'])/b['volume_ml'],
                shape_tensor_relative_difference=float(np.linalg.norm(sb-sa)/np.linalg.norm(sb)),
                max_axis_relative_difference=float(np.max(np.abs(np.array(b['semi_axes_mm'])-a['semi_axes_mm'])/b['semi_axes_mm'])),
                dice_difference=high['metrics']['dice']-low['metrics']['dice'],
                full_surface_RMS_difference_mm=high['metrics']['target_surface_to_candidate_RMS_mm']-low['metrics']['target_surface_to_candidate_RMS_mm'])
        differences.append(row)
    for count, fits in variants.items():
        for mid in [f'M{i}' for i in range(1,10)]:
            for cycle in sorted({r['cycle_index'] for r in baseline}):
                rows = sorted([r for r in fits.values() if r['method_id']==mid and r['cycle_index']==cycle],
                    key=lambda r:refs[r['case_id']]['derived_time_from_R0_ms'])
                rec = dict(sample_count=count,method_id=mid,cycle_index=cycle,
                    phase_count=len(rows),complete=all(r['metrics'] is not None for r in rows))
                if not rec['complete']:
                    dynamics.append(rec)
                    continue
                predicted = np.array([r['parameters']['volume_ml'] for r in rows])
                truth = np.array([r['target_volume_ml'] for r in rows])
                cp = np.array([r['parameters']['center_mm'] for r in rows])
                ct = np.array([r['target_center_mm'] for r in rows])
                ref_indices = [i for i,r in enumerate(rows) if r['is_reference_phase']]
                if len(ref_indices)!=1:
                    raise ValueError('Expected one reference phase per cycle')
                anchor=ref_indices[0]
                rec.update(method_name=rows[0]['method_name'], reference_phase_id=rows[anchor]['phase_id'],
                    phase_order=','.join(r['phase_id'] for r in rows),
                    volume_RMSE_ml=rms(predicted-truth),
                    volume_change_from_reference_RMSE_ml=rms((predicted-predicted[anchor])-(truth-truth[anchor])),
                    adjacent_volume_increment_RMSE_ml=rms(np.diff(predicted)-np.diff(truth)),
                    target_volume_range_ml=float(np.ptp(truth)),
                    fitted_volume_range_ml=float(np.ptp(predicted)),
                    volume_range_error_ml=float(np.ptp(predicted)-np.ptp(truth)),
                    volume_Pearson_r=float(np.corrcoef(predicted,truth)[0,1]) if np.std(predicted)>0 and np.std(truth)>0 else None,
                    center_absolute_RMS_mm=float(np.sqrt(np.mean(np.sum((cp-ct)**2,axis=1)))),
                    center_displacement_from_reference_RMS_mm=float(np.sqrt(np.mean(np.sum(((cp-cp[anchor])-(ct-ct[anchor]))**2,axis=1)))),
                    adjacent_center_increment_RMS_mm=float(np.sqrt(np.mean(np.sum((np.diff(cp,axis=0)-np.diff(ct,axis=0))**2,axis=1)))))
                dynamics.append(rec)
    summary = []
    for mid in [f'M{i}' for i in range(1,10)]:
        good=[r for r in differences if r['method_id']==mid and r['comparable']]
        record={'method_id':mid,'comparable_phases':len(good)}
        for metric in ('center_difference_mm','volume_difference_ml','volume_difference_percent_of_high',
                       'shape_tensor_relative_difference','max_axis_relative_difference','dice_difference','full_surface_RMS_difference_mm'):
            record['max_abs_'+metric]=max((abs(r[metric]) for r in good),default=None)
        summary.append(record)
    target.mkdir()
    data={'schema':'nik_whole_heart_library_comparison_v1',
        'baseline_sha256':sha(args.baseline),'source_code_sha256':sha(__file__),
        'evidence':evidence,'sampling_differences':differences,'sampling_summary':summary,'cycle_dynamics':dynamics,
        'limitations':['Shape fit and evaluation use existing automatic WHOLE_HEART masks, not independent anatomy.',
            'Surface RMS is one-sided full-target area quadrature; Dice is the common voxel-centre approximation.',
            '8192-to-32768 comparison evaluates input quadrature sensitivity, not CT resolution or segmentation accuracy.',
            'Dynamics never bridge different R-R cycles; sample range is not stroke volume.',
            'Principal-axis sign/order is arbitrary; convergence uses shape tensors rather than raw rotations.',
            'No prior numerical acceptance threshold was established; close rankings remain provisional.']}
    write_json(target/'comparison.json', data)
    table(target/'sampling_summary.csv',summary)
    table(target/'cycle_dynamics.csv',dynamics)
    table(target/'sampling_differences.csv',differences)
    table(target/'parameters.csv',parameter_rows)
    write_json(target/'completion.json',{'artifacts_sha256':{p.name:sha(p) for p in target.iterdir() if p.is_file()},
        'adapter_sha256':sha(__file__)})
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',required=True)
    parser.add_argument('--baseline',required=True)
    run(parser.parse_args())
