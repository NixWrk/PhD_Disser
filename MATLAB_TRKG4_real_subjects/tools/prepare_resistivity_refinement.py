"""Freeze the first full rho refinement, its checks, reuse provenance and batches.

No FEM is executed. Existing studies/results are never changed. The 5^3 grid
contains a nested 3^3 grid; 4^3 cell centres are independent checking states.
"""
from pathlib import Path
import copy
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os

from configure_resistivity_grid import configure_grid
from electrode_sensitivity_contract import validate_study

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'output/exploratory/arm_sigma_20260908'
PRIOR = ROOT / 'output/exploratory/resistivity_sensitivity_20260910'
BASE = ROOT / 'output/exploratory/resistivity_refinement_20260910'
PARAMS = ['soft', 'heart', 'lung']


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def state_key(values):
    return tuple(round(float(values[k]), 12) for k in PARAMS)


def main():
    if BASE.exists():
        raise FileExistsError(f'Refusing to replace a prepared refinement: {BASE}')
    old_identity = json.loads((OLD / 'identity.json').read_text(encoding='utf-8'))
    assert sha(OLD / 'prepared_fem.mat') == old_identity['prepared_fem_sha256']
    integrity = json.loads((OLD / 'result_integrity.json').read_text(encoding='utf-8'))['files']
    for name, expected in integrity.items():
        assert sha(OLD / name) == expected, name
    profile_file = PRIOR / 'literature/frozen/rho_profile.json'
    profile = json.loads(profile_file.read_text(encoding='utf-8'))
    montages = json.loads((PRIOR / 'coverage_plan.json').read_text())['montages']
    axes = {k: [lo + (hi-lo)*i/4 for i in range(5)] for k, (lo, hi) in profile['bounds'].items()}
    criteria = {
        'derivative_rtol': .05, 'derivative_grid_atol_fraction': .005,
        'Q_atol': .005, 'pair_rtol': .05,
        'pair_S_scale_atol_fraction': .0001, 'pair_Q_atol': .0001,
        'min_B_refinement_rtol': .01, 'min_Q_refinement_atol': .0005,
        'preferred_B_relative_band': .0001, 'preferred_Q_absolute_band': .0001,
    }
    BASE.mkdir(parents=True)
    (BASE / 'rho_profile.json').write_bytes(profile_file.read_bytes())
    reuse = []
    batches_by_montage = {}
    all_states = []
    for montage in montages:
        master_path = BASE / 'studies' / f'{montage}_all.json'
        study = configure_grid(PRIOR / 'studies' / f'{montage}.json', profile_file, 5, master_path)
        source_csv = OLD / f'{montage}_results.csv'
        with source_csv.open(encoding='utf-8-sig', newline='') as handle:
            old_rows = list(csv.DictReader(handle))
        lookup = {state_key({k: r['sigma_'+k] for k in PARAMS}): r for r in old_rows}
        indexed = list(zip(itertools.product(range(5), repeat=3), study['states']))
        indexed.sort(key=lambda item: (not all(i % 2 == 0 for i in item[0]), item[0]))
        requested = []
        for idx, state in indexed:
            state = copy.deepcopy(state)
            role = 'grid3' if all(i % 2 == 0 for i in idx) else 'grid5_added'
            all_states.append(dict(montage=montage, state=state['id'], role=role,
                                   rho={k: 1/state['conductivity'][k] for k in PARAMS}))
            match = lookup.get(state_key(state['conductivity']))
            if match is not None:
                assert all(abs(float(match['sigma_'+k])-state['conductivity'][k]) < 1e-12 for k in PARAMS)
                reuse.append(dict(montage=montage, state=state['id'], role=role,
                    source_csv=os.path.relpath(source_csv, BASE).replace('\\','/'),
                    source_sha256=sha(source_csv), case_index=match['case_index'],
                    conductivity=state['conductivity']))
            else:
                requested.append((role, state))
        nominal = {p['id']: p['nominal_sigma'] for p in study['parameters']}
        for number, idx in enumerate(itertools.product(range(4), repeat=3)):
            rho = {k: (axes[k][i]+axes[k][i+1])/2 for k, i in zip(PARAMS, idx)}
            sigma = nominal | {k: 1/rho[k] for k in PARAMS}
            state = dict(id=f'rho5_check_{number:06d}', conductivity=sigma)
            requested.append(('check5', state))
            all_states.append(dict(montage=montage, state=state['id'], role='check5', rho=rho))
        batches = []
        for role in ['grid3', 'grid5_added', 'check5']:
            states = [state for r, state in requested if r == role]
            for offset in range(0, len(states), 8):
                batch = copy.deepcopy(study)
                batch['states'] = states[offset:offset+8]
                name = f'{montage}__{role}_{offset//8:02d}'
                path = BASE / 'studies' / f'{name}.json'
                write_json(path, validate_study(batch))
                batches.append(dict(id=name, montage=montage, role=role,
                    study=f'studies/{name}.json', study_sha256=sha(path),
                    output=f'fem/{name}', states=len(batch['states'])))
        batches_by_montage[montage] = batches
    # Stage-first round robin makes completed common parameter blocks available
    # for all assemblies before moving deeper into the cube.
    batches = []
    for role in ['grid3', 'grid5_added', 'check5']:
        groups = [[b for b in batches_by_montage[m] if b['role'] == role] for m in montages]
        for i in range(max(map(len, groups))):
            batches.extend(g[i] for g in groups if i < len(g))
    contract = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        status='preregistered_exploratory_first_refinement_not_computed',
        frequency_hz=50000, geometry_fixed=True, compartments_uniform=True,
        mesh_sha256=old_identity['mesh_sha256'], prepared_sha256=old_identity['prepared_fem_sha256'],
        profile_sha256=sha(profile_file), axes_rho=axes, montages=montages,
        grid_levels=[3,5], grid_states_per_montage=125, independent_centres_per_montage=64,
        total_required_states=len(all_states), reused_states=len(reuse), new_fem_states=sum(b['states'] for b in batches),
        criteria=criteria, states=all_states, reuse=reuse, batches=batches,
        stopping_rule={
            'adequate_on_tested_states': 'All direct-derivative, Q and paired-difference interpolation checks pass; min_B/min_Q and preferred sets stable from nested 3 to 5 and after adding checking states.',
            'unresolved_pairs': 'Differences within the declared numerical band remain unresolved; no single winner forced.',
            'requires_refinement': 'Any failed criterion requires further rho coverage; failed cells and implicated axes are reported. Level 9 is not silently launched in this first attempt.',
            'scope': 'Empirical coverage of this CT/FEM model and fixed literature profile. No continuous-domain proof, spatial convergence or physical validation.',
        },
        sources_sha256={name:sha(ROOT/'tools'/name) for name in ['run_electrode_sensitivity.py','electrode_sensitivity_contract.py','compartment_sensitivity.py','configure_resistivity_grid.py','prepare_resistivity_refinement.py']},
    )
    write_json(BASE / 'refinement_plan.json', contract)
    (BASE / 'refinement_plan.sha256').write_text(sha(BASE/'refinement_plan.json')+'\n', encoding='ascii')
    print(json.dumps({k:contract[k] for k in ['total_required_states','reused_states','new_fem_states']}, indent=2))


if __name__ == '__main__':
    main()
