"""Freeze the 50 kHz literature profile and prepare independent rho-cell checks.

Run from any directory. This prepares inputs only; FEM is launched through
run_pipeline('Profile','electrode_sensitivity', ...). Existing results are read-only.
"""
from pathlib import Path
import csv
import hashlib
import itertools
import json
import math

from electrode_sensitivity_contract import validate_study

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'output/exploratory/resistivity_sensitivity_20260910'
OLD = BASE.parent / 'arm_sigma_20260908'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def conductivity(row, frequency=50000):
    omega = 2 * math.pi * frequency
    result = float(row[34])
    for i, scale in [(28, 1e-12), (31, 1e-9), (35, 1e-6), (38, 1e-3)]:
        delta, tau, alpha = map(float, row[i:i+3])
        power = (omega*tau*scale)**(1-alpha)
        real = 1 + power*math.cos((1-alpha)*math.pi/2)
        imag = power*math.sin((1-alpha)*math.pi/2)
        result += omega*8.8542e-12*delta*imag/(real*real+imag*imag)
    return result


def main():
    acquisition = BASE/'literature/acquisition'
    source = next(acquisition.glob('Thermal*.txt'))
    rows = list(csv.reader(source.read_text(encoding='latin1').splitlines(), delimiter='\t'))
    wanted = ['Fat', 'Muscle', 'Heart Muscle', 'Blood', 'Lung (Inflated)', 'Lung (Deflated)', 'Bone (Cortical)']
    anchors = {}
    for row in rows[3:]:
        if len(row)>40 and row[1] in wanted:
            sigma = conductivity(row)
            anchors[row[1]] = {'sigma_S_m': sigma, 'rho_ohm_m': 1/sigma,
                               'coefficients': dict(zip(rows[2][27:41], row[27:41]))}
    assert set(anchors) == set(wanted)
    identity = json.loads((OLD/'identity.json').read_text(encoding='utf-8'))
    old = identity['study']
    fixed = json.loads((OLD/'fixed_parameters.json').read_text(encoding='utf-8'))
    bounds = {key: sorted([anchors[x]['rho_ohm_m'] for x in names])
              for key, names in {'soft':['Fat','Muscle'], 'heart':['Heart Muscle','Blood'],
                                 'lung':['Lung (Inflated)','Lung (Deflated)']}.items()}
    axes = {key: sorted([1/s for s in old[key+'_levels'] if bounds[key][0]-1e-9 <= 1/s <= bounds[key][1]+1e-9])
            for key in bounds}
    assert [len(v) for v in axes.values()] == [3,3,2]
    profile = {'profile_id':'itis_gabriel_50khz_whole_compartment_envelope_v1',
               'frequency_hz':50000, 'rho_units':'ohm*m', 'bounds':bounds,
               'bone_fixed_rho_ohm_m':1/fixed['bone_conductivity_S_m'],
               'anchors':anchors, 'database_doi':'10.13099/VIP21000-05-0',
               'database_url':'https://itis.swiss/virtual-population/tissue-properties/downloads/database-v5-0',
               'database_archive_sha256':sha(acquisition/'Database-V5-0.zip'),
               'database_table_sha256':sha(source), 'original_rho_axes':axes,
               'interpretation':'Analyst-defined scalar effective-compartment envelope between literature tissue/state anchors. Not a population interval, joint probability law, pulse or breathing amplitude; not exhaustive of all literature.',
               'heart_assumption':'Entire heart mask represents effective blood and myocardium; no separate blood mask or inferred blood fraction.',
               'excluded_historical_lung_rho_ohm_m':1/old['lung_levels'][0],
               'coverage_protocol':{'method':'independent FEM at all four cell centres of the 3x3x2 rho grid',
                 'derivative_error_rule':'abs(interpolated-direct) <= .05*abs(direct)+.005*max_abs_coarse_for_same_montage_and_compartment',
                 'selectivity_absolute_tolerance':.005,
                 'criterion_status':'Analyst-selected interpolation tolerance; not measurement accuracy or a continuous-domain certificate',
                 'continuous_domain_certified':False}}
    write_json(BASE/'literature/frozen/rho_profile.json', profile)
    manifest = json.loads((OLD/'result_integrity.json').read_text(encoding='utf-8'))['files']
    manifest['prepared_fem.mat'] = identity['prepared_fem_sha256']
    for name in ['prepared_fem.mat','fixed_parameters.json']:
        assert sha(OLD/name) == manifest[name], name
    midpoints = {k:[(a+b)/2 for a,b in zip(v[:-1],v[1:])] for k,v in axes.items()}
    states=[]
    for index, rho in enumerate(itertools.product(*midpoints.values()), 1):
        states.append({'id':f'cell_{index:02d}', 'conductivity':
                       dict(zip(bounds, [1/x for x in rho])) | {'bone':fixed['bone_conductivity_S_m']}})
    prepared=[]
    for result in sorted(OLD.glob('in*_results.csv')):
        assert sha(result) == manifest[result.name], result.name
        montage = result.name.removesuffix('_results.csv')
        contact = montage+'_contact.mat'
        assert sha(OLD/contact) == manifest[contact], contact
        study = {'schema_version':'electrode_sensitivity_v1','status':'exploratory_hypothesis_not_validated',
                 'source':{'prepared_fem':'../../arm_sigma_20260908/prepared_fem.mat',
                   'prepared_sha256':manifest['prepared_fem.mat'], 'frequency_hz':50000},
                 'parameters':[{'id':k,'block_indices':blocks, 'nominal_sigma':nominal, 'fixed':k=='bone'}
                   for k,blocks,nominal in [('soft',[0,1],old['baseline'][0]),('heart',[2],old['baseline'][1]),
                        ('lung',[3],anchors['Lung (Inflated)']['sigma_S_m']),('bone',[4],fixed['bone_conductivity_S_m'])]],
                 'states':states,
                 'montages':[{'id':montage,'contact_matrix':'../../arm_sigma_20260908/'+contact,
                   'contact_sha256':manifest[contact],'electrodes':['E1','E2','E3','E4'],
                   'contact_geometry_provenance':'Verified original 40.05 CEM contacts; outer planes only, inner electrodes are cuffs in the mixed assembly.',
                   'channels':[{'id':'standard','drive':[1,0,0,-1],'measure':[0,1,-1,0]}]}],
                 'analysis':{'map_states':[], 'save_fields':False}}
        validate_study(study)
        write_json(BASE/'studies'/f'{montage}.json', study)
        prepared.append(montage)
    write_json(BASE/'coverage_plan.json', {'montages':prepared, 'new_states_per_montage':len(states),
               'existing_states_per_montage':18,'new_states_total':len(states)*len(prepared),
               'rho_cell_centres':list(itertools.product(*midpoints.values())),
               'profile_sha256':sha(BASE/'literature/frozen/rho_profile.json'),
               'input_manifest_sha256':sha(OLD/'result_integrity.json')})
    print(json.dumps({'montages':len(prepared), 'states':len(prepared)*len(states),'bounds':bounds}))


if __name__ == '__main__':
    main()
