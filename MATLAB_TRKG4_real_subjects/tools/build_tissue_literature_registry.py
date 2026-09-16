"""Extract the pinned IT'IS V5.0 source into the project's tissue registry.

Usage: python build_tissue_literature_registry.py DATABASE.zip OUTPUT.json
Only the verified public database is read; no patient data or FEM results change.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

ARCHIVE_SHA256 = '531adceeca8a108b0c6ec0fdb276381cba4f3f0627990e3bb66aaa6673a063c0'
MEMBER = 'Thermal_dielectric_acoustic_MR properties_database_V5.0(ASCII).txt'
TISSUES = ('Muscle', 'Fat', 'Heart Muscle', 'Blood', 'Lung (Inflated)',
           'Lung (Deflated)', 'Bone (Cortical)', 'Bone (Cancellous)', 'Skin')


def build(archive):
    raw = Path(archive).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ARCHIVE_SHA256:
        raise ValueError('Database differs from the pinned V5.0 archive; review a new version explicitly.')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        source = z.read(MEMBER)
    rows = list(csv.reader(io.StringIO(source.decode('cp1252')), delimiter='\t'))
    if rows[2][27:35] != ['ef', 'del1', 'tau1 (ps)', 'alf1', 'del2', 'tau2 (ns)', 'alf2', 'sig']:
        raise ValueError('Unexpected dielectric column layout')
    tissues = {}
    for row in rows[3:]:
        if len(row) < 63 or row[1] not in TISSUES:
            continue
        name = row[1]
        if name in tissues:
            raise ValueError('Duplicate tissue: ' + name)
        c = list(map(float, row[27:41]))
        lf = {}
        for label, start in [('mixed', 43), ('across', 48), ('along', 53), ('normal', 58)]:
            lf[label] = {k: float(v) if v.strip() else None for k, v in zip(
                ('mean_S_m', 'sd_S_m', 'sample_size', 'min_S_m', 'max_S_m'), row[start:start+5])}
        bounds = [1 / lf['mixed']['max_S_m'], 1 / lf['mixed']['min_S_m']]
        frequency = {}
        for f in (50000, 100000):
            w = 2 * math.pi * f
            eps = complex(c[0], -c[7] / (w * 8.8542e-12))
            for d, t, a in [(c[1], c[2]*1e-12, c[3]), (c[4], c[5]*1e-9, c[6]),
                            (c[8], c[9]*1e-6, c[10]), (c[11], c[12]*1e-3, c[13])]:
                eps += d / (1 + (1j*w*t)**(1-a))
            sigma = -eps.imag * w * 8.8542e-12
            frequency[str(f)] = {'sigma_S_m': sigma, 'rho_ohm_m': 1/sigma,
                                 'relative_permittivity': eps.real}
        tissues[name] = {'source_assignment': row[42], 'source_row_name': name,
                         'coefficients': dict(zip(rows[2][27:41], c)),
                         'gabriel_at_frequency': frequency,
                         'low_frequency_up_to_1MHz': lf,
                         'low_frequency_mixed_rho_bounds_ohm_m': bounds}
    if set(tissues) != set(TISSUES):
        raise ValueError('Missing tissue rows')
    groups = {'muscle': ['Muscle'], 'fat': ['Fat'],
              'blood': ['Blood'], 'myocardium': ['Heart Muscle'],
              'soft': ['Muscle', 'Fat'], 'heart': ['Blood', 'Heart Muscle'],
              'lung_inflated': ['Lung (Inflated)'], 'lung_deflated': ['Lung (Deflated)'],
              'lung_unspecified': ['Lung (Inflated)', 'Lung (Deflated)'],
              'bone_cortical': ['Bone (Cortical)'], 'bone_cancellous': ['Bone (Cancellous)']}
    profiles = {}
    for f in ('50000', '100000'):
        compartments = {}
        for key, names in groups.items():
            candidates = []
            for name in names:
                t = tissues[name]
                for side, value in zip(('min', 'max'), t['low_frequency_mixed_rho_bounds_ohm_m']):
                    candidates.append({'rho_ohm_m': value, 'tissue': name, 'origin': 'LF_mixed_rho_'+side})
                candidates.append({'rho_ohm_m': t['gabriel_at_frequency'][f]['rho_ohm_m'],
                                   'tissue': name, 'origin': 'Gabriel_'+f+'Hz'})
            lo = min(candidates, key=lambda x: x['rho_ohm_m'])
            hi = max(candidates, key=lambda x: x['rho_ohm_m'])
            compartments[key] = {'rho_bounds_ohm_m': [lo['rho_ohm_m'], hi['rho_ohm_m']],
                                 'lower_source': lo, 'upper_source': hi, 'tissue_members': names}
        profiles[f] = {'id': 'itis_v5_lf_gabriel_screening_'+f+'hz_v2',
                       'frequency_hz': int(f), 'compartments': compartments,
                       'default_nominal': None,
                       'frequency_transfer_assumption': 'LF mixed min/max applied as uncertainty scenarios at this frequency; not a measured frequency-specific interval',
                       'physiological_population_interval': False,
                       'pulse_or_breathing_amplitude': False,
                       'independent_joint_distribution_assumed': False,
                       'anisotropy_covered': False}
    return {'schema_version': 1, 'registry_id': 'itis_v5_tissue_reference_20260916_v2',
            'source': {'version': 'ITIS V5.0', 'doi': '10.13099/VIP21000-05-0',
                       'url': 'https://itis.swiss/assets/Downloads/TissueDb/Database-V5-0.zip',
                       'archive_sha256': ARCHIVE_SHA256, 'member': MEMBER,
                       'member_sha256': hashlib.sha256(source).hexdigest()},
            'tissues': tissues, 'screening_profiles': profiles,
            'policy': {'scope': 'prospective direct tissue-property calculations and sensitivity',
                       'no_automatic_midpoint_nominal': True,
                       'separate_tissues_require_separate_material_maps': True,
                       'combined_and_separate_materials_are_alternative_partitions': True,
                       'legacy_runs_are_immutable': True,
                       'inverse_search_bounds_are_separate': True,
                       'required_run_metadata': ['registry_id', 'registry_sha256', 'profile_id',
                          'frequency_hz', 'material_mapping', 'rho_ohm_m', 'nominal_rationale',
                          'literature_frequency_transfer_assumption', 'fixed_parameters']}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = build(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
