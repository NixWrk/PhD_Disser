"""Prepare paired motion probes using existing, hash-verified L1 materials.

This writes only a new local configuration and material-plan index. It neither
changes a mesh nor assumes that the +/-5 mm probes are measured heart motion.
The established run_pipeline('heart_stl_electrical_check') performs the solves.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False)+'\n', encoding='utf-8')


def run(data_root, output):
    b = data_root.resolve()
    source_config = b/'stl_pose_electrical_q64_v1.local.json'
    base_config = b/'stl_electrical_q64_v1.local.json'
    fit_path = b/'stl_effective_fit_q64_v1/effective_fit.local.json'
    cfg, base, fit = read(source_config), read(base_config), read(fit_path)
    shared = ('mesh_path', 'stl_path', 'eit_model_file', 'ttrkg_model_file',
              'reference_prediction_file', 'frequency_Hz', 'current_A')
    for key in shared:
        if cfg[key] != base[key]:
            raise ValueError(f'Different baseline/pose condition: {key}')
    for field, key in [('eit_model_sha256','eit_model_file'),
                       ('ttrkg_model_sha256','ttrkg_model_file'),
                       ('reference_sha256','reference_prediction_file')]:
        if fit['identity'][field] != sha(cfg[key]):
            raise ValueError(f'Effective fit has different source: {field}')
    fitted = [row for row in fit['rows'] if row['id']=='L1']
    if len(fitted)!=1 or not fitted[0]['converged']:
        raise ValueError('One completed L1 conditional fit required')
    diagnostic = cfg['jobs'][0]['rho_ohm_m']
    if any(job['rho_ohm_m'] != diagnostic for job in cfg['jobs']):
        raise ValueError('Pose probes have inconsistent material scenarios')
    scenarios = {'diagnostic4': diagnostic,
                 'effectiveL1': fitted[0]['rho_ohm_m']}
    wanted = {f'L1_v{v}_q64' for v in ('095','100','105')}
    wanted |= {f'L1_{axis}{direction}5_v{v}_q64'
               for axis in 'xyz' for direction in ('m','p')
               for v in ('095','100','105')}
    variants, provenance = {}, {}
    plan_template = None
    for config in (base, cfg):
        path = Path(config['material_plan'])
        plan = read(path)
        if plan_template is None:
            plan_template = copy.deepcopy(plan)
        for field,key in [('mesh_sha256','mesh_path'),('stl_sha256','stl_path')]:
            if plan[field] != sha(cfg[key]):
                raise ValueError(f'Unmatched source geometry: {field}')
        provenance[str(path)] = sha(path)
        for row in plan['variants']:
            if row['id'] not in wanted:
                continue
            item = copy.deepcopy(row)
            material = (path.parent/item['material_file']).resolve()
            if sha(material) != item['material_sha256']:
                raise ValueError(f'Material changed: {material}')
            # MATLAB fullfile(prefix, absolute_path) does not discard prefix.
            item['material_file'] = os.path.relpath(material, output.resolve().parent)
            if item['id'] in variants and variants[item['id']] != item:
                raise ValueError('Ambiguous variant '+item['id'])
            variants[item['id']] = item
    if set(variants) != wanted:
        raise ValueError('Missing variants: '+str(sorted(wanted-set(variants))))
    out = output.resolve()
    config_path = out.with_suffix('.local.json')
    plan_path = out.parent/(out.name+'_materials.local.json')
    if out.exists() or config_path.exists() or plan_path.exists():
        raise FileExistsError('Use a new output name')
    out.mkdir(parents=True)
    plan_template['variants'] = [variants[k] for k in sorted(variants)]
    plan_template['source_plans_sha256'] = provenance
    plan_template['motion_probe_basis'] = 'fixed LPS +/-5 mm, not registered dynamic CT'
    write(plan_path, plan_template)
    cfg['output_dir'], cfg['material_plan'] = str(out), str(plan_path)
    cfg['jobs'] = [dict(id=f'{scenario}_{name}', variant=name,
                        scenario=scenario, rho_ohm_m=rho, eit=True)
                   for scenario,rho in scenarios.items() for name in sorted(variants)]
    cfg['motion_comparison'] = dict(
        shape='L1 volume-preserving moment ellipsoid',
        position_step_mm=5, coordinate_system='LPS_mm',
        prescribed_volume_factors=[0.95,1,1.05],
        source_config_sha256=sha(source_config),
        baseline_config_sha256=sha(base_config),
        effective_fit_sha256=sha(fit_path),
        preparation_sha256=sha(__file__),
        no_dynamic_CT_inputs=True, no_AI_training=True,
        limits=['Two conditional material scenarios, not tissue truth',
                'No dynamic CT to static FEM registration assumed',
                'Diagnostic TT montage, not verified experimental geometry',
                'Existing q64 materials; global mesh convergence not proven'])
    write(config_path, cfg)
    print(json.dumps(dict(config=str(config_path), output=str(out),
                          jobs=len(cfg['jobs'])), ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    run(args.data_root,args.output)
