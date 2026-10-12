"""Simultaneous contact sets on Georg; immutable preparation for the next fit.

Creates own-source CEM input sets for both late-block arm hypotheses.
Does not authorize inversion of the unconverged current FEM reference.
"""
from pathlib import Path
import sys
import numpy as np
from georg_model_contract import resolve
from scorpion_georg_fem_prepare import PIPE, read,write,sha

ROOT=PIPE.parent
VIEW=ROOT/'Colab Notebooks/scorpion_v2'
sys.path.insert(0,str(VIEW))
from scorpion_protocol import ttrkg_configuration,arm_groups
from scorpion_comparisons import OUT as OBS, verify as verify_observations

OUT=PIPE/'output/exploratory/georg_paired_montages_20261012'


def prepare():
    model,current=resolve();verify_observations()
    contacts=read(model/'fem_contacts.local.json')
    geometry=read(model/'contacts.local.json')
    arms={name:next(r['groups_mm'] for r in contacts['scenarios'] if r['id']=='TTRKG_'+name)
          for name in ['1x1','1x4','2x2']}
    # Same top I/U positions in every configuration, as confirmed by the author.
    np.testing.assert_allclose(np.vstack(arms['1x1']),np.vstack(arm_groups(geometry,{'variant':'1x1'})),atol=1e-8)
    union=np.unique(np.vstack(arms['1x4']+arms['2x2']),axis=0)
    scenarios=[]
    # The illustrative side pose is retained as a forward control, not a localization.
    base_lateral={int(r['montage']):r['groups_mm'][4:]
        for r in contacts['scenarios'] if r['family']=='reference' and r['id'].endswith('_lateral')}
    for rid in range(2,11):
        if rid in (5,6):continue  # Perpendicular and cross poses belong to localization.
        size={2:140,3:140,4:140,7:120,8:100,9:80,10:60}[rid]
        variants=['1x1','1x4'] if 7<=rid<=10 else [ttrkg_configuration(f'SCV2-G-{rid:03d}')['variant']]
        for variant in variants:
            groups=arms[variant]+base_lateral[size]
            scenarios.append(dict(id=f'{rid:03d}_{variant}',record_id=f'SCV2-G-{rid:03d}',family='lateral',
                variant=variant,variant_is_confirmed=rid<=4,channel_roles={'1':'ttrkg' if rid<=4 else 'lateral',
                '2':'lateral' if rid<=4 else 'ttrkg'},groups_mm=groups,
                geometry_is_measured=False,lateral_pose_is_fitted=False,size_mm=size))
    for number in range(2,8):
        record=number+10
        tepc=next(r for r in contacts['scenarios'] if r['id']==f'TEPC{number}_nominal')
        scenarios.append(dict(id=f'{record:03d}_TEPC{number}_1x1',record_id=f'SCV2-G-{record:03d}',family='TEPC',
            variant='1x1',variant_is_confirmed=True,channel_roles={'1':'ttrkg','2':'tepc'},
            groups_mm=arms['1x1']+tepc['groups_mm'],geometry_is_measured=False,
            TEPC_centres_are_author_reconstruction=True,lateral_pose_is_fitted=False))
    for row in scenarios:
        active=np.vstack(row['groups_mm'][:4])
        disconnected=[p for p in union if np.min(np.linalg.norm(active-p,axis=1))>1e-8]
        row.update(physical_arm_positions_unchanged=True,
            inactive_arm_contacts_mm=[p.tolist() for p in disconnected],
            contact_presence_policy='connected contacts; disconnected pads retained as separate floating terminals in a second control',
            drive_ports=[[1,4],[2,3],[5,8],[6,7]])
    fields=sorted({key for row in scenarios for key in row})
    scenarios=[{key:row.get(key) for key in fields} for row in scenarios]
    OUT.mkdir(parents=True,exist_ok=True)
    import pandas as pd
    frame=pd.read_csv(OBS/'measured_levels.local.csv')
    frame[frame.role.isin(['ttrkg','lateral_horizontal','tepc'])].to_csv(OUT/'observations.local.csv',index=False,encoding='utf-8-sig')
    sources=[model/'fem_contacts.local.json',model/'contacts.local.json',OBS/'contract.local.json',
        OUT/'observations.local.csv',VIEW/'scorpion_protocol.py',Path(__file__)]
    plan=dict(schema='georg_paired_montages_v1',frequency_hz=100000,subject='georg',scenarios=scenarios,
        model_directory=str(model.relative_to(PIPE)),reference_materials_file=str((PIPE/current['fem_directory']/'reference_materials.local.json').relative_to(PIPE)),
        diameter_mm=5.,nominal_area_mm2=contacts['nominal_area_mm2'],
        contact_impedance_ohm_m2=read(PIPE/current['fem_directory']/'plan.local.json')['contact_impedance_ohm_m2'],
        free_materials=['soft_background','lung','heart','fat'],free_material_indices_matlab=[1,2,4,5],
        notation={'rho1':'soft_background','rho2':'lung','rho3':'heart','rho4':'fat'},
        fixed_material='bone',fixed_material_index_matlab=3,
        excitation_scenario='own_drive',source_confirmation='author: same as Nix, 2026-10-12',
        current_per_channel_uA=7000,current_value_is_approximate=True,current_amplitude_convention='not established',
        fit_states='separate inspiration and expiration, unchanged inspiratory CT is an assumption',
        lateral_pose_must_be_fitted_jointly_with_resistivities=True,
        AEPC_coordinates_unavailable=True,quantitative_inversion_allowed=current['quantitative_inversion_allowed'],
        numerical_gate='georg_model_contract.resolve(require_converged=True)',
        sources=[dict(path=str(p.resolve()),sha256=sha(p)) for p in sources])
    write(OUT/'plan.local.json',plan)
    print('Prepared',len(scenarios),'paired montages; quantitative inversion allowed:',plan['quantitative_inversion_allowed'])
    return OUT


if __name__=='__main__':prepare()
