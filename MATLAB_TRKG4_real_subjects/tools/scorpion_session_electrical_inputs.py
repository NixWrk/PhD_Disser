"""Pin observations and 100-kHz reference scenarios for a fresh subject FEM."""
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
import ct_fem_core as core


def build(subject,model,summary_file):
    model=Path(model).resolve();summary_file=Path(summary_file).resolve()
    registry=ROOT/'MATLAB_TRKG4_real_subjects/configs/tissue_literature_v2.json'
    r=core.read(registry)
    def rho(name):return r['tissues'][name]['gabriel_at_frequency']['100000']['rho_ohm_m']
    reference=dict(frequency_hz=100000,material_order=core.MATERIALS,
        inhale_rho_ohm_m=[rho(n) for n in ('Muscle','Lung (Inflated)','Bone (Cortical)','Heart Muscle','Fat')],
        exhale_rho_ohm_m=[rho(n) for n in ('Muscle','Lung (Deflated)','Bone (Cortical)','Heart Muscle','Fat')],
        nominal_heart_is_myocardial_endpoint_of_effective_whole_heart=True,
        blood_fraction_is_measured=False,source='Frequency-specific Gabriel reference entries in the project registry',
        registry_sha256=core.sha(registry),individual_tissue_values_are_measured=False,
        geometric_breathing_motion_modelled=False)
    core.write_new(model/'reference_materials.local.json',reference)
    summary=core.read(summary_file);records={v['record_id']:v for v in summary['records'] if v['subject']==subject}
    rows=[]
    for v in summary['levels']:
        if v['record_id'] not in records:continue
        rec=records[v['record_id']];context=rec['context']
        if context['record_kind']!='both':continue
        rows.append(dict(record_id=v['record_id'],case_id=f"L{context['lateral_size_mm']}_TT{context['ttrkg_variant']}",
            role=v['role'],state=v['state'],Z_ohm=v['baseline_ohm'],source_sha256=rec['source_sha256'],
            annotation_status='pending_manual_review'))
    core.write_new(model/'electrical_inputs.local.json',dict(subject_alias=subject,
        summary_source=dict(path=str(summary_file),sha256=core.sha(summary_file)),observations=rows,
        reference_materials_sha256=core.sha(model/'reference_materials.local.json'),
        current_per_source_A=.007,own_drive_is_working_source_rule=True,source_relative_phase='unknown',
        producer_sha256=core.sha(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('subject','model','summary'):p.add_argument('--'+k,required=True)
    a=p.parse_args();build(a.subject,a.model,a.summary)
