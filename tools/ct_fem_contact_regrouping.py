"""New electrical wiring on immutable CT/mesh/materials and physical contacts."""
from pathlib import Path
import argparse
import copy
import shutil

import ct_fem_core as core
from ct_fem_contacts import normalize


def validate_change(parent, current):
    old=copy.deepcopy(parent);new=copy.deepcopy(current)
    for value in [old,new]:
        value.pop('_config_path',None)
        value['contacts'].pop('scenarios')
    if old!=new:
        raise ValueError('Regrouping may change only electrical scenarios; geometry, materials, centres and areas must remain identical')


def prepare(config_path, source, out, *, reason, protocol=None):
    if not isinstance(reason,str) or not reason.strip():raise ValueError('Declare the reason for the new wiring')
    source=Path(source).resolve();out=Path(out).resolve();config_path=Path(config_path).resolve()
    if out.exists() or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError('Regrouping requires a fresh separate output')
    parent=core.verify_artifact(source)
    if parent['stage']!='materials':raise ValueError('A completed material artifact is required')
    cfg=core.load_config(config_path);validate_change(parent['configuration'],cfg)
    actual=core.read(source/'contacts.local.json')
    if actual['scenarios']!=parent['configuration']['contacts']['scenarios']:
        raise ValueError('Parent actual wiring and passport disagree')
    candidate=copy.deepcopy(actual);candidate['scenarios']=copy.deepcopy(cfg['contacts']['scenarios'])
    normalize(cfg,candidate)
    sources=core.artifact_sources(cfg,source,parent['sources'])
    sources[str(Path(__file__).resolve())]=core.sha(__file__)
    if protocol:sources[str(Path(protocol).resolve())]=core.sha(protocol)
    out.mkdir(parents=True)
    try:
        for name,digest in parent['products'].items():
            if name=='contacts.local.json':continue
            target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,target)
            if core.sha(target)!=digest:raise ValueError('Bulk product changed during regrouping: '+name)
        core.write_new(out/'contacts.local.json',candidate)
        audit=dict(schema='ct_fem_electrical_regrouping_v1',reason=reason,
            parent_manifest_sha256=core.sha(source/'preparation.local.json'),
            source_contact_sha256=core.sha(source/'contacts.local.json'),
            output_contact_sha256=core.sha(out/'contacts.local.json'),
            physical_contact_catalog_unchanged=candidate['physical_contacts']==actual['physical_contacts'],
            mesh_and_material_arrays_unchanged=True,old_scenarios=actual['scenarios'],new_scenarios=candidate['scenarios'],**core.FLAGS)
        core.write_new(out/'electrical_regrouping.local.json',audit)
        core.verify_sources(sources)
        return core.publish(out,'materials',cfg,parent['input_qc'],sources,
            parent.get('completed_geometry_settings',{}),dict(kind='explicit_electrical_regrouping',
                parent_manifest_sha256=audit['parent_manifest_sha256'],physical_geometry_preserved=True,
                source_contacts_sha256=audit['source_contact_sha256'],reason=reason))
    except BaseException as error:
        core.write_new(out/'preparation_failed.local.json',dict(status='failed',error=str(error),**core.FLAGS))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['config','input','out','reason']:p.add_argument('--'+name,required=True)
    p.add_argument('--protocol');a=p.parse_args()
    prepare(a.config,a.input,a.out,reason=a.reason,protocol=a.protocol)
