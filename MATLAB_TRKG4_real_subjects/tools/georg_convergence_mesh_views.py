"""Read-only plane sections of saved parent/child meshes at a frozen pose."""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.io import loadmat
from georg_quality_mesh_section import section_lines
from georg_quality_selected_refinement import read,sha,write_new


def prepare(parent, child, candidate, out):
    parent, child, candidate, out = map(lambda p: Path(p).resolve(), (parent,child,candidate,out))
    if out.exists():raise ValueError('Fresh immutable display output required')
    row=read(candidate);row=row.get('best',row);pose=row['pose']
    centre=np.asarray(pose['centre_lps_mm'],float)
    inward=-np.asarray(pose['normal_unit_xyz'],float)
    sections=[];sources=[candidate,Path(__file__),Path(__file__).with_name('georg_quality_mesh_section.py')]
    for label, model in [('parent',parent),('child',child)]:
        meshfile=model/'mesh.mat';qualityfile=model/'mesh_quality.local.json'
        quality=read(qualityfile)
        if not quality['passed'] or quality['mesh_sha256']!=sha(meshfile):
            raise ValueError('Displayed mesh shape/identity failed')
        mesh=loadmat(meshfile,variable_names=['nodes_mm','elems'])
        elements=mesh['elems'].astype(np.int64)-1
        for axis, field in [('longitudinal','primary_unit_xyz'),('perpendicular','perpendicular_unit_xyz')]:
            horizontal=np.asarray(pose[field],float)
            lines=section_lines(mesh['nodes_mm'],elements,centre,horizontal,inward,window=(-80,80,-5,85))
            sections.append(dict(model=label,axis=axis,mesh_sha256=quality['mesh_sha256'],
                centre_lps_mm=centre.tolist(),horizontal_unit_xyz=horizontal.tolist(),
                inward_unit_xyz=inward.tolist(),**lines))
        sources.extend([meshfile,qualityfile])
        del mesh,elements
    write_new(out,dict(schema='georg_frozen_pose_mesh_sections_v1',sections=sections,
        coordinate_units='mm',source_candidate_sha256=sha(candidate),
        scope='Actual tetrahedral plane intersections at the frozen initial pose; no new fitted location',
        numerical_or_physiological_acceptance=False,FEM_was_run=False,
        sources=[dict(path=str(p),sha256=sha(p)) for p in sources]))
    print({f"{r['model']}_{r['axis']}":r['intersected_tetrahedra_in_window'] for r in sections})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','child','candidate','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();prepare(args.parent,args.child,args.candidate,args.out)
