"""New bulk-identical model with an explicitly chosen contact master.

This changes only the discretization master of finite contact footprints.
Pose construction, tissue fractions, physical inventory and electrical groups
are unchanged. Historical plans and prepared matrices remain immutable.
"""
from pathlib import Path
import argparse
import copy
import shutil
from georg_quality_selected_refinement import read, sha, write_new, _sources


def prepare(source, out, master, boundary_map):
    source, out, master = [Path(p).resolve() for p in (source, out, master)]
    boundary_map = Path(boundary_map).resolve() if boundary_map else None
    if out.exists() or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError('A fresh separate model is required')
    planfile = source / 'plan.local.json'
    plan = read(planfile)
    for row in plan['sources']:
        if sha(row['path']) != row['sha256']:
            raise ValueError('Source model dependency changed: ' + row['path'])
    if not boundary_map and sha(source / 'mesh.mat') != sha(master):
        raise ValueError('An identity contact map requires the same mesh')
    out.mkdir(parents=True)
    names = ['mesh.mat', 'mesh_quality.local.json', 'reference_materials.local.json',
             *[f'fractions_q{q}.mat' for q in plan['numerical_quadrature_points']]]
    for name in ('parent_maps.mat', 'inheritance.local.json'):
        if (source / name).exists():
            names.append(name)
    for name in names:
        shutil.copyfile(source / name, out / name)
        if sha(source / name) != sha(out / name):
            raise ValueError('Copied coefficient or geometry changed')
    variant = copy.deepcopy(plan)
    variant.update(parent_directory=str(source), exact_contact_parent_mesh=str(master),
        boundary_parent_map=str(boundary_map) if boundary_map else '',
        contact_parent_mesh_sha256=sha(master), contact_discretization_changed=True,
        contact_geometry_rule='Explicit numerical contact master; unchanged pose, bulk and physical connections',
        quantitative_inversion_allowed=False, global_mesh_convergence_proven=False,
        producer_sha256=sha(Path(__file__)))
    variant.pop('validation_scope', None)
    variant['sources'] = _sources([*[r['path'] for r in plan['sources']], planfile,
        *[out / name for name in names], master, *([boundary_map] if boundary_map else []), Path(__file__)])
    write_new(out / 'plan.local.json', variant)
    write_new(out / 'contact_master_change.local.json', dict(
        source_plan_sha256=sha(planfile), variant_plan_sha256=sha(out / 'plan.local.json'),
        source_mesh_sha256=sha(source / 'mesh.mat'), variant_mesh_sha256=sha(out / 'mesh.mat'),
        contact_master_sha256=sha(master), boundary_map_sha256=sha(boundary_map) if boundary_map else '',
        bulk_identical=True, physical_inventory_and_connections_changed=False,
        pose_reference_changed=False, FEM_was_run=False, quantitative_result_accepted=False))
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'out', 'master'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--boundary-map', default='')
    a = p.parse_args()
    print(prepare(a.source, a.out, a.master, a.boundary_map))
