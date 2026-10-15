"""Audit selected fine CEM faces against an immediate boundary-parent map.

This reads saved geometry only. It neither solves the PDE nor establishes
volume, whole search-window, or global numerical convergence.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
from scipy.io import loadmat


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _indices(value, label, maximum):
    array = np.asarray(value)
    if array.dtype.kind not in 'iuf' or not array.size:
        raise ValueError(label + ': expected nonempty one-based integer indices')
    if (not np.all(np.isfinite(array)) or np.any(array < 1)
            or np.any(array > maximum) or np.any(array != np.floor(array))):
        raise ValueError(label + ': invalid one-based integer index')
    return array.astype(np.int64)


def _vector(value, label):
    array = np.asarray(value)
    if array.ndim > 2 or (array.ndim == 2 and 1 not in array.shape):
        raise ValueError(label + ': expected a scalar, row or column vector')
    return array.reshape(-1)


def fine_patch_coverage(forward, nodes_mm, boundary, boundary_parent):
    """All three index arrays use MATLAB's one-based convention.

    A fine face is refined iff its immediate parent has more than one child
    in the FULL fine boundary, regardless of any parent's selected CEM patch.
    Repeated contacts across descriptions are retained, not deduplicated.
    """
    nodes = np.asarray(nodes_mm)
    if (nodes.ndim != 2 or nodes.shape[1] != 3 or not len(nodes)
            or nodes.dtype.kind not in 'iuf' or not np.all(np.isfinite(nodes))):
        raise ValueError('nodes_mm: expected finite N-by-3 coordinates')
    faces = np.asarray(boundary)
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError('boundary: expected nonempty N-by-3 triangles')
    faces = _indices(faces, 'boundary', len(nodes)) - 1
    # Convert before subtraction: integer coordinates must not overflow.
    nodes = nodes.astype(float)
    with np.errstate(over='ignore', invalid='ignore'):
        areas = np.linalg.norm(np.cross(nodes[faces[:, 1]] - nodes[faces[:, 0]],
                                        nodes[faces[:, 2]] - nodes[faces[:, 0]]), axis=1) / 2
    if not np.all(np.isfinite(areas) & (areas > 0)):
        raise ValueError('boundary: all triangle areas must be positive and finite')

    mapping = _vector(boundary_parent, 'boundary_parent')
    if len(mapping) != len(faces):
        raise ValueError('boundary_parent length differs from current fine boundary')
    parents = _indices(mapping, 'boundary_parent', len(faces)) - 1
    counts = np.bincount(parents)
    if np.any(counts == 0):
        raise ValueError('boundary_parent: parent IDs must be dense and one-based')
    refined = counts[parents] > 1

    patches = forward.get('patches')
    if not isinstance(patches, list) or not patches:
        raise ValueError('forward.patches: expected nonempty physical descriptions')
    contacts, descriptions = [], []
    for ordinal, record in enumerate(patches, 1):
        selected = record.get('faces')
        if not isinstance(selected, list) or not selected:
            raise ValueError('patch faces: expected nonempty per-contact list')
        groups = _indices(_vector(record.get('gids'), 'gids'), 'gids', len(selected))
        if len(groups) != len(selected):
            raise ValueError('gids count differs from physical contact count')
        if 'physical_patches' in record and record['physical_patches'] != len(selected):
            raise ValueError('physical_patches differs from physical contact count')
        if 'electrical_groups' in record:
            group_count = _indices(record['electrical_groups'], 'electrical_groups', len(selected))
            if group_count.size != 1 or np.max(groups) > group_count.item():
                raise ValueError('gids exceeds declared electrical_groups')
        centres = record.get('xyz')
        if centres is not None:
            centres = np.asarray(centres)
            if (centres.shape != (len(selected), 3) or centres.dtype.kind not in 'iuf'
                    or not np.all(np.isfinite(centres))):
                raise ValueError('xyz: expected one finite centre per physical contact')
        rows = []
        for index, value in enumerate(selected, 1):
            ids = _indices(_vector(value, 'patch faces'), 'patch faces', len(faces)) - 1
            if len(np.unique(ids)) != len(ids):
                raise ValueError('patch faces: duplicate fine face within one contact')
            mask = refined[ids]
            area = float(areas[ids].sum())
            refined_area = float(areas[ids[mask]].sum())
            if not np.isfinite(area) or not np.isfinite(refined_area):
                raise ValueError('patch faces: total area must be finite')
            row = dict(description_index=ordinal, record_id=record['record_id'],
                       physical_contact=index, electrical_group=int(groups[index-1]),
                       fine_face_count=len(ids), refined_fine_face_count=int(mask.sum()),
                       unrefined_fine_face_count=int((~mask).sum()),
                       parent_face_count=len(np.unique(parents[ids])),
                       minimum_parent_child_count=int(counts[parents[ids]].min()),
                       maximum_parent_child_count=int(counts[parents[ids]].max()),
                       area_mm2=area, refined_area_mm2=refined_area,
                       unrefined_area_mm2=float(areas[ids[~mask]].sum()),
                       refined_boundary_area_fraction=refined_area / area,
                       all_faces_refined=bool(mask.all()))
            if centres is not None:
                row['centre_mm'] = centres[index-1].tolist()
            rows.append(row)
        contacts.extend(rows)
        descriptions.append(dict(description_index=ordinal, record_id=record['record_id'],
                                 physical_contact_count=len(rows), **_totals(rows)))
    return dict(schema='georg_actual_fine_contact_coverage_v1', contacts=contacts,
                descriptions=descriptions, physical_description_count=len(descriptions),
                physical_contact_description_count=len(contacts), **_totals(contacts),
                minimum_refined_boundary_area_fraction=min(
                    row['refined_boundary_area_fraction'] for row in contacts),
                current_boundary_face_count=len(faces), immediate_parent_face_count=len(counts),
                index_base=1, qualification='immediate parent has >1 children in the full fine boundary',
                aggregation='per physical description; repeated pads/faces are retained, not unique surface area',
                scope='selected actual fine CEM boundary faces at this saved pose only',
                volume_refinement_proven=False, global_mesh_convergence_proven=False,
                whole_search_window_convergence_proven=False,
                independent_physical_validation=False, quantitative_result_accepted=False)


def _totals(rows):
    area = sum(row['area_mm2'] for row in rows)
    refined_area = sum(row['refined_area_mm2'] for row in rows)
    if not np.isfinite(area) or not np.isfinite(refined_area):
        raise ValueError('description aggregate area must be finite')
    return dict(fine_face_count=sum(row['fine_face_count'] for row in rows),
                refined_fine_face_count=sum(row['refined_fine_face_count'] for row in rows),
                unrefined_fine_face_count=sum(row['unrefined_fine_face_count'] for row in rows),
                area_mm2=area, refined_area_mm2=refined_area,
                unrefined_area_mm2=sum(row['unrefined_area_mm2'] for row in rows),
                refined_boundary_area_fraction=refined_area / area,
                all_contact_faces_refined=all(row['all_faces_refined'] for row in rows))


def compute_coverage(forward_path, mesh_path, parent_maps_path):
    paths = [Path(forward_path), Path(mesh_path), Path(parent_maps_path)]
    stats = [(path.stat().st_size, path.stat().st_mtime_ns) for path in paths]
    hashes = [sha(path) for path in paths]
    forward = json.loads(paths[0].read_text(encoding='utf-8'))
    recorded_mesh = forward.get('identity', {}).get('contact_inheritance', {}).get('current_mesh_sha256')
    if recorded_mesh and recorded_mesh != hashes[1]:
        raise ValueError('Actual current mesh SHA differs from forward identity')
    mesh = loadmat(paths[1], variable_names=['nodes_mm', 'boundary'])
    maps = loadmat(paths[2], variable_names=['boundary_parent'])
    result = fine_patch_coverage(forward, mesh['nodes_mm'], mesh['boundary'], maps['boundary_parent'])
    if stats != [(path.stat().st_size, path.stat().st_mtime_ns) for path in paths]:
        raise ValueError('Coverage input changed while reading')
    result.update(forward_sha256=hashes[0], current_mesh_sha256=hashes[1],
                  parent_maps_sha256=hashes[2], producer_sha256=sha(__file__),
                  forward_current_mesh_identity_verified=bool(recorded_mesh),
                  parent_mapping_binding='supplied immediate map; SHA pinned and length/index checked; no parent selected patches used',
                  sources=[dict(role=role, path=str(path.resolve()), sha256=digest)
                           for role, path, digest in zip(('forward', 'current_mesh', 'immediate_parent_maps'), paths, hashes)])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forward', type=Path, required=True)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--parent-maps', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError('Coverage output must be fresh: ' + str(args.out))
    result = compute_coverage(args.forward, args.mesh, args.parent_maps)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with args.out.open('x', encoding='utf-8') as stream:
        stream.write(payload)
    print(json.dumps(dict(physical_descriptions=result['physical_description_count'],
                          physical_contacts=result['physical_contact_description_count'],
                          all_contact_faces_refined=result['all_contact_faces_refined'],
                          minimum_refined_boundary_area_fraction=result['minimum_refined_boundary_area_fraction'])))


if __name__ == '__main__':
    main()
