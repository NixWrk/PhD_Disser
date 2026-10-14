"""Read saved before/after body products; no surface rebuild, remesh or FEM.

Exports local patient diagnostics only to an explicitly supplied fresh root.
Distances use the historical deterministic sample seed, plus a declared local
vertex probe around changed voxels. Neither is a certified Hausdorff maximum.
"""
import argparse
import gc
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import trimesh

import ct_fem_core as core


def components(voxels, lps, added, voxel_volume_mm3, connectivity):
    radius = 1.001 if connectivity == 6 else np.sqrt(3)+.001
    pairs = np.asarray(list(cKDTree(voxels).query_pairs(radius)), dtype=np.int64).reshape(-1, 2)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(voxels), len(voxels)))
    count, labels = connected_components(graph, directed=False)
    rows = []
    for key in range(count):
        ix = np.flatnonzero(labels == key); points = lps[ix]; v = voxels[ix]
        rows.append(dict(component=key, voxels=len(ix), added=int(added[ix].sum()), removed=int((~added[ix]).sum()),
            volume_ml=len(ix)*voxel_volume_mm3/1000, bbox_voxel=[v.min(0).tolist(), v.max(0).tolist()],
            bbox_lps_mm=[points.min(0).tolist(), points.max(0).tolist()], centre_lps_mm=points.mean(0).tolist()))
    return sorted(rows, key=lambda r: -r["voxels"]), labels


def distances(source, target, count, threshold, changed_tree, voxels, labels, chunk=128):
    points, source_faces = trimesh.sample.sample_surface(source, count, seed=20261011)
    closest = np.empty_like(points); values = np.empty(count); target_faces = np.empty(count, np.int64)
    for start in range(0, count, chunk):
        stop = start+chunk
        closest[start:stop], values[start:stop], target_faces[start:stop] = target.nearest.on_surface(points[start:stop])
    worst = np.argsort(values)[-12:][::-1]
    near, index = changed_tree.query(points[worst])
    locations = [dict(distance_mm=float(values[k]), source_lps_mm=points[k].tolist(),
        closest_target_lps_mm=closest[k].tolist(), source_face_id=int(source_faces[k]),
        target_face_id=int(target_faces[k]), nearest_changed_voxel=voxels[j].tolist(),
        distance_to_changed_voxel_mm=float(d), changed_component_26=int(labels[j]))
        for k, j, d in zip(worst, index, near)]
    return dict(sample_count=count, seed=20261011, max_mm=float(values.max()),
        median_mm=float(np.median(values)), p95_mm=float(np.percentile(values, 95)),
        p99_mm=float(np.percentile(values, 99)), p999_mm=float(np.percentile(values, 99.9)),
        above_declared_threshold=int((values > threshold).sum()), threshold_mm=threshold,
        worst_locations=locations, maximum_is_sampled_not_certified=True), (points, values, source_faces)


def local_vertex_probe(source, target, tree, radius, voxels, labels, cap=40000):
    distance, _ = tree.query(source.vertices, workers=1)
    ids = np.flatnonzero(distance <= radius)
    if len(ids) > cap:
        raise ValueError("Local affected-vertex probe exceeds explicit cap; no silent subsampling")
    values = np.empty(len(ids)); closest = np.empty((len(ids), 3))
    for start in range(0, len(ids), 64):
        ix = ids[start:start+64]
        closest[start:start+64], values[start:start+64], _ = target.nearest.on_surface(source.vertices[ix])
    if not len(ids):
        return dict(vertex_count=0, support_radius_mm=radius)
    worst = np.argsort(values)[-12:][::-1]
    near, vi = tree.query(source.vertices[ids[worst]])
    rows = [dict(vertex_id=int(ids[k]), distance_mm=float(values[k]),
        source_lps_mm=source.vertices[ids[k]].tolist(), closest_target_lps_mm=closest[k].tolist(),
        nearest_changed_voxel=voxels[j].tolist(), distance_to_changed_voxel_mm=float(d),
        changed_component_26=int(labels[j])) for k, j, d in zip(worst, vi, near)]
    return dict(vertex_count=len(ids), support_radius_mm=radius, max_mm=float(values.max()),
        p95_mm=float(np.percentile(values, 95)), worst_locations=rows,
        maximum_is_local_vertex_probe_not_certified_surface_bound=True)


def run(source, out, cfg_path, count=15000, vertex_cap=40000):
    source=Path(source).resolve(); out=Path(out).resolve(); cfg_path=Path(cfg_path).resolve()
    if out.exists() or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError("Fresh separate diagnostic root required")
    cfg=core.read(cfg_path)
    files=[source/n for n in ("body_hu_mask.nii.gz", "body_regularized_mask.nii.gz",
        "body_before_regularization.stl", "skin_regularized.stl")]+[cfg_path, Path(__file__), Path(core.__file__)]
    inputs={str(p):core.sha(p) for p in files}
    before=nib.load(files[0]); after=nib.load(files[1])
    if before.shape!=after.shape or not np.array_equal(before.affine,after.affine):
        raise ValueError("Before/after mask grids differ")
    old=np.asanyarray(before.dataobj)>0; new=np.asanyarray(after.dataobj)>0
    change=old!=new; voxels=np.column_stack(np.nonzero(change)); added=new[tuple(voxels.T)]
    if not len(voxels):
        raise ValueError("No changed voxels; use unchanged-export QC instead")
    lps=nib.affines.apply_affine(before.affine,voxels)*[-1,-1,1]
    voxel_volume=abs(float(np.linalg.det(before.affine[:3,:3])))
    c6,_=components(voxels,lps,added,voxel_volume,6); c26,labels=components(voxels,lps,added,voxel_volume,26)
    body=dict(shape=list(before.shape),affine_ras_mm=before.affine.tolist(),
        spacing_mm=list(map(float,before.header.get_zooms())), original_voxels=int(old.sum()),
        regularized_voxels=int(new.sum()), changed_voxels=len(voxels),added=int(added.sum()),removed=int((~added).sum()),
        changed_relative=float(len(voxels)/old.sum()),changed_volume_ml=len(voxels)*voxel_volume/1000,
        components_6=c6,components_26=c26)
    del old,new,change; gc.collect()
    # Read each medical array once; release before constructing triangle trees.
    tissue=[]
    for name,p in cfg["inputs"]["masks"].items():
        file=(cfg_path.parent/Path(p)).resolve(); inputs[str(file)]=core.sha(file)
        image=nib.load(file)
        if image.shape!=before.shape or not np.array_equal(image.affine,before.affine):
            raise ValueError("Diagnostic tissue grid mismatch")
        data=np.asanyarray(image.dataobj); selected=data[tuple(voxels.T)]>0
        tissue.append(dict(id=name,changed_voxels_inside=int(selected.sum()),
            by_component_26=[dict(component=r["component"],voxels=int(selected[labels==r["component"]].sum())) for r in c26]))
        del data
    ct_path=(cfg_path.parent/Path(cfg["inputs"]["ct"])).resolve(); inputs[str(ct_path)]=core.sha(ct_path)
    ct=nib.load(ct_path)
    if ct.shape!=before.shape or not np.array_equal(ct.affine,before.affine):
        raise ValueError("Diagnostic CT grid mismatch")
    hu_data=np.asanyarray(ct.dataobj); hu=np.asarray(hu_data[tuple(voxels.T)],float); del hu_data
    body["HU_changed_voxels"]=dict(min=float(hu.min()),max=float(hu.max()),median=float(np.median(hu)),
        p05=float(np.percentile(hu,5)),p95=float(np.percentile(hu,95)),
        above_HU_threshold=int((hu>cfg["body"]["hu_threshold"]).sum()))
    for r in c26:
        v=hu[labels==r["component"]]
        r["HU"]=dict(min=float(v.min()),max=float(v.max()),median=float(np.median(v)))
    tree=cKDTree(lps)
    contacts=[]
    for name,row in cfg["contacts"]["physical_centres_lps_mm"].items():
        d,j=tree.query(row["raw_lps_mm"])
        contacts.append(dict(id=name,nearest_changed_voxel=voxels[j].tolist(),distance_mm=float(d),component_26=int(labels[j])))
    before_stl=trimesh.load(files[2],force="mesh"); after_stl=trimesh.load(files[3],force="mesh")
    meshes=dict(before=dict(vertices=len(before_stl.vertices),faces=len(before_stl.faces),
        volume_ml=float(before_stl.volume/1000),euler=int(before_stl.euler_number),watertight=bool(before_stl.is_watertight)),
        after=dict(vertices=len(after_stl.vertices),faces=len(after_stl.faces),
        volume_ml=float(after_stl.volume/1000),euler=int(after_stl.euler_number),watertight=bool(after_stl.is_watertight)))
    threshold=cfg["surface_qc"].get("maximum_sampled_topology_correction_mm",
                                  cfg["surface_qc"].get("maximum_sampled_correction_mm"))
    threshold=core.positive(threshold,"declared topology diagnostic threshold")
    out.mkdir(parents=True,exist_ok=False)
    core.write_new(out/"changed_voxels.local.json",dict(body,contacts=contacts,tissues=tissue,sources=inputs))
    np.savez_compressed(out/"changed_voxels.local.npz",voxels=voxels,lps_mm=lps,added=added,HU=hu,component_26=labels)
    print(json.dumps(dict(progress="changed voxels summarized",changed=len(voxels),components_26=len(c26))),flush=True)
    reports={}; radius=float(np.linalg.norm(before.header.get_zooms()))
    for name,a,b in [("original_to_regularized",before_stl,after_stl),("regularized_to_original",after_stl,before_stl)]:
        report,sample=distances(a,b,count,threshold,tree,voxels,labels)
        core.write_new(out/(name+".local.json"),report)
        np.savez_compressed(out/(name+".local.npz"),points_lps_mm=sample[0],distance_mm=sample[1],source_face_id=sample[2])
        print(json.dumps(dict(progress=name,**{k:report[k] for k in ("max_mm","p95_mm","above_declared_threshold")})),flush=True)
        report["local_vertex_probe"]=local_vertex_probe(a,b,tree,radius,voxels,labels,vertex_cap)
        core.write_new(out/(name+"_local_vertices.local.json"),report["local_vertex_probe"])
        reports[name]=report
        a._cache.clear(); b._cache.clear(); gc.collect()
    core.verify_sources(inputs)
    result=dict(schema="ct_fem_saved_body_distance_diagnostic_v1",scope="Saved body STL/voxel diagnostics only",
        status="diagnostic_not_acceptance",source_directory=str(source),body=body,mesh=meshes,
        contacts=contacts,tissues=tissue,distances=reports,sources=inputs,
        FEM_was_run=False,MATLAB_was_run=False,TetGen_was_run=False,remesh_was_run=False,
        quantitative_result_accepted=False,independent_anatomical_validation=False)
    core.write_new(out/"body_distance_diagnostic.local.json",result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True);parser.add_argument("--out",required=True);parser.add_argument("--config",required=True)
    parser.add_argument("--sample-count",type=int,default=15000);parser.add_argument("--local-vertex-cap",type=int,default=40000)
    args=parser.parse_args()
    if not 1<=args.sample_count<=15000 or not 1<=args.local_vertex_cap<=40000:
        raise ValueError("Explicit bounded diagnostic resources required")
    run(args.input,args.out,args.config,args.sample_count,args.local_vertex_cap)
