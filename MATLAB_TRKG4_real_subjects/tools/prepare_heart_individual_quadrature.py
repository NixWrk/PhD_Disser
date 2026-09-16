"""Refine only quadrature for an already sealed individual-heart probe design.

The source design is read-only. Independent new material states and EIDORS
blocks are required; no existing result seal is rewritten.
"""
from pathlib import Path
import argparse
import copy
import shutil
import time
import numpy as np

import heart_replacement_electrical_pilot as core
import heart_fractional_electrical_pilot as fractional
import heart_individual_electrical_stage as stage
from heart_individual_fractional_materials import TetrahedralHeart
from heart_fractional_materials import uniform_tet_barycentric
from heart_individual_fractional_fast import integrate_individual_fractions_fast

read, write, sha, require=core.read,core.write,core.sha,core.require
ROOT=Path(__file__).resolve().parents[1]


def prepare(a):
    source=a.source.resolve();out=a.output.resolve()
    original=fractional.load_plan(source)
    require(original.get("geometry")=="individual","Only original individual geometry")
    require(not out.exists() or not any(out.iterdir()),"Use a new output directory")
    runner=core.configure(Path(original["runtime"]["deps"]),original["threads"])
    mesh=Path(original["source_prepared"])
    geometric_source=next(Path(x["path"]) for x in original["inputs"]
                          if x["sha256"]==original["original_geometric_source_sha256"])
    with runner.h5py.File(geometric_source) as f:
        nodes=f["fmdl_m/nodes"][:].T
        heartids=np.flatnonzero(f["tissue_id"][:].ravel()==3)
        elems=f["fmdl_m/elems"][:,heartids].T.astype(np.int64)-1
        heart=TetrahedralHeart(nodes[elems])
    bounds=[]
    for state in original["states"]:
        scale=np.cbrt(1+state["volume_fraction"])
        shift=np.asarray(state["translation_m"])
        bounds.append(tuple(heart.centre_m+scale*(b-heart.centre_m)+shift for b in heart.bounds_m))
    union=(np.min([b[0] for b in bounds],axis=0),np.max([b[1] for b in bounds],axis=0))
    ids,vertices,ml,selected_labels,labels,total=stage.select_mesh(mesh,union)
    levels=tuple(sorted(set(a.levels)))
    require(a.electrical_level in levels,"Electrical level absent")
    bary=uniform_tet_barycentric(max(levels))
    out.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source/"background.mat",out/"background.mat")
    require(sha(out/"background.mat")==original["background_mat_sha256"],"Background copy mismatch")
    states=[];rows=[];diagnostic_rows=[]
    for old in original["states"]:
        tick=time.time()
        fractions,diagnostics=integrate_individual_fractions_fast(
            vertices,heart,volume_fraction=old["volume_fraction"],
            translation_m=old["translation_m"],levels=levels,barycentric=bary,
            batch_size=a.batch_size)
        if old["parameter"]=="baseline":
            require(np.array_equal(fractions,np.broadcast_to(selected_labels==3,fractions.shape)),
                    "Identity differs from original labels")
        for j,level in enumerate(levels):
            volume=float(ml@fractions[j])
            rows.append(dict(state=old["id"],parameter=old["parameter"],
                signed_step=old["signed_step"],points_per_tet=level,
                analytic_volume_ml=old["analytic_volume_ml"],material_volume_ml=volume,
                volume_error_percent=100*(volume/old["analytic_volume_ml"]-1)))
        state=copy.deepcopy(old)
        state["id"]=old["id"].replace("_q"+str(old["points_per_tet"])+"_",
                                    "_q"+str(a.electrical_level)+"_")
        state["points_per_tet"]=a.electrical_level
        mat=out/(state["id"]+"_fraction.mat")
        state["fraction_mat"]=mat.name
        fraction=np.zeros(len(labels),dtype=np.float32)
        fraction[ids]=fractions[levels.index(a.electrical_level)]
        runner.sio.savemat(mat,{"heart_fraction":fraction[:,None]},do_compression=True)
        state["fraction_mat_sha256"]=sha(mat)
        state["material_volume_ml"]=float(ml@fraction[ids])
        array=out/(state["id"]+"_geometry.npz")
        np.savez_compressed(array,levels=levels,fractions=fractions,element_indices=ids)
        state["geometry_npz_sha256"]=sha(array)
        state["geometry_npz"]=array.name
        states.append(state)
        diagnostic_rows.append(dict(state=state["id"],seconds=time.time()-tick,diagnostics=diagnostics))
        write(out/"geometry_progress.json",dict(completed=len(states),required=len(original["states"]),
              last_state=state["id"],last_seconds=time.time()-tick))
        print(state["id"],"geometry seconds",round(time.time()-tick,2),flush=True)
    write(out/"geometry_qc.json",dict(
        original_volume_ml=heart.volume_ml,original_centre_m=heart.centre_m.tolist(),
        original_heart_tetrahedra=len(heartids),selected_tetrahedra=len(ids),
        all_tetrahedra_scanned=len(labels),total_domain_volume_ml=total,
        original_identity_exact=True,geometry="individual",quadrature_levels=list(levels),
        rows=rows,classification_diagnostics=diagnostic_rows))
    p=copy.deepcopy(original)
    p.update(states=states,geometry_qc_sha256=sha(out/"geometry_qc.json"),
             quadrature_source_plan_sha256=sha(source/"plan.json"))
    additions=[Path(__file__),ROOT/"tools/heart_individual_fractional_fast.py"]
    p["implementation"]+= [dict(path=str(v.resolve()),sha256=sha(v)) for v in additions]
    p["inputs"]+= [dict(path=str(source/"plan.json"),sha256=sha(source/"plan.json"))]
    write(out/"plan.json",p)
    (out/"plan.sha256").write_text(sha(out/"plan.json"),encoding="ascii")
    write(out/"status.json",dict(status="ready_for_weighted_assembly",states=len(states),FEM_solved=False))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--levels",type=int,nargs="+",default=[4096,16384,65536])
    p.add_argument("--electrical-level",type=int,default=65536)
    p.add_argument("--batch-size",type=int,default=32)
    prepare(p.parse_args())
