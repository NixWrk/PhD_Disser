"""Bounded, isolated TetGen diagnosis. Outputs are never prepared FEM artifacts."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def worker(args):
    if args.runtime:sys.path.insert(0,str(Path(args.runtime).resolve()))
    import numpy as np
    import trimesh
    import tetgen
    from scipy.io import savemat
    root=Path(__file__).resolve().parent;sys.path.insert(0,str(root))
    import ct_fem_core as core
    out=Path(args.out);source=Path(args.surface);mesh=trimesh.load(source,force='mesh')
    quality=args.mode!='plc'
    settings=dict(plc=True,quality=quality,verbose=2,quiet=False,nobisect=False,
        steinerleft=500000,opt_scheme=0 if args.mode!='quality_opt1' else 1,smooth_maxiter=0,
        nomergefacet=not args.merge_coplanar,epsilon=1e-10 if args.epsilon is None else args.epsilon)
    if args.mode in ('quality_georg_original','quality_stronger_optimization'):
        # Exact explicit equivalents of 0.8.3 defaults in the preserved
        # bff3ad3f... producer, before later changes to georg_quality_mesh_prepare.
        settings.update(opt_scheme=7,smooth_maxiter=7,nomergefacet=False,
                        epsilon=1e-8 if args.epsilon is None else args.epsilon)
    if args.mode=='quality_stronger_optimization':
        settings.update(opt_max_asp_ratio=100.,optmaxdihedral=170.,opt_iterations=6)
    if quality:settings.update(minratio=1.5,mindihedral=8.,fixedvolume=True,maxvolume=8.**3/6)
    write(out/'inputs.local.json',dict(source_sha256=core.sha(source),tetgen_version=tetgen.__version__,settings=settings))
    print('TetGen',tetgen.__version__,'mode',args.mode,'vertices',len(mesh.vertices),'faces',len(mesh.faces),flush=True)
    start=time.monotonic()
    try:
        nodes,tets,*_=tetgen.TetGen(mesh.vertices,mesh.faces.astype(np.int32)).tetrahedralize(**settings)
        tets=core.kernels()['orient'](nodes,tets)
        # Save the diagnostic before shape QC, so rejected elements remain inspectable.
        savemat(out/'diagnostic_mesh.local.mat',dict(nodes_mm=nodes,elems=tets+1))
        qc=core.kernels()['quality'](nodes,tets)
        write(out/'result.local.json',dict(status='diagnostic_mesh_generated',elapsed_s=time.monotonic()-start,
            nodes=len(nodes),tetrahedra=len(tets),shape_qc=qc,FEM_was_run=False,quantitative_result_accepted=False))
        print('diagnostic result',len(nodes),len(tets),'shape passed',qc['passed'],flush=True)
    except Exception as error:
        write(out/'result.local.json',dict(status='failed',error_type=type(error).__name__,message=str(error),
            elapsed_s=time.monotonic()-start,FEM_was_run=False))
        raise


def run(args):
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-X','utf8','-u',str(Path(__file__).resolve()),'--worker','--surface',args.surface,
        '--out',str(out),'--mode',args.mode]
    if args.epsilon is not None:command+=['--epsilon',str(args.epsilon)]
    if args.runtime:command+=['--runtime',args.runtime]
    if args.merge_coplanar:command+=['--merge-coplanar']
    with (out/'stdout.local.log').open('w',encoding='utf8') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        write(out/'process.local.json',dict(pid=child.pid,timeout_s=args.timeout,started_unix_s=time.time()))
        try:code=child.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            child.kill();child.wait();code=child.returncode
            write(out/'result.local.json',dict(status='timeout',timeout_s=args.timeout,FEM_was_run=False))
    write(out/'process_exit.local.json',dict(exit_code=code))
    print(json.dumps(json.loads((out/'result.local.json').read_text(encoding='utf8')),ensure_ascii=False)[:1100] if (out/'result.local.json').exists() else 'worker crashed before a result',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--surface',required=True);p.add_argument('--out',required=True)
    p.add_argument('--runtime');p.add_argument('--mode',choices=['plc','quality_opt0','quality_opt1','quality_georg_original','quality_stronger_optimization'],default='plc')
    p.add_argument('--timeout',type=float,default=120);p.add_argument('--worker',action='store_true')
    p.add_argument('--merge-coplanar',action='store_true',help='Diagnostic change of one TetGen boundary option; geometry QC remains separate')
    p.add_argument('--epsilon',type=float,help='Explicit diagnostic TetGen tolerance; not an anatomical error budget')
    a=p.parse_args();(worker if a.worker else run)(a)
