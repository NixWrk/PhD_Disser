"""Fresh EIDORS material assembly and bounded 3 x 6 static heart pilot.

prepare is read/convert/hash only. run exports one geometry per MATLAB process,
then solves one montage per Python process (two RHS, direct and reciprocal).
No old rho blocks/Jacobians are used. Contact C may be reused ONLY on the exact
bound unchanged mesh. Different montages NEVER share a factorization.
Outputs and local machine paths remain in --output, not in tracked examples.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
GEOMETRIES = ('nominal', 'sphere', 'ellipsoid')
MONTAGES = tuple('tepc_'+str(i) for i in range(2, 8))
STATUS = 'static_electrical_pilot_not_anatomically_or_physically_validated'

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)

def require(value,message):
    if not value: raise ValueError(message)

def configure(deps,threads):
    sys.path.insert(0,str(ROOT/'tools'))
    import run_electrode_sensitivity as runner
    runner.initialize(deps,threads)
    global np, sp, sio, h5py, psutil
    np,sp,sio,h5py=runner.np,runner.sp,runner.sio,runner.h5py
    import psutil
    return runner

def resource_snapshot():
    mem=psutil.virtual_memory()
    return {'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'total_gib':mem.total/2**30,'available_gib':mem.available/2**30,
            'logical_cpus':os.cpu_count(),'own_rss_gib':psutil.Process().memory_info().rss/2**30}

def verify_files(records):
    for rec in records:
        require(sha(rec['path'])==rec['sha256'],'Changed input/code: '+rec['path'])

def prepare(args):
    configure(args.deps,args.threads)
    out=args.output.resolve(); maps=args.maps.resolve()
    require(not (out/'plan.json').exists(),'Plan already exists; use run to resume')
    out.mkdir(parents=True,exist_ok=True)
    pre=read(maps/'preflight.json'); checks={}
    def bind(path,expected=None):
        path=Path(path).resolve(); digest=sha(path)
        require(expected is None or digest==expected,'Hash mismatch: '+str(path))
        checks[str(path)]={'path':str(path),'sha256':digest}
        return path
    bind(maps/'preflight.json')
    require(pre['input_jacobian_qc_passed'] and pre['overlay_audit_passed'],'Input geometry/overlay QC failed')
    require(all(not v['external_boundary_conflict'] for v in pre['boundary_qc'].values()),'External boundary conflict')
    for entry in pre['sources'].values(): bind(maps/entry['path'],entry['sha256'])
    manifest_path=(maps/pre['sources']['manifest']['path']).resolve(); baseline=manifest_path.parent
    manifest=read(manifest_path)
    source=bind(baseline/manifest['source']['prepared_fem'],pre['sources']['prepared_fem']['sha256'])
    prep_path=bind(baseline/'contacts/preparation.json'); prep=read(prep_path)
    profile=read(bind(baseline/'rho_profile.json'))
    plan_old=read(bind(baseline/'surface_plan.json'))
    require(prep['status']=='passed' and prep['prepared_sha256']==sha(source),'Unbound contacts')
    require(prep['manifest_sha256']==sha(manifest_path),'Contact manifest changed')
    require(plan_old['profile_sha256']==sha(baseline/'rho_profile.json'),'Rho profile changed')
    require(prep['contact']['diameter_mm']==5 and profile['frequency_hz']==50000,'Expected 5 mm / 50 kHz')
    require(prep['electrode_order']==['I_plus','V_plus','V_minus','I_minus'],'Electrode roles mismatch')
    # This pilot uses the exact midpoint state of the existing five-node rho grid.
    rho={k:float(plan_old['axes_rho'][k][2]) for k in ('soft','heart','lung')}
    rho['bone']=float(profile['bone_fixed_rho_ohm_m'])
    sigma=[1/rho[k] for k in ('soft','lung','heart','bone')]
    contacts=[]
    require(tuple(v['id'] for v in prep['montages'])==MONTAGES,'Expected TEPC2..7')
    for v in prep['montages']:
        require(v['status']=='passed' and v['matrix_error']<1e-12,'Contact preparation failed')
        contact=bind(baseline/v['contact_matrix'],v['contact_sha256'])
        bind(baseline/v['geometry_file'],v['geometry_sha256'])
        bind(baseline/v['source_json'],v['source_json_sha256'])
        contacts.append({'id':v['id'],'path':str(contact),'sha256':sha(contact),
                         'areas_mm2':v['areas_mm2'],'actual_centres_xyz_mm':v['actual_centres_xyz_mm']})
    with h5py.File(source) as f:
        nn=f['fmdl_m/nodes'].shape[1]; ne=f['fmdl_m/elems'].shape[1]
        original=f['tissue_id'][:].ravel().astype(np.uint8)
        ground=int(f['fmdl_m/gnd_node'][0,0])-1
    require(nn==pre['metadata']['counts']['nodes'] and ne==len(original),'Mesh count mismatch')
    bg=bind(maps/pre['background']['path'],pre['background']['sha256'])
    background=np.load(bg,allow_pickle=False)
    require(np.array_equal(background,np.where(original==3,1,original)),'Immutable background mismatch')
    candidates={}
    for name in GEOMETRIES:
        cc=pre['candidate_contracts'][name]; cp=bind(maps/cc['path'],cc['sha256']); c=read(cp)
        lp=bind(cp.parent/c['labels'],c['labels_sha256']); labels=np.load(lp,allow_pickle=False)
        require(c['source_prepared_sha256']==sha(source),'Candidate belongs to another mesh')
        require(labels.shape==(ne,) and labels.dtype==np.uint8 and set(np.unique(labels))=={1,2,3,4},'Invalid labels')
        require(np.array_equal(labels, np.where(labels==3,3,background)),'Overlay does not restore background')
        if name=='nominal': require(np.array_equal(labels,original),'Nominal differs from source labels')
        mat=out/(name+'_labels.mat'); sio.savemat(mat,{'tissue_id':labels[:,None]},do_compression=True)
        candidates[name]={'labels_mat':mat.name,'labels_mat_sha256':sha(mat),
                          'labels_npy_sha256':sha(lp),'export_directory':name+'/export',
                          'counts_by_id':[int(np.sum(labels==i)) for i in range(1,5)]}
    code=[Path(__file__),ROOT/'src/run_heart_replacement_electrical_pilot.m',
          ROOT/'src/trkg4_export_sensitivity_model.m',ROOT/'src/trkg4_cem_contact_matrix.m',
          ROOT/'src/trkg4_file_sha256.m',ROOT/'tools/run_electrode_sensitivity.py']
    implementation=[{'path':str(p.resolve()),'sha256':sha(p)} for p in code]
    plan={'schema':'heart_replacement_electrical_pilot_v1','status':STATUS,
          'source_prepared':str(source),'source_prepared_sha256':sha(source),
          'inputs':list(checks.values()),'implementation':implementation,'candidates':candidates,
          'contacts':contacts,'threads':args.threads,'memory_limit_gib':args.memory_limit_gib,
          'rho_ohm_m':rho,'conductivity_by_tissue_id':sigma,
          'rho_choice':'Existing TEPC five-node-grid midpoint; modelling state, not measured physiology',
          'frequency_hz':50000,'model':'real scalar resistive CEM; no permittivity',
          'contact':prep['contact'],'node_count':nn,'element_count':ne,'ground_zero_based':ground,
          'runtime':{'python':sys.executable,'deps':str(args.deps.resolve()),
                     'matlab':str(args.matlab.resolve()),'eidors_startup':str(args.eidors_startup.resolve())},
          'criteria':{'relative_residual':1e-7,'reciprocity_atol_ohm':1e-8,'reciprocity_rtol':1e-8,
                      'current_atol_A':1e-7,'ground_residual_atol_A':1e-7,'matrix_rtol':1e-12},
          'criteria_status':'Numerical consistency checks; not physical/clinical accuracy thresholds',
          'limitations':['Static mesh; no registration to 4D CT','Whole heart is effective blood/myocardium material',
                         'Centroid element classification; no partial-volume integration',
                         'Spatial convergence and physical validation not established',
                         'Different CEM contacts require separate matrix factorizations']}
    write(out/'plan.json',plan); (out/'plan.sha256').write_text(sha(out/'plan.json'))
    write(out/'preflight.json',{'status':'ready_for_serial_assembly','resources':resource_snapshot(),
          'nodes':nn,'elements':ne,'states':18,'new_material_blocks_per_geometry':4,
          'old_blocks_loaded':False,'plan_sha256':sha(out/'plan.json')})
    print(json.dumps({'output':str(out),'rho':rho,'resources':resource_snapshot()},indent=2))

def load_plan(out):
    p=read(out/'plan.json')
    require(sha(out/'plan.json')==(out/'plan.sha256').read_text().strip(),'Plan seal changed')
    verify_files(p['implementation']); verify_files(p['inputs'])
    for v in p['candidates'].values():
        require(sha(out/v['labels_mat'])==v['labels_mat_sha256'],'Converted labels changed')
    return p

def monitored(command,out,stage,limit):
    log=out/(stage+'.log')
    env=os.environ.copy(); env['PYTHONDONTWRITEBYTECODE']='1'
    with log.open('a',encoding='utf-8') as f:
        child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env,
                 creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        peak=0.; started=time.time()
        while child.poll() is None:
            try:
                proc=psutil.Process(child.pid); family=[proc]+proc.children(recursive=True)
                rss=sum(p.memory_info().rss for p in family if p.is_running())/2**30
                peak=max(peak,rss)
                state={'status':'running','stage':stage,'pid':child.pid,'command':command,
                       'rss_gib':rss,'peak_rss_gib':peak,'elapsed_s':time.time()-started,
                       'available_gib':psutil.virtual_memory().available/2**30}
                write(out/'status.json',state)
                if rss>limit or state['available_gib']<4:
                    for p in reversed(family):
                        try:p.kill()
                        except psutil.NoSuchProcess:pass
                    child.wait(); raise MemoryError('Memory guard stopped '+stage)
            except psutil.NoSuchProcess:pass
            time.sleep(2)
    state={'status':'completed' if child.returncode==0 else 'failed','stage':stage,
           'pid':child.pid,'exit_code':child.returncode,'peak_rss_gib':peak,'elapsed_s':time.time()-started,
           'command':command}
    write(out/(stage+'_process.json'),state);write(out/'status.json',state)
    require(child.returncode==0,'Process failed; see '+str(log))

def validate_export(out,p,name):
    directory=out/p['candidates'][name]['export_directory']; report=read(directory/'export.json')
    require(report['plan_sha256']==sha(out/'plan.json') and report['candidate']==name,'Export identity mismatch')
    require(report['source_prepared_sha256']==p['source_prepared_sha256'] and
            report['labels_mat_sha256']==p['candidates'][name]['labels_mat_sha256'],'Export input mismatch')
    require(report['matrix_error']<1e-12 and report['heterogeneous_matrix_error']<1e-12,'Export assembly failed')
    for filename,key in [('prepared_fem.mat','prepared_fem_sha256'),('contact.mat','contact_sha256')]:
        require(sha(directory/filename)==report[key],'Export artifact changed')
    return directory,report

def solve_checked(K,C,q,ground,criteria,memory_limit,out,backend='pardiso'):
    require(np.isfinite(K.data).all() and np.isfinite(q).all(),'Nonfinite system')
    require(np.max(np.abs(q.sum(axis=0)))<1e-14,'Unbalanced imposed currents')
    asym=float(sp.linalg.norm(K-K.T)/sp.linalg.norm(K))
    require(asym<criteria['matrix_rtol'],'Matrix not symmetric')
    free=np.delete(np.arange(K.shape[0]),ground); reduced=K[free,:][:,free].tocsr()
    require(np.all(reduced.diagonal()>0),'Nonpositive diagonal')
    scale=1/np.sqrt(reduced.diagonal()); D=sp.diags(scale)
    A=sp.triu(D@reduced@D,format='csr'); rhs=np.asfortranarray(scale[:,None]*q[free])
    memory={}; started=time.time()
    if backend=='pardiso':
        from pypardiso import PyPardisoSolver
        solver=PyPardisoSolver(mtype=2)
        try:
            for i,v in {1:1,2:3,8:3,10:13,18:-1,19:-1}.items():solver.set_iparm(i,v)
            # Explicit phases: symbolic estimate BEFORE numerical factorization.
            solver._check_A(A); rhs=solver._check_b(A,rhs)
            solver.set_phase(11);solver._call_pardiso(A,rhs)
            kb=[int(solver.get_iparm(i)) for i in (15,16,17)]
            require(all(v>=0 for v in kb) and kb[2]>0,'PARDISO memory estimate unavailable')
            rss=psutil.Process().memory_info().rss/2**30
            extra=max(kb[0],kb[1]+kb[2])*1024/2**30
            memory={'iparm_15_16_17_kib':kb,'rss_after_symbolic_gib':rss,
                    'conservative_rss_plus_solver_estimate_gib':rss+extra+1,
                    'available_gib':psutil.virtual_memory().available/2**30}
            if out: write(out/'factorization_memory.json',memory)
            require(rss+extra+1 < memory_limit and extra+5<memory['available_gib'],
                    'Symbolic memory estimate exceeds serial pilot budget')
            solver.set_phase(22);solver._call_pardiso(A,rhs)
            solver.set_phase(33);x=solver._call_pardiso(A,rhs)
        finally:solver.free_memory(everything=True)
    else:
        require(A.shape[0]<=2000,'Scipy backend only for synthetic self-check')
        full=A+A.T-sp.diags(A.diagonal()); np.linalg.cholesky(full.toarray())
        x=sp.linalg.spsolve(full,rhs)
    fields=np.zeros_like(q); fields[free]=scale[:,None]*x
    residual=K@fields-q
    relative=np.linalg.norm(residual[free],axis=0)/np.linalg.norm(q[free],axis=0)
    currents=(C@fields)[-4:]; current_error=float(np.max(np.abs(currents-q[-4:])))
    ground_error=float(np.max(np.abs(residual[ground])))
    z=float(q[:,1]@fields[:,0]); zr=float(q[:,0]@fields[:,1]); rec=abs(z-zr)
    require(np.isfinite(fields).all() and np.isfinite(relative).all(),'Nonfinite solution')
    require(float(max(relative))<criteria['relative_residual'],'Residual failed')
    require(current_error<criteria['current_atol_A'],'Electrode current recovery failed')
    require(ground_error<criteria['ground_residual_atol_A'],'Ground residual failed')
    require(rec<=criteria['reciprocity_atol_ohm']+criteria['reciprocity_rtol']*max(abs(z),abs(zr)),'Reciprocity failed')
    row={'Z_ohm':z,'reciprocal_Z_ohm':zr,'max_relative_residual':float(max(relative)),
         'reciprocity_absolute_ohm':rec,'max_current_error_A':current_error,
         'max_ground_residual_A':ground_error,'recovered_currents_A':currents.tolist(),
         'matrix_asymmetry_relative':asym,'solve_seconds':time.time()-started,
         'factorization_memory':memory,'rhs_count':2,'factorizations':1}
    return fields,row

def worker(args):
    out=args.output.resolve(); p=load_plan(out); runner=configure(Path(p['runtime']['deps']),p['threads'])
    name=args.geometry; mid=args.montage; directory,export=validate_export(out,p,name)
    target=out/name/mid
    require(not target.exists(),'State directory exists; complete states resume through run, incomplete require audit')
    target.mkdir(parents=True)
    model=runner.load_model(directory/'prepared_fem.mat')
    require(len(model['blocks'])==4 and model['ground']==p['ground_zero_based'],'New block model mismatch')
    expected=sio.loadmat(out/p['candidates'][name]['labels_mat'])['tissue_id'].ravel()
    require(np.array_equal(model['tissue_id'],expected),'Exported tissue map differs')
    params=[{'id':k,'block_indices':[i]} for i,k in enumerate(('soft','lung','heart','bone'))]
    blocks=runner.parameter_blocks(model,params,4); nn=len(model['nodes'])
    contact=next(c for c in p['contacts'] if c['id']==mid)
    # Original hash is intentional: C is bound to ORIGINAL unchanged geometry.
    # The new export is independently bound to that same source + new labels.
    C=runner.load_contact(contact['path'],nn,4,model['ground'],p['source_prepared_sha256'])
    K=C.copy()
    for k,B in blocks.items(): K=K+(1/p['rho_ohm_m'][k])*B
    montage={'electrodes':['E1','E2','E3','E4'],'channels':[{'drive':[1,0,0,-1],'measure':[0,1,-1,0]}]}
    q=runner.channel_rhs(nn,montage)
    fields,row=solve_checked(K,C,q,model['ground'],p['criteria'],p['memory_limit_gib'],target)
    row.update({'geometry':name,'montage':mid,'scientific_status':STATUS,'frequency_hz':p['frequency_hz']})
    for k,B in blocks.items():
        ds=float(-fields[:,1]@(B@fields[:,0])); row['dZ_dsigma_'+k]=ds
        row['dZ_drho_'+k]=-ds/p['rho_ohm_m'][k]**2
    # Arrays are retained for audit/recovered-current and residual re-evaluation.
    np.savez_compressed(target/'fields.npz',potentials_ohm=fields)
    row['plan_sha256']=sha(out/'plan.json');row['prepared_sha256']=export['prepared_fem_sha256']
    row['contact_sha256']=contact['sha256'];row['rho_ohm_m']=p['rho_ohm_m']
    row['runtime']={k:importlib.metadata.version(k) for k in ('numpy','scipy','pypardiso','h5py')}
    write(target/'result.json',row)
    verify_files(p['implementation']);verify_files(p['inputs'])
    write(target/'completion.json',{'status':'completed_numerical_checks_passed','plan_sha256':sha(out/'plan.json'),
        'hashes':{f:sha(target/f) for f in ('fields.npz','result.json','factorization_memory.json')}})
    print(json.dumps(row,indent=2))

def completed(out,p,name,mid):
    target=out/name/mid
    if not (target/'completion.json').exists():return None
    c=read(target/'completion.json')
    require(c['status']=='completed_numerical_checks_passed' and c['plan_sha256']==sha(out/'plan.json'),'Completion identity mismatch')
    for f,h in c['hashes'].items():require(sha(target/f)==h,'Completed artifact changed')
    row=read(target/'result.json'); _,report=validate_export(out,p,name)
    require(row['prepared_sha256']==report['prepared_fem_sha256'],'Result prepared export mismatch')
    require(row['contact_sha256']==next(c['sha256'] for c in p['contacts'] if c['id']==mid),'Contact changed')
    return row

def summarize(out,p):
    rows=[]
    for name in GEOMETRIES:
        for mid in MONTAGES:
            if (out/name/mid/'completion.json').exists(): rows.append(completed(out,p,name,mid))
    reference={r['montage']:r['Z_ohm'] for r in rows if r['geometry']=='nominal'}
    table=[]
    for r in rows:
        v={k:r[k] for k in ('geometry','montage','Z_ohm','reciprocal_Z_ohm','max_relative_residual',
                            'reciprocity_absolute_ohm','max_current_error_A','max_ground_residual_A')}
        z0=reference.get(r['montage']); v['delta_Z_vs_nominal_ohm']=None if z0 is None else r['Z_ohm']-z0
        v['delta_Z_vs_nominal_percent']=None if z0 is None or abs(z0)<1e-12 else 100*(r['Z_ohm']-z0)/abs(z0)
        table.append(v)
    write(out/'summary.json',{'status':'complete' if len(rows)==18 else 'partial','completed':len(rows),
                             'required':18,'scientific_status':STATUS,'rows':table,'limitations':p['limitations']})
    if table:
        with (out/'summary.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    return len(rows)

def run(args):
    out=args.output.resolve(); p=load_plan(out); configure(Path(p['runtime']['deps']),p['threads'])
    lock=out/'run.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        for name in args.geometries:
            exp=out/p['candidates'][name]['export_directory']
            if not (exp/'export.json').exists():
                require(not exp.exists(),'Incomplete export requires audit before new attempt')
                quote=lambda s: "'"+str(s).replace('\\','/').replace("'","''")+"'"
                code='run('+quote(p['runtime']['eidors_startup'])+');addpath('+quote(ROOT/'src')+');'
                code+='run_heart_replacement_electrical_pilot('+quote(out/'plan.json')+','+quote(name)+');'
                env_threads=str(p['threads'])
                for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]=env_threads
                monitored([p['runtime']['matlab'],'-batch',code],out,'export_'+name,p['memory_limit_gib'])
            validate_export(out,p,name)
            for mid in args.montages:
                if completed(out,p,name,mid) is not None:continue
                cmd=[p['runtime']['python'],'-B','-X','utf8',str(Path(__file__).resolve()),'worker',
                     '--output',str(out),'--geometry',name,'--montage',mid]
                monitored(cmd,out,'solve_'+name+'_'+mid,p['memory_limit_gib'])
                completed(out,p,name,mid);summarize(out,p)
        n=summarize(out,p);write(out/'status.json',{'status':'complete' if n==18 else 'selected_states_complete','completed_states':n,'required_states':18})
    finally:lock.unlink(missing_ok=True)

def selftest(args):
    runner=configure(args.deps,2)
    n=6; K=sp.lil_matrix((n+4,n+4));C=sp.lil_matrix(K.shape)
    for i in range(n-1):
        w=1+i/3
        for a,b,v in [(i,i,w),(i+1,i+1,w),(i,i+1,-w),(i+1,i,-w)]:K[a,b]+=v
    for j,i in enumerate((1,2,4,5)):
        for a,b,v in [(i,i,2),(n+j,n+j,2),(i,n+j,-2),(n+j,i,-2)]:C[a,b]+=v
    K=(K+C).tocsr();C=C.tocsr();q=np.zeros((n+4,2));q[n:]=np.array([[1,0],[0,1],[0,-1],[-1,0]])
    crit={'relative_residual':1e-7,'matrix_rtol':1e-12,'current_atol_A':1e-7,
          'ground_residual_atol_A':1e-7,'reciprocity_atol_ohm':1e-8,'reciprocity_rtol':1e-8}
    f,a=solve_checked(K,C,q,0,crit,32,None,'scipy')
    g,b=solve_checked(K,C,q,0,crit,32,None,'pardiso')
    require(np.allclose(f,g,atol=1e-11),'PARDISO disagrees with dense/sparse independent solution')
    rejected=0
    for badK,badC,badq in [(K,C,q+1),(K,C*1.01,q)]:
        try:solve_checked(badK,badC,badq,0,crit,32,None,'scipy')
        except ValueError:rejected+=1
    require(rejected==2,'Negative QC tests did not reject')
    write(args.output/'selftest.json',{'status':'passed','test':'synthetic resistor-network CEM; direct vs reciprocal; PARDISO phases vs scipy; corrupt currents rejected',
                                     'negative_tests_rejected':rejected,'scipy':a,'pardiso':b})
    print('SELFTEST PASSED')

def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('prepare'); a.add_argument('--maps',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--deps',type=Path,required=True);a.add_argument('--matlab',type=Path,required=True)
    a.add_argument('--eidors-startup',type=Path,required=True);a.add_argument('--threads',type=int,default=4)
    a.add_argument('--memory-limit-gib',type=float,default=32)
    a=sub.add_parser('run');a.add_argument('--output',type=Path,required=True)
    a.add_argument('--geometries',nargs='+',choices=GEOMETRIES,default=list(GEOMETRIES))
    a.add_argument('--montages',nargs='+',choices=MONTAGES,default=list(MONTAGES))
    a=sub.add_parser('worker');a.add_argument('--output',type=Path,required=True)
    a.add_argument('--geometry',choices=GEOMETRIES,required=True);a.add_argument('--montage',choices=MONTAGES,required=True)
    a=sub.add_parser('selftest');a.add_argument('--deps',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('status');a.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.mode=='prepare':
        require(1<=args.threads<=4 and 0<args.memory_limit_gib<=32,'Pilot limited to four threads and 32 GiB')
        prepare(args)
    elif args.mode=='run':run(args)
    elif args.mode=='worker':worker(args)
    elif args.mode=='selftest':selftest(args)
    else: print(json.dumps(read(args.output/'status.json'),indent=2))
if __name__=='__main__':main()
