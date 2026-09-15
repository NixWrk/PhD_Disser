"""Thirteen new TEPC-2 solutions from fractional EIDORS P1 material blocks.

4096 points: sphere and ellipsoid, baseline and volume +/-0.5%, +/-1%.
512 points: sphere baseline and +/-0.5%. No translation/montage sweep.
Uses the tested residual/current/reciprocity and process-memory guard from
heart_replacement_electrical_pilot, without changing that frozen calculation.
"""
from pathlib import Path
import argparse
import importlib.metadata
import json
import os
import sys
import time
import numpy as np
import heart_replacement_electrical_pilot as core
from heart_fractional_materials import expand_heart_fraction
ROOT=Path(__file__).resolve().parents[1]
read,write,sha,require=core.read,core.write,core.sha,core.require

def bind(files,path,expected=None):
    path=Path(path).resolve();digest=sha(path)
    require(expected is None or digest==expected,'Input hash mismatch: '+str(path))
    files[str(path)]={'path':str(path),'sha256':digest};return path

def prepare(a):
    source=a.fractions.resolve();old=a.baseline.resolve();out=a.output.resolve()
    require(not (out/'plan.json').exists(),'Plan already exists; use run to resume')
    base=core.load_plan(old);runner=core.configure(Path(base['runtime']['deps']),base['threads'])
    inputs={}
    identity=read(bind(inputs,source/'identity.json'));results=read(bind(inputs,source/'results.json'))
    completion=read(bind(inputs,source/'completion.json'))
    require(completion['results_sha256']==sha(source/'results.json') and completion['identity_sha256']==sha(source/'identity.json'),'Fraction completion mismatch')
    require(results['FEM_solved'] is False and results['states']==66,'Expected completed geometric fractions')
    require(identity['mesh_sha256']==base['source_prepared_sha256'],'Fractional mesh differs from electrical baseline')
    scan=read(bind(inputs,source/'mesh_scan.json',completion['mesh_scan_sha256']))
    ids=np.load(bind(inputs,source/'selected_element_indices_zero_based.npy',scan['selected_indices_sha256']))
    bind(inputs,old/'plan.json');bind(inputs,base['source_prepared'],base['source_prepared_sha256'])
    # The old plan binds all contacts; only TEPC-2 is consumed here.
    contact=next(c for c in base['contacts'] if c['id']=='tepc_2');bind(inputs,contact['path'],contact['sha256'])
    with runner.h5py.File(base['source_prepared']) as f:
        background=f['tissue_id'][:].ravel().astype(np.uint8);background[background==3]=1
    require(len(background)==scan['all_input_tetrahedra_scanned'],'Full mesh length mismatch')
    out.mkdir(parents=True,exist_ok=True);runner.sio.savemat(out/'background.mat',{'background':background[:,None]},do_compression=True)
    state_rows=read(source/'results.json')['rows'];states=[]
    design=[(g,4096,s) for g in ('sphere','ellipsoid') for s in (0.,-.01,.01,-.005,.005)]
    design += [('sphere',512,s) for s in (0.,-.005,.005)]
    for geometry,points,step in design:
        par='baseline' if step==0 else 'volume_fraction'
        row=next(r for r in state_rows if r['geometry']==geometry and r['points_per_tet']==points and r['parameter']==par and r['signed_step']==step)
        i=row['state_index'];receipt=read(bind(inputs,source/f'state_{i:03d}.json'))
        require(receipt['identity_sha256']==sha(source/'identity.json'),'Unbound fraction state')
        path=bind(inputs,source/f'state_{i:03d}.npz',receipt['array_sha256'])
        with np.load(path) as ar:
            index=int(np.flatnonzero(ar['levels']==points)[0]);h=expand_heart_fraction(len(background),ids,ar['heart_fraction'][index])
        require(np.all((h>=0)&(h<=1)),'Invalid material fraction')
        sign='base' if step==0 else ('m' if step<0 else 'p')+str(round(abs(step)*10000)).zfill(3)
        sid=f'{geometry}_q{points}_{sign}';mat=out/(sid+'_fraction.mat')
        runner.sio.savemat(mat,{'heart_fraction':h[:,None]},do_compression=True)
        states.append({'id':sid,'geometry':geometry,'points_per_tet':points,'volume_fraction':step,
                      'fraction_mat':mat.name,'fraction_mat_sha256':sha(mat),
                      'analytic_volume_ml':row['analytic_volume_ml'],'material_volume_ml':row['material_volume_ml'],
                      'source_fraction_state_index':i,'fraction_array_sha256':receipt['array_sha256']})
    code=[Path(__file__),ROOT/'src/run_heart_fractional_electrical_pilot.m',ROOT/'tools/heart_replacement_electrical_pilot.py',
          ROOT/'tools/heart_fractional_materials.py',ROOT/'tools/run_electrode_sensitivity.py',ROOT/'src/trkg4_cem_contact_matrix.m',ROOT/'src/trkg4_file_sha256.m']
    p={k:base[k] for k in ('source_prepared','source_prepared_sha256','runtime','threads','memory_limit_gib',
                          'node_count','element_count','ground_zero_based','rho_ohm_m','conductivity_by_tissue_id','criteria','frequency_hz')}
    p.update(schema='heart_fractional_electrical_pilot_v1',states=states,inputs=list(inputs.values()),contact=contact,
             background_mat_sha256=sha(out/'background.mat'),
             implementation=[{'path':str(path.resolve()),'sha256':sha(path)} for path in code],
             scientific_status='fractional_P1_static_numerical_micro_pilot_not_physically_validated',
             field_approximation='unchanged linear P1 basis; material integrals improved, no spatial mesh convergence',
             derivative='central finite differences per analytic whole-heart volume in ml; not stroke volume or physiological amplitude')
    write(out/'plan.json',p);(out/'plan.sha256').write_text(sha(out/'plan.json'))
    write(out/'preflight.json',{'status':'ready_for_weighted_assembly','states':13,'montage':'TEPC-2',
         'resources':core.resource_snapshot(),'old_material_blocks_read':False,'old_electrical_outputs_modified':False})
    print(json.dumps(read(out/'preflight.json'),indent=2))

def load_plan(out):
    p=read(out/'plan.json');require(sha(out/'plan.json')==(out/'plan.sha256').read_text(),'Plan seal changed')
    core.verify_files(p['inputs']);core.verify_files(p['implementation'])
    require(sha(out/'background.mat')==p['background_mat_sha256'],'Background conversion changed')
    for s in p['states']:require(sha(out/s['fraction_mat'])==s['fraction_mat_sha256'],'Fraction conversion changed')
    return p

def checked_export(out,p,s):
    d=out/s['id'];r=read(d/'export.json')
    require(r['plan_sha256']==sha(out/'plan.json') and r['fraction_mat_sha256']==s['fraction_mat_sha256'],'Weighted export provenance mismatch')
    require(r['status']=='weighted_P1_assembly_checks_passed' and r['homogeneous_error']<1e-12 and r['heterogeneous_error']<1e-12,'Weighted export checks failed')
    require(sha(d/'blocks.mat')==r['blocks_sha256'],'Weighted blocks changed')
    return r

def checked_result(out,p,s):
    d=out/s['id'];file=d/'completion.json'
    if not file.exists():return None
    seal=read(file);require(seal['plan_sha256']==sha(out/'plan.json'),'Completed state plan mismatch')
    for f,h in seal['hashes'].items():require(sha(d/f)==h,'Completed artifact changed')
    exp=checked_export(out,p,s);row=read(d/'result.json')
    require(row['blocks_sha256']==exp['blocks_sha256'] and row['contact_sha256']==p['contact']['sha256'],'Completed matrix/contact mismatch')
    return row

def worker(a):
    out=a.output.resolve();p=load_plan(out);runner=core.configure(Path(p['runtime']['deps']),p['threads'])
    s=next(s for s in p['states'] if s['id']==a.state);d=out/s['id'];exp=checked_export(out,p,s)
    require(not (d/'result.json').exists() and not (d/'fields.npz').exists(),'Incomplete solution needs inspection')
    blocks=[];sp=runner.sp
    with runner.h5py.File(d/'blocks.mat') as f:
        nn=int(f['nn'][0,0]);ground=int(f['ground'][0,0])-1
        require(nn==p['node_count'] and ground==p['ground_zero_based'],'Block dimensions or ground mismatch')
        for ref in f['blocks'][:].ravel():
            g=f[ref];n=int(g.attrs['MATLAB_sparse']);B=sp.csc_matrix((g['data'][:],g['ir'][:],g['jc'][:]),shape=(n,n)).tocsr()
            require(n==nn+4 and np.isfinite(B.data).all() and not B[nn:,:].nnz and not B[:,nn:].nnz,'Invalid fractional volume block')
            blocks.append(B)
    require(len(blocks)==4,'Expected four tissue blocks')
    C=runner.load_contact(p['contact']['path'],nn,4,ground,p['source_prepared_sha256'])
    K=C.copy()
    for sigma,B in zip(p['conductivity_by_tissue_id'],blocks):K=K+sigma*B
    q=np.zeros((nn+4,2));q[nn:]=np.array([[1,0],[0,1],[0,-1],[-1,0]])
    fields,row=core.solve_checked(K,C,q,ground,p['criteria'],p['memory_limit_gib'],d)
    row.update(s);row.update(montage='tepc_2',blocks_sha256=exp['blocks_sha256'],contact_sha256=p['contact']['sha256'],
                             plan_sha256=sha(out/'plan.json'),frequency_hz=50000,
                             scientific_status=p['scientific_status'])
    row['runtime']={k:importlib.metadata.version(k) for k in ('numpy','scipy','pypardiso','h5py')}
    np.savez_compressed(d/'fields.npz',potentials_ohm=fields);write(d/'result.json',row)
    core.verify_files(p['implementation']);core.verify_files(p['inputs'])
    write(d/'completion.json',{'status':'completed_numerical_checks_passed','plan_sha256':sha(out/'plan.json'),
                              'hashes':{f:sha(d/f) for f in ('fields.npz','result.json','factorization_memory.json')}})
    print(json.dumps(row,indent=2))

def summarize(out,p):
    rows=[r for s in p['states'] if (r:=checked_result(out,p,s)) is not None]
    lookup={(r['geometry'],r['points_per_tet'],r['volume_fraction']):r for r in rows};ds=[]
    for geometry,points,steps in [('sphere',4096,[.005,.01]),('ellipsoid',4096,[.005,.01]),('sphere',512,[.005])]:
        for step in steps:
            if not all((geometry,points,s) in lookup for s in (0.,-step,step)):continue
            base=lookup[geometry,points,0.];minus=lookup[geometry,points,-step];plus=lookup[geometry,points,step]
            analytic=2*step*base['analytic_volume_ml'];material=plus['material_volume_ml']-minus['material_volume_ml']
            ds.append({'geometry':geometry,'points_per_tet':points,'step_fraction':step,
                       'Z_base_ohm':base['Z_ohm'],'Z_minus_ohm':minus['Z_ohm'],'Z_plus_ohm':plus['Z_ohm'],
                       'dZ_dV_analytic_ohm_per_ml':(plus['Z_ohm']-minus['Z_ohm'])/analytic,
                       'dZ_dV_material_ohm_per_ml_diagnostic':(plus['Z_ohm']-minus['Z_ohm'])/material,
                       'dVmaterial_dVanalytic':material/analytic})
    pairs=[]
    for g in ('sphere','ellipsoid'):
        found={d['step_fraction']:d for d in ds if d['geometry']==g and d['points_per_tet']==4096}
        if len(found)==2:
            fine=found[.005]['dZ_dV_analytic_ohm_per_ml'];coarse=found[.01]['dZ_dV_analytic_ohm_per_ml']
            pairs.append({'comparison':'step_halving','geometry':g,'difference_ohm_per_ml':coarse-fine,
                          'relative_difference':None if abs(fine)<1e-14 else abs(coarse-fine)/abs(fine)})
    fine=next((d for d in ds if d['geometry']=='sphere' and d['points_per_tet']==4096 and d['step_fraction']==.005),None)
    coarse=next((d for d in ds if d['geometry']=='sphere' and d['points_per_tet']==512),None)
    if fine and coarse:
        x=fine['dZ_dV_analytic_ohm_per_ml'];y=coarse['dZ_dV_analytic_ohm_per_ml']
        pairs.append({'comparison':'quadrature_512_vs_4096','geometry':'sphere','difference_ohm_per_ml':y-x,
                      'relative_difference':None if abs(x)<1e-14 else abs(y-x)/abs(x)})
    write(out/'summary.json',{'status':'complete' if len(rows)==13 else 'partial','completed':len(rows),'required':13,
         'states':rows,'derivatives':ds,'comparisons':pairs,'scientific_status':p['scientific_status'],
         'spatial_convergence':False,'physical_validation':False,
         'limitations':['No geometric-derivative acceptance tolerance was fixed; observed differences are reported, not declared convergence',
           'Only one CEM montage and one conductivity state; no translation or patient generalization',
           'Residual/current/reciprocity test linear solves, not coarse-P1 field accuracy',
           'Finite shared quadrature can correlate geometry errors across +/- perturbations']})
    if rows:
        core.write(out/'numerical_qc.json',{'states':len(rows),'max_relative_residual':max(r['max_relative_residual'] for r in rows),
           'max_reciprocity_absolute_ohm':max(r['reciprocity_absolute_ohm'] for r in rows),
           'max_current_error_A':max(r['max_current_error_A'] for r in rows)})
    return len(rows)

def run(a):
    out=a.output.resolve();p=load_plan(out);core.configure(Path(p['runtime']['deps']),p['threads'])
    lock=out/'run.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        if any(not (out/s['id']/'export.json').exists() for s in p['states']):
            quote=lambda v:"'"+str(v).replace('\\','/').replace("'","''")+"'"
            code='run('+quote(p['runtime']['eidors_startup'])+');addpath('+quote(ROOT/'src')+');'
            code+='run_heart_fractional_electrical_pilot('+quote(out/'plan.json')+');'
            core.monitored([p['runtime']['matlab'],'-batch',code],out,'weighted_export',p['memory_limit_gib'])
        for s in p['states']:
            checked_export(out,p,s)
            if checked_result(out,p,s) is not None:continue
            cmd=[p['runtime']['python'],'-B','-X','utf8',str(Path(__file__).resolve()),'worker','--output',str(out),'--state',s['id']]
            core.monitored(cmd,out,'solve_'+s['id'],p['memory_limit_gib']);summarize(out,p)
        n=summarize(out,p);require(n==13,'Incomplete micro-pilot')
        write(out/'status.json',{'status':'complete','states':n,'new_weighted_exports':13,'FEM_solved':True})
    finally:lock.unlink(missing_ok=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('prepare');a.add_argument('--fractions',type=Path,required=True);a.add_argument('--baseline',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('run');a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('worker');a.add_argument('--output',type=Path,required=True);a.add_argument('--state',required=True)
    a=sub.add_parser('summarize');a.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a)
    elif a.mode=='run':run(a)
    elif a.mode=='worker':worker(a)
    else:print(summarize(a.output.resolve(),load_plan(a.output.resolve())))
if __name__=='__main__':main()
