"""Geometry-only audit of centroid classification; no FEM solve or material export."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
out=Path(__file__).resolve().parent
plan=json.loads((out/'plan.json').read_text());sys.path.insert(0,plan['runtime']['deps'])
import h5py
maps=next(Path(v['path']).parent for v in plan['inputs'] if Path(v['path']).name=='preflight.json')
params=json.loads((maps/'candidate_parameters.json').read_text())
with h5py.File(plan['source_prepared']) as f:
 nodes=f['fmdl_m/nodes'][:].T
 centre=np.array(params['original_heart']['centre_m'])
 ne=f['fmdl_m/elems'].shape[1]
 locations=[];weights=[];indices=[]
 # Bounding cube contains every tested candidate, including max 8 mm shifts.
 for start in range(0,ne,100000):
  e=f['fmdl_m/elems'][:,start:start+100000].T.astype('int64')-1
  v=nodes[e];c=v.mean(axis=1)
  take=np.max(np.abs(c-centre),axis=1)<.12
  if take.any():
   vv=v[take];vol=np.abs(np.einsum('ij,ij->i',vv[:,1]-vv[:,0],np.cross(vv[:,2]-vv[:,0],vv[:,3]-vv[:,0])))/6*1e6
   locations.append(c[take]);weights.append(vol);indices.append(start+np.flatnonzero(take))
centroids=np.concatenate(locations);volume=np.concatenate(weights);ids=np.concatenate(indices)
rows=[]
for name in ('sphere','ellipsoid'):
 p=params['candidates'][name];c=np.array(p['centre_m']);axes=np.array(p['semiaxes_m']);R=np.array(p['axes_columns'])
 def mask(cc,scale=1):return np.sum((((centroids-cc)@R)/(axes*scale))**2,axis=1)<=1+1e-12
 nominal=mask(c);allmask=np.load(maps/name/'labels.npy')==3
 assert np.array_equal(nominal,allmask[ids]) and nominal.sum()==allmask.sum()
 v0=float(volume[nominal].sum());V=p['analytic_volume_ml']
 for axis in range(3):
  for h in (.25,.5,1,2,4,8):
   vv=[];changes=[]
   for sign in (-1,1):
    shift=np.zeros(3);shift[axis]=sign*h*.001;m=mask(c+shift)
    vv.append(float(volume[m].sum()));changes.append(int(np.count_nonzero(m!=nominal)))
   rows.append({'geometry':name,'parameter':'xyz'[axis],'h_mm':h,'delta_volume_fraction':None,
                'V0_discrete_ml':v0,'Vminus_discrete_ml':vv[0],'Vplus_discrete_ml':vv[1],
                'analytic_deltaV_each_ml':0.,'max_spurious_volume_change_ml':max(abs(v-v0) for v in vv),
                'changed_elements_minus_plus':changes})
 for fraction in (.005,.01,.02,.04,.08):
  vv=[];changes=[]
  for sign in (-1,1):
   m=mask(c,(1+sign*fraction)**(1/3));vv.append(float(volume[m].sum()));changes.append(int(np.count_nonzero(m!=nominal)))
  rows.append({'geometry':name,'parameter':'volume','h_mm':None,'delta_volume_fraction':fraction,
               'V0_discrete_ml':v0,'Vminus_discrete_ml':vv[0],'Vplus_discrete_ml':vv[1],
               'analytic_deltaV_each_ml':fraction*V,
               'actual_central_volume_step_ml':(vv[1]-vv[0])/2,
               'central_step_error_percent':100*((vv[1]-vv[0])/2/(fraction*V)-1),
               'changed_elements_minus_plus':changes})
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
result={'status':'geometry_only_no_FEM','rows':rows,'script_sha256':sha(__file__),
        'source_prepared_sha256':plan['source_prepared_sha256'],
        'candidate_parameters_sha256':sha(maps/'candidate_parameters.json'),
        'scope':'Centroid labels only; sphere and ellipsoid; no individual-shape transformation; no physiological amplitudes',
        'conclusion':'Do not equate analytic constant volume with discrete constant material volume. This audit cannot establish electrical derivative convergence.'}
(out/'perturbation_map_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
for row in rows:
 if row['h_mm'] in (1,2) or row['delta_volume_fraction'] in (.01,.02):print(row)
