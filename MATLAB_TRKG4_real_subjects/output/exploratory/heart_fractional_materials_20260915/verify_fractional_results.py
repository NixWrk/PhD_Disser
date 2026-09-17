"""Independent artifact/analytic checks for all completed fractional geometry states."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
out=Path(__file__).resolve().parent;root=out.parents[2];sys.path.insert(0,str(root/'tools'))
import heart_fractional_materials as model
r=model.read(out/'results.json');identity=model.read(out/'identity.json');scan=model.read(out/'mesh_scan.json')
assert model.sha(root/'tools/heart_fractional_materials.py')==identity['implementation_sha256']
assert r['identity_sha256']==model.sha(out/'identity.json')
assert len(r['rows'])==198 and len(r['derivatives'])==96 and len(r['receipts'])==66
ids=np.load(out/'selected_element_indices_zero_based.npy');bg=np.load(out/'selected_background_labels.npy')
assert model.sha(out/'selected_element_indices_zero_based.npy')==scan['selected_indices_sha256']
assert ids.shape==bg.shape==(scan['selected_tetrahedra'],)
assert np.array_equal(ids,np.unique(ids)) and ids.min()>=0 and ids.max()<scan['all_input_tetrahedra_scanned']
assert np.isin(bg,[1,2,4]).all()
max_analytic_error=0;unique_arrays=set()
for entry in r['receipts']:
 path=out/entry['file'];assert model.sha(path)==entry['sha256']
 s=model.read(path);assert s['identity_sha256']==r['identity_sha256']
 p=s['state'];analytic=4*np.pi/3*np.prod(p['semiaxes_m'])*1e6
 max_analytic_error=max(max_analytic_error,abs(analytic-p['analytic_volume_ml']))
 assert abs(analytic-p['analytic_volume_ml'])<1e-9
 array_path=path.with_suffix('.npz');assert model.sha(array_path)==s['array_sha256'];unique_arrays.add(s['array_sha256'])
 with np.load(array_path) as data:
  f=data['heart_fraction'];assert f.shape==(3,len(ids)) and np.array_equal(data['levels'],[64,512,4096])
  assert np.isfinite(f).all() and np.all((f>=0)&(f<=1))
  assert all(np.all(f[j]*n==np.floor(f[j]*n)) for j,n in enumerate([64,512,4096]))
  for row in f:
   fractions=model.material_fractions(row,bg)
   assert np.max(abs(fractions.sum(axis=1)-1))<1e-14
   assert np.array_equal(bg,np.load(out/'selected_background_labels.npy'))
   assert np.allclose(fractions@np.full(4,.2),.2,atol=1e-15,rtol=0)
   full=model.expand_heart_fraction(scan['all_input_tetrahedra_scanned'],ids,row)
   assert np.array_equal(full[ids],row) and np.count_nonzero(full)==np.count_nonzero(row)
   del full
checks={'status':'passed','geometric_states':66,'point_level_rows':198,'derivatives':96,
        'all_input_tetrahedra_scanned':scan['all_input_tetrahedra_scanned'],
        'conservative_selected_tetrahedra':len(ids),'unique_state_fraction_arrays':len(unique_arrays),
        'max_analytic_parameter_volume_consistency_ml':max_analytic_error,
        'fraction_bounds_checked':True,'immutable_background_checked':True,
        'sigma_uniform_limit_checked_all_states_levels':True,
        'full_global_expansion_checked_all_states_levels':True,
        'receipt_hashes_verified':True,'FEM_solved':False,
        'verification_implementation_sha256':model.sha(__file__),'results_sha256':model.sha(out/'results.json')}
model.write(out/'verification.json',checks);print(json.dumps(checks,indent=2))
