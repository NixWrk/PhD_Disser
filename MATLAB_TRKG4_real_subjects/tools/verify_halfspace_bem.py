"""Independent sphere solution, homogeneous limit and derivative controls."""
import json
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from halfspace_bem import InclusionBEM,montage
from halfspace_study import OUT,write,sha

def run():
    rows=[]
    for resolution in [12,20,32]:
        source=vtk.vtkSphereSource();source.SetRadius(.04)
        source.SetThetaResolution(resolution);source.SetPhiResolution(resolution);source.Update()
        poly=source.GetOutput();v=vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
        f=vtk_to_numpy(poly.GetPolys().GetData()).reshape(-1,4)[:,1:]
        bem=InclusionBEM(v,f,mirror=False)
        points=np.array([[0,0,.09],[.08,.02,.01],[.03,.03,.08]])
        for q in [1.,4.,12.]:
            u=bem.uniform_field(q,np.array([0.,0.,1.]),points)
            exact=points[:,2]*(1+(q-1)/(2*q+1)*.04**3/np.linalg.norm(points,axis=1)**3)
            rows.append(dict(kind='sphere_exact',triangles=len(f),q=q,max_rel_error=float(max(abs((u-exact)/exact))),solid_angle_error=bem.row_sum_error,assembly_seconds=bem.assembly_seconds))
        v[:,2]+=.10;bem=InclusionBEM(v,f,montage([50,90,140]))
        a=bem.solve(4.);b=bem.solve(4*np.exp(.0001));c=bem.solve(4*np.exp(-.0001));hom=bem.solve(1.)
        rows.append(dict(kind='halfspace',triangles=len(f),relative_reciprocity=float(max(a['reciprocity_abs']/a['f'])),derivative_relative_error=float(max(abs((b['f']-c['f'])/.0002-a['df_dlogq'])/np.maximum(abs(a['df_dlogq']),1e-10))),homogeneous_error=float(max(abs(hom['f']-bem.base))),solve_seconds=a['seconds'],solid_angle_error=bem.row_sum_error))
    assert rows[-1]['relative_reciprocity']<1e-5
    assert rows[-1]['derivative_relative_error']<1e-6
    assert rows[-2]['max_rel_error']<.0005
    assert rows[-1]['homogeneous_error']==0
    for q in [4.,12.]:
        errors=[r['max_rel_error'] for r in rows if r['kind']=='sphere_exact' and r['q']==q]
        assert all(a>b for a,b in zip(errors,errors[1:]))
    write(OUT/'solver_validation.json',rows)
    write(OUT/'solver_validation_identity.json',dict(solver_sha256=sha(__import__('pathlib').Path(__file__).with_name('halfspace_bem.py')),test_sha256=sha(__file__),result_sha256=sha(OUT/'solver_validation.json'),status='mathematical_controls_pass_not_anatomical_validation'))
    print(json.dumps(rows[-4:],indent=2))

if __name__=='__main__':run()
