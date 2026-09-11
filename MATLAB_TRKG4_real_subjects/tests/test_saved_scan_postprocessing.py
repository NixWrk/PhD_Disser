import sys
from pathlib import Path
import numpy as np
import vtk
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from analyze_saved_scan import robustness
from scan_geometry_profiles import Surface,describe

def test_neighbourhood_worst_value_and_missing_are_distinct():
    state=np.ones((49,7),np.uint8)
    d=dict(u=np.arange(7),v=np.arange(7),phi=np.arange(7),status=state,flat=np.arange(state.size))
    value=np.ones(state.size);spike=np.ravel_multi_index((3,3,4),(7,7,7));value[spike]=20
    worst,known,fraction=robustness(d,value,radius=1)
    assert worst[3*7+3,3]==20
    assert worst[1*7+1,1]==1
    assert np.isnan(worst[0,0]) and not known[0,0]
    state.ravel()[spike]=0;d['flat']=np.flatnonzero(state.ravel()==1)
    worst,known,fraction=robustness(d,value[d['flat']],radius=1)
    assert np.isnan(worst[3*7+3,3]) and not known[3*7+3,3]
    assert abs(fraction[3*7+3,3]-26/27)<1e-12
    state.ravel()[spike]=2
    worst,known,fraction=robustness(d,value[d['flat']],radius=1)
    assert np.isnan(worst[3*7+3,3]) and known[3*7+3,3]

def sphere(path,z,r):
    source=vtk.vtkSphereSource();source.SetCenter(0,0,z);source.SetRadius(r);source.SetThetaResolution(64);source.SetPhiResolution(64);source.Update()
    writer=vtk.vtkSTLWriter();writer.SetFileName(str(path));writer.SetInputData(source.GetOutput());writer.Write()
    return Surface(path)

def test_first_lung_interval_and_bone_path_on_known_spheres(tmp_path):
    lung=sphere(tmp_path/'lung.stl',5,1);bone=sphere(tmp_path/'bone.stl',2,.5)
    result=describe(lung,bone,[0,0,0],[0,0,-1])
    assert result['lung_ray_qc']=='ok'
    assert abs(result['lung_entry_mm']-4)<.005
    assert abs(result['lung_first_chord_mm']-2)<.005
    assert abs(result['bone_before_lung_mm']-1)<.005
    assert abs(result['lung_nearest_surface_mm']-4)<.005
    miss=describe(lung,bone,[0,0,0],[1,0,0])
    assert miss['lung_ray_qc']=='no_intersection' and np.isnan(miss['lung_entry_mm'])

def test_open_surface_cannot_supply_anatomical_interval(tmp_path):
    source=vtk.vtkPlaneSource();source.Update()
    triangle=vtk.vtkTriangleFilter();triangle.SetInputData(source.GetOutput());triangle.Update()
    writer=vtk.vtkSTLWriter();writer.SetFileName(str(tmp_path/'open.stl'));writer.SetInputData(triangle.GetOutput());writer.Write()
    surface=Surface(tmp_path/'open.stl')
    intervals,status=surface.intervals(np.array([0.,0.,1.]),np.array([0.,0.,-1.]))
    assert intervals is None and status=='nonclosed_surface'
