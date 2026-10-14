from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'MATLAB_TRKG4_real_subjects/tools'))
from georg_quality_mesh_section import section_lines

N=np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.],[0.,0.,2.]])
E=np.array([[0,1,2,3]])

def test_exact_triangle_at_middle_height():
    s=section_lines(N,E,np.array([0,0,1]),[1,0,0],[0,1,0],(-2,2,-2,2))
    assert s['intersected_tetrahedra_in_window']==1
    points=set(zip(s['x'][:-2],s['y'][:-2]))
    assert points=={(0.,0.),(1.,0.),(0.,1.)}

def test_coplanar_face_not_duplicated_and_point_touch_ignored():
    s=section_lines(N,E,np.zeros(3),[1,0,0],[0,1,0],(-2,2,-2,2))
    assert len(s['x'])==5
    assert section_lines(N,E,np.array([0,0,2]),[1,0,0],[0,1,0])['x']==[]

def test_disjoint_window_and_nonorthogonal_axes():
    assert section_lines(N,E,np.zeros(3),[1,0,0],[0,1,0],(10,20,10,20))['x']==[]
    with pytest.raises(ValueError,match='orthonormal'):
        section_lines(N,E,np.zeros(3),[1,0,0],[1,0,0])
