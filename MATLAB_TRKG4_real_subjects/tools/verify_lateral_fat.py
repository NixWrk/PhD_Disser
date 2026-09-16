"""Independent geometry checks and final browser verification for the fat test."""
import argparse
import base64
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import qmc
from lateral_fat_prepare import OUT,PIPE,BASE,PREVIOUS,sample_labels,sha,read,write,volumes


def geometry():
    u=np.sort(qmc.Sobol(3,scramble=True,seed=20260916).random_base2(10),axis=1)
    b=np.diff(np.c_[np.zeros(1024),u,np.ones(1024)],axis=1)
    assert np.max(np.abs(b.mean(0)-.25))<.002
    estimate=float(np.mean(b[:,1]>.3));exact=(1-.3)**3;assert abs(estimate-exact)<.006
    grid=np.zeros((4,5,6),np.uint8);grid[1,2,3]=4
    result,inside=sample_labels(grid,np.array([[1,2,3],[1,2,3.49],[1,2,3.51],[-.51,2,3]]))
    assert list(result)==[4,4,0,0] and list(inside)==[True,True,True,False]
    m=loadmat(BASE/'refined_mesh.mat',simplify_cells=True)
    f=loadmat(OUT/'fractions_q1024.mat',simplify_cells=True)['fractions']
    v=volumes(np.asarray(m['nodes_mm']),np.asarray(m['elems'],int)-1)
    old=loadmat(PREVIOUS/'material_masks.mat',simplify_cells=True);comparison=[]
    for j,name in [(1,'lung'),(2,'bone'),(3,'heart')]:
        mask=np.asarray(m['reference_mask'] if name=='lung' else old[name+'_mask']).ravel().astype(bool)
        comparison.append(dict(tissue=name,previous_volume_ml=float(v[mask].sum()/1000),
            new_volume_ml=float(v@f[:,j]/1000),fractional_overlap_dice=float(2*v[mask]@f[mask,j]/(v[mask].sum()+v@f[:,j]))))
    pd.DataFrame(comparison).to_csv(OUT/'mask_comparison.csv',index=False)
    write(OUT/'geometry_qc.json',dict(complete=True,uniform_tetrahedron_first_moment_error=float(np.max(np.abs(b.mean(0)-.25))),
        analytical_plane_cut_volume_fraction=exact,quadrature_plane_cut_volume_fraction=estimate,
        voxel_indexing_and_outside_test=True,nonnegative_unit_sum_material_fractions=bool(np.all(f>=0) and np.all(f.sum(1)==1)),
        mask_comparison_sha256=sha(OUT/'mask_comparison.csv'),script_sha256=sha(__file__),
        overlap_scope='discrete model agreement at 1024 points per element, not segmentation accuracy'))


def visual():
    OUT=globals()['OUT']/'refined'
    from playwright.sync_api import sync_playwright
    d=pd.read_csv(OUT/'increments.csv');expected=[]
    for f in [50000,100000]:
        for state in ['inhale','exhale']:
            a=d[(d.frequency_hz==f)&(d.state==state)].sort_values('L_mm')
            for field in ['Z_experiment','previous_four_tissue','ts_without_fat','ts_with_fat','fat_delta_ohm','segmentation_delta_ohm']:
                expected.append((a.L_mm.to_numpy(),a[field].to_numpy()))
    def array(a):
        return np.frombuffer(base64.b64decode(a['bdata']),dtype=np.dtype(a['dtype'])) if isinstance(a,dict) else np.asarray(a)
    nb=PIPE.parent/'Colab Notebooks/33.09_Выбор_размеров_и_числа_боковых_сборок.html'
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True);page=browser.new_page(viewport={'width':1360,'height':1000})
        page.goto((OUT/'comparison.html').as_uri(),wait_until='networkidle');plot=page.locator('.js-plotly-plot')
        data=plot.evaluate('(g)=>g.data.map(t=>({x:t.x,y:t.y}))');assert len(data)==24
        for got,(x,y) in zip(data,expected):
            np.testing.assert_allclose(array(got['x']),x,rtol=1e-12);np.testing.assert_allclose(array(got['y']),y,rtol=1e-12)
        plot.screenshot(path=str(OUT/'comparison.png'))
        page.locator('.updatemenu-header').click();page.locator('.updatemenu-dropdown-button').filter(has_text='100 кГц').click()
        assert '100 кГц' in page.locator('.gtitle').text_content();plot.screenshot(path=str(OUT/'comparison_100khz.png'))
        page.goto(nb.as_uri(),wait_until='networkidle');page.wait_for_function('window.MathJax && MathJax.Hub')
        page.evaluate("new Promise(resolve=>MathJax.Hub.Queue(['Typeset',MathJax.Hub],resolve))")
        assert page.locator('.jp-CodeCell .jp-InputArea').count()==0
        count=page.locator('.js-plotly-plot').count();assert count==7,count
        page.locator('.js-plotly-plot').nth(6).screenshot(path=str(OUT/'notebook_section55.png'))
        math=page.locator('.MathJax_CHTML').count();assert math>=26;tables=page.locator('table.dataframe').count();browser.close()
    write(OUT/'visual_qc.json',dict(complete=True,all_24_traces_match_csv=True,frequency_switch_checked=True,
        code_inputs_absent=True,notebook_plot_count=count,MathJax_expressions=math,table_count=tables,
        html_sha256=sha(nb),standalone_sha256=sha(OUT/'comparison.html'),verification_script_sha256=sha(__file__),
        outputs_sha256={n:sha(OUT/n) for n in ['comparison.png','comparison_100khz.png','notebook_section55.png']}))
    print('Verified 24 fat traces, both frequency views, seven notebook figures and hidden code.')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['geometry','visual']);a=ap.parse_args()
    (geometry if a.stage=='geometry' else visual)()
