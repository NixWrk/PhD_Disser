"""Verify saved numerical data, montage coordinates and executed reader artifacts."""
from pathlib import Path
import base64
import hashlib
import json
import re

from bs4 import BeautifulSoup
import nbformat
import numpy as np
import trimesh
from nbconvert.filters import markdown2html

from lateral_pilot import ROOT, EXPERIMENT, read, save, sha, DiskModel
from lateral_identifiability import vector, SIZES
from lateral_admissible_sets import OUT as ADDITIONAL
from lateral_admissible_sets_report import verify_additional
from build_lateral_identifiability_report import STEM
from build_georg_fem_reports import GEOMETRY, INVERSE, OUT as OLD
from build_georg_geometry_revision_report import OUT, SOURCE, verify
from build_lateral_solution_surfaces_report import verify as verify_surfaces
from build_scorpion_experiment_viewer import STEM as EXPERIMENT_VIEW, verify as verify_viewer


def verify_skin_alignment():
    """Check preserved ray distances and the original first lung boundary."""
    depth=read(OUT/'lung_depth_profile.local.json')
    provenance=read(OUT/'ct_skin_aligned_section_provenance.local.json')
    assert provenance['depth_profile_sha256']==sha(OUT/'lung_depth_profile.local.json')
    assert provenance['surface_sha256']==sha(SOURCE)
    assert provenance['producer_sha256']==sha(Path(__file__).with_name('lung_depth_profile.py'))
    assert provenance['builder_sha256']==sha(Path(__file__).with_name('build_georg_geometry_revision_report.py'))
    assert provenance['ct_sha256']==sha(OLD/'segmentation/static_ct.nii.gz')
    for name,digest in provenance['mask_sha256'].items():
        assert digest==sha(OLD/'segmentation'/(name+'.nii.gz'))
    assert provenance['image_sha256']==sha(OUT/'ct_skin_aligned_section.png')
    with np.load(OUT/'ct_skin_aligned_section.local.npz') as aligned:
        np.testing.assert_array_equal(aligned['s_mm'],depth['s_mm'])
        np.testing.assert_allclose(aligned['x_lps_mm']-aligned['skin_x_lps_mm'][:,None],
            np.broadcast_to(aligned['depth_mm'],aligned['x_lps_mm'].shape),atol=1e-12)
        np.testing.assert_allclose(np.diff(aligned['depth_mm']),.5,atol=1e-12)
        assert not np.any(aligned['valid']&(aligned['x_lps_mm']>0))
        found=[]
        for i,expected in enumerate(depth['depth_mm']):
            first=np.flatnonzero(aligned['valid'][i]&(aligned['labels'][i]==1))
            if expected is not None:
                assert len(first) and aligned['valid'][i,:first[0]+1].all()
                found.append(aligned['depth_mm'][first[0]]-expected)
        np.testing.assert_allclose(found,0,atol=1e-12)
    return dict(compared_lung_boundaries=len(found),maximum_boundary_difference_mm=float(max(map(abs,found))),
        ray_distances_preserved=True,depth_scaling_applied=False)


def extract_reader_text():
    """Expose all prose, tables and plot labels for a subsequent model review."""
    decoder=json.JSONDecoder()
    def labels(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key in ['name','text','ticktext','hovertemplate']:yield str(item)
                elif key=='title':yield item if isinstance(item,str) else str(item.get('text',''))
                elif key not in ['template','x','y','z','i','j','k','customdata','bdata','font']:
                    yield from labels(item)
        elif isinstance(value,list):
            for item in value:yield from labels(item)
    folder=Path(__file__).parent
    for stem in (EXPERIMENT_VIEW,STEM,GEOMETRY,INVERSE):
        chunks=[]
        for cell in nbformat.read(folder/(stem+'.ipynb'),as_version=4).cells:
            if cell.cell_type=='markdown':chunks.append(cell.source)
            else:
                for output in cell.outputs:
                    html=output.get('data',{}).get('text/html','')
                    if html:
                        soup=BeautifulSoup(html,'html.parser')
                        for script in soup.select('script,style'):script.decompose()
                        text=soup.get_text(' ',strip=True)
                        if text:chunks.append(text)
                        matches=list(re.finditer(r'Plotly\.newPlot\(\s*"[0-9a-f-]+"',html))
                        if matches:
                            pos=html.index('(',matches[-1].start())+1;arguments=[]
                            for _ in range(3):
                                while html[pos].isspace() or html[pos]==',':pos+=1
                                value,used=decoder.raw_decode(html[pos:]);arguments.append(value);pos+=used
                            chunks.append('ПОДПИСИ ГРАФИКА: '+' | '.join(dict.fromkeys(labels(arguments[1:]))))
                    elif 'image/png' in output.get('data',{}):
                        chunks.append('ИЗОБРАЖЕНИЕ: встроенный PNG, проверяется визуально.')
        (OUT/(stem[:5]+'_full_reader_text.local.txt')).write_text('\n\n'.join(chunks),encoding='utf-8')


def run():
    primary=EXPERIMENT/'derived/exploratory/33.21_identifiability'
    manifest=read(primary/'manifest.local.json')
    assert all(sha(primary/name)==expected for name,expected in manifest['outputs'].items())
    verify_additional(ADDITIONAL)
    verify_surfaces();verify_viewer()
    scalar_error=0.;cloud_error=0.
    for case in read(ADDITIONAL/'results.local.json'):
        y=np.array(case['fit']['observations_ohm']);bounds=np.array(case['fit']['bounds_ohm_m'])
        p=np.array(case['fit']['parameters']);r=vector(p)-y
        np.testing.assert_allclose(r@r,case['fit']['rss'],rtol=1e-10)
        with np.load(ADDITIONAL/(case['key']+'.npz')) as d:
            a=d['cloud']
            assert np.all(a[:,:2]>=bounds[:,0]-1e-9) and np.all(a[:,:2]<=bounds[:,1]+1e-9)
            assert np.max(a[:,3])<=1.1*case['fit']['rss']+1e-8
            for point in a[::max(1,len(a)//30)]:
                r=vector(point[:3])-y
                cloud_error=max(cloud_error,abs(r@r-point[3]))
            for i,size in enumerate(SIZES):
                for ih in np.linspace(0,len(d['h'])-1,7,dtype=int):
                    model=DiskModel([size],d['h'][ih],radial=2,angular=8,terms=2048)
                    for k in range(2):
                        fixed=d['fixed_rho2'] if k==0 else d['fixed_rho1']
                        for j,value in enumerate(d['conditional_lines'][i,k,:,ih]):
                            if np.isfinite(value):
                                p=[value,fixed[j]] if k==0 else [fixed[j],value]
                                scalar_error=max(scalar_error,abs(model.evaluate(p)[0][0]-y[i]))
    assert cloud_error<1e-7 and scalar_error<1e-7
    comparisons=read(ADDITIONAL/'cross_comparison.local.json')
    for area in ('technical','literature'):
        for ch in (1,2):
            for state in ('inhale','exhale'):
                pair=[r for r in comparisons if (r['area'],r['channel'],r['state'])==(area,ch,state)]
                assert len(pair)==2
                assert pair[0]['four_size_prediction_140_ohm']==pair[1]['four_size_prediction_140_ohm']
    skin_alignment=verify_skin_alignment()
    g=verify(OUT);c=read(OUT/'contacts.local.json');radius=g['radius_mm']
    source=trimesh.load(SOURCE,force='mesh');angles=np.arange(256)*2*np.pi/256
    for key,j in g['joins'].items():
        yz=np.array(j['centre_yz_mm'])+radius*np.c_[np.cos(angles),np.sin(angles)]
        points=np.c_[np.full(len(yz),j['x_mm']),yz]
        assert source.contains(points).all()
    for montage in ('1x4','2x2'):
        for key,index in [('left',0),('right',3)]:
            yz=np.array(g['joins'][key]['centre_yz_mm'])
            xyz=np.array(c['ttrkg_'+montage][index])
            np.testing.assert_allclose(xyz[:,1],yz[0],atol=.02)
            target=np.array([radius]) if montage=='1x4' else np.array([radius,-radius])
            np.testing.assert_allclose(xyz[:,2]-yz[1],target,atol=.02)
    assert c['potential_pair_rotation_is_scenario'] is False
    for key,index in [('left',1),('right',2)]:
        yz=np.array(g['joins'][key]['centre_yz_mm']);xyz=np.array(c['ttrkg_2x2'][index])
        np.testing.assert_allclose(xyz[:,1],yz[0],atol=.02)
        np.testing.assert_allclose(xyz[:,2]-yz[1],[radius,-radius],atol=.02)
    centres=np.array(c['lateral']['centres_by_size_mm'])
    np.testing.assert_allclose(centres[:,:,1],c['lateral']['y_mm'],atol=1e-9)
    np.testing.assert_allclose(centres[:,:,2].mean(axis=1),c['lateral']['z_mm'],atol=1e-9)
    np.testing.assert_allclose(centres[:,3,2]-centres[:,0,2],SIZES,atol=1e-9)
    surface=trimesh.load(OUT/'body_embedded_arms_370mm.stl',force='mesh')
    assert surface.is_watertight and surface.is_volume and len(surface.split())==1
    inventory=[];technical=[]
    folder=Path(__file__).parent
    for stem in (EXPERIMENT_VIEW,STEM,GEOMETRY,INVERSE):
        nb=nbformat.read(folder/(stem+'.ipynb'),as_version=4)
        codes=[c for c in nb.cells if c.cell_type=='code']
        assert all(c.execution_count is not None for c in codes)
        assert not any(o.output_type=='error' for c in codes for o in c.outputs)
        soup=BeautifulSoup((folder/(stem+'.html')).read_text(encoding='utf-8'),'html.parser')
        assert not soup.select('.jp-CodeCell .jp-InputArea') and not soup.select('.highlight')
        if stem==GEOMETRY:
            image_sha=sha(OUT/'ct_skin_aligned_section.png')
            pngs=[o.get('data',{}).get('image/png','') for c in codes for o in c.outputs]
            assert image_sha in [hashlib.sha256(base64.b64decode(p)).hexdigest() for p in pngs if p]
            images=[i.get('src','') for i in soup.select('img')]
            assert image_sha in [hashlib.sha256(base64.b64decode(s.split(',',1)[1])).hexdigest()
                for s in images if s.startswith('data:image/png;base64,')]
        for anchor in soup.select('.anchor-link'):anchor.decompose()
        actual=' '.join(soup.get_text(' ',strip=True).split())
        for cell in nb.cells:
            if cell.cell_type!='markdown':continue
            expected_soup=BeautifulSoup(markdown2html(cell.source),'html.parser')
            for anchor in expected_soup.select('.anchor-link'):anchor.decompose()
            expected=expected_soup.get_text(' ',strip=True)
            assert ' '.join(expected.split()) in actual, f'Stale HTML prose: {stem}, cell {cell.id}'
        markdown='\n\n'.join(c.source for c in nb.cells if c.cell_type=='markdown')+'\n'
        assert (folder/(stem+'.md')).read_text(encoding='utf-8')==markdown
        technical.append(dict(stem=stem,executed_cells=len(codes),errors=0,code_excluded=True,html_prose_matches_notebook=True))
        for ext in ('.ipynb','.html','.md'):
            p=folder/(stem+ext)
            inventory.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size))
    result=dict(primary_numerical_manifest_unchanged=True,additional_manifest_verified=True,
        active_bounded_surface_manifest_verified=True,experiment_viewer_manifest_verified=True,
        maximum_cloud_rss_difference_ohm2=cloud_error,maximum_scalar_residual_ohm=scalar_error,
        actual_joins_full_circle_inside_source=True,current_contacts_top_and_opposite_bottom=True,
        lateral_axis_longitudinal=True,old_fem_transferred=False,new_fem_executed=False,
        skin_alignment=skin_alignment,
        reader_checks=technical,artifact_inventory=inventory)
    save(OUT/'revision_verification.local.json',result)
    save(OUT/'review_inventory.local.json',inventory)
    print(json.dumps({k:v for k,v in result.items() if k not in ('artifact_inventory','reader_checks')},ensure_ascii=False))


if __name__=='__main__':run()
