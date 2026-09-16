"""Matched finite-contact CEM comparison: background -> bone -> bone and heart."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lateral_literature_forward import PIPE, BASE, LIT, sha, read, write, source
import refine_surface_contact_mesh as refine

OUT=PIPE/'output/exploratory/lateral_tissue_ablation_20260916'
GEO=PIPE/'output/exploratory/transverse_consistency_20260914'
NAMES={'two_tissue':'Фон + лёгкое','bones':'+ кости','bones_heart':'+ кости + сердце'}


def prepare(out=OUT):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    assert not (out/'plan.json').exists(), 'Frozen plan already exists'
    bp=read(BASE/'plan.json');contract=read(GEO/'input_contract.json')
    a=loadmat(out/'original_labels.mat',simplify_cells=True)
    assert a['mesh_sha256']==contract['mesh_sha256']
    assert list(a['tissue_names'])==['soft_tissue','lungs','heart','bones']
    g=loadmat(GEO/'geometry_input.mat',simplify_cells=True)
    old=loadmat(GEO/'geometry_masks.mat',simplify_cells=True)
    final=loadmat(BASE/'refined_mesh.mat',simplify_cells=True)
    labels=np.asarray(a['tissue_id']).ravel().astype(np.int32)
    lung=np.asarray(old['masks'][:,0],bool)
    # Preserve the closed-lung mask used by every current baseline calculation.
    bone=(labels==4)&~lung;heart=(labels==3)&~lung
    ids=np.select([lung,bone,heart],[2,4,3],default=1).astype(np.int32)
    nodes=np.asarray(g['nodes_mm']);tets=np.asarray(g['elems'],np.int32)-1
    boundary=np.asarray(g['boundary_faces'],np.int32)-1
    refine.np=np;v=refine.volumes(nodes,tets)
    overlap=dict(bone_elements=int(np.count_nonzero((labels==4)&lung)),
        bone_volume_mm3=float(np.abs(v)[(labels==4)&lung].sum()),
        heart_elements=int(np.count_nonzero((labels==3)&lung)))
    flipped=v<0;tets[flipped,1:3]=tets[flipped,1:3][:,::-1]
    before=np.bincount(ids,weights=np.abs(v),minlength=5)
    nodes,tets,boundary,ids,lung2,history=refine.refine(nodes,tets,boundary,ids,lung.astype(np.int32),
        np.asarray(final['electrode_centres_mm']),radius=4.,target=.5)
    assert np.array_equal(nodes,final['nodes_mm'])
    assert np.array_equal(tets+1,final['elems']) and np.array_equal(boundary+1,final['boundary'])
    assert np.array_equal(lung2,np.asarray(final['reference_mask']).ravel())
    v=refine.volumes(nodes,tets);assert np.all(v>0)
    after=np.bincount(ids,weights=v,minlength=5);np.testing.assert_allclose(after,before,rtol=1e-10,atol=1e-5)
    savemat(out/'material_masks.mat',dict(soft_mask=(ids==1)[:,None],bone_mask=(ids==4)[:,None],
        heart_mask=(ids==3)[:,None]),do_compression=False)
    pd.DataFrame([dict(tissue=n,tetrahedra=int((ids==i).sum()),volume_ml=float(v[ids==i].sum()/1000))
        for i,n in [(1,'soft_background'),(2,'lung'),(4,'bone'),(3,'heart')]]).to_csv(out/'material_inventory.csv',index=False)
    selection=read(LIT/'literature_selection.json');raw_name,raw_sha=next(iter(selection['source_files'].items()))
    raw=PIPE/raw_name.replace('\\','/');assert sha(raw)==raw_sha
    required={'Muscle','Lung (Inflated)','Lung (Deflated)','Bone (Cortical)','Heart Muscle'}
    values=[]
    for line,row in enumerate(csv.reader(raw.read_text(encoding='cp1252').splitlines(),delimiter='\t'),1):
        if len(row)<48 or row[1] not in required:continue
        for f in [50000,100000]:
            delta=np.array([float(row[i]) for i in [28,31,35,38]])
            tau=np.array([float(row[i]) for i in [29,32,36,39]])*[1e-12,1e-9,1e-6,1e-3]
            alpha=np.array([float(row[i]) for i in [30,33,37,40]])
            w=2*np.pi*f;disp=np.sum(delta/(1+(1j*w*tau)**(1-alpha)))
            sigma=float(row[34])-w*8.8542e-12*disp.imag
            values.append(dict(tissue=row[1],frequency_hz=f,conductivity_S_m=sigma,rho_ohm_m=1/sigma,source_line=line))
    lit=pd.DataFrame(values);assert len(lit)==10 and set(lit.tissue)==required
    lit.to_csv(out/'literature_values.csv',index=False)
    requests=[]
    for f in [50000,100000]:
        a=lit[lit.frequency_hz==f].set_index('tissue').rho_ohm_m
        for state,tissue in [('inhale','Lung (Inflated)'),('exhale','Lung (Deflated)')]:
            for model in NAMES:
                requests.append(dict(request_id=f'{f}_{state}_{model}',kind='literature',frequency_hz=f,
                    state=state,model=model,rho1=a['Muscle'],rho2=a[tissue],rho_bone=a['Bone (Cortical)'],rho_heart=a['Heart Muscle']))
    for model in NAMES:
        requests.append(dict(request_id='equal_'+model,kind='equal_material_limit',frequency_hz=50000,
            state='control',model=model,rho1=4.,rho2=16.,rho_bone=4.,rho_heart=4.))
    pd.DataFrame(requests).to_csv(out/'requests.csv',index=False)
    paths=[BASE/'plan.json',BASE/'prepared_fem.mat',BASE/'refined_mesh.mat',BASE/'contact_qc.csv',
        GEO/'input_contract.json',GEO/'geometry_input.mat',GEO/'geometry_masks.mat',out/'original_labels.mat',
        PIPE/'output/nik_trkg4_right_rib_050mm_full_v5_1mm_local2mm.mat',raw,
        PIPE/bp['observations_source'].replace('\\','/'),PIPE/'src/run_lateral_array_design.m',Path(refine.__file__)]
    write(out/'plan.json',dict(status='exploratory_hypothesis_not_validated',candidate='C01',
        purpose='matched sequential restoration of bone then heart; no geometry or resistivity fitting',
        models=list(NAMES),sizes_mm=bp['sizes_mm'],finite_electrodes=True,diameter_mm=5,
        contact_ohm_m2=bp['contact_ohm_m2'],electrodes_per_measurement=4,
        sources=[source(p) for p in paths],requests_sha256=sha(out/'requests.csv'),masks_sha256=sha(out/'material_masks.mat'),
        literature_values_sha256=sha(out/'literature_values.csv'),material_inventory_sha256=sha(out/'material_inventory.csv'),
        literature_doi=selection['doi'],literature_urls=selection['selected_source_urls'],
        literature_scope='frozen ITIS V5.0 corpus extended to existing-model Bone (Cortical) and Heart Muscle rows',
        mask_overlap=overlap,overlap_policy='current closed lung has priority; no change to lung or external surface',
        geometry_checks=dict(identical_refined_nodes_elements_boundary=True,positive_volumes=True,
            parent_material_volumes_preserved=True,before_mm3=before.tolist(),after_mm3=after.tolist()),
        preparation_script_sha256=sha(__file__),
        limitations=['Fixed inspiratory geometry in both electrical respiratory scenarios.',
            'Deflated is not independently established as this volunteer exhalation.',
            'Whole bone mask assigned cortical bone; marrow and cancellous bone not separated.',
            'Whole heart mask assigned heart muscle; blood not separately segmented.',
            'Real isotropic conductivities only; fixed transferred contact resistance.',
            'Five mm contact is a design scenario, not measured historical diameter.',
            'Sequential increments depend on background; not independent impedance fractions.',
            'No new pose search, fat segmentation, or population validation.']))
    print(lit.to_string(index=False),flush=True)


def analyze(out=OUT):
    p=read(out/'plan.json');e=read(out/'execution.json');assert e['complete'] and e['rows']==135
    for s in p['sources']:assert sha(PIPE/s['path'])==s['sha256'],s['path']
    assert sha(out/'requests.csv')==p['requests_sha256'] and sha(out/'responses.csv')==e['responses_sha256']
    assert sha(out/'material_masks.mat')==p['masks_sha256']
    assert e['identity']['plan_sha256']==sha(out/'plan.json')
    assert e['identity']['worker_sha256']==sha(PIPE/'src/run_lateral_tissue_ablation.m')
    d=pd.read_csv(out/'responses.csv');q=pd.read_csv(out/'requests.csv').set_index('request_id')
    assert not d.duplicated(['request_id','L_mm']).any() and len(d)==9*len(q)
    assert set(d.request_id)==set(q.index)
    for rid,g in d.groupby('request_id'):
        assert sorted(g.L_mm)==p['sizes_mm']
        np.testing.assert_allclose(g[['rho1','rho2','rho_bone','rho_heart']],np.tile(q.loc[rid,['rho1','rho2','rho_bone','rho_heart']].to_numpy(float),(9,1)),rtol=1e-12)
        assert all(g.model==q.loc[rid,'model'])
    assert np.isfinite(d.select_dtypes('number')).all().all()
    assert d.relative_residual.max()<1e-7 and d.reciprocity_abs.max()<1e-7
    c=d[d.kind=='equal_material_limit'].pivot(index='L_mm',columns='model',values='Z')
    equal=float((c.max(axis=1)-c.min(axis=1)).max());assert equal<1e-6
    lit=d[d.kind=='literature'].copy();old_errors=[]
    for f,folder in [(50000,'lateral_literature_forward_20260916'),(100000,'lateral_literature_100khz_20260916')]:
        old=pd.read_csv(PIPE/'output/exploratory'/folder/'comparison.csv')
        a=lit[(lit.frequency_hz==f)&(lit.model=='two_tissue')].merge(old[old.model=='reference'],on=['L_mm','state'],suffixes=('','_old'),validate='one_to_one')
        assert len(a)==18;err=float((a.Z-a.Z_old).abs().max());assert err<1e-6;old_errors.append(err)
    obs=pd.read_csv(PIPE/'output/exploratory/lateral_real_application_20260915/observations.csv')
    lit['Z_experiment']=[float(obs.set_index('L_mm').loc[row.L_mm,f'Z_{row.state}_hold_ohm']) for row in lit.itertuples()]
    lit['residual_ohm']=lit.Z-lit.Z_experiment;lit['residual_pct']=100*lit.residual_ohm/lit.Z_experiment
    lit.to_csv(out/'comparison.csv',index=False)
    wide=lit.pivot(index=['frequency_hz','state','L_mm'],columns='model',values='Z').reset_index()
    wide['bone_delta_ohm']=wide.bones-wide.two_tissue
    wide['heart_after_bone_delta_ohm']=wide.bones_heart-wide.bones
    wide['both_delta_ohm']=wide.bones_heart-wide.two_tissue
    wide['bone_delta_pct']=100*wide.bone_delta_ohm/wide.two_tissue
    wide['heart_after_bone_delta_pct']=100*wide.heart_after_bone_delta_ohm/wide.bones
    wide.to_csv(out/'increments.csv',index=False)
    metrics=[]
    for (f,state,model),g in lit.groupby(['frequency_hz','state','model']):
        metrics.append(dict(frequency_hz=f,state=state,model=model,rmse_ohm=float(np.sqrt(np.mean(g.residual_ohm**2))),
            mean_bias_ohm=float(g.residual_ohm.mean()),mape_pct=float(g.residual_pct.abs().mean())))
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    # A local diagnostic only: equivalent log-parameter displacement and uncaptured residual.
    biases=[]
    for (f,state),g in lit.groupby(['frequency_hz','state']):
        original=g[g.model=='two_tissue'].sort_values('L_mm')
        for reduced,target,label in [('two_tissue','bones','omit_bone'),('bones','bones_heart','omit_heart_after_bone'),('two_tissue','bones_heart','omit_both')]:
            a=g[g.model==reduced].sort_values('L_mm');b=g[g.model==target].sort_values('L_mm')
            J=a[['dZ_dlogrho1','dZ_dlogrho2']].to_numpy();delta=b.Z.to_numpy()-a.Z.to_numpy()
            step=np.linalg.lstsq(J,delta,rcond=None)[0];res=J@step-delta
            biases.append(dict(frequency_hz=f,state=state,comparison=label,
                delta_logrho1=float(step[0]),delta_logrho2=float(step[1]),
                small_step_under_0_1=bool(np.max(np.abs(step))<=.1),
                unexplained_rmse_ohm=float(np.sqrt(np.mean(res**2))),condition_number=float(np.linalg.cond(J))))
    pd.DataFrame(biases).to_csv(out/'local_inverse_diagnostic.csv',index=False)
    qc=dict(complete=True,new_forward_solves=108,equal_material_control_solves=27,derivative_solves=9,
        prior_reference_max_difference_ohm=max(old_errors),equal_material_max_difference_ohm=equal,
        max_system_relative_residual=float(d.relative_residual.max()),max_reciprocity_abs_ohm=float(d.reciprocity_abs.max()),
        global_spatial_convergence=False,physical_validation=False,nonlinear_inverse_performed=False,
        input_sha256={n:sha(out/n) for n in ['plan.json','requests.csv','responses.csv','execution.json','literature_values.csv','material_inventory.csv','preparation_qc.json','derivative_qc.json']},
        output_sha256={n:sha(out/n) for n in ['comparison.csv','increments.csv','metrics.csv','local_inverse_diagnostic.csv']},analysis_script_sha256=sha(__file__))
    write(out/'qc.json',qc)
    print(pd.DataFrame(metrics).to_string(index=False));print(wide.groupby(['frequency_hz','state'])[['bone_delta_ohm','heart_after_bone_delta_ohm','bone_delta_pct','heart_after_bone_delta_pct']].agg(['min','max']).to_string())
    return lit,wide


def figure(out=OUT):
    qc=read(out/'qc.json')
    for n,h in {**qc['input_sha256'],**qc['output_sha256']}.items():assert sha(out/n)==h,n
    d=pd.read_csv(out/'comparison.csv');w=pd.read_csv(out/'increments.csv')
    fig=make_subplots(rows=2,cols=2,horizontal_spacing=.09,vertical_spacing=.22,
        subplot_titles=['Вдох: абсолютный импеданс','Условный выдох: абсолютный импеданс',
                        'Вдох: изменение при возврате органа','Условный выдох: изменение при возврате органа'])
    colors={'two_tissue':'#8293a2','bones':'#ad6b17','bones_heart':'#2166ac'}
    for f in [50000,100000]:
        visible=f==50000
        for col,state in enumerate(['inhale','exhale'],1):
            a=d[(d.frequency_hz==f)&(d.state==state)]
            exp=a[a.model=='two_tissue'].sort_values('L_mm')
            fig.add_trace(go.Scatter(x=exp.L_mm,y=exp.Z_experiment,name='Эксперимент',mode='lines+markers',
                line=dict(color='#cf3f3f'),legendgroup='exp',showlegend=col==1,visible=visible),row=1,col=col)
            for model,label in NAMES.items():
                g=a[a.model==model].sort_values('L_mm')
                fig.add_trace(go.Scatter(x=g.L_mm,y=g.Z,name=label,mode='lines+markers',line=dict(color=colors[model]),
                    legendgroup=model,showlegend=col==1,visible=visible),row=1,col=col)
            g=w[(w.frequency_hz==f)&(w.state==state)].sort_values('L_mm')
            for field,label,color in [('bone_delta_ohm','Изменение: кости','#ad6b17'),('heart_after_bone_delta_ohm','Изменение: сердце после костей','#2166ac')]:
                fig.add_trace(go.Scatter(x=g.L_mm,y=g[field],customdata=g[[field.replace('_ohm','_pct')]],
                    hovertemplate='L=%{x} мм<br>ΔZ=%{y:.5f} Ом<br>%{customdata[0]:.4f}%<extra>%{fullData.name}</extra>',
                    name=label,mode='lines+markers',line=dict(color=color),legendgroup=field,
                    showlegend=col==1,visible=visible),row=2,col=col)
    fig.update_yaxes(title='Z, Ом',range=[0,115],row=1,col=1);fig.update_yaxes(range=[0,115],row=1,col=2)
    for col in [1,2]:
        fig.update_xaxes(title='Размер сборки L, мм',tickvals=[50,70,90,110,130,140],row=2,col=col)
        fig.update_yaxes(title='Изменение Z, Ом' if col==1 else None,zeroline=True,zerolinecolor='#555',row=2,col=col)
    fig.update_layout(template='plotly_white',width=1280,height=880,font=dict(family='Arial',size=14),
        margin=dict(l=80,r=35,t=115,b=120),hovermode='x unified',
        title=dict(text='Последовательное восстановление костей и сердца · 50 кГц',x=.5),
        legend=dict(orientation='h',y=-.16,x=.5,xanchor='center'),
        updatemenus=[dict(x=.0,y=1.14,buttons=[dict(label=f'{f//1000} кГц',method='update',args=[
            {'visible':[i//12==j for i in range(24)]},{'title.text':f'Последовательное восстановление костей и сердца · {f//1000} кГц'}]) for j,f in enumerate([50000,100000])])])
    return fig


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');args=ap.parse_args()
    if args.prepare:prepare()
    else:
        analyze();fig=figure();fig.write_html(OUT/'comparison.html',include_plotlyjs=True,config={'displaylogo':False})
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True);page=browser.new_page(viewport={'width':1310,'height':930},device_scale_factor=1.5)
            page.goto((OUT/'comparison.html').as_uri(),wait_until='networkidle')
            page.locator('.js-plotly-plot').screenshot(path=str(OUT/'comparison.png'));browser.close()
