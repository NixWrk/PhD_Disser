function report=verify_acceleration_smoke(tag,python)
if nargin<2,python=getenv('TRKG4_REPORT_PYTHON');if isempty(python),python='python';end;end
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));cd(root);
addpath('src','tools','tools/full_pose_scan','notebooks/full_scan_2026_09_08');
r=run_pipeline('Profile','full_pose_scan','ResultTag',tag,'UValues',[0,1], ...
    'VValues',-30,'PhiValues',-20:20,'CentreChunk',1,'Workers',2,'GreenNodeLimit',2500, ...
    'Streaming',true,'ReportPython',python);
assert(r.complete&&r.counts.poses==82);
best_qc=verify_trkg4_full_scan_best(tag);
cfg=trkg4_config('nik');trkg4_prepare_runtime(cfg);
shared=load(fullfile(root,'output','nik_trkg4_right_rib_050mm_full_v5_1mm_local2mm.mat'), ...
    'fmdl_mesh_units','elem_sigma','tissue_id','tissue_names');
si=trkg4_scale_fmdl_to_si(shared.fmdl_mesh_units,cfg);sigma=shared.elem_sigma;
sigma(shared.tissue_id==find(strcmp(shared.tissue_names,'soft_tissue'),1))=1/r.rho1;
sigma(shared.tissue_id==find(strcmp(shared.tissue_names,'lungs'),1))=1/r.rho2;
out=fullfile(root,'output',['full_pose_scan_' tag]);identity=struct('scan_fingerprint',r.fingerprint);
a=load(fullfile(out,'geometry','block_00001.mat'));b=load(fullfile(out,'geometry','block_00002.mat'));
ra=find(a.phi_deg(a.valid_rows)==0);rb=find(b.phi_deg(b.valid_rows)==0);assert(isscalar(ra)&&isscalar(rb));
fa=reshape(a.patch_faces(ra,:,:),[],1);fb=reshape(b.patch_faces(rb,:,:),[],1);
na=unique(reshape(si.boundary(unique(vertcat(fa{:})),:),[],1));
nb=unique(reshape(si.boundary(unique(vertcat(fb{:})),:),[],1));
[Ga,ma]=full_scan_green(si,sigma,na,'',identity);
[Gr,mr]=full_scan_green(si,sigma,nb,'',identity,struct('green',Ga,'meta',ma));
[Gf,mf]=full_scan_green(si,sigma,nb,'',identity);
err=max(abs(Gr-Gf),[],'all');assert(err<1e-8&&mr.reused_green_columns>0&&mr.new_green_columns<numel(nb));
solved=load(fullfile(out,'solved','block_00002.mat'));map=zeros(size(si.nodes,1),1);map(nb)=1:numel(nb);
z=zeros(1,9);for j=1:9,z(j)=full_scan_cem_transfer(si,reshape(b.patch_faces(rb,j,:),1,4),map,Gf,cfg.z_contact);end
row=b.valid_rows(rb);difference=max(abs(z-solved.z(row,:)));assert(difference<1e-7);
report=struct('passed',true,'best_full_EIDORS',best_qc,'rolling_green_max_difference',err, ...
    'rolling_reused_columns',mr.reused_green_columns,'rolling_new_columns',mr.new_green_columns, ...
    'fresh_green_columns',mf.new_green_columns,'second_centre_all_nine_Z_max_difference',difference, ...
    'poses',82,'scope','bounded integration and numerical equivalence, not a complete production search');
full_scan_write_json(fullfile(out,'acceleration_verification.json'),report);disp(report);
end
