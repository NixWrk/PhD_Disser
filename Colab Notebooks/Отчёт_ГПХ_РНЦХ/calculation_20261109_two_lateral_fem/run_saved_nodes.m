function run_saved_nodes(config_file)
% Reuse geometry-only material matrices; solve at newly estimated two-record rho.
cfg=jsondecode(fileread(config_file));addpath(fullfile(cfg.root,'MATLAB_TRKG4_real_subjects','src'));
maxNumCompThreads(2);job=cfg.job;
baseline=jsondecode(fileread(cfg.baseline_file));rho=baseline.fit.rho_ohm_m(:);
assert(baseline.fit_used_lateral_record_count==2);
source=load(job.cem_file,'result');sc=source.result.scenarios;clear source;
if ~iscell(sc),sc=num2cell(sc);end
ix=find(cellfun(@(s)strcmp(regexprep(s.id,'_(ttrkg|lateral)$',''),job.case_ids{1}),sc),1);
assert(~isempty(ix));C=sc{ix}.grouped_contact_matrix_SI;P=ports(sc{ix}.ports);clear sc;
identity=struct('config_sha256',trkg4_file_sha256(config_file),'baseline_sha256',trkg4_file_sha256(cfg.baseline_file),...
 'driver_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']),...
 'kernel_sha256',trkg4_file_sha256(which('trkg4_lateral_cem_response')),'matlab_version',version);
for vi=1:numel(cfg.variants)
 v=cfg.variants(vi);target=fullfile(cfg.output,[v.id '.json']);
 if isfile(target),old=jsondecode(fileread(target));assert(isequal(old.identity,identity));continue;end
 assert(strcmp(trkg4_file_sha256(v.parts_file),v.parts_sha256));
 data=load(v.parts_file,'parts');parts=data.parts;clear data;
 n=size(parts{1},1);answers=struct([]);clock=tic;
 for phase=[0 .1 .4]
  rr=rho;
  if phase~=0,[~,ii]=min(abs(baseline.phase_rr-phase));rr(1:2)=rr(1:2).*exp(baseline.dlogrho_FEM(:,ii));end
  K=sparse(n,n);for k=1:numel(parts),K=K+parts{k}/rr(k);end
  [T,J,qc]=trkg4_lateral_cem_response(K,C,parts,rr,P,[1 2 4 5],1);
  answers=[answers;struct('phase_rr',phase,'rho_ohm_m',rr','Z_t_ohm',T(2,1),'Z_b_ohm',T(4,3),...
   'transfer_matrix_ohm',T,'J_t_logrho',reshape(J(2,1,:),1,[]),'J_b_logrho',reshape(J(4,3,:),1,[]),'system_checks',qc)];
 end
 result=struct('identity',identity,'variant',v,'job_id',job.id,'participant',job.participant_code,...
  'case_id',job.case_ids{1},'answers',answers,'elapsed_s',toc(clock),'mesh_unchanged',true,...
  'material_matrices_reused_geometry_only',true,'new_two_record_material_estimate',true,'CT_phase_targets_used',false);
 fid=fopen(target,'w','n','UTF-8');assert(fid>=0);fprintf(fid,'%s\n',jsonencode(result,'PrettyPrint',true));fclose(fid);
 fprintf('DIRECT %s %s %.1f s\n',job.id,v.id,result.elapsed_s);clear parts K;
end
end
function P=ports(groups)
names={groups.id};
if any(strcmp(names,'TT_Ileft')),pairs={{'TT_Ileft','TT_Iright'},{'TT_Uleft','TT_Uright'},{'A','B'},{'M','N'}};
else,pairs={{'TT_I0','TT_I1'},{'TT_U0','TT_U1'},{'A','B'},{'M','N'}};end
P=zeros(numel(groups),4);for k=1:4,[found,ix]=ismember(pairs{k},names);assert(all(found));P(ix,k)=[1;-1];end
end
