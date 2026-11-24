function run_zero_limit(config_file)
% Zero-heart-volume limit from the canonical frozen background, not a patient state.
cfg=jsondecode(fileread(config_file));job=cfg.job;maxNumCompThreads(2);
addpath(fullfile(cfg.root,'MATLAB_TRKG4_real_subjects','src'));
assert(strcmp(trkg4_file_sha256(cfg.original_CT_parts),cfg.original_CT_parts_sha256));
data=load(cfg.original_CT_parts,'parts');parts=data.parts;clear data;
parts{1}=parts{1}+parts{4};parts{4}=sparse(size(parts{4},1),size(parts{4},2));
base=jsondecode(fileread(cfg.baseline_file));rho=base.fit.rho_ohm_m(:);
data=load(job.cem_file,'result');sc=data.result.scenarios;clear data;
if ~iscell(sc),sc=num2cell(sc);end
ix=find(cellfun(@(s)strcmp(regexprep(s.id,'_(ttrkg|lateral)$',''),job.case_ids{1}),sc),1);
C=sc{ix}.grouped_contact_matrix_SI;P=ports(sc{ix}.ports);clear sc;
answers=struct([]);clock=tic;n=size(parts{1},1);
for phase=[0 .1 .4]
 rr=rho;if phase~=0,[~,ii]=min(abs(base.phase_rr-phase));rr(1:2)=rr(1:2).*exp(base.dlogrho_FEM(:,ii));end
 K=sparse(n,n);for k=1:5,K=K+parts{k}/rr(k);end
 [T,J,qc]=trkg4_lateral_cem_response(K,C,parts,rr,P,[1 2 4 5],1);
 answers=[answers;struct('phase_rr',phase,'rho_ohm_m',rr','Z_t_ohm',T(2,1),'Z_b_ohm',T(4,3),...
  'transfer_matrix_ohm',T,'J_t_logrho',reshape(J(2,1,:),1,[]),'J_b_logrho',reshape(J(4,3,:),1,[]),'system_checks',qc)];
end
d=struct('participant',job.participant_code,'variant',struct('id','zero_volume_limit','volume_factor',0),...
 'answers',answers,'diagnostic_nonphysiological_limit',true,'CT_phase_targets_used',false,...
 'background_rule','Original CT heart material becomes soft background; all other original materials retained',...
 'elapsed_s',toc(clock),'identity',struct('driver_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']),...
 'baseline_sha256',trkg4_file_sha256(cfg.baseline_file),'original_CT_parts_sha256',cfg.original_CT_parts_sha256));
fid=fopen(cfg.output,'w','n','UTF-8');assert(fid>=0);fprintf(fid,'%s\n',jsonencode(d,'PrettyPrint',true));fclose(fid);
fprintf('ZERO LIMIT %s %.1f s\n',job.id,d.elapsed_s);
end
function P=ports(groups)
names={groups.id};
if any(strcmp(names,'TT_Ileft')),pairs={{'TT_Ileft','TT_Iright'},{'TT_Uleft','TT_Uright'},{'A','B'},{'M','N'}};
else,pairs={{'TT_I0','TT_I1'},{'TT_U0','TT_U1'},{'A','B'},{'M','N'}};end
P=zeros(numel(groups),4);for k=1:4,[found,ix]=ismember(pairs{k},names);assert(all(found));P(ix,k)=[1;-1];end
end
