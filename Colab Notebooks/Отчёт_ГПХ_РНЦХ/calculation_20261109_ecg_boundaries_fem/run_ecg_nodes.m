function run_ecg_nodes(config_file,participant)
% Only orchestration; physical CEM and analytic J use the unchanged canonical kernel.
cfg=jsondecode(fileread(config_file));addpath(fullfile(cfg.root,'MATLAB_TRKG4_real_subjects','src'));
maxNumCompThreads(cfg.threads);
ix=find(strcmp({cfg.subjects.participant_code},participant),1);assert(~isempty(ix));job=cfg.subjects(ix);
kernel=which('trkg4_lateral_cem_response');assert(strcmp(trkg4_file_sha256(kernel),job.evidence.kernel_sha256));
assert(strcmp(trkg4_file_sha256(job.cem_file),job.cem_sha256));
source=load(job.cem_file,'result');sc=source.result.scenarios;clear source;
if ~iscell(sc),sc=num2cell(sc);end
Cs=cell(2,1);Ps=Cs;
for c=1:2
 ix=find(cellfun(@(s)strcmp(regexprep(s.id,'_(ttrkg|lateral)$',''),job.case_ids{c}),sc),1);assert(~isempty(ix));
 Cs{c}=sc{ix}.grouped_contact_matrix_SI;Ps{c}=ports(sc{ix}.ports);
end
clear sc;
if ~isfolder(job.output),mkdir(job.output);end
identity=struct('config_sha256',trkg4_file_sha256(config_file),'driver_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']),...
 'kernel_sha256',trkg4_file_sha256(kernel),'matlab_version',version);
for vi=1:numel(job.nodes)
 node=job.nodes(vi);target=fullfile(job.output,[node.id '.json']);
 if isfile(target),previous=jsondecode(fileread(target));assert(isequal(previous.identity,identity));continue;end
 assert(strcmp(trkg4_file_sha256(node.parts_file),node.parts_sha256));
 data=load(node.parts_file,'parts');parts=data.parts;clear data;
 if strcmp(node.kind,'zero_limit'),parts{1}=parts{1}+parts{4};parts{4}=sparse(size(parts{4},1),size(parts{4},2));end
 n=size(parts{1},1);answers=struct([]);clock=tic;
 for ei=1:numel(job.endpoints)
  ep=job.endpoints(ei);rho=ep.rho_ohm_m(:);K=sparse(n,n);
  for k=1:numel(parts),K=K+parts{k}/rho(k);end
  [T,J,qc]=trkg4_lateral_cem_response(K,Cs{1},parts,rho,Ps{1},[1 2 4 5],1);
  second=struct('calculated',false);
  if abs(node.volume_factor-1)<1e-12
   [T2,J2,qc2]=trkg4_lateral_cem_response(K,Cs{2},parts,rho,Ps{2},[1 2 4 5],1);
   second=struct('calculated',true,'Z_b_ohm',T2(4,3),'J_b_logrho',reshape(J2(4,3,:),1,[]),'system_checks',qc2);
  end
  answers=[answers;struct('endpoint',ep.endpoint,'rho_ohm_m',rho','Z_t_ohm',T(2,1),'Z_b_ohm',T(4,3),...
   'J_t_logrho',reshape(J(2,1,:),1,[]),'J_b_logrho',reshape(J(4,3,:),1,[]),'system_checks',qc,'second_lateral_at_V0',second)]; %#ok<AGROW>
 end
 result=struct('identity',identity,'participant_code',participant,'node',node,'answers',answers,...
  'elapsed_s',toc(clock),'CT_phase_targets_used',false,'new_endpoint_properties_used',true);
 fid=fopen(target,'w','n','UTF-8');assert(fid>=0);fprintf(fid,'%s\n',jsonencode(result,'PrettyPrint',true));fclose(fid);
 fprintf('ECG DIRECT %s %s %.1f s\n',participant,node.id,result.elapsed_s);clear parts K;
end
end
function P=ports(groups)
names={groups.id};
if any(strcmp(names,'TT_Ileft')),pairs={{'TT_Ileft','TT_Iright'},{'TT_Uleft','TT_Uright'},{'A','B'},{'M','N'}};
else,pairs={{'TT_I0','TT_I1'},{'TT_U0','TT_U1'},{'A','B'},{'M','N'}};end
P=zeros(numel(groups),4);for k=1:4,[found,ix]=ismember(pairs{k},names);assert(all(found));P(ix,k)=[1;-1];end
end
