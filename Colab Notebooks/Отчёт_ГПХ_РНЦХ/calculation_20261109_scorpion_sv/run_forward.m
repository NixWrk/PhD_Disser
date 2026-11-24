function run_forward(jobs_file)
% New experiment wrapper. All electrical solves use the canonical CEM kernel.
% Reuses SHA-bound, independently assembled own-CT material blocks; no new solver.
cfg=jsondecode(fileread(jobs_file));
addpath(fullfile(cfg.root,'MATLAB_TRKG4_real_subjects','src'));
maxNumCompThreads(cfg.threads);
if ~isfolder(cfg.output),mkdir(cfg.output);end
prep=jsondecode(fileread(fullfile(fileparts(jobs_file),'prepared.json')));
kernel=which('trkg4_lateral_cem_response');
identity=struct('driver_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']),...
 'jobs_sha256',trkg4_file_sha256(jobs_file),'kernel_sha256',trkg4_file_sha256(kernel),...
 'matlab_version',version,'threads',cfg.threads);
current_subject='';
for i=1:numel(cfg.jobs)
 job=cfg.jobs(i);target=fullfile(cfg.output,[job.id '.json']);
 if isfile(target)
  previous=jsondecode(fileread(target));
  assert(strcmp(previous.identity.jobs_sha256,identity.jobs_sha256));
  assert(strcmp(previous.identity.driver_sha256,identity.driver_sha256));
  fprintf('REUSE %s\n',job.id);continue;
 end
 if ~strcmp(current_subject,job.subject)
  assert(strcmp(trkg4_file_sha256(job.cem_file),job.cem_sha256));
  data=load(job.cem_file,'result');c=data.result;clear data;
  scenarios=c.scenarios;if ~iscell(scenarios),scenarios=num2cell(scenarios);end
  ix=find(cellfun(@(s)strcmp(regexprep(s.id,'_(ttrkg|lateral)$',''),job.case_id),scenarios),1);
  assert(~isempty(ix));s=scenarios{ix};C=s.grouped_contact_matrix_SI;
  P=make_ports(s.ports);current_subject=job.subject;clear c scenarios;
 end
 answers=struct([]);control=struct([]);
 for v=1:numel(job.material_blocks)
  block=job.material_blocks(v);
  assert(strcmp(trkg4_file_sha256(block.parts_file),block.sha256));
  data=load(block.parts_file,'parts');parts=data.parts;clear data;
  rho=double(job.rho_ohm_m(:));
  tic;row=solve(parts,C,P,rho);row.seconds=toc;row.variant=block.variant;
  row.volume_factor=[.95 1 1.05];row.volume_factor=row.volume_factor(v);
  row.volume_ml=job.sphere.volume_ml*row.volume_factor;
  answers=[answers;row]; %#ok<AGROW>
  fprintf('FORWARD %s %s Z=%.10g seconds=%.2f\n',job.id,block.variant,row.Z_t_ohm,row.seconds);
  % Direct property-change checks at two endpoints for the middle h scenario.
  if v==2 && contains(job.id,'_h60_')
   ti=find(strcmp({prep.tissue_scenarios.id},job.tissue_scenario));assert(numel(ti)==1);
   tissue=prep.tissue_scenarios(ti);
   for phase=[.10 .40]
    [~,k]=min(abs(prep.phase_rr-phase));d=double(tissue.dlogrho(:,k));
    rr=rho;rr([1 5])=rr([1 5])*exp(d(1));rr(2)=rr(2)*exp(d(2));
    tic;r=solve(parts,C,P,rr);r.seconds=toc;r.phase_rr=phase;
    r.linear_Z_t_ohm=row.Z_t_ohm+(row.J_t_logrho(1)+row.J_t_logrho(4))*d(1)+row.J_t_logrho(2)*d(2);
    r.linearization_error_ohm=r.Z_t_ohm-r.linear_Z_t_ohm;
    control=[control;r]; %#ok<AGROW>
    fprintf('PULSE_CONTROL %s phase=%.2f error_mohm=%.6g\n',job.id,phase,1000*r.linearization_error_ohm);
   end
  end
  clear parts;
 end
 result=struct('schema','scorpion_sv_fem_v1','job',job,'identity',identity,...
  'answers',answers,'property_controls',control,'FEM_was_run',true,...
  'global_mesh_convergence_proven',false,'individual_tissue_properties_validated',false,...
  'source_excitation','time_division_own_drive','active_material_order',{{'soft','lung','heart','fat'}});
 fid=fopen(target,'w','n','UTF-8');assert(fid>=0);
 fprintf(fid,'%s\n',jsonencode(result,'PrettyPrint',true));fclose(fid);
end
end

function row=solve(parts,C,P,rho)
n=size(parts{1},1);K=sparse(n,n);
for k=1:numel(parts),K=K+parts{k}/rho(k);end
[T,J,qc]=trkg4_lateral_cem_response(K,C,parts,rho,P,[1 2 4 5],1);
row=struct('rho_ohm_m',rho','Z_t_ohm',T(2,1),'Z_b_ohm',T(4,3),...
 'J_t_logrho',reshape(J(2,1,:),1,[]),'J_b_logrho',reshape(J(4,3,:),1,[]),...
 'transfer_matrix_ohm',T,'system_checks',qc);
end

function P=make_ports(groups)
names={groups.id};
if any(strcmp(names,'TT_Ileft'))
 pairs={{'TT_Ileft','TT_Iright'},{'TT_Uleft','TT_Uright'},{'A','B'},{'M','N'}};
else
 pairs={{'TT_I0','TT_I1'},{'TT_U0','TT_U1'},{'A','B'},{'M','N'}};
end
P=zeros(numel(groups),4);
for k=1:4,[found,ix]=ismember(pairs{k},names);assert(all(found));P(ix,k)=[1;-1];end
end
