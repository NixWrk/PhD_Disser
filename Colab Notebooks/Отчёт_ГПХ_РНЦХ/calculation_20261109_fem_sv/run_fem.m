function run_fem(jobs_file)
% Experiment orchestration only; all physics in canonical trkg4_lateral_cem_response.
cfg=jsondecode(fileread(jobs_file));addpath(fullfile(cfg.root,'MATLAB_TRKG4_real_subjects','src'));
maxNumCompThreads(cfg.threads);sig=jsondecode(fileread(cfg.signals_file));
identity=struct('jobs_sha256',trkg4_file_sha256(jobs_file),...
 'driver_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']),...
 'kernel_sha256',trkg4_file_sha256(which('trkg4_lateral_cem_response')),...
 'matlab_version',version,'threads',cfg.threads);
for ji=1:numel(cfg.jobs)
 job=cfg.jobs(ji);target=fullfile(cfg.output,[job.id '.json']);
 if isfile(target),v=jsondecode(fileread(target));assert(isequal(v.identity,identity));fprintf('REUSE %s\n',job.id);continue;end
 fprintf('START %s\n',job.id);jobclock=tic;
 assert(strcmp(trkg4_file_sha256(job.cem_file),job.cem_sha256));
 data=load(job.cem_file,'result');sc=data.result.scenarios;clear data;
 if ~iscell(sc),sc=num2cell(sc);end
 Cs=cell(numel(job.case_ids),1);Ps=Cs;
 for c=1:numel(Cs)
  ix=find(cellfun(@(s)strcmp(regexprep(s.id,'_(ttrkg|lateral)$',''),job.case_ids{c}),sc),1);
  assert(~isempty(ix));Cs{c}=sc{ix}.grouped_contact_matrix_SI;Ps{c}=ports(sc{ix}.ports);
 end
 clear sc;block=job.material_blocks(2);checkblock(block);data=load(block.parts_file,'parts');parts=data.parts;clear data;
 history=struct([]);last_x=[];last_rows=[];
 options=optimoptions('lsqnonlin','SpecifyObjectiveGradient',true,'Display','iter',...
  'MaxFunctionEvaluations',job.max_function_evaluations,'MaxIterations',job.max_iterations,...
  'FunctionTolerance',1e-7,'StepTolerance',1e-6,'OptimalityTolerance',1e-5);
 [x,resnorm,residual,exitflag,output,lambda,J]=lsqnonlin(@objective,log(job.initial_rho_ohm_m(:)),...
    log(job.lower_rho_ohm_m(:)),log(job.upper_rho_ohm_m(:)),options);
 [~,~]=objective(x);baseline=last_rows;rho=job.rho_fixed_ohm_m(:);rho([1 2])=exp(x);
 fit=struct('rho_ohm_m',rho','rmse_lateral_ohm',sqrt(mean(residual.^2)),...
  'residual_lateral_ohm',residual,'resnorm',resnorm,'exitflag',exitflag,'output',output,...
  'lambda',lambda,'J_free_logrho',full(J),'singular_values',svd(full(J)),'baseline',baseline,'history',history);
 result=struct('schema','individual_fem_sv_v1','job',job,'identity',identity,'fit',fit,...
   'stage','baseline_fitted','global_mesh_convergence_proven',false,'validated',false,...
   'active_material_order',{{'soft_background','lung','heart','fat'}});
 savejson([target '.partial'],result);
 variants=struct([]);
 for vi=[1 3]
  block=job.material_blocks(vi);checkblock(block);data=load(block.parts_file,'parts');vparts=data.parts;clear data;
  rows=solveall(vparts,rho);variants=[variants;struct('volume_factor',.9+.05*vi,'rows',rows)]; %#ok<AGROW>
  fprintf('VOLUME %s %.2f\n',job.id,.9+.05*vi);clear vparts;
 end
 result.volume_variants=variants;result.stage='volume_checked';savejson([target '.partial'],result);
 % Both tissue pulse parameters come ONLY from the matching FEM lateral J.
 records=sig.subjects.(job.subject).records;yb=[];
 for c=1:numel(records),yb(c,:)=records(c).signal.lateral.delta_Z_from_QRS_ohm(:)';end
 jb=vertcat(baseline.J_b_logrho);jb=jb(:,1:2);dlog=pinv(jb)*yb;
 controls=struct([]);
 for phase=job.phase_control_rr(:)'
  [~,ix]=min(abs(sig.phase_rr-phase));rr=rho;rr(1:2)=rr(1:2).*exp(dlog(:,ix));
  rows=solveall(parts,rr);
  for c=1:numel(rows)
   rows(c).linear_Z_t_ohm=baseline(c).Z_t_ohm+baseline(c).J_t_logrho(1:2)*dlog(:,ix);
   rows(c).linear_Z_b_ohm=baseline(c).Z_b_ohm+baseline(c).J_b_logrho(1:2)*dlog(:,ix);
  end
  controls=[controls;struct('phase_rr',phase,'rho_ohm_m',rr','rows',rows)]; %#ok<AGROW>
  fprintf('PULSE_CHECK %s %.2f\n',job.id,phase);
 end
 result.property_controls=controls;result.phase_rr=sig.phase_rr;result.dlogrho_FEM=dlog;
 result.stage='completed';result.elapsed_s=toc(jobclock);savejson(target,result);
 fprintf('DONE %s rmse=%.6g rho=%s seconds=%.1f\n',job.id,fit.rmse_lateral_ohm,mat2str(rho',7),result.elapsed_s);
 clear parts Cs Ps;
end
 function [f,jac]=objective(xx)
  if ~isequal(last_x,xx)
   rr=job.rho_fixed_ohm_m(:);rr(1:2)=exp(xx);last_rows=solveall(parts,rr);last_x=xx;
   z=[last_rows.Z_b_ohm]';f=z-job.observed_lateral_ohm(:);
   history=[history;struct('rho_ohm_m',rr','rmse_ohm',sqrt(mean(f.^2)),'rows',last_rows)]; %#ok<AGROW>
   savejson([target '.evaluations'],struct('identity',identity,'job',job,'history',history));
   fprintf('EVAL %s %d rho=%s RMSE=%.6g\n',job.id,numel(history),mat2str(rr(1:2)',6),sqrt(mean(f.^2)));
  end
  f=[last_rows.Z_b_ohm]'-job.observed_lateral_ohm(:);
  jac=vertcat(last_rows.J_b_logrho);jac=jac(:,1:2);
 end
 function rows=solveall(pp,rr)
  n=size(pp{1},1);K=sparse(n,n);for kk=1:numel(pp),K=K+pp{kk}/rr(kk);end
  rows=struct([]);
  for cc=1:numel(Cs)
   t0=tic;[T,JJ,qc]=trkg4_lateral_cem_response(K,Cs{cc},pp,rr,Ps{cc},[1 2 4 5],1);
   row=struct('case_id',job.case_ids{cc},'Z_t_ohm',T(2,1),'Z_b_ohm',T(4,3),...
    'J_t_logrho',reshape(JJ(2,1,:),1,[]),'J_b_logrho',reshape(JJ(4,3,:),1,[]),...
    'transfer_matrix_ohm',T,'system_checks',qc,'seconds',toc(t0));
   rows=[rows;row]; %#ok<AGROW>
  end
 end
end
function checkblock(b),assert(strcmp(trkg4_file_sha256(b.parts_file),b.sha256));end
function savejson(path,data)
 f=fopen(path,'w','n','UTF-8');assert(f>=0);cleanup=onCleanup(@()fclose(f));fprintf(f,'%s\n',jsonencode(data,'PrettyPrint',true));
end
function P=ports(groups)
 names={groups.id};
 if any(strcmp(names,'TT_Ileft')),pairs={{'TT_Ileft','TT_Iright'},{'TT_Uleft','TT_Uright'},{'A','B'},{'M','N'}};
 else,pairs={{'TT_I0','TT_I1'},{'TT_U0','TT_U1'},{'A','B'},{'M','N'}};end
 P=zeros(numel(groups),4);for k=1:4,[found,ix]=ismember(pairs{k},names);assert(all(found));P(ix,k)=[1;-1];end
end
