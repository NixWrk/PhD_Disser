function benchmark_arm_sigma_solver
addpath('src'); cfg=trkg4_config('nik');trkg4_prepare_runtime(cfg);maxNumCompThreads(4);
out=fullfile(cfg.project_root,'output','exploratory','arm_sigma_20260908');
s=load(fullfile(out,'prepared_fem.mat'),'fmdl_mm','blocks','bone_sigma');
spec=trkg4_arm_montage_spec(cfg,'outer_planes_inner_rings',20,80);
[model,~]=trkg4_build_mixed_arm_electrodes(s.fmdl_mm,spec,cfg.z_contact,cfg);
fmdl=trkg4_scale_fmdl_to_si(model,cfg);C=trkg4_cem_contact_matrix(fmdl);
design=trkg4_arm_conductivity_design(cfg); sig=design.baseline;
K=sig(1)*(s.blocks{1}+s.blocks{2})+sig(2)*s.blocks{3}+sig(3)*s.blocks{4}+s.bone_sigma*s.blocks{5}+C;
n=size(fmdl.nodes,1);q=zeros(n+4,2);q(n+[1 4],1)=[1;-1];q(n+[2 3],2)=[1;-1];
free=setdiff(1:n+4,fmdl.gnd_node); A=K(free,free); b=q(free,:);clear K s model fmdl C;
d=1./sqrt(diag(A));D=spdiags(d,0,numel(d),numel(d));A=D*A*D; b=D*b;
fprintf('Prepared normalized %d x %d matrix nnz %d.\n',size(A,1),size(A,2),nnz(A));
for alpha=[0 .001 .01 .1]
 try
  timer=tic;L=ichol(A,struct('type','ict','droptol',.01,'michol','off','diagcomp',alpha));
  fprintf('IC alpha %g: %.2fs, nnz %d\n',alpha,toc(timer),nnz(L));break
 catch exc
  fprintf('IC failed: %s\n',exc.message);
 end
end
v=zeros(size(b));rows=struct([]);
for j=1:2
 timer=tic;[v(:,j),flag,relres,iter]=pcg(A,b(:,j),1e-11,3000,L,L');
 rows(j)=struct('rhs',j,'flag',flag,'relative_residual',relres,'iterations',iter,'seconds',toc(timer));
 fprintf('PCG rhs %d flag %d residual %.3g iterations %d %.2fs\n',j,flag,relres,iter,toc(timer));
end
v=D*v; z=q(free,2)'*v(:,1);zr=q(free,1)'*v(:,2);
fprintf('Z %.12g reciprocal %.12g\n',z,zr);
save(fullfile(out,'solver_benchmark.mat'),'v','free','rows','z','zr','-v7.3');
writetable(struct2table(rows),fullfile(out,'solver_benchmark.csv'));
end
