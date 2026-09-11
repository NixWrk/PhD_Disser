function report=run_trkg4_geometry_validation_derivative_qc()
% Independent centered-difference check inside the accepted parameter box.
cfg=trkg4_config('nik');root=cfg.project_root;
out=fullfile(root,'output','exploratory','geometry_validation_20260910');
plan=jsondecode(fileread(fullfile(out,'plan.json')));
prepared=fullfile(root,'output','exploratory','geometry_reduction_20260910','prepared_fem.mat');
load(prepared,'Kall','Klung','C','free','sizes');
maxNumCompThreads(6);n=size(Kall,1);Ksoft=Kall-Klung{1};Klung=Klung{1};
q=sparse(n+4,2);q(n+[1 4],1)=[1;-1];q(n+[2 3],2)=[1;-1];af=[free;n+(1:4)'];
x=mean(log(plan.bounds_ohm_m),2);rows=struct([]);
for k=[1 numel(sizes)]
 [~,J]=solve(x,k);
 for j=1:2
  for h=[1e-3 5e-4]
   d=zeros(2,1);d(j)=h;fd=(solve(x+d,k)-solve(x-d,k))/(2*h);
   err=abs(fd-J(j))/max(abs(J(j)),1e-10);
   row=struct('L_mm',sizes(k),'parameter_index',j,'log_step',h, ...
    'adjoint_ohm',J(j),'finite_difference_ohm',fd,'relative_error',err);
   if isempty(rows),rows=row;else,rows(end+1)=row;end %#ok<AGROW>
   fprintf('Derivative QC: L=%g parameter=%d h=%g relative_error=%.3g\n',sizes(k),j,h,err);
   assert(err<1e-4,'CEM adjoint derivative failed independent difference check');
  end
 end
end
writetable(struct2table(rows),fullfile(out,'cem_derivative_qc.csv'));
report=struct('passed',true,'rho1',exp(x(1)),'rho2',exp(x(2)), ...
 'max_relative_error',max([rows.relative_error]),'prepared_sha256',trkg4_file_sha256(prepared), ...
 'plan_sha256',trkg4_file_sha256(fullfile(out,'plan.json')), ...
 'source_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']));
f=fopen(fullfile(out,'cem_derivative_qc.json'),'w');assert(f>0);c=onCleanup(@()fclose(f));
fprintf(f,'%s',jsonencode(report,'PrettyPrint',true));
 function [z,J]=solve(v,k)
  rho=exp(v);H=C{k};H(1:n,1:n)=H(1:n,1:n)+Ksoft/rho(1)+Klung/rho(2);
  A=H(af,af);V=zeros(n+4,2);V(af,:)=decomposition((A+A')/2,'chol')\q(af,:);
  assert(norm(H(af,:)*V-q(af,:),'fro')/norm(q(af,:),'fro')<1e-7);
  z=q(:,2)'*V(:,1);
  J=[V(1:n,2)'*(Ksoft*V(1:n,1))/rho(1),V(1:n,2)'*(Klung*V(1:n,1))/rho(2)];
 end
end
