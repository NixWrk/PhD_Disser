function tests=test_lateral_cem_response
tests=functiontests(localfunctions);
end

function test_nix_two_rhs_and_georg_four_ports_agree(testCase)
[parts,rho,K,C,I]=fixture();n=size(K,1);
[T,J,qc]=trkg4_lateral_cem_response(K,C,parts,rho,I,1:3,1);
% The historical Nix solve, with the selected current and adjoint pair.
% The other electrode groups remain physically present with zero net current.
H=C;H(1:n,1:n)=H(1:n,1:n)+K;free=(2:size(H,1))';
rhs=sparse(size(H,1),2);rhs(n+(1:8),:)=I(:,[3,4]);U=zeros(size(rhs));
B=H(free,free);U(free,:)=decomposition((B+B')/2,'chol')\rhs(free,:);
z=rhs(:,2)'*U(:,1);oldJ=zeros(1,3);
for j=1:3,oldJ(j)=U(1:n,2)'*(parts{j}*U(1:n,1))/rho(j);end
verifyEqual(testCase,T(4,3),z,'AbsTol',1e-12);
verifyEqual(testCase,reshape(J(4,3,:),1,3),oldJ,'AbsTol',1e-12);
verifyLessThan(testCase,qc.system_relative_residual,1e-12);
verifyEqual(testCase,qc.factorizations,1);
end

function test_all_port_log_derivatives_by_finite_differences(testCase)
[parts,rho,K,C,I]=fixture();
[~,J]=trkg4_lateral_cem_response(K,C,parts,rho,I,1:3,1);h=1e-4;
for j=1:3
 rp=rho;rm=rho;rp(j)=rp(j)*exp(h);rm(j)=rm(j)*exp(-h);
 Tp=trkg4_lateral_cem_response(bulk(parts,rp),C,parts,rp,I,1:3,1);
 Tm=trkg4_lateral_cem_response(bulk(parts,rm),C,parts,rm,I,1:3,1);
 fd=(Tp-Tm)/(2*h);
 verifyLessThan(testCase,norm(fd-J(:,:,j),'fro')/norm(fd,'fro'),1e-7);
end
end

function test_ground_and_reciprocity(testCase)
[parts,rho,K,C,I]=fixture();
[T,J]=trkg4_lateral_cem_response(K,C,parts,rho,I,1:3,1);
[T2,J2]=trkg4_lateral_cem_response(K,C,parts,rho,I,1:3,12);
verifyEqual(testCase,T,T2,'AbsTol',1e-12);
verifyEqual(testCase,J,J2,'AbsTol',1e-12);
verifyEqual(testCase,T,T','AbsTol',1e-12);
end

function test_three_material_synthetic_inverse(testCase)
[parts,rho,K,C,I]=fixture();truth=trkg4_lateral_cem_response(K,C,parts,rho,I,1:3,1);
options=optimoptions('lsqnonlin','SpecifyObjectiveGradient',true,'Display','off',...
 'FunctionTolerance',1e-14,'OptimalityTolerance',1e-12,'StepTolerance',1e-12);
[x,~,~,flag]=lsqnonlin(@residual,log([5,8,30]),log([1,2,3]),log([10,30,50]),options);
verifyGreaterThan(testCase,flag,0);verifyLessThan(testCase,max(abs(exp(x)./rho-1)),1e-6);
 function [r,J]=residual(x)
  fitted=exp(x);[T,D]=trkg4_lateral_cem_response(bulk(parts,fitted),C,parts,fitted,I,1:3,1);
  r=T(:)-truth(:);J=reshape(D,[],3);
 end
end

function [parts,rho,K,C,I]=fixture()
rng(20261011);n=24;parts=cell(1,3);rho=[3,12,23];
for j=1:3
 a=triu(rand(n),1);a=a+a';parts{j}=sparse(diag(sum(a,2))-a);
end
K=bulk(parts,rho);C=sparse(n+8,n+8);groups=[1,1,2,2,3,4,5,6,7,8];
for e=1:10
 ids=[2*e-1,2*e,n+groups(e)];D=[3,1;1,4];one=ones(2,1);
 C(ids,ids)=C(ids,ids)+[D,-D*one;-one'*D,one'*D*one];
end
I=zeros(8,4);pairs={[1,4],[2,3],[5,8],[6,7]};
for p=1:4,I(pairs{p},p)=[1;-1];end
end

function K=bulk(parts,rho)
K=sparse(size(parts{1},1),size(parts{1},2));
for j=1:numel(parts),K=K+parts{j}/rho(j);end
end
