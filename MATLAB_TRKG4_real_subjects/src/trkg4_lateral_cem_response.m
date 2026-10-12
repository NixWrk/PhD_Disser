function [T,J,qc]=trkg4_lateral_cem_response(K,C,parts,rho,ports,active,ground)
% FEM/adjoint core from run_lateral_fat_inverse (Nix, 33.09, section 5.6).
% Multiple unit-current ports share one full CEM factorization. No reduction,
% truncation, fitted gain or change of material/finite-contact model is used.
n=size(K,1);ng=size(C,1)-n;
assert(ng>0&&size(C,2)==n+ng&&size(ports,1)==ng);
assert(all(abs(sum(ports,1))<1e-12),'Port currents must be balanced');
assert(ground>=1&&ground<=n&&ground==fix(ground));
assert(numel(rho)==numel(parts)&&all(isfinite(rho))&&all(rho>0));
% Sparse submatrix assignment scans a huge mostly-zero rectangle. The block
% sum is exactly the same CEM operator and scales with stored entries.
H=C+blkdiag(K,sparse(ng,ng));
free=(1:n+ng)';free(ground)=[];
rhs=sparse(n+ng,size(ports,2));rhs(n+(1:ng),:)=ports;
B=H(free,free);U=zeros(size(rhs));
U(free,:)=decomposition((B+B')/2,'chol')\rhs(free,:);
res=norm(H(free,:)*U-rhs(free,:),'fro')/norm(rhs(free,:),'fro');
T=ports'*U(n+(1:ng),:);rec=max(abs(T-T'),[],'all');
assert(res<1e-7&&rec<1e-7&&all(isfinite(T),'all'),'Full CEM solve failed');
J=zeros(size(T,1),size(T,2),numel(active));
for j=1:numel(active)
 m=active(j);J(:,:,j)=U(1:n,:)'*(parts{m}*U(1:n,:))/rho(m);
end
qc=struct('system_relative_residual',res,'reciprocity_ohm',rec,...
 'factorizations',1,'right_hand_sides',size(ports,2));
end
