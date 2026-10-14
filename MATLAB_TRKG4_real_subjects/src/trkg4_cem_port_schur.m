function [T,meta,fields] = trkg4_cem_port_schur(K,contacts,group_ids,currents,ground)
% Exact grouped finite-contact CEM, with every source/measurement port retained.
% K: nodal bulk matrix in SI. contacts: compiled full_scan_contact structures.
% Contacts tied to one terminal have the same positive integer group_id.
% currents: terminal currents by port; each column must sum to zero.
% T(i,j) is voltage at port i for unit drive at port j, in ohms.
% The Schur formula avoids cancellation; no inverse truncation is used.
if nargin<5,ground=1;end
n=size(K,1);g=double(group_ids(:));ng=max(g);
assert(size(K,2)==n && ground>=1 && ground<=n && ground==fix(ground));
assert(iscell(contacts)&&numel(contacts)==numel(g)&&all(g==fix(g)&g>0));
assert(isequal(unique(g),(1:ng)') && size(currents,1)==ng);
assert(all(abs(sum(currents,1))<1e-12),'Every port must conserve current');
nodes=cellfun(@(c)c.nodes(:),contacts,'UniformOutput',false);nodes=vertcat(nodes{:});
assert(numel(unique(nodes))==numel(nodes),'Physical contacts share nodes');
np=numel(nodes);R=zeros(np);P=zeros(np,ng);off=0;
for e=1:numel(contacts)
 c=contacts{e};ids=off+(1:numel(c.nodes));
 assert(isequal(size(c.R),[numel(ids),numel(ids)]));
 R(ids,ids)=c.R;P(ids,g(e))=1;off=off+numel(ids);
end
keep=(1:n)';keep(ground)=[];map=zeros(n,1);map(keep)=1:numel(keep);
Kg=(K(keep,keep)+K(keep,keep)')/2;timer=tic;
factor=decomposition(Kg,'chol');factor_seconds=toc(timer);
active=find(nodes~=ground);G=zeros(np);max_residual=0;max_backward=0;
timer=tic;knorm=norm(Kg,1);
for first=1:32:numel(active)
 ids=active(first:min(first+31,numel(active)));
 rhs=sparse(map(nodes(ids)),1:numel(ids),1,numel(keep),numel(ids));
 V=factor\rhs;defect=rhs-Kg*V;
 max_residual=max(max_residual,norm(defect,'fro')/sqrt(numel(ids)));
 max_backward=max(max_backward,norm(defect,'fro')/(knorm*norm(V,'fro')+sqrt(numel(ids))));
 G(active,ids)=V(map(nodes(active)),:);
end
green_seconds=toc(timer);clear V defect rhs
skew=norm(G-G','fro')/max(norm(G,'fro'),realmin);
assert(max_backward<1e-12&&skew<1e-9,'Selected bulk inverse failed checks');
G=(G+G')/2;M=eye(np)+R*G*R';L=chol((M+M')/2,'lower');
W=L\(R*P);S=W'*W;Ls=chol((S+S')/2,'lower');
U=Ls'\(Ls\currents);T=currents'*U;
reciprocity=max(abs(T-T'),[],'all');
assert(all(isfinite(T),'all')&&reciprocity<1e-7,'Port reciprocity failed');
meta=struct('patch_nodes',np,'terminals',ng,'ports',size(currents,2),...
 'factor_seconds',factor_seconds,'green_seconds',green_seconds,...
 'bulk_relative_residual',max_residual,'bulk_backward_error',max_backward,...
 'green_relative_skew',skew,'reciprocity_ohm',reciprocity,...
 'source_rule','linear port matrix, all physical contacts, passive zero-current terminals',...
 'regularization_used',false,'units','ohm');
fields=[];
if nargout>2
 q=R'*(L'\(L\(R*P*U)));rhs=zeros(numel(keep),size(currents,2));
 rhs(map(nodes(active)),:)=q(active,:);fields=zeros(n,size(currents,2));
 fields(keep,:)=factor\rhs;
end
end
