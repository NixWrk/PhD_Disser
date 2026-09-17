function K = trkg4_cem_contact_matrix(fmdl)
% Exact linear-triangle CEM surface mass matrix, in SI units.
% Integral [N;-1]*[N;-1]' / z_c over each electrode triangle.
% Its coefficients are independently checked against EIDORS calc_system_mat.
n=size(fmdl.nodes,1); nall=n+numel(fmdl.electrode);
rows=cell(numel(fmdl.electrode),1); cols=rows; values=rows;
M=[2 1 1 -4;1 2 1 -4;1 1 2 -4;-4 -4 -4 12]/12;
for k=1:numel(fmdl.electrode)
    e=fmdl.electrode(k);
    if isfield(e,'faces') && ~isempty(e.faces)
        f=e.faces;
    else
        f=fmdl.boundary(all(ismember(fmdl.boundary,e.nodes),2),:);
    end
    assert(~isempty(f), 'Empty CEM surface');
    p=fmdl.nodes;
    area=vecnorm(cross(p(f(:,2),:)-p(f(:,1),:),p(f(:,3),:)-p(f(:,1),:),2),2,2)/2;
    idx=[f,repmat(n+k,size(f,1),1)];
    [i,j]=ndgrid(1:4,1:4);
    rows{k}=reshape(idx(:,i(:)),[],1);
    cols{k}=reshape(idx(:,j(:)),[],1);
    values{k}=reshape((area/e.z_contact)*M(:)',[],1);
end
K=sparse(vertcat(rows{:}),vertcat(cols{:}),vertcat(values{:}),nall,nall);
end
