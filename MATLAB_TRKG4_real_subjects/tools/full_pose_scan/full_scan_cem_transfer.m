function z = full_scan_cem_transfer(fmdl_si, patch_faces, global_to_green, green, z_contact, prepared)
%FULL_SCAN_CEM_TRANSFER Exact four-electrode CEM transfer resistance in ohms.
% patch_faces: four cells of induced triangular boundary face IDs, in order
% I+, V+, V-, I-. Patches must be node-disjoint (canonical builder + QC).
% Coordinates are metres, scalar positive contact impedance is ohm*m^2.
%
% The full grounded CEM is [K+D, -D*P; -P'*D, P'*D*P]. With D=R'*R,
% M=I+R*G*R', its electrode Schur complement equals
% S=(R*P)'*M^(-1)*(R*P). This is the SAME triangular consistent mass CEM
% as EIDORS, without the cancellation in C-B'*(I+G*D)^(-1)*G*B.
% Both solves are SPD Cholesky solves; no regularisation/truncation is used.
% A grounded patch node keeps a positive Green index and an exactly zero
% Green row/column. Its contact mass entries are retained, as in full CEM.

if nargin<6, prepared=[]; end
if ~isempty(prepared) && (~iscell(prepared)||numel(prepared)~=4)
    error('full_scan:contactCache','Expected four compiled contacts');
end
if ~iscell(patch_faces) || numel(patch_faces)~=4
    error('full_scan:patchInput','Expected four induced-face lists.');
end
if ~isa(green,'double') || ~ismatrix(green) || ~isreal(green) || issparse(green) || size(green,1)~=size(green,2)
    error('full_scan:greenInput','Expected a real dense double Green matrix.');
end
validateattributes(z_contact,{'numeric'},{'real','finite','scalar','positive'});
if numel(global_to_green)~=size(fmdl_si.nodes,1)
    error('full_scan:greenMap','Expected one Green index per mesh node (zero means absent).');
end
nodes_by_electrode=cell(4,1); triangles=cell(4,1);
for e=1:4
    faces=patch_faces{e};
    validateattributes(faces,{'numeric'},{'vector','nonempty','integer','>=',1,'<=',size(fmdl_si.boundary,1)});
    if numel(unique(faces))~=numel(faces)
        error('full_scan:patchInput','An electrode contains duplicate faces.');
    end
    triangles{e}=fmdl_si.boundary(faces,:);
    nodes_by_electrode{e}=unique(triangles{e}(:));
end
patch_nodes=vertcat(nodes_by_electrode{:});
if numel(unique(patch_nodes))~=numel(patch_nodes)
    error('full_scan:patchOverlap','CEM electrode patches must not share nodes.');
end
index=double(global_to_green(patch_nodes));
if any(~isfinite(index) | index<1 | index>size(green,1) | index~=fix(index)) || numel(unique(index))~=numel(index)
    error('full_scan:greenMap','Missing or nonunique patch-node mapping (including ground).');
end
G=green(index,index);
if any(~isfinite(G),'all')
    error('full_scan:greenInput','Non-finite patch Green entries.');
end
skew=norm(G-G',1)/max(norm(G,1),realmin);
if skew>1e-10, error('full_scan:greenSymmetry','Patch Green is not symmetric: %.3g.',skew); end
G=(G+G')/2;
ground=find(patch_nodes==fmdl_si.gnd_node);
if ~isempty(ground) && (any(G(ground,:)~=0) || any(G(:,ground)~=0))
    error('full_scan:groundGreen','The grounded patch node must have a zero Green row and column.');
end
n=numel(patch_nodes); R=zeros(n); P=zeros(n,4); offset=0;
for e=1:4
    nodes=nodes_by_electrode{e}; tri=triangles{e};
    indices=offset+(1:numel(nodes));
    if isempty(prepared)
        compiled=full_scan_contact(fmdl_si,patch_faces{e},z_contact);
    else
        compiled=prepared{e};
        if ~isequal(compiled.faces,double(patch_faces{e}(:))) || ...
                ~isequal(compiled.triangles,tri) || ~isequal(compiled.nodes,nodes) || ...
                ~isequal(compiled.xyz,fmdl_si.nodes(nodes,:)) || compiled.z_contact~=z_contact
            error('full_scan:contactCache','Compiled contact does not match this mesh/contact');
        end
    end
    contact_R=compiled.R;
    R(indices,indices)=contact_R; P(indices,e)=1; offset=offset+numel(nodes);
end
M=R*G*R'; M(1:n+1:end)=M(1:n+1:end)+1;
[L,flag]=chol((M+M')/2,'lower');
if flag~=0, error('full_scan:localSPD','Local bulk/contact system is not positive definite.'); end
W=L\(R*P);
S=W'*W;
[Ls,flag]=chol((S+S')/2,'lower');
if flag~=0, error('full_scan:electrodeSPD','Electrode Schur complement is not positive definite.'); end
voltage=Ls'\(Ls\[1;0;0;-1]);
z=voltage(2)-voltage(3);
if ~isfinite(z), error('full_scan:nonfiniteCEM','CEM transfer is not finite.'); end
end
