function [signature,parent_rows]=trkg4_georg_material_rules(cfg,points,fraction_sha256,scale_helper_sha256,parent)
% Describe assembly units and select only identical parent material rules.
% A zero row means recompute. Geometry and field-builder equality are checked
% by the caller; equal point counts alone do not authorize matrix reuse.
if nargin<5,parent=[];end
points=reshape(points,1,[]);
assert(isnumeric(points)&&numel(points)==2&&all(isfinite(points))&&...
 all(points>0)&&all(points==fix(points))&&points(2)>points(1),...
 'trkg4:invalidMaterialRules','Expected two ascending positive integer point counts');
fraction_sha256=cellstr(string(fraction_sha256));fraction_sha256=reshape(fraction_sha256,1,[]);
scale_helper_sha256=char(scale_helper_sha256);
assert(numel(fraction_sha256)==numel(points)&&...
 all(cellfun(@valid_sha,fraction_sha256))&&valid_sha(scale_helper_sha256),...
 'trkg4:invalidMaterialHash','Expected SHA256 for each fractions file and the scale helper');
unit=lower(strtrim(char(cfg.length_unit)));
assert(ismember(unit,{'m','meter','metre','mm','millimeter','millimetre'}),...
 'trkg4:invalidMaterialUnits','Unsupported geometry length unit');
flag=cfg.scale_nodes_to_si;
assert((islogical(flag)||isnumeric(flag))&&isscalar(flag)&&isfinite(flag)&&...
 (flag==0||flag==1),'trkg4:invalidMaterialUnits','scale_nodes_to_si must be a boolean');
scale=1;
if logical(flag)&&ismember(unit,{'mm','millimeter','millimetre'}),scale=1e-3;end
signature=struct('length_unit',unit,'scale_nodes_to_si',logical(flag),...
 'node_scale_to_si',scale,'scale_helper_sha256',lower(scale_helper_sha256),...
 'quadrature_points',points,'fractions_sha256',{cellfun(@lower,fraction_sha256,'UniformOutput',false)});
parent_rows=zeros(1,numel(points));
if isempty(parent),return;end
for name={'length_unit','scale_nodes_to_si','node_scale_to_si','scale_helper_sha256'}
 if ~isfield(parent,name{1})||~isequal(signature.(name{1}),parent.(name{1})),return;end
end
if ~isfield(parent,'quadrature_points')||~isfield(parent,'fractions_sha256'),return;end
parent_points=reshape(parent.quadrature_points,1,[]);
parent_hashes=reshape(cellstr(string(parent.fractions_sha256)),1,[]);
assert(numel(parent_points)==numel(parent_hashes)&&...
 numel(unique(parent_points))==numel(parent_points),...
 'trkg4:invalidParentMaterialRules','Parent rule metadata is ambiguous');
for rule=1:numel(points)
 row=find(parent_points==points(rule));
 if ~isempty(row)&&strcmp(signature.fractions_sha256{rule},parent_hashes{row})
  parent_rows(rule)=row;
 end
end
end

function yes=valid_sha(value)
yes=~isempty(regexp(value,'^[0-9a-fA-F]{64}$','once'));
end
