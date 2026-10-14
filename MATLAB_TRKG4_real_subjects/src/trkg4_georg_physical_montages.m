function [C,drive,measure,qc,patches]=trkg4_georg_physical_montages(pose,mm,contacts,O,plan,cfg,context,inherited)
% Permanent physical arm patches, explicit connections and optional exact
% boundary-face inheritance. Inactive patches have zero net external current.
if nargin<8,inherited=[];end
if ~isempty(inherited)&&isfield(inherited,'coarse_mm')
 [~,~,~,~,parent_patches]=trkg4_georg_physical_montages(pose,inherited.coarse_mm,contacts,O,plan,cfg,inherited.coarse_context,[]);
 inherited.patches=parent_patches;
end
physical=double(contacts.arm_physical_centres_lps_mm);
assert(size(physical,1)==12&&size(physical,2)==3);
C=cell(height(O),1);drive=C;measure=C;rows=struct([]);patches=struct([]);
for i=1:height(O)
 previous=find(O.record_id(1:i-1)==O.record_id(i),1);
 if ~isempty(previous)
  C{i}=C{previous};
  if O.is_cross(i),first=1+4*(O.axis(i)=="perpendicular");
  else,first=drive{previous}(1);end
  drive{i}=first+[0,3];measure{i}=first+[1,2];continue;
 end
 points=pose.electrodes([pose.electrodes.size_mm]==O.size_mm(i)&strcmp({pose.electrodes.axis},O.axis(i))).centres_lps_mm;
 if O.is_cross(i)
  a=pose.electrodes([pose.electrodes.size_mm]==O.size_mm(i)&strcmp({pose.electrodes.axis},'longitudinal')).centres_lps_mm;
  b=pose.electrodes([pose.electrodes.size_mm]==O.size_mm(i)&strcmp({pose.electrodes.axis},'perpendicular')).centres_lps_mm;
  groups=[num2cell([a;b],2);num2cell(physical,2)];first=1+4*(O.axis(i)=="perpendicular");
 else
  r=contacts.scenarios(strcmp({contacts.scenarios.id},['TTRKG_' char(O.arm_montage(i))]));
  assert(isscalar(r),'Unknown arm connection');groups=r.groups_mm;
  if isnumeric(groups),groups=num2cell(groups,[2,3]);end
  groups=groups(:);used=false(12,1);
  for g=1:numel(groups)
   p=reshape(groups{g},[],3);
   for k=1:size(p,1)
    [distance,index]=min(vecnorm(physical-p(k,:),2,2));assert(distance<1e-7&&~used(index));used(index)=true;
   end
  end
  groups=[groups;num2cell(physical(~used,:),2)];first=numel(groups)+1;
  groups=[groups;num2cell(points,2)];
 end
 xyz=[];gids=[];
 for g=1:numel(groups),p=reshape(groups{g},[],3);xyz=[xyz;p];gids=[gids;repmat(g,size(p,1),1)];end %#ok<AGROW>
 if isempty(inherited)
  [el,faces,certificate]=trkg4_local_exact_patches(context,xyz,contacts.nominal_area_mm2,plan.contact_impedance_ohm_m2);
 else
  prior=inherited.patches(strcmp({inherited.patches.record_id},char(O.record_id(i))));
  assert(isscalar(prior)&&isequal(prior.gids,gids)&&max(abs(prior.xyz(:)-xyz(:)))<1e-9,'Inherited physical layout changed');
  faces=cell(numel(prior.faces),1);el=struct([]);
  for e=1:numel(faces)
   if iscell(inherited.child_faces)
    ids=vertcat(inherited.child_faces{prior.faces{e}});
   else
    ids=reshape(inherited.child_faces(prior.faces{e},:)',[],1);
   end
   faces{e}=ids;
   el(e).nodes=unique(mm.boundary(ids,:)); %#ok<AGROW>
  end
  certificate=struct('scope','exact inherited parent contact faces','original_contact_geometry_preserved',true);
 end
 assert(numel(unique(vertcat(el.nodes)))==numel(vertcat(el.nodes)),'trkg4:patchOverlap','Physical patches overlap');
 grouped=struct([]);
 for e=1:numel(el)
  f=mm.boundary(faces{e},:);v=mm.nodes;
  area=vecnorm(cross(v(f(:,2),:)-v(f(:,1),:),v(f(:,3),:)-v(f(:,1),:),2),2,2)/2;
  centroid=sum((v(f(:,1),:)+v(f(:,2),:)+v(f(:,3),:))/3.*area,1)/sum(area);
  row=struct('row',i,'contact',e,'group',gids(e),'area_mm2',sum(area),...
   'area_error',abs(sum(area)/contacts.nominal_area_mm2-1),...
   'centroid_offset_mm',norm(centroid-xyz(e,:)),...
   'extent_mm',max(vecnorm(v(el(e).nodes,:)-xyz(e,:),2,2)));
  rows=append(rows,row);
 end
 for g=1:numel(groups)
  ix=find(gids==g);grouped(g).nodes=unique(vertcat(el(ix).nodes));
  grouped(g).faces=mm.boundary(vertcat(faces{ix}),:);grouped(g).z_contact=plan.contact_impedance_ohm_m2;
 end
 tmp=mm;tmp.electrode=grouped;C{i}=trkg4_cem_contact_matrix(trkg4_scale_fmdl_to_si(tmp,cfg));
 drive{i}=first+[0,3];measure{i}=first+[1,2];
 patches=append(patches,struct('record_id',char(O.record_id(i)),'faces',{faces},'xyz',xyz,'gids',gids,...
  'electrical_groups',numel(groups),'physical_patches',size(xyz,1),'certificate',certificate));
end
qc=struct('passed',max([rows.area_error])<.12&&max([rows.centroid_offset_mm])<1&&max([rows.extent_mm])<3.7,...
 'contacts',rows,'constant_arm_physical_contacts',12,...
 'unconnected_patches_are_floating_CEM_assumption',true,'patches',patches);
end

function answer=append(answer,row)
if isempty(answer),answer=row;else,answer(end+1)=row;end
end
