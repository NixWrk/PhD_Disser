function pose=trkg4_lateral_surface_pose(context,reference,offsets,angle_deg)
% Surface pose construction from Nix run_trkg4_inverse_inhale.m.
% Both cross axes and every size have one common projected centre. Spacings
% are tangent-plane offsets followed by triangle projection, not arc lengths.
[c0,n0]=full_scan_project(context,reshape(reference,1,3));
u0=[0,0,1]-n0(3)*n0;
assert(norm(u0)>1e-12,'trkg4:invalidSurfaceNormal','Longitudinal tangent is undefined');
u0=u0/norm(u0);
v0=cross(n0,u0);v0=v0/norm(v0);
[centre,normal]=full_scan_project(context,c0+offsets(1)*u0+offsets(2)*v0);
base=u0-dot(u0,normal)*normal;
assert(norm(base)>1e-12,'trkg4:invalidSurfaceNormal','Moved tangent is undefined');
base=base/norm(base);
side=cross(normal,base);side=side/norm(side);
primary=cosd(angle_deg)*base+sind(angle_deg)*side;
perpendicular=-sind(angle_deg)*base+cosd(angle_deg)*side;
rows=struct([]);
for axis={'longitudinal','perpendicular'}
 sizes=[60,80,100,120,140];direction=primary;
 if strcmp(axis{1},'perpendicular'),sizes=[60,140];direction=perpendicular;end
 for size_mm=sizes
  raw=centre+[-size_mm/2;-size_mm/4;size_mm/4;size_mm/2]*direction;
  [points,~,~]=full_scan_project(context,raw);
  row=struct('size_mm',size_mm,'axis',axis{1},'centres_lps_mm',points);
  if isempty(rows),rows=row;else,rows(end+1)=row;end %#ok<AGROW>
 end
end
assert(all(isfinite(centre))&&all(isfinite(primary))&&all(isfinite(perpendicular)),...
 'trkg4:invalidSurfaceNormal','Surface tangent basis is undefined');
pose=struct('centre_lps_mm',centre,'angle_deg',angle_deg,...
 'normal_unit_xyz',normal,'primary_unit_xyz',primary,...
 'perpendicular_unit_xyz',perpendicular,'offsets_uv_mm',reshape(offsets,1,2),...
 'electrodes',rows,'nominal_spacing_is_geodesic',false,...
 'origin','Nix common-centre tangent construction and exact triangle projection');
end
