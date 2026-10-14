function tests=test_lateral_surface_pose
tests=functiontests(localfunctions);
end

function setupOnce(~)
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'src'),fullfile(root,'vendor_stl_eidors'),fullfile(root,'tools','full_pose_scan'));
end

function test_common_centre_and_nominal_sizes(t)
ctx=plane_context();pose=trkg4_lateral_surface_pose(ctx,[-210,0,0],[20,30],0);
verifyEqual(t,pose.centre_lps_mm,[-200,30,20],'AbsTol',1e-10);
verifyEqual(t,pose.primary_unit_xyz,[0,0,1],'AbsTol',1e-12);
verifyEqual(t,pose.perpendicular_unit_xyz,[0,1,0],'AbsTol',1e-12);
verifyEqual(t,numel(pose.electrodes),7);
for r=pose.electrodes
 direction=pose.primary_unit_xyz;
 if strcmp(r.axis,'perpendicular'),direction=pose.perpendicular_unit_xyz;end
 expected=pose.centre_lps_mm+[-r.size_mm/2;-r.size_mm/4;r.size_mm/4;r.size_mm/2]*direction;
 verifyEqual(t,r.centres_lps_mm,expected,'AbsTol',1e-10);
end
end

function test_cross_rotates_with_one_common_angle(t)
ctx=plane_context();p=trkg4_lateral_surface_pose(ctx,[-200,0,0],[0,0],17);
verifyEqual(t,p.primary_unit_xyz,[0,sind(17),cosd(17)],'AbsTol',1e-12);
verifyEqual(t,p.perpendicular_unit_xyz,[0,cosd(17),-sind(17)],'AbsTol',1e-12);
verifyEqual(t,dot(p.primary_unit_xyz,p.perpendicular_unit_xyz),0,'AbsTol',1e-12);
for r=p.electrodes
 verifyEqual(t,mean(r.centres_lps_mm,1),p.centre_lps_mm,'AbsTol',1e-10);
 verifyEqual(t,norm(r.centres_lps_mm(4,:)-r.centres_lps_mm(1,:)),r.size_mm,'AbsTol',1e-10);
end
end

function ctx=plane_context()
[y,z]=ndgrid(-300:50:300,-300:50:300);nodes=[repmat(-200,numel(y),1),y(:),z(:)];
faces=delaunay(y(:),z(:));f=struct('nodes',nodes,'boundary',faces);
pc=electrode_patch_context(f);
ctx=struct('nodes',nodes,'boundary',faces,'normals',repmat([-1,0,0],size(nodes,1),1),...
 'boundary_nodes',(1:size(nodes,1))','tree',KDTreeSearcher(nodes),...
 'node_face_list',double(pc.node_face_list),'node_face_start',double(pc.node_face_start));
end
