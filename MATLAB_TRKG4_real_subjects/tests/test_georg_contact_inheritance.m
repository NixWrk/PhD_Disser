function tests=test_georg_contact_inheritance
tests=functiontests(localfunctions);
end
function setupOnce(t)
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'src'));addpath(fullfile(root,'tools'));addpath(fullfile(root,'vendor_stl_eidors'));
t.TestData.folder=fullfile(root,'output','exploratory','georg_contact_inheritance_unit');
if ~isfolder(t.TestData.folder),mkdir(t.TestData.folder);end
end
function testDirectContactMasterCanDifferFromPoseReference(t)
[mm,plan,file]=fixture(t,'direct');plan.pose_reference_mesh='different pose reference';
[inherit,id]=trkg4_georg_contact_inheritance(mm,plan,file);
verifyEmpty(t,inherit);verifyEqual(t,id.contact_parent_mesh_sha256,id.current_mesh_sha256);
end
function testDirectContactsRequireMatchingContactMaster(t)
[mm,plan,file]=fixture(t,'different');other=fullfile(t.TestData.folder,'other.mat');
nodes_mm=mm.nodes*2;elems=mm.elems;boundary=mm.boundary;save(other,'nodes_mm','elems','boundary');
plan.exact_contact_parent_mesh=other;
verifyError(t,@()trkg4_georg_contact_inheritance(mm,plan,file),'trkg4:wrongContactMaster');
end
function testMapUsesContactMasterNotPoseMaster(t)
[mm,plan,file]=fixture(t,'identity');plan.pose_reference_mesh='unused pose reference';
boundary_parent=(1:size(mm.boundary,1))';boundary_child_counts=ones(size(boundary_parent));
map=fullfile(t.TestData.folder,'identity_map.mat');save(map,'boundary_parent','boundary_child_counts');plan.boundary_parent_map=map;
[inherit,id]=trkg4_georg_contact_inheritance(mm,plan,file);
verifyEqual(t,inherit.coarse_mm.nodes,mm.nodes);verifyEqual(t,inherit.child_faces,num2cell(boundary_parent));
verifyLessThan(t,id.master_boundary_identity.maximum_relative_area_difference,1e-12);
end
function testWrongIndexBaseRejected(t)
[mm,plan,file]=fixture(t,'zero');boundary_parent=(0:size(mm.boundary,1)-1)';
map=fullfile(t.TestData.folder,'zero_map.mat');save(map,'boundary_parent');plan.boundary_parent_map=map;
verifyError(t,@()trkg4_georg_contact_inheritance(mm,plan,file),'trkg4:wrongContactMap');
end
function testFacePermutationRejectedGeometrically(t)
[mm,plan,file]=fixture(t,'permuted');boundary_parent=[2;1;3;4];
map=fullfile(t.TestData.folder,'permuted_map.mat');save(map,'boundary_parent');plan.boundary_parent_map=map;
verifyError(t,@()trkg4_georg_contact_inheritance(mm,plan,file),'trkg4:wrongContactMap');
end
function testWrongDeclaredCountsRejected(t)
[mm,plan,file]=fixture(t,'counts');boundary_parent=(1:4)';boundary_child_counts=2*ones(4,1);
map=fullfile(t.TestData.folder,'counts_map.mat');save(map,'boundary_parent','boundary_child_counts');plan.boundary_parent_map=map;
verifyError(t,@()trkg4_georg_contact_inheritance(mm,plan,file),'trkg4:wrongContactMap');
end
function [mm,plan,file]=fixture(t,name)
nodes_mm=[0,0,0;1,0,0;0,1,0;0,0,1];elems=[1,2,3,4];
boundary=[1,3,2;1,2,4;1,4,3;2,3,4];
file=fullfile(t.TestData.folder,[name '.mat']);save(file,'nodes_mm','elems','boundary');
mm=struct('nodes',nodes_mm,'elems',elems,'boundary',boundary);
plan=struct('exact_contact_parent_mesh',file,'boundary_parent_map','');
end
