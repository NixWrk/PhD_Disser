function tests=test_electrode_faces_by_area_single_owned
% Minimal regressions for logical-vector indexing by a single boundary row.
% Five nodes, two triangles, two centres reproduce the production index error:
% f.nodes=[0 0 0;-2 0 0;0 -2 0;2 0 0;0 2 0];
% f.boundary=[1 2 3;1 4 5];
% electrode_faces_by_area(f,[-.8 -.8 0;.8 .8 0],.5,.1)
% Expected noSeed: the second electrode's only owned face shares node 1.
tests=functiontests(localfunctions);
end

function setupOnce(~)
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'vendor_stl_eidors'));
end

function testSingleOwnedFaceSharedFirstNode(t)
assert_shared_rejected(t,[1 4 5]);
end

function testSingleOwnedFaceSharedSecondNode(t)
assert_shared_rejected(t,[4 1 5]);
end

function testSingleOwnedFaceSharedThirdNode(t)
assert_shared_rejected(t,[4 5 1]);
end

function assert_shared_rejected(t,second_face)
f=struct('nodes',[0 0 0;-2 0 0;0 -2 0;2 0 0;0 2 0], ...
    'boundary',[1 2 3;second_face]);
centres=[-.8 -.8 0;.8 .8 0];
% Establish the first patch's used nodes with an independently checked output.
[first,faces]=electrode_faces_by_area(f,centres(1,:),.5,.1);
verifyEqual(t,first.nodes,[1;2;3]);
verifyEqual(t,faces,{1});
% Before the fix: shared-first crashes with an index error; shared-second
% and shared-third silently accept overlapping electrodes. All must noSeed.
verifyError(t,@() electrode_faces_by_area(f,centres,.5,.1), ...
    'electrode_faces_by_area:noSeed');
end

function testSingleBoundaryFaceAccepted(t)
f=struct('nodes',[0 0 0;2 0 0;0 2 0],'boundary',[3 1 2]);
[el,faces]=electrode_faces_by_area(f,[.3 .2 0],.5,.27);
verifyEqual(t,size(el),[1 1]);
verifyEqual(t,el.nodes,[1;2;3]);
verifyEqual(t,el.z_contact,.27);
verifyEqual(t,faces,{1});
end

function testOneOwnedFacePerElectrodeAccepted(t)
f=struct('nodes',[0 0 0;2 0 0;0 2 0;10 0 0;12 0 0;10 2 0], ...
    'boundary',[3 1 2;5 6 4]);
[el,faces]=electrode_faces_by_area(f,[.3 .2 0;10.3 .2 0],.5,.27);
verifyEqual(t,size(el),[2 1]);
verifyEqual(t,el(1).nodes,[1;2;3]);
verifyEqual(t,el(2).nodes,[4;5;6]);
verifyEmpty(t,intersect(el(1).nodes,el(2).nodes));
verifyEqual(t,[el.z_contact],[.27 .27]);
verifyEqual(t,faces,{1;2});
end

function testSingleBoundaryFaceNoSecondOwner(t)
f=struct('nodes',[0 0 0;2 0 0;0 2 0],'boundary',[1 2 3]);
verifyError(t,@() electrode_faces_by_area(f,repmat([.3 .2 0],2,1),.5,.27), ...
    'electrode_faces_by_area:noSeed');
end
