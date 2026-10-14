function tests=test_georg_local_search_parameters
tests=functiontests(localfunctions);
end

function setupOnce(~)
root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'src'));
end

function test_local_offsets_are_relative_not_LPS(t)
material_starts=log([2,9,23,3,8,22;4,15,40,4,15,40]);
[lo,hi,seeds]=trkg4_georg_local_search_parameters(material_starts,zeros(1,6),log(50)*ones(1,6),[30,45],20);
verifyEqual(t,lo(7:9),[-30,-45,-20]);verifyEqual(t,hi(7:9),[30,45,20]);
verifyEqual(t,size(seeds),[18,9]);verifyEqual(t,seeds([1,10],7:9),zeros(2,3));
verifyEqual(t,seeds(1:9,1:6),repmat(material_starts(1,:),9,1));
verifyEqual(t,seeds(10:18,1:6),repmat(material_starts(2,:),9,1));
verifyEqual(t,max(abs(seeds(:,7:8)),[],1),[20,30],'AbsTol',1e-12);
verifyTrue(t,all(all(seeds>=lo & seeds<=hi)));
end

function test_invalid_window_cannot_silently_fix_centre(t)
verifyError(t,@()trkg4_georg_local_search_parameters(ones(2,6),zeros(1,6),2*ones(1,6),[0,30],20),...
 'MATLAB:assertion:failed');
end
