function tests=test_reordered_pattern_search
tests=functiontests(localfunctions);
end

function test_original_coordinates_are_restored(t)
f=@(x)(x(1)-.4)^2+3*(x(2)-12)^2;
[a,va,ia]=trkg4_reordered_pattern_search(f,[.1,10],[0,8],[1,16],[.1,1],[.01,.1],100,[2,1]);
[b,vb,ib]=trkg4_bounded_pattern_search(@(x)f(x([2,1])),[10,.1],[8,0],[16,1],[1,.1],[.1,.01],100);
verifyEqual(t,a,b([2,1]));verifyEqual(t,va,vb);
verifyEqual(t,ia.final_step,ib.final_step([2,1]));
verifyGreaterThanOrEqual(t,a,[0,8]);verifyLessThanOrEqual(t,a,[1,16]);
end

function test_finite_budget_reaches_priority_coordinate(t)
[q,~,info]=trkg4_reordered_pattern_search(@(x)(x(2)-1)^2,[0,0],[-2,-2],[2,2],...
 [1,1],[.1,.1],3,[2,1]);
verifyEqual(t,q,[0,1]);verifyFalse(t,info.resolution_reached);
verifyEqual(t,info.parameter_evaluation_order,[2,1]);
verifyError(t,@()trkg4_reordered_pattern_search(@(~)0,[0,0],[-1,-1],[1,1],[1,1],[.1,.1],3,[1,1]),...
 'trkg4:invalidSearchOrder');
end
