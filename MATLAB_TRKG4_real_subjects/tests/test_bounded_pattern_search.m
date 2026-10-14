function tests=test_bounded_pattern_search
tests=functiontests(localfunctions);
end

function test_material_fit_with_discrete_pose(t)
% Infinitesimal pose derivatives vanish; finite spatial steps must move it.
fun=@(q)(exp(q(1))-4)^2+(round(q(2)/5)-3)^2;
[q,value,info]=trkg4_bounded_pattern_search(fun,[log(2),0],[0,-20],[log(8),20],...
 [.25,5],[.001,.5],150);
verifyLessThan(t,value,.001);
verifyEqual(t,round(q(2)/5),3);
verifyLessThanOrEqual(t,info.evaluations,150);
verifyFalse(t,info.global_minimum_certified);
end

function test_invalid_geometry_is_not_a_candidate(t)
[q,value,info]=trkg4_bounded_pattern_search(@geometry_objective,[-1;1],-2,2,.5,.01,50);
verifyGreaterThanOrEqual(t,q,0);verifyLessThan(t,value,1e-12);
verifyLessThanOrEqual(t,info.evaluations,50);
verifyError(t,@()trkg4_bounded_pattern_search(@(~)Inf,0,-1,1,.5,.1,10),...
 'trkg4:noAdmissibleInversePoint');
end

function test_incomplete_sweep_does_not_claim_resolution(t)
[~,~,info]=trkg4_bounded_pattern_search(@(~)1,[0,0],[-1,-1],[1,1],...
 [.2,.2],[.15,.15],3);
verifyFalse(t,info.resolution_reached);verifyEqual(t,info.completed_sweeps,0);
verifyEqual(t,info.final_step,[.2,.2]);
end

function value=geometry_objective(q)
if q<0,value=Inf;else,value=(q-.5)^2;end
end
