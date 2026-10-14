function tests=test_replay_search_point
tests=functiontests(localfunctions);
end

function test_interrupted_point_is_not_reused(t)
row=struct('parameters',[1 2],'status','forward_calculation_running','mean_square_ohm2',[]);
[~,~,hit]=trkg4_replay_search_point({row},[1 2]);verifyFalse(t,hit);
row.status='invalid_geometry';[~,value,hit]=trkg4_replay_search_point({row},[1 2]);
verifyTrue(t,hit);verifyEqual(t,value,Inf);
row.status='numerical_candidate_not_validated';
verifyError(t,@()trkg4_replay_search_point({row},[1 2]),'trkg4:invalidReplayValue');
end

function test_replay_preserves_coupled_search(t)
addpath(fileparts(fileparts(mfilename('fullpath'))));
checkpoints={};calls=0;fun=@(q)(exp(q(1))-4)^2+(round(q(2)/5)-3)^2;
args={[log(2),0],[0,-20],[log(8),20],[.25,5],[.001,.5],35};
[expected,v,info]=trkg4_bounded_pattern_search(@record,args{:});
original_calls=calls;checkpoints=checkpoints(1:12);calls=0;
[actual,w,resumed]=trkg4_bounded_pattern_search(@reuse,args{:});
verifyEqual(t,actual,expected);verifyEqual(t,w,v);verifyEqual(t,resumed,info);
verifyEqual(t,calls,original_calls-12);
 function value=record(q)
  calls=calls+1;value=fun(q);
  checkpoints{end+1}=struct('parameters',q,'status','numerical_candidate_not_validated','mean_square_ohm2',value);
 end
 function value=reuse(q)
  [~,value,hit]=trkg4_replay_search_point(checkpoints,q);
  if ~hit,calls=calls+1;value=fun(q);end
 end
end
