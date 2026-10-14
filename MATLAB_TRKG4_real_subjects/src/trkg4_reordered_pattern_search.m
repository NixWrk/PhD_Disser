function [best,value,info]=trkg4_reordered_pattern_search(fun,starts,lb,ub,step,minstep,budget,order)
% Reorder evaluation only; return parameters and step sizes in original order.
% The canonical bounded search and objective remain unchanged.
n=numel(lb);order=reshape(order,1,[]);
assert(isequal(sort(order),1:n),'trkg4:invalidSearchOrder','Order must be a parameter permutation');
restore=zeros(1,n);restore(order)=1:n;
[q,value,info]=trkg4_bounded_pattern_search(@(y)fun(y(restore)),starts(:,order),...
 lb(order),ub(order),step(order),minstep(order),budget);
best=q(restore);info.final_step=info.final_step(restore);
info.parameter_evaluation_order=order;
end
