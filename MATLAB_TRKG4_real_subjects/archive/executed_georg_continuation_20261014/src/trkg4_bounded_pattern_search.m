function [best,value,info]=trkg4_bounded_pattern_search(fun,starts,lb,ub,step,minstep,budget)
% Bounded direct search from Nix run_trkg4_inverse_inhale.m.
% Meaningful spatial steps avoid false zero gradients of discrete contacts.
assert(size(starts,2)==numel(lb)&&numel(lb)==numel(ub));
assert(all(lb<ub)&&all(step>0)&&all(minstep>0)&&all(step>=minstep));
assert(budget>=size(starts,1)&&budget==fix(budget));
starts=unique(min(max(starts,lb),ub),'rows','stable');
best=starts(1,:);value=Inf;evaluated=zeros(0,numel(lb));count=0;
for k=1:size(starts,1)
 v=evaluate(starts(k,:));
 if v<value,best=starts(k,:);value=v;end
end
assert(isfinite(value),'trkg4:noAdmissibleInversePoint','No admissible starting pose');
sweep=0;
while count<budget&&any(step>=minstep)
 sweep=sweep+1;next=best;nextvalue=value;
 for j=1:numel(best)
  if step(j)<minstep(j),continue;end
  for sign=[-1,1]
   if count>=budget,break;end
   q=best;q(j)=min(max(q(j)+sign*step(j),lb(j)),ub(j));
   if any(all(abs(evaluated-q)<1e-12,2)),continue;end
   v=evaluate(q);
   if v<nextvalue,next=q;nextvalue=v;end
  end
 end
 if nextvalue<value,best=next;value=nextvalue;else,step=step/2;end
 fprintf('NIX PATTERN sweep=%d objective=%.8g evaluations=%d\n',sweep,value,count);
end
info=struct('evaluations',count,'sweeps',sweep,'resolution_reached',all(step<minstep),...
 'budget_reached',count>=budget,'final_step',step,'global_minimum_certified',false);

 function v=evaluate(q)
  v=fun(q);assert(isscalar(v)&&isreal(v)&&~isnan(v),'Invalid objective');
  evaluated(end+1,:)=q;count=count+1;
 end
end
