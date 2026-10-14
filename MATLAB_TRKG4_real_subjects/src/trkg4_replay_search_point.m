function [row,value,hit]=trkg4_replay_search_point(rows,q)
% Return only a completed checkpoint; running FEM points must be recomputed.
row=[];value=[];hit=false;
for k=1:numel(rows)
 r=rows{k};
 if ~any(strcmp(r.status,{'numerical_candidate_not_validated','invalid_geometry'})),continue;end
 if numel(r.parameters)~=numel(q)||any(abs(r.parameters(:)-q(:))>1e-12),continue;end
 if strcmp(r.status,'invalid_geometry')
  value=Inf;
 else
  assert(isscalar(r.mean_square_ohm2)&&isfinite(r.mean_square_ohm2)&&r.mean_square_ohm2>=0,...
   'trkg4:invalidReplayValue','Completed checkpoint has no valid objective');
  value=r.mean_square_ohm2;
 end
 row=r;hit=true;return;
end
end
