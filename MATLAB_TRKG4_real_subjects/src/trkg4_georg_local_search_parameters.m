function [lower,upper,seeds]=trkg4_georg_local_search_parameters(material_starts,material_lower,material_upper,halfwidth,angle_halfwidth)
% Explicit tangent-plane search scenario around an author-supplied centre.
% The bounds are computational limits, never a measured placement error.
halfwidth=reshape(halfwidth,1,[]);
assert(numel(halfwidth)==2&&all(isfinite(halfwidth))&&all(halfwidth>0));
assert(isscalar(angle_halfwidth)&&isfinite(angle_halfwidth)&&angle_halfwidth>0);
assert(size(material_starts,2)==6&&numel(material_lower)==6&&numel(material_upper)==6);
assert(all(material_lower<material_upper));
lower=[reshape(material_lower,1,[]),-halfwidth,-angle_halfwidth];
upper=[reshape(material_upper,1,[]),halfwidth,angle_halfwidth];
fraction=[0,0;-2/3,0;2/3,0;0,-2/3;0,2/3;-2/3,-2/3;-2/3,2/3;2/3,-2/3;2/3,2/3];
offsets=fraction.*halfwidth;
seeds=zeros(0,9);
for k=1:size(material_starts,1)
 materials=min(max(material_starts(k,:),material_lower),material_upper);
 seeds=[seeds;repmat(materials,size(offsets,1),1),offsets,zeros(size(offsets,1),1)]; %#ok<AGROW>
end
end
