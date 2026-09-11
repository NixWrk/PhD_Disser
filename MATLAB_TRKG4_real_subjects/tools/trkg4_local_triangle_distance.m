function [distance, closest, best_face] = trkg4_local_triangle_distance(ctx,point,ids)
%TRKG4_LOCAL_TRIANGLE_DISTANCE Canonical arithmetic on specified global faces.
% No broad-phase approximation: omitted faces are NOT searched here.
p1=ctx.canonical.p1(ids,:); p2=ctx.canonical.p2(ids,:); p3=ctx.canonical.p3(ids,:);
ab=p2-p1; ac=p3-p1; bc=p3-p2;
normal=cross(ab,ac,2); normal_squared=sum(normal.^2,2);
signed_distance=sum((point-p1).*normal,2)./normal_squared;
plane_point=point-signed_distance.*normal;
d00=sum(ab.*ab,2); d01=sum(ab.*ac,2); d11=sum(ac.*ac,2);
v2=plane_point-p1; d20=sum(v2.*ab,2); d21=sum(v2.*ac,2);
v=(d11.*d20-d01.*d21)./normal_squared;
w=(d00.*d21-d01.*d20)./normal_squared; u=1-v-w;
inside=u>=-1e-12 & v>=-1e-12 & w>=-1e-12;
plane_d=inf(size(inside));
plane_d(inside)=sum((plane_point(inside,:)-point).^2,2);
[qab,dab]=local_segment(point,p1,ab);
[qbc,dbc]=local_segment(point,p2,bc);
[qca,dca]=local_segment(point,p3,-ac);
[d,kind]=min([plane_d,dab,dbc,dca],[],2);
[d,index]=min(d); candidates={plane_point,qab,qbc,qca};
distance=sqrt(max(0,d)); closest=candidates{kind(index)}(index,:);
best_face=ids(index);
end

function [q,d]=local_segment(p,a,e)
t=sum((p-a).*e,2)./sum(e.*e,2);
t=max(0,min(1,t)); q=a+t.*e; d=sum((q-p).^2,2);
end
