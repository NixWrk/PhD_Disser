"""Read-only intersections of saved tetrahedra with a specified plane (mm)."""
import numpy as np


def section_lines(nodes, elements, centre, horizontal, inward, window=(-80,80,-5,55)):
    nodes=np.asarray(nodes,float);elements=np.asarray(elements,np.int64)
    h=np.asarray(horizontal,float);v=np.asarray(inward,float)
    if not np.allclose([np.linalg.norm(h),np.linalg.norm(v),np.dot(h,v)],[1,1,0],atol=1e-9):
        raise ValueError('Section axes must be orthonormal')
    relative=nodes-np.asarray(centre,float)
    normal=np.cross(h,v);distance=relative@normal
    signed=distance[elements]
    keep=(signed.min(1)<=1e-9)&(signed.max(1)>=-1e-9)
    candidates=elements[keep];del signed
    projected_x=(relative@h)[candidates];projected_y=(relative@v)[candidates]
    in_window=((projected_x.max(1)>=window[0])&(projected_x.min(1)<=window[1])&
               (projected_y.max(1)>=window[2])&(projected_y.min(1)<=window[3]))
    candidates=candidates[in_window];del relative,projected_x,projected_y
    x=[];y=[];count=0
    for tet in candidates:
        xyz=nodes[tet]-centre;d=distance[tet];points=[]
        for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):
            if abs(d[i])<=1e-9:points.append(xyz[i])
            if abs(d[j])<=1e-9:points.append(xyz[j])
            if d[i]*d[j]<0 and abs(d[i])>1e-9 and abs(d[j])>1e-9:
                points.append(xyz[i]+d[i]/(d[i]-d[j])*(xyz[j]-xyz[i]))
        if len(points)<3:continue
        points=np.unique(np.round(points,9),axis=0)
        if len(points)<3:continue
        xy=np.column_stack([points@h,points@v]);middle=xy.mean(0)
        xy=xy[np.argsort(np.arctan2(xy[:,1]-middle[1],xy[:,0]-middle[0]))]
        if abs(np.dot(xy[:,0],np.roll(xy[:,1],-1))-np.dot(xy[:,1],np.roll(xy[:,0],-1)))<1e-12:continue
        if xy[:,0].max()<window[0] or xy[:,0].min()>window[1] or xy[:,1].max()<window[2] or xy[:,1].min()>window[3]:continue
        xy=np.round(np.vstack([xy,xy[0]]),6)  # Display coordinates only; original mesh is unchanged.
        x.extend([*xy[:,0],None]);y.extend([*xy[:,1],None]);count+=1
    return dict(x=x,y=y,intersected_tetrahedra_in_window=count,window_mm=list(window),
                is_plane_intersection=True,is_2D_shape_quality_test=False)
