"""Declared model montages on a new skin; preserve input coordinates separately."""
import copy
import numpy as np


def skin_arc(surface, y, z, offsets):
    section = surface.section(plane_origin=[0.,y,z],plane_normal=[0.,1.,0.])
    if section is None: raise ValueError('Declared lateral plane misses the new skin')
    candidates = []
    for trace in section.discrete:
        if not np.allclose(trace[0],trace[-1],rtol=0,atol=1e-6): continue
        for index,(a,b) in enumerate(zip(trace[:-1],trace[1:])):
            if a[2] == b[2] or not min(a[2],b[2]) <= z <= max(a[2],b[2]): continue
            point = a + (b-a)*(z-a[2])/(b[2]-a[2])
            candidates.append((point[0],point,trace,index))
    if not candidates: raise ValueError('Lateral centre does not cross a closed skin section')
    _,origin,trace,index = min(candidates,key=lambda item:item[0])
    cycle = np.vstack([origin,trace[index+1:-1],trace[:index+1],origin])
    cycle = cycle[np.r_[True,np.linalg.norm(np.diff(cycle,axis=0),axis=1)>1e-9]]
    if cycle[1,2] < origin[2]: cycle = cycle[::-1]
    arc = np.r_[0.,np.linalg.norm(np.diff(cycle,axis=0),axis=1).cumsum()]
    if max(abs(np.asarray(offsets))) >= arc[-1]/2:
        raise ValueError('Montage exceeds the local half-contour')
    interpolated = np.column_stack([np.interp(np.asarray(offsets)%arc[-1],arc,cycle[:,i]) for i in range(3)])
    projected,dist,_ = surface.nearest.on_surface(interpolated)
    if max(dist) > .1: raise ValueError('Skin-arc interpolation departed from the source skin')
    return projected


def apply(cfg, skin, plan):
    import ct_fem_core as core
    physical = cfg['contacts']['physical_centres_lps_mm']
    before = copy.deepcopy(physical); assigned = set()
    joins = [core.full_section_join(skin,a) for a in cfg['synthetic_cylinders']]
    for rule in plan:
        if not rule.get('basis'): raise ValueError('Every placement rule needs its provenance')
        if rule['kind'] == 'skin_arc':
            points = skin_arc(skin,rule['y_mm'],rule['z_mm'],rule['offsets_mm'])
            ids = rule['ids']
        elif rule['kind'] == 'cylinder':
            index = rule['cylinder_index']; arm = cfg['synthetic_cylinders'][index];join = joins[index]
            angle = np.deg2rad(rule['angle_degrees'])
            points = [[arm['cut_x_mm'] + arm['direction']*rule['axial_distance_mm'],
                join['centre_yz_mm'][0]+arm['radius_mm']*np.cos(angle),
                join['centre_yz_mm'][1]+arm['radius_mm']*np.sin(angle)]]
            ids = [rule['id']]
        else: raise ValueError('Unsupported montage placement rule')
        if len(ids) != len(points): raise ValueError('Placement IDs and offsets differ')
        for name,point in zip(ids,points):
            if name not in physical or name in assigned: raise ValueError('Unknown or duplicate contact rule')
            assigned.add(name)
            physical[name]['raw_lps_mm'] = np.asarray(point).tolist()
            physical[name]['provenance'] = rule['basis']+'; model placement reconstructed on TotalSegmentator surface, not a measured coordinate'
    if assigned != set(physical): raise ValueError('Every physical contact needs an explicit rule')
    return dict(input_centres=before,reconstructed_centres=copy.deepcopy(physical),rules=plan,
        joins=joins,measured_positions=False,original_centres_preserved_separately=True)
