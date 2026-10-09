"""Directional CT lung depth; this geometric distance is not an electrical h."""
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates


def profile(surface, lung, affine_ras, y_mm, centre_z_mm, length_mm=140.,
            axis_step_mm=1., ray_step_mm=.5):
    if min(length_mm, axis_step_mm, ray_step_mm) <= 0:
        raise ValueError('Sampling lengths must be positive')
    s = np.linspace(-length_mm/2, length_mm/2,
                    int(np.ceil(length_mm/axis_step_mm))+1)
    z = centre_z_mm+s
    origins = np.c_[np.full(len(z), surface.bounds[0, 0]-10.),
                    np.full(len(z), y_mm), z]
    hits, rays, _ = surface.ray.intersects_location(origins,
        np.tile([1., 0., 0.], (len(z), 1)))
    skin = np.full(len(z), np.nan)
    for i in range(len(z)):
        selected = hits[rays == i]
        if len(selected):
            skin[i] = selected[:, 0].min()
    transform = np.linalg.inv(affine_ras) @ np.diag([-1., -1., 1., 1.])
    lung = lung.astype(np.uint8, copy=False)
    lung_x = np.full(len(z), np.nan)
    flags = []
    for i, x in enumerate(skin):
        if not np.isfinite(x) or x >= 0:
            flags.append('no_right_skin'); continue
        # Stop at the median plane: never substitute the contralateral lung.
        samples = x + np.arange(int(np.floor(-x/ray_step_mm))+1)*ray_step_mm
        xyz = np.c_[samples, np.full(len(samples), y_mm), np.full(len(samples), z[i])]
        ijk = nib.affines.apply_affine(transform, xyz)
        inside = np.all((ijk >= -.5) & (ijk < np.array(lung.shape)-.5), axis=1)
        values = map_coordinates(lung, ijk.T,
            order=0, mode='grid-constant', cval=0) > 0
        first = np.flatnonzero(values & inside)
        if not len(first):
            flags.append('no_right_lung_on_ray'); continue
        j = first[0]
        if not inside[:j+1].all():
            flags.append('incomplete_ct_field'); continue
        lung_x[i] = samples[j]
        flags.append('sampled_lung_boundary')
    nullable = lambda values: [float(v) if np.isfinite(v) else None for v in values]
    return dict(s_mm=s.tolist(), z_lps_mm=z.tolist(), skin_x_lps_mm=nullable(skin),
        lung_x_lps_mm=nullable(lung_x), depth_mm=nullable(lung_x-skin), flags=flags,
        direction_lps=[1., 0., 0.], y_lps_mm=float(y_mm), centre_z_lps_mm=float(centre_z_mm),
        length_mm=float(length_mm), axis_step_mm=float(s[1]-s[0]), ray_step_mm=float(ray_step_mm),
        definition='first lung voxel boundary sampled along +X from outer STL surface in right half-space',
        boundary_sampling_error_upper_mm=float(ray_step_mm), segmentation_error_quantified=False,
        is_electrical_equivalent_h=False, electrode_pose_is_illustrative=True)
