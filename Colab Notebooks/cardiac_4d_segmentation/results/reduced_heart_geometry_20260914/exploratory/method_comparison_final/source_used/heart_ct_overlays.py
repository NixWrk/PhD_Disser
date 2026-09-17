"""Render saved reduced-heart candidates in their own source CT coordinate frame.

No segmentation, model refit, 4D-to-static registration or FEM solve is performed.
Run with --config (machine-local geometry run JSON), --baseline and a NEW --output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from cardiac_function_analysis import verify_grid
from run_reduced_heart_geometry import digest, load_crop, mask_path

MODELS = ('sphere', 'ellipsoid_volume_scaled', 'individual_isotropic')
LABELS = ('Сфера', 'Эллипсоид', 'Индивидуальная форма')
COLORS = ('#FFB000', '#F26DA8', '#A6FF4D')
TARGET_COLOR = '#32D6FF'
PLANES = (('Аксиальная', 0, 1, 2, 'R', 'L', 'A', 'P'),
          ('Корональная', 0, 2, 1, 'R', 'L', 'S', 'I'),
          ('Сагиттальная', 1, 2, 0, 'A', 'P', 'S', 'I'))


def world_to_index(world, affine):
    inv = np.linalg.inv(np.asarray(affine, dtype=float))
    return np.asarray(world) @ inv[:3, :3].T + inv[:3, 3]


def voxel_membership(mask, affine, world):
    indices = np.floor(world_to_index(world, affine) + .5).astype(np.int64)
    flat = indices.reshape(-1, 3)
    inside = np.all((flat >= 0) & (flat < np.asarray(mask.shape)), axis=1)
    result = np.zeros(len(flat), dtype=bool)
    result[inside] = mask[tuple(flat[inside].T)]
    return result.reshape(indices.shape[:-1])


def candidate_membership(parameters, world, reference_mask=None):
    if parameters['kind'] == 'individual_isotropic_volume_centroid':
        affine = np.asarray(parameters['transform_world']) @ np.asarray(parameters['reference_affine'])
        return voxel_membership(reference_mask, affine, world)
    delta = np.asarray(world) - np.asarray(parameters['center_mm'])
    normalised = (delta @ np.asarray(parameters['axes_matrix'])) / np.asarray(parameters['semi_axes_mm'])
    return np.sum(normalised ** 2, axis=-1) <= 1


def plane_grid(center, bounds, horizontal, vertical, fixed, pixels=480):
    # RAS+: increasing R or A is displayed to the left; A/S upward.
    u = np.linspace(bounds[horizontal, 1], bounds[horizontal, 0], pixels)
    v = np.linspace(bounds[vertical, 0], bounds[vertical, 1], pixels)
    uu, vv = np.meshgrid(u, v)
    world = np.broadcast_to(center, (pixels, pixels, 3)).copy()
    world[..., horizontal] = uu
    world[..., vertical] = vv
    world[..., fixed] = center[fixed]
    return world, u, v


def sample_ct(data, affine, world):
    index = world_to_index(world, affine)
    return map_coordinates(data, np.moveaxis(index, -1, 0), order=1,
                           mode='constant', cval=np.nan, prefilter=False)


def select_examples(records):
    selected = []
    for subject in ('adam', 'nix', 'georg'):
        rows = [r for r in records if r['subject'] == subject and r['region'] == 'whole_heart']
        cycles = sorted({r['cycle_index'] for r in rows})
        groups = [[r for r in rows if r['cycle_index'] == cycle] for cycle in cycles]
        group = max(groups, key=len)
        reference = next(r for r in group if r['is_reference_phase'])
        target = max((r for r in group if not r['is_reference_phase']),
                     key=lambda r: abs(r['moments']['volume_ml'] / reference['moments']['volume_ml'] - 1))
        selected.append((target, reference))
    return selected


def field_bounds(target, models=MODELS):
    p = target['comparisons']['individual_isotropic']['parameters']
    lo, hi = np.asarray(p['reference_bbox_index_centers'])
    corners = np.array(list(product(*zip(lo - .5, hi + .5))))
    affine = np.asarray(p['transform_world']) @ np.asarray(p['reference_affine'])
    points = corners @ affine[:3, :3].T + affine[:3, 3]
    low, high = points.min(axis=0), points.max(axis=0)
    # Include target and all candidates, including any extent outside source CT.
    lo, hi = np.asarray(target['moments']['bbox_index_centers'])
    crop = np.asarray(target['provenance']['crop_origin_index'])
    affine = np.asarray(target['provenance']['source_affine'])
    corners = np.array(list(product(*zip(lo + crop - .5, hi + crop + .5))))
    points = corners @ affine[:3, :3].T + affine[:3, 3]
    low, high = np.minimum(low, points.min(axis=0)), np.maximum(high, points.max(axis=0))
    for model in models:
        if model == "individual_isotropic": continue
        p = target['comparisons'][model]['parameters']
        half = np.sqrt(np.sum((np.asarray(p['axes_matrix']) * np.asarray(p['semi_axes_mm']))**2, axis=1))
        low = np.minimum(low, np.asarray(p['center_mm']) - half)
        high = np.maximum(high, np.asarray(p['center_mm']) + half)
    center = np.asarray(target['moments']['centroid_mm'])
    half = np.maximum(np.maximum(center - low, high - center) + 18, [105, 95, 0])
    return np.column_stack((center - half, center + half))


def make_figure(data, ct_affine, target_mask, target_affine, reference_mask, row,
                *, models=MODELS, labels=LABELS, colors=COLORS):
    center = np.asarray(row['moments']['centroid_mm'])
    bounds = field_bounds(row, models)
    fig, axes = plt.subplots(3, len(models), figsize=(4 * len(models), 11.5), squeeze=False)
    cmap = plt.get_cmap('gray').copy(); cmap.set_bad('#383242')
    sample_checks = []
    for i, (plane, horizontal, vertical, fixed, left, right, top, bottom) in enumerate(PLANES):
        world, u, v = plane_grid(center, bounds, horizontal, vertical, fixed)
        ct = sample_ct(data, ct_affine, world)
        target = voxel_membership(target_mask, target_affine, world)
        x, y = u - center[horizontal], v - center[vertical]
        dx, dy = x[1] - x[0], y[1] - y[0]
        extent = [x[0] - dx/2, x[-1] + dx/2, y[0] - dy/2, y[-1] + dy/2]
        sample_checks.append({'plane': plane, 'fixed_RAS_axis': fixed,
            'fixed_coordinate_mm': float(center[fixed]), 'outside_CT_fraction': float(np.isnan(ct).mean()),
            'pixel_spacing_mm': [abs(float(dx)), abs(float(dy))]})
        for j, (model, label, color) in enumerate(zip(models, labels, colors)):
            ax = axes[i, j]
            candidate = candidate_membership(row['comparisons'][model]['parameters'], world, reference_mask)
            ax.imshow(ct, origin='lower', extent=extent, cmap=cmap, vmin=-200, vmax=300,
                      interpolation='nearest', aspect='equal')
            ax.contourf(x, y, candidate.astype(float), levels=[.5, 1.5], colors=[color], alpha=.10)
            ax.contour(x, y, candidate, levels=[.5], colors=[color], linewidths=1.25)
            ax.contour(x, y, target, levels=[.5], colors=[TARGET_COLOR], linewidths=1.15, linestyles='--')
            ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                dice = row['comparisons'][model]['overlap']['dice_voxel_center_approx']
                ax.set_title(f'{label}\n3D Dice = {dice:.3f}'.replace('.', ','), fontsize=11)
            if j == 0: ax.set_ylabel(plane, fontsize=11, labelpad=10)
            for label, position in ((left,(.025,.5)),(right,(.975,.5)),(top,(.5,.97)),(bottom,(.5,.025))):
                ax.text(*position, label, transform=ax.transAxes, color='white', ha='center', va='center', fontsize=8,
                        bbox={'facecolor':'black','alpha':.5,'edgecolor':'none','pad':1})
            # A physical 50 mm scale, identical in meaning in every panel.
            start = x[0] - 10
            yy = y[0] + 10
            ax.plot([start, start - 50], [yy, yy], color='white', lw=2)
            ax.text(start - 25, yy + 4, '50 мм', color='white', ha='center', fontsize=8)
    phase = row['phase_percent_within_cycle']
    title = f"{row['subject'].capitalize()} · цикл {row['cycle_index']+1} · фаза {phase:g}% R–R · целое сердце"
    fig.suptitle(title, fontsize=13, y=.99)
    handles=[Line2D([0],[0],color=TARGET_COLOR,lw=1.5,ls='--',label='Автоматическая маска целевой фазы')]
    handles += [Line2D([0],[0],color=c,lw=1.8,label=t) for c,t in zip(colors,labels)]
    fig.legend(handles=handles, loc='lower center', ncol=4 if len(models)>3 else 2,
               frameon=False, bbox_to_anchor=(.5,.005), fontsize=9 if len(models)>3 else 10)
    fig.tight_layout(rect=[0,.12 if len(models)>3 else .055,1,.965])
    return fig, sample_checks


def run(config_path, baseline, output):
    output=Path(output)
    if output.exists(): raise ValueError('Output must be new; existing images are never overwritten')
    cfg=json.loads(Path(config_path).read_text(encoding='utf-8'))
    root=Path(cfg['derived_root']); baseline=Path(baseline)
    record_path=baseline/'geometry_comparison.jsonl'
    records=[json.loads(x) for x in record_path.read_text(encoding='utf-8').splitlines()]
    assert json.loads((baseline/'run_status.json').read_text())['status']=='completed'
    sources=[]; results=[]
    output.mkdir(parents=True)
    for row, reference in select_examples(records):
        subject=row['subject']; manifest_path=root/subject/'phases/conversion_manifest.json'
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        infos={x['phase_id']:x for x in manifest['phases']}
        target_ct_path=root/subject/'phases'/infos[row['phase_id']]['nifti']
        ref_ct_path=root/subject/'phases'/infos[reference['phase_id']]['nifti']
        ct=nib.load(str(target_ct_path)); ref_ct=nib.load(str(ref_ct_path))
        verify_grid(ct,infos[row['phase_id']]); verify_grid(ref_ct,infos[reference['phase_id']])
        assert digest(target_ct_path)==row['ct_sha256'], 'CT changed since geometry run'
        target,affine,provenance=load_crop(mask_path(root,subject,row['phase_id'],'whole_heart'),infos[row['phase_id']],ct)
        ref,ref_affine,ref_provenance=load_crop(mask_path(root,subject,reference['phase_id'],'whole_heart'),infos[reference['phase_id']],ref_ct)
        assert provenance['sha256']==row['provenance']['sha256']
        assert ref_provenance['sha256']==row['reference_mask_sha256']
        assert np.allclose(ref_affine,row['comparisons']['individual_isotropic']['parameters']['reference_affine'],rtol=0,atol=1e-10)
        data=np.asarray(ct.dataobj,dtype=np.float32)
        fig,checks=make_figure(data,ct.affine,target,affine,ref,row)
        filename=f'ct_{subject}.png'
        fig.savefig(output/filename,dpi=155,bbox_inches='tight',facecolor='white');plt.close(fig)
        results.append({'subject':subject,'phase_id':row['phase_id'],'phase_percent_within_cycle':row['phase_percent_within_cycle'],
            'cycle_index':row['cycle_index'],'reference_phase_id':row['reference_phase_id'],
            'reference_phase_percent':reference['phase_percent_within_cycle'], 'volume_ml':row['moments']['volume_ml'],
            'figure':filename,'figure_sha256':digest(output/filename),'planes':checks,
            'target_touches_CT_boundary':row['provenance']['touches_source_image_boundary'],
            'CT_sha256':row['ct_sha256'],'target_mask_sha256':provenance['sha256'],'reference_mask_sha256':ref_provenance['sha256'],
            'saved_candidate_parameters_used_without_refit':True})
        print(subject,filename,flush=True)
        del data,target,ref,ct,ref_ct
    receipt={'schema':'heart_CT_overlay_v1','status':'rendered_not_anatomically_validated',
        'baseline_sha256':digest(record_path),'code_sha256':digest(__file__),
        'selection_rule':'For each subject, longest original cycle; nonreference phase with greatest absolute relative whole-heart volume difference from its own reference. Ties: first cycle/record.',
        'coordinate_frame':'Original CT NIfTI RAS+ in mm; exact saved candidate transforms; no CT-to-static-FEM registration',
        'display':{'HU_window':[-200,300],'CT_interpolation':'trilinear','mask_interpolation':'nearest voxel, half-open cells',
            'plane_definition':'World axial/coronal/sagittal planes through target heart volume centroid',
            'orientations':'R/L right/left; A/P anterior/posterior; S/I superior/inferior',
            'outside_CT':'muted violet gray, never filled with invented anatomy'},
        'examples':results,'raw_inputs_modified':False,'FEM_run':False}
    (output/'manifest.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--baseline',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();run(args.config,args.baseline,args.output)
