"""Prepare model-derived prompt candidates; they are not blinded manual labels."""
import argparse,json
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cardiac_function_analysis import verify_grid
from georg_ventricular_mask_pilot import PHASES,CHAMBERS
from pilot_blood_segmentation_candidates import sha256_file


def pick_points(region,spacing,count=2,min_separation_mm=22):
    coords=np.column_stack(np.nonzero(region))
    if not len(coords):return []
    lo=coords.min(0);hi=coords.max(0)+1;sl=tuple(slice(a,b) for a,b in zip(lo,hi))
    depth=ndimage.distance_transform_edt(np.pad(region[sl],1),sampling=spacing)[1:-1,1:-1,1:-1]
    scores=depth[tuple((coords-lo).T)];points=[]
    for _ in range(count):
        if not np.any(scores>0):break
        idx=int(np.argmax(scores));point=coords[idx];points.append(point.tolist())
        scores[np.linalg.norm((coords-point)*spacing,axis=1)<min_separation_mm]=0
    return points


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--derived-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True);subject=a.derived_root/'georg';items=[]
    for pid in PHASES:
        ctpath=subject/'phases'/f'{pid}.nii.gz';img=nib.load(str(ctpath));ct=np.asarray(img.dataobj,dtype=np.float32);spacing=np.linalg.norm(img.affine[:3,:3],axis=0)
        names=['heart_ventricle_left','heart_ventricle_right','heart_atrium_left','heart_atrium_right','heart_myocardium','aorta','pulmonary_artery']
        masks={};hashes={}
        for name in names:
            path=subject/'automatic_masks_v2'/pid/'heartchambers_highres'/f'{name}.nii.gz';m=nib.load(str(path))
            if m.shape!=img.shape or not np.allclose(m.affine,img.affine):raise ValueError('Grid mismatch')
            masks[name]=np.asarray(m.dataobj)>0;hashes[name]=sha256_file(path)
        for ch in CHAMBERS:
            raw=masks[ch];positive_region=ndimage.binary_erosion(raw,iterations=5)
            median=np.median(ct[positive_region]);positive_region &= ct>=median
            # HU selects a bright seed inside the existing cavity; it is not a segmentation threshold.
            positive=pick_points(positive_region,spacing,1)
            negatives=[];excluded=ndimage.binary_dilation(raw,iterations=5)
            structures=['heart_myocardium','aorta','heart_atrium_left'] if ch.endswith('left') else ['pulmonary_artery','heart_atrium_right','heart_myocardium']
            for structure in structures:
                region=ndimage.binary_erosion(masks[structure],iterations=4) & ~excluded
                for point in pick_points(region,spacing,3 if structure=='heart_myocardium' else 1):
                    negatives.append({'point_native_ijk':point,'source_structure':structure,'HU':float(ct[tuple(point)])})
            allpoints=[{'point_native_ijk':point,'source_structure':'positive_cavity','HU':float(ct[tuple(point)])} for point in positive]+negatives
            item={'phase_id':pid,'chamber':ch,'ct_sha256':sha256_file(ctpath),'source_mask_sha256':hashes,
                  'positive_points':positive,'negative_points':[x['point_native_ijk'] for x in negatives],'point_details':allpoints,
                  'point_origin':'derived_from_TS_geometry_and_CT_interior_intensity; proposals_for_assisted_review_not_manual_reference',
                  'anatomical_planes_accepted':False}
            # A slice at each prompt; canonical RAS radiological display with the marker.
            can=nib.as_closest_canonical(img);data=np.asarray(can.dataobj);transform=np.linalg.inv(can.affine)@img.affine
            rawcan=np.asarray(nib.as_closest_canonical(nib.Nifti1Image(raw.astype(np.uint8),img.affine)).dataobj)>0
            n=len(allpoints);fig,axes=plt.subplots(2,int(np.ceil(n/2)),figsize=(16,8),squeeze=False)
            xyz=np.column_stack(np.nonzero(rawcan));lo=np.maximum(xyz.min(0)-60,0);hi=np.minimum(xyz.max(0)+61,data.shape)
            for ax,point in zip(axes.flat,allpoints):
                q=np.rint((transform@np.r_[point['point_native_ijk'],1])[:3]).astype(int);z=q[2]
                lx=min(lo[0],max(q[0]-20,0));hx=max(hi[0],min(q[0]+21,data.shape[0]));ly=min(lo[1],max(q[1]-20,0));hy=max(hi[1],min(q[1]+21,data.shape[1]))
                crop=data[lx:hx,ly:hy,z].T
                ax.imshow(crop,origin='lower',cmap='gray',vmin=-25,vmax=225,extent=[lx-.5,hx-.5,ly-.5,hy-.5]);ax.invert_xaxis()
                contour=rawcan[lx:hx,ly:hy,z].T
                if contour.any():ax.contour(np.arange(lx,hx),np.arange(ly,hy),contour,levels=[.5],colors=['orange'],linewidths=.7)
                color='lime' if point['source_structure']=='positive_cavity' else 'red';ax.plot(q[0],q[1],'+',color=color,ms=14,mew=2)
                ax.set_title(f"{point['source_structure']} | {point['HU']:.0f} HU\nRAS {q.tolist()}",fontsize=9);ax.set_aspect('equal');ax.tick_params(labelsize=7)
            for ax in axes.flat[n:]:ax.axis('off')
            fig.suptitle(f'Georg {pid} {ch} | candidate prompts: green include, red exclude');fig.tight_layout();fig.savefig(a.output_dir/f'{pid}_{ch}_prompts.png',dpi=130);plt.close(fig)
            items.append(item);print('Prepared prompts',pid,ch,flush=True)
    result={'schema_version':1,'method':'TS_initial_mask_with_sparse_structure_derived_points','status':'exploratory_hypothesis_not_validated','independent_reference':False,'selection_rule':'1 positive point in bright eroded cavity; negatives in eroded myocardium and adjacent structure masks at least 5 native dilation steps away from raw cavity; farthest-inside candidates separated by 22 mm','items':items,'code_sha256':sha256_file(Path(__file__))}
    (a.output_dir/'prompt_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
