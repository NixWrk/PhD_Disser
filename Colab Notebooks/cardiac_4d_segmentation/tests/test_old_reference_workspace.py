"""Synthetic-only tests of independent annotation preparation."""
from pathlib import Path
import tempfile
import unittest
import sys
import json
import numpy as np
import nibabel as nib
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_old_reference_workspace import prepare, sha256, checked_ct


def fixture(root):
    ct_path = root/'phase_00.nii.gz'
    array = np.zeros((11,12,13), dtype=np.float32)
    array[3:8,4:9,4:10] = 400
    affine = np.array([[0,-2,0,20],[-3,0,0,30],[0,0,4,-40],[0,0,0,1]], float)
    image = nib.Nifti1Image(array,affine)
    image.header.set_xyzt_units('mm')
    nib.save(image,ct_path)
    entry = {'phase_id':'phase_00','phase_percent':0,'nifti':ct_path.name,'sha256':sha256(ct_path),
             'shape':list(array.shape),'affine_ras_mm':affine.tolist(),'intensity_units':'HU','slice_thickness_mm':4}
    manifest = root/'conversion_manifest.json'
    manifest.write_text(json.dumps({'case_id':'synthetic','sequence_id':'synthetic','phases':[entry]}))
    config = root/'config.json'
    config.write_text(json.dumps({'conversion_manifest':str(manifest),'output_dir':str(root/'workspace'),
        'phase_ids':['phase_00'],'selection_rule':'synthetic_only','guide_lower_hu':250}))
    return config, entry, array, image


class PreparationTest(unittest.TestCase):
    def test_independent_guide_grid_and_no_fake_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config,entry,array,image=fixture(root)
            out=prepare(config)
            p=out['phases'][0]
            guide=nib.load(p['guide_path'])
            np.testing.assert_array_equal(np.asanyarray(guide.dataobj),array>=250)
            np.testing.assert_allclose(guide.affine,image.affine)
            self.assertEqual(guide.header.get_xyzt_units()[0],'mm')
            self.assertFalse(out['reference_available'])
            self.assertFalse(out['automatic_comparator_masks_loaded'])
            self.assertTrue(all(v['reference_path'] is None and not v['accepted'] for v in p['chambers'].values()))
            self.assertEqual(sha256(root/'phase_00.nii.gz'),entry['sha256'])
            with self.assertRaises(FileExistsError): prepare(config)

    def test_hash_and_intensity_provenance_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,entry,_,_=fixture(root)
            bad=dict(entry,intensity_units='unknown')
            with self.assertRaisesRegex(ValueError,'HU'): checked_ct(root/'phase_00.nii.gz',bad)
            bad=dict(entry,sha256='wrong')
            with self.assertRaisesRegex(ValueError,'hash'): checked_ct(root/'phase_00.nii.gz',bad)


if __name__=='__main__': unittest.main()
