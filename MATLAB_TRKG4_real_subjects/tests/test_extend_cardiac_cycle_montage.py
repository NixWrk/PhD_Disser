from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import extend_cardiac_cycle_montage as ext


def test_changed_bound_file_is_rejected(tmp_path):
    p=tmp_path/'contact.mat';p.write_bytes(b'original')
    r=ext.record(p);ext.verify([r])
    p.write_bytes(b'modified')
    with pytest.raises(ValueError,match='Changed sealed input'):ext.verify([r])


def test_result_from_another_phase_is_rejected_even_with_valid_receipt(tmp_path):
    ext.core.write(tmp_path/'plan.json',{})
    target=tmp_path/'phase_01';target.mkdir()
    ext.core.write(target/'result.json',dict(state_id='phase_00',montage='ttrkg',blocks_sha256='b',contact_sha256='c'))
    ext.core.write(target/'completion.json',dict(plan_sha256=ext.core.sha(tmp_path/'plan.json'),hashes={'result.json':ext.core.sha(target/'result.json')}))
    with pytest.raises(ValueError,match='Wrong result identity'):
        ext.completed(tmp_path,dict(montage='ttrkg',contact={'sha256':'c'}),dict(id='phase_01',blocks_sha256='b'))
