"""A continuation must retain the exact executed worker, not today's source."""
import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'MATLAB_TRKG4_real_subjects/tools'))
extension=importlib.import_module('georg_quality_search_extension')


def fixture_run(tmp_path):
    parent=tmp_path/'parent';inputs=parent/'inputs';run=inputs/'joint_previous'
    run.mkdir(parents=True)
    source=parent/'source_snapshots/src/run_georg_lateral_inverse.m'
    source.parent.mkdir(parents=True);source.write_bytes(b'% executed worker\r\n')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    (run/'plan.local.json').write_text(json.dumps({'identity':{'worker_sha256':digest}}))
    return parent,inputs,source,digest


def test_archive_uses_verified_executed_bytes(tmp_path):
    parent,inputs,source,digest=fixture_run(tmp_path)
    target=extension.preserve_resume_source(parent,inputs,'joint_previous')
    assert target.name=='run_georg_lateral_inverse_'+digest+'.m'
    assert target.read_bytes()==source.read_bytes()
    assert extension.preserve_resume_source(parent,inputs,'joint_previous')==target


def test_changed_existing_archive_is_not_overwritten(tmp_path):
    parent,inputs,source,digest=fixture_run(tmp_path)
    target=inputs/'source_snapshots'/('run_georg_lateral_inverse_'+digest+'.m')
    target.parent.mkdir();target.write_bytes(b'wrong source')
    with pytest.raises(ValueError,match='archive'):
        extension.preserve_resume_source(parent,inputs,'joint_previous')
    assert target.read_bytes()==b'wrong source'


def test_missing_original_bytes_blocks_preparation(tmp_path):
    parent,inputs,source,_=fixture_run(tmp_path)
    source.write_bytes(b'changed source')
    with pytest.raises(ValueError,match='executed worker'):
        extension.preserve_resume_source(parent,inputs,'joint_previous')
    assert not (inputs/'source_snapshots').exists()
