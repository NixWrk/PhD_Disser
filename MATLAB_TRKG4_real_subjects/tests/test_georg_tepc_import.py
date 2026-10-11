"""Guard reconstruction identity and preserve author-selected coordinates."""
from pathlib import Path
import copy
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from import_georg_tepc_landmarks import normalise
from build_surface_landmark_picker import landmark_payload_template


@pytest.fixture
def inputs():
    payload = landmark_payload_template('body_embedded_arms_370mm.stl')
    payload['preliminary_electrode_centers_xyz_mm'] = dict(
        I_plus=[-120.,-70.,-40.], V_plus=[-110.,-90.,-60.],
        V_minus=[100.,-120.,-250.], I_minus=[120.,-140.,-300.])
    expected = dict(body_sha256='body', ct_sha256='ct', tissue_mask_sha256={'fat':'fat'})
    payload['measurement_context'] = dict(subject_pseudonym='Georg', experiment='scorpion_v2_georg',
        physical_coordinate_frame='LPS', units='mm', electrode_configuration_family='TEPC',
        bei_channel=None, recording=dict(id='012', montage_id='georg_tepc_2'), **expected)
    return payload, expected


def test_filename_mapping_requires_confirmation(inputs):
    payload, hashes = inputs
    with pytest.raises(ValueError, match='confirmation'):
        normalise(payload, 'georg_tepc_7.json', hashes)


@pytest.mark.parametrize('number', range(2,8))
def test_confirmed_mapping_keeps_raw_points_and_unknown_channel(inputs, number):
    payload, hashes = inputs
    before = copy.deepcopy(payload)
    result = normalise(payload, f'georg_tepc_{number}.json', hashes, filename_mapping_confirmed=True)
    assert payload == before
    assert result['measurement_context']['recording']['id'] == f'{number+10:03d}'
    assert result['preliminary_electrode_centers_xyz_mm'] == before['preliminary_electrode_centers_xyz_mm']
    assert result['measurement_context']['bei_channel'] is None
    assert result['import_provenance']['original_recording'] == before['measurement_context']['recording']
    assert not result['import_provenance']['anatomical_accuracy_validated']


@pytest.mark.parametrize('bad_field,bad_value', [('body_sha256','other-body'),('physical_coordinate_frame','RAS')])
def test_different_surface_or_frame_is_rejected(inputs, bad_field, bad_value):
    payload, hashes = inputs
    payload['measurement_context'][bad_field] = bad_value
    with pytest.raises(ValueError):
        normalise(payload, 'georg_tepc_2.json', hashes, filename_mapping_confirmed=True)
