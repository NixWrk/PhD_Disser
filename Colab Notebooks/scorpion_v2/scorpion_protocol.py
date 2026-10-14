"""Dated protocol overlay; signal caches and candidate annotations are immutable.

The author confirmed late-block 1x1 and complete cross-record arm-wire removal
on 2026-10-14. Original filename labels remain historical source metadata.
"""
from scorpion_qc import channel_assignment as historical_assignment

REVISION = '2026-10-14'
LATE_VARIANT_SCENARIOS = ('1x1',)
HISTORICAL_LATE_VARIANT_SCENARIOS = ('1x4',)


def channel_assignment(record_id):
    number = int(record_id.rsplit('-', 1)[-1])
    result = historical_assignment(record_id)
    if 12 <= number <= 18:
        result = dict(physical_channel_roles={'1': 'ttrkg',
            '2': 'tepc' if number <= 17 else 'aepc'},
            status='author_confirmed_2026-10-12',
            source='direct author clarification 2026-10-12',
            ecg_quality_is_supporting_evidence_only=True)
    result['ttrkg'] = ttrkg_configuration(record_id)
    if number in (6, 11):
        result.update(arm_wires_removed=True,
            arm_wires_status='author_confirmed_2026-10-14',
            arm_wires_source='direct author clarification 2026-10-14',
            remaining_arm_pad_presence_confirmed=False)
    return result


def ttrkg_configuration(record_id, late_variant='1x1', *, historical_diagnostic=False):
    if late_variant not in LATE_VARIANT_SCENARIOS + HISTORICAL_LATE_VARIANT_SCENARIOS:
        raise ValueError('Unknown late-block montage')
    number = int(record_id.rsplit('-', 1)[-1])
    if 7 <= number <= 10 and late_variant == '1x4' and not historical_diagnostic:
        raise ValueError('Late-block 1x4 requires explicit historical_diagnostic=True; current protocol is confirmed 1x1')
    if number in (6, 11):
        return None
    if number in (1, 2) or 12 <= number <= 18:
        variant, status = '1x1', 'author_confirmed'
    elif number in (3, 4):
        variant, status = ('2x2' if number == 3 else '1x4'), 'author_confirmed'
    elif number == 5:
        variant, status = '1x4', 'author_protocol_recollection_tentative'
    elif 7 <= number <= 10:
        variant = late_variant
        status = 'author_confirmed' if variant == '1x1' else 'historical_diagnostic_not_current_protocol'
    else:
        raise ValueError('Unknown record')
    result = dict(variant=variant, status=status, source='author clarification 2026-10-12',
        single_contact_angle_author_confirmed=True,
        physical_positions_unchanged=True,
        probe_frequency_hz=100000,
        current_uA_author_estimate=7000, current_amplitude_is_approximate=True,
        current_applies_to_each_channel_author_confirmed=True,
        current_amplitude_convention='unknown',
        excitation_scenario='own_drive',
        excitation_source='author: same as Nix; Nix main FEM own-source transfer',
        physical_multiplexing_or_relative_phase_measured=False)
    if 7 <= number <= 10:
        result.update(source=('direct author clarification 2026-10-14' if variant == '1x1'
            else 'historical filename label 1x4; superseded by author clarification 2026-10-14'),
            confirmation_date='2026-10-14', confirmed_current_variant='1x1',
            accepted_current_protocol=variant == '1x1',
            historical_diagnostic_only=variant == '1x4',
            original_filename_1x4_label_is_historical=True)
    return result


def arm_groups(contacts, configuration):
    if configuration is None:
        return []
    variant = configuration['variant']
    if variant == '1x1':
        return [group[1:2] if i in (1, 2) else group[:1]
            for i, group in enumerate(contacts['ttrkg_1x4'])]
    return contacts['ttrkg_' + variant]
