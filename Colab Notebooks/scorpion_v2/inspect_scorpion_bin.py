"""Read-only, exploratory inspection of Scorpion BIN packet streams.

Field names deliberately describe byte positions. A value resembling a rate
does not establish its unit or meaning without the device protocol.
"""
from __future__ import annotations

import argparse
from array import array
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


def cobs_decode(encoded: bytes) -> bytes:
    output = bytearray()
    position = 0
    while position < len(encoded):
        code = encoded[position]
        if code == 0 or position + code > len(encoded):
            raise ValueError('invalid COBS block length')
        output.extend(encoded[position + 1:position + code])
        position += code
        if code < 255 and position < len(encoded):
            output.append(0)
    return bytes(output)


def signed24(data: bytes) -> int:
    value = int.from_bytes(data, 'little')
    return value - (1 << 24) if value & (1 << 23) else value


def control_fields(packet: bytes) -> dict:
    result = {'decoded_hex': packet.hex(' '), 'decoded_length': len(packet)}
    if len(packet) in (22, 64):
        result.update({
            'byte_1': packet[1],
            'u16_at_2': struct.unpack_from('<H', packet, 2)[0],
            'u16_at_4': struct.unpack_from('<H', packet, 4)[0],
            'u16_at_6': struct.unpack_from('<H', packet, 6)[0],
            'u16_at_8': struct.unpack_from('<H', packet, 8)[0],
            'u16_at_10': struct.unpack_from('<H', packet, 10)[0],
            'byte_12': packet[12], 'byte_13': packet[13],
            'groups_from_14_u32_u16': [
                list(struct.unpack_from('<IH', packet, i))
                for i in range(14, len(packet) - 2, 6)
            ],
            'semantics': 'unconfirmed_byte_fields_not_named_as_sampling_rate',
        })
    return result


def inspect_file(path: Path, file_id: str, out: Path) -> dict:
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    counts = Counter()
    controls = {}
    transitions = []
    other_examples = {}
    failures = []
    sample_count = array('B')
    ecg1, ecg2 = array('i'), array('i')
    sample_byte_offsets = array('Q')
    frequency_count = array('B')
    settings_state = array('h')
    state = -1
    offset = 0
    frame_count = 0
    parts = source.split(b'\0')
    trailing_bytes = len(parts[-1])
    for encoded in parts[:-1]:
        start = offset
        offset += len(encoded) + 1
        frame_count += 1
        if not encoded:
            counts['empty_delimited_block'] += 1
            continue
        try:
            packet = cobs_decode(encoded)
        except ValueError:
            failures.append({'byte_offset': start, 'encoded_bytes': len(encoded)})
            continue
        if not packet:
            counts['empty_decoded_packet'] += 1
            continue
        kind = packet[0]
        key = f'0x{kind:02x}_length_{len(packet)}'
        counts[key] += 1
        if kind == 0x11:
            if packet not in controls:
                controls[packet] = {'state_id': len(controls), **control_fields(packet),
                                    'first_byte_offset': start, 'occurrences': 0}
            control = controls[packet]
            control['occurrences'] += 1
            if state != control['state_id']:
                state = control['state_id']
                transitions.append({'byte_offset': start, 'next_data_index': len(ecg1),
                                    'state_id': state})
        elif kind == 0x10 and len(packet) >= 15 and len(packet) == 15 + 22 * packet[8]:
            sample_count.append(packet[1])
            ecg1.append(signed24(packet[2:5]))
            ecg2.append(signed24(packet[5:8]))
            frequency_count.append(packet[8])
            sample_byte_offsets.append(start)
            settings_state.append(state)
        elif key not in other_examples:
            other_examples[key] = {'byte_offset': start, 'decoded_hex': packet.hex(' '),
                                   'ascii': packet[1:-2].decode('ascii', errors='replace')}
    counter = np.asarray(sample_count)
    gaps = np.flatnonzero((np.diff(counter.astype(int)) % 256) != 1)
    np.savez_compressed(out / f'{file_id}_decoded.local.npz',
                        count=counter, ecg1_signed24=np.asarray(ecg1),
                        ecg2_signed24=np.asarray(ecg2),
                        frequency_count=np.asarray(frequency_count),
                        sample_byte_offsets=np.asarray(sample_byte_offsets),
                        settings_state=np.asarray(settings_state))
    return {'file_id': file_id, 'source_name': path.name, 'bytes': len(source),
            'sha256': digest, 'delimited_blocks': frame_count,
            'decoded_packet_counts': dict(counts), 'decode_failures': failures,
            'trailing_unterminated_bytes': trailing_bytes,
            'data_packets': len(ecg1), 'counter_nonunit_steps': len(gaps),
            'counter_nonunit_step_examples': gaps[:20].tolist(),
            'control_variants': list(controls.values()), 'control_transitions': transitions,
            'other_packet_examples': other_examples,
            'checksum_status': 'last_two_bytes_not_verified',
            'rate_status': 'control_fields_require_protocol_or_cross_validation'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    inventory_path = experiment / 'metadata/inventory.local.json'
    inventory = json.loads(inventory_path.read_text(encoding='utf-8-sig'))
    out = experiment / 'derived/scorpion_bin_review'
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for entry in inventory['files']:
        if not entry['raw_relative_path'].lower().endswith('.bin'):
            continue
        path = experiment / entry['raw_relative_path']
        if path.stat().st_size == 0:
            record = {'file_id': entry['record_id'], 'source_name': path.name,
                      'bytes': 0, 'sha256': hashlib.sha256(b'').hexdigest(), 'status': 'empty'}
        else:
            record = inspect_file(path, entry['record_id'], out)
        if record['sha256'] != entry['sha256']:
            raise ValueError(f'Raw input hash differs from inventory: {path.name}')
        records.append(record)
        print(record['file_id'], record['bytes'], record.get('data_packets', 0),
              len(record.get('control_variants', [])), 'control variants', flush=True)
    result = {'status': 'exploratory_binary_structure_not_device_protocol',
              'generated_utc': datetime.now(timezone.utc).isoformat(),
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'source_archive_sha256': inventory['source_archive_sha256'],
              'inventory_sha256_at_read': hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
              'files': records}
    (out / 'binary_structure.local.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (out / 'inspection_source.local.py').write_bytes(Path(__file__).read_bytes())


if __name__ == '__main__':
    main()
