from io import StringIO
import unittest

import numpy as np
from inspect_scorpion_bin import cobs_decode, control_fields, signed24
from match_scorpion_bin import read_txt_ecg


class BinaryStructureTests(unittest.TestCase):
    def test_cobs_known_zero_and_nonzero_payloads(self):
        self.assertEqual(cobs_decode(bytes.fromhex('01 01')), b'\0')
        self.assertEqual(cobs_decode(bytes.fromhex('03 11 03 03 e8 03 01')), bytes.fromhex('11 03 00 e8 03 00'))
        self.assertEqual(cobs_decode(b'\xff' + b'X' * 254 + b'\x02Y'), b'X' * 254 + b'Y')

    def test_cobs_rejects_incomplete_and_zero_codes(self):
        for malformed in (b'\0', b'\x03A', b'\x02A\0'):
            with self.assertRaises(ValueError):
                cobs_decode(malformed)

    def test_signed_24_bit_limits(self):
        self.assertEqual(signed24(bytes.fromhex('ff ff 7f')), 8388607)
        self.assertEqual(signed24(bytes.fromhex('00 00 80')), -8388608)
        self.assertEqual(signed24(bytes.fromhex('ff ff ff')), -1)

    def test_control_offset_mapping_keeps_raw_units(self):
        packet = bytes.fromhex('11 03 c4 09 e8 03 31 00 32 00 14 05 01 01 00 e1 f5 05 58 1b 3b a1')
        fields = control_fields(packet)
        self.assertEqual(fields['u16_at_2'], 2500)
        self.assertEqual(fields['u16_at_10'], 1300)
        self.assertEqual(fields['groups_from_14_u32_u16'], [[100000000, 7000]])
        self.assertNotIn('sampling_rate_hz', fields)

    def test_txt_trailing_field_does_not_shift_counter_or_ecg(self):
        header = 'COUNT\tEMG1mV\tEMG2mV\tB1\tB2\tP1\tP2\tC1\tC2\tQ1\tQ2\n'
        row = '105\t1,72\t-0,10\t64\t75\t0\t0\t600\t600\t0\t0\t\n'
        np.testing.assert_array_equal(read_txt_ecg(StringIO(header + row)), [[105, 1.72, -.10]])


if __name__ == '__main__':
    unittest.main()
