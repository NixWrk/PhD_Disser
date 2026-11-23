"""Verify the 25 TXT exports against their known intervals in the long BIN.

Run: python -B verify_exports_20261119.py --archive /data/Scorp_19_11_26.zip
--archive also accepts the directory containing that ZIP. Requires NumPy and
the tracked ../../scorpion_v2/inspect_scorpion_bin.py (COBS decoder). No local
audit caches are required. Reads the ZIP directly; writes no files, even caches.

CSV indices are zero-based counts of complete 0x10 packets in this BIN, with
exclusive ends. Byte offsets address the first encoded byte of a packet in the
uncompressed BIN, not in the ZIP. The initial 129-byte fragment is not a sample.
The BIN hash and all packet boundaries are checked; only the 25 known windows
and control packets are fully decoded. Memory is bounded by the 230 MB BIN plus
one export/window. Original timestamps are ZIP modification times, not device
timestamps. A 1000-Hz nominal rate is supported by settings and the timestamp
cross-check; the unit of u16@2 and oscillator accuracy remain undocumented.

Condition labels only locate exports (U1/U2 and back_arm/bilateral); they do not
assign anatomical sides to channels 1/2. Bilateral U2 lying/100 and breathing
order are author information, not decoded metadata. No separate REO field is
present. CRC of the ZIP entries is checked by zipfile; the last two bytes of
device packets are not verified as a checksum. COUNT continuity is modulo 256.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import sys
from zipfile import ZipFile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scorpion_v2"))
from inspect_scorpion_bin import cobs_decode, signed24  # noqa: E402
import numpy as np  # noqa: E402

ARCHIVE_SHA256 = "5e51f6bf4522894cafd09d0a99336ca293b68d184c2ab0a001ac2088136c1696"
BIN_SHA256 = "4bb288b9050d3cc3590fca4decb850721c9b087bd5832a165d8243376091bc44"
BIN_NAME = "2026-11-19 10.48.13.bin"
SETTINGS_HEX = "1103e803c201310032002602010100e1f505581baac0"
HEADER = ("COUNT", "EMG1mV", "EMG2mV", "BEI1ohm_100kHz", "BEI2ohm_100kHz",
          "BPH1grad_100kHz", "BPH2grad_100kHz", "CE1ohm_100kHz", "CE2ohm_100kHz",
          "CPH1grad_100kHz", "CPH2grad_100kHz")
KEY = ("participant", "posture", "pair", "size_mm")
TEXT_FIELDS = {"participant", "posture", "pair", "settings_hex"}
# Half the printed decimal step; only 1e-9 is allowed for floating-point error.
TOLERANCE = np.array([0, .005, .005] + [.0000005] * 8)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def condition_key(row):
    return tuple(row[k] for k in KEY)


def identify(member):
    """Source names label conditions only; sample equality proves the BIN link."""
    path = PurePosixPath(member)
    participant = {"ник": "U1", "георг": "U2"}[path.parent.name.casefold()]
    words = path.stem.casefold().replace("ё", "е").split()
    posture = {"сидя": "seated", "лежа": "lying"}[words[0]]
    sizes = [int(w) for w in words if w.isdecimal()]
    terms = set(words[1:]) - set(map(str, sizes)) - {"эксперимент"}
    pair = "back_arm" if terms == {"ттркг"} else "bilateral"
    require(len(sizes) == 1 and sizes[0] in (60, 100, 140, 200)
            and terms in ({"ттркг"}, {"два"}, {"оба"}), "Unexpected export condition")
    return participant, posture, pair, sizes[0]


def read_intervals(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = [{k: v if k in TEXT_FIELDS else int(v) for k, v in r.items()}
                for r in csv.DictReader(stream)]
    rows.sort(key=lambda r: r["start_data_index"])
    require(len(rows) == len({condition_key(r) for r in rows}) == 25,
            "CSV must contain exactly 25 distinct conditions")
    previous_end = 0
    for r in rows:
        require(r["start_data_index"] >= previous_end
                and r["end_data_index_exclusive"] - r["start_data_index"] == r["samples"] > 0,
                "Invalid or overlapping sample interval")
        require(r["first_byte_offset"] < r["last_packet_byte_offset"], "Invalid byte interval")
        require(r["settings_hex"] == SETTINGS_HEX and r["settings_u16_at_2"] == 1000
                and r["frequency_blocks"] == 1 and r["nominal_fs_hz"] == 1000,
                "Unexpected settings in interval CSV")
        previous_end = r["end_data_index_exclusive"]
    return rows


def unsigned(a, offset, width):
    value = a[:, offset].astype(np.int64)
    for k in range(1, width):
        value |= a[:, offset + k].astype(np.int64) << (8 * k)
    return value


def signed(a, offset, width):
    value = unsigned(a, offset, width)
    return np.where(value >= 1 << (8 * width - 1), value - (1 << (8 * width)), value)


def compare_export(packets, raw_text, row):
    header, _, body = raw_text.partition(b"\n")
    require(tuple(header.decode("ascii").split()) == HEADER, "TXT header/units changed")
    text = np.loadtxt(BytesIO(body.replace(b",", b".")), ndmin=2)
    require(text.shape == (row["samples"], 11) and np.isfinite(text).all(),
            "TXT shape or finite-value check failed")
    a = np.frombuffer(packets, dtype=np.uint8).reshape(-1, 37)
    require(len(a) == len(text) and np.all(a[:, 0] == 0x10) and np.all(a[:, 8] == 1)
            and not np.any(a[:, 9:13]), "Unexpected measurement packet layout")
    # Little-endian fields of a one-block decoded packet; fixed scales, no fit.
    values = np.column_stack((a[:, 1], signed(a, 2, 3) * .001, signed(a, 5, 3) * .001,
        unsigned(a, 13, 4) * .00001, unsigned(a, 17, 4) * .00001,
        signed(a, 21, 2) / 360, signed(a, 23, 2) / 360,
        unsigned(a, 25, 3) * .001, unsigned(a, 28, 3) * .001,
        signed(a, 31, 2) / 360, signed(a, 33, 2) / 360))
    # Check the vectorized s24 conversion against the shared scalar decoder.
    require(values[0, 1] == signed24(bytes(a[0, 2:5])) * .001,
            "Signed24 conversion disagrees with shared reader")
    require(np.array_equal(values[:, 0], text[:, 0])
            and np.all(np.diff(text[:, 0]) % 256 == 1)
            and text[0, 0] == row["first_count"] and text[-1, 0] == row["last_count"],
            "COUNT mismatch or discontinuity")
    errors = np.max(np.abs(values - text), axis=0)
    require(np.all(errors <= TOLERANCE + 1e-9),
            f"Export differs from BIN beyond rounding: {condition_key(row)}")
    return errors


def verify(archive, rows):
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    require(digest.hexdigest() == ARCHIVE_SHA256, "ZIP differs from the audited source")
    with ZipFile(archive) as z:
        bins = [i for i in z.infolist() if PurePosixPath(i.filename).name == BIN_NAME]
        require(len(bins) == 1, "Expected exactly one long BIN")
        exports = {}
        for info in z.infolist():
            if info.filename.lower().endswith(".txt"):
                key = identify(info.filename)
                require(key not in exports, "Duplicate TXT condition")
                exports[key] = info
        require(set(exports) == {condition_key(r) for r in rows}, "CSV/TXT coverage mismatch")
        source = z.read(bins[0])
        require(hashlib.sha256(source).hexdigest() == BIN_SHA256, "Long BIN hash mismatch")
        require(source.endswith(b"\0"), "Unterminated final packet")
        start = 0
        data_index = -1
        current = 0
        control = None
        packets = bytearray()
        frame_sizes = Counter()
        controls = Counter()
        max_error = np.zeros(11)
        for delimiter in re.finditer(b"\0", source):
            end = delimiter.start()
            if start == 0:
                require(end + 1 == 129, "Unexpected initial fragment length")
            else:
                require(end - start >= 2, "Empty/short packet")
                # Packet types are nonzero: COBS byte 1 holds the decoded type.
                kind = source[start + 1]
                if kind == 0x10:
                    data_index += 1
                    frame_sizes[end - start] += 1
                    if current < len(rows):
                        row = rows[current]
                        if data_index >= row["start_data_index"]:
                            require(control == row["settings_hex"], "Settings changed in export")
                            if data_index == row["start_data_index"]:
                                require(start == row["first_byte_offset"], "Start index/offset mismatch")
                            packet = cobs_decode(source[start:end])
                            require(len(packet) == 37, "Expected one-block packet")
                            packets.extend(packet)
                            if data_index + 1 == row["end_data_index_exclusive"]:
                                require(start == row["last_packet_byte_offset"], "End index/offset mismatch")
                                error = compare_export(packets, z.read(exports[condition_key(row)]), row)
                                max_error = np.maximum(max_error, error)
                                packets.clear()
                                current += 1
                elif kind == 0x11:
                    packet = cobs_decode(source[start:end])
                    control = packet.hex()
                    controls[control] += 1
                else:
                    require(kind in (0x01, 0x2f), "Unexpected service packet type")
            start = end + 1
        require(current == 25 and data_index + 1 == 5883965 and not packets,
                "Incomplete interval verification or changed data-packet count")
        require(frame_sizes == {192: 2275, 38: 5881690}
                and controls[SETTINGS_HEX] == 5882 and sum(controls.values()) == 5904,
                "Unexpected packet/control counts")
        first, last = rows[0], rows[-1]
        seconds = (datetime(*exports[condition_key(last)].date_time)
                   - datetime(*exports[condition_key(first)].date_time)).total_seconds()
        samples = last["end_data_index_exclusive"] - first["end_data_index_exclusive"]
        require(samples == 5200705 and seconds == 5202, "Timestamp cross-check changed")
    return {"status": "PASS", "exports": current, "matched_rows": sum(r["samples"] for r in rows),
            "columns_verified": 11, "maximum_residual_by_column": dict(zip(HEADER, max_error.tolist())),
            "nominal_fs_hz": 1000, "nominal_dt_s": .001,
            "settings_u16_at_2": 1000, "zip_mtime_delta_s": seconds,
            "end_sample_delta": samples, "mtime_implied_fs_hz": samples / seconds,
            "rate_status": "file_supported_nominal; field_unit_and_clock_accuracy_undocumented",
            "device_packet_checksum": "unverified", "right_side_channel": "not_established"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, required=True, help="ZIP file or its containing directory")
    parser.add_argument("--intervals", type=Path,
                        default=Path(__file__).with_name("bin_intervals_20261119.csv"))
    args = parser.parse_args()
    archive = args.archive / "Scorp_19_11_26.zip" if args.archive.is_dir() else args.archive
    print(json.dumps(verify(archive, read_intervals(args.intervals)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
