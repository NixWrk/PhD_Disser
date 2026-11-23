"""Analyze the 19 November posterior electrode recordings directly from the supplied ZIP."""
from __future__ import annotations

import argparse
import csv
import hashlib
from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np


ARCHIVE_SHA256 = "5e51f6bf4522894cafd09d0a99336ca293b68d184c2ab0a001ac2088136c1696"
SUBJECTS = {"ник": "U1", "георг": "U2"}
SIZES = (60, 100, 140, 200)
HEADER = (
    "COUNT", "EMG1mV", "EMG2mV", "BEI1ohm_100kHz", "BEI2ohm_100kHz",
    "BPH1grad_100kHz", "BPH2grad_100kHz", "CE1ohm_100kHz", "CE2ohm_100kHz",
    "CPH1grad_100kHz", "CPH2grad_100kHz",
)


def identify(member):
    path = PurePosixPath(member)
    participant = SUBJECTS[path.parent.name.casefold()]
    text = path.stem.casefold().replace("ё", "е")
    posture, *labels = text.split()
    position = {"сидя": "seated", "лежа": "lying"}[posture]
    sizes = [int(word) for word in labels if word.isdecimal()]
    if len(sizes) != 1 or sizes[0] not in SIZES:
        raise ValueError(f"Unexpected size in {member}")
    terms = set(labels) - {str(sizes[0]), "эксперимент"}
    if terms == {"ттркг"}:
        pair = "back_arm"
    elif terms in ({"оба"}, {"два"}):
        pair = "bilateral"
    else:
        raise ValueError(f"Unknown lead pairing in {member}")
    return participant, position, pair, sizes[0]


def expected_records():
    rows = {(u, p, "back_arm", size) for u in SUBJECTS.values()
            for p in ("seated", "lying") for size in SIZES}
    rows.update((u, "seated", "bilateral", size) for u in SUBJECTS.values() for size in SIZES)
    rows.add(("U2", "lying", "bilateral", 100))
    return rows


def iter_exports(archive):
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError("Archive differs from the documented 19 November source.")
    with ZipFile(archive) as z:
        exports = {}
        for name in z.namelist():
            if PurePosixPath(name).suffix.lower() != ".txt":
                continue
            key = identify(name)
            if key in exports:
                raise ValueError(f"Duplicate condition {key}")
            exports[key] = name
        if set(exports) != expected_records():
            raise ValueError("The recordings do not cover the documented conditions.")
        for key, name in sorted(exports.items()):
            raw = z.read(name)
            header, _, body = raw.partition(b"\n")
            if tuple(header.decode("utf-8-sig").split()) != HEADER:
                raise ValueError(f"Unexpected field order or units in {key}")
            data = np.loadtxt(BytesIO(body.replace(b",", b".")), ndmin=2)
            if data.shape[1] != len(HEADER) or not np.isfinite(data).all():
                raise ValueError(f"Missing fields or non-finite samples in {key}")
            counter = data[:, 0]
            if not np.equal(counter, counter.astype(int)).all() or counter.min() < 0 or counter.max() > 255:
                raise ValueError(f"Invalid device counter in {key}")
            if np.any(np.diff(counter) % 256 != 1):
                raise ValueError(f"Discontinuous count in {key}. Split into contiguous runs before filtering.")
            yield key, data


def record_metrics(key, data):
    u, position, pair, size = key
    row = dict(participant=u, posture=position, pair=pair, size_mm=size, samples=len(data))
    even_sign = 1 - 2 * (data[1:, 0] % 2)
    for i in (1, 2):
        z = data[:, 2 + i]
        q1, median, q3 = np.quantile(z, [.25, .5, .75])
        paired_z = z[:len(z) // 2 * 2].reshape(-1, 2).mean(axis=1)
        row.update({
            f"ch{i}_median_ohm": median,
            f"ch{i}_q1_ohm": q1, f"ch{i}_q3_ohm": q3,
            f"ch{i}_even_minus_odd_ohm": np.median(np.diff(z) * even_sign),
            f"ch{i}_median_of_adjacent_means_ohm": np.median(paired_z),
        })
    # These samples are genuinely simultaneous. For two postures only file-level summaries are paired.
    row["synchronous_ch1_minus_ch2_median_ohm"] = np.median(data[:, 3] - data[:, 4])
    return row


def contrasts(records):
    rows = []
    for u in sorted(SUBJECTS.values()):
        for size in SIZES:
            mixed_s = records[u, "seated", "back_arm", size]
            mixed_l = records[u, "lying", "back_arm", size]
            both_s = records[u, "seated", "bilateral", size]
            both_l = records.get((u, "lying", "bilateral", size))
            row = dict(participant=u, size_mm=size)
            for i in (1, 2):
                field = f"ch{i}_median_ohm"
                row[f"mixed_seated_minus_lying_ch{i}_ohm"] = mixed_s[field] - mixed_l[field]
                row[f"bilateral_seated_minus_lying_ch{i}_ohm"] = (
                    both_s[field] - both_l[field] if both_l else None)
                row[f"seated_mixed_minus_bilateral_ch{i}_ohm"] = mixed_s[field] - both_s[field]
            row["bilateral_seated_synchronous_ch1_minus_ch2_ohm"] = both_s["synchronous_ch1_minus_ch2_median_ohm"]
            row["bilateral_lying_synchronous_ch1_minus_ch2_ohm"] = (
                both_l["synchronous_ch1_minus_ch2_median_ohm"] if both_l else None)
            rows.append(row)
    return rows


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: round(float(value), 6) if isinstance(value, (float, np.floating)) else value
                             for key, value in row.items()})


def plot_levels(path, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.0), sharex=True, sharey=True, layout="constrained")
    for col, u in enumerate(sorted(SUBJECTS.values())):
        top, bottom = axes[:, col]
        for ax, pair, lines in [
            (top, "back_arm", (("seated", 1, "Сидя"), ("lying", 1, "Лёжа"))),
            (bottom, "bilateral", (("seated", 1, "Канал 1"), ("seated", 2, "Канал 2"))),
        ]:
            for (position, i, label), color, marker in zip(lines, ("#126B77", "#A94E25"), ("o", "s")):
                y = [records[u, position, pair, size][f"ch{i}_median_ohm"] for size in SIZES]
                ax.plot(SIZES, y, color=color, marker=marker, lw=2, label=label)
            ax.set_xticks(SIZES)
            ax.yaxis.set_major_locator(MultipleLocator(20))
            ax.grid(axis="y", color="#D9E0E0", lw=.7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False)
        top.set_title(f"У{u[-1]}  ·  Серия с ТТРКГ, канал 1")
        bottom.set_title(f"У{u[-1]}  ·  Два отведения со спины сидя")
        bottom.set_xlabel("Размер задней сборки, мм")
    for ax in axes[:, 0]:
        ax.set_ylabel("Медиана импеданса, Ом")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = {key: record_metrics(key, data) for key, data in iter_exports(args.archive)}
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "levels_20261119.csv", list(records.values()))
    write_csv(args.output / "contrasts_20261119.csv", contrasts(records))
    plot_levels(args.output / "levels_20261119.png", records)
    print(f"{len(records)} exports, {sum(r['samples'] for r in records.values())} rows, 8 size-matched comparisons")


if __name__ == "__main__":
    main()
