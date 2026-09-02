#!/usr/bin/env python
"""Extract the colored ECG/gating strip and GE-highlighted scan window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def close_short_false_gaps(mask: np.ndarray, max_gap: int = 4) -> np.ndarray:
    """Join dense colored regions separated by a narrow grayscale trace."""
    closed = mask.copy()
    start = None
    for index, value in enumerate(mask):
        if not value and start is None:
            start = index
        if value and start is not None:
            if start > 0 and index - start <= max_gap:
                closed[start:index] = True
            start = None
    return closed


def highlighted_window_bounds(safe_strip: np.ndarray) -> tuple[int, int]:
    """Locate the wide colored interval highlighted by the GE report."""
    nonwhite = np.any(safe_strip < 245, axis=2)
    dense_columns = nonwhite.sum(axis=0) > 0.35 * safe_strip.shape[0]
    dense_columns = close_short_false_gaps(dense_columns)
    padded = np.pad(dense_columns.astype(np.int8), (1, 1))
    starts = np.flatnonzero(np.diff(padded) == 1)
    stops = np.flatnonzero(np.diff(padded) == -1)
    runs = [(int(start), int(stop)) for start, stop in zip(starts, stops)]
    candidates = [run for run in runs if run[1] - run[0] >= max(20, int(0.08 * safe_strip.shape[1]))]
    if not candidates:
        raise ValueError("No wide highlighted scan/reconstruction window detected")
    return max(candidates, key=lambda run: run[1] - run[0])


def parse_mapping(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected ALIAS=PATH")
    alias, raw_path = value.split("=", 1)
    return alias.strip().lower(), Path(raw_path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", required=True, type=parse_mapping)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "deidentified_colored_trace_only_not_calibrated_waveform", "items": []}
    for alias, path in args.image:
        image = mpimg.imread(path)[..., :3]
        if image.dtype.kind == "f":
            image = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
        rgb = image.astype(int)
        colored = np.logical_and(
            rgb.max(axis=2) - rgb.min(axis=2) > 35,
            np.min(rgb, axis=2) < 230,
        )
        coords = np.argwhere(colored)
        if not len(coords):
            raise ValueError(f"No colored ECG/gating strip detected: {path}")
        top = max(int(coords[:, 0].min()) - 3, 0)
        bottom = min(int(coords[:, 0].max()) + 4, image.shape[0])
        crop_mask = colored[top:bottom]
        crop_rgb = image[top:bottom]
        safe = np.full_like(crop_rgb, 255)
        safe[crop_mask] = crop_rgb[crop_mask]
        output = output_dir / f"{alias}_series307_ecg_gating_strip.png"
        plt.imsave(output, safe)

        highlight_left, highlight_right = highlighted_window_bounds(safe)
        margin = 8
        window_left = max(highlight_left - margin, 0)
        window_right = min(highlight_right + margin, safe.shape[1])
        window = safe[:, window_left:window_right]
        window_output = output_dir / f"{alias}_series307_ecg_scan_window.png"
        plt.imsave(window_output, window)

        manifest["items"].append(
            {
                "subject": alias,
                "output": output.name,
                "scan_window_output": window_output.name,
                "source_page": path.name,
                "source_crop_xyxy": [0, top, int(image.shape[1]), bottom],
                "highlight_bounds_xyxy_in_strip": [highlight_left, 0, highlight_right, int(safe.shape[0])],
                "scan_window_crop_xyxy_in_strip": [window_left, 0, window_right, int(safe.shape[0])],
                "retained_pixel_rule": "colored pixels only; all grayscale text/background removed",
                "scan_window_detection": "columns with colored-pixel coverage above 35%; grayscale gaps up to 4 px closed; longest wide interval selected; 8 px display margin added",
                "interpretation": "Recorded scan ECG/gating raster from GE Secondary Capture; no voltage calibration and no DICOM WaveformSequence.",
                "scan_window_interpretation": "Pixel-exact crop of the interval highlighted by the GE report; no resampling, interpolation, or waveform reconstruction.",
            }
        )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"outputs": len(manifest["items"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
