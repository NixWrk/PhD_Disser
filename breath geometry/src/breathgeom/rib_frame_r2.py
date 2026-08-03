"""R2: rib identification after the R1 preflight failed 1/10.

R1 established that the detection itself works -- fill fraction separates ribs
from scapulae cleanly -- but that the *indexing* drifts.  Three measured
failure modes are fixed here: fragments of one rib counted twice, partially
imaged ribs entering the field as the chest expands, and sides paired by
absolute height when the counts disagree.

One ambiguity is made explicit rather than fixed, because it cannot be fixed by
better bookkeeping: ribs are spaced about 20-25 mm apart while the measured
axial displacement of material points reaches 12-42 mm.  Position-based
matching can therefore be off by a whole rib, which shifts the entire map and
looks plausible.  The anchor gate fails loudly instead of guessing.

Protocol frozen in ``configs/wall_rib_frame_r2.json``; see
``docs/WALL_RIB_FRAME_R2.md``.  No R1 threshold is changed.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from breathgeom.measure.segmentation import body_mask_3d
from breathgeom.measure.wall import IntArray, WallParams, anatomical_midline

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class RibParamsR2:
    bone_hu_min: float = 200.0
    midline_slab_half_width_mm: float = 35.0
    min_component_voxels: int = 400
    max_fill_fraction: float = 0.02
    # Two fragments of one rib sit at nearly the same height; two ribs are a
    # full gap apart.
    fragment_merge_gap_fraction: float = 0.4
    drop_ribs_touching_volume_edge: bool = True


@dataclass(frozen=True)
class RibGatesR2:
    ribs_per_side_min: int = 8
    ribs_per_side_max: int = 12
    count_must_match_between_phases: bool = True
    max_gap_over_median_gap: float = 1.6
    left_right_z_tolerance_mm: float = 15.0
    anchor_agreement_max_gap_fraction: float = 0.5


@dataclass(frozen=True)
class RibConfigR2:
    frame_version: str
    dataset_id: str
    subjects: tuple[str, ...]
    phases: tuple[str, ...]
    params: RibParamsR2
    gates: RibGatesR2


@dataclass(frozen=True)
class PhaseRibsR2:
    subject_id: str
    phase: str
    left_count: int
    right_count: int
    dropped_at_edge: int
    merged_fragments: int
    left_gap_ratio: float
    right_gap_ratio: float
    median_gap_mm: float
    left_anchor_z_mm: float
    right_anchor_z_mm: float
    left_right_max_z_difference_mm: float


@dataclass(frozen=True)
class PreflightR2:
    subject_id: str
    count_in_range: bool
    count_matches_between_phases: bool
    gaps_regular: bool
    left_right_symmetric: bool
    anchor_agrees: bool
    passes: bool
    failure_reasons: str


def load_rib_config_r2(path: Path) -> RibConfigR2:
    top = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    policy = cast(dict[str, object], top.get("preflight_policy", {}))
    if policy.get("delta_blocked_until_preflight_passes") is not True:
        raise ValueError("R2 must block the delta map until the preflight passes")
    identification = cast(dict[str, object], top["rib_identification"])
    gates = cast(dict[str, object], top["rib_preflight_gates"])
    anchor = cast(dict[str, object], top["anchor"])
    if gates.get("anchor_must_agree") is not True:
        raise ValueError("R2 must keep the anchor gate; it is what refuses to guess")
    return RibConfigR2(
        frame_version=str(top["frame_version"]),
        dataset_id=str(top["dataset_id"]),
        subjects=tuple(str(item) for item in cast(list[object], top["subjects"])),
        phases=tuple(str(item) for item in cast(list[object], top["phases"])),
        params=RibParamsR2(
            bone_hu_min=float(cast(float, identification["bone_hu_min"])),
            midline_slab_half_width_mm=float(
                cast(float, identification["midline_slab_half_width_mm"])
            ),
            min_component_voxels=int(cast(int, identification["min_component_voxels"])),
            max_fill_fraction=float(cast(float, identification["max_fill_fraction"])),
            fragment_merge_gap_fraction=float(
                cast(float, identification["fragment_merge_gap_fraction"])
            ),
            drop_ribs_touching_volume_edge=bool(
                cast(bool, identification["drop_ribs_touching_volume_edge"])
            ),
        ),
        gates=RibGatesR2(
            ribs_per_side_min=int(cast(int, gates["ribs_per_side_min"])),
            ribs_per_side_max=int(cast(int, gates["ribs_per_side_max"])),
            count_must_match_between_phases=bool(
                cast(bool, gates["count_must_match_between_phases"])
            ),
            max_gap_over_median_gap=float(cast(float, gates["max_gap_over_median_gap"])),
            left_right_z_tolerance_mm=float(cast(float, gates["left_right_z_tolerance_mm"])),
            anchor_agreement_max_gap_fraction=float(
                cast(float, anchor["agreement_max_gap_fraction"])
            ),
        ),
    )


def merge_fragments(heights: FloatArray, merge_fraction: float) -> tuple[FloatArray, int]:
    """Collapse components that sit at nearly the same height into one rib."""
    if len(heights) < 2:
        return heights, 0
    ordered = np.sort(heights)
    gaps = np.diff(ordered)
    reference = float(np.median(gaps))
    if not np.isfinite(reference) or reference <= 0.0:
        return ordered, 0
    limit = merge_fraction * reference
    merged: list[float] = [float(ordered[0])]
    group: list[float] = [float(ordered[0])]
    collapsed = 0
    for value in ordered[1:]:
        if float(value) - group[-1] < limit:
            group.append(float(value))
            merged[-1] = float(np.mean(group))
            collapsed += 1
        else:
            group = [float(value)]
            merged.append(float(value))
    return np.asarray(merged, dtype=np.float64), collapsed


def detect_ribs_r2(
    volume_ras: IntArray,
    spacing: tuple[float, float, float],
    params: RibParamsR2,
    *,
    subject_id: str,
    phase: str,
    wall_params: WallParams | None = None,
) -> PhaseRibsR2:
    """Detect, drop partially imaged ribs, merge fragments, index from the top."""
    wall_params = wall_params or WallParams()
    body = body_mask_3d(volume_ras, wall_params)
    midline = anatomical_midline(volume_ras, wall_params)
    bone: BoolArray = body & (volume_ras > params.bone_hu_min)

    half = params.midline_slab_half_width_mm / spacing[0]
    lateral = bone.copy()
    lateral[max(int(midline - half), 0) : int(midline + half), :, :] = False

    last_slice = volume_ras.shape[2] - 1
    labels, _ = ndimage.label(lateral)
    sides: dict[str, list[float]] = {"left": [], "right": []}
    dropped = 0
    for index, box in enumerate(ndimage.find_objects(labels), start=1):
        mask = labels[box] == index
        voxels = int(mask.sum())
        if voxels < params.min_component_voxels:
            continue
        extent = np.asarray([item.stop - item.start for item in box], dtype=np.float64)
        if voxels / float(np.prod(extent)) > params.max_fill_fraction:
            continue
        # A rib cut by the end of the volume is not a rib; it enters and leaves
        # detection as the chest expands, which is what drifted the R1 counts.
        if params.drop_ribs_touching_volume_edge and (
            box[2].start == 0 or box[2].stop - 1 >= last_slice
        ):
            dropped += 1
            continue
        where = np.nonzero(mask)
        centre_row = box[0].start + float(where[0].mean())
        side = "right" if centre_row > midline else "left"
        sides[side].append((box[2].start + float(np.median(where[2]))) * spacing[2])

    merged_total = 0
    per_side: dict[str, FloatArray] = {}
    ratios: dict[str, float] = {}
    for side, values in sides.items():
        heights, collapsed = merge_fragments(
            np.asarray(values, dtype=np.float64), params.fragment_merge_gap_fraction
        )
        merged_total += collapsed
        # Index 0 is the topmost rib, so the anchor sits where ribs move least.
        per_side[side] = heights[::-1]
        gaps = np.diff(np.sort(heights))
        ratios[side] = (
            float(gaps.max() / np.median(gaps)) if len(gaps) > 1 else float("nan")
        )

    all_gaps = np.concatenate(
        [np.diff(np.sort(values)) for values in per_side.values() if len(values) > 1]
        or [np.asarray([np.nan])]
    )
    paired = min(len(per_side["left"]), len(per_side["right"]))
    symmetry = (
        float(np.abs(per_side["left"][:paired] - per_side["right"][:paired]).max())
        if paired
        else float("nan")
    )
    return PhaseRibsR2(
        subject_id=subject_id,
        phase=phase,
        left_count=len(per_side["left"]),
        right_count=len(per_side["right"]),
        dropped_at_edge=dropped,
        merged_fragments=merged_total,
        left_gap_ratio=ratios["left"],
        right_gap_ratio=ratios["right"],
        median_gap_mm=float(np.nanmedian(all_gaps)),
        left_anchor_z_mm=float(per_side["left"][0]) if len(per_side["left"]) else float("nan"),
        right_anchor_z_mm=(
            float(per_side["right"][0]) if len(per_side["right"]) else float("nan")
        ),
        left_right_max_z_difference_mm=symmetry,
    )


def evaluate_preflight_r2(
    phases: tuple[PhaseRibsR2, ...], gates: RibGatesR2
) -> PreflightR2:
    """Every R1 gate unchanged, plus the anchor gate that refuses to guess."""
    if not phases:
        raise ValueError("no phases to evaluate")
    reasons: list[str] = []
    counts = [(item.left_count, item.right_count) for item in phases]

    in_range = all(
        gates.ribs_per_side_min <= value <= gates.ribs_per_side_max
        for pair in counts
        for value in pair
    )
    if not in_range:
        reasons.append("rib_count_out_of_range")

    matches = len(set(counts)) == 1
    if gates.count_must_match_between_phases and not matches:
        reasons.append("rib_count_differs_between_phases")

    ratios = [
        value
        for item in phases
        for value in (item.left_gap_ratio, item.right_gap_ratio)
        if np.isfinite(value)
    ]
    regular = all(value <= gates.max_gap_over_median_gap for value in ratios)
    if not regular:
        reasons.append("irregular_rib_spacing")

    symmetric = all(
        np.isfinite(item.left_right_max_z_difference_mm)
        and item.left_right_max_z_difference_mm <= gates.left_right_z_tolerance_mm
        for item in phases
    )
    if not symmetric:
        reasons.append("left_right_asymmetry")

    anchor_agrees = True
    if len(phases) > 1:
        spacing = float(np.nanmedian([item.median_gap_mm for item in phases]))
        limit = gates.anchor_agreement_max_gap_fraction * spacing
        for attribute in ("left_anchor_z_mm", "right_anchor_z_mm"):
            values = [getattr(item, attribute) for item in phases]
            if not all(np.isfinite(value) for value in values):
                anchor_agrees = False
                break
            if max(values) - min(values) > limit:
                anchor_agrees = False
                break
    if not anchor_agrees:
        reasons.append("anchor_ambiguous")

    return PreflightR2(
        subject_id=phases[0].subject_id,
        count_in_range=in_range,
        count_matches_between_phases=matches,
        gaps_regular=regular,
        left_right_symmetric=symmetric,
        anchor_agrees=anchor_agrees,
        passes=not reasons,
        failure_reasons=";".join(reasons),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_preflight_r2(
    output_dir: Path,
    config_path: Path,
    config: RibConfigR2,
    phases: tuple[PhaseRibsR2, ...],
    verdicts: tuple[PreflightR2, ...],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_path = output_dir / "phases.csv"
    rows = [asdict(item) for item in phases]
    with phase_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    verdict_path = output_dir / "verdicts.csv"
    verdict_rows = [asdict(item) for item in verdicts]
    with verdict_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(verdict_rows[0]))
        writer.writeheader()
        writer.writerows(verdict_rows)

    passed = [item for item in verdicts if item.passes]
    summary = {
        "frame_version": config.frame_version,
        "dataset_id": config.dataset_id,
        "supersedes": "wall-dual-frame-r1",
        "subjects_evaluated": len(verdicts),
        "subjects_passed": len(passed),
        "passing_subjects": [item.subject_id for item in passed],
        "delta_map_produced": False,
        "failure_counts": {
            reason: sum(reason in item.failure_reasons for item in verdicts)
            for reason in (
                "rib_count_out_of_range",
                "rib_count_differs_between_phases",
                "irregular_rib_spacing",
                "left_right_asymmetry",
                "anchor_ambiguous",
            )
        },
        "verdicts": verdict_rows,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    manifest = {
        "frame_version": config.frame_version,
        "config": {"path": str(config_path).replace("\\", "/"), "sha256": _sha256(config_path)},
        "phases_sha256": _sha256(phase_path),
        "verdicts_sha256": _sha256(verdict_path),
        "summary_sha256": _sha256(summary_path),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return manifest_path


__all__ = [
    "PhaseRibsR2",
    "PreflightR2",
    "RibConfigR2",
    "RibGatesR2",
    "RibParamsR2",
    "detect_ribs_r2",
    "evaluate_preflight_r2",
    "load_rib_config_r2",
    "merge_fragments",
    "write_preflight_r2",
]
