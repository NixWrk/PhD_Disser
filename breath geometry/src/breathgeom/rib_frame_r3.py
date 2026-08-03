"""R3: the last attempt at a rib-indexed material frame.

R2 made things worse -- 0/10 against R1's 1/10 -- because its edge rule was
applied per phase and so was itself phase dependent: at inspiration the
expanded chest pushes more ribs against the volume boundary, so more got
dropped and the counts diverged further.  R3 replaces it with one window
computed from the intersection of both phases and applied to both, which
cannot depend on which phase is being processed.

The ultrasound literature says the remaining obstacle is physics, not code:
Huang 2024 measured a significant effect of respiration on intercostal width
(F[1,58]=75.27, p<0.001), so rib spacing is not a respiratory invariant.  This
is therefore the last attempt; failure on anchor_ambiguous is a result about
the anatomy, not a reason for a fourth version.

Protocol frozen in ``configs/wall_rib_frame_r3.json``; see
``docs/WALL_RIB_FRAME_R3.md``.  No gate threshold is changed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from breathgeom.measure.segmentation import body_mask_3d
from breathgeom.measure.wall import IntArray, WallParams, anatomical_midline
from breathgeom.rib_frame_r2 import (
    PhaseRibsR2,
    PreflightR2,
    RibGatesR2,
    RibParamsR2,
    evaluate_preflight_r2,
    merge_fragments,
)

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class RawRib:
    side: str
    centre_z_mm: float
    touches_edge: bool


@dataclass(frozen=True)
class RibConfigR3:
    frame_version: str
    dataset_id: str
    subjects: tuple[str, ...]
    phases: tuple[str, ...]
    params: RibParamsR2
    gates: RibGatesR2


def load_rib_config_r3(path: Path) -> RibConfigR3:
    top = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    identification = cast(dict[str, object], top["rib_identification"])
    if identification.get("common_axial_window") is not True:
        raise ValueError("R3 must use one common axial window, not per-phase edge dropping")
    if identification.get("drop_ribs_touching_volume_edge") is not False:
        raise ValueError("R3 must not drop edge ribs per phase; that is what failed in R2")
    gates = cast(dict[str, object], top["rib_preflight_gates"])
    anchor = cast(dict[str, object], top["anchor"])
    if gates.get("anchor_must_agree") is not True:
        raise ValueError("R3 must keep the anchor gate")
    return RibConfigR3(
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
            drop_ribs_touching_volume_edge=False,
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


def detect_raw_ribs(
    volume_ras: IntArray,
    spacing: tuple[float, float, float],
    params: RibParamsR2,
    wall_params: WallParams | None = None,
) -> tuple[RawRib, ...]:
    """Every rib component with its height and whether the volume cut it."""
    wall_params = wall_params or WallParams()
    body = body_mask_3d(volume_ras, wall_params)
    midline = anatomical_midline(volume_ras, wall_params)
    bone: BoolArray = body & (volume_ras > params.bone_hu_min)
    half = params.midline_slab_half_width_mm / spacing[0]
    lateral = bone.copy()
    lateral[max(int(midline - half), 0) : int(midline + half), :, :] = False

    last = volume_ras.shape[2] - 1
    labels, _ = ndimage.label(lateral)
    found: list[RawRib] = []
    for index, box in enumerate(ndimage.find_objects(labels), start=1):
        mask = labels[box] == index
        voxels = int(mask.sum())
        if voxels < params.min_component_voxels:
            continue
        extent = np.asarray([item.stop - item.start for item in box], dtype=np.float64)
        if voxels / float(np.prod(extent)) > params.max_fill_fraction:
            continue
        where = np.nonzero(mask)
        centre_row = box[0].start + float(where[0].mean())
        found.append(
            RawRib(
                side="right" if centre_row > midline else "left",
                centre_z_mm=(box[2].start + float(np.median(where[2]))) * spacing[2],
                touches_edge=box[2].start == 0 or box[2].stop - 1 >= last,
            )
        )
    return tuple(found)


def common_axial_window(
    phases: dict[str, tuple[RawRib, ...]],
) -> tuple[float, float]:
    """Intersection of the fully imaged rib span of every phase.

    Computed once across phases and applied to all of them, so unlike the R2
    rule it cannot depend on which phase is being processed.
    """
    lows: list[float] = []
    highs: list[float] = []
    for ribs in phases.values():
        inside = [item.centre_z_mm for item in ribs if not item.touches_edge]
        if not inside:
            raise ValueError("a phase has no fully imaged rib; the window is undefined")
        lows.append(min(inside))
        highs.append(max(inside))
    low, high = max(lows), min(highs)
    if low >= high:
        raise ValueError("the phases share no axial window")
    return low, high


def summarise_phase(
    ribs: tuple[RawRib, ...],
    window: tuple[float, float],
    params: RibParamsR2,
    *,
    subject_id: str,
    phase: str,
) -> PhaseRibsR2:
    """Apply the shared window, merge fragments and index from the top."""
    low, high = window
    per_side: dict[str, FloatArray] = {}
    ratios: dict[str, float] = {}
    merged_total = 0
    for side in ("left", "right"):
        heights = np.asarray(
            [
                item.centre_z_mm
                for item in ribs
                if item.side == side and low <= item.centre_z_mm <= high
            ],
            dtype=np.float64,
        )
        merged, collapsed = merge_fragments(heights, params.fragment_merge_gap_fraction)
        merged_total += collapsed
        per_side[side] = merged[::-1]
        gaps = np.diff(np.sort(merged))
        ratios[side] = float(gaps.max() / np.median(gaps)) if len(gaps) > 1 else float("nan")

    pieces = [np.diff(np.sort(values)) for values in per_side.values() if len(values) > 1]
    all_gaps = np.concatenate(pieces) if pieces else np.asarray([np.nan])
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
        dropped_at_edge=sum(item.touches_edge for item in ribs),
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


def evaluate_subject_r3(
    volumes: dict[str, IntArray],
    spacing: tuple[float, float, float],
    config: RibConfigR3,
    subject_id: str,
) -> tuple[tuple[PhaseRibsR2, ...], PreflightR2]:
    """Detect in every phase, share one window, then apply the unchanged gates."""
    raw = {
        phase: detect_raw_ribs(volume, spacing, config.params)
        for phase, volume in volumes.items()
    }
    window = common_axial_window(raw)
    phases = tuple(
        summarise_phase(
            raw[phase], window, config.params, subject_id=subject_id, phase=phase
        )
        for phase in config.phases
    )
    return phases, evaluate_preflight_r2(phases, config.gates)


def relaxed_for_diagnosis(gates: RibGatesR2) -> RibGatesR2:
    """Gates with the count-match requirement off, for diagnosis only.

    Never used for a verdict. It answers one question the strict run cannot:
    once the counts are allowed to differ, do the remaining geometric gates
    still hold, or is the frame broken in more than one way?
    """
    return replace(gates, count_must_match_between_phases=False)


__all__ = [
    "RawRib",
    "RibConfigR3",
    "common_axial_window",
    "detect_raw_ribs",
    "evaluate_subject_r3",
    "load_rib_config_r3",
    "relaxed_for_diagnosis",
    "summarise_phase",
]
