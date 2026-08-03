"""R1 preflight: is a rib-indexed material frame available at all?

A rib-anchored coordinate is the only discrete, identity-preserving landmark
the chest wall offers, and identity is what a Lagrangian delta-h needs.  But a
miscounted rib shifts the whole map and produces a systematic error that looks
exactly like a real result, which is worse than noise because the spread does
not reveal it.

So this module identifies ribs and then tries hard to break its own indexing:
each gate fails under a different miscount mode.  Until they all pass, the
delta map is not produced -- the same rule the J1.2 truth preflight used.
Protocol frozen in ``configs/wall_dual_frame_r1.json``; see
``docs/WALL_DUAL_FRAME_R1.md``.
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


@dataclass(frozen=True)
class RibDetectionParams:
    """How bone is broken into individual ribs."""

    bone_hu_min: float = 200.0
    # Cutting a midline slab severs the costovertebral junctions; without it
    # the whole skeleton is one component.
    midline_slab_half_width_mm: float = 35.0
    min_component_voxels: int = 400
    # A rib is a thin curved arc, so it fills very little of its bounding box.
    # Bounding-box elongation does NOT separate ribs from scapulae, because a
    # rib's box is nearly cubic; measured fill separates them cleanly.
    max_fill_fraction: float = 0.02


@dataclass(frozen=True)
class RibPreflightGates:
    ribs_per_side_min: int = 8
    ribs_per_side_max: int = 12
    count_must_match_between_phases: bool = True
    max_gap_over_median_gap: float = 1.6
    left_right_z_tolerance_mm: float = 15.0


@dataclass(frozen=True)
class RibFrameConfig:
    frame_version: str
    dataset_id: str
    subjects: tuple[str, ...]
    phases: tuple[str, ...]
    detection: RibDetectionParams
    gates: RibPreflightGates
    delta_blocked_until_preflight_passes: bool


@dataclass(frozen=True)
class Rib:
    side: str
    order: int
    voxels: int
    fill_fraction: float
    centre_z_mm: float
    arc_extent_mm: float


@dataclass(frozen=True)
class PhaseRibs:
    subject_id: str
    phase: str
    left_count: int
    right_count: int
    left_gap_ratio: float
    right_gap_ratio: float
    left_right_max_z_difference_mm: float
    median_arc_extent_mm: float
    ribs: tuple[Rib, ...]


@dataclass(frozen=True)
class SubjectPreflight:
    subject_id: str
    count_in_range: bool
    count_matches_between_phases: bool
    gaps_regular: bool
    left_right_symmetric: bool
    passes: bool
    failure_reasons: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_rib_frame_config(path: Path) -> RibFrameConfig:
    top = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    for key in ("frame_version", "dataset_id", "subjects", "phases", "rib_identification",
                "rib_preflight_gates", "preflight_policy"):
        if key not in top:
            raise ValueError(f"{path} is missing required key {key}")
    identification = cast(dict[str, object], top["rib_identification"])
    gates = cast(dict[str, object], top["rib_preflight_gates"])
    policy = cast(dict[str, object], top["preflight_policy"])
    if policy.get("delta_blocked_until_preflight_passes") is not True:
        raise ValueError("R1 must block the delta map until the preflight passes")
    if "max_fill_fraction" not in identification:
        raise ValueError(
            "rib_identification must state max_fill_fraction; bounding-box elongation "
            "was measured not to separate ribs from scapulae"
        )
    return RibFrameConfig(
        frame_version=str(top["frame_version"]),
        dataset_id=str(top["dataset_id"]),
        subjects=tuple(str(item) for item in cast(list[object], top["subjects"])),
        phases=tuple(str(item) for item in cast(list[object], top["phases"])),
        detection=RibDetectionParams(
            bone_hu_min=float(cast(float, identification["bone_hu_min"])),
            midline_slab_half_width_mm=float(
                cast(float, identification["midline_slab_half_width_mm"])
            ),
            min_component_voxels=int(cast(int, identification["min_component_voxels"])),
            max_fill_fraction=float(cast(float, identification["max_fill_fraction"])),
        ),
        gates=RibPreflightGates(
            ribs_per_side_min=int(cast(int, gates["ribs_per_side_min"])),
            ribs_per_side_max=int(cast(int, gates["ribs_per_side_max"])),
            count_must_match_between_phases=bool(
                cast(bool, gates["count_must_match_between_phases"])
            ),
            max_gap_over_median_gap=float(cast(float, gates["max_gap_over_median_gap"])),
            left_right_z_tolerance_mm=float(
                cast(float, gates["left_right_z_tolerance_mm"])
            ),
        ),
        delta_blocked_until_preflight_passes=True,
    )


def detect_ribs(
    volume_ras: IntArray,
    spacing: tuple[float, float, float],
    params: RibDetectionParams,
    *,
    subject_id: str,
    phase: str,
    wall_params: WallParams | None = None,
) -> PhaseRibs:
    """Split chest-wall bone into individual ribs and order them along z."""
    wall_params = wall_params or WallParams()
    body = body_mask_3d(volume_ras, wall_params)
    midline = anatomical_midline(volume_ras, wall_params)
    bone: BoolArray = body & (volume_ras > params.bone_hu_min)

    half = params.midline_slab_half_width_mm / spacing[0]
    lateral = bone.copy()
    lateral[max(int(midline - half), 0) : int(midline + half), :, :] = False

    labels, _ = ndimage.label(lateral)
    sides: dict[str, list[Rib]] = {"left": [], "right": []}
    for index, box in enumerate(ndimage.find_objects(labels), start=1):
        mask = labels[box] == index
        voxels = int(mask.sum())
        if voxels < params.min_component_voxels:
            continue
        extent = np.asarray([item.stop - item.start for item in box], dtype=np.float64)
        fill = voxels / float(np.prod(extent))
        if fill > params.max_fill_fraction:
            continue
        where = np.nonzero(mask)
        centre_row = box[0].start + float(where[0].mean())
        side = "right" if centre_row > midline else "left"
        sides[side].append(
            Rib(
                side=side,
                order=-1,
                voxels=voxels,
                fill_fraction=fill,
                centre_z_mm=(box[2].start + float(np.median(where[2]))) * spacing[2],
                arc_extent_mm=float(np.linalg.norm(extent * np.asarray(spacing))),
            )
        )

    ordered: list[Rib] = []
    statistics: dict[str, tuple[int, float, npt.NDArray[np.float64]]] = {}
    for side, found in sides.items():
        found.sort(key=lambda item: item.centre_z_mm)
        heights = np.asarray([item.centre_z_mm for item in found], dtype=np.float64)
        gaps = np.diff(heights)
        ratio = (
            float(gaps.max() / np.median(gaps)) if len(gaps) > 1 else float("nan")
        )
        statistics[side] = (len(found), ratio, heights)
        ordered.extend(
            Rib(
                side=item.side,
                order=position,
                voxels=item.voxels,
                fill_fraction=item.fill_fraction,
                centre_z_mm=item.centre_z_mm,
                arc_extent_mm=item.arc_extent_mm,
            )
            for position, item in enumerate(found)
        )

    paired = min(statistics["left"][0], statistics["right"][0])
    symmetry = (
        float(np.abs(statistics["left"][2][:paired] - statistics["right"][2][:paired]).max())
        if paired
        else float("nan")
    )
    return PhaseRibs(
        subject_id=subject_id,
        phase=phase,
        left_count=statistics["left"][0],
        right_count=statistics["right"][0],
        left_gap_ratio=statistics["left"][1],
        right_gap_ratio=statistics["right"][1],
        left_right_max_z_difference_mm=symmetry,
        median_arc_extent_mm=float(
            np.median([item.arc_extent_mm for item in ordered]) if ordered else float("nan")
        ),
        ribs=tuple(ordered),
    )


def evaluate_preflight(
    phases: tuple[PhaseRibs, ...], gates: RibPreflightGates
) -> SubjectPreflight:
    """Apply every frozen gate; each one breaks under a different miscount."""
    if not phases:
        raise ValueError("no phases to evaluate")
    subject = phases[0].subject_id
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

    return SubjectPreflight(
        subject_id=subject,
        count_in_range=in_range,
        count_matches_between_phases=matches,
        gaps_regular=regular,
        left_right_symmetric=symmetric,
        passes=not reasons,
        failure_reasons=";".join(reasons),
    )


def write_preflight(
    output_dir: Path,
    config_path: Path,
    config: RibFrameConfig,
    phases: tuple[PhaseRibs, ...],
    verdicts: tuple[SubjectPreflight, ...],
) -> Path:
    """Write per-phase detections, per-subject verdicts and provenance."""
    output_dir.mkdir(parents=True, exist_ok=True)

    phase_path = output_dir / "phases.csv"
    phase_rows = [
        {key: value for key, value in asdict(item).items() if key != "ribs"}
        for item in phases
    ]
    with phase_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(phase_rows[0]))
        writer.writeheader()
        writer.writerows(phase_rows)

    rib_path = output_dir / "ribs.csv"
    rib_rows = [
        {"subject_id": item.subject_id, "phase": item.phase, **asdict(rib)}
        for item in phases
        for rib in item.ribs
    ]
    with rib_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rib_rows[0]))
        writer.writeheader()
        writer.writerows(rib_rows)

    passed = [item for item in verdicts if item.passes]
    summary = {
        "frame_version": config.frame_version,
        "dataset_id": config.dataset_id,
        "subjects_evaluated": len(verdicts),
        "subjects_passed": len(passed),
        "passing_subjects": [item.subject_id for item in passed],
        "delta_map_produced": False,
        "delta_block_reason": (
            "the frozen protocol blocks any delta map until the rib preflight passes"
            if len(passed) != len(verdicts)
            else "preflight passed; the delta map is a separate frozen step"
        ),
        "verdicts": [asdict(item) for item in verdicts],
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
        "ribs_sha256": _sha256(rib_path),
        "summary_sha256": _sha256(summary_path),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return manifest_path


__all__ = [
    "PhaseRibs",
    "Rib",
    "RibDetectionParams",
    "RibFrameConfig",
    "RibPreflightGates",
    "SubjectPreflight",
    "detect_ribs",
    "evaluate_preflight",
    "load_rib_frame_config",
    "write_preflight",
]
