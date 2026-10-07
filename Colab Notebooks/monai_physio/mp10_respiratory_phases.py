"""Explicit acquisition-phase contract for the published lung stage parser.

TCIA g070 is a shape-model reference, NOT a new origin of the breathing cycle.
The training driver parses the acquisition percentage directly / 100.
No phase is selected or shifted using target CT or prediction accuracy.
"""

from __future__ import annotations

import math
import re


SOURCES = [
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC5912888/",
    "https://purehost.bath.ac.uk/ws/portalfiles/portal/287959945/EIT_Guided_by_DT_and_DL_final.pdf",
    "https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/4dct.html",
    "https://github.com/Project-MONAI/monai-physio/blob/2026.09.1/tutorials/tutorial_09_lung_train_physicsnemo_mgn.py",
    "https://github.com/Project-MONAI/monai-physio/blob/2026.09.1/data/TCIA-4DLung/convert.py",
]


def phase_stage(phase: str, dataset: str | None = None) -> float:
    """Parse only documented ten-bin TCIA or DIR-Lab acquisition labels."""
    match = re.fullmatch(r"(?:T([0-9]{2})|g([0-9]{3}))", phase)
    if match is None:
        raise ValueError(f"Unverified respiratory phase {phase!r}; expected T00..T90 or g000..g090")
    inferred = "dirlab" if match[1] is not None else "tcia"
    if dataset is not None and dataset != inferred:
        raise ValueError(f"Phase {phase!r} does not belong to {dataset}")
    percent = int(match[1] if match[1] is not None else match[2])
    if percent not in range(0, 100, 10):
        raise ValueError(f"Undocumented acquisition bin: {phase}")
    return percent / 100.0


def phase_contract(source_phase: str, anchor: float | None, stages: list[float],
                   *, allow_unverified: bool = False) -> dict:
    """Derive the source anchor; reject contradictory manual values."""
    try:
        expected = phase_stage(source_phase)
    except ValueError:
        if not allow_unverified or anchor is None:
            raise
        expected = anchor
        dataset = "unverified"
    else:
        dataset = "dirlab" if source_phase.startswith("T") else "tcia"
    if anchor is not None and not math.isclose(anchor, expected, abs_tol=1e-9):
        raise ValueError(f"Source {source_phase} requires anchor {expected}; received {anchor}")
    if not stages or any(not math.isfinite(s) or not 0 <= s < 1 for s in stages):
        raise ValueError("Stages must be finite fractions in [0, 1)")
    if len(set(stages)) != len(stages) or expected not in stages:
        raise ValueError("Stages must be unique and include the source anchor")
    return {
        "schema": "mp10_respiratory_phase_contract_v1",
        "dataset": dataset,
        "status": "published_acquisition_convention_verified" if dataset != "unverified"
                  else "unverified_phase_scenario_only",
        "source_phase": source_phase,
        "source_stage": expected,
        "stage_rule": "acquisition percentage / 100; no shift, inversion or outcome-based alignment",
        "cycle_zero": "end_inhalation" if dataset != "unverified" else None,
        "end_exhalation_stage": 0.5 if dataset != "unverified" else None,
        "shape_model_reference_stage": 0.7,
        "shape_model_reference_is_end_exhalation": False,
        "source_matches_published_shape_reference": expected == 0.7,
        "target_stages": stages,
        "checkpoint_physiological_manifest_present": False,
        "individual_breathing_amplitude_verified": False,
        "sources": SOURCES,
    }


def validate_scenario(report: dict, dataset: str) -> dict:
    phase_stage(report["source_phase"], dataset)
    contract = phase_contract(report["source_phase"], report["stage_anchor"],
                              [float(row["stage"]) for row in report["stages"]])
    recorded = report.get("phase_contract")
    if recorded is not None and recorded != contract:
        raise ValueError("Saved phase contract disagrees with scenario stages")
    return contract


def check_fit_phase(fit_report: dict, source_phase: str) -> None:
    """Reject reuse of a fit explicitly tagged with another CT phase."""
    fitted_phase = fit_report.get("source_phase")
    if fitted_phase is not None and fitted_phase != source_phase:
        raise ValueError(f"Fit belongs to {fitted_phase}; requested source is {source_phase}")
