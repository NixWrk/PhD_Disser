"""Reader for DIR-Lab COPDgene raw volumes.

These files are headerless Analyze ``.img`` dumps: no dimensions, no voxel size,
no orientation, no intensity scale. Everything has to come from the published
table or be recovered from the image itself, and every recovered fact is
reported so it can be audited rather than trusted.

Three traps, all silent if unhandled:

* dimensions and spacing live only on the DIR-Lab download page;
* intensities are stored as ``HU + 1024``, so an unshifted read puts air at
  about ``+25`` and no thresholding works;
* out-of-field voxels carry ``-2000`` raw, which is below air and would be
  counted as lung by any ``< -400`` test.

Left and right are not encoded anywhere, so they are decided anatomically: the
heart displaces the left lung, making the right lung the larger one at mid
thorax. Getting this backwards mirrors the subject without any error.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from zipfile import BadZipFile, ZipFile

import numpy as np
import numpy.typing as npt
from scipy import ndimage

IntArray = npt.NDArray[np.int16]
BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]

# Stored value minus this is Hounsfield units.
HU_OFFSET = 1024
# Raw values at or below this mark voxels outside the reconstruction field.
RAW_OUTSIDE = -1500
# The recovered air peak must land this close to -1000 HU or the file is not
# what we think it is.
AIR_PEAK_TOLERANCE_HU = 60.0
# Nothing in a CT is less dense than air, so only noise may sit below -1100 HU.
IMPLAUSIBLE_FRACTION = 0.02
# Dense enough to be vertebra or rib rather than calcified soft tissue.
SPINE_HU = 300
# Fewer bone voxels than this in a slice is speckle, not a vertebra.
MIN_BONE_PX = 50
BODY_HU = -300
AIR_HU = -400
# Below this the recovered right/left asymmetry is too small to decide on.
MIN_LUNG_ASYMMETRY = 0.02


@dataclass(frozen=True)
class CaseGeometry:
    """Published acquisition parameters; the file itself carries none."""

    columns: int
    rows: int
    slices: int
    spacing_x_mm: float
    spacing_y_mm: float
    spacing_z_mm: float


# Published on the DIR-Lab COPDgene download page. Slice counts are re-derived
# from file size at load time, so a wrong entry here cannot pass unnoticed.
DIRLAB_COPDGENE: dict[str, CaseGeometry] = {
    "copd1": CaseGeometry(512, 512, 121, 0.625, 0.625, 2.5),
    "copd2": CaseGeometry(512, 512, 102, 0.645, 0.645, 2.5),
    "copd3": CaseGeometry(512, 512, 126, 0.652, 0.652, 2.5),
    "copd4": CaseGeometry(512, 512, 126, 0.590, 0.590, 2.5),
    "copd5": CaseGeometry(512, 512, 131, 0.647, 0.647, 2.5),
    "copd6": CaseGeometry(512, 512, 119, 0.633, 0.633, 2.5),
    "copd7": CaseGeometry(512, 512, 112, 0.625, 0.625, 2.5),
    "copd8": CaseGeometry(512, 512, 115, 0.586, 0.586, 2.5),
    "copd9": CaseGeometry(512, 512, 116, 0.664, 0.664, 2.5),
    "copd10": CaseGeometry(512, 512, 135, 0.742, 0.742, 2.5),
}

COPDGENE_REQUIRED_SUFFIXES = (
    "_iBHCT.img",
    "_eBHCT.img",
    "_300_iBH_xyz_r1.txt",
    "_300_eBH_xyz_r1.txt",
)


@dataclass(frozen=True)
class COPDgeneCaseInventory:
    """Availability of one COPDgene case without reading image pixel data."""

    case_id: str
    archive_path: Path | None
    archive_complete: bool
    extracted_dir: Path | None
    inhale_image: Path | None
    exhale_image: Path | None
    inhale_landmarks: Path | None
    exhale_landmarks: Path | None

    @property
    def extracted_complete(self) -> bool:
        return all(
            path is not None
            for path in (
                self.inhale_image,
                self.exhale_image,
                self.inhale_landmarks,
                self.exhale_landmarks,
            )
        )


def _case_number(case_id: str) -> int:
    return int(case_id.removeprefix("copd"))


def _archive_has_case(archive: Path, case_id: str) -> bool:
    try:
        with ZipFile(archive) as stream:
            names = [Path(name).name for name in stream.namelist() if not name.endswith("/")]
    except (BadZipFile, OSError):
        return False
    return all(f"{case_id}{suffix}" in names for suffix in COPDGENE_REQUIRED_SUFFIXES)


def _find_case_dir(root: Path, case_id: str) -> Path | None:
    candidates = (root / "extracted" / case_id, root / case_id)
    return next((path for path in candidates if path.is_dir()), None)


def _existing(path: Path | None, name: str) -> Path | None:
    if path is None:
        return None
    candidate = path / name
    return candidate if candidate.is_file() else None


def inventory_copdgene(root: Path) -> tuple[COPDgeneCaseInventory, ...]:
    """Inventory all ten COPDgene cases in archives and extracted folders."""
    rows: list[COPDgeneCaseInventory] = []
    for case_id in sorted(DIRLAB_COPDGENE, key=_case_number):
        archive_candidate = root / f"{case_id}.zip"
        archive = archive_candidate if archive_candidate.is_file() else None
        folder = _find_case_dir(root, case_id)
        rows.append(
            COPDgeneCaseInventory(
                case_id=case_id,
                archive_path=archive,
                archive_complete=bool(archive and _archive_has_case(archive, case_id)),
                extracted_dir=folder,
                inhale_image=_existing(folder, f"{case_id}_iBHCT.img"),
                exhale_image=_existing(folder, f"{case_id}_eBHCT.img"),
                inhale_landmarks=_existing(folder, f"{case_id}_300_iBH_xyz_r1.txt"),
                exhale_landmarks=_existing(folder, f"{case_id}_300_eBH_xyz_r1.txt"),
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class OrientationReport:
    """How left/right and head/foot were decided, and how convincingly.

    Both axes are decided once per subject from the fuller-inflated scan and
    then reused for the other phase: two phases of one person cannot have
    different anatomy, so inferring twice can only introduce disagreement.
    """

    right_is_high_index: bool
    lung_asymmetry: float
    superior_is_high_index: bool
    caudal_air_excess: float
    diaphragm_sharpness_ratio: float
    posterior_is_high_row: bool
    spine_offset_px: float
    air_peak_hu: float

    @property
    def sides_agree(self) -> bool:
        """Do the two independent head/foot cues point the same way?"""
        by_volume = self.caudal_air_excess > 0
        by_sharpness = self.diaphragm_sharpness_ratio > 1.0
        return by_volume == by_sharpness

    @property
    def confident(self) -> bool:
        return abs(self.lung_asymmetry) >= MIN_LUNG_ASYMMETRY and self.sides_agree


def _to_hu(raw: IntArray) -> tuple[npt.NDArray[np.int32], BoolArray, float]:
    """Shift to Hounsfield units and confirm the shift against the air peak.

    The check uses the *mode*, not the mean or median: external air is a sharp
    spike at -1000 while emphysematous lung spreads broadly from -1000 upwards,
    so an average over everything below -700 HU sits far above the true peak.
    """
    outside: BoolArray = raw <= RAW_OUTSIDE
    hu = raw.astype(np.int32) - HU_OFFSET
    inside = hu[~outside]
    if inside.size == 0:
        raise ValueError("Every voxel is out of field: the file is not what was assumed")

    # The guard is not the position of a peak but the floor: air is the least
    # dense thing a scan contains, so with the right shift almost nothing sits
    # below -1100 HU. A file already in Hounsfield units puts its air at about
    # -2024 after the shift, and that shows up here immediately. Looking for a
    # peak instead would not work: how much external air a volume holds depends
    # on the field size, and in a tight field soft tissue outnumbers it.
    below_air = float(np.count_nonzero(inside < -1100)) / float(inside.size)
    if below_air > IMPLAUSIBLE_FRACTION:
        raise ValueError(
            f"{100 * below_air:.0f}% of in-field voxels fall below -1100 HU after the "
            "shift; the stored intensity offset differs from the assumed HU+1024"
        )

    window = inside[(inside >= -1100) & (inside <= -900)]
    if window.size == 0:
        raise ValueError("No air-like voxels: the intensity scale is not what was assumed")
    counts = np.bincount((window + 1100).astype(np.int64), minlength=201)
    peak = float(int(np.argmax(counts)) - 1100)
    if abs(peak - (-1000.0)) > AIR_PEAK_TOLERANCE_HU:
        raise ValueError(
            f"Air peak recovered at {peak:.0f} HU, expected about -1000; "
            "the stored intensity offset differs from the assumed HU+1024"
        )
    return hu, outside, peak


def _body_of(slice_hu: npt.NDArray[np.int32], inside: BoolArray) -> BoolArray | None:
    mask = ndimage.binary_closing((slice_hu > BODY_HU) & inside, np.ones((5, 5), dtype=bool))
    labels, count = ndimage.label(mask)
    if count == 0:
        return None
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    largest = labels == (int(np.argmax(sizes)) + 1)
    filled: BoolArray = ndimage.binary_fill_holes(largest)
    return filled if int(filled.sum()) > 500 else None


def _infer_orientation(
    hu: npt.NDArray[np.int32],
    outside: BoolArray,
    air_peak: float,
) -> OrientationReport:
    """Recover the anatomical meaning of all three axes of a (slice, row, column) stack.

    Which array axis is which is not a free choice. In an axial DICOM the first
    in-plane index runs along the *column* direction cosine, so it is the
    anteroposterior axis, and the second runs left-right. Reading them the other
    way round makes "the lung with the larger row centroid" a statement about
    depth rather than side, and it silently selects nonsense.

    Front/back: the spine is the densest structure and it lies posterior.
    Right/left: the heart displaces the left lung, so the right side holds more
    air, summed over the whole stack rather than one slice.
    Head/foot: two cues that must agree — the caudal half of the lung holds more
    air, and the diaphragm truncates it abruptly while the apex tapers away.
    """
    slices = hu.shape[0]
    air_per_slice = np.zeros(slices, dtype=np.int64)
    low_air = high_air = 0
    bone_rows: list[float] = []
    body_rows: list[float] = []
    for index in range(slices):
        body = _body_of(hu[index], ~outside[index])
        if body is None:
            continue
        air = body & (hu[index] < AIR_HU)
        air_per_slice[index] = int(air.sum())
        # Columns are the left-right axis, so the side split happens on axis 1.
        rows, columns = np.where(air)
        if columns.size:
            occupied = np.flatnonzero(body.any(axis=0))
            midline = 0.5 * (float(occupied[0]) + float(occupied[-1]))
            low_air += int((columns < midline).sum())
            high_air += int((columns >= midline).sum())
        bone = body & (hu[index] > SPINE_HU)
        if int(bone.sum()) > MIN_BONE_PX:
            bone_rows.append(float(np.where(bone)[0].mean()))
            spanned = np.flatnonzero(body.any(axis=1))
            body_rows.append(0.5 * (float(spanned[0]) + float(spanned[-1])))

    total_air = low_air + high_air
    asymmetry = (high_air - low_air) / total_air if total_air else 0.0
    if not bone_rows:
        raise ValueError("No bone found: the anteroposterior axis cannot be oriented")
    spine_offset = float(np.mean(bone_rows) - np.mean(body_rows))

    if air_per_slice.max() == 0:
        raise ValueError("No aerated lung found: orientation cannot be recovered")
    lung = np.flatnonzero(air_per_slice > 0.25 * air_per_slice.max())
    first, last = int(lung[0]), int(lung[-1])
    middle = (first + last) // 2
    caudal_excess = float(
        air_per_slice[first : middle + 1].sum() - air_per_slice[middle:last + 1].sum()
    ) / max(float(total_air), 1.0)

    # How fast the lung disappears at each end: the diaphragm is a wall, the
    # apex is a slope.
    span = max((last - first) // 6, 2)
    low_drop = float(air_per_slice[first : first + span].mean())
    high_drop = float(air_per_slice[last - span + 1 : last + 1].mean())
    peak = float(air_per_slice.max()) or 1.0
    sharpness = (low_drop + 1.0) / (high_drop + 1.0)

    # caudal_excess > 0 means the low-index half holds more air, i.e. it is the
    # base, so superior lies at the high index. The sharper end is likewise the
    # base; both cues are reported so a disagreement is visible.
    _ = peak
    return OrientationReport(
        right_is_high_index=asymmetry > 0,
        lung_asymmetry=asymmetry,
        superior_is_high_index=caudal_excess > 0,
        caudal_air_excess=caudal_excess,
        diaphragm_sharpness_ratio=sharpness,
        posterior_is_high_row=spine_offset > 0,
        spine_offset_px=spine_offset,
        air_peak_hu=air_peak,
    )


def load_copdgene(
    path: Path,
    case_id: str,
    geometry: CaseGeometry | None = None,
    orientation: OrientationReport | None = None,
) -> tuple[IntArray, tuple[float, float, float], OrientationReport]:
    """Read one ``.img`` and return it in RAS+ with its spacing and provenance.

    The returned array follows the same convention as
    :func:`breathgeom.measure.wall.load_ras`: axis 0 towards the patient's
    right, axis 1 anterior, axis 2 superior.

    Pass ``orientation`` from the subject's better-inflated scan to apply the
    same anatomy to both phases; inferring it separately per phase can silently
    mirror or invert one of them relative to the other.
    """
    raw = np.fromfile(path, dtype=np.int16)
    return _decode_copdgene(
        raw, case_id, geometry=geometry, orientation=orientation, source_name=path.name
    )


def _decode_copdgene(
    raw: IntArray,
    case_id: str,
    geometry: CaseGeometry | None = None,
    orientation: OrientationReport | None = None,
    source_name: str | None = None,
) -> tuple[IntArray, tuple[float, float, float], OrientationReport]:
    """Decode raw COPDgene voxels supplied by a file or an archive member."""
    geometry = geometry or DIRLAB_COPDGENE[case_id]
    source_name = source_name or case_id
    voxels_per_slice = geometry.rows * geometry.columns
    if raw.size % voxels_per_slice:
        raise ValueError(f"{source_name}: {raw.size} voxels is not a whole number of slices")
    slices = raw.size // voxels_per_slice
    if slices != geometry.slices:
        raise ValueError(
            f"{source_name}: file holds {slices} slices, published table says {geometry.slices}"
        )

    stack = raw.reshape(slices, geometry.rows, geometry.columns)
    hu, outside, air_peak = _to_hu(stack)
    hu[outside] = -1024  # outside the field is air, not something denser than bone
    report = orientation or _infer_orientation(hu, outside, air_peak)

    # (slice, row=AP, column=LR) -> (LR, AP, SI), then point every axis the way
    # RAS+ requires: right, anterior, superior.
    volume = np.transpose(hu, (2, 1, 0))
    if not report.right_is_high_index:
        volume = volume[::-1, :, :]
    if report.posterior_is_high_row:
        volume = volume[:, ::-1, :]
    if not report.superior_is_high_index:
        volume = volume[:, :, ::-1]

    spacing = (geometry.spacing_x_mm, geometry.spacing_y_mm, geometry.spacing_z_mm)
    return np.ascontiguousarray(volume, dtype=np.int16), spacing, report


def _read_locator(locator: str) -> bytes:
    if not locator.startswith("zip://"):
        return Path(locator).read_bytes()
    archive_text, separator, member = locator.removeprefix("zip://").partition("!/")
    if not separator or not archive_text or not member:
        raise ValueError(f"Malformed ZIP locator: {locator}")
    with ZipFile(Path(archive_text)) as archive:
        return archive.read(member)


def load_copdgene_locator(
    locator: str,
    case_id: str,
    geometry: CaseGeometry | None = None,
    orientation: OrientationReport | None = None,
) -> tuple[IntArray, tuple[float, float, float], OrientationReport]:
    """Read a COPDgene image from an extracted path or ``zip://...!/member``."""
    if not locator.startswith("zip://"):
        return load_copdgene(
            Path(locator), case_id, geometry=geometry, orientation=orientation
        )
    raw = np.frombuffer(_read_locator(locator), dtype=np.int16).copy()
    return _decode_copdgene(
        raw, case_id, geometry=geometry, orientation=orientation, source_name=locator
    )


def load_copdgene_landmarks(
    locator: str,
    case_id: str,
    orientation: OrientationReport,
    *,
    indexing: Literal["one_based", "zero_based"] = "one_based",
    geometry: CaseGeometry | None = None,
) -> FloatArray:
    """Read DIR-Lab ``x y z`` landmarks and convert them to RAS+ millimetres."""
    geometry = geometry or DIRLAB_COPDGENE[case_id]
    text = _read_locator(locator).decode("ascii")
    points = np.loadtxt(text.splitlines(), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{case_id}: expected an N x 3 landmark table")
    if indexing == "one_based":
        points -= 1.0

    if not orientation.right_is_high_index:
        points[:, 0] = geometry.columns - 1 - points[:, 0]
    if orientation.posterior_is_high_row:
        points[:, 1] = geometry.rows - 1 - points[:, 1]
    if not orientation.superior_is_high_index:
        points[:, 2] = geometry.slices - 1 - points[:, 2]
    spacing = np.array(
        [geometry.spacing_x_mm, geometry.spacing_y_mm, geometry.spacing_z_mm],
        dtype=np.float64,
    )
    converted: FloatArray = points * spacing
    return converted


__all__ = [
    "COPDGENE_REQUIRED_SUFFIXES",
    "DIRLAB_COPDGENE",
    "COPDgeneCaseInventory",
    "CaseGeometry",
    "OrientationReport",
    "inventory_copdgene",
    "load_copdgene",
    "load_copdgene_landmarks",
    "load_copdgene_locator",
]
