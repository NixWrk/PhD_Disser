from pathlib import Path
from zipfile import ZipFile

from breathgeom.io.dirlab import COPDGENE_REQUIRED_SUFFIXES
from breathgeom.io.pairs import (
    add_source_checksums,
    inventory_copdgene_pairs,
    inventory_lungct_pairs,
    read_pair_manifest,
    write_pair_manifest,
)


def test_copdgene_inventory_uses_zip_members_without_extraction(tmp_path: Path) -> None:
    with ZipFile(tmp_path / "copd1.zip", "w") as archive:
        for suffix in COPDGENE_REQUIRED_SUFFIXES:
            archive.writestr(f"nested/copd1{suffix}", b"test")

    rows = inventory_copdgene_pairs(tmp_path)

    assert len(rows) == 10
    assert rows[0].complete
    assert rows[0].fixed_image.startswith("zip://")
    assert rows[0].fixed_image.endswith("!/nested/copd1_eBHCT.img")
    assert rows[0].has_expert_landmarks
    assert not rows[1].complete


def test_lungct_inventory_keeps_expert_landmarks_distinct_from_keypoints(
    tmp_path: Path,
) -> None:
    base = tmp_path / "extracted" / "LungCT"
    base.mkdir(parents=True)
    (base / "LungCT_dataset.json").write_text(
        '{"img_shift":{"fixed":"exhale","moving":"inhale"}}', encoding="utf-8"
    )
    for folder in ("imagesTr", "masksTr", "landmarksTr", "keypointsTr"):
        (base / folder).mkdir()
    for phase in ("0000", "0001"):
        name = f"LungCT_0001_{phase}"
        (base / "imagesTr" / f"{name}.nii.gz").touch()
        (base / "masksTr" / f"{name}.nii.gz").touch()
        (base / "keypointsTr" / f"{name}.csv").touch()

    without_expert = inventory_lungct_pairs(tmp_path)[0]
    assert without_expert.complete
    assert without_expert.has_keypoints
    assert not without_expert.has_expert_landmarks

    for phase in ("0000", "0001"):
        (base / "landmarksTr" / f"LungCT_0001_{phase}.csv").touch()
    with_expert = inventory_lungct_pairs(tmp_path)[0]
    assert with_expert.has_expert_landmarks


def test_manifest_has_archive_checksum_and_serialized_missing(tmp_path: Path) -> None:
    archive_path = tmp_path / "copd1.zip"
    with ZipFile(archive_path, "w") as archive:
        for suffix in COPDGENE_REQUIRED_SUFFIXES:
            archive.writestr(f"copd1{suffix}", b"test")
    rows = add_source_checksums(inventory_copdgene_pairs(tmp_path))
    output = tmp_path / "pairs.csv"

    assert write_pair_manifest(output, rows) == 10
    text = output.read_text(encoding="utf-8-sig")
    assert rows[0].source_sha256 in text
    assert "fixed_image;moving_image" in text

    restored = read_pair_manifest(output)
    assert restored == rows
