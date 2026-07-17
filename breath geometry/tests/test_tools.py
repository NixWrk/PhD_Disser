from breathgeom.tools import select_version_line

DCM2NIIX_OUTPUT = (
    "Compression will be faster with C:\\tools\\bin\\pigz.exe "
    "in the same folder as the executable\n"
    "Chris Rorden's dcm2niiX version v1.0.20260416 (JP2:OpenJPEG) (JP-LS:CharLS)\n"
    "v1.0.20260416\n"
)


def test_advisory_line_does_not_mask_version() -> None:
    line = select_version_line(DCM2NIIX_OUTPUT)

    assert line is not None
    assert "v1.0.20260416" in line
    assert "pigz" not in line


def test_plain_version_output() -> None:
    assert select_version_line("4.15.2\n") == "4.15.2"
    assert select_version_line("cmake version 3.30.3\n") == "cmake version 3.30.3"


def test_output_without_version_falls_back_to_first_line() -> None:
    assert select_version_line("\nunknown tool\nsecond line\n") == "unknown tool"


def test_empty_output() -> None:
    assert select_version_line("   \n\n") is None
