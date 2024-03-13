from .. import io
from pathlib import Path
import pytest


@pytest.fixture(scope="module")
def file_trove(tmp_path_factory):
    return tmp_path_factory.mktemp("file_trove")


@pytest.fixture(scope='module')
def empty_uvh5(file_trove) -> list[str]:
    datas = [file_trove / f"data{i}.uvh5" for i in range(6)]
    for d in datas:
        d.touch()
    return [str(d) for d in datas]


@pytest.fixture(scope='module')
def empty_calfile(empty_uvh5) -> list[str]:
    cals = [Path(d).with_suffix(".calfile") for d in empty_uvh5]
    for fl in cals:
        fl.touch()
    return [str(c) for c in cals]


class TestApplyFilenameRules:
    def test_single_depth(self, empty_uvh5: list[str], empty_calfile: list[str]):
        calfiles = io.apply_filename_rules(empty_uvh5, [(".uvh5", ".calfile")])
        assert len(calfiles) == len(empty_uvh5)

    def test_single_path(self, empty_uvh5: list[str], empty_calfile: list[str]):
        calfiles = io.apply_filename_rules(empty_uvh5[0], [(".uvh5", ".calfile")])
        assert isinstance(calfiles, str)

    def test_multiple_depth(self, empty_uvh5: list[str], empty_calfile: list[str]):
        calfiles = io.apply_filename_rules(
            [[empty_uvh5, empty_uvh5]],
            [(".uvh5", ".calfile")]
        )
        assert len(calfiles) == 1

    def test_warn_if_missing(self, empty_uvh5: list[str]):
        with pytest.warns(UserWarning, match="does not exist"):
            io.apply_filename_rules(
                empty_uvh5,
                [(".uvh5", ".non_existent.h5")],
                missing="warn"
            )

    def test_raise_if_missing(self, empty_uvh5: list[str]):
        with pytest.raises(IOError, match="does not exist"):
            io.apply_filename_rules(
                empty_uvh5,
                [(".uvh5", ".non_existent.h5")],
                missing="raise"
            )

    def test_ignore_if_missing(self, empty_uvh5: list[str]):
        out = io.apply_filename_rules(
            empty_uvh5,
            [(".uvh5", ".non_existent.h5")],
            missing="ignore"
        )
        assert all(o is None for o in out)


def test_configure_inpainted_mode():
    modes = io._configure_inpainted_mode(
        output_flagged=True, output_inpainted=True, where_inpainted_files=[]
    )
    assert len(modes) == 2

    modes = io._configure_inpainted_mode(
        output_flagged=True, output_inpainted=True, where_inpainted_files=["a_file.h5"]
    )
    assert len(modes) == 2

    modes = io._configure_inpainted_mode(
        output_flagged=True, output_inpainted=False, where_inpainted_files=[]
    )
    assert len(modes) == 1
    assert not modes[0]

    modes = io._configure_inpainted_mode(
        output_flagged=True, output_inpainted=False, where_inpainted_files=["a_file.h5"]
    )
    assert len(modes) == 1
    assert not modes[0]

    modes = io._configure_inpainted_mode(
        output_flagged=True, output_inpainted=None, where_inpainted_files=[]
    )
    assert len(modes) == 1
    assert not modes[0]

    modes = io._configure_inpainted_mode(
        output_flagged=True, output_inpainted=None, where_inpainted_files=["a_file.h5"]
    )
    assert len(modes) == 2

    modes = io._configure_inpainted_mode(
        output_flagged=False, output_inpainted=True, where_inpainted_files=[]
    )
    assert len(modes) == 1
    assert modes[0]

    with pytest.raises(ValueError, match="Both output_inpainted and output_flagged"):
        io._configure_inpainted_mode(
            output_flagged=False, output_inpainted=False, where_inpainted_files=[]
        )
