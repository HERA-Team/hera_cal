from .. import io
from pathlib import Path
import pytest
from ..config import _nested_list_of
from pyuvdata.uvdata import FastUVH5Meta, UVData
import numpy as np


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

    def test_empty_list(self):
        out = io.apply_filename_rules([], [(".uvh5", ".calfile")])
        assert out == []


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


@pytest.fixture(scope="module")
def redavg_metas(season_redavg):
    return _nested_list_of(FastUVH5Meta)(season_redavg)


@pytest.fixture(scope="module")
def uvd_template(season_redavg):
    return UVData.from_file(season_redavg[0][0], use_future_array_shapes=True)


class TestFilterRequiredFilesByTimes:
    def test_lstmin_larger_than_max(self, redavg_metas):

        tinds, lsts, files, cals, inp = io.filter_required_files_by_times(
            lst_range=(np.pi + 0.01, np.pi - 0.01),
            data_metas=redavg_metas
        )
        # should have all files.
        assert len(files) == sum(len(m) for m in redavg_metas)


class TestCreateLSTBinOutputFile:
    def test_passing_kind(self, uvd_template, tmp_path):
        out = io.create_lstbin_output_file(
            uvd_template, outdir=tmp_path, fname="{kind}.uvh5",
            kind="test"
        )

        assert "test" in out.name

    def test_poorly_formed_fname(self, uvd_template, tmp_path):
        io.create_lstbin_output_file(
            uvd_template, outdir=tmp_path, fname="/thisfile.uvh5", kind='yoho'
        )

        assert (tmp_path / 'thisfile.uvh5').exists()

    def test_dir_not_existing(self, uvd_template, tmp_path):
        io.create_lstbin_output_file(
            uvd_template, outdir=tmp_path / "non_existent", fname="{kind}.uvh5", kind='yoho'
        )

        assert (tmp_path / 'non_existent' / 'yoho.uvh5').exists()

    def test_no_overwrite(self, uvd_template, tmp_path):
        io.create_lstbin_output_file(
            uvd_template, outdir=tmp_path, fname="test.uvh5", kind='yoho'
        )

        with pytest.raises(FileExistsError):
            io.create_lstbin_output_file(
                uvd_template, outdir=tmp_path, fname="test.uvh5", kind='yoho'
            )
