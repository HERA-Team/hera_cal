from .. import config
import numpy as np
import pytest
from .conftest import create_small_array_uvd
from ...tests import mock_uvdata as mockuvd
from hypothesis import given, strategies as st, assume
from pyuvdata.uvdata import FastUVH5Meta


class TestFixdLST:

    # Happy path tests with various realistic test values
    @given(n=st.integers(1, 1000000))
    def test_happy_path(self, n):
        dlst = 2 * np.pi / n
        result = config._fix_dlst(dlst)

        rng = (2 * np.pi / (n + 1), 2 * np.pi / max(0.1, (n - 1)))
        assert abs(result - dlst) < abs(dlst - rng[0])
        assert abs(result - dlst) < abs(dlst - rng[1])

        assert result == pytest.approx(dlst, rel=1e-6)

    @given(dlst=st.one_of(st.integers(1, 6), st.floats(2 * np.pi / 1000000, 2 * np.pi)))
    def test_output_properties(self, dlst: float):
        # Act
        result = config._fix_dlst(dlst)

        # Assert
        assert isinstance(result, float)
        assert 0 <= result <= 2 * np.pi
        n = 2 * np.pi / result
        rounded = round(n, 0)

        assert abs(rounded - n) < 1e-8

    @given(dlst=st.floats(max_value=2 * np.pi / 1000000 - 1e-10))
    def test_error_too_small(self, dlst: float):
        # Act & Assert
        with pytest.raises(ValueError, match='dlst must be more'):
            config._fix_dlst(dlst)

    @given(dlst=st.floats(min_value=2 * np.pi + 1e-10))
    def test_error_too_large(self, dlst: float):
        # Act & Assert
        with pytest.raises(ValueError, match='dlst must be less'):
            config._fix_dlst(dlst)


class TestMakeLSTGrid:
    @given(
        dlst=st.floats(2 * np.pi / 1000000, 2 * np.pi),
        begin_lst=st.floats(),
        lst_width=st.floats(1e-10, np.inf),
    )
    def test_make_lst_grid(self, dlst, begin_lst, lst_width):
        dlst = config._fix_dlst(dlst)
        lst_width = dlst + lst_width

        lst_grid = config.make_lst_grid(dlst, begin_lst=begin_lst, lst_width=lst_width)
        assert len(lst_grid) > 0
        assert np.all(np.diff(lst_grid) > 0)
        assert len(lst_grid) <= 1000000
        assert np.all(lst_grid >= 0)
        assert lst_grid[0] <= 2 * np.pi
        assert lst_grid[-1] - lst_grid[0] <= 2 * np.pi


class TestGetAllAntpairs:
    @pytest.mark.parametrize("redundantly_averaged", [True, False])
    @pytest.mark.parametrize("only_last_file_per_night", [True, False])
    def test_get_all_baselines(
        self, tmp_path_factory, redundantly_averaged, only_last_file_per_night
    ):
        tmp = tmp_path_factory.mktemp("get_all_baselines")
        uvds = mockuvd.make_dataset(
            ndays=3,
            nfiles=4,
            ntimes=2,
            ants=np.arange(10),
            creator=create_small_array_uvd,
            redundantly_averaged=redundantly_averaged,
        )
        data_files = mockuvd.write_files_in_hera_format(uvds, tmp)

        antpairs, pols = config.get_all_antpairs(
            data_files,
            redundantly_averaged=redundantly_averaged,
            only_last_file_per_night=only_last_file_per_night,
        )

        assert len(antpairs) == len(uvds[0][0].get_antpairs())

    def test_bad_inputs(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("get_all_baselines")
        uvds = mockuvd.make_dataset(
            ndays=3,
            nfiles=4,
            ntimes=2,
            ants=np.arange(10),
            creator=create_small_array_uvd,
            redundantly_averaged=True,
        )
        data_files = mockuvd.write_files_in_hera_format(uvds, tmp)

        with pytest.raises(
            ValueError,
            match="Cannot ignore antennas if the files are redundantly averaged",
        ):
            config.get_all_antpairs(data_files, ignore_ants=[0, 1])

        with pytest.raises(
            ValueError,
            match="Cannot exclude antennas if the files are redundantly averaged",
        ):
            config.get_all_antpairs(
                data_files,
                ex_ant_yaml_files=["non-existent-file.yaml"],
            )

        uvds_different_xorient = mockuvd.make_dataset(
            ndays=1,
            nfiles=4,
            ntimes=2,
            ants=np.arange(10),
            creator=create_small_array_uvd,
            x_orientation="east",
            redundantly_averaged=True,
        )

        data_files = mockuvd.write_files_in_hera_format(
            uvds + uvds_different_xorient, tmp
        )

        with pytest.raises(
            ValueError, match="Not all files have the same xorientation!"
        ):
            config.get_all_antpairs(data_files)


@pytest.fixture(scope="module")
def default_dataset(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("get_all_baselines")
    uvds = mockuvd.make_dataset(
        ndays=3,
        nfiles=4,
        ntimes=2,
        ants=np.arange(10),
        creator=create_small_array_uvd,
        redundantly_averaged=True,
    )
    return mockuvd.write_files_in_hera_format(uvds, tmp)


@pytest.fixture(scope="module")
def default_config(default_dataset):
    return config.LSTBinConfiguration(default_dataset)


class TestLSTBinConfiguration:
    def test_datameta(self, default_config):
        assert isinstance(default_config.datameta, FastUVH5Meta)

    def test_get_earliest_jd_in_set(self, default_config):
        earliest = default_config.get_earliest_jd_in_set()
        assert isinstance(earliest, float)
        all_metas = [FastUVH5Meta(f) for fls in self.data_files for f in fls]
        all_times = [meta.times[0] for meta in all_metas]
        assert earliest == min(all_times)

    def test_lst_grid(self, default_config):
        assert default_config.lst_grid is not None
        assert len(default_config.lst_grid) > 0
        assert np.all(np.diff(default_config.lst_grid) > 0)

    def test_lst_grid_edges(self, default_config):
        assert len(default_config.lst_grid_edges) == len(default_config.lst_grid) + 1

    def test_nfiles(self, default_config):
        assert default_config.nfiles > 0

    def test_red_avg(self, default_config):
        assert default_config.is_redundantly_averaged

    def test_get_matched_files(self, default_config):
        matched_files = default_config.get_matched_files()

        assert isinstance(matched_files, list)
        assert len(matched_files) > 0
        assert isinstance(matched_files[0], list)
        assert len(matched_files[0]) > 0
        assert isinstance(matched_files[0][0], FastUVH5Meta)

    def test_create_config(self, default_config):
        mf = default_config.get_matched_files()
        config = default_config.create_config(mf)

        assert len(config.matched_files) == 4  # 2 outfiles
        assert len(config.matched_files[0]) == 3  # 3 nights
        assert isinstance(config.matched_files[0][0], list)
        assert isinstance(config.matched_files[0][0][0], FastUVH5Meta)
