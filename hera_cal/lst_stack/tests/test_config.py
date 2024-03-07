from .. import config
import numpy as np
import pytest
from .conftest import create_small_array_uvd
from ...tests import mock_uvdata as mockuvd
from hypothesis import given, strategies as st, assume
from pyuvdata.uvdata import FastUVH5Meta
from pathlib import Path


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


all_seasons = [
    'redavg',
    'redavg_inpaint',
    'notredavg',
]


class TestLSTBinConfiguration:
    def get_config(self, season, request):
        return config.LSTBinConfiguration(request.getfixturevalue(f"season_{season}"))

    @pytest.mark.parametrize('season', all_seasons)
    def test_datameta(self, season, request):
        cfg = self.get_config(season, request)
        assert isinstance(cfg.datameta, FastUVH5Meta)

    @pytest.mark.parametrize('season', all_seasons)
    def test_get_earliest_jd_in_set(self, season, request):
        cfg = self.get_config(season, request)

        earliest = cfg.get_earliest_jd_in_set()
        assert isinstance(earliest, float)
        all_metas = [FastUVH5Meta(f) for fls in cfg.data_files for f in fls]
        all_times = [meta.times[0] for meta in all_metas]
        assert earliest == min(all_times)

    @pytest.mark.parametrize('season', all_seasons)
    def test_lst_grid(self, season, request):
        cfg = self.get_config(season, request)
        assert cfg.lst_grid is not None
        assert len(cfg.lst_grid) == 24
        assert np.all(np.diff(cfg.lst_grid) > 0)

    @pytest.mark.parametrize('season', all_seasons)
    def test_lst_grid_edges(self, season, request):
        cfg = self.get_config(season, request)
        assert len(cfg.lst_grid_edges) == len(cfg.lst_grid) + 1

    @pytest.mark.parametrize('season', all_seasons)
    def test_get_file_lst_edges(self, season, request):
        cfg = self.get_config(season, request)
        edges = cfg.get_file_lst_edges()
        assert len(edges) == 13  # one for each file, each of which has 2 bins, and one at the last edge.
        assert np.all(np.diff(edges) > 0)

    @pytest.mark.parametrize('season', all_seasons)
    def test_nfiles(self, season, request):
        cfg = self.get_config(season, request)
        assert cfg.nfiles > 0

    @pytest.mark.parametrize('season', all_seasons)
    def test_red_avg(self, season, request):
        cfg = self.get_config(season, request)
        assert cfg.is_redundantly_averaged == ('notredavg' not in season)

    @pytest.mark.parametrize('season', all_seasons)
    def test_get_matched_files(self, season, request):
        cfg = self.get_config(season, request)
        matched_files = cfg.get_matched_files()

        assert isinstance(matched_files, list)
        assert len(matched_files) == 12  # LSTs
        assert isinstance(matched_files[0], list)
        assert all(len(m) <= 3 for m in matched_files)  # number of nights
        assert isinstance(matched_files[0][0], list)
        assert len(matched_files[0][0]) in {0, 1}  # At most one file corresponds to each bin

    @pytest.mark.parametrize('season', all_seasons)
    def test_create_config(self, season, request):
        cfg = self.get_config(season, request)
        mf = cfg.get_matched_files()
        cfg = cfg.create_config(mf)

        assert len(cfg.matched_files) == 4  # 2 outfiles
        assert len(cfg.matched_files[0]) == 3  # 3 nights
        assert isinstance(cfg.matched_files[0][0], list)
        assert isinstance(cfg.matched_files[0][0][0], Path)


class TestLSTConfig:
    def get_lstconfig(self, season: str, request) -> config.LSTConfig:
        cfg = config.LSTBinConfiguration(request.getfixturevalue(f"season_{season}"))
        mf = cfg.get_matched_files()
        return cfg.create_config(mf)

    @pytest.mark.parametrize('season', all_seasons)
    def test_lst_grid_edges(self, season, request):
        cfg = self.get_lstconfig(season, request)
        assert len(cfg.lst_grid_edges[0]) == len(cfg.lst_grid[0]) + 1

    @pytest.mark.parametrize('season', all_seasons)
    def test_read_write_roundtrip(
        self,
        season,
        tmp_path, request
    ):
        cfg = self.get_lstconfig(season, request)
        fl = tmp_path / "lstconfig.h5"
        cfg.write(fl)
        new_config = config.LSTConfig.from_file(fl)
        assert new_config.config == cfg.config
        assert new_config == cfg

    @pytest.mark.parametrize('season', all_seasons)
    def test_at_single_outfile(self, season, request):
        cfg = self.get_lstconfig(season, request)
        cfgout = cfg.at_single_outfile(outfile=0)

        assert len(cfgout.matched_metas) == 3  # number of nights * lst bins per outfile
        assert len(cfgout.time_indices) == len(cfgout.matched_metas)
        assert all(isinstance(x, np.ndarray) for x in cfgout.time_indices)
        assert len(cfgout.lst_grid_edges) == 3  # 2 bins in the file, so 3 edges
        assert cfgout.n_lsts == 2
        lsts = cfgout.get_lsts()
        assert len(lsts) == len(cfgout.matched_metas)
        assert all(np.all((lst > cfgout.lst_grid_edges[0]) & (lst < cfgout.lst_grid_edges[-1])) for lst in lsts)
