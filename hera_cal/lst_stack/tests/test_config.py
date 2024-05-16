from .. import config
import numpy as np
import pytest
from .conftest import create_small_array_uvd
from ...tests import mock_uvdata as mockuvd
from hypothesis import given, strategies as st
from pyuvdata.uvdata import FastUVH5Meta
from pathlib import Path
import toml
import attrs
import re
import copy


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

    def test_bad_input(self):
        with pytest.raises(ValueError, match="lst_width must be greater than dlst"):
            config.make_lst_grid(dlst=0.1, lst_width=0.05)

    def test_default_begin_lst(self):
        grid = config.make_lst_grid(dlst=0.01)
        dlst = grid[1] - grid[0]
        assert np.isclose(grid[0], 0.0 + dlst / 2)


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
    'redavg_irregular'
]


class TestLSTBinConfigurator:
    def get_config(self, season, request):
        return config.LSTBinConfigurator(
            request.getfixturevalue(f"season_{season}"),
            where_inpainted_file_rules=[(".uvh5", ".where_inpainted.h5")] if 'inpaint' in season else None
        )

    def test_bad_inputs(self, season_redavg):
        with pytest.raises(ValueError, match="LST must be between 0 and 2pi"):
            config.LSTBinConfigurator(season_redavg, lst_start=-1)

        with pytest.raises(ValueError, match="LST must be between 0 and 2pi"):
            config.LSTBinConfigurator(season_redavg, lst_start=3 * np.pi)

        with pytest.raises(ValueError, match="calfile_rules must be a list of tuples of length 2"):
            config.LSTBinConfigurator(season_redavg, calfile_rules=[1, 2, 3])

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

    @pytest.mark.parametrize("form", ['file', 'str', 'dict'])
    def test_from_toml_list(self, season_redavg, form, tmp_path):
        dct = {
            'datadir': str(Path(season_redavg[0][0]).parent.parent),
            'datafiles': [f"{Path(d[0]).parent.name}/*" for d in season_redavg]
        }

        if form == 'str':
            x = toml.dumps(dct)
        elif form == 'file':
            x = tmp_path / 'a.toml'
            with x.open('w') as _fl:
                toml.dump(dct, _fl)
        else:
            x = dct

        cfg = config.LSTBinConfigurator.from_toml(x)
        assert [str(fl) for fl in cfg.data_files[0]] == season_redavg[0]
        assert len(cfg.data_files) == len(season_redavg)

    def test_from_toml_bad_input(self):
        with pytest.raises(ValueError, match="toml_file must be a valid path, toml-serialized string, or a dictionary."):
            config.LSTBinConfigurator.from_toml(3)

        with pytest.raises(ValueError, match='datafiles must be specified'):
            config.LSTBinConfigurator.from_toml({})

        with pytest.raises(ValueError, match="No data files found"):
            config.LSTBinConfigurator.from_toml({'datafiles': [], 'datadir': '.'})

        with pytest.raises(ValueError, match="No data files found"):
            config.LSTBinConfigurator.from_toml({'datafiles': ["non-existent/*", "other-non-exists*"], 'datadir': '.'})

    def test_from_toml_datafile_dict(self, tmp_path):
        # Make a few files to find.
        allfiles = []
        nights = [str(d) for d in range(2459811, 2459815)]
        for night in nights:
            ndir = tmp_path / night
            ndir.mkdir()
            this = []
            allfiles.append(this)

            for fl in range(3):
                _fl = (ndir / f'file-{fl}.uvh5')
                _fl.touch()
                this.append(_fl)

        dct = {
            'nlsts_per_file': 2,
            'dlst': 0.01,
            'datafiles': {
                'datadir': tmp_path,
                'nights': nights,
                'fileglob': '{night}/file-?.uvh5'
            }
        }
        cfg = config.LSTBinConfigurator.from_toml(dct)
        assert cfg.data_files == allfiles


class TestLSTConfig:
    def get_lstconfig(self, season: str, request) -> config.LSTConfig:
        cfg = config.LSTBinConfigurator(
            request.getfixturevalue(f"season_{season}"),
            where_inpainted_file_rules=[(".uvh5", ".where_inpainted.h5")] if 'inpaint' in season else None
        )
        mf = cfg.get_matched_files()
        return cfg.create_config(mf)

    def test_bad_parameters(self, season_redavg):
        cfg = config.LSTBinConfigurator(season_redavg)
        mf = cfg.get_matched_files()
        good_cfg = cfg.create_config(mf)

        with pytest.raises(ValueError, match='lst_grid must be a 1D or 2D array'):
            attrs.evolve(good_cfg, lst_grid=np.arange(16).reshape((2, 2, 2, 2)))

        with pytest.raises(ValueError, match=re.escape("lst_grid must have shape (n_output_files, nlsts_per_file)")):
            attrs.evolve(good_cfg, lst_grid=np.arange(16).reshape((2, 8)))

        with pytest.raises(ValueError, match="matched_files must be a list with one entry per output file"):
            attrs.evolve(good_cfg, matched_files=good_cfg.matched_files[:-1])

        with pytest.raises(ValueError, match="each list in matched_files should be n_nights long"):
            attrs.evolve(good_cfg, matched_files=[mmf + [mmf[-1]] for mmf in good_cfg.matched_files])

        with pytest.raises(ValueError, match="antpairs must be a list of tuples of length 2"):
            attrs.evolve(good_cfg, antpairs=[(1, 1, 2.5)])

        with pytest.raises(ValueError, match="Autos must have the same antenna number on both sides"):
            attrs.evolve(good_cfg, autopairs=[(1, 2)])

        with pytest.raises(ValueError, match="calfiles must have the same shape as matched_files"):
            attrs.evolve(good_cfg, calfiles=good_cfg.matched_files[:-1])

        with pytest.raises(ValueError, match="does not exist"):
            cf = copy.deepcopy(good_cfg.matched_files)
            cf[-1][-1][-1] = Path("non-existent.file")
            attrs.evolve(good_cfg, calfiles=cf)

        with pytest.raises(ValueError, match="calfiles has a different shape than matched_files"):
            cf = [[[[fl, fl] for fl in night] for night in outfl] for outfl in good_cfg.matched_files]
            attrs.evolve(good_cfg, calfiles=cf)

        with pytest.raises(ValueError, match='pols must be a list of strings'):
            attrs.evolve(good_cfg, pols=[1, 2, 3])

        with pytest.raises(ValueError, match='pols must have at most 4 elements'):
            attrs.evolve(good_cfg, pols=['xx', 'yy', 'xy', 'yx', 'extra'])

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

        with pytest.raises(ValueError, match='Either lst or outfile must be specified'):
            cfg.at_single_outfile()

        with pytest.raises(ValueError, match='Only one of lst or outfile can be specified'):
            cfg.at_single_outfile(outfile=0, lst=0)

        cfgout = cfg.at_single_outfile(outfile=0)
        assert len(cfgout.matched_metas) == 1 if season == 'redavg_irregular' else 3  # number of nights
        assert len(cfgout.time_indices) == len(cfgout.matched_metas)
        assert all(isinstance(x, np.ndarray) for x in cfgout.time_indices)
        assert len(cfgout.lst_grid_edges) == 3  # 2 bins in the file, so 3 edges
        assert cfgout.n_lsts == 2
        lsts = cfgout.get_lsts()
        assert len(lsts) == len(cfgout.matched_metas)
        assert all(np.all((lst > cfgout.lst_grid_edges[0]) & (lst < cfgout.lst_grid_edges[-1])) for lst in lsts)

        getlst = np.mean(cfgout.lst_grid)
        cfgnew = cfg.at_single_outfile(lst=getlst)
        assert cfgnew == cfgout

    def test_at_single_bin(self, request):
        cfg = self.get_lstconfig('redavg', request)

        with pytest.raises(ValueError, match='Either lst or bin_index must be specified'):
            cfg.at_single_bin()

        with pytest.raises(ValueError, match='Only one of lst or bin_index can be specified'):
            cfg.at_single_bin(bin_index=0, lst=0)

        cfgout = cfg.at_single_bin(bin_index=0)
        assert len(cfgout.matched_metas) == 3  # number of nights
        assert len(cfgout.time_indices) == len(cfgout.matched_metas)
        assert all(isinstance(x, np.ndarray) for x in cfgout.time_indices)
        assert len(cfgout.lst_grid_edges) == 2  # 2 bins in the file, so 3 edges
        assert cfgout.n_lsts == 1
        lsts = cfgout.get_lsts()
        assert len(lsts) == len(cfgout.matched_metas)
        assert all(np.all((lst > cfgout.lst_grid_edges[0]) & (lst < cfgout.lst_grid_edges[-1])) for lst in lsts)

        getlst = np.mean(cfgout.lst_grid)
        cfgnew = cfg.at_single_bin(lst=getlst)
        assert cfgnew == cfgout

    def test_bad_matched_files_config_single(self, request):
        cfg = self.get_lstconfig('redavg', request).at_single_bin(bin_index=0)

        with pytest.raises(ValueError, match='matched_files must be a list of Path objects'):
            attrs.evolve(cfg, matched_files=[[fl] for fl in cfg.matched_files])

    def test_write_none_property(self, request, tmp_path):
        cfg = self.get_lstconfig('redavg', request)

        cfg.properties['nonetype'] = None

        cfg.write(tmp_path / 'outfile.h5')
        new = config.LSTConfig.from_file(tmp_path / 'outfile.h5')

        assert new.properties['nonetype'] is None

    def test_write_bad_property(self, request, tmp_path):
        cfg = self.get_lstconfig('redavg', request)

        cfg.properties['badprop'] = cfg

        with pytest.raises(ValueError, match='Cannot write attribute badprop'):
            cfg.write(tmp_path / 'somefile.h5')


@pytest.fixture(scope='module')
def redavg_configurator(season_redavg):
    return config.LSTBinConfigurator(season_redavg)


@pytest.fixture(scope='module')
def redavg_single(season_redavg, redavg_configurator):
    mf = redavg_configurator.get_matched_files()
    return redavg_configurator.create_config(mf).at_single_bin(bin_index=0)


class TestLSTConfigSingle:
    def test_default_time_indices(self, redavg_configurator, redavg_single):
        new = config.LSTConfigSingle(
            config=redavg_configurator,
            lst_grid=redavg_single.lst_grid,
            matched_files=redavg_single.matched_files,
            autopairs=redavg_single.autopairs,
            antpairs=redavg_single.antpairs,
            pols=redavg_single.pols,
        )

        assert all(np.allclose(ttnew, tt) for ttnew, tt in zip(new.time_indices, redavg_single.time_indices))

    def test_bad_time_indices(self, redavg_single):
        with pytest.raises(ValueError, match='time_indices must have the same length as matched_metas'):
            attrs.evolve(redavg_single, time_indices=np.arange(len(redavg_single.matched_files) - 1))

        with pytest.raises(ValueError, match='time_indices must be integer arrays'):
            attrs.evolve(redavg_single, time_indices=[np.linspace(0, 1, 10) for _ in redavg_single.matched_files])

        with pytest.raises(ValueError, match='time_indices must be shorter than the LSTs in the file'):
            tidx = copy.deepcopy(redavg_single.time_indices)
            tidx[-1] += 1000
            attrs.evolve(redavg_single, time_indices=tidx)
