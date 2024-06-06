import pytest
from .. import binning
from ...tests import mock_uvdata as mockuvd
import numpy as np
from ...red_groups import RedundantGroups
from pyuvdata import utils as uvutils
from pathlib import Path
from ..config import LSTBinConfigurator
import shutil
from hera_cal.lst_stack.io import apply_filename_rules
from pyuvdata import UVFlag, UVData


class TestAdjustLSTBinEdges:
    @pytest.mark.parametrize(
        'edges', [
            np.linspace(0, 1, 10),
            np.linspace(700, 800, 10),
            np.linspace(-2, 2, 2)
        ]
    )
    def test_properties(self, edges):
        binning.adjust_lst_bin_edges(edges)
        assert 0 <= edges[0] <= 2 * np.pi
        assert np.all(np.diff(edges) > 0)

    def test_errors(self):
        x = np.linspace(1, 0, 10)
        with pytest.raises(ValueError, match='lst_bin_edges must be monotonically increasing'):
            binning.adjust_lst_bin_edges(x)

        x = np.zeros((2, 3))
        with pytest.raises(ValueError, match='lst_bin_edges must be a 1D array'):
            binning.adjust_lst_bin_edges(x)


class TestLSTAlign:
    @classmethod
    def get_lst_align_data(
        cls,
        ntimes: int = 10,
        nfreqs: int = 5,
        npols: int = 4,
        nants: int = 4,
        ndays: int = 1,
        creator: callable = mockuvd.create_uvd_identifiable,
    ):
        uvds = mockuvd.make_dataset(
            ndays=ndays,
            nfiles=1,
            creator=creator,
            pols=("xx", "yy", "xy", "yx")[:npols],
            freqs=np.linspace(100e6, 200e6, nfreqs),
            ants=np.arange(nants),
            ntimes=ntimes,
            time_axis_faster_than_bls=False,
        )
        uvds = [uvd[0] for uvd in uvds]  # flatten to single list

        data = np.concatenate(
            [uvd.data_array.reshape((ntimes, -1, nfreqs, npols)) for uvd in uvds],
            axis=0,
        )
        freq_array = uvds[0].freq_array
        data_lsts = np.concatenate(
            [np.unique(uvds[0].lst_array)] * len(uvds)
        )
        antpairs = uvds[0].get_antpairs()
        antpos = uvds[0].antenna_positions

        # Ensure that each LST is in its own bin
        lsts = np.sort(np.unique(data_lsts))

        lst_bin_edges = np.concatenate(
            (
                [lsts[0] - (lsts[1] - lsts[0]) / 2],
                (lsts[1:] + lsts[:-1]) / 2,
                [lsts[-1] + (lsts[-1] - lsts[-2]) / 2],
            )
        )

        return dict(
            data=data,
            data_lsts=data_lsts,
            antpairs=antpairs,
            lst_bin_edges=lst_bin_edges,
            freq_array=freq_array,
            antpos=antpos,
        )

    def test_bad_inputs(self):
        # Test that we get the right errors for bad inputs
        kwargs = self.get_lst_align_data()

        def lst_align_with(**kw):
            return binning.lst_align(**{**kwargs, **kw})

        data = np.ones(kwargs["data"].shape[:-1] + (5,))
        with pytest.raises(ValueError, match="data has more than 4 pols"):
            lst_align_with(data=data)

        with pytest.raises(ValueError, match="data should have shape"):
            lst_align_with(freq_array=kwargs["freq_array"][:-1])

        with pytest.raises(ValueError, match="flags should have shape"):
            lst_align_with(flags=np.ones(13, dtype=bool))

        # Make a wrong-shaped nsample array.
        with pytest.raises(ValueError, match="nsamples should have shape"):
            lst_align_with(nsamples=np.ones(13))

        # Use only one bin edge
        with pytest.raises(
            ValueError, match="lst_bin_edges must have at least 2 elements"
        ):
            lst_align_with(lst_bin_edges=np.ones(1))

        # Try rephasing without freq_array or antpos
        with pytest.raises(
            ValueError, match="freq_array and antpos is needed for rephase"
        ):
            lst_align_with(rephase=True, antpos=None)

    def test_increasing_lsts_one_per_bin(self):
        kwargs = self.get_lst_align_data(ntimes=6)
        bins, d, f, n, inp = binning.lst_align(rephase=False, **kwargs)

        # We should not be changing the data at all.
        d = np.squeeze(np.asarray(d))
        f = np.squeeze(np.asarray(f))
        n = np.squeeze(np.asarray(n))

        np.testing.assert_allclose(d, kwargs["data"])
        assert not np.any(f)
        assert np.all(n == 1.0)
        assert len(bins) == 6

    def test_multi_days_one_per_bin(self):
        kwargs = self.get_lst_align_data(ndays=2, ntimes=7)
        bins, d, f, n, inp = binning.lst_align(rephase=False, **kwargs)

        # We should not be changing the data at all.
        d = np.squeeze(np.asarray(d))
        f = np.squeeze(np.asarray(f))
        n = np.squeeze(np.asarray(n))

        np.testing.assert_allclose(d[:, 0], kwargs["data"][:7])
        assert not np.any(f)
        assert np.all(n == 1.0)
        assert len(bins) == 7

    def test_multi_days_with_flagging(self):
        kwargs = self.get_lst_align_data(ndays=2, ntimes=7)

        # Flag everything after the first day, and make the data there crazy.
        flags = np.zeros_like(kwargs["data"], dtype=bool)
        flags[7:] = True
        kwargs["data"][7:] = 1000.0

        bins, d, f, n, inp = binning.lst_align(rephase=False, flags=flags, **kwargs)

        d = np.squeeze(np.asarray(d))
        f = np.squeeze(np.asarray(f))
        n = np.squeeze(np.asarray(n))

        np.testing.assert_allclose(d[:, 0], kwargs["data"][:7])
        assert not np.any(f[:, 0])
        assert np.all(f[:, 1])
        assert len(bins) == 7

    def test_multi_days_with_nsamples_zero(self):
        kwargs = self.get_lst_align_data(ndays=2, ntimes=7)

        # Flag everything after the first day, and make the data there crazy.
        nsamples = np.ones_like(kwargs["data"], dtype=float)
        nsamples[7:] = 0.0
        kwargs["data"][7:] = 1000.0

        bins, d, f, n, inp = binning.lst_align(
            rephase=False, nsamples=nsamples, **kwargs
        )

        d = np.squeeze(np.asarray(d))
        f = np.squeeze(np.asarray(f))
        n = np.squeeze(np.asarray(n))

        np.testing.assert_allclose(d[:, 0], kwargs["data"][:7])
        assert not np.any(f)
        assert np.all(n[:, 0] == 1.0)
        assert np.all(n[:, 1] == 0.0)
        assert len(bins) == 7

    def test_rephase(self):
        """Test that rephasing where each bin is already at center does nothing."""
        kwargs = self.get_lst_align_data(ntimes=7)

        bins0, d0, f0, n0, inp = binning.lst_align(rephase=True, **kwargs)
        bins, d, f, n, inp = binning.lst_align(rephase=False, **kwargs)
        np.testing.assert_allclose(d, d0, rtol=1e-6)
        np.testing.assert_allclose(f, f0, rtol=1e-6)
        np.testing.assert_allclose(n, n0, rtol=1e-6)
        assert len(bins) == len(bins0)

    def test_lstbinedges_modulus(self):
        kwargs = self.get_lst_align_data(ntimes=7)
        edges = kwargs.pop("lst_bin_edges")

        lst_bin_edges = edges.copy()
        lst_bin_edges -= 4 * np.pi

        bins, d0, f0, n0, inp = binning.lst_align(lst_bin_edges=lst_bin_edges, **kwargs)

        lst_bin_edges = edges.copy()
        lst_bin_edges += 4 * np.pi

        bins, d, f, n, inp = binning.lst_align(
            lst_bin_edges=lst_bin_edges, **kwargs
        )

        np.testing.assert_allclose(d, d0)
        np.testing.assert_allclose(f, f0)
        np.testing.assert_allclose(n, n0)

        with pytest.raises(
            ValueError, match="lst_bin_edges must be monotonically increasing."
        ):
            binning.lst_align(lst_bin_edges=lst_bin_edges[::-1], **kwargs)


@pytest.mark.filterwarnings("ignore", message="Getting antpos from the first file only")
class TestLSTBinFilesForBaselines:
    def test_defaults(self, uvd, uvd_file):
        _, d0, f0, n0, inpflg, _, _ = binning.lst_bin_files_for_baselines(
            data_files=[uvd_file],
            lst_bin_edges=[uvd.lst_array.min(), uvd.lst_array.max() + 0.01],
            antpairs=uvd.get_antpairs(),
            rephase=False,
        )

        _, d, f, n, inpflg, _, _ = binning.lst_bin_files_for_baselines(
            data_files=[uvd_file],
            lst_bin_edges=[uvd.lst_array.min(), uvd.lst_array.max() + 0.01],
            antpairs=uvd.get_antpairs(),
            freqs=uvd.freq_array,
            pols=uvd.polarization_array,
            time_idx=[np.ones(uvd.Ntimes, dtype=bool)],
            # time_arrays=[np.unique(uvd.time_array)],
            lsts=np.unique(uvd.lst_array),
            rephase=False,
        )

        np.testing.assert_allclose(d0, d)
        np.testing.assert_allclose(f0, f)
        np.testing.assert_allclose(n0, n)
        # np.testing.assert_allclose(times0, times)

    def test_empty(self, uvd, uvd_file):
        f0 = binning.lst_bin_files_for_baselines(
            data_files=[uvd_file],
            lst_bin_edges=[uvd.lst_array.min(), uvd.lst_array.max()],
            antpairs=[(127, 128)],
            rephase=True,
        )[2]

        assert np.all(f0)

    def test_extra(self, uvd, uvd_file):
        # Providing baselines that don't exist in the file is fine, they're just ignored.
        f0 = binning.lst_bin_files_for_baselines(
            data_files=[uvd_file],
            lst_bin_edges=[uvd.lst_array.min(), uvd.lst_array.max()],
            antpairs=uvd.get_antpairs() + [(127, 128)],
            rephase=True,
        )[2]

        # last baseline is the extra one that's all flagged.
        assert np.all(f0[0][:, -1])

    def test_freqrange(self, uvd, uvd_file, uvc_file):
        """Test that providing freq_range works."""
        data = binning.lst_bin_files_for_baselines(
            data_files=[uvd_file],
            lst_bin_edges=[uvd.lst_array.min(), uvd.lst_array.max()],
            cal_files=[uvc_file],
            freq_min=153e6,
            freq_max=158e6,
            antpairs=uvd.get_antpairs(),
        )[1]

        assert data[0].shape[-2] < uvd.Nfreqs

    def test_bad_pols(self, uvd, uvd_file):
        with pytest.raises(KeyError, match="7"):
            binning.lst_bin_files_for_baselines(
                data_files=[uvd_file],
                lst_bin_edges=[uvd.lst_array.min(), uvd.lst_array.max()],
                pols=[3.0, 7, -1],
                antpairs=uvd.get_antpairs(),
            )

    def test_incorrect_red_input(self, uvd, uvd_file, uvc_file):
        with pytest.raises(
            ValueError, match="reds must be provided if redundantly_averaged is True"
        ):
            binning.lst_bin_files_for_baselines(
                data_files=[uvd_file],
                lst_bin_edges=[uvd.lst_array.min(), uvd.lst_array.max()],
                redundantly_averaged=True,
                antpairs=uvd.get_antpairs(),
            )

        with pytest.raises(
            ValueError, match="Cannot apply calibration if redundantly_averaged is True"
        ):
            binning.lst_bin_files_for_baselines(
                data_files=[uvd_file],
                lst_bin_edges=[uvd.lst_array.min(), uvd.lst_array.max()],
                cal_files=[uvc_file],
                redundantly_averaged=True,
                reds=RedundantGroups.from_antpos(
                    dict(zip(uvd.antenna_numbers, uvd.antenna_positions)),
                    pols=uvutils.polnum2str(
                        uvd.polarization_array, x_orientation=uvd.x_orientation
                    ),
                ),
                antpairs=uvd.get_antpairs(),
            )

    def test_simple_redundant_averaged_file(self, uvd_redavg, uvd_redavg_file):
        d0 = binning.lst_bin_files_for_baselines(
            data_files=[uvd_redavg_file],
            lst_bin_edges=[
                uvd_redavg.lst_array.min() - 0.1,
                uvd_redavg.lst_array.max() + 0.1,
            ],
            redundantly_averaged=True,
            rephase=False,
            antpairs=uvd_redavg.get_antpairs(),
            reds=RedundantGroups.from_antpos(
                dict(zip(uvd_redavg.antenna_numbers, uvd_redavg.antenna_positions)),
            ),
        )[1]

        assert len(d0) == 1
        assert d0[0].shape == (
            uvd_redavg.Ntimes,
            uvd_redavg.Nbls,
            uvd_redavg.Nfreqs,
            uvd_redavg.Npols,
        )


class TestLSTBinFilesFromConfig:
    def get_config(self, season, request, outfile_index=0):
        cfg = LSTBinConfigurator(
            request.getfixturevalue(f"season_{season}"),
            where_inpainted_file_rules=[(".uvh5", ".where_inpainted.h5")] if 'inpaint' in season else None,
        )
        mf = cfg.get_matched_files()
        return cfg.create_config(mf).at_single_outfile(outfile=outfile_index)

    @pytest.mark.parametrize('season', ['redavg', 'notredavg', 'redavg_inpaint'])
    def test_bin_files(self, season, request):
        cfg = self.get_config(season, request)

        uvd, uvd1 = binning.lst_bin_files_from_config(cfg)
        assert uvd.Ntimes == uvd1.Ntimes == 3  # ndays
        assert uvd.Nbls == uvd1.Nbls == len(cfg.antpairs)
        assert uvd.Nfreqs == uvd1.Nfreqs == len(cfg.config.datameta.freq_array)
        assert uvd.Npols == uvd1.Npols == len(cfg.config.datameta.pols)

        # test that exposes bug fixed in 3a3ead0fd13400578b50b5fe05af39be61717206
        uvd0 = UVData.from_file(cfg.matched_files[0])
        assert np.allclose(uvd.get_ENU_antpos()[0], uvd0.get_ENU_antpos()[0])

    def test_redavg_with_where_inpainted(self, request, tmp_path_factory):
        # This is kind of a dodgy way to test that if the inpainted files don't have
        # all the baselines, things will fail. We copy the original data files,
        # write a whole new dataset in the same place but with fewer baselines, then
        # copy the data files (but not the where_inpainted files) back, so they mismatch.

        cfg = self.get_config('redavg_inpaint', request)
        cfg_bad = self.get_config("redavg_inpaint_fewer_ants", request)

        newobs = tmp_path_factory.mktemp("newobs")
        shutil.copytree(cfg.config.datameta.path.parent.parent, newobs, dirs_exist_ok=True)

        bad_inpaint = apply_filename_rules(
            cfg_bad.config.data_files, cfg_bad.config.where_inpainted_file_rules
        )
        for night in bad_inpaint:
            for fl in night:
                fl = Path(fl)
                shutil.copyfile(fl, newobs / fl.parent.name / fl.name)

        cfg = LSTBinConfigurator(
            [sorted(pth.glob("*.uvh5")) for pth in sorted(newobs.glob("*"))],
            where_inpainted_file_rules=[(".uvh5", ".where_inpainted.h5")],
        )
        mf = cfg.get_matched_files()
        cfg = cfg.create_config(mf).at_single_outfile(outfile=0)

        with pytest.raises(ValueError, match="Could not find any baseline from group"):
            binning.lst_bin_files_from_config(cfg)


class TestLSTStack:
    def setup_class(self):
        self.uvd = mockuvd.create_mock_hera_obs(
            integration_time=1.0, ntimes=20, jd_start=2459844.0, ants=[0, 1, 2, 3],
            time_axis_faster_than_bls=False
        )
        self.stack = binning.LSTStack(self.uvd)

    def test_bad_uvd(self):
        uvd = mockuvd.create_mock_hera_obs(
            integration_time=1.0, ntimes=20, jd_start=2459844.0, ants=[0, 1, 2, 3],
            time_axis_faster_than_bls=True
        )
        print('bl', uvd.time_axis_faster_than_bls)
        with pytest.raises(ValueError, match='time_axis_faster_than_bls must be False'):
            binning.LSTStack(uvd)

        uvf = UVFlag(self.uvd, use_future_array_shapes=True)
        uvf.to_waterfall()
        with pytest.raises(ValueError, match="UVFlag type must be 'baseline'"):
            binning.LSTStack(uvf)

        rng = np.random.default_rng(0)
        uvd.reorder_blts(order=rng.permutation(uvd.Nblts))
        with pytest.raises(ValueError, match='blts_are_rectangular must be True'):
            binning.LSTStack(uvd)
