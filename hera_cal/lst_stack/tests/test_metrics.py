from __future__ import annotations

import pytest
import numpy as np
from scipy.stats import gamma, chi2, norm
from hera_cal.lst_stack import metrics as mt
from hera_cal.lst_stack.config import LSTBinConfigurator
from hera_cal.lst_stack import lst_bin_files_from_config, reduce_lst_bins
from hera_cal.lst_stack.binning import LSTStack
from hera_cal.lst_stack import stats
from hera_cal.datacontainer import DataContainer
from hera_cal.tests import mock_uvdata as mockuvd
from pyuvdata import UVFlag
import re


@pytest.fixture(scope='module')
def lstconfig(season_nonredavg_with_noise):
    config = LSTBinConfigurator(season_nonredavg_with_noise, nlsts_per_file=2)
    mf = config.get_matched_files()
    config = config.create_config(mf)
    return config.at_single_outfile(0)


@pytest.fixture(scope='module')
def auto_stack(lstconfig) -> LSTStack:
    return LSTStack(
        mockuvd.create_uvd_identifiable(
            with_noise=True,
            freqs=mockuvd.PHASEII_FREQS[:10],
            pols=['xx', 'yy'],
            antpairs=[(i, i) for i in range(4)],
            jd_start=2459844.1,
            integration_time=1.0,  # day
            ntimes=3000,  # enough to get good stats
            time_axis_faster_than_bls=False,
        )  # lst_bin_files_from_config(lstconfig, bl_chunk_to_load='autos')[0]
    )


@pytest.fixture(scope='module')
def cross_stack(lstconfig) -> LSTStack:
    data = mockuvd.create_uvd_identifiable(
        with_noise=True,
        freqs=mockuvd.PHASEII_FREQS[:10],
        pols=['xx', 'yy'],
        antpairs=[(i, j) for i in range(4) for j in range(i, 4)],
        jd_start=2459844.1,
        integration_time=1.0,  # day
        ntimes=3000,  # enough to get good stats
        time_axis_faster_than_bls=False,
    )
    data.select(bls=[(i, j) for i in range(7) for j in range(i + 1, 4)])

    return LSTStack(data)


@pytest.fixture(scope='module')
def auto_rdc(auto_stack) -> dict[str, np.ndarray]:
    return reduce_lst_bins(auto_stack, get_mad=False)


@pytest.fixture(scope='module')
def cross_rdc(cross_stack) -> dict[str, np.ndarray]:
    return reduce_lst_bins(cross_stack, get_mad=True)


@pytest.fixture(scope='module')
def auto_stats(auto_rdc, auto_stack):
    return mt.LSTBinStats.from_reduced_data(
        antpairs=auto_stack.antpairs,
        pols=auto_stack.pols,
        rdc=auto_rdc,
    )


@pytest.fixture(scope='module')
def cross_stats(cross_rdc, cross_stack):
    return mt.LSTBinStats.from_reduced_data(
        antpairs=cross_stack.antpairs,
        pols=cross_stack.pols,
        rdc=cross_rdc,
    )


class TestGetNightlyPredictedVariance:
    def test_cross(self, lstconfig, cross_stack, auto_stats, cross_stats):

        for ap in lstconfig.antpairs:
            for pol in lstconfig.pols:
                bl = ap + (pol,)
                print(bl)
                assert bl in cross_stats.bls
                predicted_var = mt.get_nightly_predicted_variance(
                    bl=bl, stack=cross_stack, auto_stats=auto_stats
                )

                # Each night should have the same predicted variance, since there
                # are no flags and nsamples==1
                np.testing.assert_allclose(np.diff(predicted_var, axis=0), 0)

                std = cross_stats.std[bl]

                # The tolerance here is pretty high, because we have only 100 nights,
                # so the estimated variance is not highly accurate.
                np.testing.assert_allclose(predicted_var[0] / 2, std.real**2, atol=0, rtol=0.5)
                np.testing.assert_allclose(predicted_var[0] / 2, std.imag**2, atol=0, rtol=0.5)

    def test_as_stack(self, cross_stack, auto_stats):
        v = mt.get_nightly_predicted_variance_stack(
            stack=cross_stack,
            auto_stats=auto_stats,
        )
        assert v.shape == cross_stack.data.shape
        assert np.all(v >= 0)


class TestGetZSquared:
    @pytest.mark.parametrize("central", ['mean', 'median'])
    @pytest.mark.parametrize("std", ['autos', 'std', 'mad'])
    def test_uniform_nsamples(self, auto_stats, cross_stats, cross_stack, central, std):

        zsq = mt.get_squared_zscores(auto_stats, cross_stats, cross_stack, central=central, std=std)
        assert isinstance(zsq, LSTStack)
        assert zsq.metrics.shape == cross_stack.data.shape

        dist = stats.zsquare(absolute=True)

        # Check if the mean lines up. The variance on the mean is equal to
        # the variance of chi2(df=2nnights)/nnights^2 == 4*nnights/nnights^2.
        nnights = len(zsq.nights)

        # Ensure we're within 3 sigma of the mean
        if central == 'mean' and std == 'autos':
            np.testing.assert_allclose(dist.mean(), np.mean(zsq.metrics), atol=3 * 2 / np.sqrt(nnights))

            # Tolerance here is large because we have only 100 nights.
            np.testing.assert_allclose(dist.var(), np.var(zsq.metrics), rtol=0.6)

    def test_wrong_central(self, auto_stats, cross_stats, cross_stack):
        with pytest.raises(ValueError, match="central must be 'mean' or 'median'"):
            mt.get_squared_zscores(auto_stats, cross_stats, cross_stack, central='bad')

    def test_wrong_std(self, auto_stats, cross_stats, cross_stack):
        with pytest.raises(ValueError, match="std must be 'autos', 'std' or 'mad'"):
            mt.get_squared_zscores(auto_stats, cross_stats, cross_stack, std='bad')

    def test_get_flagged_z(self, auto_stats, cross_stats, cross_stack):
        zsq = mt.get_squared_zscores_flagged(
            cross_stack, auto_stats=auto_stats
        )
        v = mt.get_nightly_predicted_variance_stack(
            cross_stack, auto_stats=auto_stats, flag_if_inpainted=True
        ) / 2
        zsq2 = mt.get_squared_zscores_flagged(cross_stack, variance=v)

        assert np.all(zsq.metrics == zsq2.metrics)


class TestGetSelectedBls:
    def test_get_all_bls(self):
        bls = [(0, 1, 'ee'), (1, 1, 'nn'), (2, 3, 'en')]
        assert mt.get_selected_bls(bls, min_days=0) == bls

    def test_selectors_is_callable(self):
        bls = [(0, 1, 'ee'), (1, 1, 'nn'), (2, 3, 'en')]
        assert mt.get_selected_bls(
            bls,
            selectors=lambda bl: bl[0] == 0,
            min_days=0
        ) == [(0, 1, 'ee')]

    def test_selectors_is_iterable(self):
        bls = [(0, 1, 'ee'), (1, 1, 'nn'), (2, 3, 'en')]
        assert mt.get_selected_bls(
            bls,
            selectors=[(lambda bl: bl[0] in (0, 1)), (lambda bl: bl[1] == 1)],
            min_days=0
        ) == [(0, 1, 'ee'), (1, 1, 'nn')]

    def test_with_min_days(self):
        bls = [(0, 1, 'ee'), (1, 1, 'nn'), (2, 3, 'en')]
        days_binned = DataContainer({bl: np.ones(100) for bl in bls})
        assert mt.get_selected_bls(bls, days_binned=days_binned, min_days=2) == []


class TestDownSelectZscores:
    def setup_class(self):
        uvd = mockuvd.create_mock_hera_obs(
            integration_time=24 * 3600, ntimes=20, jd_start=2459844.0, ants=[0, 1, 2, 3],
            time_axis_faster_than_bls=False
        )
        uvf = UVFlag(uvd, mode='metric', use_future_array_shapes=True)
        uvf.metric_array[:] = 2.0
        self.zscores = LSTStack(uvf)

        (self.nnights, self.nbls, self.nfreqs, self.npols) = self.zscores.metrics.shape

    def test_downselect_antpairs(self):
        ma = mt.downselect_zscores(self.zscores, antpairs=(0, 1))
        assert isinstance(ma, np.ma.MaskedArray)
        assert ma.shape == (self.nnights, 1, self.nfreqs, self.npols)

        ma = mt.downselect_zscores(self.zscores, antpairs=[(0, 1)])
        assert isinstance(ma, np.ma.MaskedArray)
        assert ma.shape == (self.nnights, 1, self.nfreqs, self.npols)

    def test_downselect_pols(self):
        ma = mt.downselect_zscores(self.zscores, pols=['ee'])
        assert isinstance(ma, np.ma.MaskedArray)
        assert ma.shape == (self.nnights, self.nbls, self.nfreqs, 1)

    def test_downselect_freqs(self):
        ma = mt.downselect_zscores(self.zscores, band=slice(12, 15))
        assert isinstance(ma, np.ma.MaskedArray)
        assert ma.shape == (self.nnights, self.nbls, 3, self.npols)

        ma = mt.downselect_zscores(self.zscores, band=(12, 15))
        assert isinstance(ma, np.ma.MaskedArray)
        assert ma.shape == (self.nnights, self.nbls, 3, self.npols)

        with pytest.raises(TypeError, match=re.escape("band must be a tuple of (low, high) or a slice")):
            mt.downselect_zscores(self.zscores, band=12)

    def test_downselect_nights(self):
        ma = mt.downselect_zscores(self.zscores, nights=self.zscores.nights[:3])
        assert isinstance(ma, np.ma.MaskedArray)
        assert ma.shape == (3, self.nbls, self.nfreqs, self.npols)

        ma = mt.downselect_zscores(self.zscores, nights=self.zscores.nights[0])
        assert isinstance(ma, np.ma.MaskedArray)
        assert ma.shape == (1, self.nbls, self.nfreqs, self.npols)

    def test_downselect_bl_selectors(self):
        ma = mt.downselect_zscores(self.zscores, bl_selectors=lambda bl: bl[0] == 0)
        nbls = len([bl for bl in self.zscores.antpairs if bl[0] == 0])
        assert isinstance(ma, np.ma.MaskedArray)
        assert ma.shape == (self.nnights, nbls, self.nfreqs, self.npols)

        # Test doing pol-selection via the bl_selector
        ma = mt.downselect_zscores(self.zscores, bl_selectors=lambda bl: bl[2] == 'ee')
        assert isinstance(ma, np.ma.MaskedArray)
        assert ma.shape == (self.nnights, self.nbls, self.nfreqs, 1)

        ma = mt.downselect_zscores(self.zscores, bl_selectors=lambda bl: bl[2] == 'nn')
        assert isinstance(ma, np.ma.MaskedArray)
        assert ma.shape == (self.nnights, self.nbls, self.nfreqs, 1)

    def test_downselect_with_flags(self):
        rng = np.random.default_rng(0)

        newz = self.zscores.copy()
        flags = rng.binomial(1, 0.1, size=newz.metrics.shape).astype(bool)
        newz.metrics[flags] == 12345  # Some bad metrics...

        ma = mt.downselect_zscores(self.zscores, flags=flags)
        assert isinstance(ma, np.ma.MaskedArray)
        assert np.ma.all(ma == 2.0)

    def test_compressed_downselect(self):
        newz = self.zscores.copy()
        newz.metrics[0] = 12345
        newz.metrics[:, 0] = 12345
        newz.metrics[:, :, 0] = 12345
        newz.metrics[:, :, :, 0] = 12345

        z = mt.get_compressed_zscores(
            [self.zscores],
            nights=self.zscores.nights[1:],
            antpairs=self.zscores.antpairs[1:],
            pols=self.zscores.pols[1:],
            band=slice(1, None)
        )

        # test that all the bad metrics are gone
        assert z.ndim == 1
        assert np.all(z == 2)

    def test_compress_with_flags(self):
        rng = np.random.default_rng(0)

        newz = self.zscores.copy()
        flags = rng.binomial(1, 0.1, size=newz.metrics.shape).astype(bool)
        newz.metrics[flags] == 12345  # Some bad metrics...

        z = mt.get_compressed_zscores([self.zscores], flags=[flags])
        assert np.all(z == 2.0)


class TestReduceStackOverAxis:
    def setup_class(self):
        uvd = mockuvd.create_mock_hera_obs(
            integration_time=1.0, ntimes=20, jd_start=2459844.0, ants=[0, 1, 2, 3],
            time_axis_faster_than_bls=False
        )
        uvf = UVFlag(uvd, mode='metric', use_future_array_shapes=True)
        uvf.metric_array[:] = 2.0
        self.zscores = LSTStack(uvf)

        (self.nnights, self.nbls, self.nfreqs, self.npols) = self.zscores.metrics.shape

    def test_reduce_bls(self):
        reduced = mt.reduce_stack_over_axis(np.mean, self.zscores.metrics, axis='bls')
        assert reduced.shape == (self.nnights, self.nfreqs)

    def test_reduce_antpairs(self):
        reduced = mt.reduce_stack_over_axis(np.std, self.zscores.metrics, axis='antpairs')
        assert reduced.shape == (self.nnights, self.nfreqs, self. npols)

    def test_reduce_freqs(self):
        reduced = mt.reduce_stack_over_axis(np.max, self.zscores.metrics, axis='freqs')
        assert reduced.shape == (self.nnights, self.nbls, self.npols)

    def test_reduce_pols(self):
        reduced = mt.reduce_stack_over_axis(np.min, self.zscores.metrics, axis='pols')
        assert reduced.shape == (self.nnights, self.nbls, self.nfreqs)

    def test_reduce_nights(self):
        reduced = mt.reduce_stack_over_axis(np.mean, self.zscores.metrics, axis='nights')
        assert reduced.shape == (self.nbls, self.nfreqs, self.npols)

    def test_reduce_multi(self):
        reduced = mt.reduce_stack_over_axis(
            np.mean, self.zscores.metrics, axis=['nights', 'freqs']
        )
        assert reduced.shape == (self.nbls, self.npols)
