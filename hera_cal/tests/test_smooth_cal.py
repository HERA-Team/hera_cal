# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

import pytest
import numpy as np
from copy import deepcopy
import os
import glob
import sys
import shutil
from scipy import constants
import warnings
from pyuvdata import UVCal, UVData
import pytest
from hera_filters import dspec

from .. import io, smooth_cal, utils
from ..datacontainer import DataContainer
from ..data import DATA_PATH


class Test_Smooth_Cal_Helper_Functions(object):

    def setup_method(self):
        np.random.seed(21)

    def test_time_kernel(self):
        kernel = smooth_cal.time_kernel(100, 10.0, filter_scale=1.0)
        assert np.allclose(np.sum(kernel), 1.0)
        assert np.max(kernel) == kernel[100]
        assert len(kernel) == 201

    def test_smooth_cal_argparser(self):
        sys.argv = [sys.argv[0], 'a', 'b', '--flag_file_list', 'c', '--lst_blacklists', '3-4', '10-12', '23-.5']
        a = smooth_cal.smooth_cal_argparser()
        assert a.calfits_list == ['a', 'b']
        assert a.flag_file_list == ['c']
        assert a.lst_blacklists == [(3, 4), (10, 12), (23, .5)]

    def test_dpss_filters(self):
        times = np.linspace(0, 10 * 10 / 60. / 60. / 24., 40, endpoint=False)
        freqs = np.linspace(100., 200., 50, endpoint=False) * 1e6
        freq_scale = 0.5 / (np.diff(freqs)[0] / 1e6)
        time_scale = 0.5 / (np.diff(times)[0] * 3600 * 24)
        # Show that dpss_filter generates filters for a 2D grid of times and freqs
        time_filters, freq_filters = smooth_cal.dpss_filters(
            times=times, freqs=freqs, time_scale=1.01 / time_scale, freq_scale=1.01 / freq_scale
        )
        assert time_filters.shape[0] == times.shape[0]
        assert freq_filters.shape[0] == freqs.shape[0]
        # test that error is thrown when filtering scale is too fine
        assert pytest.raises(ValueError, smooth_cal.dpss_filters, times=times, freqs=freqs,
                             time_scale=0.99 / time_scale, freq_scale=1.01 / freq_scale)
        assert pytest.raises(ValueError, smooth_cal.dpss_filters, times=times, freqs=freqs,
                             time_scale=1.01 / time_scale, freq_scale=0.99 / freq_scale)

        v = dspec.dpss_operator(times * 60 * 60 * 24, [0], [time_scale / 1.01], eigenval_cutoff=[1e-9])[0].real
        for i in range(v.shape[1]):
            np.testing.assert_allclose(v[:, i], time_filters[:, i])

        v = dspec.dpss_operator(freqs, [0], [freq_scale / 1e6 / 1.01], eigenval_cutoff=[1e-9])[0].real
        for i in range(v.shape[1]):
            np.testing.assert_allclose(v[:, i], freq_filters[:, i])

    def test_solve_2D_DPSS(self):
        time_filters = np.random.uniform(0, 1, size=(50, 2))
        freq_filters = np.random.uniform(0, 1, size=(40, 5))
        weights = np.random.uniform(0, 1, size=(50, 40))
        gains = np.random.uniform(0, 1, size=(50, 40))
        fit1, info = smooth_cal.solve_2D_DPSS(gains, weights, time_filters, freq_filters)
        assert fit1.shape == gains.shape

        # Check XTXinv
        fit2, info = smooth_cal.solve_2D_DPSS(gains, weights, time_filters, freq_filters, XTXinv=info['XTXinv'])
        assert fit1.shape == fit2.shape
        np.testing.assert_array_equal(fit1, fit2)

        # Check to see that this function matches the true result
        X = np.kron(time_filters, freq_filters)
        fit_lsq = X @ np.linalg.pinv((X.T * weights.ravel()) @ X) @ (X.T * weights.ravel()) @ gains.ravel()
        np.testing.assert_array_almost_equal(fit_lsq, fit2.ravel())

    def test_time_filter(self):
        gains = np.ones((10, 10), dtype=complex)
        gains[3, 5] = 10.0
        wgts = np.ones((10, 10), dtype=float)
        wgts[3, 5] = 0
        times = np.linspace(0, 10 * 10 / 60. / 60. / 24., 10, endpoint=False)
        tf = smooth_cal.time_filter(gains, wgts, times, filter_scale=1800.0, nMirrors=1)
        np.testing.assert_array_almost_equal(tf, np.ones((10, 10), dtype=complex))

    @pytest.mark.filterwarnings("ignore: Mean of empty slice")
    def test_single_iterative_fft_dly(self):
        # try without flags
        gains = np.ones((2, 1000), dtype=complex)
        wgts = np.ones((2, 1000), dtype=float)
        freqs = np.linspace(100., 200., 1000, endpoint=False) * 1e6
        gains *= np.exp(2.0j * np.pi * np.outer(151e-9 * np.ones(2), freqs))
        dly = smooth_cal.single_iterative_fft_dly(gains, wgts, freqs)
        np.testing.assert_array_almost_equal(dly, 151e-9)

        # try with flags
        gains = np.ones((2, 1000), dtype=complex)
        wgts = np.ones((2, 1000), dtype=float)
        wgts[:, 0:40] = 0.0
        wgts[:, 900:] = 0.0
        freqs = np.linspace(100., 200., 1000, endpoint=False) * 1e6
        gains *= np.exp(2.0j * np.pi * np.outer(-151e-9 * np.ones(2), freqs))
        dly = smooth_cal.single_iterative_fft_dly(gains, wgts, freqs)
        np.testing.assert_array_almost_equal(dly, -151e-9)

        # try all flagged
        gains = np.ones((2, 1000), dtype=complex)
        wgts = np.zeros((2, 1000), dtype=float)
        freqs = np.linspace(100., 200., 1000, endpoint=False) * 1e6
        gains *= np.exp(2.0j * np.pi * np.outer(-151e-9 * np.ones(2), freqs))
        assert smooth_cal.single_iterative_fft_dly(gains, wgts, freqs) == 0

    @pytest.mark.parametrize("ax", ['freq', 'time'])
    def test_filter_1d(self, ax):
        gains = np.ones((10, 10), dtype=complex)
        gains[3, 5] = 10.0
        wgts = np.ones((10, 10), dtype=float)
        wgts[3, 5] = 0
        freqs = np.linspace(100., 200., 10, endpoint=False) * 1e6
        times = np.linspace(0, 100, 10, endpoint=False) / (24 * 3600.)
        # choose time scale equivalent to 10 MHz freq scale
        df = np.mean(np.diff(freqs))
        dt = np.mean(np.diff(times))
        if ax == 'freq':
            xaxis = freqs
            filter_scale = 10e6
        else:
            xaxis = times
            filter_scale = 10e6 * dt / df

        ff, info = smooth_cal.filter_1d(gains, wgts, xaxis, ax=ax, filter_scale=filter_scale)
        np.testing.assert_array_almost_equal(ff, np.ones((10, 10), dtype=complex), decimal=5)

        # test rephasing
        if ax == 'freq':
            gains = np.ones((2, 1000), dtype=complex)
            wgts = np.ones((2, 1000), dtype=float)
            freqs = np.linspace(100., 200., 1000, endpoint=False) * 1e6
            gains *= np.exp(2.0j * np.pi * np.outer(150e-9 * np.ones(2), freqs))
            ff, info = smooth_cal.filter_1d(gains, wgts, freqs, ax=ax)
            np.testing.assert_array_almost_equal(ff, gains, decimal=5)

        # test skip_wgt
        gains = np.random.randn(10, 10) + 1.0j * np.random.randn(10, 10)
        wgts = np.ones((10, 10), dtype=float)
        freqs = np.linspace(100., 200., 10, endpoint=False) * 1e6
        times = np.linspace(0, 100, 10, endpoint=False) / (24 * 3600.)

        if ax == 'freq':
            wgts[0, 0:8] = 0
            xaxis = freqs
        else:
            wgts[0:8, 0] = 0
            xaxis = times

        ff, info = smooth_cal.filter_1d(gains, wgts, xaxis, skip_wgt=.5, ax=ax, filter_scale=filter_scale)
        if ax == 'freq':
            np.testing.assert_array_almost_equal(ff[0, :], gains[0, :], decimal=5)
        else:
            np.testing.assert_array_almost_equal(ff[:, 0], gains[:, 0], decimal=5)

        if ax == 'freq':
            assert info['status']['axis_1'][0] == 'skipped'
            filter_scale = 50.
            xaxis = freqs
        else:
            assert info['status']['axis_0'][0] == 'skipped'
            filter_scale = 50e6 * dt / df
            xaxis = times

        ff, info = smooth_cal.filter_1d(gains, wgts, xaxis, skip_wgt=.5, filter_scale=filter_scale,
                                        mode='dpss_leastsq', ax=ax)

        if ax == 'freq':
            assert info['status']['axis_1'][0] == 'skipped'
            assert info['status']['axis_1'][1] == 'success'
            np.testing.assert_array_almost_equal(ff[0, :], gains[0, :], decimal=5)
        else:
            assert info['status']['axis_0'][0] == 'skipped'
            assert info['status']['axis_0'][1] == 'success'
            np.testing.assert_array_almost_equal(ff[:, 0], gains[:, 0], decimal=5)

    def test_freq_filter_dpss(self):
        # run freq_filter tests for dpss/dft modes.
        gains = np.ones((10, 100), dtype=complex)
        gains[3, 5] = 10.0
        wgts = np.ones((10, 100), dtype=float)
        wgts[3, 5] = 0
        freqs = np.linspace(100., 200., 100, endpoint=False) * 1e6
        ff, info = smooth_cal.filter_1d(gains, wgts, freqs, mode='dpss_leastsq')
        np.testing.assert_array_almost_equal(ff, np.ones((10, 100), dtype=complex))

        # test rephasing
        gains = np.ones((2, 1000), dtype=complex)
        wgts = np.ones((2, 1000), dtype=float)
        freqs = np.linspace(100., 200., 1000, endpoint=False) * 1e6
        gains *= np.exp(2.0j * np.pi * np.outer(150e-9 * np.ones(2), freqs))
        ff, info = smooth_cal.filter_1d(gains, wgts, freqs, mode='dpss_leastsq')
        np.testing.assert_array_almost_equal(ff, gains, decimal=5)

        # test skip_wgt
        gains = np.random.randn(10, 100) + 1.0j * np.random.randn(10, 100)
        wgts = np.ones((10, 100), dtype=float)
        wgts[0, 0:80] = 0
        freqs = np.linspace(100., 200., 100, endpoint=False) * 1e6
        ff, info = smooth_cal.filter_1d(gains, wgts, freqs, skip_wgt=.5, mode='dpss_leastsq')
        np.testing.assert_array_almost_equal(ff[0, :], gains[0, :])
        info['status']['axis_1'][0] == 'skipped'

        assert pytest.raises(ValueError, smooth_cal.filter_1d, gains=gains, wgts=wgts,
                             xvals=freqs, fitting_options=None, mode='dpss_leastsq')

    def test_freq_filter_dpss_skip_flagged_edges(self):
        # run freq_filter tests dpss modes.
        # test skip_wgt
        gains = (np.random.randn(10) + 1.0j * np.random.randn(10))[:, None] * np.ones((10, 100))
        wgts = np.ones((10, 100), dtype=float)
        wgts[0, 0:80] = 0.
        wgts[7, 10:90] = 0.
        wgts[:, :3] = 0.
        freqs = np.linspace(100., 200., 100, endpoint=False) * 1e6
        ff, info = smooth_cal.filter_1d(gains, wgts, freqs, skip_wgt=.5, mode='dpss_leastsq',
                                        skip_flagged_edges=True, filter_scale=100)
        for i in range(10):
            if i in [0, 7]:
                info['status']['axis_1'][i] == 'skipped'
                np.testing.assert_array_almost_equal(ff[i, :], gains[i, :])
            else:
                info['status']['axis_1'][i] == 'success'
                np.testing.assert_array_almost_equal(ff[i, :3], 0.)
                np.testing.assert_array_almost_equal(ff[i, 3:], gains[i, 3:], decimal=4)

    def test_time_freq_2D_filter(self):
        gains = np.ones((100, 100), dtype=complex)
        gains[3, 5] = 10.0
        wgts = np.ones((100, 100), dtype=float)
        wgts[3, 5] = 0
        freqs = np.linspace(100., 200., 100, endpoint=False) * 1e6
        times = np.linspace(0, 10 * 10 / 60. / 60. / 24., 100, endpoint=False)
        ff, info = smooth_cal.time_freq_2D_filter(gains, wgts, freqs, times, filter_mode='rect')
        np.testing.assert_array_almost_equal(ff, np.ones((100, 100), dtype=complex))
        ff, info = smooth_cal.time_freq_2D_filter(gains, wgts, freqs, times, filter_mode='plus')
        np.testing.assert_array_almost_equal(ff, np.ones((100, 100), dtype=complex))
        ff, info = smooth_cal.time_freq_2D_filter(gains, wgts, freqs, times, method='DPSS', eigenval_cutoff=1e-12)
        np.testing.assert_array_almost_equal(ff, np.ones((100, 100), dtype=complex))
        ff, info = smooth_cal.time_freq_2D_filter(gains, wgts, freqs, times, method='DPSS', skip_flagged_edges=False,
                                                  eigenval_cutoff=1e-12)
        np.testing.assert_array_almost_equal(ff, np.ones((100, 100), dtype=complex))

        # test rephasing
        gains = np.ones((100, 100), dtype=complex)
        wgts = np.ones((100, 100), dtype=float)
        gains *= np.exp(2.0j * np.pi * np.outer(-151e-9 * np.ones(100), freqs))
        ff, info = smooth_cal.time_freq_2D_filter(gains, wgts, freqs, times)
        np.testing.assert_array_almost_equal(ff, gains, 4)

        # test errors
        with pytest.raises(ValueError):
            ff, info = smooth_cal.time_freq_2D_filter(gains, wgts, freqs, times, filter_mode='blah')
        with pytest.raises(ValueError):
            ff, info = smooth_cal.time_freq_2D_filter(gains, wgts, freqs, times, method='DPSS', filter_mode='blah')
        with pytest.raises(ValueError):
            ff, info = smooth_cal.time_freq_2D_filter(gains, wgts, freqs, times, method='blah')
        with pytest.raises(NotImplementedError):
            ff, info = smooth_cal.time_freq_2D_filter(gains, wgts, freqs, times, method='DPSS', filter_mode='plus')

    def flag_threshold_and_broadcast(self):
        flags = {(i, 'Jxx'): np.zeros((10, 10), dtype=bool) for i in range(3)}
        for ant in flags.keys():
            flags[ant][4, 0:6] = True
            flags[ant][0:4, 4] = True
        flag_threshold_and_broadcast(flags, freq_threshold=0.35, time_threshold=0.5, ant_threshold=1.0)
        for ant in flags.keys():
            assert np.all(flags[ant][4, :])
            assert np.all(flags[ant][:, 4])

        assert not np.all(flags[(0, 'Jxx')])
        flags[(0, 'Jxx')][0:8, :] = True
        flag_threshold_and_broadcast(flags, freq_threshold=1.0, time_threshold=1.0, ant_threshold=0.5)
        assert np.all(flags[0, 'Jxx'])
        assert not np.all(flags[1, 'Jxx'])

    @pytest.mark.filterwarnings("ignore:Mean of empty slice")
    def test_pick_reference_antenna(self):
        gains = {(n, 'Jxx'): np.ones((10, 10), dtype=complex) for n in range(10)}
        flags = {(n, 'Jxx'): np.zeros((10, 10), dtype=bool) for n in range(10)}
        freqs = np.linspace(100e6, 200e6, 10)
        for n in range(0, 7):  # add flags to disqualify antennas 0, 1, 2, 3, 4, 5, 6
            flags[(n, 'Jxx')][:, 4] = True
        for n in range(6, 9):  # add phase noise to disqualify antennas 6, 7, 8
            gains[(n, 'Jxx')] *= np.exp(.1j * np.pi * np.random.rand(10, 10))  # want this to be << 2pi to avoid phase wraps
        assert smooth_cal.pick_reference_antenna(gains, flags, freqs, per_pol=False) == (9, 'Jxx')
        assert smooth_cal.pick_reference_antenna(gains, flags, freqs) == {'Jxx': (9, 'Jxx')}

    def test_rephase_to_refant(self):
        gains = {(0, 'Jxx'): np.array([1. + 1.0j, 1. - 1.0j]),
                 (1, 'Jxx'): np.array([-1. + 1.0j, -1. - 1.0j])}
        smooth_cal.rephase_to_refant(gains, (0, 'Jxx'))
        np.testing.assert_array_almost_equal(np.imag(gains[(0, 'Jxx')]), np.zeros_like(np.imag(gains[(0, 'Jxx')])))
        flags = {(0, 'Jxx'): np.array([False, True]),
                 (1, 'Jxx'): np.array([True, False])}
        with pytest.raises(ValueError):
            smooth_cal.rephase_to_refant(gains, (0, 'Jxx'), flags=flags)
        smooth_cal.rephase_to_refant(gains, (0, 'Jxx'), flags=flags, propagate_refant_flags=True)
        np.testing.assert_array_equal(flags[0, 'Jxx'], np.array([False, True]))
        np.testing.assert_array_equal(flags[1, 'Jxx'], np.array([True, True]))

    def test_build_time_blacklist(self):
        time_grid = np.array([2458838.30008962, 2458838.30020147, 2458838.30031332, 2458838.30042517, 2458838.30053701, 2458838.30064886])
        lst_grid = np.array([2.56920604, 2.57189775, 2.57458945, 2.57728116, 2.57997286, 2.58266456])

        # test time cuts
        time_blacklist = smooth_cal.build_time_blacklist(time_grid, time_blacklists=[(2458838.3000, 2458838.3004)])
        np.testing.assert_array_equal(time_blacklist, [True, True, True, False, False, False])

        # test LST cuts
        time_blacklist = smooth_cal.build_time_blacklist(time_grid, lst_blacklists=[(2.5692, 2.5746)])
        np.testing.assert_array_equal(time_blacklist, [True, True, True, False, False, False])

        # try shifting hera position so that the time_grd spans the branch cut in LSTs
        hera_lat_lon_alt_degrees = (-30.721526120689507, 21.428303826863015, 1051.690000018105)
        shifted_llad = np.array(hera_lat_lon_alt_degrees) + [0, 321.38115825, 0]
        time_blacklist = smooth_cal.build_time_blacklist(time_grid, lst_blacklists=[(23.995, 0.005)], lat_lon_alt_degrees=shifted_llad)
        np.testing.assert_array_equal(time_blacklist, [False, True, True, True, False, False])

        # test errors
        with pytest.raises(AssertionError):
            time_blacklist = smooth_cal.build_time_blacklist(time_grid, time_blacklists=[(2458838.3000)])
        with pytest.raises(AssertionError):
            time_blacklist = smooth_cal.build_time_blacklist(time_grid, time_blacklists=[(2458838.3004, 2458838.3000)])
        with pytest.raises(AssertionError):
            time_blacklist = smooth_cal.build_time_blacklist(time_grid, lst_blacklists=[(2.5692)])
        with pytest.raises(NotImplementedError):
            time_blacklist = smooth_cal.build_time_blacklist(time_grid, lst_blacklists=[(2.5692, 2.5746)], telescope_name='NOT_A_REAL_TELESCOPE')

    def test_build_freq_blacklist(self):
        freqs = np.array([100e6, 120e6, 140e6, 160e6, 180e6, 200e6])
        freq_blacklist = smooth_cal.build_freq_blacklist(freqs, freq_blacklists=[(0e6, 137e6)])
        np.testing.assert_array_equal(freq_blacklist, [True, True, False, False, False, False])

        freq_blacklist = smooth_cal.build_freq_blacklist(freqs, chan_blacklists=[(4, 5)])
        np.testing.assert_array_equal(freq_blacklist, [False, False, False, False, True, True])

        # test errors
        with pytest.raises(AssertionError):
            time_blacklist = smooth_cal.build_freq_blacklist(freqs, freq_blacklists=[(137e6)])
        with pytest.raises(AssertionError):
            time_blacklist = smooth_cal.build_freq_blacklist(freqs, freq_blacklists=[(137e6, 0e6)])
        with pytest.raises(AssertionError):
            time_blacklist = smooth_cal.build_freq_blacklist(freqs, chan_blacklists=[(1)])
        with pytest.raises(AssertionError):
            time_blacklist = smooth_cal.build_freq_blacklist(freqs, chan_blacklists=[(3, 1)])

    def test_build_wgts_grid(self):
        flag_grid = np.zeros((2, 3), dtype=bool)
        flag_grid[0, 0] = True
        wgts_grid = smooth_cal._build_wgts_grid(flag_grid, time_blacklist=[False, True], freq_blacklist=[False, False, True])
        np.testing.assert_array_equal(wgts_grid, [[0, 1, 0], [0, 0, 0]])

        wgts_grid = smooth_cal._build_wgts_grid(flag_grid, time_blacklist=[False, True], freq_blacklist=[False, False, True], blacklist_wgt=.1)
        np.testing.assert_array_equal(wgts_grid, [[0, 1, .1], [.1, .1, .1]])


class Test_Calibration_Smoother(object):

    def setup_method(self):
        calfits_list = sorted(glob.glob(os.path.join(DATA_PATH, 'test_input/*.abs.calfits_54x_only')))[0::2]
        flag_file_list = sorted(glob.glob(os.path.join(DATA_PATH, 'test_input/*.uvOCR_53x_54x_only.flags.applied.npz')))[0::2]
        self.cs = smooth_cal.CalibrationSmoother(calfits_list, flag_file_list=flag_file_list, flag_filetype='npz',
                                                 time_blacklists=[(2458101.44795386, 2458101.44919662)],
                                                 lst_blacklists=[(6.17226452, 6.17824609)],
                                                 freq_blacklists=[(136e6, 138e6), (149e6, 151e6)], chan_blacklists=[(0, 64)])

    @pytest.mark.filterwarnings("ignore:invalid value encountered in true_divide")
    @pytest.mark.filterwarnings("ignore:overflow encountered in square")
    @pytest.mark.filterwarnings("ignore:invalid value encountered in reduce")
    @pytest.mark.filterwarnings("ignore:Mean of empty slice")
    def test_ref_ant(self):
        calfits_list = sorted(glob.glob(os.path.join(DATA_PATH, 'test_input/*.abs.calfits_54x_only')))[0::2]
        flag_file_list = sorted(glob.glob(os.path.join(DATA_PATH, 'test_input/*.uvOCR_53x_54x_only.flags.applied.npz')))[0::2]
        cs = smooth_cal.CalibrationSmoother(calfits_list, flag_file_list=flag_file_list, flag_filetype='npz', pick_refant=True)
        assert cs.refant['Jee'] == (54, 'Jee')
        cs.time_freq_2D_filter(window='tukey', alpha=.45)
        cs.rephase_to_refant()
        np.testing.assert_array_almost_equal(np.imag(cs.gain_grids[54, 'Jee']),
                                             np.zeros_like(np.imag(cs.gain_grids[54, 'Jee'])))

    def test_check_consistency(self):
        temp_time = self.cs.cal_times[self.cs.cals[0]][0]
        self.cs.cal_times[self.cs.cals[0]][0] = self.cs.cal_times[self.cs.cals[0]][1]
        self.cs.time_indices = {cal: np.searchsorted(self.cs.time_grid, times) for cal, times in self.cs.cal_times.items()}
        with pytest.raises(AssertionError):
            self.cs.check_consistency()
        self.cs.cal_times[self.cs.cals[0]][0] = temp_time
        self.cs.time_indices = {cal: np.searchsorted(self.cs.time_grid, times) for cal, times in self.cs.cal_times.items()}

        self.cs.cal_freqs[self.cs.cals[0]] += 1
        with pytest.raises(AssertionError):
            self.cs.check_consistency()
        self.cs.cal_freqs[self.cs.cals[0]] -= 1

        self.cs.flag_freqs[self.cs.flag_files[0]] += 1
        with pytest.raises(AssertionError):
            self.cs.check_consistency()
        self.cs.flag_freqs[self.cs.flag_files[0]] -= 1

        temp_time = self.cs.flag_times[self.cs.flag_files[0]][0]
        self.cs.flag_times[self.cs.flag_files[0]][0] = self.cs.flag_times[self.cs.flag_files[0]][1]
        self.cs.flag_time_indices = {ff: np.searchsorted(self.cs.time_grid, times) for ff, times in self.cs.flag_times.items()}
        with pytest.raises(AssertionError):
            self.cs.check_consistency()
        self.cs.flag_times[self.cs.flag_files[0]][0] = temp_time
        self.cs.flag_time_indices = {ff: np.searchsorted(self.cs.time_grid, times) for ff, times in self.cs.flag_times.items()}

    def test_load_cal_and_flags(self):
        assert len(self.cs.freqs) == 1024
        assert len(self.cs.time_grid) == 180
        assert np.allclose(self.cs.dt, 10.737419128417969 / 24 / 60 / 60)
        assert (54, 'Jee') in self.cs.gain_grids
        assert (54, 'Jee') in self.cs.flag_grids
        assert self.cs.gain_grids[54, 'Jee'].shape == (180, 1024)
        assert self.cs.flag_grids[54, 'Jee'].shape == (180, 1024)
        np.testing.assert_array_equal(self.cs.flag_grids[54, 'Jee'][60:120, :], True)

    @pytest.mark.parametrize("ax", ['freq', 'time'])
    def test_1D_filtering(self, ax):
        g = deepcopy(self.cs.gain_grids[54, 'Jee'])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.cs.filter_1d(window='tukey', alpha=.45, ax=ax)
        g2 = deepcopy(self.cs.gain_grids[54, 'Jee'])
        assert not np.all(g == g2)
        assert g2.shape == g.shape

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.cs.time_filter()
        g3 = deepcopy(self.cs.gain_grids[54, 'Jee'])
        assert not np.all(g == g3)
        assert g3.shape == g.shape

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.cs.time_filter()
        g4 = deepcopy(self.cs.gain_grids[54, 'Jee'])
        assert not np.all(g3 == g4)
        assert g4.shape == g.shape

        self.setup_method()
        assert not np.all(self.cs.flag_grids[(54, 'Jee')] == np.ones_like(self.cs.flag_grids[(54, 'Jee')]))
        self.cs.flag_grids[(54, 'Jee')] = np.zeros_like(self.cs.flag_grids[(54, 'Jee')])
        if ax == 'freq':
            self.cs.flag_grids[(54, 'Jee')][:, 0:1000] = True
        else:
            self.cs.flag_grids[(54, 'Jee')][0:1000, :] = True

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.cs.filter_1d(ax=ax)
            np.testing.assert_array_equal(self.cs.gain_grids[(54, 'Jee')], g)
            self.cs.time_filter()
            np.testing.assert_array_equal(self.cs.gain_grids[(54, 'Jee')], g)
            # test skip_wgt propagation to flags
            np.testing.assert_array_equal(self.cs.flag_grids[(54, 'Jee')],
                                          np.ones_like(self.cs.flag_grids[(54, 'Jee')]))
        self.setup_method()
        self.cs.gain_grids[54, 'Jee'] = g

    def test_2D_filtering(self):
        g = deepcopy(self.cs.gain_grids[54, 'Jee'])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.cs.time_freq_2D_filter(window='tukey', alpha=.45)
        g2 = deepcopy(self.cs.gain_grids[54, 'Jee'])
        assert not np.all(g == g2)
        assert g2.shape == g.shape

        # Test DPSS mode
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.cs.time_freq_2D_filter(method='DPSS')

        g3 = deepcopy(self.cs.gain_grids[54, 'Jee'])
        assert not np.all(g3 == g2)
        assert g2.shape == g3.shape

        # Test DPSS mode w/ skip_flagged_edges
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.cs.time_freq_2D_filter(method='DPSS', skip_flagged_edges=True)

        g4 = deepcopy(self.cs.gain_grids[54, 'Jee'])
        assert not np.all(g4 == g2)
        assert g2.shape == g4.shape

        # Test case of skip_flagged_edges where all grids aren't the same
        cs2 = deepcopy(self.cs)
        cs2.ants.append((55, 'Jee'))
        cs2.ants.append((56, 'Jee'))
        cs2.flag_grids[cs2.ants[1]] = np.ones_like(cs2.flag_grids[cs2.ants[0]], dtype=bool)
        cs2.flag_grids[cs2.ants[1]][10:-10, 10:-10] = False
        cs2.flag_grids[cs2.ants[2]] = deepcopy(cs2.flag_grids[cs2.ants[0]])
        cs2.gain_grids[cs2.ants[1]] = deepcopy(cs2.gain_grids[cs2.ants[0]])
        cs2.gain_grids[cs2.ants[2]] = deepcopy(cs2.gain_grids[cs2.ants[0]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cs2.time_freq_2D_filter(method='DPSS', skip_flagged_edges=True)
        del cs2

    @pytest.mark.filterwarnings("ignore:Mean of empty slice")
    def test_write(self):
        outfilename = os.path.join(DATA_PATH, 'test_output/smooth_test.calfits')
        g = deepcopy(self.cs.gain_grids[54, 'Jee'])
        self.cs.write_smoothed_cal(output_replace=('test_input/', 'test_output/smoothed_'),
                                   add_to_history='hello world', clobber=True, telescope_name='PAPER')
        for cal in self.cs.cals:
            new_cal = io.HERACal(cal.replace('test_input/', 'test_output/smoothed_'))
            gains, flags, qual, total_qual = new_cal.read()
            old_cal = io.HERACal(cal)
            old_gains, _, _, _ = old_cal.read()
            assert 'helloworld' in new_cal.history.replace('\n', '').replace(' ', '')
            assert 'Thisfilewasproducedbythefunction' in new_cal.history.replace('\n', '').replace(' ', '')
            assert new_cal.telescope_name == 'PAPER'
            np.testing.assert_array_equal(gains[54, 'Jee'], g[self.cs.time_indices[cal], :])

            relative_diff, avg_relative_diff = utils.gain_relative_difference(gains, old_gains, flags)
            np.testing.assert_array_equal(qual[54, 'Jee'], relative_diff[54, 'Jee'])
            np.testing.assert_array_equal(total_qual['Jee'], avg_relative_diff['Jee'])
            os.remove(cal.replace('test_input/', 'test_output/smoothed_'))
