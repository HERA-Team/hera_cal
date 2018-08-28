# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

from hera_cal import io
from hera_cal import smooth_cal
from hera_cal.datacontainer import DataContainer
import numpy as np
import unittest
from copy import deepcopy
from pyuvdata.utils import check_histories
from pyuvdata import UVCal, UVData
from hera_cal.data import DATA_PATH
import os
import glob
import sys
import shutil
from scipy import constants
import warnings


class Test_Smooth_Cal_Helper_Functions(unittest.TestCase):

    def test_time_kernel(self):
        kernel = smooth_cal.time_kernel(100, 10.0, filter_scale=1.0)
        self.assertAlmostEqual(np.sum(kernel), 1.0)
        self.assertEqual(np.max(kernel), kernel[100])
        self.assertEqual(len(kernel), 201)

    def test_smooth_cal_argparser(self):
        sys.argv = [sys.argv[0], 'a', 'b', '--flags_npz_list', 'c']
        a = smooth_cal.smooth_cal_argparser()
        self.assertEqual(a.calfits_list, ['a', 'b'])
        self.assertEqual(a.flags_npz_list, ['c'])

    def test_time_filter(self):
        gains = np.ones((10, 10), dtype=complex)
        gains[3, 5] = 10.0
        wgts = np.ones((10, 10), dtype=float)
        wgts[3, 5] = 0
        times = np.linspace(0, 10 * 10 / 60. / 60. / 24., 10, endpoint=False)
        tf = smooth_cal.time_filter(gains, wgts, times, filter_scale=1800.0, nMirrors=1)
        np.testing.assert_array_almost_equal(tf, np.ones((10, 10), dtype=complex))

    def test_freq_filter(self):
        gains = np.ones((10, 10), dtype=complex)
        gains[3, 5] = 10.0
        wgts = np.ones((10, 10), dtype=float)
        wgts[3, 5] = 0
        freqs = np.linspace(100., 200., 10, endpoint=False) * 1e6
        ff, info = smooth_cal.freq_filter(gains, wgts, freqs)
        np.testing.assert_array_almost_equal(ff, np.ones((10, 10), dtype=complex))

        # test rephasing
        gains = np.ones((2, 1000), dtype=complex)
        wgts = np.ones((2, 1000), dtype=float)
        freqs = np.linspace(100., 200., 1000, endpoint=False) * 1e6
        gains *= np.exp(2.0j * np.pi * np.outer(150e-9 * np.ones(2), freqs))
        ff, info = smooth_cal.freq_filter(gains, wgts, freqs)
        np.testing.assert_array_almost_equal(ff, gains)

        # test skip_wgt
        gains = np.random.randn(10, 10) + 1.0j * np.random.randn(10, 10)
        wgts = np.ones((10, 10), dtype=float)
        wgts[0, 0:8] = 0
        freqs = np.linspace(100., 200., 10, endpoint=False) * 1e6
        ff, info = smooth_cal.freq_filter(gains, wgts, freqs, skip_wgt=.5)
        np.testing.assert_array_equal(ff[0, :], gains[0, :])
        self.assertTrue(info[0]['skipped'])


class Test_Calibration_Smoother(unittest.TestCase):

    def setUp(self):
        calfits_list = sorted(glob.glob(os.path.join(DATA_PATH, 'test_input/*.abs.calfits_54x_only')))[0::2]
        flags_npz_list = sorted(glob.glob(os.path.join(DATA_PATH, 'test_input/*.uvOCR_53x_54x_only.flags.applied.npz')))[0::2]
        self.cs = smooth_cal.CalibrationSmoother(calfits_list, flags_npz_list=flags_npz_list)

    def test_check_consistency(self):
        temp_time = self.cs.cal_times[self.cs.cals[0]][0]
        self.cs.cal_times[self.cs.cals[0]][0] = self.cs.cal_times[self.cs.cals[0]][1]
        self.cs.time_indices = {cal: np.searchsorted(self.cs.time_grid, times) for cal, times in self.cs.cal_times.items()}
        with self.assertRaises(AssertionError):
            self.cs.check_consistency()
        self.cs.cal_times[self.cs.cals[0]][0] = temp_time
        self.cs.time_indices = {cal: np.searchsorted(self.cs.time_grid, times) for cal, times in self.cs.cal_times.items()}

        self.cs.cal_freqs[self.cs.cals[0]] += 1
        with self.assertRaises(AssertionError):
            self.cs.check_consistency()
        self.cs.cal_freqs[self.cs.cals[0]] -= 1

        self.cs.npz_freqs[self.cs.npzs[0]] += 1
        with self.assertRaises(AssertionError):
            self.cs.check_consistency()
        self.cs.npz_freqs[self.cs.npzs[0]] -= 1

        temp_time = self.cs.npz_times[self.cs.npzs[0]][0]
        self.cs.npz_times[self.cs.npzs[0]][0] = self.cs.npz_times[self.cs.npzs[0]][1]
        self.cs.npz_time_indices = {npz: np.searchsorted(self.cs.time_grid, times) for npz, times in self.cs.npz_times.items()}
        with self.assertRaises(AssertionError):
            self.cs.check_consistency()
        self.cs.npz_times[self.cs.npzs[0]][0] = temp_time
        self.cs.npz_time_indices = {npz: np.searchsorted(self.cs.time_grid, times) for npz, times in self.cs.npz_times.items()}

    def test_load_cal_and_flags(self):
        self.assertEqual(len(self.cs.freqs), 1024)
        self.assertEqual(len(self.cs.time_grid), 180)
        self.assertAlmostEqual(self.cs.dt, 10.737419128417969 / 24 / 60 / 60)
        self.assertFalse(self.cs.freq_filtered)
        self.assertFalse(self.cs.time_filtered)
        self.assertTrue((54, 'Jxx') in self.cs.gain_grids)
        self.assertTrue((54, 'Jxx') in self.cs.flag_grids)
        self.assertEqual(self.cs.gain_grids[54, 'Jxx'].shape, (180, 1024))
        self.assertEqual(self.cs.flag_grids[54, 'Jxx'].shape, (180, 1024))
        np.testing.assert_array_equal(self.cs.flag_grids[54, 'Jxx'][60:120, :], True)

    def test_filtering(self):
        g = deepcopy(self.cs.filtered_gain_grids[54, 'Jxx'])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.cs.freq_filter(window='tukey', alpha=.45)
        g2 = deepcopy(self.cs.filtered_gain_grids[54, 'Jxx'])
        self.assertFalse(np.all(g == g2))
        self.assertTrue(self.cs.freq_filtered)
        self.assertEqual(g2.shape, g.shape)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.cs.time_filter()
        g3 = deepcopy(self.cs.filtered_gain_grids[54, 'Jxx'])
        self.assertFalse(np.all(g == g3))
        self.assertTrue(self.cs.time_filtered)
        self.assertEqual(g3.shape, g.shape)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.cs.time_filter()
        g4 = deepcopy(self.cs.filtered_gain_grids[54, 'Jxx'])
        self.assertFalse(np.all(g3 == g4))
        self.assertTrue(self.cs.time_filtered)
        self.assertEqual(g4.shape, g.shape)

        self.cs.reset_filtering()
        self.assertFalse(np.all(self.cs.flag_grids[(54, 'Jxx')] == np.ones_like(self.cs.flag_grids[(54, 'Jxx')])))
        self.cs.filtered_flag_grids[(54, 'Jxx')] = np.zeros_like(self.cs.flag_grids[(54, 'Jxx')])
        self.cs.filtered_flag_grids[(54, 'Jxx')][:, 0:1000] = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.cs.freq_filter()
            np.testing.assert_array_equal(self.cs.filtered_gain_grids[(54, 'Jxx')], g)
            self.cs.time_filter()
            np.testing.assert_array_equal(self.cs.filtered_gain_grids[(54, 'Jxx')], g)
            # test skip_wgt propagation to flags
            np.testing.assert_array_equal(self.cs.filtered_flag_grids[(54, 'Jxx')],
                                          np.ones_like(self.cs.filtered_flag_grids[(54, 'Jxx')]))
        self.cs.reset_filtering()
        self.cs.filtered_gain_grids[54, 'Jxx'] = g
        self.cs.time_filtered, self.cs.freq_filtered = False, False

    def test_write(self):
        outfilename = os.path.join(DATA_PATH, 'test_output/smooth_test.calfits')
        g = deepcopy(self.cs.filtered_gain_grids[54, 'Jxx'])
        self.cs.write_smoothed_cal(output_replace=('test_input/', 'test_output/smoothed_'),
                                   add_to_history='hello world', clobber=True, telescope_name='PAPER')
        for cal in self.cs.cals:
            old_cal, new_cal = UVCal(), UVCal()
            old_cal.read_calfits(cal)
            new_cal.read_calfits(cal.replace('test_input/', 'test_output/smoothed_'))
            self.assertTrue(check_histories(new_cal.history, old_cal.history + 'hello world'))
            self.assertEqual(new_cal.telescope_name, 'PAPER')
            gains, flags = io.load_cal(new_cal)
            np.testing.assert_array_equal(gains[54, 'Jxx'], g[self.cs.time_indices[cal], :])
            os.remove(cal.replace('test_input/', 'test_output/smoothed_'))


if __name__ == '__main__':
    unittest.main()
