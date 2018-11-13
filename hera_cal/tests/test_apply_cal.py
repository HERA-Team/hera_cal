# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

"""Unit tests for the hera_cal.apply_cal module."""

from __future__ import absolute_import, division, print_function

import unittest
import numpy as np
from copy import deepcopy
import os
import sys
import shutil
from scipy import constants
import warnings
from pyuvdata.utils import check_histories
from pyuvdata import UVCal, UVData

from hera_cal import io
from hera_cal import apply_cal as ac
from hera_cal.datacontainer import DataContainer
from hera_cal.data import DATA_PATH


class Test_Update_Cal(unittest.TestCase):

    def test_calibrate_in_place(self):
        np.random.seed(21)
        vis = np.random.randn(10, 10) + 1.0j * np.random.randn(10, 10)
        dc = DataContainer({(0, 1, 'xx'): deepcopy(vis)})
        f = np.random.randn(10, 10) > 0
        flags = DataContainer({(0, 1, 'xx'): deepcopy(f)})
        g0_new = np.random.randn(10, 10) + 1.0j * np.random.randn(10, 10)
        g1_new = np.random.randn(10, 10) + 1.0j * np.random.randn(10, 10)
        g_new = {(0, 'Jxx'): g0_new, (1, 'Jxx'): g1_new}
        g0_old = np.random.randn(10, 10) + 1.0j * np.random.randn(10, 10)
        g1_old = np.random.randn(10, 10) + 1.0j * np.random.randn(10, 10)
        g_old = {(0, 'Jxx'): g0_old, (1, 'Jxx'): g1_old}
        cal_flags = {(0, 'Jxx'): np.random.randn(10, 10) > 0, (1, 'Jxx'): np.random.randn(10, 10) > 0}
        # test standard operation
        ac.calibrate_in_place(dc, g_new, flags, cal_flags, old_gains=g_old, gain_convention='divide')
        for i in range(10):
            for j in range(10):
                self.assertAlmostEqual(dc[(0, 1, 'xx')][i, j], vis[i, j] * g0_old[i, j] * np.conj(g1_old[i, j]) / g0_new[i, j] / np.conj(g1_new[i, j]))
                if f[i, j] or cal_flags[(0, 'Jxx')][i, j] or cal_flags[(1, 'Jxx')][i, j]:
                    self.assertTrue(flags[(0, 1, 'xx')][i, j])
                else:
                    self.assertFalse(flags[(0, 1, 'xx')][i, j])

        # test without old cal
        dc = DataContainer({(0, 1, 'xx'): deepcopy(vis)})
        flags = DataContainer({(0, 1, 'xx'): deepcopy(f)})
        ac.calibrate_in_place(dc, g_new, flags, cal_flags, gain_convention='divide')
        for i in range(10):
            for j in range(10):
                self.assertAlmostEqual(dc[(0, 1, 'xx')][i, j], vis[i, j] / g0_new[i, j] / np.conj(g1_new[i, j]))

        # test multiply
        dc = DataContainer({(0, 1, 'xx'): deepcopy(vis)})
        flags = DataContainer({(0, 1, 'xx'): deepcopy(f)})
        ac.calibrate_in_place(dc, g_new, flags, cal_flags, old_gains=g_old, gain_convention='multiply')
        for i in range(10):
            for j in range(10):
                self.assertAlmostEqual(dc[(0, 1, 'xx')][i, j], vis[i, j] / g0_old[i, j] / np.conj(g1_old[i, j]) * g0_new[i, j] * np.conj(g1_new[i, j]))

        # test flag propagation when missing antennas in gains
        dc = DataContainer({(0, 1, 'xx'): deepcopy(vis)})
        flags = DataContainer({(0, 1, 'xx'): deepcopy(f)})
        ac.calibrate_in_place(dc, {}, flags, cal_flags, gain_convention='divide')
        np.testing.assert_array_equal(flags[(0, 1, 'xx')], True)
        dc = DataContainer({(0, 1, 'xx'): deepcopy(vis)})
        flags = DataContainer({(0, 1, 'xx'): deepcopy(f)})
        ac.calibrate_in_place(dc, g_new, flags, cal_flags, old_gains={}, gain_convention='divide')
        np.testing.assert_array_equal(flags[(0, 1, 'xx')], True)

        # test error
        dc = DataContainer({(0, 1, 'xx'): deepcopy(vis)})
        flags = DataContainer({(0, 1, 'xx'): deepcopy(f)})
        with self.assertRaises(KeyError):
            ac.calibrate_in_place(dc, g_new, flags, cal_flags, old_gains=g_old, gain_convention='blah')

        # test w/ data weights
        dc = DataContainer({(0, 1, 'xx'): deepcopy(vis)})
        flags = DataContainer({(0, 1, 'xx'): deepcopy(f)})
        wgts = DataContainer(dict(map(lambda k: (k, (~flags[k]).astype(np.float)), flags.keys())))
        del g_new[(0, 'Jxx')]
        ac.calibrate_in_place(dc, g_new, wgts, cal_flags, gain_convention='divide', flags_are_wgts=True)
        self.assertAlmostEqual(wgts[(0, 1, 'xx')].max(), 0.0)

    def test_apply_cal(self):
        miriad = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uvOCR_53x_54x_only")
        uvh5 = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5")
        outname_miriad = os.path.join(DATA_PATH, "test_output/out.uv")
        outname_uvh5 = os.path.join(DATA_PATH, "test_output/out.h5")
        calout = os.path.join(DATA_PATH, "test_output/out.cal")
        old_cal = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only")
        new_cal = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only")
        flags_npz = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uvOCR_53x_54x_only.flags.applied.npz")

        hd_old = io.HERAData(miriad, filetype='miriad')
        hd_old.read()
        hd_old.flag_array = np.logical_or(hd_old.flag_array, np.load(flags_npz)['flag_array'])
        data, data_flags, _ = hd_old.build_datacontainers()

        new_gains, new_flags = io.load_cal(new_cal)
        uvc_old = UVCal()
        uvc_old.read_calfits(old_cal)
        uvc_old.gain_array *= (3.0 + 4.0j)
        uvc_old.write_calfits(calout, clobber=True)

        ac.apply_cal(miriad, outname_miriad, new_cal, old_calibration=calout, gain_convention='divide',
                     flag_nchan_low=450, flag_nchan_high=400, flags_npz=flags_npz,
                     filetype_in='miriad', filetype_out='miriad', clobber=True, vis_units='Jy')
        hd = io.HERAData(outname_miriad, filetype='miriad')
        new_data, new_flags, _ = hd.read()
        self.assertEqual(hd.vis_units, 'Jy')
        for k in new_data.keys():
            for i in range(new_data[k].shape[0]):
                for j in range(new_data[k].shape[1]):
                    if not new_flags[k][i, j]:
                        self.assertAlmostEqual(new_data[k][i, j] / 25.0 / data[k][i, j], 1.0, 4)
                    # from flag_nchan_low and flag_nchan_high above with 1024 total channels
                    if j < 450 or j > 623:
                        self.assertTrue(new_flags[k][i, j])

        # test partial load
        ac.apply_cal(uvh5, outname_uvh5, new_cal, old_calibration=calout, gain_convention='divide',
                     flag_nchan_low=450, flag_nchan_high=400, flags_npz=flags_npz, nbl_per_load=1,
                     filetype_in='uvh5', filetype_out='uvh5', clobber=True, vis_units='Jy')
        hd = io.HERAData(outname_uvh5, filetype='uvh5')
        new_data, new_flags, _ = hd.read()
        self.assertEqual(hd.vis_units, 'Jy')
        for k in new_data.keys():
            for i in range(new_data[k].shape[0]):
                for j in range(new_data[k].shape[1]):
                    if not new_flags[k][i, j]:
                        self.assertAlmostEqual(new_data[k][i, j] / 25.0 / data[k][i, j], 1.0, 4)
                    # from flag_nchan_low and flag_nchan_high above with 1024 total channels
                    if j < 450 or j > 623:
                        self.assertTrue(new_flags[k][i, j])
        os.remove(outname_uvh5)

        # test errors
        with self.assertRaises(ValueError):
            ac.apply_cal(miriad, outname_miriad, None)
        with self.assertRaises(NotImplementedError):
            ac.apply_cal(miriad, outname_uvh5, new_cal, filetype_in='miriad', nbl_per_load=1)
        shutil.rmtree(outname_miriad)

    def test_apply_cal_argparser(self):
        sys.argv = [sys.argv[0], 'a', 'b', '--new_cal', 'd']
        a = ac.apply_cal_argparser()
        args = a.parse_args()
        self.assertEqual(args.infilename, 'a')
        self.assertEqual(args.outfilename, 'b')
        self.assertEqual(args.new_cal, ['d'])


if __name__ == '__main__':
    unittest.main()
