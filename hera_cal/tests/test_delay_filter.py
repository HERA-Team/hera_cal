# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

from __future__ import print_function, division, absolute_import

import unittest
import numpy as np
from copy import deepcopy
import os
import sys
import shutil
from scipy import constants
from pyuvdata import UVCal, UVData
from uvtools.dspec import delay_filter

from hera_cal import io
import hera_cal.delay_filter as df
from hera_cal.data import DATA_PATH


class Test_DelayFilter(unittest.TestCase):

    def test_init(self):
        dfil = df.DelayFilter()

    def test_load_data(self):
        fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        dfil = df.DelayFilter()
        dfil.load_data(fname, filetype='miriad')
        self.assertEqual(dfil.data[(24, 25, 'xx')].shape, (60, 64))
        self.assertEqual(dfil.flags[(24, 25, 'xx')].shape, (60, 64))
        self.assertEqual(len(dfil.antpos), 47)
        self.assertEqual(type(dfil.antpos[24]), np.ndarray)
        self.assertEqual(len(dfil.antpos[24]), 3)
        self.assertEqual(len(dfil.freqs), 64)

        fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        uvd = UVData()
        uvd.read_miriad(fname)
        dfil = df.DelayFilter()
        dfil.load_data(uvd, filetype='miriad')
        self.assertEqual(dfil.data[(24, 25, 'xx')].shape, (60, 64))
        self.assertEqual(dfil.flags[(24, 25, 'xx')].shape, (60, 64))
        self.assertEqual(len(dfil.antpos), 47)
        self.assertEqual(type(dfil.antpos[24]), np.ndarray)
        self.assertEqual(len(dfil.antpos[24]), 3)
        self.assertEqual(len(dfil.freqs), 64)

        filename1 = os.path.join(DATA_PATH, 'zen.2458043.12552.xx.HH.uvORA')
        filename2 = os.path.join(DATA_PATH, 'zen.2458043.13298.xx.HH.uvORA')
        dfil = df.DelayFilter()
        dfil.load_data([filename1, filename2], filetype='miriad')

        # test uvh5 with calibration
        cal = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only")
        uvh5 = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5")
        hc = io.HERACal(cal)
        gains, calflags, _, _ = hc.read()
        hd = io.HERAData(uvh5, filetype='uvh5')
        data, flags, _ = hd.read()

        dfil = df.DelayFilter()
        dfil.load_data(hd, filetype='uvh5', inp_cal=hc)
        flag_sum = flags[54, 54, 'xx'] + calflags[54, 'Jxx']
        np.testing.assert_array_equal(flag_sum, dfil.flags[54, 54, 'xx'])
        calibrated = (data[54, 54, 'xx'] / gains[54, 'Jxx'] / np.conj(gains[54, 'Jxx']))[~flag_sum]
        np.testing.assert_array_almost_equal(dfil.data[54, 54, 'xx'][~flag_sum] / calibrated, 1.0 + 0.0j, decimal=5)

        dfil = df.DelayFilter()
        dfil.load_data(uvh5, filetype='uvh5', inp_cal=cal)
        flag_sum = flags[54, 54, 'xx'] + calflags[54, 'Jxx']
        np.testing.assert_array_equal(flag_sum, dfil.flags[54, 54, 'xx'])
        calibrated = (data[54, 54, 'xx'] / gains[54, 'Jxx'] / np.conj(gains[54, 'Jxx']))[~flag_sum]
        np.testing.assert_array_almost_equal(dfil.data[54, 54, 'xx'][~flag_sum] / calibrated, 1.0 + 0.0j, decimal=5)

    def test_run_filter(self):
        fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        k = (24, 25, 'xx')
        dfil = df.DelayFilter()
        dfil.load_data(fname, filetype='miriad')
        bl = np.linalg.norm(dfil.antpos[24] - dfil.antpos[25]) / constants.c * 1e9
        sdf = (dfil.freqs[1] - dfil.freqs[0]) / 1e9

        dfil.run_filter(to_filter=dfil.data.keys(), tol=1e-2)
        for k in dfil.data.keys():
            self.assertEqual(dfil.clean_resid[k].shape, (60, 64))
            self.assertEqual(dfil.clean_model[k].shape, (60, 64))
            self.assertTrue(k in dfil.clean_info)

        # test skip_wgt imposition of flags
        dfil = df.DelayFilter()
        dfil.load_data(fname, filetype='miriad')
        wgts = {k: np.ones_like(dfil.flags[k], dtype=np.float)}
        wgts[k][0, :] = 0.0
        dfil.run_filter(to_filter=[k], weight_dict=wgts, standoff=0., horizon=1., tol=1e-5, window='blackman-harris', skip_wgt=0.1, maxiter=100)
        self.assertTrue(dfil.clean_info[k][0]['skipped'])
        np.testing.assert_array_equal(dfil.clean_flags[k][0, :], np.ones_like(dfil.flags[k][0, :]))
        np.testing.assert_array_equal(dfil.clean_model[k][0, :], np.zeros_like(dfil.clean_resid[k][0, :]))
        np.testing.assert_array_equal(dfil.clean_resid[k][0, :], dfil.data[k][0, :])

    def test_write_filtered_data(self):
        dfil = df.DelayFilter()
        with self.assertRaises(ValueError):
            dfil.write_filtered_data('')

        fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        dfil = df.DelayFilter()
        dfil.load_data(fname, filetype='miriad')
        data = dfil.data
        dfil.run_filter(standoff=0., horizon=1., tol=1e-9, window='blackman-harris', skip_wgt=0.1, maxiter=100, edgecut_low=5, edgecut_hi=5)
        outfilename = os.path.join(DATA_PATH, 'test_output/zen.2458043.12552.xx.HH.filter_test.ORAD.uvh5')
        with self.assertRaises(ValueError):
            dfil.write_filtered_data()
        with self.assertRaises(NotImplementedError):
            dfil.write_filtered_data(res_outfilename=outfilename, partial_write=True)
        dfil.write_filtered_data(res_outfilename=outfilename, add_to_history='Hello_world.', clobber=True, telescope_name='PAPER')

        uvd = UVData()
        uvd.read_uvh5(outfilename)
        self.assertEqual(uvd.history.replace('\n', '').replace(' ', '')[-12:], 'Hello_world.')
        self.assertEqual(uvd.telescope_name, 'PAPER')

        filtered_residuals, flags = io.load_vis(uvd)

        dfil.write_filtered_data(CLEAN_outfilename=outfilename, clobber=True)
        clean_model, _flags = io.load_vis(outfilename, filetype='uvh5')

        dfil.write_filtered_data(filled_outfilename=outfilename, clobber=True)
        filled_data, filled_flags = io.load_vis(outfilename, filetype='uvh5')

        for k in data.keys():
            np.testing.assert_array_almost_equal(filled_data[k][~flags[k]], data[k][~flags[k]])
            np.testing.assert_array_almost_equal(dfil.clean_model[k], clean_model[k])
            np.testing.assert_array_almost_equal(dfil.clean_resid[k], filtered_residuals[k])
            np.testing.assert_array_almost_equal(data[k][~flags[k]], (clean_model[k] + filtered_residuals[k])[~flags[k]], 5)
        os.remove(outfilename)

    def test_partial_load_delay_filter_and_write(self):
        uvh5 = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5")
        outfilename = os.path.join(DATA_PATH, 'test_output/temp.h5')
        df.partial_load_delay_filter_and_write(uvh5, res_outfilename=outfilename, Nbls=2, tol=1e-4, clobber=True)
        hd = io.HERAData(outfilename)
        d, f, n = hd.read(bls=[(53, 54, 'xx')])

        dfil = df.DelayFilter()
        dfil.load_data(uvh5, filetype='uvh5')
        dfil.run_filter(to_filter=[(53, 54, 'xx')], tol=1e-4, verbose=True)
        np.testing.assert_almost_equal(d[(53, 54, 'xx')], dfil.clean_resid[(53, 54, 'xx')])
        np.testing.assert_array_equal(f[(53, 54, 'xx')], dfil.flags[(53, 54, 'xx')])

        cal = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only")
        outfilename = os.path.join(DATA_PATH, 'test_output/temp.h5')
        df.partial_load_delay_filter_and_write(uvh5, calfile=cal, tol=1e-4, res_outfilename=outfilename, Nbls=2, clobber=True)
        hd = io.HERAData(outfilename)
        d, f, n = hd.read(bls=[(53, 54, 'xx')])
        np.testing.assert_array_equal(f[(53, 54, 'xx')], True)
        os.remove(outfilename)

    def test_delay_filter_argparser(self):
        sys.argv = [sys.argv[0], 'a', '--clobber']
        parser = df.delay_filter_argparser()
        a = parser.parse_args()
        self.assertEqual(a.infilename, 'a')
        self.assertTrue(a.clobber)


if __name__ == '__main__':
    unittest.main()
