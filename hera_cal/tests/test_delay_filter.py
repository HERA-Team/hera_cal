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

    def test_run_filter(self):
        fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        k = (24, 25, 'xx')
        dfil = df.DelayFilter(fname, filetype='miriad')
        dfil.read(bls=[k])
        bl = np.linalg.norm(dfil.antpos[24] - dfil.antpos[25]) / constants.c * 1e9
        sdf = (dfil.freqs[1] - dfil.freqs[0]) / 1e9

        dfil.run_filter(to_filter=dfil.data.keys(), tol=1e-2)
        for k in dfil.data.keys():
            self.assertEqual(dfil.clean_resid[k].shape, (60, 64))
            self.assertEqual(dfil.clean_model[k].shape, (60, 64))
            self.assertTrue(k in dfil.clean_info)

        # test skip_wgt imposition of flags
        fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        k = (24, 25, 'xx')
        dfil = df.DelayFilter(fname, filetype='miriad')
        dfil.read(bls=[k])
        wgts = {k: np.ones_like(dfil.flags[k], dtype=np.float)}
        wgts[k][0, :] = 0.0
        dfil.run_filter(to_filter=[k], weight_dict=wgts, standoff=0., horizon=1., tol=1e-5, window='blackman-harris', skip_wgt=0.1, maxiter=100)
        self.assertTrue(dfil.clean_info[k][0]['skipped'])
        np.testing.assert_array_equal(dfil.clean_flags[k][0, :], np.ones_like(dfil.flags[k][0, :]))
        np.testing.assert_array_equal(dfil.clean_model[k][0, :], np.zeros_like(dfil.clean_resid[k][0, :]))
        np.testing.assert_array_equal(dfil.clean_resid[k][0, :], np.zeros_like(dfil.clean_resid[k][0, :]))

    def test_write_filtered_data(self):
        fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        k = (24, 25, 'xx')
        dfil = df.DelayFilter(fname, filetype='miriad')
        dfil.read(bls=[k])

        data = dfil.data
        dfil.run_filter(standoff=0., horizon=1., tol=1e-9, window='blackman-harris', skip_wgt=0.1, maxiter=100, edgecut_low=0, edgecut_hi=0)
        outfilename = os.path.join(DATA_PATH, 'test_output/zen.2458043.12552.xx.HH.filter_test.ORAD.uvh5')
        with self.assertRaises(ValueError):
            dfil.write_filtered_data()
        with self.assertRaises(NotImplementedError):
            dfil.write_filtered_data(res_outfilename=outfilename, partial_write=True)
        dfil.write_filtered_data(res_outfilename=outfilename, add_to_history='Hello_world.', clobber=True, telescope_name='PAPER')

        uvd = UVData()
        uvd.read_uvh5(outfilename)
        self.assertTrue('Hello_world.' in uvd.history.replace('\n', '').replace(' ', ''))
        self.assertTrue('Thisfilewasproducedbythefunction' in uvd.history.replace('\n', '').replace(' ', ''))
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
        df.partial_load_delay_filter_and_write(uvh5, res_outfilename=outfilename, Nbls=1, tol=1e-4, clobber=True)
        hd = io.HERAData(outfilename)
        d, f, n = hd.read(bls=[(53, 54, 'xx')])

        dfil = df.DelayFilter(uvh5, filetype='uvh5')
        dfil.read(bls=[(53, 54, 'xx')])
        dfil.run_filter(to_filter=[(53, 54, 'xx')], tol=1e-4, verbose=True)
        np.testing.assert_almost_equal(d[(53, 54, 'xx')], dfil.clean_resid[(53, 54, 'xx')], decimal=5)
        np.testing.assert_array_equal(f[(53, 54, 'xx')], dfil.flags[(53, 54, 'xx')])

        cal = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only")
        outfilename = os.path.join(DATA_PATH, 'test_output/temp.h5')
        df.partial_load_delay_filter_and_write(uvh5, calfile=cal, tol=1e-4, res_outfilename=outfilename, Nbls=2, clobber=True)
        hd = io.HERAData(outfilename)
        self.assertTrue('Thisfilewasproducedbythefunction' in hd.history.replace('\n', '').replace(' ', ''))
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
