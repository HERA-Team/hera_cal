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
import nose.tools as nt

from hera_cal import io
import hera_cal.vis_clean as vc
from hera_cal.data import DATA_PATH


class Test_VisClean(unittest.TestCase):

    def test_load_write(self):
        # test basic init
        fname = os.path.join(DATA_PATH, "zen.2458043.40141.xx.HH.uvXRAA")
        V = vc.VisClean(fname, filetype='miriad')
        nt.assert_true(hasattr(V, 'data'))
        nt.assert_true(isinstance(V.hd, io.HERAData))
        nt.assert_true(isinstance(V.hd.data_array, np.ndarray))

        # test read data can be turned off for uvh5
        fname = os.path.join(DATA_PATH, "zen.2458098.43124.downsample.uvh5")
        V = vc.VisClean(fname, filetype='uvh5', read_data=False)
        nt.assert_false(hasattr(V, 'data'))

        # test apply cal
        fname = os.path.join(DATA_PATH, "zen.2458043.13298.xx.HH.uvORA")
        uvd = UVData()
        uvd.read_miriad(fname)
        cname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA.abs.calfits")
        uvc = UVCal()
        uvc.read_calfits(cname)
        V = vc.VisClean(uvd, filetype='miriad', inp_cal=uvc)
        bl = (24, 25, 'xx')
        g = (uvc.gain_array[uvc.ant_array.tolist().index(bl[0])].squeeze() * uvc.gain_array[uvc.ant_array.tolist().index(bl[1])].conj().squeeze()).T
        nt.assert_almost_equal(uvd.get_data(bl)[30, 30] / g[30, 30], V.data[bl][30, 30])

        # test read-write-read
        fname = os.path.join(DATA_PATH, "zen.2458043.13298.xx.HH.uvORA")
        V = vc.VisClean(fname)
        V.write_data(V.data, "./ex", overwrite=True, filetype='miriad')
        V2 = vc.VisClean("./ex")
        V.hd.history, V2.hd.history = '', ''
        nt.assert_equal(V.hd, V2.hd)
        shutil.rmtree("./ex")

        # exceptions
        nt.assert_raises(ValueError, V.load_data, 1.0)
        nt.assert_raises(ValueError, V.apply_cal, 1.0)
        nt.assert_raises(ValueError, V.write_data, V.data, 'foo', filetype='what')

    def test_vis_clean(self):
        fname = os.path.join(DATA_PATH, "zen.2458043.40141.xx.HH.uvXRAA")
        V = vc.VisClean(fname, filetype='miriad')

        # just need to make sure various kwargs run through
        # actual code unit-testing coverage has been done in uvtools.dspec

        # basic freq clean
        V.vis_clean(keys=[(24, 25, 'xx'), (24, 24, 'xx')], ax='freq', overwrite=True)
        nt.assert_true(np.all([i['success'] for i in V.clean_info[(24, 25, 'xx')]]))

        # basic time clean
        V.vis_clean(keys=[(24, 25, 'xx'), (24, 24, 'xx')], ax='time', max_frate=10e-3, overwrite=True)
        nt.assert_true('skipped' in V.clean_info[(24, 25, 'xx')][0])
        nt.assert_true('success' in V.clean_info[(24, 25, 'xx')][3])

        # basic 2d clean
        V.vis_clean(keys=[(24, 25, 'xx'), (24, 24, 'xx')], ax='both', max_frate=10e-3, overwrite=True,
                    filt2d_mode='plus')
        nt.assert_true('success' in V.clean_info[(24, 25, 'xx')])

        V.vis_clean(keys=[(24, 25, 'xx'), (24, 24, 'xx')], ax='both', flags=V.flags + True, max_frate=10e-3,
                    overwrite=True, filt2d_mode='plus')
        nt.assert_true('skipped' in V.clean_info[(24, 25, 'xx')])

        # test fft data
        V.vis_clean(keys=[(24, 25, 'xx'), (24, 24, 'xx')], ax='both', max_frate=10e-3, overwrite=True,
                    filt2d_mode='rect')

        # assert foreground peak is at 0 delay bin
        V.fft_data(data=V.clean_model, keys=[(24, 25, 'xx')], ax='freq', window='hann', edgecut_low=10, edgecut_hi=10, overwrite=True)
        nt.assert_equal(np.argmax(np.mean(np.abs(V.dfft[(24, 25, 'xx')]), axis=0)), 32)

        # assert foreground peak is at 0 FR bin (just due to FR resolution)
        V.fft_data(data=V.clean_model, keys=[(24, 25, 'xx')], ax='time', window='hann', edgecut_low=10, edgecut_hi=10, overwrite=True)
        nt.assert_equal(np.argmax(np.mean(np.abs(V.dfft[(24, 25, 'xx')]), axis=1)), 30)

        # assert foreground peak is at both 0 FR and 0 delay bin
        V.fft_data(data=V.clean_model, keys=[(24, 25, 'xx')], ax='both', window='tukey', alpha=0.5, edgecut_low=10, edgecut_hi=10, overwrite=True)
        nt.assert_equal(np.argmax(np.mean(np.abs(V.dfft[(24, 25, 'xx')]), axis=0)), 32)
        nt.assert_equal(np.argmax(np.mean(np.abs(V.dfft[(24, 25, 'xx')]), axis=1)), 30)
