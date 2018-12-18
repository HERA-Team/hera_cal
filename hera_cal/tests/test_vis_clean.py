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
from hera_cal.vis_clean import VisClean
from hera_cal.data import DATA_PATH


class Test_VisClean(unittest.TestCase):

    def test_init(self):
        # test basic init w/ miriad
        fname = os.path.join(DATA_PATH, "zen.2458043.40141.xx.HH.uvXRAA")
        V = VisClean(fname, filetype='miriad')
        nt.assert_false(hasattr(V, 'data'))
        nt.assert_false(hasattr(V, 'antpos'))
        V.read(bls=[(24, 25, 'xx')])
        nt.assert_true(hasattr(V, 'data'))
        nt.assert_true(hasattr(V, 'antpos'))
        nt.assert_true(isinstance(V.hd, io.HERAData))
        nt.assert_true(isinstance(V.hd.data_array, np.ndarray))

        # test basic init w/ uvh5
        fname = os.path.join(DATA_PATH, 'zen.2458098.43124.subband.uvh5')
        V = VisClean(fname, filetype='uvh5')
        nt.assert_false(hasattr(V, 'data'))
        nt.assert_true(hasattr(V, 'antpos'))
        V.read(bls=[(13, 14, 'xx')])
        nt.assert_equal(set(V.hd.ant_1_array), set([13]))
        nt.assert_true(isinstance(V.hd, io.HERAData))
        nt.assert_true(isinstance(V.hd.data_array, np.ndarray))

        # test input cal
        fname = os.path.join(DATA_PATH, 'zen.2458043.12552.xx.HH.uvORA')
        uvc = io.HERACal(os.path.join(DATA_PATH, 'zen.2458043.12552.xx.HH.uvORA.abs.calfits'))
        gains, _, _, _ = uvc.read()
        V1 = VisClean(fname, filetype='miriad')
        bl = (52, 53, 'xx')
        V1.read(bls=[bl])
        V2 = VisClean(fname, filetype='miriad', input_cal=uvc)
        V2.read(bls=[bl])
        g = gains[(bl[0], 'Jxx')] * gains[(bl[1], 'Jxx')].conj()
        nt.assert_almost_equal((V1.data[bl] / g)[30, 30], V2.data[bl][30, 30])
        V2.apply_calibration(V2.hc, unapply=True)
        nt.assert_almost_equal(V1.data[bl][30, 30], V2.data[bl][30, 30], places=5)

        # test clear
        V1.clear_containers()
        nt.assert_false(np.any([hasattr(V1, c) for c in ['data', 'flags', 'nsamples']]))
        V2.clear_calibration()
        nt.assert_false(hasattr(V2, 'hc'))

    def test_read_write(self):
        # test read data can be turned off for uvh5
        fname = os.path.join(DATA_PATH, 'zen.2458098.43124.subband.uvh5')
        V = VisClean(fname, filetype='uvh5')
        V.read(read_data=False)
        nt.assert_equal(set(V.hd.ant_1_array), set([1, 11, 12, 13, 14]))

        # test read-write-read
        V.read()
        V.write_data(V.data, "./ex.uvh5", overwrite=True, filetype='uvh5', vis_units='Jy')
        V2 = VisClean("./ex.uvh5", filetype='uvh5')
        V2.read()
        nt.assert_equal(V2.hd.vis_units, 'Jy')
        V.hd.history, V2.hd.history, V2.hd.vis_units = '', '', V.hd.vis_units
        nt.assert_equal(V.hd, V2.hd)
        os.remove("./ex.uvh5")

        # exceptions
        nt.assert_raises(ValueError, V.write_data, V.data, 'foo', filetype='what')

    def test_vis_clean(self):
        fname = os.path.join(DATA_PATH, "zen.2458043.40141.xx.HH.uvXRAA")
        V = VisClean(fname, filetype='miriad')
        V.read()

        # just need to make sure various kwargs run through
        # actual code unit-testing coverage has been done in uvtools.dspec

        # basic freq clean
        V.vis_clean(keys=[(24, 25, 'xx'), (24, 24, 'xx')], ax='freq', overwrite=True)
        nt.assert_true(np.all([i['success'] for i in V.clean_info[(24, 25, 'xx')]]))

        # basic time clean
        V.vis_clean(keys=[(24, 25, 'xx'), (24, 24, 'xx')], ax='time', max_frate=10., overwrite=True)
        nt.assert_true('skipped' in V.clean_info[(24, 25, 'xx')][0])
        nt.assert_true('success' in V.clean_info[(24, 25, 'xx')][3])

        # basic 2d clean
        V.vis_clean(keys=[(24, 25, 'xx'), (24, 24, 'xx')], ax='both', max_frate=10., overwrite=True,
                    filt2d_mode='plus')
        nt.assert_true('success' in V.clean_info[(24, 25, 'xx')])

        V.vis_clean(keys=[(24, 25, 'xx'), (24, 24, 'xx')], ax='both', flags=V.flags + True, max_frate=10.,
                    overwrite=True, filt2d_mode='plus')
        nt.assert_true('skipped' in V.clean_info[(24, 25, 'xx')])

        # test fft data
        V.vis_clean(keys=[(24, 25, 'xx'), (24, 24, 'xx')], ax='both', max_frate=10., overwrite=True,
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

        # check various kwargs
        V.fft_data(keys=[(24, 25, 'xx')], assign='foo', ifft=True, fftshift=True)
        delays = V.delays
        nt.assert_true(hasattr(V, 'foo'))
        V.fft_data(keys=[(24, 25, 'xx')], assign='foo', overwrite=True, ifft=False, fftshift=False)
        np.testing.assert_array_almost_equal(delays, np.fft.fftshift(V.delays))
