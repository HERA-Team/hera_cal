# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"""Unit tests for the hera_cal.apply_cal module."""

import pytest
import numpy as np
import os
import sys

from .. import io, autos
from ..data import DATA_PATH
from ..utils import split_pol
from ..apply_cal import apply_cal


@pytest.mark.filterwarnings("ignore:It seems that the latitude and longitude are in radians")
@pytest.mark.filterwarnings("ignore:The default for the `center` keyword has changed")
class Test_Autos(object):
    def test_read_and_write_autocorrelations(self):
        infile = os.path.join(DATA_PATH, 'zen.2458098.43124.downsample.uvh5')
        outfile = os.path.join(DATA_PATH, 'test_output/autos.uvh5')
        autos.read_and_write_autocorrelations(infile, outfile, clobber=True, add_to_history='testing')

        hd_full = io.HERAData(infile)
        d_full, f_full, _ = hd_full.read()
        hd = io.HERAData(outfile)
        d, f, _ = hd.read()
        for bl in d.keys():
            assert bl[0] == bl[1]
            assert split_pol(bl[2])[0] == split_pol(bl[2])[1]
            np.testing.assert_array_equal(d_full[bl], d[bl])
            np.testing.assert_array_equal(f_full[bl], f[bl])
        assert 'testing' in hd.history.replace('\n', '').replace(' ', '')
        assert 'Thisfilewasproducedbythefunction' in hd.history.replace('\n', '').replace(' ', '')
        os.remove(outfile)

    def test_read_calibrate_and_write_autocorrelations(self):
        infile = os.path.join(DATA_PATH, 'zen.2458098.43124.downsample.uvh5')
        outfile = os.path.join(DATA_PATH, 'test_output/autos.uvh5')
        calfile = os.path.join(DATA_PATH, 'test_input/zen.2458098.43124.downsample.omni.calfits')
        calibrated = os.path.join(DATA_PATH, 'test_output/calibrated.uvh5')
        autos.read_and_write_autocorrelations(infile, outfile, calfile=calfile, clobber=True)
        apply_cal(infile, calibrated, calfile, clobber=True)

        hd_full_cal = io.HERAData(calibrated)
        d_full_cal, f_full_cal, _ = hd_full_cal.read()
        hd = io.HERAData(outfile)
        d, f, _ = hd.read()
        for bl in d.keys():
            assert bl[0] == bl[1]
            assert split_pol(bl[2])[0] == split_pol(bl[2])[1]
            np.testing.assert_array_equal(d_full_cal[bl], d[bl])
            np.testing.assert_array_equal(f_full_cal[bl], f[bl])
        os.remove(outfile)
        os.remove(calibrated)

    def test_extract_autos_argparser(self):
        sys.argv = [sys.argv[0], 'a', 'b', '--calfile', 'd']
        a = autos.extract_autos_argparser()
        args = a.parse_args()
        assert args.infile == 'a'
        assert args.outfile == 'b'
        assert args.calfile == ['d']
